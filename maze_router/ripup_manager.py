"""
RipupManager: 拆线重布管理器

总体流程：
  1. 按策略顺序逐个布线所有线网
  2. 若某线网布线失败：
     a. 分析阻塞的其他线网
     b. 由策略决定拆线方案（区域/单网/多网）
     c. 执行拆线，清除约束标记
     d. 重新布线失败的线网
     e. 被拆除的线网加入待布线队列
  3. 重复直到全部布通或达到最大迭代次数
"""

from __future__ import annotations
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set

from maze_router.data.net import Net, Node, RoutingResult, RoutingSolution
from maze_router.data.grid import GridGraph
from maze_router.data.region import (
    Region, build_conflict_region, analyze_regional_ripup,
    find_nodes_in_region, find_connected_components,
)
from maze_router.constraint_manager import ConstraintManager
from maze_router.cost_manager import CostManager
from maze_router.ripup_strategy import RipupStrategy, RipupAction, CongestionAwareRipupStrategy
from maze_router.router import Router
from maze_router.maze_router_algo import MazeRouter

logger = logging.getLogger(__name__)


class RipupManager:
    """
    拆线重布管理器。

    由 MazeRouterEngine 调度，管理整个拆线重布流程。
    """

    def __init__(
        self,
        grid: GridGraph,
        constraint_mgr: ConstraintManager,
        cost_mgr: CostManager,
        strategy: RipupStrategy,
    ):
        self.grid = grid
        self.constraint_mgr = constraint_mgr
        self.cost_mgr = cost_mgr
        self.strategy = strategy
        self.router = Router(grid, constraint_mgr, cost_mgr, strategy)
        self._is_congestion_aware = isinstance(strategy, CongestionAwareRipupStrategy)

    def run(self, nets: List[Net]) -> RoutingSolution:
        """
        执行完整的布线流程（含拆线重布）。

        参数:
            nets: 所有需要布线的线网
        返回:
            RoutingSolution
        """
        solution = RoutingSolution()
        max_iter = self.strategy.get_max_iterations()

        ordered_nets = self.strategy.order_nets(nets)
        net_map: Dict[str, Net] = {net.name: net for net in nets}

        pending = list(ordered_nets)
        ripup_count: Dict[str, int] = defaultdict(int)
        iteration = 0

        while pending and iteration < max_iter:
            iteration += 1
            logger.info(
                f"=== 布线迭代 {iteration}/{max_iter}，待布线 {len(pending)} 个线网 ==="
            )

            next_pending: List[Net] = []
            progress_made = False

            for net in pending:
                result = self.router.route_net(net, iteration=iteration)

                if result.success:
                    self.constraint_mgr.mark_route(net.name, result.routed_nodes)
                    solution.add_result(result)
                    progress_made = True
                    logger.info(f"线网 {net.name} 布线成功，代价 {result.total_cost:.2f}")
                else:
                    logger.warning(f"线网 {net.name} 布线失败，分析阻塞...")

                    blocking_nets = self.router.find_blocked_nets(net)

                    region = build_conflict_region(
                        net, blocking_nets, self.constraint_mgr, self.grid,
                    )
                    region_ratios = analyze_regional_ripup(
                        net, blocking_nets, self.constraint_mgr,
                        solution.results, region,
                    )

                    decision = self.strategy.decide_ripup(
                        failed_net=net,
                        blocking_nets=blocking_nets,
                        net_results=solution.results,
                        iteration=iteration,
                        region_ratios=region_ratios,
                        region=region,
                    )

                    logger.info(f"拆线决策: {decision.action.value} - {decision.reason}")

                    if decision.action == RipupAction.SKIP:
                        next_pending.append(net)
                        continue

                    if decision.action == RipupAction.FAIL:
                        solution.add_result(result)
                        continue

                    if decision.action == RipupAction.RIPUP_REGION:
                        ripped = self._execute_regional_ripup(
                            decision, solution, net_map, ripup_count, max_iter,
                        )
                        retry = self.router.route_net(net, iteration=iteration)
                        if retry.success:
                            self.constraint_mgr.mark_route(net.name, retry.routed_nodes)
                            solution.add_result(retry)
                            progress_made = True
                            logger.info(f"线网 {net.name} 区域拆线后重布成功")
                        else:
                            next_pending.append(net)
                        next_pending.extend(ripped)
                    else:
                        # 整网拆线
                        ripped: List[Net] = []
                        for rn_name in decision.nets_to_ripup:
                            if (rn_name in solution.results
                                    and solution.results[rn_name].success):
                                ripup_count[rn_name] += 1
                                if ripup_count[rn_name] > max_iter // 2:
                                    logger.warning(
                                        f"线网 {rn_name} 已被拆 {ripup_count[rn_name]} 次，跳过"
                                    )
                                    continue
                                if self._is_congestion_aware:
                                    for node in solution.results[rn_name].routed_nodes:
                                        self.strategy.record_congestion(node)
                                self.constraint_mgr.unmark_route(rn_name)
                                solution.results[rn_name].success = False
                                ripped.append(net_map[rn_name])
                                logger.info(f"已拆除线网 {rn_name}")

                        retry = self.router.route_net(net, iteration=iteration)
                        if retry.success:
                            self.constraint_mgr.mark_route(net.name, retry.routed_nodes)
                            solution.add_result(retry)
                            progress_made = True
                            logger.info(f"线网 {net.name} 重布成功")
                        else:
                            next_pending.append(net)
                        next_pending.extend(ripped)

            pending = next_pending

            if not progress_made and pending:
                logger.warning(f"本轮无进展，剩余 {len(pending)} 个线网未布通")

        # 处理最终未布通的线网
        for net in pending:
            if net.name not in solution.results or not solution.results[net.name].success:
                solution.add_result(RoutingResult(net=net, success=False))

        self._log_summary(solution, iteration)
        return solution

    # ------------------------------------------------------------------
    # 区域拆线
    # ------------------------------------------------------------------

    def _execute_regional_ripup(
        self,
        decision,
        solution: RoutingSolution,
        net_map: Dict[str, Net],
        ripup_count: Dict[str, int],
        max_iter: int,
    ) -> List[Net]:
        """执行区域拆线，返回需要整网重布的线网列表。"""
        region = decision.region
        needs_full_reroute: List[Net] = []

        for rn_name in decision.nets_to_ripup:
            if rn_name not in solution.results or not solution.results[rn_name].success:
                continue

            ripup_count[rn_name] += 1
            if ripup_count[rn_name] > max_iter // 2:
                logger.warning(f"线网 {rn_name} 拆除次数过多，跳过区域拆线")
                continue

            result = solution.results[rn_name]
            nodes_in_region = find_nodes_in_region(result.routed_nodes, region)

            if not nodes_in_region:
                continue

            if self._is_congestion_aware:
                for node in nodes_in_region:
                    self.strategy.record_congestion(node)

            self.constraint_mgr.partial_unmark_route(rn_name, nodes_in_region)

            remaining_nodes = result.routed_nodes - nodes_in_region
            remaining_edges = [
                (u, v) for u, v in result.routed_edges
                if u not in nodes_in_region and v not in nodes_in_region
            ]

            if not remaining_nodes:
                result.success = False
                needs_full_reroute.append(net_map[rn_name])
                logger.info(f"线网 {rn_name} 全部节点在区域内，降级为整网拆线")
                continue

            components = find_connected_components(remaining_nodes, remaining_edges)

            if len(components) <= 1:
                result.routed_nodes = remaining_nodes
                result.routed_edges = remaining_edges
                result.total_cost *= len(remaining_nodes) / (
                    len(remaining_nodes) + len(nodes_in_region)
                )
            else:
                reconnected = self._reconnect_net(
                    net_map[rn_name], components, remaining_edges, result,
                )
                if not reconnected:
                    self.constraint_mgr.partial_unmark_route(rn_name, remaining_nodes)
                    result.success = False
                    needs_full_reroute.append(net_map[rn_name])

        return needs_full_reroute

    def _reconnect_net(
        self,
        net: Net,
        components: List[Set[Node]],
        remaining_edges: List,
        result: RoutingResult,
    ) -> bool:
        """重连区域拆线后断开的子树。"""
        maze = MazeRouter(self.grid, self.constraint_mgr, self.cost_mgr)

        components_sorted = sorted(components, key=len, reverse=True)
        tree_nodes = set(components_sorted[0])
        all_edges = list(remaining_edges)
        total_extra = 0.0

        for comp in components_sorted[1:]:
            path = maze.route(
                sources=tree_nodes,
                targets=comp,
                net_name=net.name,
                cable_locs=net.cable_locs,
            )
            if not path.success:
                return False

            new_nodes = set(path.nodes) - result.routed_nodes
            if new_nodes:
                self.constraint_mgr.partial_mark_route(net.name, new_nodes)

            tree_nodes.update(path.nodes)
            tree_nodes.update(comp)
            all_edges.extend(path.edges)
            total_extra += path.cost

        result.routed_nodes = tree_nodes
        result.routed_edges = all_edges
        result.total_cost += total_extra
        result.success = True
        return True

    # ------------------------------------------------------------------
    # 摘要
    # ------------------------------------------------------------------

    @staticmethod
    def _log_summary(solution: RoutingSolution, iterations: int):
        total = len(solution.results)
        routed = solution.routed_count
        failed = solution.failed_nets

        logger.info("=" * 50)
        logger.info(f"布线完成: {routed}/{total} 个线网布通")
        logger.info(f"总代价: {solution.total_cost:.2f}")
        logger.info(f"总迭代次数: {iterations}")
        if failed:
            logger.warning(f"未布通线网: {failed}")
        logger.info("=" * 50)
