"""
RipupManager: 拆线重布管理器

总体流程：
  0. 预布线（pre-route）：为含 must_keep_edges 的线网固定必走边
     - 在 SpaceConstraint 中标记占用
     - 在 CostManager 中更新拥塞地图（与普通布线相同）
     - 后续拆线时，预布线节点受保护（仅撤销非预布线部分）
  1. 按策略顺序逐个布线所有线网
  2. 若某线网布线失败：
     a. 分析阻塞的其他线网
     b. 由策略决定拆线方案（区域/单网/多网）
     c. 执行拆线，清除约束标记（预布线节点除外）
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

    预布线（pre-route）机制：
        在正式布线迭代前，对含有 must_keep_edges 约束的线网执行预布线阶段：
          1. 仅将 must-keep 边及其端点写入预布线结果
          2. 在 SpaceConstraint 中标记这些节点为已占用
          3. 在 CostManager 中以 1.0 的增量更新拥塞地图（与普通布线一致）
        预布线节点在整个拆线重布过程中受保护：
          - 整网拆线时：全量撤标后立即重新标记预布线节点
          - 区域拆线时：跳过预布线节点，仅撤标非预布线部分
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
        执行完整的布线流程（含预布线和拆线重布）。

        参数:
            nets: 所有需要布线的线网
        返回:
            RoutingSolution
        """
        solution = RoutingSolution()
        max_iter = self.strategy.get_max_iterations()

        ordered_nets = self.strategy.order_nets(nets)
        net_map: Dict[str, Net] = {net.name: net for net in nets}

        # ── 预布线阶段：固定 must-keep 边 ─────────────────────────────────
        pre_routes = self._build_pre_routes(nets)

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
                            pre_routes,
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
                                # 拥塞记录：仅对非预布线节点记录
                                if self._is_congestion_aware:
                                    fixed = pre_routes[rn_name].routed_nodes \
                                        if rn_name in pre_routes else set()
                                    for node in solution.results[rn_name].routed_nodes:
                                        if node not in fixed:
                                            self.strategy.record_congestion(node)
                                # 整网拆除（含预布线保护）
                                self._do_ripup(rn_name, solution, pre_routes)
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
    # 预布线（pre-route）
    # ------------------------------------------------------------------

    def _build_pre_routes(self, nets: List[Net]) -> Dict[str, RoutingResult]:
        """
        预布线阶段：为含 must_keep_edges 的线网固定必走边。

        对每条 must-keep 边 (a, b)：
          1. 验证两端点及边存在于网格中
          2. 将边写入预布线结果（routed_nodes / routed_edges）
          3. 标记节点到 SpaceConstraint（等同于正常布线的 mark_route）
          4. 更新拥塞地图（与普通布线一致，初始拥塞 = 1.0）

        返回:
            {net_name: RoutingResult}  ── 仅含预布线节点的固定结果
        """
        pre_routes: Dict[str, RoutingResult] = {}

        for net in nets:
            must_keep = self.constraint_mgr.get_must_keep_edges(net.name)
            if not must_keep:
                continue

            pre_result = RoutingResult(net=net)
            valid = True

            for a, b in must_keep:
                if not self.grid.is_valid_node(a) or not self.grid.is_valid_node(b):
                    logger.warning(
                        f"pre_route: 线网 {net.name} must_keep_edge {(a, b)} "
                        f"端点不在网格中，预布线跳过"
                    )
                    valid = False
                    break
                edge_cost = self.grid.get_edge_cost(a, b)
                if edge_cost >= float('inf'):
                    logger.warning(
                        f"pre_route: 线网 {net.name} must_keep_edge {(a, b)} "
                        f"在网格中不存在，预布线跳过"
                    )
                    valid = False
                    break
                pre_result.routed_nodes.add(a)
                pre_result.routed_nodes.add(b)
                pre_result.routed_edges.append((a, b))
                pre_result.total_cost += edge_cost

            if not valid or not pre_result.routed_nodes:
                continue

            pre_result.success = True

            # 标记占用（与 mark_route 相同语义）
            self.constraint_mgr.mark_route(net.name, pre_result.routed_nodes)

            # 更新拥塞地图：为预布线节点赋予初始拥塞（与普通布线进入拥塞地图的方式一致）
            for node in pre_result.routed_nodes:
                if not self.grid.is_virtual_node(node):
                    self.cost_mgr.update_congestion(node, 1.0)

            pre_routes[net.name] = pre_result
            logger.info(
                f"pre_route: 线网 {net.name} 固定 {len(pre_result.routed_edges)} 条边，"
                f"{len(pre_result.routed_nodes)} 个节点，代价 {pre_result.total_cost:.2f}"
            )

        return pre_routes

    # ------------------------------------------------------------------
    # 整网拆线辅助（含预布线保护）
    # ------------------------------------------------------------------

    def _do_ripup(
        self,
        net_name: str,
        solution: RoutingSolution,
        pre_routes: Dict[str, RoutingResult],
    ):
        """
        撤销线网的全部约束标记，并立即重新标记预布线节点（固定保护）。

        若线网有预布线（pre_routes[net_name]），执行：
          1. 全量撤标（unmark_route）
          2. 重新标记预布线节点（mark_route with fixed nodes）
        若无预布线，直接全量撤标。
        """
        self.constraint_mgr.unmark_route(net_name)
        if net_name in pre_routes:
            # 重新标记固定节点：保护预布线部分不被其他线网占用
            self.constraint_mgr.mark_route(net_name, pre_routes[net_name].routed_nodes)
        solution.results[net_name].success = False

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
        pre_routes: Dict[str, RoutingResult],
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

            # 预布线节点受保护：从区域拆线范围中排除
            fixed = pre_routes[rn_name].routed_nodes if rn_name in pre_routes else set()
            nodes_to_unmark = nodes_in_region - fixed
            nodes_in_region_unfixed = nodes_to_unmark   # 实际被拆除的节点

            if not nodes_in_region_unfixed:
                logger.debug(
                    f"线网 {rn_name} 区域内节点均为预布线固定节点，跳过区域拆线"
                )
                continue

            if self._is_congestion_aware:
                for node in nodes_in_region_unfixed:
                    self.strategy.record_congestion(node)

            self.constraint_mgr.partial_unmark_route(rn_name, nodes_in_region_unfixed)

            remaining_nodes = result.routed_nodes - nodes_in_region_unfixed
            remaining_edges = [
                (u, v) for u, v in result.routed_edges
                if u not in nodes_in_region_unfixed and v not in nodes_in_region_unfixed
            ]

            if not remaining_nodes:
                result.success = False
                # 回退到整网拆线：保护预布线节点
                if rn_name in pre_routes:
                    self.constraint_mgr.mark_route(
                        rn_name, pre_routes[rn_name].routed_nodes
                    )
                needs_full_reroute.append(net_map[rn_name])
                logger.info(f"线网 {rn_name} 全部非固定节点在区域内，降级为整网拆线")
                continue

            components = find_connected_components(remaining_nodes, remaining_edges)

            if len(components) <= 1:
                result.routed_nodes = remaining_nodes
                result.routed_edges = remaining_edges
                result.total_cost *= len(remaining_nodes) / (
                    len(remaining_nodes) + len(nodes_in_region_unfixed)
                )
            else:
                reconnected = self._reconnect_net(
                    net_map[rn_name], components, remaining_edges, result,
                )
                if not reconnected:
                    self.constraint_mgr.partial_unmark_route(rn_name, remaining_nodes)
                    result.success = False
                    if rn_name in pre_routes:
                        self.constraint_mgr.mark_route(
                            rn_name, pre_routes[rn_name].routed_nodes
                        )
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

