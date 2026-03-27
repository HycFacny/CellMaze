"""
拆线重布模块

管理布线失败时的拆线和重新布线流程。
支持拆除单个线网、多个线网或区域内线网，
并通过迭代和拥塞感知逐步改善布线结果。
"""

import logging
from typing import List, Dict, Set, Optional
from collections import defaultdict

from maze_router.net import Net, Node, RoutingResult, RoutingSolution
from maze_router.grid import RoutingGrid
from maze_router.spacing import SpacingManager
from maze_router.steiner import SteinerTreeBuilder
from maze_router.steiner_router import SteinerRouter
from maze_router.router import MazeRouter
from maze_router.strategy import (
    RoutingStrategy,
    RipupAction,
    CongestionAwareStrategy,
    SteinerMethod,
)
from maze_router.region import (
    ConflictRegion,
    build_conflict_region,
    analyze_regional_ripup,
    find_nodes_in_region,
    find_connected_components,
)

logger = logging.getLogger(__name__)


class RipupManager:
    """
    拆线重布管理器。

    总体流程:
    1. 按策略顺序逐个布线所有线网（构建Steiner树）
    2. 若某个线网布线失败：
       a. 分析阻塞的其他线网
       b. 由策略决定拆线方案
       c. 执行拆线，清除间距标记
       d. 重新布线失败的线网
       e. 被拆除的线网加入待布线队列
    3. 重复直到全部布通或达到最大迭代次数
    4. 每轮迭代可选地增加拥塞区域代价（PathFinder风格）
    """

    def __init__(
        self,
        grid: RoutingGrid,
        spacing_mgr: SpacingManager,
        strategy: RoutingStrategy,
    ):
        self.grid = grid
        self.spacing_mgr = spacing_mgr
        self.strategy = strategy
        self.steiner_builder = SteinerTreeBuilder(grid, spacing_mgr)

        # 拥塞感知（如果策略支持）
        self._is_congestion_aware = isinstance(strategy, CongestionAwareStrategy)

    def run(self, nets: List[Net]) -> RoutingSolution:
        """
        执行完整的布线流程（含拆线重布）。

        参数:
            nets: 所有需要布线的线网
        返回:
            RoutingSolution 布线方案
        """
        solution = RoutingSolution()
        max_iter = self.strategy.get_max_iterations()

        # 按策略排序线网
        ordered_nets = self.strategy.order_nets(nets)
        net_map = {net.name: net for net in nets}

        # 待布线队列
        pending = list(ordered_nets)
        # 记录每个线网被拆线的次数，避免无限循环
        ripup_count: Dict[str, int] = defaultdict(int)

        iteration = 0

        while pending and iteration < max_iter:
            iteration += 1
            logger.info(f"=== 布线迭代 {iteration}/{max_iter}，待布线 {len(pending)} 个线网 ===")

            next_pending = []
            progress_made = False

            for net in pending:
                result = self._route_single_net(net, solution, iteration)

                if result.success:
                    # 布线成功，标记间距
                    self.spacing_mgr.mark_route(net.name, result.routed_nodes)
                    solution.add_result(result)
                    progress_made = True
                    logger.info(f"线网 {net.name} 布线成功，代价 {result.total_cost:.2f}")
                else:
                    # 布线失败，尝试拆线重布
                    logger.warning(f"线网 {net.name} 布线失败，分析阻塞原因...")

                    blocking_nets = self._find_blocking_nets(net)

                    # 构建冲突区域并分析区域拆线可行性
                    region = build_conflict_region(
                        net, blocking_nets, self.spacing_mgr, self.grid,
                    )
                    region_ratios = analyze_regional_ripup(
                        net, blocking_nets, self.spacing_mgr,
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
                        # 区域拆线：只拆除阻塞线网在冲突区域内的片段
                        ripped_nets = self._execute_regional_ripup(
                            decision, solution, net_map, ripup_count, max_iter,
                        )

                        # 重新布线失败的线网
                        retry_result = self._route_single_net(net, solution, iteration)
                        if retry_result.success:
                            self.spacing_mgr.mark_route(net.name, retry_result.routed_nodes)
                            solution.add_result(retry_result)
                            progress_made = True
                            logger.info(f"线网 {net.name} 区域拆线后重布成功")
                        else:
                            next_pending.append(net)
                            logger.warning(f"线网 {net.name} 区域拆线后重布仍然失败")

                        # 被区域拆线影响的线网加入待布线队列（仅重连失败的）
                        next_pending.extend(ripped_nets)
                    else:
                        # 整网拆线 (RIPUP_SINGLE / RIPUP_MULTIPLE)
                        ripped_nets = []
                        for rn_name in decision.nets_to_ripup:
                            if rn_name in solution.results and solution.results[rn_name].success:
                                ripup_count[rn_name] += 1
                                # 防止某个线网被无限拆线
                                if ripup_count[rn_name] > max_iter // 2:
                                    logger.warning(
                                        f"线网 {rn_name} 已被拆除 {ripup_count[rn_name]} 次，跳过"
                                    )
                                    continue

                                self.spacing_mgr.unmark_route(rn_name)
                                solution.results[rn_name].success = False

                                # 记录拥塞
                                if self._is_congestion_aware:
                                    for node in solution.results[rn_name].routed_nodes:
                                        self.strategy.record_congestion(node)

                                ripped_nets.append(net_map[rn_name])
                                logger.info(f"已拆除线网 {rn_name}")

                        # 重新布线失败的线网
                        retry_result = self._route_single_net(net, solution, iteration)
                        if retry_result.success:
                            self.spacing_mgr.mark_route(net.name, retry_result.routed_nodes)
                            solution.add_result(retry_result)
                            progress_made = True
                            logger.info(f"线网 {net.name} 重布成功")
                        else:
                            next_pending.append(net)
                            logger.warning(f"线网 {net.name} 重布仍然失败")

                        # 被拆除的线网加入待布线队列
                        next_pending.extend(ripped_nets)

            pending = next_pending

            if not progress_made and pending:
                logger.warning(f"本轮迭代无进展，剩余 {len(pending)} 个线网未布通")
                # 如果完全没有进展，尝试调整策略
                if iteration >= max_iter:
                    break

        # 处理最终仍未布通的线网
        for net in pending:
            if net.name not in solution.results or not solution.results[net.name].success:
                fail_result = RoutingResult(net=net, success=False)
                solution.add_result(fail_result)

        self._log_summary(solution, iteration)
        return solution

    def _route_single_net(
        self, net: Net, solution: RoutingSolution, iteration: int
    ) -> RoutingResult:
        """
        布线单个线网，根据策略选择贪心或DP方法。

        策略通过 get_steiner_method(net, iteration) 决定使用哪种方法：
        - GREEDY: 增量贪心法，需要terminal_order
        - DP: Dreyfus-Wagner精确DP，自行处理端口顺序

        两种方法都会传入 cost_multiplier 和 congestion_map，
        以便在拆线重布循环中正确反映拥塞代价和间距约束。
        """
        # 获取拥塞映射（如果策略支持）
        congestion_map = None
        cost_multiplier = None
        if self._is_congestion_aware:
            congestion_map = self.strategy.get_congestion_map()
            cost_multiplier = self.strategy.get_cost_multiplier
        elif hasattr(self.strategy, 'get_cost_multiplier'):
            cost_multiplier = self.strategy.get_cost_multiplier

        method = self.strategy.get_steiner_method(net, iteration)

        if method == SteinerMethod.DP:
            result = self.steiner_builder.build_tree_dp(
                net=net,
                cost_multiplier=cost_multiplier,
                congestion_map=congestion_map,
            )
            # DP失败时回退到贪心（可能是节点过多导致内存问题等）
            if not result.success:
                logger.info(
                    f"线网 {net.name}: DP方法失败，回退到贪心方法"
                )
                terminal_order = self.strategy.order_terminals(net, set())
                result = self.steiner_builder.build_tree(
                    net=net,
                    terminal_order=terminal_order,
                    cost_multiplier=cost_multiplier,
                    congestion_map=congestion_map,
                )
        else:
            terminal_order = self.strategy.order_terminals(net, set())
            result = self.steiner_builder.build_tree(
                net=net,
                terminal_order=terminal_order,
                cost_multiplier=cost_multiplier,
                congestion_map=congestion_map,
            )

        return result

    def _find_blocking_nets(self, net: Net) -> Set[str]:
        """
        分析阻塞线网。

        使用SteinerRouter.find_blocked_info()统一分析，
        与MazeRouter.find_blocked_path_info()使用相同的约束检查逻辑。
        """
        router = SteinerRouter(self.grid, self.spacing_mgr)
        return router.find_blocked_info(net)

    def _execute_regional_ripup(
        self,
        decision,
        solution: RoutingSolution,
        net_map: Dict[str, Net],
        ripup_count: Dict[str, int],
        max_iter: int,
    ) -> List[Net]:
        """
        执行区域拆线：只拆除阻塞线网在冲突区域内的片段，然后尝试重连。

        流程:
        1. 对每个阻塞线网，找出其在冲突区域内的节点
        2. 调用 partial_unmark_route 清除这些节点的间距标记
        3. 从布线结果中移除这些节点和相关边
        4. 检查剩余节点是否仍然连通
        5. 如果断成多个连通分量，用MazeRouter重连
        6. 重连成功则更新结果；失败则将该线网加入待布线队列

        参数:
            decision: 拆线决策（含region信息）
            solution: 当前布线方案
            net_map: 线网名称到线网对象的映射
            ripup_count: 每个线网被拆除的次数
            max_iter: 最大迭代次数
        返回:
            需要重新完整布线的线网列表（重连失败的）
        """
        region = decision.region
        needs_full_reroute: List[Net] = []

        for rn_name in decision.nets_to_ripup:
            if rn_name not in solution.results or not solution.results[rn_name].success:
                continue

            ripup_count[rn_name] += 1
            if ripup_count[rn_name] > max_iter // 2:
                logger.warning(
                    f"线网 {rn_name} 已被拆除 {ripup_count[rn_name]} 次，跳过区域拆线"
                )
                continue

            result = solution.results[rn_name]
            nodes_in_region = find_nodes_in_region(result.routed_nodes, region)

            if not nodes_in_region:
                continue

            # 记录拥塞
            if self._is_congestion_aware:
                for node in nodes_in_region:
                    self.strategy.record_congestion(node)

            # 执行部分拆线
            self.spacing_mgr.partial_unmark_route(rn_name, nodes_in_region)

            # 更新布线结果：移除区域内节点和相关边
            remaining_nodes = result.routed_nodes - nodes_in_region
            remaining_edges = [
                (u, v) for u, v in result.routed_edges
                if u not in nodes_in_region and v not in nodes_in_region
            ]

            logger.info(
                f"线网 {rn_name} 区域拆线: 移除 {len(nodes_in_region)} 个节点，"
                f"剩余 {len(remaining_nodes)} 个节点"
            )

            if not remaining_nodes:
                # 所有节点都在区域内，等同于整网拆线
                result.success = False
                needs_full_reroute.append(net_map[rn_name])
                logger.info(f"线网 {rn_name} 全部节点在区域内，降级为整网拆线")
                continue

            # 检查剩余节点的连通性
            components = find_connected_components(remaining_nodes, remaining_edges)

            if len(components) <= 1:
                # 仍然连通，直接更新结果
                result.routed_nodes = remaining_nodes
                result.routed_edges = remaining_edges
                result.total_cost *= len(remaining_nodes) / (
                    len(remaining_nodes) + len(nodes_in_region)
                )
                logger.info(f"线网 {rn_name} 区域拆线后仍连通")
            else:
                # 断成多个分量，需要重连
                logger.info(
                    f"线网 {rn_name} 区域拆线后断成 {len(components)} 个分量，尝试重连"
                )
                reconnected = self._reconnect_net(
                    net_map[rn_name], components, remaining_edges, result,
                )
                if not reconnected:
                    # 重连失败，需要整网重布
                    self.spacing_mgr.partial_unmark_route(rn_name, remaining_nodes)
                    result.success = False
                    needs_full_reroute.append(net_map[rn_name])
                    logger.warning(f"线网 {rn_name} 重连失败，降级为整网重布")

        return needs_full_reroute

    def _reconnect_net(
        self,
        net: Net,
        components: List[Set[Node]],
        remaining_edges: List,
        result: RoutingResult,
    ) -> bool:
        """
        重连区域拆线后断开的线网子树。

        采用增量连接策略：选最大分量为初始树，
        依次用MazeRouter将其他分量连接到树上。

        参数:
            net: 线网对象
            components: 断开的连通分量列表
            remaining_edges: 剩余的边
            result: 当前布线结果（将被就地更新）
        返回:
            是否成功重连所有分量
        """
        # 获取拥塞映射
        congestion_map = None
        cost_multiplier = None
        if self._is_congestion_aware:
            congestion_map = self.strategy.get_congestion_map()
            cost_multiplier = self.strategy.get_cost_multiplier
        elif hasattr(self.strategy, 'get_cost_multiplier'):
            cost_multiplier = self.strategy.get_cost_multiplier

        router = MazeRouter(
            self.grid, self.spacing_mgr,
            cost_multiplier=cost_multiplier,
            congestion_map=congestion_map,
        )

        # 按大小降序排列，从最大分量开始
        components_sorted = sorted(components, key=len, reverse=True)
        tree_nodes = set(components_sorted[0])
        all_edges = list(remaining_edges)
        total_reconnect_cost = 0.0

        for comp in components_sorted[1:]:
            # 用MazeRouter从当前树到该分量找路径
            path_result = router.route(
                sources=tree_nodes,
                targets=comp,
                net_name=net.name,
                cable_locs=net.cable_locs,
            )

            if not path_result.success:
                return False

            # 合并重连路径
            tree_nodes.update(path_result.nodes)
            tree_nodes.update(comp)
            all_edges.extend(path_result.edges)
            total_reconnect_cost += path_result.cost

            # 标记新路径节点的间距
            new_nodes = set(path_result.nodes) - result.routed_nodes
            if new_nodes:
                self.spacing_mgr.partial_mark_route(net.name, new_nodes)

        # 更新布线结果
        result.routed_nodes = tree_nodes
        result.routed_edges = all_edges
        result.total_cost = result.total_cost + total_reconnect_cost
        result.success = True

        logger.info(
            f"线网 {net.name} 重连成功，重连代价 {total_reconnect_cost:.2f}"
        )
        return True

    @staticmethod
    def _log_summary(solution: RoutingSolution, iterations: int):
        """输出布线结果摘要"""
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
