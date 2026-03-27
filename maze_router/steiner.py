"""
Steiner树构建调度模块

SteinerTreeBuilder 作为统一调度器，根据方法选择委托给：
- MazeRouter: 增量贪心法（build_tree）
- SteinerRouter: Dreyfus-Wagner DP法（build_tree_dp）

两种路由器具有相同的约束处理接口（spacing、cable_locs、congestion）。
"""

import logging
from typing import Optional, Callable, Dict, Set

from maze_router.net import Net, Node, RoutingResult
from maze_router.grid import RoutingGrid
from maze_router.spacing import SpacingManager
from maze_router.router import MazeRouter
from maze_router.steiner_router import SteinerRouter

logger = logging.getLogger(__name__)


class SteinerTreeBuilder:
    """
    Steiner树构建调度器。

    提供两种构建方法，统一管理约束参数的传递：
    - build_tree: 委托MazeRouter的增量贪心法
    - build_tree_dp: 委托SteinerRouter的Dreyfus-Wagner DP法
    """

    def __init__(self, grid: RoutingGrid, spacing_mgr: SpacingManager):
        self.grid = grid
        self.spacing_mgr = spacing_mgr

    def build_tree(
        self,
        net: Net,
        terminal_order: list,
        cost_multiplier: Optional[Callable[[Node, str], float]] = None,
        congestion_map: Optional[Dict[Node, float]] = None,
    ) -> RoutingResult:
        """
        增量贪心法构建Steiner树。

        委托MazeRouter逐步将端口连接到已有树上。

        算法流程:
        1. 第一个端口作为初始树 T
        2. 每次选择离T最近的未连接端口
        3. 以T上所有节点为源，运行MazeRouter到该端口
        4. 将路径并入T

        参数:
            net: 线网对象
            terminal_order: 端口连接顺序（由策略类提供）
            cost_multiplier: 可选的代价乘数函数
            congestion_map: 可选的拥塞代价映射
        返回:
            RoutingResult 布线结果
        """
        result = RoutingResult(net=net)

        if len(terminal_order) == 0:
            result.success = True
            return result

        if len(terminal_order) == 1:
            result.routed_nodes.add(terminal_order[0])
            result.success = True
            return result

        router = MazeRouter(
            self.grid,
            self.spacing_mgr,
            cost_multiplier=cost_multiplier,
            congestion_map=congestion_map,
        )

        # 第一个端口作为初始树
        tree_nodes = {terminal_order[0]}
        result.routed_nodes.add(terminal_order[0])
        remaining = list(terminal_order[1:])

        while remaining:
            # 动态排序：选择离当前树估计最近的端口
            remaining.sort(key=lambda t: _min_manhattan_to_set(t, tree_nodes))

            target_terminal = remaining.pop(0)

            # 以整棵树为源，目标端口为目标
            path_result = router.route(
                sources=tree_nodes,
                targets={target_terminal},
                net_name=net.name,
                cable_locs=net.cable_locs,
            )

            if not path_result.success:
                logger.warning(
                    f"线网 {net.name}: 无法连接端口 {target_terminal}，"
                    f"已连接 {len(terminal_order) - len(remaining) - 1}"
                    f"/{len(terminal_order)} 个端口"
                )
                result.success = False
                return result

            # 合并路径到树中
            result.merge(path_result.nodes, path_result.edges, path_result.cost)
            tree_nodes.update(path_result.nodes)

        result.success = True
        logger.info(
            f"线网 {net.name}: 贪心Steiner树构建成功，"
            f"共 {len(result.routed_nodes)} 个节点，代价 {result.total_cost:.2f}"
        )
        return result

    def build_tree_dp(
        self,
        net: Net,
        cost_multiplier: Optional[Callable[[Node, str], float]] = None,
        congestion_map: Optional[Dict[Node, float]] = None,
    ) -> RoutingResult:
        """
        Dreyfus-Wagner DP法构建最优Steiner树。

        委托SteinerRouter处理约束子图构建和DP求解。
        两端口时退化为MazeRouter最短路径。

        参数:
            net: 线网对象
            cost_multiplier: 可选的代价乘数函数
            congestion_map: 可选的拥塞代价映射
        返回:
            RoutingResult 布线结果
        """
        # 两端口退化为最短路径，MazeRouter更高效
        if len(net.terminals) <= 2:
            return self.build_tree(
                net, list(net.terminals),
                cost_multiplier=cost_multiplier,
                congestion_map=congestion_map,
            )

        router = SteinerRouter(
            self.grid,
            self.spacing_mgr,
            cost_multiplier=cost_multiplier,
            congestion_map=congestion_map,
        )
        return router.route(net)


def _min_manhattan_to_set(target: Node, node_set: Set[Node]) -> float:
    """计算目标节点到节点集合的最小曼哈顿距离"""
    tx, ty = target[1], target[2]
    return min(abs(tx - n[1]) + abs(ty - n[2]) for n in node_set)
