"""
Router: 派发层

负责根据 RipupStrategy 决策调用 SteinerRouter（DP）或 MazeRouter（贪心），
并在路由完成后处理 Pin 引出逻辑。

Pin 引出逻辑（net.pin_spec 不为 None 时）：
  1. 检查 routed_nodes 中是否有满足目标层的节点 → 选最近的作为 pin_point
  2. 若无，找可通过 via 到达目标层的节点：
     a. 走 via 到目标层
     b. 从 via 点找第一个可用邻居
     c. 邻居作为 pin_point
"""

from __future__ import annotations
import logging
from typing import List, Optional, Set

from maze_router.data.net import Net, Node, RoutingResult
from maze_router.data.grid import GridGraph
from maze_router.constraint_manager import ConstraintManager
from maze_router.cost_manager import CostManager
from maze_router.ripup_strategy import RipupStrategy, RouterType
from maze_router.maze_router_algo import MazeRouter, build_steiner_greedy
from maze_router.steiner_router_algo import SteinerRouter

logger = logging.getLogger(__name__)

# pin 引出时要求 via 点周围的最小可用空间（Manhattan 邻居数，1 即够）
_PIN_MIN_NEIGHBORS = 1


class Router:
    """
    路由器派发层。

    根据 strategy.get_router_type() 选择 SteinerRouter（DP）或 MazeRouter（贪心），
    路由完成后执行 pin 引出。
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
        self._steiner = SteinerRouter(grid, constraint_mgr, cost_mgr)
        self._maze = MazeRouter(grid, constraint_mgr, cost_mgr)

    def route_net(self, net: Net, iteration: int = 1) -> RoutingResult:
        """
        布线单个线网，含路由器派发和 pin 引出。

        参数:
            net:       线网对象
            iteration: 当前迭代轮次
        返回:
            RoutingResult
        """
        router_type = self.strategy.get_router_type(net, iteration)

        if router_type == RouterType.STEINER_DP:
            result = self._steiner.route(net, iteration=iteration)
            # DP 失败时回退到贪心
            if not result.success:
                logger.info(f"线网 {net.name}: DP 失败，回退到贪心")
                terminal_order = self.strategy.order_terminals(net, set())
                result = build_steiner_greedy(
                    net, terminal_order,
                    self.grid, self.constraint_mgr, self.cost_mgr,
                    iteration=iteration,
                )
        else:
            terminal_order = self.strategy.order_terminals(net, set())
            result = build_steiner_greedy(
                net, terminal_order,
                self.grid, self.constraint_mgr, self.cost_mgr,
                iteration=iteration,
            )

        # Pin 引出
        if result.success and net.pin_spec is not None:
            self._extract_pin(net, result)

        # 后处理 hook（如 MinAreaConstraint 节点扩展）
        if result.success:
            self.constraint_mgr.post_process_results(net.name, result, self.grid)

        return result

    def find_blocked_nets(self, net: Net) -> Set[str]:
        """分析阻塞线网（使用 SteinerRouter 的 BFS 分析）。"""
        return self._steiner.find_blocked_nets(net)

    # ------------------------------------------------------------------
    # Pin 引出
    # ------------------------------------------------------------------

    def _extract_pin(self, net: Net, result: RoutingResult):
        """
        在布线结果中选取或生成 Pin 引出点。

        修改 result.pin_point（以及可能的 routed_nodes/routed_edges）。
        """
        pin_spec = net.pin_spec
        terminals = set(net.terminals)

        # 1. 检查 routed_nodes 中是否已有满足层要求的节点
        candidates = [
            n for n in result.routed_nodes
            if pin_spec.allows_layer(n[0])
        ]
        if candidates:
            # 选距离 terminal 最近的
            result.pin_point = min(
                candidates,
                key=lambda n: min(
                    abs(n[1] - t[1]) + abs(n[2] - t[2])
                    for t in terminals
                )
            )
            logger.debug(f"线网 {net.name}: pin 引出至 {result.pin_point}（已有节点）")
            return

        # 2. 寻找可通过 via 到达目标层的节点
        target_layers: List[str] = []
        if pin_spec.layer == "Any":
            target_layers = ["M1", "M2"]
        else:
            target_layers = [pin_spec.layer]

        for node in result.routed_nodes:
            for nb in self.grid.get_neighbors(node):
                if nb[0] not in target_layers:
                    continue
                if not self.constraint_mgr.is_available(nb, net.name):
                    continue

                # 检查 nb 的邻居中有无可用节点（保证 pin 周围有空间）
                via_neighbors = [
                    n for n in self.grid.get_neighbors(nb)
                    if n != node and self.constraint_mgr.is_available(n, net.name)
                ]
                if len(via_neighbors) < _PIN_MIN_NEIGHBORS:
                    continue

                # 找一个可用邻居作为 pin_point
                pin_point = via_neighbors[0]

                result.routed_nodes.add(nb)
                result.routed_nodes.add(pin_point)
                result.routed_edges.append((node, nb))
                result.routed_edges.append((nb, pin_point))
                result.pin_point = pin_point

                logger.debug(
                    f"线网 {net.name}: pin 引出 via {nb} → {pin_point}"
                )
                return

        logger.warning(f"线网 {net.name}: 无法找到 pin 引出点（层要求={pin_spec.layer}）")
