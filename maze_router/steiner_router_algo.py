"""
SteinerRouter: Dreyfus-Wagner DP Steiner 树路由器

基于精确 DP 算法（Dreyfus-Wagner）的多端口 Steiner 树求解器。
适用于 terminal 数量 ≤ 10（默认 ≤ 8）的线网。

折角代价集成方案：
  1. 子集合并（split）：在节点 v 处合并两棵子树，施加 T 型折角代价。
  2. Dijkstra 松弛：沿边扩展，方向改变时叠加 L 型折角代价。
  最终 dp[mask][v] 反映含折角代价的最优 Steiner 树代价。
"""

from __future__ import annotations
import heapq
import logging
from typing import Dict, List, Optional, Set, Tuple

from maze_router.data.net import Node, Edge, Net, RoutingResult
from maze_router.data.grid import GridGraph
from maze_router.constraint_manager import ConstraintManager
from maze_router.cost_manager import CostManager, RoutingContext
from maze_router.maze_router_algo import move_dir_code, DIR_NONE, N_DIRS

logger = logging.getLogger(__name__)


class SteinerRouter:
    """
    基于 Dreyfus-Wagner DP 的 Steiner 树路由器。

    时间复杂度: O(3^k·N + 2^k·N·log N)，k = terminal 数量，N = 节点数。
    """

    def __init__(
        self,
        grid: GridGraph,
        constraint_mgr: ConstraintManager,
        cost_mgr: CostManager,
    ):
        self.grid = grid
        self.constraint_mgr = constraint_mgr
        self.cost_mgr = cost_mgr

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def route(self, net: Net, iteration: int = 1) -> RoutingResult:
        """
        为线网构建最优 Steiner 树，连接其所有端口。

        参数:
            net:       线网对象
            iteration: 当前迭代轮次
        返回:
            RoutingResult
        """
        result = RoutingResult(net=net)
        terminals = list(net.terminals)
        k = len(terminals)

        if k == 0:
            result.success = True
            return result

        if k == 1:
            result.routed_nodes.add(terminals[0])
            result.success = True
            return result

        ctx = self.cost_mgr.make_context(
            net_name=net.name,
            constraint_mgr=self.constraint_mgr,
            iteration=iteration,
        )

        # 步骤1: 构建约束子图
        valid_nodes, node_to_idx, adj = self._build_constrained_graph(net, ctx)
        n_nodes = len(valid_nodes)

        # 检查所有端口是否在约束子图中
        terminal_indices = []
        for terminal in terminals:
            if terminal not in node_to_idx:
                logger.warning(
                    f"线网 {net.name}: 端口 {terminal} 被约束排除"
                )
                result.success = False
                return result
            terminal_indices.append(node_to_idx[terminal])

        # 步骤2: Dreyfus-Wagner DP
        full_mask = (1 << k) - 1
        dp, parent = self._run_dp(
            k, n_nodes, terminal_indices, adj, full_mask,
            valid_nodes, self.cost_mgr,
        )

        # 步骤3: 找最优根节点
        INF = float('inf')
        best_cost = INF
        best_root = -1
        for v in range(n_nodes):
            if dp[full_mask][v] < best_cost:
                best_cost = dp[full_mask][v]
                best_root = v

        if best_root < 0 or best_cost >= INF:
            logger.warning(f"线网 {net.name}: DP 求解失败")
            result.success = False
            return result

        # 步骤4: 回溯重建 Steiner 树
        tree_nodes: Set[Node] = set()
        tree_edges: List[Edge] = []
        self._backtrace(full_mask, best_root, parent, valid_nodes, tree_nodes, tree_edges)

        result.routed_nodes = tree_nodes
        result.routed_edges = tree_edges
        result.total_cost = best_cost
        result.success = True

        logger.info(
            f"线网 {net.name}: DP 成功，{len(tree_nodes)} 个节点，代价 {best_cost:.2f}"
        )
        return result

    def find_blocked_nets(self, net: Net) -> Set[str]:
        """分析阻塞线网集合（BFS）。"""
        blocking: Set[str] = set()
        terminal_set = set(net.terminals)
        visited: Set[Node] = set()
        queue = list(net.terminals)

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            for neighbor in self.grid.get_neighbors(current):
                if neighbor in visited:
                    continue
                if net.cable_locs is not None and neighbor[0] == "M0":
                    if neighbor not in net.cable_locs and neighbor not in terminal_set:
                        continue
                if not self.constraint_mgr.is_available(neighbor, net.name):
                    blocking.update(
                        self.constraint_mgr.get_blocking_nets(neighbor, net.name)
                    )
                    continue
                queue.append(neighbor)

        return blocking

    # ------------------------------------------------------------------
    # 约束子图构建
    # ------------------------------------------------------------------

    def _build_constrained_graph(
        self, net: Net, ctx: RoutingContext,
    ) -> Tuple[List[Node], Dict[Node, int], List[List[Tuple[int, float]]]]:
        terminal_set = set(net.terminals)
        all_nodes = self.grid.get_all_nodes()

        valid_nodes: List[Node] = []
        for node in all_nodes:
            if self._is_node_available(node, net.name, terminal_set, net.cable_locs):
                valid_nodes.append(node)

        valid_set = set(valid_nodes)
        node_to_idx: Dict[Node, int] = {n: i for i, n in enumerate(valid_nodes)}
        n_nodes = len(valid_nodes)

        adj: List[List[Tuple[int, float]]] = [[] for _ in range(n_nodes)]
        for i, node in enumerate(valid_nodes):
            for neighbor in self.grid.get_neighbors(node):
                if neighbor not in valid_set:
                    continue
                j = node_to_idx[neighbor]
                cost = self.cost_mgr.get_edge_cost(node, neighbor, ctx)
                adj[i].append((j, cost))

        return valid_nodes, node_to_idx, adj

    def _is_node_available(
        self,
        node: Node,
        net_name: str,
        terminal_set: Set[Node],
        cable_locs: Optional[Set[Node]],
    ) -> bool:
        if node in terminal_set:
            return True
        if not self.constraint_mgr.is_available(node, net_name):
            return False
        if cable_locs is not None and node[0] == "M0":
            if node not in cable_locs:
                return False
        return True

    # ------------------------------------------------------------------
    # Dreyfus-Wagner DP
    # ------------------------------------------------------------------

    @staticmethod
    def _run_dp(
        k: int,
        n_nodes: int,
        terminal_indices: List[int],
        adj: List[List[Tuple[int, float]]],
        full_mask: int,
        valid_nodes: List[Node],
        cost_mgr: CostManager,
    ) -> Tuple[List[List[float]], List[List[Optional[tuple]]]]:
        """
        执行 Dreyfus-Wagner DP（含折角感知 Dijkstra 松弛和 T 型折角分支惩罚）。

        dp[S][v]: 以节点 v 为根、连接端口子集 S 的最小代价
        转移:
          1. 子集合并（T 型折角）: dp[S][v] = dp[S'][v] + dp[S\\S'][v] + t_corner(v)
          2. Dijkstra 松弛（L 型折角）: 方向改变时叠加 L_corner(layer, d_in, d_out)
        """
        INF = float('inf')
        dp: List[List[float]] = [[INF] * n_nodes for _ in range(1 << k)]
        parent: List[List[Optional[tuple]]] = [
            [None] * n_nodes for _ in range(1 << k)
        ]

        # 基础情况
        for i, tidx in enumerate(terminal_indices):
            dp[1 << i][tidx] = 0.0

        for mask in range(1, full_mask + 1):
            if mask & full_mask != mask:
                continue

            # 步骤1: 子集合并（T 型折角分支惩罚）
            if bin(mask).count('1') >= 2:
                sub = (mask - 1) & mask
                while sub > 0:
                    comp = mask ^ sub
                    if sub < comp:
                        dp_sub = dp[sub]
                        dp_comp = dp[comp]
                        dp_mask = dp[mask]
                        parent_mask = parent[mask]
                        for v in range(n_nodes):
                            t_cp = cost_mgr.get_corner_t_cost_flat(valid_nodes[v][0])
                            val = dp_sub[v] + dp_comp[v] + t_cp
                            if val < dp_mask[v]:
                                dp_mask[v] = val
                                parent_mask[v] = ("split", sub, comp)
                    sub = (sub - 1) & mask

            # 步骤2: 方向感知 Dijkstra 松弛
            dp_mask = dp[mask]
            parent_mask = parent[mask]

            dist_dir: List[List[float]] = [[INF] * N_DIRS for _ in range(n_nodes)]
            pred_v: List[List[int]] = [[-1] * N_DIRS for _ in range(n_nodes)]

            pq: List = []
            dijk_counter = 0

            for v in range(n_nodes):
                if dp_mask[v] < INF:
                    dist_dir[v][DIR_NONE] = dp_mask[v]
                    heapq.heappush(pq, (dp_mask[v], dijk_counter, v, DIR_NONE))
                    dijk_counter += 1

            while pq:
                cost_v, _, v, d_in = heapq.heappop(pq)
                if cost_v > dist_dir[v][d_in]:
                    continue
                v_node = valid_nodes[v]
                for u, base_cost in adj[v]:
                    u_node = valid_nodes[u]
                    d_out = move_dir_code(v_node, u_node)

                    cp = 0.0
                    if d_in != DIR_NONE and d_out != DIR_NONE and d_in != d_out:
                        cp = cost_mgr.get_corner_l_cost(v_node[0], d_in, d_out)

                    new_cost = cost_v + base_cost + cp
                    if new_cost < dist_dir[u][d_out]:
                        dist_dir[u][d_out] = new_cost
                        pred_v[u][d_out] = v
                        heapq.heappush(pq, (new_cost, dijk_counter, u, d_out))
                        dijk_counter += 1

            # 用 Dijkstra 结果更新 dp_mask
            for v in range(n_nodes):
                best_cost_v = min(dist_dir[v])
                if best_cost_v < dp_mask[v]:
                    dp_mask[v] = best_cost_v
                    best_d = dist_dir[v].index(best_cost_v)
                    pv = pred_v[v][best_d]
                    if pv >= 0:
                        parent_mask[v] = ("edge", mask, pv)

        return dp, parent

    # ------------------------------------------------------------------
    # 回溯重建 Steiner 树
    # ------------------------------------------------------------------

    def _backtrace(
        self,
        mask: int,
        v_idx: int,
        parent: List[List[Optional[tuple]]],
        valid_nodes: List[Node],
        tree_nodes: Set[Node],
        tree_edges: List[Edge],
    ):
        """递归回溯 DP 解，重建 Steiner 树。"""
        tree_nodes.add(valid_nodes[v_idx])

        p = parent[mask][v_idx]
        if p is None:
            return

        if p[0] == "split":
            _, mask1, mask2 = p
            self._backtrace(mask1, v_idx, parent, valid_nodes, tree_nodes, tree_edges)
            self._backtrace(mask2, v_idx, parent, valid_nodes, tree_nodes, tree_edges)

        elif p[0] == "edge":
            _, prev_mask, prev_idx = p
            tree_nodes.add(valid_nodes[prev_idx])
            tree_edges.append((valid_nodes[prev_idx], valid_nodes[v_idx]))
            self._backtrace(prev_mask, prev_idx, parent, valid_nodes, tree_nodes, tree_edges)
