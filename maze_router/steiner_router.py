"""
Steiner树DP路由器模块

基于Dreyfus-Wagner动态规划算法的多端口Steiner树精确求解器。
与MazeRouter平行的路由器类，统一处理间距约束、cable_locs约束、拥塞代价和折角代价。

折角代价集成方案：
  Dreyfus-Wagner DP 分两阶段：
  1. 子集合并（split）：在同一节点合并两棵子树，无方向概念，不施加折角代价。
  2. Dijkstra松弛：沿边扩展路径，此阶段使用方向感知Dijkstra：
       状态 = (node_idx, incoming_dir_code)
       方向改变时叠加 corner_costs[layer]
  最终 dp[mask][v] 反映含折角代价的最优Steiner树代价。
"""

import heapq
import logging
from typing import Optional, Callable, Dict, Tuple, Set, List

from maze_router.net import Net, Node, Edge, RoutingResult
from maze_router.grid import RoutingGrid
from maze_router.spacing import SpacingManager
from maze_router.corner import CornerManager
from maze_router.router import (
    _move_dir_code, _DIR_NONE, _N_DIRS, _resolve_corner_mgr,
)

logger = logging.getLogger(__name__)


class SteinerRouter:
    """
    基于Dreyfus-Wagner DP的Steiner树路由器。

    与MazeRouter采用相同的构造参数和约束处理模式：
    - grid: 布线网格
    - spacing_mgr: 间距约束管理器
    - cost_multiplier: 代价乘数函数
    - congestion_map: 拥塞代价映射
    - corner_costs: 各层折角代价（None→默认5.0；{}→关闭）

    约束处理流程（类比MazeRouter的邻居扩展检查）：
    1. 构建约束子图：过滤不可用节点、计算有效边权
    2. 在约束子图上运行Dreyfus-Wagner DP（含方向感知Dijkstra松弛）
    3. 回溯重建Steiner树

    适用范围：端口数 k ≤ 8，时间复杂度 O(3^k·N + 2^k·N·log N)
    """

    def __init__(
        self,
        grid: RoutingGrid,
        spacing_mgr: SpacingManager,
        cost_multiplier: Optional[Callable[[Node, str], float]] = None,
        congestion_map: Optional[Dict[Node, float]] = None,
        corner_mgr: Optional[CornerManager] = None,
        corner_costs: Optional[Dict[str, float]] = None,
    ):
        """
        参数:
            grid: 布线网格
            spacing_mgr: 间距约束管理器
            cost_multiplier: 可选的代价乘数函数 (node, net_name) -> float
            congestion_map: 可选的拥塞代价映射 node -> extra_cost
            corner_mgr: 折角代价管理器（CornerManager），优先于 corner_costs。
            corner_costs: 向后兼容参数；仅在 corner_mgr=None 时生效。
        """
        self.grid = grid
        self.spacing_mgr = spacing_mgr
        self.cost_multiplier = cost_multiplier
        self.congestion_map = congestion_map or {}
        self.corner_mgr = _resolve_corner_mgr(corner_mgr, corner_costs)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def route(self, net: Net) -> RoutingResult:
        """
        为线网构建最优Steiner树，连接其所有端口。

        参数:
            net: 线网对象（包含terminals和cable_locs）
        返回:
            RoutingResult 布线结果
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

        # 步骤1: 构建约束子图
        valid_nodes, node_to_idx, adj = self._build_constrained_graph(net)
        n_nodes = len(valid_nodes)

        # 检查所有端口是否在约束子图中
        terminal_indices = []
        for terminal in terminals:
            if terminal not in node_to_idx:
                logger.warning(
                    f"线网 {net.name}: 端口 {terminal} 被间距约束或cable_locs排除"
                )
                result.success = False
                return result
            terminal_indices.append(node_to_idx[terminal])

        # 步骤2: 运行Dreyfus-Wagner DP（含方向感知Dijkstra）
        full_mask = (1 << k) - 1
        dp, parent = self._run_dp(
            k, n_nodes, terminal_indices, adj, full_mask,
            valid_nodes, self.corner_mgr,
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
            logger.warning(f"线网 {net.name}: DP求解失败，无法连接所有端口")
            result.success = False
            return result

        # 步骤4: 回溯重建Steiner树
        tree_nodes: Set[Node] = set()
        tree_edges: List[Edge] = []
        self._backtrace(
            full_mask, best_root,
            parent, valid_nodes,
            tree_nodes, tree_edges,
        )

        result.routed_nodes = tree_nodes
        result.routed_edges = tree_edges
        result.total_cost = best_cost
        result.success = True

        logger.info(
            f"线网 {net.name}: DP Steiner树构建成功，"
            f"共 {len(tree_nodes)} 个节点，代价 {best_cost:.2f}"
        )
        return result

    def find_blocked_info(self, net: Net) -> Set[str]:
        """
        分析哪些线网阻塞了端口之间的连接。

        遍历所有端口的邻域，收集被间距约束阻塞的线网名称。
        类比MazeRouter.find_blocked_path_info()。

        参数:
            net: 线网对象
        返回:
            阻塞线网名称集合
        """
        blocking_nets: Set[str] = set()
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

                if not self.spacing_mgr.is_available(neighbor, net.name):
                    blocking_nets.update(
                        self.spacing_mgr.get_blocking_nets(neighbor, net.name)
                    )
                    continue

                queue.append(neighbor)

        return blocking_nets


    def _build_constrained_graph(
        self, net: Net,
    ) -> Tuple[List[Node], Dict[Node, int], List[List[Tuple[int, float]]]]:
        """
        构建约束子图：过滤节点、计算边权（不含折角，折角在DP Dijkstra阶段计算）。

        返回:
            valid_nodes: 过滤后的有效节点列表
            node_to_idx: 节点到索引的映射
            adj: 邻接表 adj[i] = [(j, base_cost), ...]
        """
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
                cost = self._compute_edge_cost(node, neighbor, net.name)
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
        if not self.spacing_mgr.is_available(node, net_name):
            return False
        if cable_locs is not None and node[0] == "M0":
            if node not in cable_locs:
                return False
        return True

    def _compute_edge_cost(
        self, src: Node, dst: Node, net_name: str,
    ) -> float:
        """计算单条边的基础代价（不含折角，折角在Dijkstra阶段叠加）。"""
        cost = self.grid.get_edge_cost(src, dst)
        if self.cost_multiplier:
            cost *= self.cost_multiplier(dst, net_name)
        if dst in self.congestion_map:
            cost += self.congestion_map[dst]
        return cost


    @staticmethod
    def _run_dp(
        k: int,
        n_nodes: int,
        terminal_indices: List[int],
        adj: List[List[Tuple[int, float]]],
        full_mask: int,
        valid_nodes: List[Node],
        corner_mgr: CornerManager,
    ) -> Tuple[List[List[float]], List[List[Optional[tuple]]]]:
        """
        执行Dreyfus-Wagner DP（含折角感知Dijkstra松弛和T型折角分支惩罚）。

        状态: dp[S][v] = 以节点v为根、连接端口子集S的最小代价（含折角）
        转移:
            1. 子集合并（T型折角）:
               dp[S][v] = min{ dp[S'][v] + dp[S\\S'][v] + t_corner(v) }
               在节点v处合并两棵子树，若T型折角代价>0则施加分支惩罚。
            2. 方向感知Dijkstra松弛（L型折角）:
               Dijkstra状态 (v, dir_code)，方向改变时叠加 L_corner(layer, d_in, d_out)
               最终 dp[S][v] = min_dir{ dist_dir[v][dir] }

        参数:
            k: 端口数
            n_nodes: 约束子图节点数
            terminal_indices: 各端口在约束子图中的索引
            adj: 约束子图邻接表（基础代价，不含折角）
            full_mask: 全端口掩码 (1<<k)-1
            valid_nodes: 索引→节点的映射（用于计算方向和层名）
            corner_mgr: 折角代价管理器
        返回:
            (dp表, parent回溯表)
        """
        INF = float('inf')
        dp: List[List[float]] = [[INF] * n_nodes for _ in range(1 << k)]
        parent: List[List[Optional[tuple]]] = [
            [None] * n_nodes for _ in range(1 << k)
        ]

        # 基础情况：单端口子集，代价为0
        for i, tidx in enumerate(terminal_indices):
            dp[1 << i][tidx] = 0.0

        for mask in range(1, full_mask + 1):
            if mask & full_mask != mask:
                continue

            # ---- 步骤1: 子集合并（含T型折角分支惩罚）--------------------
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
                            # T型折角代价：在节点v处合并两棵子树（分支惩罚）
                            t_cp = corner_mgr.get_t_cost_flat(valid_nodes[v][0])
                            val = dp_sub[v] + dp_comp[v] + t_cp
                            if val < dp_mask[v]:
                                dp_mask[v] = val
                                parent_mask[v] = ("split", sub, comp)
                    sub = (sub - 1) & mask

            # ---- 步骤2: 方向感知Dijkstra松弛 ----------------------------
            dp_mask = dp[mask]
            parent_mask = parent[mask]

            # dist_dir[v][d]: 以方向码d到达节点v的最小代价
            # pred_v[v][d]:   在最优路径中v的前驱节点索引（-1=无）
            dist_dir: List[List[float]] = [[INF] * _N_DIRS for _ in range(n_nodes)]
            pred_v: List[List[int]] = [[-1] * _N_DIRS for _ in range(n_nodes)]

            pq: List = []
            dijk_counter = 0

            # 以当前dp_mask的有限值为初始点（方向=_DIR_NONE，表示子树根，无来向）
            for v in range(n_nodes):
                if dp_mask[v] < INF:
                    dist_dir[v][_DIR_NONE] = dp_mask[v]
                    heapq.heappush(pq, (dp_mask[v], dijk_counter, v, _DIR_NONE))
                    dijk_counter += 1

            while pq:
                cost_v, _, v, d_in = heapq.heappop(pq)
                if cost_v > dist_dir[v][d_in]:
                    continue
                v_node = valid_nodes[v]
                for u, base_cost in adj[v]:
                    u_node = valid_nodes[u]
                    d_out = _move_dir_code(v_node, u_node)

                    # 折角代价：同层 && 有来向 && 方向改变
                    cp = 0.0
                    if (d_in != _DIR_NONE
                            and d_out != _DIR_NONE
                            and d_in != d_out):
                        cp = corner_mgr.get_l_cost(v_node[0], d_in, d_out)

                    new_cost = cost_v + base_cost + cp
                    if new_cost < dist_dir[u][d_out]:
                        dist_dir[u][d_out] = new_cost
                        pred_v[u][d_out] = v
                        heapq.heappush(pq, (new_cost, dijk_counter, u, d_out))
                        dijk_counter += 1

            # 用方向感知结果更新 dp_mask 和 parent_mask
            for v in range(n_nodes):
                best_cost_v = min(dist_dir[v])
                if best_cost_v < dp_mask[v]:
                    dp_mask[v] = best_cost_v
                    best_d = dist_dir[v].index(best_cost_v)
                    pv = pred_v[v][best_d]
                    if pv >= 0:
                        parent_mask[v] = ("edge", mask, pv)

        return dp, parent


    def _backtrace(
        self,
        mask: int,
        v_idx: int,
        parent: List[List[Optional[tuple]]],
        valid_nodes: List[Node],
        tree_nodes: Set[Node],
        tree_edges: List[Edge],
    ):
        """递归回溯DP解，重建Steiner树。"""
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
