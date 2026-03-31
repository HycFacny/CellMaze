"""
MazeRouter: 增量式方向感知 A* 迷宫路由器

基于多源 Dijkstra/A* 算法从一组源节点寻找到一组目标节点的最短路径。
同时提供增量 Steiner 树构建（贪心逐一连接 terminal）。

方向编码（模块级常量，供其他模块导入）：
  DIR_NONE = 0  — 初始/via后，无方向
  1=右(+x), 2=左(-x), 3=上(+y), 4=下(-y)
"""

from __future__ import annotations
import heapq
from typing import Dict, List, Optional, Set, Tuple

from maze_router.data.net import Node, Edge, Net, RoutingResult
from maze_router.data.grid import GridGraph
from maze_router.constraint_manager import ConstraintManager
from maze_router.cost_manager import CostManager, RoutingContext


# -----------------------------------------------------------------------
# 方向工具（模块级，供 steiner_router_algo 共用）
# -----------------------------------------------------------------------

DIR_NONE = 0
_DIR_MAP: Dict[Tuple[int, int], int] = {
    (1,  0): 1,   # 右
    (-1, 0): 2,   # 左
    (0,  1): 3,   # 上
    (0, -1): 4,   # 下
}
N_DIRS = 5


def move_dir_code(src: Node, dst: Node) -> int:
    """
    计算从 src 到 dst 的移动方向编码。

    跨层 via（层名不同）返回 DIR_NONE（方向重置）；
    同层移动返回 1-4 对应方向码。
    """
    if src[0] != dst[0]:
        return DIR_NONE
    dx, dy = dst[1] - src[1], dst[2] - src[2]
    return _DIR_MAP.get((dx, dy), DIR_NONE)


# -----------------------------------------------------------------------
# 单次路径搜索结果
# -----------------------------------------------------------------------

class PathResult:
    """单次路径搜索的结果。"""

    def __init__(
        self,
        nodes: List[Node],
        edges: List[Edge],
        cost: float,
        success: bool,
    ):
        self.nodes = nodes
        self.edges = edges
        self.cost = cost
        self.success = success

    @staticmethod
    def failure() -> 'PathResult':
        return PathResult([], [], 0.0, False)


# -----------------------------------------------------------------------
# MazeRouter: 单路径 A*
# -----------------------------------------------------------------------

class MazeRouter:
    """
    增量式方向感知 A* 迷宫路由器。

    Dijkstra 状态: (node, incoming_dir)
      - 所有源节点初始方向为 DIR_NONE
      - 跨层 via 后方向重置为 DIR_NONE
      - 同层移动方向改变时，叠加 cost_mgr.get_corner_l_cost()

    约束检查:
      - constraint_mgr.is_available()
      - M0 cable_locs 约束
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

    def route(
        self,
        sources: Set[Node],
        targets: Set[Node],
        net_name: str,
        cable_locs: Optional[Set[Node]] = None,
        use_astar: bool = True,
        iteration: int = 1,
    ) -> PathResult:
        """
        从源节点集合寻找到目标节点集合的最短路径。

        参数:
            sources:    源节点集合（已布线树节点）
            targets:    目标节点集合
            net_name:   当前线网名
            cable_locs: M0 层可用节点集合（None=不限制）
            use_astar:  是否使用 A* 启发式
            iteration:  当前迭代轮次（影响拥塞代价）
        """
        if not sources or not targets:
            return PathResult.failure()

        overlap = sources & targets
        if overlap:
            node = next(iter(overlap))
            return PathResult([node], [], 0.0, True)

        ctx = self.cost_mgr.make_context(
            net_name=net_name,
            constraint_mgr=self.constraint_mgr,
            iteration=iteration,
        )

        target_coords = [(t[1], t[2]) for t in targets]

        def heuristic(node: Node) -> float:
            if not use_astar:
                return 0.0
            x, y = node[1], node[2]
            return min(abs(x - tx) + abs(y - ty) for tx, ty in target_coords)

        counter = 0
        pq: List = []
        dist: Dict[Tuple[Node, int], float] = {}
        prev: Dict[Tuple[Node, int], Tuple[Node, int]] = {}

        for s in sources:
            if not self.grid.is_valid_node(s):
                continue
            state = (s, DIR_NONE)
            if state not in dist:
                dist[state] = 0.0
                h = heuristic(s)
                heapq.heappush(pq, (h, counter, 0.0, s, DIR_NONE))
                counter += 1

        found_state: Optional[Tuple[Node, int]] = None

        while pq:
            est_cost, _, actual_cost, current, dir_in = heapq.heappop(pq)
            state = (current, dir_in)

            if actual_cost > dist.get(state, float('inf')):
                continue

            if current in targets:
                found_state = state
                break

            for neighbor in self.grid.get_neighbors(current):
                # 约束检查（源集合内节点属于本线网）
                if neighbor not in sources:
                    if not self.constraint_mgr.is_available(neighbor, net_name):
                        continue

                # M0 cable_locs 约束
                if cable_locs is not None and neighbor[0] == "M0":
                    if neighbor not in cable_locs and neighbor not in targets:
                        continue

                dir_out = move_dir_code(current, neighbor)

                # L 型折角代价
                cp = 0.0
                if dir_in != DIR_NONE and dir_out != DIR_NONE and dir_in != dir_out:
                    cp = self.cost_mgr.get_corner_l_cost(current[0], dir_in, dir_out)

                edge_cost = self.cost_mgr.get_edge_cost(current, neighbor, ctx)
                new_cost = actual_cost + edge_cost + cp
                new_state = (neighbor, dir_out)

                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    prev[new_state] = state
                    h = heuristic(neighbor)
                    heapq.heappush(
                        pq, (new_cost + h, counter, new_cost, neighbor, dir_out)
                    )
                    counter += 1

        if found_state is None:
            return PathResult.failure()

        # 回溯路径
        path_nodes: List[Node] = []
        path_edges: List[Edge] = []
        cur_state = found_state
        while cur_state in prev:
            node = cur_state[0]
            path_nodes.append(node)
            par_state = prev[cur_state]
            path_edges.append((par_state[0], node))
            cur_state = par_state
        path_nodes.append(cur_state[0])
        path_nodes.reverse()
        path_edges.reverse()

        return PathResult(
            nodes=path_nodes,
            edges=path_edges,
            cost=dist[found_state],
            success=True,
        )

    def find_blocked_nets(
        self,
        sources: Set[Node],
        targets: Set[Node],
        net_name: str,
        cable_locs: Optional[Set[Node]] = None,
    ) -> Set[str]:
        """分析阻塞路径的线网集合（BFS，不含折角逻辑）。"""
        blocking: Set[str] = set()
        visited: Set[Node] = set()
        queue = list(sources)

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            for neighbor in self.grid.get_neighbors(current):
                if neighbor in visited:
                    continue
                if cable_locs is not None and neighbor[0] == "M0":
                    if neighbor not in cable_locs and neighbor not in targets:
                        continue
                if neighbor not in sources:
                    if not self.constraint_mgr.is_available(neighbor, net_name):
                        blocking.update(
                            self.constraint_mgr.get_blocking_nets(neighbor, net_name)
                        )
                        continue
                queue.append(neighbor)

        return blocking


# -----------------------------------------------------------------------
# 增量贪心 Steiner 树构建
# -----------------------------------------------------------------------

def build_steiner_greedy(
    net: Net,
    terminal_order: List[Node],
    grid: GridGraph,
    constraint_mgr: ConstraintManager,
    cost_mgr: CostManager,
    iteration: int = 1,
) -> RoutingResult:
    """
    增量贪心法构建 Steiner 树：逐一将 terminal 连接到当前树。

    参数:
        net:            线网对象
        terminal_order: 端口布线顺序
        grid, constraint_mgr, cost_mgr: 路由基础设施
        iteration:      当前迭代（影响拥塞代价）
    返回:
        RoutingResult
    """
    result = RoutingResult(net=net)
    terminals = terminal_order or list(net.terminals)

    if not terminals:
        result.success = True
        return result

    if len(terminals) == 1:
        result.routed_nodes.add(terminals[0])
        result.success = True
        return result

    router = MazeRouter(grid, constraint_mgr, cost_mgr)

    # 以第一个 terminal 为起点
    tree_nodes: Set[Node] = {terminals[0]}
    total_cost = 0.0

    for target in terminals[1:]:
        path = router.route(
            sources=tree_nodes,
            targets={target},
            net_name=net.name,
            cable_locs=net.cable_locs,
            iteration=iteration,
        )
        if not path.success:
            result.success = False
            return result

        tree_nodes.update(path.nodes)
        result.routed_edges.extend(path.edges)
        total_cost += path.cost

    result.routed_nodes = tree_nodes
    result.total_cost = total_cost
    result.success = True
    return result
