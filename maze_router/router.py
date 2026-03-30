"""
核心迷宫路由器模块

基于多源Dijkstra/A*算法的单路径布线引擎。
支持间距约束、cable_locs约束、可配置代价函数和折角(corner)代价。

折角代价说明：
  折角发生在同层移动方向改变时（如由横向变纵向）。跨层 via 会重置方向，
  不触发折角惩罚。路由器通过将 (node, incoming_dir) 作为 Dijkstra 状态来
  精确计算折角代价，而非用平均值近似。

方向编码（模块级常量，供 steiner_router 共用）：
  _DIR_NONE = 0  —— 初始/via后，无方向
  1=向右, 2=向左, 3=向上, 4=向下
"""

import heapq
from typing import List, Set, Tuple, Optional, Callable, Dict

from maze_router.net import Node, Edge
from maze_router.grid import RoutingGrid
from maze_router.spacing import SpacingManager
from maze_router.corner import CornerManager


# ======================================================================
# 方向工具（模块级，供 steiner_router 导入复用）
# ======================================================================

_DIR_NONE = 0
_DIR_MAP: Dict[Tuple[int, int], int] = {
    (1,  0): 1,
    (-1, 0): 2,
    (0,  1): 3,
    (0, -1): 4,
}
_N_DIRS = 5

DEFAULT_CORNER_COST = 5.0


def _move_dir_code(src: Node, dst: Node) -> int:
    """
    计算从 src 到 dst 的移动方向编码。

    跨层 via（层名不同）返回 _DIR_NONE，表示方向重置；
    同层移动返回 1-4 对应方向码。
    """
    if src[0] != dst[0]:
        return _DIR_NONE
    dx, dy = dst[1] - src[1], dst[2] - src[2]
    return _DIR_MAP.get((dx, dy), _DIR_NONE)


def _resolve_corner_mgr(
    corner_mgr: Optional[CornerManager],
    corner_costs: Optional[Dict[str, float]] = None,
) -> CornerManager:
    """
    解析折角代价参数，返回 CornerManager 实例。

    优先使用 corner_mgr；若未提供则从向后兼容的 corner_costs 字典构建。
    - corner_mgr 非 None → 直接使用
    - corner_mgr=None, corner_costs=None → 默认 CornerManager（L=5.0，T=0）
    - corner_mgr=None, corner_costs={} → 禁用所有折角代价
    - corner_mgr=None, corner_costs={...} → 按层L代价，无T代价
    """
    if corner_mgr is not None:
        return corner_mgr
    return CornerManager(l_costs=corner_costs, t_costs={} if corner_costs is not None else None)


# 向后兼容别名（供 steiner_router、steiner、ripup 等旧代码调用）
def _resolve_corner_costs(
    corner_costs: Optional[Dict[str, float]]
) -> 'CornerManager':
    """向后兼容：从 Dict[str, float] 构建 CornerManager。"""
    return _resolve_corner_mgr(None, corner_costs)


class PathResult:
    """单次路径搜索的结果"""

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


class MazeRouter:
    """
    迷宫布线核心引擎。

    使用方向感知多源Dijkstra算法（可选A*启发式）从一组源节点寻找到达
    一组目标节点的最短路径，同时遵守间距约束、cable_locs限制和折角代价。

    Dijkstra状态为 (node, incoming_dir)，在同层方向改变时施加折角代价；
    跨层 via 后方向重置为 _DIR_NONE，不产生折角惩罚。
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
            corner_mgr: 折角代价管理器（CornerManager），优先于 corner_costs；
                        None 时按 corner_costs 参数构建默认管理器。
            corner_costs: 向后兼容参数，{layer: float} 字典；仅在 corner_mgr=None 时生效：
                          None → 使用默认值 DEFAULT_CORNER_COST=5.0；
                          {}   → 不施加折角代价；{...} → 按层指定代价。
        """
        self.grid = grid
        self.spacing_mgr = spacing_mgr
        self.cost_multiplier = cost_multiplier
        self.congestion_map = congestion_map or {}
        self.corner_mgr = _resolve_corner_mgr(corner_mgr, corner_costs)

    def route(
        self,
        sources: Set[Node],
        targets: Set[Node],
        net_name: str,
        cable_locs: Optional[Set[Node]] = None,
        use_astar: bool = True,
    ) -> PathResult:
        """
        从源节点集合寻找到目标节点集合的最短路径（含折角代价）。

        Dijkstra 状态：(node, incoming_dir)
          - 所有源节点初始方向为 _DIR_NONE（无来向，第一步不触发折角）
          - 跨层 via 后方向重置为 _DIR_NONE
          - 同层移动方向改变时，在进入新节点前叠加 corner_costs[layer]

        参数:
            sources: 源节点集合（如已构建的部分Steiner树上所有节点）
            targets: 目标节点集合（如下一个要连接的terminal）
            net_name: 当前线网名称
            cable_locs: 该线网在M0层可用的节点集合（None表示无限制）
            use_astar: 是否使用A*启发式加速
        返回:
            PathResult 包含路径节点、边和代价
        """
        if not sources or not targets:
            return PathResult.failure()

        # 源和目标有交集则直接返回
        overlap = sources & targets
        if overlap:
            node = next(iter(overlap))
            return PathResult([node], [], 0.0, True)

        target_coords = [(t[1], t[2]) for t in targets]

        def heuristic(node: Node) -> float:
            if not use_astar:
                return 0.0
            x, y = node[1], node[2]
            return min(abs(x - tx) + abs(y - ty) for tx, ty in target_coords)

        # 优先队列: (estimated_cost, counter, actual_cost, node, dir_code)
        # dist / prev 以 (node, dir_code) 为键
        counter = 0
        pq = []
        dist: Dict[Tuple[Node, int], float] = {}
        prev: Dict[Tuple[Node, int], Tuple[Node, int]] = {}

        for s in sources:
            if not self.grid.is_valid_node(s):
                continue
            state = (s, _DIR_NONE)
            if state not in dist:
                dist[state] = 0.0
                h = heuristic(s)
                heapq.heappush(pq, (h, counter, 0.0, s, _DIR_NONE))
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
                # 间距约束（源集合内节点属于本线网，无需检查）
                if neighbor not in sources:
                    if not self.spacing_mgr.is_available(neighbor, net_name):
                        continue

                # M0 cable_locs 约束
                if cable_locs is not None and neighbor[0] == "M0":
                    if neighbor not in cable_locs and neighbor not in targets:
                        continue

                # 计算移动方向
                dir_out = _move_dir_code(current, neighbor)

                # 折角代价：同层 && 有来向 && 方向改变
                cp = 0.0
                if (dir_in != _DIR_NONE
                        and dir_out != _DIR_NONE
                        and dir_in != dir_out):
                    cp = self.corner_mgr.get_l_cost(current[0], dir_in, dir_out)

                # 边代价
                edge_cost = self.grid.get_edge_cost(current, neighbor)
                if self.cost_multiplier:
                    edge_cost *= self.cost_multiplier(neighbor, net_name)
                if neighbor in self.congestion_map:
                    edge_cost += self.congestion_map[neighbor]

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

        # 回溯路径（从 (node, dir) 状态链中提取节点序列）
        path_nodes: List[Node] = []
        path_edges: List[Edge] = []
        cur_state = found_state
        while cur_state in prev:
            node = cur_state[0]
            path_nodes.append(node)
            par_state = prev[cur_state]
            path_edges.append((par_state[0], node))
            cur_state = par_state
        path_nodes.append(cur_state[0])   # 起始源节点
        path_nodes.reverse()
        path_edges.reverse()

        return PathResult(
            nodes=path_nodes,
            edges=path_edges,
            cost=dist[found_state],
            success=True,
        )

    def find_blocked_path_info(
        self,
        sources: Set[Node],
        targets: Set[Node],
        net_name: str,
        cable_locs: Optional[Set[Node]] = None,
    ) -> Set[str]:
        """
        当布线失败时，分析哪些线网阻塞了路径。

        使用 BFS 扩展，收集所有无法通过的节点上阻塞的线网名称。
        （此方法仅用于阻塞分析，不涉及方向/折角，保持原逻辑。）

        参数:
            sources: 源节点集合
            targets: 目标节点集合
            net_name: 当前线网名称
            cable_locs: M0层可用节点集合
        返回:
            阻塞线网名称集合
        """
        blocking_nets = set()
        visited = set()
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
                    if not self.spacing_mgr.is_available(neighbor, net_name):
                        blocking_nets.update(
                            self.spacing_mgr.get_blocking_nets(neighbor, net_name)
                        )
                        continue

                queue.append(neighbor)

        return blocking_nets
