"""
核心迷宫路由器模块

基于多源Dijkstra/A*算法的单路径布线引擎。
支持间距约束、cable_locs约束和可配置的代价函数。
"""

import heapq
from typing import List, Set, Tuple, Optional, Callable, Dict

from maze_router.net import Node, Edge
from maze_router.grid import RoutingGrid
from maze_router.spacing import SpacingManager


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

    使用多源Dijkstra算法（可选A*启发式）从一组源节点寻找到达
    一组目标节点的最短路径，同时遵守间距约束和M0层cable_locs限制。
    """

    def __init__(
        self,
        grid: RoutingGrid,
        spacing_mgr: SpacingManager,
        cost_multiplier: Optional[Callable[[Node, str], float]] = None,
        congestion_map: Optional[Dict[Node, float]] = None,
    ):
        """
        参数:
            grid: 布线网格
            spacing_mgr: 间距约束管理器
            cost_multiplier: 可选的代价乘数函数 (node, net_name) -> float
            congestion_map: 可选的拥塞代价映射 node -> extra_cost
        """
        self.grid = grid
        self.spacing_mgr = spacing_mgr
        self.cost_multiplier = cost_multiplier
        self.congestion_map = congestion_map or {}

    def route(
        self,
        sources: Set[Node],
        targets: Set[Node],
        net_name: str,
        cable_locs: Optional[Set[Node]] = None,
        use_astar: bool = True,
    ) -> PathResult:
        """
        从源节点集合寻找到目标节点集合的最短路径。

        使用多源Dijkstra：所有源节点初始代价为0。
        当扩展到任意目标节点时终止并回溯路径。

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

        # 预计算目标集合用于启发式
        target_coords = [(t[1], t[2]) for t in targets]

        def heuristic(node: Node) -> float:
            if not use_astar:
                return 0.0
            x, y = node[1], node[2]
            # 到最近目标的曼哈顿距离
            return min(abs(x - tx) + abs(y - ty) for tx, ty in target_coords)

        # 优先队列: (estimated_cost, counter, actual_cost, node)
        counter = 0
        pq = []
        dist = {}
        prev = {}

        for s in sources:
            if not self.grid.is_valid_node(s):
                continue
            dist[s] = 0.0
            h = heuristic(s)
            heapq.heappush(pq, (h, counter, 0.0, s))
            counter += 1

        found_target = None

        while pq:
            est_cost, _, actual_cost, current = heapq.heappop(pq)

            # 已经有更优路径到达此节点
            if actual_cost > dist.get(current, float('inf')):
                continue

            # 到达目标
            if current in targets:
                found_target = current
                break

            # 扩展邻居
            for neighbor in self.grid.get_neighbors(current):
                # 检查间距约束（邻居不在源集合内才需要检查，源节点已属于本线网）
                if neighbor not in sources:
                    if not self.spacing_mgr.is_available(neighbor, net_name):
                        continue

                # 检查M0层cable_locs约束
                if cable_locs is not None and neighbor[0] == "M0":
                    if neighbor not in cable_locs and neighbor not in targets:
                        continue

                # 计算边代价
                edge_cost = self.grid.get_edge_cost(current, neighbor)

                # 加入代价乘数
                if self.cost_multiplier:
                    edge_cost *= self.cost_multiplier(neighbor, net_name)

                # 加入拥塞代价
                if neighbor in self.congestion_map:
                    edge_cost += self.congestion_map[neighbor]

                new_cost = actual_cost + edge_cost

                if new_cost < dist.get(neighbor, float('inf')):
                    dist[neighbor] = new_cost
                    prev[neighbor] = current
                    h = heuristic(neighbor)
                    heapq.heappush(pq, (new_cost + h, counter, new_cost, neighbor))
                    counter += 1

        if found_target is None:
            return PathResult.failure()

        # 回溯路径
        path_nodes = []
        path_edges = []
        current = found_target
        while current in prev:
            path_nodes.append(current)
            parent = prev[current]
            path_edges.append((parent, current))
            current = parent
        path_nodes.append(current)  # 添加起始源节点
        path_nodes.reverse()
        path_edges.reverse()

        return PathResult(
            nodes=path_nodes,
            edges=path_edges,
            cost=dist[found_target],
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

        使用BFS扩展，收集所有无法通过的节点上阻塞的线网名称。

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

                # 检查cable_locs
                if cable_locs is not None and neighbor[0] == "M0":
                    if neighbor not in cable_locs and neighbor not in targets:
                        continue

                if neighbor not in sources:
                    if not self.spacing_mgr.is_available(neighbor, net_name):
                        # 记录阻塞的线网
                        blocking_nets.update(
                            self.spacing_mgr.get_blocking_nets(neighbor, net_name)
                        )
                        continue

                queue.append(neighbor)

        return blocking_nets
