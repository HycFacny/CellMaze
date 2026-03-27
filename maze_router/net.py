"""
Net数据结构模块

定义线网(Net)、布线结果(RoutingResult)和布线方案(RoutingSolution)。
"""

from dataclasses import dataclass, field
from typing import List, Set, Tuple, Dict, Optional


# 节点类型：(层名, x坐标, y坐标)
Node = Tuple[str, int, int]

# 边类型：(节点1, 节点2)
Edge = Tuple[Node, Node]


@dataclass
class Net:
    """
    线网：需要连接的一组端口。

    属性:
        name: 线网名称（唯一标识）
        terminals: 该线网的所有端口节点
        cable_locs: 在M0层上该线网可以走线的节点集合
    """
    name: str
    terminals: List[Node]
    cable_locs: Optional[Set[Node]] = None

    def __post_init__(self):
        if self.cable_locs is not None and not isinstance(self.cable_locs, set):
            self.cable_locs = set(self.cable_locs)

    @property
    def terminal_count(self) -> int:
        return len(self.terminals)

    def bounding_box(self) -> Tuple[int, int, int, int]:
        """返回所有端口的包围盒 (min_x, min_y, max_x, max_y)"""
        xs = [t[1] for t in self.terminals]
        ys = [t[2] for t in self.terminals]
        return min(xs), min(ys), max(xs), max(ys)

    def bounding_box_area(self) -> int:
        """返回包围盒面积"""
        min_x, min_y, max_x, max_y = self.bounding_box()
        return (max_x - min_x + 1) * (max_y - min_y + 1)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Net):
            return self.name == other.name
        return NotImplemented


@dataclass
class RoutingResult:
    """
    单个线网的布线结果。

    属性:
        net: 对应的线网
        routed_nodes: 布线经过的所有节点
        routed_edges: 布线使用的所有边
        total_cost: 总布线代价
        success: 是否成功完成布线
    """
    net: Net
    routed_nodes: Set[Node] = field(default_factory=set)
    routed_edges: List[Edge] = field(default_factory=list)
    total_cost: float = 0.0
    success: bool = False

    def merge(self, path_nodes: List[Node], path_edges: List[Edge], cost: float):
        """将一条新路径合并到当前布线结果中"""
        self.routed_nodes.update(path_nodes)
        self.routed_edges.extend(path_edges)
        self.total_cost += cost


@dataclass
class RoutingSolution:
    """
    所有线网的布线方案。

    属性:
        results: 每个线网的布线结果
        total_cost: 所有线网的总布线代价
        all_routed: 是否所有线网都成功布线
    """
    results: Dict[str, RoutingResult] = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        return sum(r.total_cost for r in self.results.values())

    @property
    def all_routed(self) -> bool:
        return all(r.success for r in self.results.values())

    @property
    def routed_count(self) -> int:
        return sum(1 for r in self.results.values() if r.success)

    @property
    def failed_nets(self) -> List[str]:
        return [name for name, r in self.results.items() if not r.success]

    def add_result(self, result: RoutingResult):
        self.results[result.net.name] = result

    def get_result(self, net_name: str) -> Optional[RoutingResult]:
        return self.results.get(net_name)
