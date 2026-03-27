"""
路由网格模块

封装nx.Graph，提供布线网格的查询接口。
"""

from typing import List, Tuple, Set, Optional, Dict
import networkx as nx

from maze_router.net import Node


class RoutingGrid:
    """
    三层金属布线网格。

    封装nx.Graph，提供邻居查询、边代价获取等布线所需接口。
    节点格式为 (layer, x, y)，边可以是同层相邻连接或跨层via连接。
    """

    def __init__(self, graph: nx.Graph):
        """
        参数:
            graph: nx.Graph，节点为(layer, x, y)三元组，
                   边属性中包含'cost'（默认为1）
        """
        self.graph = graph
        # 缓存层信息
        self._layers: Optional[Set[str]] = None

    @property
    def layers(self) -> Set[str]:
        """返回所有层名称"""
        if self._layers is None:
            self._layers = {node[0] for node in self.graph.nodes}
        return self._layers

    def is_valid_node(self, node: Node) -> bool:
        """检查节点是否存在于网格中"""
        return node in self.graph

    def get_neighbors(self, node: Node) -> List[Node]:
        """获取节点的所有邻居（包括同层和via连接）"""
        if not self.is_valid_node(node):
            return []
        return list(self.graph.neighbors(node))

    def get_edge_cost(self, n1: Node, n2: Node) -> float:
        """
        获取两个节点之间边的代价。

        返回:
            边的cost属性值，若边不存在则返回float('inf')
        """
        if not self.graph.has_edge(n1, n2):
            return float('inf')
        return self.graph[n1][n2].get('cost', 1.0)

    def get_layer(self, node: Node) -> str:
        """获取节点所在层"""
        return node[0]

    def get_coords(self, node: Node) -> Tuple[int, int]:
        """获取节点的(x, y)坐标"""
        return node[1], node[2]

    def is_via_edge(self, n1: Node, n2: Node) -> bool:
        """判断两节点之间是否为via（跨层）连接"""
        return n1[0] != n2[0]

    def get_nodes_on_layer(self, layer: str) -> List[Node]:
        """获取指定层上的所有节点"""
        return [n for n in self.graph.nodes if n[0] == layer]

    def get_all_nodes(self) -> List[Node]:
        """获取所有节点"""
        return list(self.graph.nodes)

    @staticmethod
    def build_grid(
        layers: List[str],
        width: int,
        height: int,
        via_connections: Optional[List[Tuple[Node, Node, float]]] = None,
        removed_nodes: Optional[Set[Node]] = None,
        removed_edges: Optional[Set[Tuple[Node, Node]]] = None,
        default_cost: float = 1.0,
        default_via_cost: float = 2.0,
    ) -> 'RoutingGrid':
        """
        便捷方法：构建标准三层网格。

        参数:
            layers: 层名称列表，如 ["M0", "M1", "M2"]
            width: 网格宽度（x方向）
            height: 网格高度（y方向）
            via_connections: via连接列表 [(node1, node2, cost), ...]
            removed_nodes: 需要移除的节点
            removed_edges: 需要移除的边
            default_cost: 同层边默认代价
            default_via_cost: via默认代价（仅在via_connections未提供时使用）
        """
        G = nx.Graph()

        # 添加每层的节点和同层边
        for layer in layers:
            for x in range(width):
                for y in range(height):
                    node = (layer, x, y)
                    if removed_nodes and node in removed_nodes:
                        continue
                    G.add_node(node)
                    # 连接左邻居
                    left = (layer, x - 1, y)
                    if x > 0 and G.has_node(left):
                        edge = (node, left)
                        if not (removed_edges and (edge in removed_edges or (left, node) in removed_edges)):
                            G.add_edge(node, left, cost=default_cost)
                    # 连接下邻居
                    down = (layer, x, y - 1)
                    if y > 0 and G.has_node(down):
                        edge = (node, down)
                        if not (removed_edges and (edge in removed_edges or (down, node) in removed_edges)):
                            G.add_edge(node, down, cost=default_cost)

        # 添加via连接
        if via_connections:
            for n1, n2, cost in via_connections:
                if G.has_node(n1) and G.has_node(n2):
                    G.add_edge(n1, n2, cost=cost)

        return RoutingGrid(G)
