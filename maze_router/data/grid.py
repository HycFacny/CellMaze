"""
GridGraph: nx.Graph 封装，描述三层金属层布线网格。

节点: (layer, x, y)  layer ∈ {"M0","M1","M2"}
边:   带 'cost' 属性的无向边
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx

from maze_router.data.net import Node


class GridGraph:
    """
    布线网格封装。

    内部以 nx.Graph 存储节点和边（edge 属性 'cost' 默认=1）。
    提供路由器所需的查询接口：邻居、边代价、节点有效性。

    虚拟节点（virtual nodes）：
        某些工艺中，active 区域的起点范围可变，外部通过添加虚拟层节点
        并连边到实际 M0 节点来描述这种可变性。虚拟节点不参与间距约束，
        也不计入 active 占用计数。
        使用 register_virtual_layer() 注册虚拟层，用 is_virtual_node() 查询。
    """

    def __init__(self, graph: Optional[nx.Graph] = None):
        self.graph: nx.Graph = graph if graph is not None else nx.Graph()
        # 虚拟层集合：这些层的节点不参与间距约束和 active 占用计数
        self.virtual_node_layers: Set[str] = set()

    # ------------------------------------------------------------------
    # 基础查询
    # ------------------------------------------------------------------

    def is_valid_node(self, node: Node) -> bool:
        return self.graph.has_node(node)

    def get_all_nodes(self) -> List[Node]:
        return list(self.graph.nodes)

    def get_nodes_on_layer(self, layer: str) -> List[Node]:
        return [n for n in self.graph.nodes if n[0] == layer]

    def get_neighbors(self, node: Node) -> List[Node]:
        if not self.graph.has_node(node):
            return []
        return list(self.graph.neighbors(node))

    def get_edge_cost(self, src: Node, dst: Node) -> float:
        if not self.graph.has_edge(src, dst):
            return float('inf')
        return self.graph[src][dst].get('cost', 1.0)

    def add_node(self, node: Node):
        self.graph.add_node(node)

    def add_edge(self, src: Node, dst: Node, cost: float = 1.0):
        self.graph.add_edge(src, dst, cost=cost)

    # ------------------------------------------------------------------
    # 虚拟节点管理
    # ------------------------------------------------------------------

    def register_virtual_layer(self, layer: str):
        """
        将某层注册为虚拟层（非物理层）。

        虚拟层节点用于描述 active 起点范围可变的工艺：
        外部建立虚拟节点并连边到实际 M0 节点，路由器透明穿越，
        但虚拟节点不参与间距约束计算，也不计入 active 占用计数。
        """
        self.virtual_node_layers.add(layer)

    def is_virtual_node(self, node: Node) -> bool:
        """若节点属于已注册的虚拟层，返回 True。"""
        return node[0] in self.virtual_node_layers

    def remove_node(self, node: Node):
        self.graph.remove_node(node)

    def remove_edge(self, src: Node, dst: Node):
        if self.graph.has_edge(src, dst):
            self.graph.remove_edge(src, dst)

    def remove_vertical_edges_at(self, nodes):
        """
        移除给定节点上/下方向的同层竖向边。

        用于在 active 行的 SD 列节点上禁止竖向 M0 走线（仅允许 via 到 M1）。
        对每个节点 (layer, x, y)，移除与 (layer, x, y-1) 和 (layer, x, y+1) 的边。

        参数:
            nodes: Iterable[Node]
        """
        for node in nodes:
            layer, x, y = node
            for dy in (-1, 1):
                neighbor = (layer, x, y + dy)
                self.remove_edge(node, neighbor)

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @staticmethod
    def build_grid(
        layers: List[str],
        width: int,
        height: int,
        edge_cost: float = 1.0,
        via_cost: float = 1.0,
        layer_constraints: Optional[Dict[str, Set[Tuple[int, int]]]] = None,
        via_connections: Optional[List[Tuple[Node, Node, float]]] = None,
        removed_nodes: Optional[Set[Node]] = None,
        horizontal_only_layers: Optional[Set[str]] = None,
        vertical_only_cols: Optional[Dict[str, Set[int]]] = None,
        allowed_nodes: Optional[Dict[str, Set[Tuple[int, int]]]] = None,
    ) -> 'GridGraph':
        """
        构建标准矩形网格。

        参数:
            layers:               层名列表，如 ["M0","M1","M2"]
            width:                网格列数 (x 方向，0..width-1)
            height:               网格行数 (y 方向，0..height-1)
            edge_cost:            同层水平/垂直边代价
            via_cost:             跨层 via 边代价（若 via_connections 为 None 时自动生成）
            layer_constraints:    {layer: set of (x,y)} 各层允许存在的节点（None=全部允许）
            via_connections:      [(src_node, dst_node, cost), ...] 显式 via 列表
            removed_nodes:        全局移除节点集合
            horizontal_only_layers: 只允许横向走线的层集合（如 M2）
            vertical_only_cols:   {layer: set of x_col} 只允许纵向走线的列（如 SD 列）
            allowed_nodes:        {layer: set of (x,y)} 与 layer_constraints 等价，取并集
        返回:
            GridGraph
        """
        g = GridGraph()
        removed = removed_nodes or set()

        # 收集每层允许的节点坐标
        layer_allowed: Dict[str, Set[Tuple[int, int]]] = {}
        for layer in layers:
            allowed: Set[Tuple[int, int]] = set()
            for x in range(width):
                for y in range(height):
                    allowed.add((x, y))
            if layer_constraints and layer in layer_constraints:
                allowed &= layer_constraints[layer]
            if allowed_nodes and layer in allowed_nodes:
                allowed &= allowed_nodes[layer]
            layer_allowed[layer] = allowed

        # 添加节点
        for layer in layers:
            for x, y in layer_allowed[layer]:
                node = (layer, x, y)
                if node not in removed:
                    g.add_node(node)

        h_only = horizontal_only_layers or set()
        v_only_cols = vertical_only_cols or {}

        # 添加同层边
        for layer in layers:
            allowed = layer_allowed[layer]
            only_h = layer in h_only
            v_cols = v_only_cols.get(layer, None)

            for x, y in allowed:
                src = (layer, x, y)
                if src in removed or not g.is_valid_node(src):
                    continue

                # 水平边 (+x)
                if not only_h or True:   # horizontal allowed unless restricted
                    rx, ry = x + 1, y
                    if (rx, ry) in allowed:
                        dst = (layer, rx, ry)
                        if dst not in removed and g.is_valid_node(dst):
                            # 若该列为竖向专用列，禁止横向边
                            if v_cols and (x in v_cols or rx in v_cols):
                                pass  # 不加横向边
                            elif not only_h:
                                g.add_edge(src, dst, cost=edge_cost)
                            else:
                                g.add_edge(src, dst, cost=edge_cost)

                # 垂直边 (+y)
                ux, uy = x, y + 1
                if (ux, uy) in allowed:
                    dst = (layer, ux, uy)
                    if dst not in removed and g.is_valid_node(dst):
                        if only_h:
                            pass  # 水平专用层，不加纵向边
                        else:
                            g.add_edge(src, dst, cost=edge_cost)

        # 添加 via 边
        if via_connections is not None:
            for src, dst, cost in via_connections:
                if g.is_valid_node(src) and g.is_valid_node(dst):
                    g.add_edge(src, dst, cost=cost)
        else:
            # 自动生成相邻层 via（y 坐标相同，x 相同）
            layer_order = {l: i for i, l in enumerate(layers)}
            for i in range(len(layers) - 1):
                lower, upper = layers[i], layers[i + 1]
                lower_allowed = layer_allowed.get(lower, set())
                upper_allowed = layer_allowed.get(upper, set())
                common = lower_allowed & upper_allowed
                for x, y in common:
                    src = (lower, x, y)
                    dst = (upper, x, y)
                    if (src not in removed and dst not in removed
                            and g.is_valid_node(src) and g.is_valid_node(dst)):
                        g.add_edge(src, dst, cost=via_cost)

        return g

    def __repr__(self) -> str:
        return (
            f"GridGraph(nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()})"
        )
