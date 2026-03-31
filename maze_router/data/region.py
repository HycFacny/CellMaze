"""
Region: 区域拆线用的空间范围描述。

Region 表示一个矩形区域（可跨层），用于区域拆线时判断节点是否在冲突区域内。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Set

from maze_router.data.net import Node


@dataclass
class Region:
    """
    矩形布线区域。

    Attributes:
        x_min, x_max: 横坐标范围（含端点）
        y_min, y_max: 纵坐标范围（含端点）
        layers:       受影响的层集合（None=所有层）
    """
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    layers: Optional[Set[str]] = None

    def contains(self, node: Node) -> bool:
        layer, x, y = node
        if self.layers is not None and layer not in self.layers:
            return False
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def expand(self, margin: int) -> 'Region':
        return Region(
            x_min=self.x_min - margin,
            x_max=self.x_max + margin,
            y_min=self.y_min - margin,
            y_max=self.y_max + margin,
            layers=self.layers,
        )

    def __repr__(self) -> str:
        return (
            f"Region(x=[{self.x_min},{self.x_max}], "
            f"y=[{self.y_min},{self.y_max}], layers={self.layers})"
        )


def find_nodes_in_region(nodes: Set[Node], region: Region) -> Set[Node]:
    """返回 nodes 中在 region 内的节点子集。"""
    return {n for n in nodes if region.contains(n)}


def find_connected_components(
    nodes: Set[Node],
    edges: list,
) -> list:
    """
    在给定节点和边的子图中查找连通分量。

    参数:
        nodes: 节点集合
        edges: 边列表 [(src, dst), ...]
    返回:
        连通分量列表，每个分量是一个 Set[Node]
    """
    adj: dict = {n: set() for n in nodes}
    for u, v in edges:
        if u in adj and v in adj:
            adj[u].add(v)
            adj[v].add(u)

    visited: Set[Node] = set()
    components = []

    for start in nodes:
        if start in visited:
            continue
        component: Set[Node] = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            component.add(cur)
            for nb in adj[cur]:
                if nb not in visited:
                    stack.append(nb)
        components.append(component)

    return components


def build_conflict_region(
    failed_net,
    blocking_net_names: Set[str],
    constraint_mgr,
    grid,
    margin: int = 1,
) -> Region:
    """
    根据失败线网的端口位置构建冲突区域。

    参数:
        failed_net:          布线失败的线网
        blocking_net_names:  阻塞线网名称集合
        constraint_mgr:      约束管理器（用于获取线网节点）
        grid:                布线网格
        margin:              区域扩展余量
    返回:
        Region 冲突区域
    """
    xs, ys = [], []
    for t in failed_net.terminals:
        xs.append(t[1])
        ys.append(t[2])

    for net_name in blocking_net_names:
        for node in constraint_mgr.get_net_nodes(net_name):
            xs.append(node[1])
            ys.append(node[2])

    if not xs:
        return Region(0, 0, 0, 0)

    return Region(
        x_min=min(xs) - margin,
        x_max=max(xs) + margin,
        y_min=min(ys) - margin,
        y_max=max(ys) + margin,
    )


def analyze_regional_ripup(
    failed_net,
    blocking_net_names: Set[str],
    constraint_mgr,
    results: dict,
    region: Region,
) -> dict:
    """
    分析每个阻塞线网在区域内的节点比例。

    返回: {net_name: ratio} ratio ∈ [0, 1]
    """
    ratios = {}
    for net_name in blocking_net_names:
        if net_name not in results or not results[net_name].success:
            ratios[net_name] = 0.0
            continue
        routed = results[net_name].routed_nodes
        in_region = find_nodes_in_region(routed, region)
        ratios[net_name] = len(in_region) / len(routed) if routed else 0.0
    return ratios
