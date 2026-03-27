"""
区域拆线重布模块

提供冲突区域的定义、分析和区域拆线后的线网重连接能力。
区域拆线只拆除阻塞线网在冲突区域内的布线片段（而非整条线网），
拆线后被截断的线网通过MazeRouter重新连接断开的子树。
"""

import logging
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque

from maze_router.net import Net, Node, Edge, RoutingResult
from maze_router.grid import RoutingGrid
from maze_router.spacing import SpacingManager

logger = logging.getLogger(__name__)


class ConflictRegion:
    """
    冲突区域：描述布线冲突发生的空间范围。

    通过失败线网的端口位置和阻塞节点来定义一个包围盒区域，
    可选地扩展一定的margin以确保拆线后有足够的空间重布。

    属性:
        min_x, min_y, max_x, max_y: 区域包围盒
        layers: 涉及的层集合
    """

    def __init__(
        self,
        min_x: int, min_y: int,
        max_x: int, max_y: int,
        layers: Set[str],
        margin: int = 1,
    ):
        self.min_x = min_x - margin
        self.min_y = min_y - margin
        self.max_x = max_x + margin
        self.max_y = max_y + margin
        self.layers = layers

    def contains(self, node: Node) -> bool:
        """检查节点是否在冲突区域内"""
        layer, x, y = node
        if layer not in self.layers:
            return False
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y

    def area(self) -> int:
        """区域面积"""
        return (self.max_x - self.min_x + 1) * (self.max_y - self.min_y + 1)

    def __repr__(self) -> str:
        return (
            f"ConflictRegion(x=[{self.min_x},{self.max_x}], "
            f"y=[{self.min_y},{self.max_y}], layers={self.layers})"
        )


def build_conflict_region(
    failed_net: Net,
    blocking_nets: Set[str],
    spacing_mgr: SpacingManager,
    grid: RoutingGrid,
    margin: int = 2,
) -> ConflictRegion:
    """
    根据失败线网的端口和阻塞节点构建冲突区域。

    区域的范围由以下节点的包围盒决定：
    1. 失败线网的所有端口
    2. 阻塞线网在失败线网端口附近的节点

    参数:
        failed_net: 布线失败的线网
        blocking_nets: 阻塞线网名称集合
        spacing_mgr: 间距管理器
        grid: 布线网格
        margin: 区域额外扩展的边距
    返回:
        冲突区域对象
    """
    # 收集所有相关节点的坐标来确定区域
    xs: List[int] = []
    ys: List[int] = []
    layers: Set[str] = set()

    # 失败线网的端口
    for terminal in failed_net.terminals:
        layer, x, y = terminal
        xs.append(x)
        ys.append(y)
        layers.add(layer)

    # 阻塞线网中靠近失败线网端口的节点
    # 先计算失败线网端口的包围盒
    t_min_x = min(xs)
    t_min_y = min(ys)
    t_max_x = max(xs)
    t_max_y = max(ys)

    # 扩大搜索范围以包含阻塞节点
    search_margin = margin + 3
    for bn_name in blocking_nets:
        bn_nodes = spacing_mgr.get_net_nodes(bn_name)
        for node in bn_nodes:
            layer, x, y = node
            # 只关注在失败线网端口附近的阻塞节点
            if (t_min_x - search_margin <= x <= t_max_x + search_margin and
                    t_min_y - search_margin <= y <= t_max_y + search_margin):
                xs.append(x)
                ys.append(y)
                layers.add(layer)

    if not xs:
        # 回退：使用失败线网端口的包围盒
        for t in failed_net.terminals:
            xs.append(t[1])
            ys.append(t[2])
            layers.add(t[0])

    return ConflictRegion(
        min_x=min(xs), min_y=min(ys),
        max_x=max(xs), max_y=max(ys),
        layers=layers,
        margin=margin,
    )


def analyze_regional_ripup(
    failed_net: Net,
    blocking_nets: Set[str],
    spacing_mgr: SpacingManager,
    net_results: Dict[str, RoutingResult],
    region: ConflictRegion,
) -> Dict[str, float]:
    """
    分析每个阻塞线网是否适合区域拆线（而非整网拆线）。

    决策标准：如果阻塞线网在冲突区域内的节点数远小于总节点数，
    则区域拆线更高效，因为只需拆除并重连一小部分。

    参数:
        failed_net: 布线失败的线网
        blocking_nets: 阻塞线网名称集合
        spacing_mgr: 间距管理器
        net_results: 当前布线结果
        region: 冲突区域
    返回:
        阻塞线网名称 -> 区域内节点占比（0~1）
    """
    ratios: Dict[str, float] = {}
    for bn_name in blocking_nets:
        if bn_name not in net_results or not net_results[bn_name].success:
            continue
        bn_nodes = net_results[bn_name].routed_nodes
        total = len(bn_nodes)
        if total == 0:
            continue
        in_region = sum(1 for n in bn_nodes if region.contains(n))
        ratios[bn_name] = in_region / total
    return ratios


def find_nodes_in_region(
    net_nodes: Set[Node], region: ConflictRegion,
) -> Set[Node]:
    """获取线网在冲突区域内的所有节点"""
    return {n for n in net_nodes if region.contains(n)}


def find_connected_components(
    nodes: Set[Node],
    edges: List[Edge],
) -> List[Set[Node]]:
    """
    求节点子集的连通分量。

    在区域拆线后，线网的剩余节点可能断成多个连通分量，
    需要用MazeRouter将它们重新连接。

    参数:
        nodes: 节点集合
        edges: 边列表
    返回:
        连通分量列表，每个分量是节点集合
    """
    if not nodes:
        return []

    # 构建邻接表（只考虑两端都在nodes中的边）
    adj: Dict[Node, Set[Node]] = defaultdict(set)
    for u, v in edges:
        if u in nodes and v in nodes:
            adj[u].add(v)
            adj[v].add(u)

    visited: Set[Node] = set()
    components: List[Set[Node]] = []

    for node in nodes:
        if node in visited:
            continue
        # BFS发现一个连通分量
        component: Set[Node] = set()
        queue = deque([node])
        while queue:
            curr = queue.popleft()
            if curr in visited:
                continue
            visited.add(curr)
            component.add(curr)
            for nb in adj[curr]:
                if nb not in visited:
                    queue.append(nb)
        components.append(component)

    return components
