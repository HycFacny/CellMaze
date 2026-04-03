"""
硬间距约束 (SpaceConstraint)

基于 Chebyshev 距离的硬间距约束：
  同层两条不同线网之间的 Chebyshev 距离必须 ≥ min_space[layer]。
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, Set

from maze_router.constraints.base import BaseConstraint
from maze_router.data.net import Node


class SpaceConstraint(BaseConstraint):
    """
    硬间距约束。

    对于每一层，维护已标记节点的占用记录，
    查询时检查候选节点到其他线网节点的 Chebyshev 距离是否满足最小间距要求。

    Chebyshev 距离: dist(a, b) = max(|ax-bx|, |ay-by|)
    """

    def __init__(self, rules: Dict[str, int]):
        """
        参数:
            rules: {layer: min_space}，min_space ≥ 1
        """
        self.rules: Dict[str, int] = rules
        # {layer -> {(x,y) -> set of net_names}}
        self._grid: Dict[str, Dict[tuple, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        # {net_name -> set of nodes}
        self._net_nodes: Dict[str, Set[Node]] = defaultdict(set)

    # ------------------------------------------------------------------
    # BaseConstraint 接口
    # ------------------------------------------------------------------

    def is_available(self, node: Node, net_name: str) -> bool:
        """
        判断节点是否对给定线网可用。

        规则：候选节点与所有其他线网的已标记节点的 Chebyshev 距离 ≥ min_space。
        同一线网的节点不受约束。
        若节点所在层未在 rules 中配置（如虚拟层），始终返回 True。
        """
        layer, x, y = node
        if layer not in self.rules:
            return True
        min_space = self.rules.get(layer, 1)
        layer_grid = self._grid[layer]

        for dx in range(-min_space, min_space + 1):
            for dy in range(-min_space, min_space + 1):
                # Chebyshev 距离 = max(|dx|, |dy|)
                cheby = max(abs(dx), abs(dy))
                if cheby > min_space:
                    continue
                key = (x + dx, y + dy)
                if key in layer_grid:
                    for occ_net in layer_grid[key]:
                        if occ_net != net_name:
                            return False
        return True

    def mark_route(self, net_name: str, nodes: Set[Node]):
        """标记线网占用的节点。仅标记 rules 中配置的层（虚拟层节点不参与间距管理）。"""
        for node in nodes:
            layer, x, y = node
            if layer not in self.rules:
                continue
            self._grid[layer][(x, y)].add(net_name)
            self._net_nodes[net_name].add(node)

    def unmark_route(self, net_name: str):
        """撤销线网的所有标记。"""
        if net_name not in self._net_nodes:
            return
        for node in self._net_nodes[net_name]:
            layer, x, y = node
            key = (x, y)
            if layer in self._grid and key in self._grid[layer]:
                self._grid[layer][key].discard(net_name)
                if not self._grid[layer][key]:
                    del self._grid[layer][key]
        del self._net_nodes[net_name]

    def partial_unmark_route(self, net_name: str, nodes: Set[Node]):
        """撤销线网中部分节点的标记。"""
        for node in nodes:
            layer, x, y = node
            key = (x, y)
            if layer in self._grid and key in self._grid[layer]:
                self._grid[layer][key].discard(net_name)
                if not self._grid[layer][key]:
                    del self._grid[layer][key]
            self._net_nodes[net_name].discard(node)

    def partial_mark_route(self, net_name: str, nodes: Set[Node]):
        """标记线网中部分节点。仅标记 rules 中配置的层。"""
        for node in nodes:
            layer, x, y = node
            if layer not in self.rules:
                continue
            self._grid[layer][(x, y)].add(net_name)
            self._net_nodes[net_name].add(node)

    def get_blocking_nets(self, node: Node, net_name: str) -> Set[str]:
        """返回在该节点附近阻塞给定线网的线网名称集合。虚拟层节点不产生阻塞。"""
        layer, x, y = node
        if layer not in self.rules:
            return set()
        min_space = self.rules.get(layer, 1)
        layer_grid = self._grid[layer]
        blockers: Set[str] = set()

        for dx in range(-min_space, min_space + 1):
            for dy in range(-min_space, min_space + 1):
                cheby = max(abs(dx), abs(dy))
                if cheby > min_space:
                    continue
                key = (x + dx, y + dy)
                if key in layer_grid:
                    for occ_net in layer_grid[key]:
                        if occ_net != net_name:
                            blockers.add(occ_net)
        return blockers

    def get_net_nodes(self, net_name: str) -> Set[Node]:
        return set(self._net_nodes.get(net_name, set()))

    # ------------------------------------------------------------------
    # 调试辅助
    # ------------------------------------------------------------------

    def get_occupied_layers(self) -> Dict[str, int]:
        return {layer: len(coords) for layer, coords in self._grid.items()}
