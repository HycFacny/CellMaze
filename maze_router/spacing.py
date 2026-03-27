"""
间距约束管理模块

管理不同线网之间的最小间距约束，使用Chebyshev距离度量。
维护每层的占用地图，支持标记、解除标记和可用性查询。
"""

from typing import Dict, Set, Optional, List
from collections import defaultdict

from maze_router.net import Node


class SpacingManager:
    """
    间距约束管理器。

    维护每层上各坐标被哪个线网占用的信息，
    并根据space参数在标记占用时扩展Chebyshev距离范围内的禁区。
    """

    def __init__(self, space: Dict[str, int]):
        """
        参数:
            space: 每层的最小间距，如 {"M0": 1, "M1": 1, "M2": 1}
        """
        self.space = space

        # 每层每个坐标被哪个线网实际占用（非禁区扩展）
        # layer -> (x, y) -> net_name
        self._occupied: Dict[str, Dict[tuple, str]] = defaultdict(dict)

        # 每层每个坐标的禁区属于哪些线网（包含间距扩展）
        # layer -> (x, y) -> set of net_names
        self._blocked: Dict[str, Dict[tuple, Set[str]]] = defaultdict(lambda: defaultdict(set))

        # 每个线网占用的节点，用于拆线时快速清除
        # net_name -> set of nodes
        self._net_nodes: Dict[str, Set[Node]] = defaultdict(set)

    def mark_route(self, net_name: str, nodes: Set[Node]):
        """
        标记一个线网的布线路径。

        将路径上的节点标记为已占用，并在Chebyshev距离范围内
        扩展禁区以限制其他线网。

        参数:
            net_name: 线网名称
            nodes: 布线经过的节点集合
        """
        self._net_nodes[net_name] = set(nodes)

        for node in nodes:
            layer, x, y = node
            # 标记实际占用
            self._occupied[layer][(x, y)] = net_name

            # 扩展禁区（Chebyshev距离）
            s = self.space.get(layer, 0)
            for dx in range(-s, s + 1):
                for dy in range(-s, s + 1):
                    self._blocked[layer][(x + dx, y + dy)].add(net_name)

    def unmark_route(self, net_name: str):
        """
        清除一个线网的所有标记（用于拆线重布）。

        参数:
            net_name: 需要清除的线网名称
        """
        nodes = self._net_nodes.get(net_name, set())

        for node in nodes:
            layer, x, y = node
            # 清除实际占用
            if (x, y) in self._occupied[layer] and self._occupied[layer][(x, y)] == net_name:
                del self._occupied[layer][(x, y)]

            # 清除禁区
            s = self.space.get(layer, 0)
            for dx in range(-s, s + 1):
                for dy in range(-s, s + 1):
                    coord = (x + dx, y + dy)
                    if coord in self._blocked[layer]:
                        self._blocked[layer][coord].discard(net_name)
                        if not self._blocked[layer][coord]:
                            del self._blocked[layer][coord]

        if net_name in self._net_nodes:
            del self._net_nodes[net_name]

    def is_available(self, node: Node, net_name: str) -> bool:
        """
        检查节点对于指定线网是否可用。

        一个节点可用的条件是：
        1. 没有被其他线网实际占用
        2. 不在其他线网的禁区范围内（本线网自身的禁区不影响）

        参数:
            node: 要检查的节点
            net_name: 当前正在布线的线网名称
        返回:
            是否可用
        """
        layer, x, y = node
        coord = (x, y)

        # 检查是否被其他线网实际占用
        if coord in self._occupied[layer]:
            if self._occupied[layer][coord] != net_name:
                return False

        # 检查禁区：只要有其他线网的禁区覆盖此处，就不可用
        if coord in self._blocked[layer]:
            blocking_nets = self._blocked[layer][coord]
            # 排除自身线网
            other_nets = blocking_nets - {net_name}
            if other_nets:
                return False

        return True

    def get_blocking_nets(self, node: Node, net_name: str) -> Set[str]:
        """
        获取阻塞指定节点的其他线网名称。

        参数:
            node: 被阻塞的节点
            net_name: 当前线网名称
        返回:
            阻塞该节点的其他线网名称集合
        """
        layer, x, y = node
        coord = (x, y)
        blocking = set()

        if coord in self._occupied[layer]:
            occ = self._occupied[layer][coord]
            if occ != net_name:
                blocking.add(occ)

        if coord in self._blocked[layer]:
            blocking.update(self._blocked[layer][coord] - {net_name})

        return blocking

    def get_all_blocking_nets_for_path(
        self, nodes: List[Node], net_name: str
    ) -> Set[str]:
        """
        获取阻塞一条路径上任意节点的所有其他线网。

        参数:
            nodes: 路径节点列表
            net_name: 当前线网名称
        返回:
            阻塞线网名称集合
        """
        blocking = set()
        for node in nodes:
            blocking.update(self.get_blocking_nets(node, net_name))
        return blocking

    def is_occupied(self, node: Node) -> bool:
        """检查节点是否被任何线网实际占用"""
        layer, x, y = node
        return (x, y) in self._occupied[layer]

    def get_occupying_net(self, node: Node) -> Optional[str]:
        """获取占用指定节点的线网名称"""
        layer, x, y = node
        return self._occupied[layer].get((x, y))

    def get_routed_nets(self) -> Set[str]:
        """获取当前已布线的所有线网名称"""
        return set(self._net_nodes.keys())

    def get_net_nodes(self, net_name: str) -> Set[Node]:
        """获取指定线网占用的所有节点"""
        return self._net_nodes.get(net_name, set())

    def partial_unmark_route(self, net_name: str, nodes_to_remove: Set[Node]):
        """
        部分清除一个线网的标记（用于区域拆线）。

        只清除指定节点子集的占用和禁区标记，
        线网的其余部分保持不变。

        参数:
            net_name: 线网名称
            nodes_to_remove: 需要清除的节点子集
        """
        current_nodes = self._net_nodes.get(net_name, set())
        actual_remove = nodes_to_remove & current_nodes

        for node in actual_remove:
            layer, x, y = node
            # 清除实际占用
            if (x, y) in self._occupied[layer] and self._occupied[layer][(x, y)] == net_name:
                del self._occupied[layer][(x, y)]

            # 清除禁区
            s = self.space.get(layer, 0)
            for dx in range(-s, s + 1):
                for dy in range(-s, s + 1):
                    coord = (x + dx, y + dy)
                    if coord in self._blocked[layer]:
                        self._blocked[layer][coord].discard(net_name)
                        if not self._blocked[layer][coord]:
                            del self._blocked[layer][coord]

        # 更新线网节点集合
        self._net_nodes[net_name] = current_nodes - actual_remove
        if not self._net_nodes[net_name]:
            del self._net_nodes[net_name]

    def partial_mark_route(self, net_name: str, nodes_to_add: Set[Node]):
        """
        为线网补充标记新节点（用于区域拆线后的重连接）。

        在现有标记基础上增加新节点的占用和禁区标记。

        参数:
            net_name: 线网名称
            nodes_to_add: 需要添加的节点集合
        """
        current_nodes = self._net_nodes.get(net_name, set())
        new_nodes = nodes_to_add - current_nodes

        for node in new_nodes:
            layer, x, y = node
            self._occupied[layer][(x, y)] = net_name
            s = self.space.get(layer, 0)
            for dx in range(-s, s + 1):
                for dy in range(-s, s + 1):
                    self._blocked[layer][(x + dx, y + dy)].add(net_name)

        self._net_nodes[net_name] = current_nodes | new_nodes
