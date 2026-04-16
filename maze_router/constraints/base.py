"""
约束基类 (ABC)
"""

from abc import ABC, abstractmethod
from typing import List, Set, Tuple
from maze_router.data.net import Node


class BaseConstraint(ABC):
    """
    布线约束基类。

    所有约束实现此接口，由 ConstraintManager 统一调用。
    """

    @abstractmethod
    def is_available(self, node: Node, net_name: str) -> bool:
        """判断节点对给定线网是否可用（不违反约束）。"""
        ...

    @abstractmethod
    def mark_route(self, net_name: str, nodes: Set[Node]):
        """标记线网已占用的节点。"""
        ...

    @abstractmethod
    def unmark_route(self, net_name: str):
        """撤销线网的所有标记。"""
        ...

    def partial_unmark_route(self, net_name: str, nodes: Set[Node]):
        """撤销线网的部分节点标记（区域拆线用）。默认不操作，子类可覆盖。"""
        pass

    def partial_mark_route(self, net_name: str, nodes: Set[Node]):
        """标记线网的部分节点（区域重连用）。默认不操作，子类可覆盖。"""
        pass

    def get_blocking_nets(self, node: Node, net_name: str) -> Set[str]:
        """返回在该节点阻塞给定线网的线网名称集合。"""
        return set()

    def get_net_nodes(self, net_name: str) -> Set[Node]:
        """返回线网已标记的节点集合。"""
        return set()

    def post_process_result(self, net_name: str, result, grid) -> None:
        """
        路由成功后的后处理钩子（如最小面积扩展）。

        在 Router.route_net() 完成后调用，允许约束对布线结果进行修改。
        默认不操作，子类可覆盖。

        参数:
            net_name: 线网名称
            result:   RoutingResult（可修改 routed_nodes/routed_edges）
            grid:     GridGraph（只读，用于查找邻居节点）
        """
        pass

    def is_edge_available(self, node_a: Node, node_b: Node, net_name: str) -> bool:
        """
        判断边 (node_a, node_b) 对给定线网是否可用。

        默认允许所有边，子类可覆盖以实现边级禁止约束（must_forbid_edges）。

        参数:
            node_a, node_b: 边的两个端点
            net_name:       当前线网名称
        返回:
            True = 边可用；False = 边被禁止
        """
        return True

    def get_must_keep_edges(self, net_name: str) -> List[Tuple[Node, Node]]:
        """
        返回该线网必须走的边列表（must_keep_edges 约束）。

        路由算法在构建 Steiner 树时会主动经过这些边。
        默认返回空列表，PathConstraint 覆盖此方法。

        参数:
            net_name: 线网名称
        返回:
            必须走的边列表 [(node_a, node_b), ...]
        """
        return []

    def required_terminals(self, net_name: str, grid) -> List[Node]:
        """
        返回路由*前*需注入为强制终端的节点列表（在线约束注入）。

        Router 在派发路由前调用此方法，将返回节点合并到线网终端列表，
        使路由算法自然建立物理连接以满足约束（如 active 覆盖）。
        默认返回空列表，子类可覆盖。

        参数:
            net_name: 线网名称
            grid:     GridGraph（只读）
        返回:
            需要作为强制终端的节点列表
        """
        return []
