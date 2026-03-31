"""
约束基类 (ABC)
"""

from abc import ABC, abstractmethod
from typing import Set
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
