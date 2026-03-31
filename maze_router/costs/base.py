"""
代价基类 (ABC)
"""

from abc import ABC, abstractmethod
from maze_router.data.net import Node


class BaseCost(ABC):
    """
    布线代价基类。

    所有软代价（SpaceCost、CornerCost 等）继承此类，
    由 CostManager 统一聚合。
    """

    @abstractmethod
    def get_node_penalty(self, node: Node, net_name: str, context) -> float:
        """
        计算节点的额外代价（软约束惩罚）。

        参数:
            node:     目标节点
            net_name: 当前线网名
            context:  RoutingContext（含约束管理器、拥塞图等）
        返回:
            非负浮点数代价
        """
        ...
