"""
约束管理器 (ConstraintManager)

聚合所有 BaseConstraint 实例，统一对外提供约束查询接口。
支持动态插入新约束，便于扩展。
"""

from __future__ import annotations
from typing import List, Set

from maze_router.constraints.base import BaseConstraint
from maze_router.data.net import Node


class ConstraintManager:
    """
    约束管理器。

    统一聚合多个 BaseConstraint，实现约束的统一查询和标记管理：
    - is_available: 所有约束都满足才返回 True
    - mark/unmark: 广播到所有约束
    - get_blocking_nets: 合并所有约束返回的阻塞线网集合
    """

    def __init__(self, constraints: List[BaseConstraint] = None):
        self._constraints: List[BaseConstraint] = constraints or []

    def add_constraint(self, constraint: BaseConstraint):
        """动态插入新约束。"""
        self._constraints.append(constraint)

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    def is_available(self, node: Node, net_name: str) -> bool:
        """所有约束都满足才返回 True。"""
        return all(c.is_available(node, net_name) for c in self._constraints)

    def get_blocking_nets(self, node: Node, net_name: str) -> Set[str]:
        """合并所有约束返回的阻塞线网名称集合。"""
        result: Set[str] = set()
        for c in self._constraints:
            result.update(c.get_blocking_nets(node, net_name))
        return result

    def get_net_nodes(self, net_name: str) -> Set[Node]:
        """从所有约束中收集线网已标记的节点。"""
        result: Set[Node] = set()
        for c in self._constraints:
            result.update(c.get_net_nodes(net_name))
        return result

    # ------------------------------------------------------------------
    # 标记接口
    # ------------------------------------------------------------------

    def mark_route(self, net_name: str, nodes: Set[Node]):
        """广播标记到所有约束。"""
        for c in self._constraints:
            c.mark_route(net_name, nodes)

    def unmark_route(self, net_name: str):
        """广播撤标到所有约束。"""
        for c in self._constraints:
            c.unmark_route(net_name)

    def partial_unmark_route(self, net_name: str, nodes: Set[Node]):
        """部分撤标到所有约束（区域拆线用）。"""
        for c in self._constraints:
            c.partial_unmark_route(net_name, nodes)

    def partial_mark_route(self, net_name: str, nodes: Set[Node]):
        """部分标记到所有约束（区域重连用）。"""
        for c in self._constraints:
            c.partial_mark_route(net_name, nodes)

    def post_process_results(self, net_name: str, result, grid) -> None:
        """广播后处理钩子到所有约束（如 MinAreaConstraint 的节点扩展）。"""
        for c in self._constraints:
            c.post_process_result(net_name, result, grid)

    def collect_required_terminals(self, net_name: str, grid) -> list:
        """
        收集所有约束为线网提供的强制终端节点（在线注入机制）。

        Router 在路由派发前调用此方法，将返回节点合并到线网终端列表，
        使路由算法自然建立物理连接以满足约束（如 active 覆盖）。

        返回去重后的节点列表，保留首次出现顺序。
        """
        seen: set = set()
        result: list = []
        for c in self._constraints:
            for node in c.required_terminals(net_name, grid):
                if node not in seen:
                    seen.add(node)
                    result.append(node)
        return result

    def __repr__(self) -> str:
        return f"ConstraintManager({len(self._constraints)} constraints)"
