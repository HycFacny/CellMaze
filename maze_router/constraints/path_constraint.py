"""
路径约束 (PathConstraint)

支持两类边级约束：
  must_keep_edges  — 路由结果中必须包含这些边
  must_forbid_edges — 路由过程中禁止使用这些边

数据结构
--------
must_keep_edges : Dict[str, List[Tuple[Node, Node]]]
    {net_name: [(node_a, node_b), ...]}
    每条 must-keep 边在最终路由结果的 routed_edges 中必须出现（无方向）。

must_forbid_edges : Dict[str, List[Tuple[Node, Node]]]
    {net_name: [(node_a, node_b), ...]}
    路由过程中禁止使用这些边（无方向，(a,b) 与 (b,a) 等价）。

实现机制
--------
must_forbid_edges:
  is_edge_available(a, b, net_name) — O(1) 集合查询，实时拦截禁止边。

must_keep_edges:
  required_terminals(net_name, grid) — 注入边的两个端点为强制终端，
    使路由器自然建立物理连接（在线注入机制）。
  post_process_result(net_name, result, grid) — 校验所有 must-keep 边
    确实出现在 routed_edges 中；不满足时根据 hard 标志决定是否失败。

注意：Node = Tuple[str, int, int]，即 (layer, x, y)。
"""

from __future__ import annotations
import logging
from typing import Dict, FrozenSet, List, Set, Tuple

from maze_router.constraints.base import BaseConstraint
from maze_router.data.net import Node

logger = logging.getLogger(__name__)

# Edge key: frozenset of two nodes (undirected)
_EdgeKey = FrozenSet[Node]


def _edge_key(a: Node, b: Node) -> _EdgeKey:
    return frozenset({a, b})


class PathConstraint(BaseConstraint):
    """
    边级路径约束。

    参数
    ----
    must_keep_edges : Dict[str, List[Tuple[Node, Node]]], optional
        {net_name: [(a, b), ...]}
        路由结果中必须包含这些边。

    must_forbid_edges : Dict[str, List[Tuple[Node, Node]]], optional
        {net_name: [(a, b), ...]}
        路由过程中禁止经过这些边。

    hard : bool
        True（默认）：must-keep 校验失败时 result.success = False。
        False：仅记录 debug 日志（soft rule）。
    """

    def __init__(
        self,
        must_keep_edges: Dict[str, List[Tuple[Node, Node]]] = None,
        must_forbid_edges: Dict[str, List[Tuple[Node, Node]]] = None,
        hard: bool = True,
    ):
        self.hard = hard

        # 预处理 forbid：{net_name: Set[FrozenSet[Node]]}
        self._forbid: Dict[str, Set[_EdgeKey]] = {}
        for net_name, edges in (must_forbid_edges or {}).items():
            self._forbid[net_name] = {_edge_key(a, b) for a, b in edges}

        # 预处理 keep：{net_name: List[(a, b)]}（保持原始顺序供校验使用）
        self._keep: Dict[str, List[Tuple[Node, Node]]] = dict(must_keep_edges or {})

        # keep 边 key 集合，供快速校验
        self._keep_keys: Dict[str, Set[_EdgeKey]] = {
            net_name: {_edge_key(a, b) for a, b in edges}
            for net_name, edges in self._keep.items()
        }

    # ------------------------------------------------------------------
    # BaseConstraint 接口（节点级，不阻断）
    # ------------------------------------------------------------------

    def is_available(self, node: Node, net_name: str) -> bool:
        return True

    def mark_route(self, net_name: str, nodes: Set[Node]):
        pass

    def unmark_route(self, net_name: str):
        pass

    # ------------------------------------------------------------------
    # 边级可用性（must_forbid_edges 核心逻辑）
    # ------------------------------------------------------------------

    def is_edge_available(self, node_a: Node, node_b: Node, net_name: str) -> bool:
        """
        检查边 (node_a, node_b) 对给定线网是否可用。

        若边在 must_forbid_edges 中，返回 False；否则返回 True。
        时间复杂度 O(1)。
        """
        forbid_set = self._forbid.get(net_name)
        if forbid_set is None:
            return True
        return _edge_key(node_a, node_b) not in forbid_set

    def get_must_keep_edges(self, net_name: str) -> List[Tuple[Node, Node]]:
        """返回该线网的 must-keep 边列表，供路由算法在布线过程中强制经过。"""
        return list(self._keep.get(net_name, []))

    # ------------------------------------------------------------------
    # 在线终端注入（must_keep_edges 保证连通）
    # ------------------------------------------------------------------

    def required_terminals(self, net_name: str, grid) -> List[Node]:
        """
        将 must-keep 边的两个端点注入为强制终端。

        路由器在连接这些终端时会自然经过 must-keep 边（当边的代价合理时）。
        后处理 post_process_result 负责最终校验。
        """
        keep_edges = self._keep.get(net_name)
        if not keep_edges:
            return []

        result: List[Node] = []
        seen: Set[Node] = set()
        for a, b in keep_edges:
            for node in (a, b):
                if node not in seen:
                    seen.add(node)
                    if grid.is_valid_node(node):
                        result.append(node)

        return result

    # ------------------------------------------------------------------
    # 后处理：校验 must-keep 边实际出现在路由结果中
    # ------------------------------------------------------------------

    def post_process_result(self, net_name: str, result, grid) -> None:
        """
        校验所有 must-keep 边是否出现在 routed_edges 中。

        若 must-keep 边缺失且 hard=True，标记 result.success = False。
        """
        if not result.success:
            return

        keep_keys = self._keep_keys.get(net_name)
        if not keep_keys:
            return

        routed_edge_keys: Set[_EdgeKey] = {
            _edge_key(a, b) for a, b in result.routed_edges
        }

        keep_edges = self._keep.get(net_name, [])
        for a, b in keep_edges:
            key = _edge_key(a, b)
            if key not in routed_edge_keys:
                msg = (
                    f"线网 {net_name}: must_keep_edge {(a, b)} "
                    f"未出现在路由结果中"
                )
                if self.hard:
                    logger.warning(msg + "，标记布线失败")
                    result.success = False
                    return
                else:
                    logger.debug(msg + "（soft rule，忽略）")
