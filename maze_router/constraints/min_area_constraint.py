"""
最小面积约束 (MinAreaConstraint)

Soft rule：线网在某层若有布线节点，则该层节点数必须 ≥ min_area[layer]。
默认 min_area=2，clamp 到 [1, 4]。

实现为后处理 hook：路由成功后调用 post_process_result()，
通过 BFS 扩展节点直到满足 min_area。
无法满足时仍保留 success=True（soft rule，不触发拆线）。
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, Set

from maze_router.constraints.base import BaseConstraint
from maze_router.data.net import Node


class MinAreaConstraint(BaseConstraint):
    """
    最小面积约束（Soft Rule）。

    线网在层 L 上若有任何路由节点，则该层节点总数 ≥ min_area[L]。
    违规时通过 BFS 扩展同层相邻有效节点。

    注意：此约束不阻断路由（is_available 始终返回 True），
    仅在路由完成后通过 post_process_result() 进行后处理扩展。
    """

    DEFAULT_MIN_AREA = 2
    MAX_MIN_AREA = 4

    def __init__(self, rules: Dict[str, int] = None):
        """
        参数:
            rules: {layer: min_nodes}，每层的最小节点数要求。
                   None 或未指定层 → 使用默认值 DEFAULT_MIN_AREA=2。
                   值被 clamp 到 [1, MAX_MIN_AREA=4]。
        """
        if rules is None:
            rules = {}
        self.rules: Dict[str, int] = {
            layer: max(1, min(self.MAX_MIN_AREA, v))
            for layer, v in rules.items()
        }

    # ------------------------------------------------------------------
    # BaseConstraint 接口（Soft Rule —— 不阻断路由）
    # ------------------------------------------------------------------

    def is_available(self, node: Node, net_name: str) -> bool:
        """始终返回 True：min_area 不阻断路由时序。"""
        return True

    def mark_route(self, net_name: str, nodes: Set[Node]):
        """no-op：min_area 不需要跟踪占用状态。"""

    def unmark_route(self, net_name: str):
        """no-op。"""

    # ------------------------------------------------------------------
    # 后处理 hook
    # ------------------------------------------------------------------

    def post_process_result(self, net_name: str, result, grid) -> None:
        """
        对路由成功的线网，检查每层节点数是否满足 min_area。
        不足时从已有节点 BFS 扩展同层有效相邻节点。

        扩展节点直接加入 result.routed_nodes（soft rule）。
        无法扩展到目标数量时也不影响 result.success。
        """
        if not result.success:
            return

        # 按层统计已有节点
        layer_nodes: Dict[str, Set[Node]] = defaultdict(set)
        for node in result.routed_nodes:
            layer_nodes[node[0]].add(node)

        for layer, nodes in layer_nodes.items():
            min_needed = self.rules.get(layer, self.DEFAULT_MIN_AREA)
            deficit = min_needed - len(nodes)
            if deficit <= 0:
                continue

            # BFS：从该层已有节点向外扩展
            frontier = list(nodes)
            visited: Set[Node] = set(nodes)
            added = 0

            while frontier and added < deficit:
                next_frontier = []
                for n in frontier:
                    for nb in grid.get_neighbors(n):
                        if nb[0] != layer:
                            continue
                        if nb in visited:
                            continue
                        if not grid.is_valid_node(nb):
                            continue
                        visited.add(nb)
                        result.routed_nodes.add(nb)
                        next_frontier.append(nb)
                        added += 1
                        if added >= deficit:
                            break
                    if added >= deficit:
                        break
                frontier = next_frontier

    # ------------------------------------------------------------------
    # 查询辅助
    # ------------------------------------------------------------------

    def get_min_area(self, layer: str) -> int:
        """返回指定层的 min_area 要求。"""
        return self.rules.get(layer, self.DEFAULT_MIN_AREA)

    def check_violations(self, routed_nodes: Set[Node]) -> Dict[str, int]:
        """
        检查给定节点集合是否满足 min_area。

        返回:
            {layer: actual_count} — 仅包含违规层（actual < min_area）
        """
        layer_counts: Dict[str, int] = defaultdict(int)
        for node in routed_nodes:
            layer_counts[node[0]] += 1

        violations = {}
        for layer, count in layer_counts.items():
            if count < self.get_min_area(layer):
                violations[layer] = count
        return violations
