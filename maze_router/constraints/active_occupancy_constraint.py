"""
Active 区域 M1 最小占用约束 (ActiveOccupancyConstraint)

工作原理（在线化方案）
----------------------
在路由开始前，通过 required_terminals() 将 M1 active 覆盖所需节点注入为
强制终端（由 Router 调用 ConstraintManager.collect_required_terminals()
统一收集后合并到线网终端列表）。路由算法自然建立物理连接，从而保证覆盖。

post_process_result 仅作保底校验，不再执行 BFS 扩展。

参数语义
--------
rules : Dict[Tuple[str, int, int], int]
    {(net_name, j, i): N}

    net_name : 线网名称
    j        : SD 列索引（版图中的 x 坐标）
    i        : 行类型（行索引）
                 0 → PMOS（默认）
                 1 → NMOS H6
                 2 → NMOS H4
                 以此类推
    N        : 该 SD Terminal 在 M1 层对应列 j、对应行范围内必须覆盖的
               最少节点数（通常 N = active 格点数）。

row_type_y_ranges : Dict[int, Tuple[int, int]]
    {i: (y_min, y_max)}  ——  行类型 i 对应的 y 坐标范围（含端点）。

注意：虚拟节点（grid.is_virtual_node() 为 True）不计入 M1 节点数。
"""

from __future__ import annotations
import logging
from typing import Dict, List, Set, Tuple

from maze_router.constraints.base import BaseConstraint
from maze_router.data.net import Node

logger = logging.getLogger(__name__)


class ActiveOccupancyConstraint(BaseConstraint):
    """
    Active 区域 M1 最小占用约束（per-net, per-SD-terminal）。

    在线化方案：通过 required_terminals() 在路由前注入强制 M1 终端，
    使路由器自然覆盖整个 active 列段，无需后处理 BFS 补丁。
    post_process_result 仅做最终合规性校验。

    参数
    ----
    rules : Dict[Tuple[str, int, int], int]
        {(net_name, j, i): N}
        每个 SD Terminal 在 M1 层对应列 j、行类型 i 的 y 范围内必须覆盖 N 个节点。

    row_type_y_ranges : Dict[int, Tuple[int, int]]
        {i: (y_min, y_max)}  行类型 i 对应的网格 y 坐标范围（含端点）。

    hard : bool
        True（默认）：校验不满足时 result.success = False。
        False：仅记录 debug 日志（soft rule）。
    """

    def __init__(
        self,
        rules: Dict[Tuple[str, int, int], int],
        row_type_y_ranges: Dict[int, Tuple[int, int]],
        hard: bool = True,
    ):
        # {(net_name, j, i): N}
        self.rules: Dict[Tuple[str, int, int], int] = {
            k: max(1, v) for k, v in rules.items()
        }
        # {i: (y_min, y_max)}
        self.row_type_y_ranges: Dict[int, Tuple[int, int]] = row_type_y_ranges
        self.hard: bool = hard

        # 预计算: {net_name: [(j, i, N, y_min, y_max), ...]}
        self._net_rules: Dict[str, List[Tuple[int, int, int, int, int]]] = {}
        for (net_name, j, i), n in self.rules.items():
            y_range = self.row_type_y_ranges.get(i)
            if y_range is None:
                logger.warning(
                    f"ActiveOccupancyConstraint: 行类型 i={i} 未在 "
                    f"row_type_y_ranges 中定义，跳过 ({net_name}, {j}, {i})"
                )
                continue
            y_min, y_max = y_range
            self._net_rules.setdefault(net_name, []).append(
                (j, i, n, y_min, y_max)
            )

    # ------------------------------------------------------------------
    # BaseConstraint 接口（不阻断路由）
    # ------------------------------------------------------------------

    def is_available(self, node: Node, net_name: str) -> bool:
        return True

    def mark_route(self, net_name: str, nodes: Set[Node]):
        pass

    def unmark_route(self, net_name: str):
        pass

    # ------------------------------------------------------------------
    # 在线终端注入（核心机制）
    # ------------------------------------------------------------------

    def required_terminals(self, net_name: str, grid) -> List[Node]:
        """
        返回该线网 active 覆盖所需的强制 M1 终端节点列表。

        对每个 (j, i, N) 规则，返回 M1 层列 j 在 [y_min, y_max] 范围内
        的所有有效非虚拟节点。Router 将这些节点合并到线网终端列表，
        使路由算法自然建立从 SD via 点到整个 active 列段的物理连接。

        为何此方案保证覆盖：
          M1 同列竖向路径是连接两端点的最优路径（任何绕行代价更高），
          因此贪心/DP Steiner 路由器都会选择直列路径，覆盖所有中间节点。
        """
        rules_for_net = self._net_rules.get(net_name)
        if not rules_for_net:
            return []

        result: List[Node] = []
        for (j, i, _min_needed, y_min, y_max) in rules_for_net:
            for y in range(y_min, y_max + 1):
                node = ("M1", j, y)
                if grid.is_valid_node(node) and not grid.is_virtual_node(node):
                    result.append(node)

        return result

    # ------------------------------------------------------------------
    # 后处理：保底校验（不再执行 BFS 扩展）
    # ------------------------------------------------------------------

    def post_process_result(self, net_name: str, result, grid) -> None:
        """
        路由完成后的合规性校验（保底检查）。

        在线注入机制正常工作时此处不应触发失败。
        若仍不满足（极端拥塞情况），根据 hard 标志决定是否标记失败。
        """
        if not result.success:
            return

        rules_for_net = self._net_rules.get(net_name)
        if not rules_for_net:
            return

        for (j, i, min_needed, y_min, y_max) in rules_for_net:
            actual = sum(
                1 for n in result.routed_nodes
                if n[0] == "M1"
                and n[1] == j
                and y_min <= n[2] <= y_max
                and not grid.is_virtual_node(n)
            )

            if actual >= min_needed:
                logger.debug(
                    f"线网 {net_name} (j={j}, i={i}): M1 覆盖满足 "
                    f"({actual}/{min_needed})"
                )
                continue

            msg = (
                f"线网 {net_name} (j={j}, i={i}): M1 覆盖 {actual} < "
                f"要求 {min_needed}"
                + ("，标记布线失败" if self.hard else "（soft rule，忽略）")
            )
            if self.hard:
                logger.warning(msg)
                result.success = False
                return
            else:
                logger.debug(msg)
