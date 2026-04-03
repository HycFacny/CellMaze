"""
Active 区域 M1 最小占用约束 (ActiveOccupancyConstraint)

部分工艺（如 YMTC CJ3）要求 M1 走线在 S/D 对应列的 active 行范围内覆盖足够
多的节点，以满足 Active Coverage DRC 规则。

参数语义
--------
rules : Dict[Tuple[str, int, int], int]
    {(net_name, j, i): N}

    net_name : 线网名称
    j        : SD 列索引（版图中的 x 坐标）
    i        : 行类型（行索引）
                 0 → NMOS（单行布局中的下排晶体管，或多行布局第 1 行 NMOS）
                 1 → PMOS（单行布局中的上排晶体管）
                 2 → NMOS_R / PMOS 第 2 行（多行布局扩展）
                 以此类推
    N        : 该 SD Terminal 在 M1 层对应列 j、对应行范围内必须覆盖的
               最少节点数。通常 N = active 格点数（覆盖全部 active 区域）。

row_type_y_ranges : Dict[int, Tuple[int, int]]
    {i: (y_min, y_max)}  ——  行类型 i 对应的 y 坐标范围（含端点）。
    例如单行标准单元（9行）：
        {0: (0, 3), 1: (5, 8)}

工作原理
--------
路由成功后作为 post-process hook 被调用：
  1. 在 routed_nodes 中统计 M1 层 x==j 且 y ∈ [y_min, y_max] 的节点数；
  2. 若不足 N，BFS 沿 M1 列 j 在范围内向上/下扩展节点直到满足；
  3. 若仍不足（hard=True）：将 result.success 置为 False，触发拆线重布。

注意：虚拟节点（grid.is_virtual_node() 为 True）不计入 M1 节点数。
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Set, Tuple

from maze_router.constraints.base import BaseConstraint
from maze_router.data.net import Node

logger = logging.getLogger(__name__)


class ActiveOccupancyConstraint(BaseConstraint):
    """
    Active 区域 M1 最小占用约束（per-net, per-SD-terminal）。

    参数
    ----
    rules : Dict[Tuple[str, int, int], int]
        {(net_name, j, i): N}
        每个 SD Terminal 在 M1 层对应列 j、行类型 i 的 y 范围内必须覆盖 N 个节点。

    row_type_y_ranges : Dict[int, Tuple[int, int]]
        {i: (y_min, y_max)}  行类型 i 对应的网格 y 坐标范围（含端点）。

    hard : bool
        True（默认）：扩展后仍不满足时 result.success = False。
        False：仅记录 debug 日志，不影响布线成功状态（soft rule）。
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
    # 后处理 hook
    # ------------------------------------------------------------------

    def post_process_result(self, net_name: str, result, grid) -> None:
        """
        检查并扩展 M1 列节点以满足 active 覆盖约束。

        对每个 (j, i, N) 规则：
          1. 统计 routed_nodes 中 layer==M1, x==j, y_min ≤ y ≤ y_max 的节点数
          2. 不足时 BFS 沿 M1 列 j 在 [y_min, y_max] 内扩展
          3. 仍不足（hard=True）→ result.success = False
        """
        if not result.success:
            return

        rules_for_net = self._net_rules.get(net_name)
        if not rules_for_net:
            return

        for (j, i, min_needed, y_min, y_max) in rules_for_net:
            # 统计满足条件的 M1 节点（排除虚拟节点）
            m1_col_nodes: Set[Node] = {
                n for n in result.routed_nodes
                if n[0] == "M1"
                and n[1] == j
                and y_min <= n[2] <= y_max
                and not grid.is_virtual_node(n)
            }

            deficit = min_needed - len(m1_col_nodes)
            if deficit <= 0:
                continue

            logger.debug(
                f"线网 {net_name} (j={j}, i={i}): M1 节点 {len(m1_col_nodes)} < "
                f"要求 {min_needed}，BFS 扩展 {deficit} 个节点"
            )

            # BFS：沿 M1 列 j 在 [y_min, y_max] 内扩展
            frontier = list(m1_col_nodes)
            visited: Set[Node] = set(m1_col_nodes)
            added = 0

            while frontier and added < deficit:
                next_frontier = []
                for n in frontier:
                    for dy in (-1, 1):
                        nb = ("M1", j, n[2] + dy)
                        if nb in visited:
                            continue
                        if not (y_min <= nb[2] <= y_max):
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

            final_count = sum(
                1 for n in result.routed_nodes
                if n[0] == "M1"
                and n[1] == j
                and y_min <= n[2] <= y_max
                and not grid.is_virtual_node(n)
            )

            if final_count < min_needed:
                msg = (
                    f"线网 {net_name} (j={j}, i={i}): M1 覆盖 {final_count} < "
                    f"要求 {min_needed}"
                    + ("，标记布线失败" if self.hard else "（soft rule，忽略）")
                )
                if self.hard:
                    logger.warning(msg)
                    result.success = False
                    return  # 一旦某个 SD terminal 不满足，直接失败
                else:
                    logger.debug(msg)
            else:
                logger.debug(
                    f"线网 {net_name} (j={j}, i={i}): M1 覆盖满足 "
                    f"({final_count}/{min_needed})"
                )
