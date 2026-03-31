"""
软间距代价 (SpaceCost)

当布线节点与其他线网节点的 Chebyshev 距离低于软间距阈值时，施加额外代价惩罚。
支持三种间距类型：
  S2S (side-to-side): 线段平行走线时的侧面间距代价
  T2S (tip-to-side):  线段端头与另一线段侧面的间距代价
  T2T (tip-to-tip):   线段端头与另一线段端头的间距代价

注意：当前实现将三种类型统一按 Chebyshev 距离计算，
T2S/T2T 的方向性区分留作扩展（使用统一软阈值即可满足大多数场景）。
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from maze_router.costs.base import BaseCost
from maze_router.data.net import Node


class SpaceType(Enum):
    """间距类型"""
    S2S = "side_to_side"    # 侧面到侧面
    T2S = "tip_to_side"     # 端头到侧面
    T2T = "tip_to_tip"      # 端头到端头


@dataclass
class SpaceCostRule:
    """
    软间距代价规则。

    Attributes:
        layer:       适用层名（"M0"/"M1"/"M2"）
        space_type:  间距类型
        soft_space:  软间距触发阈值（Chebyshev 距离）
        penalty:     进入该软间距区域的额外代价
    """
    layer: str
    space_type: SpaceType
    soft_space: int
    penalty: float


class SpaceCost(BaseCost):
    """
    软间距代价。

    检查候选节点与其他线网的已标记节点之间的 Chebyshev 距离：
    若距离 < soft_space（对应规则），则施加 penalty 代价。
    """

    def __init__(self, rules: List[SpaceCostRule]):
        """
        参数:
            rules: 软间距代价规则列表
        """
        self.rules = rules
        # 按层分组，加快查询
        self._layer_rules: Dict[str, List[SpaceCostRule]] = {}
        for rule in rules:
            self._layer_rules.setdefault(rule.layer, []).append(rule)

    def get_node_penalty(self, node: Node, net_name: str, context) -> float:
        """
        计算节点的软间距代价。

        context 须提供 constraint_mgr（ConstraintManager）以查询其他线网节点。
        """
        layer = node[0]
        if layer not in self._layer_rules:
            return 0.0

        constraint_mgr = getattr(context, 'constraint_mgr', None)
        if constraint_mgr is None:
            return 0.0

        total_penalty = 0.0
        layer_rules = self._layer_rules[layer]

        # 找所有其他线网的已标记节点（通过 constraint_mgr）
        max_soft_space = max(r.soft_space for r in layer_rules)
        x, y = node[1], node[2]

        for rule in layer_rules:
            ss = rule.soft_space
            # 在 ss 邻域内查找其他线网节点
            for dx in range(-ss + 1, ss):
                for dy in range(-ss + 1, ss):
                    cheby = max(abs(dx), abs(dy))
                    if cheby >= ss:
                        continue
                    neighbor = (layer, x + dx, y + dy)
                    # 检查是否有其他线网占用
                    blockers = constraint_mgr.get_blocking_nets(neighbor, net_name)
                    if blockers:
                        total_penalty += rule.penalty
                        break  # 该规则已触发，不重复计
            # 避免重复计算（每条规则只加一次）

        return total_penalty
