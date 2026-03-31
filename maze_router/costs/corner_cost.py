

from __future__ import annotations
from typing import Dict, Optional, Tuple, Union

from maze_router.costs.base import BaseCost
from maze_router.data.net import Node

# 折角代价键类型, 格式（l_costs 和 t_costs）：
#     None              → 使用默认值（L型=5.0，T型=10.0）
#     {}                → 禁用该类型折角代价
#     {layer: float}    → 按层统一代价
#     {(layer, d1, d2): float} → 按方向精细代价（精细键优先于层键）
#     方向编码（与 maze_router_algo.py 保持一致）：
#         0=无方向（初始/via后），1=右，2=左，3=上，4=下
_CostKey = Union[str, Tuple[str, int, int]]


class CornerCost(BaseCost):
    """
    折角代价管理器（兼作 BaseCost 子类）。

    L型折角：路径在同层改变方向时（如右转上）产生的代价。
    T型折角：Steiner 树中分支节点处两条子树汇聚时产生的代价。
    """

    DEFAULT_L_COST: float = 5.0
    DEFAULT_T_COST: float = 10.0

    def __init__(
        self,
        l_costs: Optional[Dict[_CostKey, float]] = None,
        t_costs: Optional[Dict[_CostKey, float]] = None,
    ):
        """
        参数:
            l_costs: L型折角代价配置（None=默认5.0, {}=禁用）
            t_costs: T型折角代价配置（None=默认10.0, {}=禁用）
        """
        self._l_none = l_costs is None
        self._t_none = t_costs is None
        self._l_disabled = l_costs is not None and len(l_costs) == 0
        self._t_disabled = t_costs is not None and len(t_costs) == 0
        self._l_costs: Dict = l_costs if l_costs else {}
        self._t_costs: Dict = t_costs if t_costs else {}

    def get_node_penalty(self, node: Node, net_name: str, context) -> float:
        """折角代价不是节点惩罚，由路由器在移动时直接查询。返回 0.0。"""
        return 0.0

    def get_l_cost(self, layer: str, dir_in: int, dir_out: int) -> float:
        """
        获取 L 型折角代价。
        查找顺序：精细键 (layer,d_in,d_out) > 层键 layer > 默认值。
        """
        if self._l_disabled:
            return 0.0
        fine_key = (layer, dir_in, dir_out)
        if fine_key in self._l_costs:
            return self._l_costs[fine_key]
        if layer in self._l_costs:
            return self._l_costs[layer]
        return self.DEFAULT_L_COST if self._l_none else 0.0

    def get_t_cost_flat(self, layer: str) -> float:
        """
        获取 T 型折角的平均代价（用于 DP 子集合并步骤）。
        """
        if self._t_disabled:
            return 0.0
        if layer in self._t_costs:
            return self._t_costs[layer]
        return self.DEFAULT_T_COST if self._t_none else 0.0

    def get_t_cost(self, layer: str, dir1: int, dir2: int) -> float:
        """
        获取 T 型折角的方向感知代价。方向对是无序的。
        """
        if self._t_disabled:
            return 0.0
        d_lo, d_hi = min(dir1, dir2), max(dir1, dir2)
        fine_key = (layer, d_lo, d_hi)
        if fine_key in self._t_costs:
            return self._t_costs[fine_key]
        return self.get_t_cost_flat(layer)

    def has_l_costs(self) -> bool:
        if self._l_disabled:
            return False
        if self._l_none:
            return self.DEFAULT_L_COST > 0
        return bool(self._l_costs) and any(v > 0 for v in self._l_costs.values())

    def has_t_costs(self) -> bool:
        if self._t_disabled:
            return False
        if self._t_none:
            return self.DEFAULT_T_COST > 0
        return bool(self._t_costs) and any(v > 0 for v in self._t_costs.values())

    @classmethod
    def default(cls) -> 'CornerCost':
        """默认配置：L型折角=5.0，T型折角=0.0"""
        return cls()

    @classmethod
    def disabled(cls) -> 'CornerCost':
        """关闭所有折角代价"""
        return cls(l_costs={}, t_costs={})

    @classmethod
    def from_layer_costs(
        cls,
        l_layer_costs: Optional[Dict[str, float]] = None,
        t_layer_costs: Optional[Dict[str, float]] = None,
    ) -> 'CornerCost':
        """从按层代价字典创建。"""
        return cls(l_costs=l_layer_costs, t_costs=t_layer_costs)

    def __repr__(self) -> str:
        if self._l_disabled and self._t_disabled:
            return "CornerCost(all_disabled)"
        l_desc = (
            "default(5.0)" if self._l_none
            else "disabled" if self._l_disabled
            else repr(self._l_costs)
        )
        t_desc = (
            "default(0.0)" if self._t_none
            else "disabled" if self._t_disabled
            else repr(self._t_costs)
        )
        return f"CornerCost(l={l_desc}, t={t_desc})"
