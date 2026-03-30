"""
折角代价管理模块

管理布线折角代价，包括：
  L型折角（L-corner）：同层路径方向改变时的代价，支持按方向差异化配置。
  T型折角（T-corner）：Steiner树分支节点处三条或更多路径汇聚时的代价。

代价配置格式（l_costs 和 t_costs 均支持以下格式）：
  None              → 使用默认值（L型=5.0，T型=0.0）
  {}                → 禁用该类型折角代价
  {layer: float}    → 按层统一代价
  {(layer, d1, d2): float} → 按方向的精细代价（精细键优先于层键）

方向编码（与 router.py 保持一致）：
  0=无方向（初始/via后），1=右，2=左，3=上，4=下
"""

from typing import Dict, Optional, Tuple, Union


# 折角代价键类型
_CostKey = Union[str, Tuple[str, int, int]]


class CornerManager:
    """
    折角代价管理器。

    统一管理L型折角和T型折角的代价配置与查询：

    L型折角：
      路径在同层改变方向时（如右转上）产生的代价。
      MazeRouter 和 SteinerRouter 在 Dijkstra 松弛阶段查询此代价。
      支持按 (layer, dir_in, dir_out) 的方向敏感代价。

    T型折角：
      Steiner 树中分支节点处两条子树汇聚时产生的代价。
      SteinerRouter 在 Dreyfus-Wagner DP 的子集合并步骤中查询此代价。
      支持按层的统一代价（DP 合并步骤不追踪方向）和按 (layer, dir1, dir2) 的
      方向敏感代价（用于后处理分析）。
    """

    DEFAULT_L_COST: float = 5.0   # 默认L型折角代价
    DEFAULT_T_COST: float = 0.0   # 默认T型折角代价（默认不启用）

    def __init__(
        self,
        l_costs: Optional[Dict[_CostKey, float]] = None,
        t_costs: Optional[Dict[_CostKey, float]] = None,
    ):
        """
        参数:
            l_costs: L型折角代价配置
                None              → 所有层和方向使用默认值 DEFAULT_L_COST=5.0
                {}                → 关闭L型折角代价（全局禁用）
                {layer: cost}     → 按层统一代价
                {(layer, dir_in, dir_out): cost} → 按方向精细代价（优先于层键）
            t_costs: T型折角代价配置
                None              → 所有层使用默认值 DEFAULT_T_COST=0.0
                {}                → 关闭T型折角代价（全局禁用）
                {layer: cost}     → 按层统一代价（用于DP合并步骤）
                {(layer, d1, d2): cost} → 按方向精细代价（方向对顺序无关）

        示例:
            CornerManager()
                → 默认L型折角5.0，T型折角0.0
            CornerManager(l_costs={})
                → 关闭所有折角代价
            CornerManager(l_costs={"M0": 3.0, "M1": 5.0}, t_costs={"M0": 2.0})
                → 按层配置L和T型代价
            CornerManager(l_costs={("M0", 1, 3): 10.0, ("M0", 3, 1): 8.0})
                → 方向敏感的L型代价（右转上=10，上转右=8）
        """
        # 状态标志
        self._l_none = l_costs is None       # None → 使用全局默认值
        self._t_none = t_costs is None       # None → 使用全局默认值
        self._l_disabled = (l_costs is not None and len(l_costs) == 0)
        self._t_disabled = (t_costs is not None and len(t_costs) == 0)

        # 存储配置
        self._l_costs: Dict = l_costs if l_costs else {}
        self._t_costs: Dict = t_costs if t_costs else {}

    # ------------------------------------------------------------------
    # L型折角
    # ------------------------------------------------------------------

    def get_l_cost(self, layer: str, dir_in: int, dir_out: int) -> float:
        """
        获取L型折角代价。

        路由器在同层方向改变时调用此方法：
        dir_in != dir_out 且两者均不为 DIR_NONE（0）时触发查询。

        查找顺序：
          1. 精细键 (layer, dir_in, dir_out)
          2. 层键 layer
          3. 默认值（None配置→5.0，{}配置→0.0，其余配置→0.0）

        参数:
            layer: 当前节点所在层名（如 "M0"）
            dir_in: 进入方向码（1=右,2=左,3=上,4=下）
            dir_out: 离开方向码

        返回:
            L型折角代价（非负浮点数）
        """
        if self._l_disabled:
            return 0.0

        # 精细键优先
        fine_key = (layer, dir_in, dir_out)
        if fine_key in self._l_costs:
            return self._l_costs[fine_key]

        # 层键
        if layer in self._l_costs:
            return self._l_costs[layer]

        # 默认值
        return self.DEFAULT_L_COST if self._l_none else 0.0

    # ------------------------------------------------------------------
    # T型折角
    # ------------------------------------------------------------------

    def get_t_cost_flat(self, layer: str) -> float:
        """
        获取T型折角的平均代价（用于 DP 子集合并步骤）。

        Dreyfus-Wagner DP 在合并两棵子树时调用，此时无法追踪方向，
        因此使用按层的统一代价作为分支惩罚。

        查找顺序：
          1. 层键 layer
          2. 默认值（None配置→0.0，{}配置→0.0）

        参数:
            layer: 分支节点所在层名

        返回:
            T型折角平均代价（非负浮点数）
        """
        if self._t_disabled:
            return 0.0

        if layer in self._t_costs:
            return self._t_costs[layer]

        return self.DEFAULT_T_COST if self._t_none else 0.0

    def get_t_cost(self, layer: str, dir1: int, dir2: int) -> float:
        """
        获取T型折角的方向感知代价（用于后处理分析）。

        dir1 和 dir2 是两条到达分支节点的子树路径方向。
        方向对是无序的：(dir1, dir2) 和 (dir2, dir1) 查询结果相同。

        查找顺序：
          1. 精细键 (layer, min(dir1,dir2), max(dir1,dir2))
          2. 层键（通过 get_t_cost_flat）

        参数:
            layer: 分支节点所在层名
            dir1: 第一条到达方向码
            dir2: 第二条到达方向码

        返回:
            T型折角代价（非负浮点数）
        """
        if self._t_disabled:
            return 0.0

        # 精细键（归一化方向对顺序）
        d_lo, d_hi = min(dir1, dir2), max(dir1, dir2)
        fine_key = (layer, d_lo, d_hi)
        if fine_key in self._t_costs:
            return self._t_costs[fine_key]

        # 回退到层键
        return self.get_t_cost_flat(layer)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def has_l_costs(self) -> bool:
        """是否存在任何L型折角代价（用于快速跳过判断）"""
        if self._l_disabled:
            return False
        if self._l_none:
            return self.DEFAULT_L_COST > 0
        return bool(self._l_costs) and any(v > 0 for v in self._l_costs.values())

    def has_t_costs(self) -> bool:
        """是否存在任何T型折角代价（用于快速跳过判断）"""
        if self._t_disabled:
            return False
        if self._t_none:
            return self.DEFAULT_T_COST > 0
        return bool(self._t_costs) and any(v > 0 for v in self._t_costs.values())

    @classmethod
    def default(cls) -> 'CornerManager':
        """默认配置：L型折角=5.0，T型折角=0.0"""
        return cls()

    @classmethod
    def disabled(cls) -> 'CornerManager':
        """关闭所有折角代价"""
        return cls(l_costs={}, t_costs={})

    @classmethod
    def from_layer_costs(
        cls,
        l_layer_costs: Optional[Dict[str, float]] = None,
        t_layer_costs: Optional[Dict[str, float]] = None,
    ) -> 'CornerManager':
        """
        从按层代价字典创建（向后兼容辅助方法）。

        参数:
            l_layer_costs: {layer: l_cost}，None表示用默认值，{}表示禁用
            t_layer_costs: {layer: t_cost}，None表示用默认值，{}表示禁用
        """
        return cls(l_costs=l_layer_costs, t_costs=t_layer_costs)

    def __repr__(self) -> str:
        if self._l_disabled and self._t_disabled:
            return "CornerManager(all_disabled)"
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
        return f"CornerManager(l={l_desc}, t={t_desc})"
