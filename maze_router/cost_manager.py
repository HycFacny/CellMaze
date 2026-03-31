"""
代价管理器 (CostManager)

聚合所有 BaseCost 实例，统一计算路由代价。
支持动态插入新代价，便于扩展。

代价由以下部分组成：
  1. 网格边代价（来自 GridGraph）
  2. 拥塞代价（PathFinder 风格，迭代累加）
  3. 软间距代价（SpaceCost，通过 BaseCost 插件接口）
  4. 折角代价（CornerCost，路由器直接查询，不走此方法）
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from maze_router.costs.base import BaseCost
from maze_router.costs.corner_cost import CornerCost
from maze_router.data.grid import GridGraph
from maze_router.data.net import Node


# -----------------------------------------------------------------------
# 路由上下文（传给 cost 插件）
# -----------------------------------------------------------------------

@dataclass
class RoutingContext:
    """
    传给 cost 计算的上下文。

    Attributes:
        net_name:        当前线网名称
        constraint_mgr:  约束管理器（SpaceCost 需要查询占用情况）
        congestion_map:  节点级拥塞代价映射
        iteration:       当前迭代轮次
    """
    net_name: str
    constraint_mgr: object = None          # ConstraintManager（循环引用避免强类型注解）
    congestion_map: Dict[Node, float] = field(default_factory=dict)
    iteration: int = 1


# -----------------------------------------------------------------------
# CostManager
# -----------------------------------------------------------------------

class CostManager:
    """
    代价管理器。

    统一管理：
    - 网格基础边代价
    - 拥塞地图（PathFinder 风格）
    - 插件式软代价（BaseCost 子类）
    - 折角代价（CornerCost，供路由器直接查询）
    """

    def __init__(
        self,
        grid: GridGraph,
        costs: Optional[List[BaseCost]] = None,
        congestion_weight: float = 1.0,
    ):
        """
        参数:
            grid:              布线网格
            costs:             软代价列表（BaseCost 子类）
            congestion_weight: 拥塞代价的权重系数
        """
        self.grid = grid
        self._costs: List[BaseCost] = costs or []
        self.congestion_weight = congestion_weight
        self._congestion_map: Dict[Node, float] = defaultdict(float)

        # 找出 CornerCost 实例（路由器直接查询）
        self._corner_cost: CornerCost = CornerCost.default()
        for c in self._costs:
            if isinstance(c, CornerCost):
                self._corner_cost = c
                break

    def add_cost(self, cost: BaseCost):
        """动态插入新代价。"""
        self._costs.append(cost)
        if isinstance(cost, CornerCost):
            self._corner_cost = cost

    # ------------------------------------------------------------------
    # 边代价（路由器主循环使用）
    # ------------------------------------------------------------------

    def get_edge_cost(self, src: Node, dst: Node, ctx: RoutingContext) -> float:
        """
        计算从 src 到 dst 的完整移动代价（不含折角）。

        代价 = 网格边代价 + 拥塞代价 + 软约束代价
        """
        base = self.grid.get_edge_cost(src, dst)

        # 拥塞代价
        cong = self._congestion_map.get(dst, 0.0) * self.congestion_weight

        # 软约束代价（BaseCost 插件）
        soft = 0.0
        for cost in self._costs:
            soft += cost.get_node_penalty(dst, ctx.net_name, ctx)

        return base + cong + soft

    # ------------------------------------------------------------------
    # 折角代价（路由器 Dijkstra 阶段直接调用）
    # ------------------------------------------------------------------

    def get_corner_l_cost(self, layer: str, dir_in: int, dir_out: int) -> float:
        """获取 L 型折角代价。"""
        return self._corner_cost.get_l_cost(layer, dir_in, dir_out)

    def get_corner_t_cost_flat(self, layer: str) -> float:
        """获取 T 型折角的平均代价（DP 合并步骤用）。"""
        return self._corner_cost.get_t_cost_flat(layer)

    def get_corner_t_cost(self, layer: str, dir1: int, dir2: int) -> float:
        """获取 T 型折角的方向感知代价。"""
        return self._corner_cost.get_t_cost(layer, dir1, dir2)

    # ------------------------------------------------------------------
    # 拥塞管理
    # ------------------------------------------------------------------

    def update_congestion(self, node: Node, increment: float = 1.0):
        """记录节点拥塞（拆线时调用，PathFinder 风格）。"""
        self._congestion_map[node] += increment

    def get_congestion_map(self) -> Dict[Node, float]:
        return dict(self._congestion_map)

    def reset_congestion(self):
        """清空拥塞计数（重新开始布线时调用）。"""
        self._congestion_map.clear()

    # ------------------------------------------------------------------
    # RoutingContext 工厂
    # ------------------------------------------------------------------

    def make_context(
        self,
        net_name: str,
        constraint_mgr=None,
        iteration: int = 1,
    ) -> RoutingContext:
        """创建路由上下文。"""
        return RoutingContext(
            net_name=net_name,
            constraint_mgr=constraint_mgr,
            congestion_map=self._congestion_map,
            iteration=iteration,
        )

    def __repr__(self) -> str:
        return (
            f"CostManager(costs={len(self._costs)}, "
            f"congestion_nodes={len(self._congestion_map)})"
        )
