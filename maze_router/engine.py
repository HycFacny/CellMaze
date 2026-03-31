"""
MazeRouterEngine: 顶层调度类

负责：
  1. 接收外部输入（GridGraph, nets, 约束参数, 代价参数）
  2. 组装 ConstraintManager、CostManager、RipupStrategy、RipupManager
  3. 调用 RipupManager.run() 执行布线
  4. 可选地调用 Visualizer 输出 SVG
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Set

from maze_router.data.net import Net, Node, PinSpec, RoutingSolution
from maze_router.data.grid import GridGraph
from maze_router.constraint_manager import ConstraintManager
from maze_router.cost_manager import CostManager
from maze_router.constraints.space_constraint import SpaceConstraint
from maze_router.constraints.min_area_constraint import MinAreaConstraint
from maze_router.costs.corner_cost import CornerCost
from maze_router.costs.space_cost import SpaceCost, SpaceCostRule
from maze_router.ripup_strategy import RipupStrategy, DefaultRipupStrategy
from maze_router.ripup_manager import RipupManager

logger = logging.getLogger(__name__)


class MazeRouterEngine:
    """
    顶层路由调度引擎。

    使用方法：
        engine = MazeRouterEngine(grid, nets, space_constr={"M0":1,"M1":1,"M2":1})
        solution = engine.run()
        engine.visualize(save_dir="results/")

    参数:
        grid:            布线网格（GridGraph）
        nets:            需要布线的线网列表
        space_constr:    硬间距约束 {layer: min_space}
        corner_cost:     折角代价（CornerCost 实例，或 None=默认）
        space_cost_rules: 软间距代价规则列表（SpaceCostRule）
        cable_locs:      {net_name: Set[Node]}，M0 可走节点集合
        pin_layers:      {net_name: layer_str}，Pin 引出层规格
        strategy:        拆线重布策略（None=DefaultRipupStrategy）
        congestion_weight: 拥塞代价权重
        min_area:        最小面积约束 {layer: min_nodes}（soft rule，默认=2，最大=4）
    """

    def __init__(
        self,
        grid: GridGraph,
        nets: List[Net],
        space_constr: Optional[Dict[str, int]] = None,
        corner_cost: Optional[CornerCost] = None,
        space_cost_rules: Optional[List[SpaceCostRule]] = None,
        cable_locs: Optional[Dict[str, Set[Node]]] = None,
        pin_layers: Optional[Dict[str, str]] = None,
        strategy: Optional[RipupStrategy] = None,
        congestion_weight: float = 1.0,
        min_area: Optional[Dict[str, int]] = None,
    ):
        self.grid = grid
        self.nets = nets

        # 应用 cable_locs 和 pin_layers 到线网对象
        if cable_locs:
            for net in self.nets:
                if net.name in cable_locs:
                    net.cable_locs = cable_locs[net.name]

        if pin_layers:
            for net in self.nets:
                if net.name in pin_layers:
                    net.pin_spec = PinSpec(layer=pin_layers[net.name])

        # 组装 ConstraintManager
        constraints = []
        if space_constr:
            constraints.append(SpaceConstraint(rules=space_constr))
        if min_area is not None:
            constraints.append(MinAreaConstraint(rules=min_area))
        self.constraint_mgr = ConstraintManager(constraints)

        # 组装 CostManager
        costs = []
        if corner_cost is not None:
            costs.append(corner_cost)
        else:
            costs.append(CornerCost.default())
        if space_cost_rules:
            costs.append(SpaceCost(space_cost_rules))
        self.cost_mgr = CostManager(
            grid=grid,
            costs=costs,
            congestion_weight=congestion_weight,
        )

        # 策略
        self.strategy = strategy or DefaultRipupStrategy()

        # RipupManager
        self.ripup_mgr = RipupManager(
            grid=grid,
            constraint_mgr=self.constraint_mgr,
            cost_mgr=self.cost_mgr,
            strategy=self.strategy,
        )

        self._solution: Optional[RoutingSolution] = None

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(self) -> RoutingSolution:
        """执行完整布线流程，返回布线方案。"""
        logger.info(
            f"MazeRouterEngine 开始: {len(self.nets)} 个线网，"
            f"网格 {self.grid}"
        )
        self._solution = self.ripup_mgr.run(self.nets)
        logger.info(f"布线结束: {self._solution}")
        return self._solution

    @property
    def solution(self) -> Optional[RoutingSolution]:
        return self._solution

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------

    def visualize(
        self,
        save_dir: str = "results",
        prefix: str = "",
        layers: Optional[List[str]] = None,
    ):
        """
        输出各层布线结果 SVG。

        参数:
            save_dir: 保存目录
            prefix:   文件名前缀（空字符串=无前缀）
            layers:   要可视化的层列表（None=所有层）
        """
        if self._solution is None:
            logger.warning("尚未运行布线，无结果可视化")
            return

        from maze_router.visualizer import Visualizer
        viz = Visualizer(self.grid, self._solution)
        viz.save_svgs(save_dir=save_dir, prefix=prefix, layers=layers)
