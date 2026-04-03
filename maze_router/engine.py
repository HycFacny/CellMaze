"""
MazeRouterEngine: 顶层调度类

负责：
  1. 接收外部原始输入（GridGraph, nets, 原始参数字典/列表/字符串）
  2. 内部转化为 ConstraintManager、CostManager、RipupStrategy、RipupManager
  3. 调用 RipupManager.run() 执行布线
  4. 可选地调用 Visualizer 输出 SVG
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Set, Tuple

from maze_router.data.net import Net, Node, PinSpec, RoutingSolution
from maze_router.data.grid import GridGraph
from maze_router.constraint_manager import ConstraintManager
from maze_router.cost_manager import CostManager
from maze_router.constraints.space_constraint import SpaceConstraint
from maze_router.constraints.min_area_constraint import MinAreaConstraint
from maze_router.constraints.active_occupancy_constraint import ActiveOccupancyConstraint
from maze_router.costs.corner_cost import CornerCost
from maze_router.costs.space_cost import SpaceCost, SpaceCostRule, SpaceType
from maze_router.ripup_strategy import (
    DefaultRipupStrategy, CongestionAwareRipupStrategy,
)
from maze_router.ripup_manager import RipupManager

logger = logging.getLogger(__name__)


class MazeRouterEngine:
    """
    顶层路由调度引擎。

    所有参数均为原始 Python 类型（dict / list / str / float），内部自动转换为
    对应的 ConstraintManager / CostManager / RipupStrategy 等内部对象。

    使用方法：
        engine = MazeRouterEngine(
            grid=grid,
            nets=nets,
            space_constr={"M0": 1, "M1": 1, "M2": 1},
            corner_l_costs={"M0": 5.0, "M1": 5.0},
            corner_t_costs={},              # 禁用 T 型折角
            strategy="congestion_aware",
            max_iterations=80,
        )
        solution = engine.run()
        engine.visualize(save_dir="results/")

    参数
    ----
    grid : GridGraph
        布线网格。

    nets : List[Net]
        需要布线的线网列表。Net 是数据类，terminal 坐标由外部传入。

    space_constr : Dict[str, int], optional
        硬间距约束，格式 {layer: min_space}，如 {"M0": 1, "M1": 1}。
        min_space=1 表示不同线网间 Chebyshev 距离必须 > 1。

    corner_l_costs : Dict, optional
        L 型折角代价配置，传入 CornerCost 的 l_costs 参数：
          None  → 使用默认值（每层 5.0）
          {}    → 禁用（所有层 0.0）
          {"M0": 3.0, "M1": 3.0, ("M0",1,3): 10.0} → 自定义（支持精细键）

    corner_t_costs : Dict, optional
        T 型折角代价配置，传入 CornerCost 的 t_costs 参数：
          None  → 使用默认值（每层 10.0）
          {}    → 禁用
          {"M0": 1.0} → 自定义

    space_cost_rules : List[Tuple[str, str, int, float]], optional
        软间距代价规则列表，每项为 (layer, space_type, soft_space, penalty)：
          layer      : 层名，如 "M0"
          space_type : "S2S" | "T2S" | "T2T"
          soft_space : 软间距触发阈值（Chebyshev 距离）
          penalty    : 进入软间距区域的额外代价

    cable_locs : Dict[str, Set[Node]], optional
        M0 可走节点集合，格式 {net_name: {node, ...}}。

    pin_layers : Dict[str, str], optional
        Pin 引出层规格，格式 {net_name: layer}，layer 可为 "M1"/"M2"/"Any"。

    strategy : str, optional
        拆线重布策略类型：
          "default"          → DefaultRipupStrategy
          "congestion_aware" → CongestionAwareRipupStrategy（默认）

    max_iterations : int
        最大迭代轮次（默认 20）。

    base_penalty : float
        拥塞感知策略的基础惩罚系数（默认 0.1）。

    penalty_growth : float
        拥塞惩罚指数增长系数（默认 0.5）。

    congestion_weight : float
        拥塞代价权重（默认 1.0）。

    min_area : Dict[str, int], optional
        最小面积约束，格式 {layer: min_nodes}（soft rule，值被 clamp 到 [1, 4]）。
        未指定层使用默认值 2。

    net_active_must_occupy_num : Dict[Tuple[str, int, int], int], optional
        per-net, per-SD-terminal 的 M1 active 覆盖要求，格式：
          {(net_name, j, i): N}
          net_name : 线网名称
          j        : SD 列索引（版图 x 坐标）
          i        : 行类型（0=NMOS, 1=PMOS, 2=PMOS多行, 3=NMOS多行, …）
          N        : M1 层在列 j、行类型 i 的 y_range 内必须覆盖的最少节点数
                     （通常 N = active_size + 1）
        需与 row_type_y_ranges 配合使用。
        Hard Rule：BFS 扩展后仍不满足时布线失败，触发拆线重布。
        虚拟层节点（grid.virtual_node_layers）不计入 M1 节点数。

    row_type_y_ranges : Dict[int, Tuple[int, int]], optional
        行类型 i 对应的网格 y 坐标范围（含端点），格式 {i: (y_min, y_max)}。
        单行标准单元（9行）默认值：{0: (0, 3), 1: (5, 8)}。
        多行需手动指定更多行类型。
        当 net_active_must_occupy_num 为 None 时忽略此参数。
    """

    def __init__(
        self,
        grid: GridGraph,
        nets: List[Net],
        # ── 硬间距约束 ──────────────────────────────────────
        space_constr: Optional[Dict[str, int]] = None,
        # ── 折角代价（原始字典，内部构建 CornerCost）────────
        corner_l_costs: Optional[Dict] = None,
        corner_t_costs: Optional[Dict] = None,
        # ── 软间距代价（原始元组列表，内部构建 SpaceCostRule）
        space_cost_rules: Optional[List[Tuple[str, str, int, float]]] = None,
        # ── 最小面积约束 ─────────────────────────────────────
        min_area: Optional[Dict[str, int]] = None,
        # ── Active 占用约束（per-net M1 列最小节点数）────────
        net_active_must_occupy_num: Optional[Dict[Tuple[str, int, int], int]] = None,
        row_type_y_ranges: Optional[Dict[int, Tuple[int, int]]] = None,
        # ── 线网绑定 ─────────────────────────────────────────
        cable_locs: Optional[Dict[str, Set[Node]]] = None,
        pin_layers: Optional[Dict[str, str]] = None,
        # ── 策略原始参数（内部构建 RipupStrategy）────────────
        strategy: str = "congestion_aware",
        max_iterations: int = 20,
        base_penalty: float = 0.1,
        penalty_growth: float = 0.5,
        # ── 拥塞代价权重 ─────────────────────────────────────
        congestion_weight: float = 1.0,
    ):
        self.grid = grid
        self.nets = nets

        # ── 1. 线网绑定：cable_locs / pin_layers ──────────────────────────
        if cable_locs:
            for net in self.nets:
                if net.name in cable_locs:
                    net.cable_locs = cable_locs[net.name]

        if pin_layers:
            for net in self.nets:
                if net.name in pin_layers:
                    net.pin_spec = PinSpec(layer=pin_layers[net.name])

        # ── 线网 active 占用数绑定（保存到 Net 对象供外部查询）──────────────
        if net_active_must_occupy_num:
            for net in self.nets:
                net_rules = {
                    (j, i): n
                    for (name, j, i), n in net_active_must_occupy_num.items()
                    if name == net.name
                }
                if net_rules:
                    net.active_must_occupy_num = net_rules

        # ── 2. 约束管理器 ─────────────────────────────────────────────────
        constraints = []
        if space_constr:
            constraints.append(SpaceConstraint(rules=space_constr))
        if min_area is not None:
            constraints.append(MinAreaConstraint(rules=min_area))
        if net_active_must_occupy_num:
            # 默认行类型 y_range（9行单行标准单元）
            _y_ranges = row_type_y_ranges if row_type_y_ranges is not None else {
                0: (0, 3),   # NMOS
                1: (5, 8),   # PMOS
            }
            constraints.append(ActiveOccupancyConstraint(
                rules=net_active_must_occupy_num,
                row_type_y_ranges=_y_ranges,
            ))
        self.constraint_mgr = ConstraintManager(constraints)

        # ── 3. 代价管理器 ─────────────────────────────────────────────────
        # 3a. 折角代价：原始字典 → CornerCost
        corner_cost = CornerCost(l_costs=corner_l_costs, t_costs=corner_t_costs)

        # 3b. 软间距代价：原始元组 → SpaceCostRule
        costs = [corner_cost]
        if space_cost_rules:
            parsed_rules = []
            for layer, type_str, soft_space, penalty in space_cost_rules:
                try:
                    stype = SpaceType[type_str.upper()]
                except KeyError:
                    raise ValueError(
                        f"未知 space_type '{type_str}'，合法值为 S2S / T2S / T2T"
                    )
                parsed_rules.append(
                    SpaceCostRule(
                        layer=layer,
                        space_type=stype,
                        soft_space=soft_space,
                        penalty=penalty,
                    )
                )
            costs.append(SpaceCost(parsed_rules))

        self.cost_mgr = CostManager(
            grid=grid,
            costs=costs,
            congestion_weight=congestion_weight,
        )

        # ── 4. 拆线重布策略：字符串 → RipupStrategy ──────────────────────
        strategy_key = strategy.lower().replace("-", "_")
        if strategy_key == "congestion_aware":
            self.strategy = CongestionAwareRipupStrategy(
                max_iterations=max_iterations,
                base_penalty=base_penalty,
                penalty_growth=penalty_growth,
            )
        elif strategy_key == "default":
            self.strategy = DefaultRipupStrategy(max_iterations=max_iterations)
        else:
            raise ValueError(
                f"未知策略 '{strategy}'，合法值为 'default' / 'congestion_aware'"
            )

        # ── 5. RipupManager ───────────────────────────────────────────────
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
