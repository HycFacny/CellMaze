"""
标准单元布线测试 — AND2_X1_ML (YMTC CJ3 工艺)

AND2 = NAND2(A,B) + INV(net2) → Y

电路结构（来自 AND2_X1_ML.xlsx）：
  T0 (x=0-1-2): PMOS S=VDD/D=net2 G=A;  NMOS S=net2/D=net0 G=A
  T1 (x=2-3-4): PMOS S=net2/D=VDD  G=B;  NMOS S=net0/D=VSS  G=B
  T2 (x=4-5-6): DUMMY (gate=None, S=D=VDD/VSS)
  T3 (x=6-7-8): PMOS S=VDD/D=Y     G=net2; NMOS S=VSS/D=Y   G=net2 [NMOS 高 4]

工艺规则 (YMTC CJ3)：
  1. S/D 列 (0,2,4,6,8) 在有源区（PMOS y=1..5，NMOS y=11..16）M0 无任何竖横连线；
     通道区（y=6..10）可参与 M0 横向连线（非 active，无扩散阻断）
  2. S/D M0↔M1 via 仅在有源区边界：PMOS→y=1，NMOS→y=16(高6)或y=14(高4)
  3. Gate M0↔M1 via 仅在非有源区 y=6..10
  4. x=5 (Dummy gate)：无任何 M0 边，无 via
  5. Active 区域在 M1 上必须全部相连（per-SD-terminal 覆盖约束）
  6. VDD 由 M1 第 0 行连接，VSS 由 M1 最后一行（y=17）连接，不能连到 M2
  7. VDD/VSS 在 M1 只能纵向连接到 power rail（M1 靠近轨道的竖向边代价=0）
  8. M2 仅 y=7 和 y=9 横向布线
  9. net_A, net_B, net_Y 需要在 M1 层引出 pin
  10. 通道区（y=6..10）相邻列间（Dummy x=5 两侧除外）均有 M0 横向边，
      代价略低于 M1 横向边（M0=0.9/列，M1=1.0/列），供 Gate net 在 M0 通道层走线

网格规格：9 列(x=0..8) × 18 行(y=0..17) × 3 层(M0,M1,M2)

  y=0 :  VDD rail (M1 only)
  y=1..5:  PMOS 有源区
  y=6..10: 布线通道（Gate via 区）
  y=11..14: NMOS 有源区（右组 x=6,8 共4行）
  y=11..16: NMOS 有源区（左组 x=0,2,4 共6行）
  y=17:  VSS rail (M1 only)
"""

from __future__ import annotations
import os
from typing import Optional
import pytest

from maze_router.data.net import Net, Node, PinSpec
from maze_router.data.grid import GridGraph
from maze_router.constraint_manager import ConstraintManager
from maze_router.cost_manager import CostManager
from maze_router.constraints.space_constraint import SpaceConstraint
from maze_router.constraints.active_occupancy_constraint import ActiveOccupancyConstraint
from maze_router.constraints.path_constraint import PathConstraint
from maze_router.costs.corner_cost import CornerCost
from maze_router.ripup_strategy import CongestionAwareRipupStrategy
from maze_router.ripup_manager import RipupManager
from maze_router import MazeRouterEngine


# ======================================================================
# 常量（从 AND2_X1_ML.xlsx 解析）
# ======================================================================

COLS = 17        # x: 0..16
ROWS = 19       # y: 0..18

SD_COLS         = {0, 2, 4, 6, 8, 10, 12, 14, 16}    # S/D 扩散列（偶数）
GATE_WHOLE_COLS = {1, 3, 7, 13, 15}            # Gate 完整列（x=5 只存在于PMOS, x=9只存在于NMOS, x=11全BREAK）
GATE_PMOS_COLS  = {5: (1, 12)}                       # Gate PMOS列（x=5, NMOS 是BREAK, 中间部分可用于给PMOS GATE 布线）
GATE_NMOS_COLS  = {9: (7, 17)}                       # Gate NMOS列（x=9, PMOS 是BREAK, 中间部分可用于给NMOS GATE 布线）
GATE_COLS       = {1, 3, 5, 7, 9, 13, 15}

# 有源区 y 范围
PMOS_ACTIVE_ROWS = set(range(1, 7))     # y=1..7
NMOS_ACTIVE_ROWS = set(range(12, 18))   # y=12..18（包含短列）

# Gate M0↔M1 via 只允许的行（非有源区通道）
GATE_VIA_ROWS = set(range(7, 12))   # y=7..11

# 电源轨
VDD_RAIL_Y = 0    # M1 y=0 = VDD power rail
VSS_RAIL_Y = 18   # M1 y=8 = VSS power rail

# M2 只在 y=7 和 y=9，仅横向
M2_ROWS = {8, 10}

# ── 行类型（用于 ActiveOccupancyConstraint）────────────────────────────
PMOS_H6_ROW = 0   # PMOS 高度 6，y=1..6   (x=0,2,4,6,8)
PMOS_H5_ROW = 1   # PMOS 高度 5，y=2..6   (x=12,14,16)
NMOS_H6_ROW = 2   # NMOS 高度 6，y=12..17（x=0,2,4,6,8,10）
NMOS_H4_ROW = 3   # NMOS 高度 4，y=12..15（x=12,14,16）

ROW_TYPE_Y_RANGES = {
    PMOS_H6_ROW: (1, 6),
    PMOS_H5_ROW: (2, 6),
    NMOS_H6_ROW: (12, 17),
    NMOS_H4_ROW: (12, 15),
}

# 每个 SD 列对应的 NMOS/PMOS 行类型
PMOS_SD_COL_NMOS_TYPE = {
    0: PMOS_H6_ROW,
    2: PMOS_H6_ROW,
    4: PMOS_H6_ROW,
    6: PMOS_H6_ROW,
    8: PMOS_H6_ROW,
    12: PMOS_H5_ROW,
    14: PMOS_H5_ROW,
    16: PMOS_H5_ROW
}
NMOS_SD_COL_NMOS_TYPE = {
    0: NMOS_H6_ROW,
    2: NMOS_H6_ROW,
    4: NMOS_H6_ROW,
    6: NMOS_H6_ROW,
    8: NMOS_H6_ROW,
    10: NMOS_H6_ROW,
    12: NMOS_H4_ROW,
    14: NMOS_H4_ROW,
    16: NMOS_H4_ROW
}

# S/D via 边界 y（PMOS=有源区顶端，NMOS=有源区底端）
SD_PMOS_VIA_Y = {0: 1, 2: 1, 4: 1, 6: 1, 8: 1, 12: 2, 14: 2, 16: 2}
SD_NMOS_VIA_Y = {0: 17, 2: 17, 4: 17, 6: 17, 8: 17, 10: 17, 12: 15, 14: 15, 16: 15}

EXT_LAYER = "EXT"

# M0 通道横向边代价（略低于 M1 横向边 1.0，鼓励 gate net 在通道区优先走 M0）
M0_H_UNIT_COST = 0   # 每列单位代价；实际边代价 = 列距 × 0.0

# M0 cable_locs 集合（用于各线网的 M0 节点约束）
GATE_CABLE_LOCS = frozenset(("M0", x, y) for x in GATE_COLS for y in range(1, 18))
SD_CABLE_LOCS   = frozenset(("M0", x, y) for x in SD_COLS   for y in range(1, 18))


# ======================================================================
# Active 覆盖规格（来自 xlsx，net_name/col_x/row_type → M1 最少节点数）
# ======================================================================

MUX2_ACTIVE_RULES = {}


# ======================================================================
# EXT 虚拟起点辅助函数（YMTC CJ3：固定单一边界起点）
# ======================================================================

def _ext_node(x: int, row_type: int) -> Node:
    return (EXT_LAYER, x, row_type)


def _add_ext_to_grid(grid: GridGraph, x: int, row_type: Optional[int] = None) -> Node:
    """
    添加 EXT 虚拟起点，连接到 M0 唯一可行边界点（零代价边）。

    YMTC CJ3：起点只和有源区边界点相连（唯一一个）
      PMOS → y_min（有源区最顶端，靠近 VDD rail）
      NMOS → y_max（有源区最底端，靠近 VSS rail）
    """
    if row_type is None:
        return
    y_min, y_max = ROW_TYPE_Y_RANGES[row_type]
    start_y = y_min if row_type in [PMOS_H5_ROW, PMOS_H6_ROW] else y_max

    vnode = _ext_node(x, row_type)
    grid.add_node(vnode)
    m0 = ("M0", x, start_y)
    if grid.is_valid_node(m0):
        grid.add_edge(vnode, m0, cost=0.0)
    return vnode


# ======================================================================
# 网格构建
# ======================================================================

def build_mux2_grid() -> GridGraph:
    """构建 MUX2_X1_ML 三层布线网格（9×18）。"""
    g = GridGraph()

    # ── M0 节点（y=1..17，y=0/y=18 无 M0）─────────────────────────────
    for x in range(COLS):
        for y in range(1, 18):
            g.add_node(("M0", x, y))

    # M0 竖向边：仅 Gate 列，y=1..16 (↔ y+1)，不延伸到 y=18（不存在的节点）
    for x in GATE_WHOLE_COLS:
        for y in range(1, 17):
            g.add_edge(("M0", x, y), ("M0", x, y + 1), cost=0.0)
    for x, (y1, y2) in (GATE_PMOS_COLS | GATE_NMOS_COLS).items():
        for y in range(y1, y2):
            g.add_edge(("M0", x, y), ("M0", x, y + 1), cost=0.0)

    # M0 横向边：仅在通道区（y=7..11），所有相邻列 x↔x+1 全连（含 Dummy x=5）
    # 代价 = M0_H_UNIT_COST（0.9），略低于 M1 横向代价（1.0）
    # 有源区（y=1..5 PMOS，y=11..16 NMOS）无任何 M0 横向边
    for y in GATE_VIA_ROWS:
        for x in range(COLS - 1):
            g.add_edge(("M0", x, y), ("M0", x + 1, y), cost=M0_H_UNIT_COST)
    # 注：S/D 列和 Dummy 列无竖向 M0 边；通道区横向边三类列均可参与

    # ── M1：完整矩形 y=0..17 ─────────────────────────────────────────
    for x in range(COLS):
        for y in range(ROWS):
            g.add_node(("M1", x, y))

    # M1 竖向边：靠近 power rail 的边代价 = 0，其余 = 1
    for x in range(COLS):
        for y in range(ROWS - 1):
            cost = 0.0 if (y == 0 or y == ROWS - 2) else 1.0
            g.add_edge(("M1", x, y), ("M1", x, y + 1), cost=cost)

    # M1 横向边：power rail 行（y=0=VDD, y=18=VSS）cost=0，其余 cost=1
    # cost=0 引导 VDD/VSS 沿 rail 行横向连接，而非占用 active 区中间 M1 列
    for y in range(ROWS):
        for x in range(COLS - 1):
            cost = 0.0 if (y == 0 or y == ROWS - 1) else 1.0
            g.add_edge(("M1", x, y), ("M1", x + 1, y), cost=cost)

    # ── M2：仅 y=8 和 y=10，仅横向 ──────────────────────────────────
    for x in range(COLS):
        for y in M2_ROWS:
            g.add_node(("M2", x, y))
    for y in M2_ROWS:
        for x in range(COLS - 1):
            g.add_edge(("M2", x, y), ("M2", x + 1, y), cost=1.0)

    # ── Via M0↔M1（受限位置）──────────────────────────────────────
    # Gate 列：仅 y=7..11
    for x in GATE_COLS:
        for y in GATE_VIA_ROWS:
            g.add_edge(("M0", x, y), ("M1", x, y), cost=2.0)

    # S/D 列：仅 PMOS 边界(y=1) 和 NMOS 边界(SD_NMOS_VIA_Y[x])
    for x in SD_COLS:
        # PMOS 边界
        pmos_y = SD_PMOS_VIA_Y.get(x)
        if pmos_y is not None:
            g.add_edge(("M0", x, pmos_y), ("M1", x, pmos_y), cost=2.0)
        # NMOS 边界（不与 PMOS 重叠）
        nmos_y = SD_NMOS_VIA_Y.get(x)
        if nmos_y is not None:
            g.add_edge(("M0", x, nmos_y), ("M1", x, nmos_y), cost=2.0)

    # Dummy 列（x=5）：无任何 via

    # ── Via M1↔M2（仅 M2 存在的行 y=7,9）──────────────────────────
    for x in range(COLS):
        for y in M2_ROWS:
            g.add_edge(("M1", x, y), ("M2", x, y), cost=2.0)

    # 注册 EXT 为虚拟层（不参与 space 约束）
    g.register_virtual_layer(EXT_LAYER)

    return g


# ======================================================================
# 线网构建
# ======================================================================

def build_and2_nets(grid: GridGraph):
    """
    构建 MUX2_X1_ML 的 10 个布线线网，向 grid 注入 EXT 虚拟起点。

    线网：
      VDD   — 2 个 PMOS SD (x=4,14) + 2 个 VDD rail 锚点
      VSS   — 3 个 NMOS SD (x=4,10,14) + 3 个 VSS rail 锚点
      net7  - PMOS SD (x=0,8,12) NMOS SD (x=0,12)
      net11 - PMOS SD (x=2,6)
      net10 - NMOS SD (x=2,8)
      net0  - NMOS SD (x=6) GATE 共栅(x=3,15, PMOS+NMOS)
      D0    - GATE 共栅(x=1, PMOS+NMOS), PinSpec(M1)
      D1    - GATE 共栅(x=7, PMOS+NMOS), PinSpec(M1)
      S     - GATE (x=5, PMOS), (x=9, NMOS), 共栅(x=13, PMOS+NMOS), PinSpec(M1)
      Q     - PMOS SD (x=16), NMOS SD (x=16), PinSpec(M1)

    返回: (nets, MUX2_ACTIVE_RULES)
    """
    # PMOS EXT 起点（连到 M0(x, y=1)）
    ext_p = {x: _add_ext_to_grid(grid, x, PMOS_SD_COL_NMOS_TYPE.get(x)) for x in SD_COLS}

    # NMOS EXT 起点（按列对应行类型）
    ext_n = {x: _add_ext_to_grid(grid, x, NMOS_SD_COL_NMOS_TYPE.get(x)) for x in SD_COLS}

    nets = [
        # VDD：3 个 PMOS SD + 每列 VDD rail 锚点（鼓励垂直连到轨道），优先最先布
        Net("VDD", [
            ext_p[4], ext_p[14],
            ("M1", 4,  VDD_RAIL_Y),
            ("M1", 14, VDD_RAIL_Y),
        ], cable_locs=set(SD_CABLE_LOCS), priority=-10),

        # VSS：2 个 NMOS SD + 每列 VSS rail 锚点，优先最先布
        Net("VSS", [
            ext_n[4], ext_n[10], ext_n[14],
            ("M1", 4,  VSS_RAIL_Y),
            ("M1", 10, VSS_RAIL_Y),
            ("M1", 14, VSS_RAIL_Y),
        ], cable_locs=set(SD_CABLE_LOCS), priority=-10),

        # net7：SD net，cable_locs 用 SD_CABLE_LOCS
        Net("net7", [
            ext_p[0], ext_p[8], ext_p[12],
            ext_n[0], ext_n[12],
        ], cable_locs=set(SD_CABLE_LOCS)),

        # net11：SD net，cable_locs 用 SD_CABLE_LOCS
        Net("net11", [
            ext_p[2], ext_p[6],
        ], cable_locs=set(SD_CABLE_LOCS)),

        # net10：SD net，cable_locs 用 SD_CABLE_LOCS
        Net("net10", [
            ext_n[2], ext_n[8],
        ], cable_locs=set(SD_CABLE_LOCS)),

        # net0：混合 net（SD EXT + Gate 端点），cable_locs=None 不限制
        Net("net0", [
            ext_n[6],
            ("M0", 3,  1), ("M0", 3,  17),
            ("M0", 15, 2), ("M0", 15, 15)
        ], cable_locs=None),

        # D0
        Net("D0", [
            ("M0", 1,  1), ("M0", 1, 17),
        ], cable_locs=set(GATE_CABLE_LOCS),
           pin_spec=PinSpec(layer="M1")),

        # D1
        Net("D1", [
            ("M0", 7,  1), ("M0", 7, 17),
        ], cable_locs=set(GATE_CABLE_LOCS),
           pin_spec=PinSpec(layer="M1")),
        
        # S
        Net("S", [
            ("M0", 5,   1), ("M0", 5,   6),
            ("M0", 9,  12), ("M0", 9,  17),
            ("M0", 13,  2), ("M0", 13, 15),
        ], cable_locs=set(GATE_CABLE_LOCS),
           pin_spec=PinSpec(layer="M1")),
        
        # Q：SD net，cable_locs 用 SD_CABLE_LOCS
        Net("Q", [
            ext_p[16], ext_n[16]
        ], cable_locs=set(SD_CABLE_LOCS),
           pin_spec=PinSpec(layer="M1"))
    ]
    
    return nets, MUX2_ACTIVE_RULES


# ======================================================================
# 固定边
# ======================================================================



# ======================================================================
# RipupManager 构建辅助
# ======================================================================

def make_mgr(
    grid: GridGraph,
    space: int = 1,
    max_iter: int = 60,
    active_rules=None,
):
    constraints = [SpaceConstraint({"M0": space, "M1": space, "M2": space})]
    if active_rules:
        constraints.append(ActiveOccupancyConstraint(
            rules=active_rules,
            row_type_y_ranges=ROW_TYPE_Y_RANGES,
        ))
    constraint_mgr = ConstraintManager(constraints)
    cost_mgr = CostManager(
        grid=grid,
        costs=[CornerCost(
            l_costs={"M0": 1.0, "M1": 5.0, "M2": 0.0},
            t_costs={"M0": 2.0, "M1": 10.0, "M2": 0.0},
        )],
    )
    strategy = CongestionAwareRipupStrategy(
        max_iterations=max_iter, base_penalty=0.1, penalty_growth=0.3
    )
    return RipupManager(grid, constraint_mgr, cost_mgr, strategy)


# ======================================================================
# Class 5: Engine 端到端测试
# ======================================================================

class TestMUX2Engine:
    """通过顶层 MazeRouterEngine 进行集成测试"""

    @pytest.fixture
    def grid(self):
        return build_mux2_grid()

    def test_engine_space1_full(self, grid):
        """Engine 端到端：MUX2 全 10 线网，space=0，至少 10/10 布通"""
        nets, active_rules = build_and2_nets(grid)

        # 工艺规则：VDD/VSS 在 M1 上只能纵向连接到 power rail，禁止横向走线
        # VDD 禁止在 y=1..17（active 区及通道区）横向走 M1
        vdd_h_forbid = [
            (("M1", x, y), ("M1", x + 1, y))
            for y in range(1, ROWS - 1)
            for x in range(COLS - 1)
        ]
        # VSS 禁止在 y=0..17（除 y=18 power rail 外）横向走 M1
        vss_h_forbid = [
            (("M1", x, y), ("M1", x + 1, y))
            for y in range(ROWS - 1)
            for x in range(COLS - 1)
        ]
        
        other_power_rail_forbid = [
            (("M1", x, 0), ("M1", x + 1, 0))
            for x in range(COLS - 1)
        ] + [
            (("M1", x, 18), ("M1", x + 1, 18))
            for x in range(COLS - 1)
        ]

        must_forbid_edges = {
            net.name: other_power_rail_forbid
            for net in nets if net.name not in ["VSS", "VDD"]
        } | {
            "VSS": vss_h_forbid, "VDD": vdd_h_forbid
        }

        engine = MazeRouterEngine(
            grid=grid,
            nets=nets,
            space_constr={"M0": 0, "M1": 0, "M2": 0},
            corner_l_costs={"M0": 2.0, "M1": 10.0, "M2": 0.0},
            corner_t_costs={"M0": 4.0, "M1": 20.0, "M2": 0.0},
            strategy="congestion_aware",
            max_iterations=80,
            net_active_must_occupy_num=active_rules,
            row_type_y_ranges=ROW_TYPE_Y_RANGES,
            must_forbid_edges=must_forbid_edges,
        )
        solution = engine.run()
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "results", "stdcell_mux2",
        )
        from maze_router.visualizer import Visualizer
        viz = Visualizer(grid, solution)
        viz.save_svgs(save_dir=save_dir)

        for name in [net.name for net in nets]:
            assert solution.results[name].success, f"共栅 {name} 必须布通"

        assert solution.routed_count == 10, \
            f"AND2 布通实际 {solution.routed_count}/10"
        assert solution.routed_count >= 4, \
            f"Engine space=1 应至少 10/10 布通，实际 {solution.routed_count}/10"

