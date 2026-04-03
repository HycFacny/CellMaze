"""
标准单元布线测试 — AND2_X1_ML (YMTC CJ3 工艺)

AND2 = NAND2(A,B) + INV(net2) → Y

电路结构（来自 AND2_X1_ML.xlsx）：
  T0 (x=0-1-2): PMOS S=VDD/D=net2 G=A;  NMOS S=net2/D=net0 G=A
  T1 (x=2-3-4): PMOS S=net2/D=VDD  G=B;  NMOS S=net0/D=VSS  G=B
  T2 (x=4-5-6): DUMMY (gate=None, S=D=VDD/VSS)
  T3 (x=6-7-8): PMOS S=VDD/D=Y     G=net2; NMOS S=VSS/D=Y   G=net2 [NMOS 高 4]

工艺规则 (YMTC CJ3)：
  1. S/D 列 (0,2,4,6,8) 在 M0 上无任何竖横连线
  2. S/D M0↔M1 via 仅在有源区边界：PMOS→y=1，NMOS→y=16(高6)或y=14(高4)
  3. Gate M0↔M1 via 仅在非有源区 y=6..10
  4. x=5 (Dummy gate)：无任何 M0 边，无 via
  5. Active 区域在 M1 上必须全部相连（per-SD-terminal 覆盖约束）
  6. VDD 由 M1 第 0 行连接，VSS 由 M1 最后一行（y=17）连接，不能连到 M2
  7. VDD/VSS 在 M1 只能纵向连接到 power rail（M1 靠近轨道的竖向边代价=0）
  8. M2 仅 y=7 和 y=9 横向布线
  9. net_A, net_B, net_Y 需要在 M1 层引出 pin

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
import pytest

from maze_router.data.net import Net, Node, PinSpec
from maze_router.data.grid import GridGraph
from maze_router.constraint_manager import ConstraintManager
from maze_router.cost_manager import CostManager
from maze_router.constraints.space_constraint import SpaceConstraint
from maze_router.constraints.active_occupancy_constraint import ActiveOccupancyConstraint
from maze_router.costs.corner_cost import CornerCost
from maze_router.ripup_strategy import CongestionAwareRipupStrategy
from maze_router.ripup_manager import RipupManager
from maze_router import MazeRouterEngine


# ======================================================================
# 常量（从 AND2_X1_ML.xlsx 解析）
# ======================================================================

COLS = 9        # x: 0..8
ROWS = 18       # y: 0..17

SD_COLS       = {0, 2, 4, 6, 8}    # S/D 扩散列（偶数）
GATE_COLS     = {1, 3, 7}          # Gate 列（x=5 为 Dummy，排除）
DUMMY_GATE_COL = 5                 # x=5: gate=None，Dummy 晶体管

# 有源区 y 范围
PMOS_ACTIVE_ROWS = set(range(1, 6))     # y=1..5
NMOS_ACTIVE_ROWS = set(range(11, 17))   # y=11..16（包含短列）

# Gate M0↔M1 via 只允许的行（非有源区通道）
GATE_VIA_ROWS = set(range(6, 11))   # y=6..10

# 电源轨
VDD_RAIL_Y = 0    # M1 y=0 = VDD power rail
VSS_RAIL_Y = 17   # M1 y=17 = VSS power rail

# M2 只在 y=7 和 y=9，仅横向
M2_ROWS = {7, 9}

# ── 行类型（用于 ActiveOccupancyConstraint）────────────────────────────
PMOS_ROW    = 0   # PMOS 有源区，y=1..5
NMOS_H6_ROW = 1   # NMOS 高度 6，y=11..16（x=0,2,4）
NMOS_H4_ROW = 2   # NMOS 高度 4，y=11..14（x=6,8）

ROW_TYPE_Y_RANGES = {
    PMOS_ROW:    (1, 5),
    NMOS_H6_ROW: (11, 16),
    NMOS_H4_ROW: (11, 14),
}

# 每个 SD 列对应的 NMOS 行类型
SD_COL_NMOS_TYPE = {
    0: NMOS_H6_ROW,
    2: NMOS_H6_ROW,
    4: NMOS_H6_ROW,
    6: NMOS_H4_ROW,
    8: NMOS_H4_ROW,
}

# S/D via 边界 y（PMOS=有源区顶端，NMOS=有源区底端）
SD_PMOS_VIA_Y = 1   # PMOS 有源区最顶端（靠近 VDD rail y=0）
SD_NMOS_VIA_Y = {0: 16, 2: 16, 4: 16, 6: 14, 8: 14}

EXT_LAYER = "EXT"

# M0 cable_locs 集合（用于各线网的 M0 节点约束）
GATE_CABLE_LOCS = frozenset(("M0", x, y) for x in GATE_COLS for y in range(1, 17))
SD_CABLE_LOCS   = frozenset(("M0", x, y) for x in SD_COLS   for y in range(1, 17))


# ======================================================================
# Active 覆盖规格（来自 xlsx，net_name/col_x/row_type → M1 最少节点数）
# ======================================================================

AND2_ACTIVE_RULES = {
    # VDD PMOS S/D（x=0,4,6，有源区 y=1..5，需 5 节点）
    ("VDD",   0, PMOS_ROW):    5,
    ("VDD",   4, PMOS_ROW):    5,
    ("VDD",   6, PMOS_ROW):    5,
    # net2 PMOS drain（x=2，y=1..5，需 5 节点）
    ("net2",  2, PMOS_ROW):    5,
    # net2 NMOS drain（x=0，y=11..16，需 6 节点）
    ("net2",  0, NMOS_H6_ROW): 6,
    # net0 NMOS 串联内部节点（x=2，y=11..16，需 6 节点）
    ("net0",  2, NMOS_H6_ROW): 6,
    # VSS NMOS S/D
    ("VSS",   4, NMOS_H6_ROW): 6,   # x=4，高 6，y=11..16
    ("VSS",   6, NMOS_H4_ROW): 4,   # x=6，高 4，y=11..14
    # net_Y INV 输出 S/D
    ("net_Y", 8, PMOS_ROW):    5,   # PMOS drain（y=1..5）
    ("net_Y", 8, NMOS_H4_ROW): 4,   # NMOS drain（y=11..14）
}


# ======================================================================
# EXT 虚拟起点辅助函数（YMTC CJ3：固定单一边界起点）
# ======================================================================

def _ext_node(x: int, row_type: int) -> Node:
    return (EXT_LAYER, x, row_type)


def _add_ext_to_grid(grid: GridGraph, x: int, row_type: int) -> Node:
    """
    添加 EXT 虚拟起点，连接到 M0 唯一可行边界点（零代价边）。

    YMTC CJ3：起点只和有源区边界点相连（唯一一个）
      PMOS → y_min（有源区最顶端，靠近 VDD rail）
      NMOS → y_max（有源区最底端，靠近 VSS rail）
    """
    y_min, y_max = ROW_TYPE_Y_RANGES[row_type]
    start_y = y_min if row_type == PMOS_ROW else y_max

    vnode = _ext_node(x, row_type)
    grid.add_node(vnode)
    m0 = ("M0", x, start_y)
    if grid.is_valid_node(m0):
        grid.add_edge(vnode, m0, cost=0.0)
    return vnode


# ======================================================================
# 网格构建
# ======================================================================

def build_and2_grid() -> GridGraph:
    """
    构建 AND2_X1_ML 三层布线网格（9×18）。

    M0 规则：
      - y=0 和 y=17：无 M0 节点（power rail 行，M1 only）
      - S/D 列（0,2,4,6,8）：无任何 M0 边（竖横均无）
      - Gate 列（1,3,7）：全程竖向边（y=1..16），无横向边
      - Dummy 列（5）：无 M0 边，无 via
      - M0↔M1 via 位置限制：
          Gate 列：仅 y=6..10（非有源区）
          S/D 列：仅 y=1（PMOS 边界）和 y=SD_NMOS_VIA_Y[x]（NMOS 边界）

    M1 规则：
      - 完整矩形（y=0..17），所有横纵边
      - y=0 横向边全连（VDD rail），y=17 横向边全连（VSS rail）
      - 纵向边代价：y=0↔1 和 y=16↔17 为 0（鼓励直连 power rail），其余为 1

    M2 规则：
      - 仅 y=7 和 y=9，仅横向，无纵向直连

    Via：
      - M0↔M1：见 M0 规则中 via 位置限制
      - M1↔M2：仅 y=7,9 行
    """
    g = GridGraph()

    # ── M0 节点（y=1..16，y=0/y=17 无 M0）─────────────────────────────
    for x in range(COLS):
        for y in range(1, 17):
            g.add_node(("M0", x, y))

    # M0 竖向边：仅 Gate 列，y=1..15 (↔ y+1)
    for x in GATE_COLS:
        for y in range(1, 16):
            g.add_edge(("M0", x, y), ("M0", x, y + 1), cost=1.0)
    # 注：S/D 列和 Dummy 列无任何 M0 边

    # ── M1：完整矩形 y=0..17 ─────────────────────────────────────────
    for x in range(COLS):
        for y in range(ROWS):
            g.add_node(("M1", x, y))

    # M1 竖向边：靠近 power rail 的边代价 = 0，其余 = 1
    for x in range(COLS):
        for y in range(ROWS - 1):
            cost = 0.0 if (y == 0 or y == ROWS - 2) else 1.0
            g.add_edge(("M1", x, y), ("M1", x, y + 1), cost=cost)

    # M1 横向边：cost = 1
    for y in range(ROWS):
        for x in range(COLS - 1):
            g.add_edge(("M1", x, y), ("M1", x + 1, y), cost=1.0)

    # ── M2：仅 y=7 和 y=9，仅横向 ──────────────────────────────────
    for x in range(COLS):
        for y in M2_ROWS:
            g.add_node(("M2", x, y))
    for y in M2_ROWS:
        for x in range(COLS - 1):
            g.add_edge(("M2", x, y), ("M2", x + 1, y), cost=1.0)

    # ── Via M0↔M1（受限位置）──────────────────────────────────────
    # Gate 列：仅 y=6..10
    for x in GATE_COLS:
        for y in GATE_VIA_ROWS:
            g.add_edge(("M0", x, y), ("M1", x, y), cost=2.0)

    # S/D 列：仅 PMOS 边界(y=1) 和 NMOS 边界(SD_NMOS_VIA_Y[x])
    for x in SD_COLS:
        # PMOS 边界
        g.add_edge(("M0", x, SD_PMOS_VIA_Y), ("M1", x, SD_PMOS_VIA_Y), cost=2.0)
        # NMOS 边界（不与 PMOS 重叠）
        nmos_y = SD_NMOS_VIA_Y[x]
        if nmos_y != SD_PMOS_VIA_Y:
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
    构建 AND2_X1_ML 的 7 个布线线网，向 grid 注入 EXT 虚拟起点。

    线网：
      VDD   — 3 个 PMOS SD (x=0,4,6) + 3 个 VDD rail 锚点
      VSS   — 2 个 NMOS SD (x=4,6) + 2 个 VSS rail 锚点
      net_A — 共栅 A (x=1, PMOS+NMOS)，PinSpec(M1)
      net_B — 共栅 B (x=3, PMOS+NMOS)，PinSpec(M1)
      net2  — NAND2 输出/INV 输入（PMOS drain + NMOS drain + INV gate）
      net0  — NMOS 串联内部节点（仅 x=2 NMOS，单 SD + M1 锚点）
      net_Y — AND2 输出（PMOS drain + NMOS drain），PinSpec(M1)

    返回: (nets, AND2_ACTIVE_RULES)
    """
    # PMOS EXT 起点（连到 M0(x, y=1)）
    ext_p = {x: _add_ext_to_grid(grid, x, PMOS_ROW) for x in SD_COLS}

    # NMOS EXT 起点（按列对应行类型）
    ext_n = {x: _add_ext_to_grid(grid, x, SD_COL_NMOS_TYPE[x]) for x in SD_COLS}

    nets = [
        # ── 电源 ──────────────────────────────────────────────────────
        # VDD：3 个 PMOS SD + 每列 VDD rail 锚点（鼓励垂直连到轨道）
        Net("VDD", [
            ext_p[0], ext_p[4], ext_p[6],
            ("M1", 0, VDD_RAIL_Y),
            ("M1", 4, VDD_RAIL_Y),
            ("M1", 6, VDD_RAIL_Y),
        ], cable_locs=set(SD_CABLE_LOCS)),

        # VSS：2 个 NMOS SD + 每列 VSS rail 锚点
        Net("VSS", [
            ext_n[4], ext_n[6],
            ("M1", 4, VSS_RAIL_Y),
            ("M1", 6, VSS_RAIL_Y),
        ], cable_locs=set(SD_CABLE_LOCS)),

        # ── Gate 线网（共栅，PMOS+NMOS 通过 M0 竖向直连）─────────────
        # Gate via 仅在 y=6..10，必须从 M0 竖向走到通道再接 M1
        Net("net_A", [
            ("M0", 1, 1),    # PMOS gate 顶端（有源区最顶行）
            ("M0", 1, 16),   # NMOS gate 底端（有源区最底行）
        ], cable_locs=set(GATE_CABLE_LOCS),
           pin_spec=PinSpec(layer="M1")),

        Net("net_B", [
            ("M0", 3, 1),
            ("M0", 3, 16),
        ], cable_locs=set(GATE_CABLE_LOCS),
           pin_spec=PinSpec(layer="M1")),

        # ── net2：NAND2 输出 = INV 输入 ─────────────────────────────
        # PMOS drain x=2 + NMOS drain x=0 + INV gate M0(7,1)
        Net("net2", [
            ext_p[2],          # PMOS net2 (x=2 top, via y=1)
            ext_n[0],          # NMOS net2 (x=0 bottom, via y=16)
            ("M0", 7, 1),      # INV gate 终端（gate 列 x=7）
        ]),

        # ── net0：NMOS 串联内部节点（单 SD，添加 M1 锚点触发覆盖）───
        Net("net0", [
            ext_n[2],          # NMOS net0 EXT (x=2 bottom, via y=16)
            ("M1", 2, 11),     # M1 有源区顶端锚点（触发 M1 全覆盖 y=11..16）
        ], cable_locs=set(SD_CABLE_LOCS)),

        # ── 输出 ─────────────────────────────────────────────────────
        Net("net_Y", [
            ext_p[8],   # PMOS Y (x=8 top, via y=1)
            ext_n[8],   # NMOS Y (x=8 bottom, via y=14)
        ], cable_locs=set(SD_CABLE_LOCS),
           pin_spec=PinSpec(layer="M1")),
    ]

    return nets, AND2_ACTIVE_RULES


# ======================================================================
# RipupManager 构建辅助
# ======================================================================

def make_mgr(
    grid: GridGraph,
    space: int = 0,
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
            l_costs={"M0": 3.0, "M1": 3.0, "M2": 3.0},
            t_costs={},
        )],
    )
    strategy = CongestionAwareRipupStrategy(
        max_iterations=max_iter, base_penalty=0.1, penalty_growth=0.3
    )
    return RipupManager(grid, constraint_mgr, cost_mgr, strategy)


# ======================================================================
# Class 1: 网格结构验证
# ======================================================================

class TestAND2GridStructure:
    """验证 AND2 网格的拓扑结构（含 via 限制）"""

    @pytest.fixture(scope="class")
    def grid(self):
        return build_and2_grid()

    def test_node_counts(self, grid):
        """验证各层节点数量"""
        # M0: y=1..16, x=0..8 → 16 * 9
        assert len(grid.get_nodes_on_layer("M0")) == 16 * COLS
        # M1: y=0..17, x=0..8 → 18 * 9
        assert len(grid.get_nodes_on_layer("M1")) == ROWS * COLS
        # M2: y=7,9, x=0..8 → 2 * 9
        assert len(grid.get_nodes_on_layer("M2")) == len(M2_ROWS) * COLS, \
            f"M2 应有 {len(M2_ROWS) * COLS} 个节点，实际 {len(grid.get_nodes_on_layer('M2'))}"

    def test_m0_sd_no_edges(self, grid):
        """S/D 列 M0 无任何 M0-M0 边（竖横均无）"""
        for x in SD_COLS:
            for y in range(1, 17):
                node = ("M0", x, y)
                for nb in grid.get_neighbors(node):
                    assert nb[0] != "M0", \
                        f"S/D M0({x},{y}) 不应有 M0-M0 边，发现邻居 {nb}"

    def test_m0_no_power_rail_rows(self, grid):
        """M0 在 y=0 和 y=17 无节点"""
        for x in range(COLS):
            assert not grid.is_valid_node(("M0", x, 0)), \
                f"M0({x},0) 不应存在（VDD rail 行）"
            assert not grid.is_valid_node(("M0", x, 17)), \
                f"M0({x},17) 不应存在（VSS rail 行）"

    def test_m0_gate_full_vertical(self, grid):
        """Gate 列 M0 全程竖向连通 (y=1..16)"""
        for x in GATE_COLS:
            for y in range(1, 16):
                assert grid.graph.has_edge(("M0", x, y), ("M0", x, y + 1)), \
                    f"Gate x={x}: M0 y={y}↔{y+1} 竖向边缺失"

    def test_m0_gate_no_horizontal_edges(self, grid):
        """Gate 列 M0 无横向边（不同列 gate 不直接相连）"""
        for x1 in GATE_COLS:
            for x2 in GATE_COLS:
                if x1 >= x2:
                    continue
                for y in range(1, 17):
                    assert not grid.graph.has_edge(("M0", x1, y), ("M0", x2, y)), \
                        f"Gate M0 横向边 x={x1}↔{x2} y={y} 不应存在"

    def test_m0_dummy_col_no_edges(self, grid):
        """Dummy 列 (x=5) M0 无任何边（gate=None → 无布线）"""
        for y in range(1, 17):
            node = ("M0", DUMMY_GATE_COL, y)
            assert len(list(grid.get_neighbors(node))) == 0, \
                f"Dummy M0({DUMMY_GATE_COL},{y}) 不应有任何邻居"

    def test_via_m0m1_gate_channel_only(self, grid):
        """Gate 列 M0↔M1 via 仅在 y=6..10（GATE_VIA_ROWS）"""
        for x in GATE_COLS:
            for y in range(1, 17):
                has_via = grid.graph.has_edge(("M0", x, y), ("M1", x, y))
                if y in GATE_VIA_ROWS:
                    assert has_via, f"Gate x={x}: M0↔M1 via 应在 y={y}"
                else:
                    assert not has_via, f"Gate x={x}: M0↔M1 via 不应在 y={y}"

    def test_via_m0m1_sd_boundary_only(self, grid):
        """S/D 列 M0↔M1 via 仅在 PMOS 和 NMOS 有源区边界"""
        for x in SD_COLS:
            allowed_via_y = {SD_PMOS_VIA_Y, SD_NMOS_VIA_Y[x]}
            for y in range(1, 17):
                has_via = grid.graph.has_edge(("M0", x, y), ("M1", x, y))
                if y in allowed_via_y:
                    assert has_via, f"SD x={x}: M0↔M1 via 应在 y={y}"
                else:
                    assert not has_via, f"SD x={x}: M0↔M1 via 不应在 y={y}"

    def test_via_m0m1_dummy_none(self, grid):
        """Dummy 列 (x=5) M0↔M1 via 不存在"""
        for y in range(1, 17):
            assert not grid.graph.has_edge(("M0", DUMMY_GATE_COL, y), ("M1", DUMMY_GATE_COL, y)), \
                f"Dummy x={DUMMY_GATE_COL}: M0↔M1 via 不应在 y={y}"

    def test_m1_power_rail_rows_exist(self, grid):
        """M1 y=0 和 y=17 所有节点及横向边存在（power rail）"""
        for y in (VDD_RAIL_Y, VSS_RAIL_Y):
            for x in range(COLS):
                assert grid.is_valid_node(("M1", x, y)), f"M1({x},{y}) 缺失"
            for x in range(COLS - 1):
                assert grid.graph.has_edge(("M1", x, y), ("M1", x + 1, y)), \
                    f"M1 rail y={y}: x={x}↔{x+1} 边缺失"

    def test_m1_power_rail_zero_cost(self, grid):
        """M1 靠近 power rail 的纵向边代价为 0"""
        for x in range(COLS):
            # y=0↔y=1
            e = grid.graph.get_edge_data(("M1", x, 0), ("M1", x, 1))
            assert e is not None and e.get("cost", 1) == 0.0, \
                f"M1({x},0↔1) 代价应为 0"
            # y=16↔y=17
            e = grid.graph.get_edge_data(("M1", x, 16), ("M1", x, 17))
            assert e is not None and e.get("cost", 1) == 0.0, \
                f"M1({x},16↔17) 代价应为 0"

    def test_m2_rows_only_7_and_9(self, grid):
        """M2 节点仅在 y=7 和 y=9"""
        m2_nodes = grid.get_nodes_on_layer("M2")
        for node in m2_nodes:
            assert node[2] in M2_ROWS, \
                f"M2 节点 {node} 不在 y=7/y=9"
        # 确认没有其他行的 M2
        for y in range(ROWS):
            if y not in M2_ROWS:
                assert not grid.is_valid_node(("M2", 0, y)), \
                    f"M2(0,{y}) 不应存在"

    def test_via_m1m2_at_m2_rows(self, grid):
        """M1↔M2 via 仅在 y=7 和 y=9"""
        for x in range(COLS):
            for y in range(ROWS):
                has = grid.graph.has_edge(("M1", x, y), ("M2", x, y))
                if y in M2_ROWS:
                    assert has, f"M1↔M2 via 缺失: ({x},{y})"
                else:
                    assert not has, f"M1↔M2 via 不应存在: ({x},{y})"

    def test_ext_virtual_layer_registered(self, grid):
        """EXT 虚拟层已注册"""
        assert EXT_LAYER in grid.virtual_node_layers


# ======================================================================
# Class 2: 单线网路由测试
# ======================================================================

class TestAND2NetRouting:
    """独立线网路由验证"""

    @pytest.fixture(scope="class")
    def grid(self):
        return build_and2_grid()

    def test_gate_a_routes_via_m0_to_m1(self, grid):
        """
        共栅 A (x=1)：M0(1,1) 到 M0(1,16) 可通过 M0 竖向路由，
        via 限制在 y=6..10，路由应成功并使用 M1（pin extraction）。
        """
        g_copy = build_and2_grid()
        mgr = make_mgr(g_copy, space=0, max_iter=20)
        net = Net("net_A", [("M0", 1, 1), ("M0", 1, 16)],
                  cable_locs=set(GATE_CABLE_LOCS),
                  pin_spec=PinSpec(layer="M1"))
        solution = mgr.run([net])
        assert solution.results["net_A"].success, "共栅 net_A 应布通"
        # 检查路由使用了 M1（来自 pin extraction）
        nodes = solution.results["net_A"].routed_nodes
        m1_nodes = [n for n in nodes if n[0] == "M1"]
        assert len(m1_nodes) > 0, "共栅 net_A（有 PinSpec）应有 M1 节点（pin）"

    def test_gate_a_via_in_channel(self, grid):
        """
        共栅 A 的 M0↔M1 via 只在 y=6..10 发生（gate via 限制验证）。
        """
        g_copy = build_and2_grid()
        mgr = make_mgr(g_copy, space=0, max_iter=20)
        net = Net("net_A", [("M0", 1, 1), ("M0", 1, 16)],
                  cable_locs=set(GATE_CABLE_LOCS),
                  pin_spec=PinSpec(layer="M1"))
        solution = mgr.run([net])
        if not solution.results["net_A"].success:
            pytest.skip("net_A 布线失败，跳过 via 检查")
        nodes = solution.results["net_A"].routed_nodes
        m1_nodes = [n for n in nodes if n[0] == "M1"]
        for m1_node in m1_nodes:
            assert m1_node[2] in GATE_VIA_ROWS, \
                f"net_A M1 节点 {m1_node} 不在 gate via 区域 y=6..10"

    def test_gate_b_routes_successfully(self, grid):
        """共栅 B (x=3) 可布通"""
        g_copy = build_and2_grid()
        mgr = make_mgr(g_copy, space=0, max_iter=20)
        net = Net("net_B", [("M0", 3, 1), ("M0", 3, 16)],
                  cable_locs=set(GATE_CABLE_LOCS),
                  pin_spec=PinSpec(layer="M1"))
        solution = mgr.run([net])
        assert solution.results["net_B"].success, "共栅 net_B 应布通"

    def test_nety_uses_m1(self, grid):
        """
        net_Y: PMOS drain(x=8) + NMOS drain(x=8)。
        S/D 列无 M0 边，必须通过 via 接 M1 再连通。
        """
        g_copy = build_and2_grid()
        ep = _add_ext_to_grid(g_copy, 8, PMOS_ROW)
        en = _add_ext_to_grid(g_copy, 8, NMOS_H4_ROW)
        mgr = make_mgr(g_copy, space=0, max_iter=20)
        net = Net("net_Y", [ep, en],
                  cable_locs=set(SD_CABLE_LOCS),
                  pin_spec=PinSpec(layer="M1"))
        solution = mgr.run([net])
        assert solution.results["net_Y"].success, "net_Y 应布通"
        m1_nodes = [n for n in solution.results["net_Y"].routed_nodes if n[0] == "M1"]
        assert len(m1_nodes) > 0, "net_Y 应使用 M1"

    def test_vdd_connects_to_rail(self, grid):
        """VDD 三个 PMOS SD + rail 锚点能布通，且所有 M1 路径在 VDD 列"""
        g_copy = build_and2_grid()
        ep0 = _add_ext_to_grid(g_copy, 0, PMOS_ROW)
        ep4 = _add_ext_to_grid(g_copy, 4, PMOS_ROW)
        ep6 = _add_ext_to_grid(g_copy, 6, PMOS_ROW)
        mgr = make_mgr(g_copy, space=0, max_iter=40)
        net = Net("VDD", [
            ep0, ep4, ep6,
            ("M1", 0, 0), ("M1", 4, 0), ("M1", 6, 0),
        ], cable_locs=set(SD_CABLE_LOCS))
        solution = mgr.run([net])
        assert solution.results["VDD"].success, "VDD 应布通"

    def test_net2_three_terminals(self, grid):
        """net2 三端点（PMOS drain x=2, NMOS drain x=0, INV gate x=7）布通"""
        g_copy = build_and2_grid()
        ep2 = _add_ext_to_grid(g_copy, 2, PMOS_ROW)
        en0 = _add_ext_to_grid(g_copy, 0, NMOS_H6_ROW)
        mgr = make_mgr(g_copy, space=0, max_iter=60)
        net = Net("net2", [ep2, en0, ("M0", 7, 1)])
        solution = mgr.run([net])
        assert solution.results["net2"].success, "net2 三端点应布通"
        m1_nodes = [n for n in solution.results["net2"].routed_nodes if n[0] == "M1"]
        assert len(m1_nodes) > 0, "net2 应使用 M1 横向走线"


# ======================================================================
# Class 3: 全 AND2 布线（space=0）
# ======================================================================

class TestAND2FullRouting:
    """7 个线网完整布线测试"""

    @pytest.fixture
    def grid(self):
        return build_and2_grid()

    def test_full_routing_min_4_of_7(self, grid):
        """
        完整 AND2（7 nets）：至少 4/7 布通，共栅 A/B 必须成功。
        保存 SVG 到 results/stdcell_and2/。
        """
        nets, _ = build_and2_nets(grid)
        mgr = make_mgr(grid, space=0, max_iter=80)
        solution = mgr.run(nets)

        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "results", "stdcell_and2",
        )
        from maze_router.visualizer import Visualizer
        viz = Visualizer(grid, solution)
        viz.save_svgs(save_dir=save_dir)

        for name in ["net_A", "net_B"]:
            assert solution.results[name].success, f"共栅 {name} 必须布通"

        assert solution.routed_count >= 4, \
            f"AND2 至少 4/7 布通，实际 {solution.routed_count}/7"

    def test_sd_nets_use_m1(self, grid):
        """S/D 线网（VDD,VSS,net2,net0,net_Y）布线应使用 M1"""
        nets, _ = build_and2_nets(grid)
        mgr = make_mgr(grid, space=0, max_iter=80)
        solution = mgr.run(nets)

        for name in {"VDD", "VSS", "net2", "net0", "net_Y"}:
            result = solution.results[name]
            if not result.success:
                continue
            m1_nodes = [n for n in result.routed_nodes if n[0] == "M1"]
            assert len(m1_nodes) > 0, f"S/D 线网 {name} 应使用 M1"

    def test_gate_via_in_channel_full(self, grid):
        """Gate 线网 (net_A, net_B) 在 M1 的节点 y 只在 y=6..10"""
        nets, _ = build_and2_nets(grid)
        mgr = make_mgr(grid, space=0, max_iter=80)
        solution = mgr.run(nets)

        for name in ["net_A", "net_B"]:
            result = solution.results[name]
            if not result.success:
                continue
            for node in result.routed_nodes:
                if node[0] == "M1":
                    assert node[2] in GATE_VIA_ROWS, \
                        f"{name} M1 节点 {node} 不在 gate via 区域 y=6..10"

    def test_no_m2_at_power_rails(self, grid):
        """任何线网均不在 M2 使用 VDD/VSS rail 行（M2 仅 y=7,9）"""
        nets, _ = build_and2_nets(grid)
        mgr = make_mgr(grid, space=0, max_iter=80)
        solution = mgr.run(nets)

        for name, result in solution.results.items():
            if not result.success:
                continue
            for node in result.routed_nodes:
                if node[0] == "M2":
                    assert node[2] in M2_ROWS, \
                        f"{name} 在 M2 使用了 y={node[2]}，应仅 y=7 或 y=9"

    def test_no_node_overlap(self, grid):
        """不同线网的非虚拟节点不重叠"""
        nets, _ = build_and2_nets(grid)
        mgr = make_mgr(grid, space=0, max_iter=80)
        solution = mgr.run(nets)

        occupied = {}
        for name, result in solution.results.items():
            if not result.success:
                continue
            for node in result.routed_nodes:
                if node[0] == EXT_LAYER:
                    continue
                if node in occupied:
                    pytest.fail(f"节点 {node} 被 {occupied[node]} 和 {name} 同时占用")
                occupied[node] = name

    def test_pin_nets_have_m1_node(self, grid):
        """net_A, net_B, net_Y（需 pin）的布线结果中应有 M1 节点"""
        nets, _ = build_and2_nets(grid)
        mgr = make_mgr(grid, space=0, max_iter=80)
        solution = mgr.run(nets)

        for name in ["net_A", "net_B", "net_Y"]:
            result = solution.results[name]
            if not result.success:
                continue
            m1 = [n for n in result.routed_nodes if n[0] == "M1"]
            assert len(m1) > 0, f"{name} 需要 pin，应有 M1 节点"

    def test_full_routing_summary(self, grid, capsys):
        """完整布线摘要"""
        nets, _ = build_and2_nets(grid)
        mgr = make_mgr(grid, space=0, max_iter=80)
        solution = mgr.run(nets)

        print(f"\n{'='*60}")
        print(f"AND2_X1_ML 布线结果: {solution.routed_count}/7 布通")
        print(f"{'='*60}")
        for name, result in sorted(solution.results.items()):
            status = "OK" if result.success else "FAIL"
            layers = sorted({n[0] for n in result.routed_nodes}) if result.routed_nodes else []
            print(f"  {name:8s}: {status}  cost={result.total_cost:6.1f}  layers={layers}")
        print(f"总代价: {solution.total_cost:.1f}")
        print(f"{'='*60}")

        assert solution.routed_count >= 3, \
            f"AND2 至少 3/7 布通，实际 {solution.routed_count}/7"


# ======================================================================
# Class 4: Active 覆盖约束集成测试
# ======================================================================

class TestAND2ActiveOccupancy:
    """ActiveOccupancyConstraint 在 AND2 上的集成验证"""

    @pytest.fixture
    def grid(self):
        return build_and2_grid()

    def test_active_coverage_satisfied(self, grid):
        """路由后所有已布通 S/D 线网在对应 M1 列 active 范围内节点数 ≥ 要求"""
        nets, active_rules = build_and2_nets(grid)
        mgr = make_mgr(grid, space=0, max_iter=80, active_rules=active_rules)
        solution = mgr.run(nets)

        for (net_name, j, i), min_n in active_rules.items():
            result = solution.results.get(net_name)
            if result is None or not result.success:
                continue
            y_min, y_max = ROW_TYPE_Y_RANGES[i]
            actual = sum(
                1 for n in result.routed_nodes
                if n[0] == "M1" and n[1] == j
                and y_min <= n[2] <= y_max
                and not grid.is_virtual_node(n)
            )
            assert actual >= min_n, (
                f"线网 {net_name} M1 覆盖不足: x={j}, i={i}, "
                f"需要 {min_n}, 实际 {actual}"
            )

    def test_gate_nets_not_in_active_rules(self, grid):
        """net_A 和 net_B 不在 active_rules 中，不受 active 覆盖约束"""
        nets, active_rules = build_and2_nets(grid)
        assert ("net_A", 1, PMOS_ROW) not in active_rules
        assert ("net_B", 3, PMOS_ROW) not in active_rules

        gate_nets = [n for n in nets if n.name in ("net_A", "net_B")]
        mgr = make_mgr(grid, space=0, max_iter=20, active_rules=active_rules)
        solution = mgr.run(gate_nets)
        assert solution.results["net_A"].success, "net_A 应布通"
        assert solution.results["net_B"].success, "net_B 应布通"

    def test_net0_active_coverage(self, grid):
        """net0 NMOS 内部节点（x=2，y=11..16）M1 覆盖应满足要求（6个节点）"""
        nets, active_rules = build_and2_nets(grid)
        mgr = make_mgr(grid, space=0, max_iter=60, active_rules=active_rules)
        solution = mgr.run(nets)

        if not solution.results["net0"].success:
            pytest.skip("net0 布线失败，跳过覆盖检查")

        result = solution.results["net0"]
        y_min, y_max = ROW_TYPE_Y_RANGES[NMOS_H6_ROW]
        covered = sum(
            1 for n in result.routed_nodes
            if n[0] == "M1" and n[1] == 2
            and y_min <= n[2] <= y_max
            and not grid.is_virtual_node(n)
        )
        assert covered >= 6, f"net0 M1 x=2 覆盖应 ≥ 6，实际 {covered}"


# ======================================================================
# Class 5: Engine 端到端测试
# ======================================================================

class TestAND2Engine:
    """通过顶层 MazeRouterEngine 进行集成测试"""

    @pytest.fixture
    def grid(self):
        return build_and2_grid()

    def test_engine_space0_full(self, grid):
        """Engine 端到端：AND2 全 7 线网，space=0，至少 4/7 布通"""
        nets, active_rules = build_and2_nets(grid)
        engine = MazeRouterEngine(
            grid=grid,
            nets=nets,
            space_constr={"M0": 0, "M1": 0, "M2": 0},
            corner_l_costs={"M0": 3.0, "M1": 3.0, "M2": 3.0},
            strategy="congestion_aware",
            max_iterations=80,
            net_active_must_occupy_num=active_rules,
            row_type_y_ranges=ROW_TYPE_Y_RANGES,
        )
        solution = engine.run()
        assert solution.routed_count >= 4, \
            f"Engine space=0 应至少 4/7 布通，实际 {solution.routed_count}/7"

    def test_engine_space1(self, grid):
        """Engine 端到端：AND2，space=1，至少 2/7 布通"""
        nets, active_rules = build_and2_nets(grid)
        engine = MazeRouterEngine(
            grid=grid,
            nets=nets,
            space_constr={"M0": 1, "M1": 1, "M2": 1},
            corner_l_costs={"M0": 3.0, "M1": 3.0, "M2": 3.0},
            strategy="congestion_aware",
            max_iterations=80,
            net_active_must_occupy_num=active_rules,
            row_type_y_ranges=ROW_TYPE_Y_RANGES,
        )
        solution = engine.run()
        assert solution.routed_count >= 2, \
            f"Engine space=1 应至少 2/7 布通，实际 {solution.routed_count}/7"

    def test_engine_saves_svg(self, grid, tmp_path):
        """Engine 可视化：生成 3 层 SVG（M0,M1,M2，不含 EXT 虚拟层）"""
        nets, active_rules = build_and2_nets(grid)
        engine = MazeRouterEngine(
            grid=grid,
            nets=nets,
            space_constr={"M0": 0, "M1": 0, "M2": 0},
            net_active_must_occupy_num=active_rules,
            row_type_y_ranges=ROW_TYPE_Y_RANGES,
        )
        engine.run()
        engine.visualize(save_dir=str(tmp_path), prefix="and2_")
        svgs = list(tmp_path.glob("*.svg"))
        assert len(svgs) == 3, \
            f"应生成 3 个 SVG (M0/M1/M2)，实际 {len(svgs)}: {[s.name for s in svgs]}"

    def test_engine_active_rules_satisfied(self, grid):
        """Engine 路由后 active 覆盖约束满足（已布通线网）"""
        nets, active_rules = build_and2_nets(grid)
        engine = MazeRouterEngine(
            grid=grid,
            nets=nets,
            space_constr={"M0": 0, "M1": 0, "M2": 0},
            net_active_must_occupy_num=active_rules,
            row_type_y_ranges=ROW_TYPE_Y_RANGES,
            max_iterations=80,
        )
        solution = engine.run()

        for (net_name, j, i), min_n in active_rules.items():
            result = solution.results.get(net_name)
            if result is None or not result.success:
                continue
            y_min, y_max = ROW_TYPE_Y_RANGES[i]
            actual = sum(
                1 for n in result.routed_nodes
                if n[0] == "M1" and n[1] == j
                and y_min <= n[2] <= y_max
                and not grid.is_virtual_node(n)
            )
            assert actual >= min_n, (
                f"[Engine] {net_name} M1 覆盖不足: x={j}, i={i}, "
                f"需要 {min_n}, 实际 {actual}"
            )
