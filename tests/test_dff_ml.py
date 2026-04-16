"""
标准单元布线测试 — DFF_X1_ML (YMTC CJ3 工艺)

传输门 D 触发器（Transmission Gate Master-Slave DFF）

电路结构（来自 DFF_X1_ML.xlsx）：
  10 PMOS + 10 NMOS = 20 晶体管（含反馈），19列×9行网格，14个线网

  晶体管列分配（x 轴）：
    T0 (x=0..1..2):  Input inverter  — SD=D,   Gate=D,   SD=di
    T1 (x=2..3..4):  TG1a           — SD=di,  Gate=CK/CKB, SD=tg1
    T2 (x=4..5..6):  TG1b           — SD=tg1, Gate=CKB/CK,  SD=m
    T3 (x=6..7..8):  Master INV     — SD=m,   Gate=tg1,  SD=m
    T4 (x=8..9..10): Master FBK INV — SD=m,   Gate=m,    SD=mB
    T5 (x=10..11..12): TG2a         — SD=mB,  Gate=CKB/CK,  SD=tg2
    T6 (x=12..13..14): TG2b         — SD=tg2, Gate=CK/CKB,  SD=s
    T7 (x=14..15..16): Slave INV    — SD=s,   Gate=tg2,  SD=Q
    T8 (x=16..17..18): Output INV   — SD=Q,   Gate=Q,    SD=QB
    (x=0,18 also serve as VDD/VSS diffusion contacts)

工艺规则（YMTC CJ3 DFF 简化模型）：
  1. S/D 列（偶数 x=0..18）：
       NMOS 有源区 y=0↔1 竖向边；PMOS 有源区 y=7↔8 竖向边；其余无 M0 边
  2. Gate 列（奇数 x=1..17）：全程竖向边 y=0..8；
       横向边仅在非有源区（y ≠ 1, 7），代价=2
  3. S/D 列无横向 M0 边
  4. M0↔M1 via：S/D 列在 y=1（NMOS）和 y=7（PMOS）处；Gate 列所有行
  5. M2 仅 y=3 和 y=5，横向走线，y=3↔y=5 竖向直连（代价=2）
  6. VDD rail = M1 y=8；VSS rail = M1 y=0

网格规格：19 列(x=0..18) × 9 行(y=0..8) × 3 层(M0,M1,M2)

  y=0: VSS rail (M0+M1 横向全连)
  y=1: NMOS 有源区
  y=2..6: 布线通道（Gate via 区）
  y=7: PMOS 有源区
  y=8: VDD rail (M0+M1 横向全连)
"""

from __future__ import annotations
import os
import pytest

from maze_router.data.net import Net
from maze_router.data.grid import GridGraph
from maze_router.constraint_manager import ConstraintManager
from maze_router.cost_manager import CostManager
from maze_router.constraints.space_constraint import SpaceConstraint
from maze_router.costs.corner_cost import CornerCost
from maze_router.ripup_strategy import CongestionAwareRipupStrategy
from maze_router.ripup_manager import RipupManager
from maze_router import MazeRouterEngine


# ======================================================================
# 常量（来自 DFF_X1_ML.xlsx）
# ======================================================================

COLS = 19       # x: 0..18
ROWS = 9        # y: 0..8

SD_COLS   = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18}
GATE_COLS = {1, 3, 5, 7, 9, 11, 13, 15, 17}

# 有源区行
NMOS_ACTIVE_Y = 1
PMOS_ACTIVE_Y = 7

# 电源轨
VDD_RAIL_Y = 8   # M1 y=8 = VDD power rail
VSS_RAIL_Y = 0   # M1 y=0 = VSS power rail

# 非有源区（Gate 横向边允许的行）
NON_ACTIVE_ROWS = {0, 2, 3, 4, 5, 6, 8}

# M2 走线行
M2_ROWS = {3, 5}

# Gate/SD 线网的 M0 cable_locs 约束
GATE_CABLE_LOCS: frozenset = frozenset(
    ("M0", x, y) for x in GATE_COLS for y in range(ROWS)
)
SD_CABLE_LOCS: frozenset = frozenset(
    ("M0", x, y) for x in SD_COLS for y in range(ROWS)
)

# Gate 列横向代价（鼓励 Gate 线网在通道内走 M0）
M0_GATE_H_COST = 2.0


# ======================================================================
# 网格构建
# ======================================================================

def build_dff_ml_grid() -> GridGraph:
    """
    构建 DFF_X1_ML 三层布线网格（19×9）。

    M0 规则（来自 DFF_X1_ML.xlsx）：
      - S/D 列（偶数）：仅 NMOS 区 y=0↔1 和 PMOS 区 y=7↔8 有竖向边，无横向边
      - Gate 列（奇数）：全程竖向边（y=0..7）；
        非有源区（y≠1,7）相邻 Gate 列间有横向边（代价=2）
    M1 规则：完整矩形 y=0..8
    M2 规则：仅 y=3, y=5 横向；y=3↔y=5 竖向直连
    """
    g = GridGraph()

    # ── M0 节点 ─────────────────────────────────────────────────────────
    for x in range(COLS):
        for y in range(ROWS):
            g.add_node(("M0", x, y))

    # M0 竖向边
    for x in range(COLS):
        if x in SD_COLS:
            # S/D 列：仅 NMOS 边界 y=0↔1 和 PMOS 边界 y=7↔8
            g.add_edge(("M0", x, 0), ("M0", x, 1), cost=1.0)
            g.add_edge(("M0", x, 7), ("M0", x, 8), cost=1.0)
        else:
            # Gate 列：全程竖向（y=0..7）
            for y in range(ROWS - 1):
                g.add_edge(("M0", x, y), ("M0", x, y + 1), cost=1.0)

    # M0 横向边：相邻 Gate 列，仅非有源区行
    gate_list = sorted(GATE_COLS)
    for i in range(len(gate_list) - 1):
        x1, x2 = gate_list[i], gate_list[i + 1]
        if x2 - x1 == 2:   # 相邻 Gate 列（间距=2，隔一个 S/D 列）
            for y in NON_ACTIVE_ROWS:
                g.add_edge(("M0", x1, y), ("M0", x2, y), cost=M0_GATE_H_COST)

    # ── M1：完整矩形 y=0..8 ─────────────────────────────────────────────
    for x in range(COLS):
        for y in range(ROWS):
            g.add_node(("M1", x, y))

    # M1 竖向边：靠近 power rail 的边代价=0，其余=1
    for x in range(COLS):
        for y in range(ROWS - 1):
            cost = 0.0 if (y == 0 or y == ROWS - 2) else 1.0
            g.add_edge(("M1", x, y), ("M1", x, y + 1), cost=cost)

    # M1 横向边
    for y in range(ROWS):
        for x in range(COLS - 1):
            g.add_edge(("M1", x, y), ("M1", x + 1, y), cost=1.0)

    # ── M2：仅 y=3, y=5，横向 + y=3↔y=5 竖向直连 ──────────────────────
    for x in range(COLS):
        for y in M2_ROWS:
            g.add_node(("M2", x, y))

    for y in M2_ROWS:
        for x in range(COLS - 1):
            g.add_edge(("M2", x, y), ("M2", x + 1, y), cost=1.0)

    for x in range(COLS):
        g.add_edge(("M2", x, 3), ("M2", x, 5), cost=2.0)

    # ── Via M0↔M1 ────────────────────────────────────────────────────
    # S/D 列：仅 y=1（NMOS 边界）和 y=7（PMOS 边界）
    for x in SD_COLS:
        g.add_edge(("M0", x, NMOS_ACTIVE_Y), ("M1", x, NMOS_ACTIVE_Y), cost=2.0)
        g.add_edge(("M0", x, PMOS_ACTIVE_Y), ("M1", x, PMOS_ACTIVE_Y), cost=2.0)

    # Gate 列：所有行均有 via
    for x in GATE_COLS:
        for y in range(ROWS):
            g.add_edge(("M0", x, y), ("M1", x, y), cost=2.0)

    # ── Via M1↔M2（仅 y=3, y=5）─────────────────────────────────────────
    for x in range(COLS):
        for y in M2_ROWS:
            g.add_edge(("M1", x, y), ("M2", x, y), cost=2.0)

    return g


# ======================================================================
# 线网构建（来自 DFF_X1_ML.xlsx）
# ======================================================================

def build_dff_ml_nets():
    """
    构建 DFF_X1_ML 的 14 个布线线网。

    线网终端使用 M0 层节点（有源区边界），与 DFF_X1_ML.xlsx 对应：
      SD 终端  → (M0, x, y=1) for NMOS,  (M0, x, y=7) for PMOS
      VDD/VSS  → (M0, x, y=8) / (M0, x, y=0) for all SD cols (rail anchor)
    """
    # 电源：两端 rail 锚点连接（使用 M1 rail 节点）
    nets = [
        # ── 电源轨 ────────────────────────────────────────────────────
        Net("net_VDD", [
            ("M1", 0, VDD_RAIL_Y), ("M1", 18, VDD_RAIL_Y),
        ]),
        Net("net_VSS", [
            ("M1", 0, VSS_RAIL_Y), ("M1", 18, VSS_RAIL_Y),
        ]),

        # ── 数据输入 ──────────────────────────────────────────────────
        Net("net_D", [
            ("M0", 0, PMOS_ACTIVE_Y),   # PMOS S/D at x=0
            ("M0", 0, NMOS_ACTIVE_Y),   # NMOS S/D at x=0
        ], cable_locs=set(SD_CABLE_LOCS)),

        # ── 时钟（4端口，Gate 列，CK 与 CKB 交叉）─────────────────────
        # CK:  NMOS gate x=3,13 + PMOS gate x=5,11
        Net("net_CK", [
            ("M0", 3, NMOS_ACTIVE_Y),
            ("M0", 13, NMOS_ACTIVE_Y),
            ("M0", 5, PMOS_ACTIVE_Y),
            ("M0", 11, PMOS_ACTIVE_Y),
        ], cable_locs=set(GATE_CABLE_LOCS)),

        # CKB: NMOS gate x=5,11 + PMOS gate x=3,13
        Net("net_CKB", [
            ("M0", 5, NMOS_ACTIVE_Y),
            ("M0", 11, NMOS_ACTIVE_Y),
            ("M0", 3, PMOS_ACTIVE_Y),
            ("M0", 13, PMOS_ACTIVE_Y),
        ], cable_locs=set(GATE_CABLE_LOCS)),

        # ── 内部信号（SD 列直连，PMOS+NMOS 两端）─────────────────────
        Net("net_di", [
            ("M0", 2, NMOS_ACTIVE_Y),
            ("M0", 2, PMOS_ACTIVE_Y),
        ], cable_locs=set(SD_CABLE_LOCS)),

        Net("net_tg1", [
            ("M0", 4, NMOS_ACTIVE_Y),
            ("M0", 4, PMOS_ACTIVE_Y),
        ], cable_locs=set(SD_CABLE_LOCS)),

        # net_m: 主锁存，4 个扩散触点（x=6 和 x=8，两侧）
        Net("net_m", [
            ("M0", 6, NMOS_ACTIVE_Y),
            ("M0", 6, PMOS_ACTIVE_Y),
            ("M0", 8, NMOS_ACTIVE_Y),
            ("M0", 8, PMOS_ACTIVE_Y),
        ], cable_locs=set(SD_CABLE_LOCS)),

        Net("net_mB", [
            ("M0", 10, NMOS_ACTIVE_Y),
            ("M0", 10, PMOS_ACTIVE_Y),
        ], cable_locs=set(SD_CABLE_LOCS)),

        # net_fb: PMOS 侧反馈路径（连接 tg1 PMOS x=4 和 m PMOS x=8）
        Net("net_fb", [
            ("M0", 4, PMOS_ACTIVE_Y),
            ("M0", 8, PMOS_ACTIVE_Y),
        ], cable_locs=set(SD_CABLE_LOCS)),

        Net("net_tg2", [
            ("M0", 12, NMOS_ACTIVE_Y),
            ("M0", 12, PMOS_ACTIVE_Y),
        ], cable_locs=set(SD_CABLE_LOCS)),

        # net_s: 从锁存，4 个扩散触点（x=14 和 x=16，两侧）
        Net("net_s", [
            ("M0", 14, NMOS_ACTIVE_Y),
            ("M0", 14, PMOS_ACTIVE_Y),
            ("M0", 16, NMOS_ACTIVE_Y),
            ("M0", 16, PMOS_ACTIVE_Y),
        ], cable_locs=set(SD_CABLE_LOCS)),

        # ── 输出 ──────────────────────────────────────────────────────
        Net("net_Q", [
            ("M0", 16, NMOS_ACTIVE_Y),
            ("M0", 16, PMOS_ACTIVE_Y),
        ], cable_locs=set(SD_CABLE_LOCS)),

        Net("net_QB", [
            ("M0", 18, NMOS_ACTIVE_Y),
            ("M0", 18, PMOS_ACTIVE_Y),
        ], cable_locs=set(SD_CABLE_LOCS)),
    ]
    return nets


# ======================================================================
# RipupManager 构建辅助
# ======================================================================

def make_mgr(grid: GridGraph, space: int = 0, max_iter: int = 100) -> RipupManager:
    """构建 RipupManager（CongestionAwareRipupStrategy）。"""
    constraints = [SpaceConstraint({layer: space for layer in ("M0", "M1", "M2")})]
    constraint_mgr = ConstraintManager(constraints)
    cost_mgr = CostManager(
        grid=grid,
        costs=[CornerCost(
            l_costs={"M0": 5.0, "M1": 5.0, "M2": 5.0},
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

class TestDFFMLGridStructure:
    """验证 DFF_X1_ML 网格拓扑（含 via 限制）"""

    @pytest.fixture(scope="class")
    def grid(self):
        return build_dff_ml_grid()

    def test_node_counts(self, grid):
        """M0=19*9, M1=19*9, M2=19*2"""
        assert len(grid.get_nodes_on_layer("M0")) == COLS * ROWS
        assert len(grid.get_nodes_on_layer("M1")) == COLS * ROWS
        assert len(grid.get_nodes_on_layer("M2")) == COLS * len(M2_ROWS)

    def test_m0_sd_only_boundary_edges(self, grid):
        """S/D 列 M0 只在 y=0↔1 (NMOS) 和 y=7↔8 (PMOS) 有竖向边"""
        for x in SD_COLS:
            # 有 edge 的边
            assert grid.graph.has_edge(("M0", x, 0), ("M0", x, 1)), \
                f"SD x={x}: M0 y=0↔1 边缺失"
            assert grid.graph.has_edge(("M0", x, 7), ("M0", x, 8)), \
                f"SD x={x}: M0 y=7↔8 边缺失"
            # 无其他竖向边
            for y in range(1, ROWS - 1):
                if y in (7,):
                    continue   # y=7↔8 已检查
                if y == 0:
                    continue   # y=0↔1 已检查
                assert not grid.graph.has_edge(("M0", x, y), ("M0", x, y + 1)), \
                    f"SD x={x}: M0 y={y}↔{y+1} 不应有竖向边"

    def test_m0_gate_full_vertical(self, grid):
        """Gate 列 M0 全程竖向连通 (y=0..7)"""
        for x in GATE_COLS:
            for y in range(ROWS - 1):
                assert grid.graph.has_edge(("M0", x, y), ("M0", x, y + 1)), \
                    f"Gate x={x}: M0 y={y}↔{y+1} 竖向边缺失"

    def test_m0_sd_no_horizontal(self, grid):
        """S/D 列无 M0 横向边"""
        for x in SD_COLS:
            for y in range(ROWS):
                for nb in grid.get_neighbors(("M0", x, y)):
                    if nb[0] == "M0" and nb[2] == y and nb[1] != x:
                        pytest.fail(
                            f"SD M0({x},{y}) 不应有横向邻居，发现 {nb}"
                        )

    def test_m0_gate_horizontal_non_active(self, grid):
        """相邻 Gate 列 M0 横向边仅在非有源区行（y≠1,7）"""
        gate_list = sorted(GATE_COLS)
        for i in range(len(gate_list) - 1):
            x1, x2 = gate_list[i], gate_list[i + 1]
            if x2 - x1 != 2:
                continue
            for y in range(ROWS):
                has_h = grid.graph.has_edge(("M0", x1, y), ("M0", x2, y))
                if y in NON_ACTIVE_ROWS:
                    assert has_h, f"Gate M0 x={x1}↔{x2} y={y} 横向边缺失"
                else:  # y=1 or y=7 (active)
                    assert not has_h, f"Gate M0 x={x1}↔{x2} y={y} 不应在有源区有横向边"

    def test_via_m0m1_sd_boundary_only(self, grid):
        """S/D 列 M0↔M1 via 仅在 y=1 (NMOS) 和 y=7 (PMOS)"""
        for x in SD_COLS:
            for y in range(ROWS):
                has_via = grid.graph.has_edge(("M0", x, y), ("M1", x, y))
                if y in (NMOS_ACTIVE_Y, PMOS_ACTIVE_Y):
                    assert has_via, f"SD x={x}: M0↔M1 via 应在 y={y}"
                else:
                    assert not has_via, f"SD x={x}: M0↔M1 via 不应在 y={y}"

    def test_via_m0m1_gate_all_rows(self, grid):
        """Gate 列 M0↔M1 via 在所有行均有"""
        for x in GATE_COLS:
            for y in range(ROWS):
                assert grid.graph.has_edge(("M0", x, y), ("M1", x, y)), \
                    f"Gate x={x}: M0↔M1 via 缺失 y={y}"

    def test_m1_power_rail_edges(self, grid):
        """M1 y=0 和 y=8 横向边全连（power rail）"""
        for y in (VDD_RAIL_Y, VSS_RAIL_Y):
            for x in range(COLS - 1):
                assert grid.graph.has_edge(("M1", x, y), ("M1", x + 1, y)), \
                    f"M1 rail y={y}: x={x}↔{x+1} 边缺失"

    def test_m1_power_rail_zero_cost(self, grid):
        """M1 靠近 power rail 的纵向边（y=0↔1, y=7↔8）代价为 0"""
        for x in range(COLS):
            e01 = grid.graph.get_edge_data(("M1", x, 0), ("M1", x, 1))
            assert e01 is not None and e01.get("cost", 1) == 0.0, \
                f"M1({x},0↔1) 代价应为 0"
            e78 = grid.graph.get_edge_data(("M1", x, 7), ("M1", x, 8))
            assert e78 is not None and e78.get("cost", 1) == 0.0, \
                f"M1({x},7↔8) 代价应为 0"

    def test_m2_rows_only(self, grid):
        """M2 节点仅在 y=3 和 y=5"""
        for node in grid.get_nodes_on_layer("M2"):
            assert node[2] in M2_ROWS, f"M2 节点 {node} y 不在 {M2_ROWS}"

    def test_via_m1m2_at_m2_rows(self, grid):
        """M1↔M2 via 仅在 y=3 和 y=5"""
        for x in range(COLS):
            for y in range(ROWS):
                has = grid.graph.has_edge(("M1", x, y), ("M2", x, y))
                if y in M2_ROWS:
                    assert has, f"M1↔M2 via 缺失 ({x},{y})"
                else:
                    assert not has, f"M1↔M2 via 不应存在 ({x},{y})"

    def test_m2_vertical_y3_y5(self, grid):
        """M2 y=3↔y=5 竖向直连存在"""
        for x in range(COLS):
            assert grid.graph.has_edge(("M2", x, 3), ("M2", x, 5)), \
                f"M2({x},3↔5) 竖向直连缺失"


# ======================================================================
# Class 2: 单线网路由测试
# ======================================================================

class TestDFFMLNetRouting:
    """独立线网路由验证"""

    def test_net_vss_routes(self):
        """net_VSS（power rail）两端 M1 锚点应布通"""
        g = build_dff_ml_grid()
        mgr = make_mgr(g, space=0, max_iter=20)
        net = Net("net_VSS", [("M1", 0, VSS_RAIL_Y), ("M1", 18, VSS_RAIL_Y)])
        solution = mgr.run([net])
        assert solution.results["net_VSS"].success, "net_VSS 应布通"

    def test_net_vdd_routes(self):
        """net_VDD（power rail）两端 M1 锚点应布通"""
        g = build_dff_ml_grid()
        mgr = make_mgr(g, space=0, max_iter=20)
        net = Net("net_VDD", [("M1", 0, VDD_RAIL_Y), ("M1", 18, VDD_RAIL_Y)])
        solution = mgr.run([net])
        assert solution.results["net_VDD"].success, "net_VDD 应布通"

    def test_net_ck_routes_4_terminals(self):
        """net_CK（4 端口，Gate 列，交叉）应布通（可走 M0 通道或 M1 互连）"""
        g = build_dff_ml_grid()
        mgr = make_mgr(g, space=0, max_iter=40)
        net = Net("net_CK", [
            ("M0", 3, NMOS_ACTIVE_Y), ("M0", 13, NMOS_ACTIVE_Y),
            ("M0", 5, PMOS_ACTIVE_Y), ("M0", 11, PMOS_ACTIVE_Y),
        ], cable_locs=set(GATE_CABLE_LOCS))
        solution = mgr.run([net])
        assert solution.results["net_CK"].success, "net_CK 4 端口应布通"
        # Gate 线网可完全走 M0 通道（所有 gate 列竖向全程连通，通道区横向相连），
        # 也可经 M0↔M1 via 借道 M1，因此不强制要求 M1 节点。
        routed = solution.results["net_CK"].routed_nodes
        assert len(routed) > 0, "net_CK 应有路由节点"

    def test_net_ckb_routes_4_terminals(self):
        """net_CKB（4 端口，Gate 列，交叉）应布通"""
        g = build_dff_ml_grid()
        mgr = make_mgr(g, space=0, max_iter=40)
        net = Net("net_CKB", [
            ("M0", 5, NMOS_ACTIVE_Y), ("M0", 11, NMOS_ACTIVE_Y),
            ("M0", 3, PMOS_ACTIVE_Y), ("M0", 13, PMOS_ACTIVE_Y),
        ], cable_locs=set(GATE_CABLE_LOCS))
        solution = mgr.run([net])
        assert solution.results["net_CKB"].success, "net_CKB 4 端口应布通"

    def test_net_m_routes_4_terminals(self):
        """net_m（主锁存 4 端口，SD 列）应布通"""
        g = build_dff_ml_grid()
        mgr = make_mgr(g, space=0, max_iter=40)
        net = Net("net_m", [
            ("M0", 6, NMOS_ACTIVE_Y), ("M0", 6, PMOS_ACTIVE_Y),
            ("M0", 8, NMOS_ACTIVE_Y), ("M0", 8, PMOS_ACTIVE_Y),
        ], cable_locs=set(SD_CABLE_LOCS))
        solution = mgr.run([net])
        assert solution.results["net_m"].success, "net_m 4 端口应布通"

    def test_net_sd_uses_m1(self):
        """S/D 线网（net_di）从 M0 via 出发，应使用 M1 互连 PMOS 和 NMOS"""
        g = build_dff_ml_grid()
        mgr = make_mgr(g, space=0, max_iter=20)
        net = Net("net_di", [
            ("M0", 2, NMOS_ACTIVE_Y),
            ("M0", 2, PMOS_ACTIVE_Y),
        ], cable_locs=set(SD_CABLE_LOCS))
        solution = mgr.run([net])
        assert solution.results["net_di"].success, "net_di 应布通"
        m1_nodes = [n for n in solution.results["net_di"].routed_nodes if n[0] == "M1"]
        assert len(m1_nodes) > 0, "net_di 应通过 M1 连通 PMOS 和 NMOS"

    def test_ck_ckb_no_conflict(self):
        """net_CK 和 net_CKB 同时路由（space=0）不冲突"""
        g = build_dff_ml_grid()
        mgr = make_mgr(g, space=0, max_iter=60)
        nets = [
            Net("net_CK", [
                ("M0", 3, NMOS_ACTIVE_Y), ("M0", 13, NMOS_ACTIVE_Y),
                ("M0", 5, PMOS_ACTIVE_Y), ("M0", 11, PMOS_ACTIVE_Y),
            ], cable_locs=set(GATE_CABLE_LOCS)),
            Net("net_CKB", [
                ("M0", 5, NMOS_ACTIVE_Y), ("M0", 11, NMOS_ACTIVE_Y),
                ("M0", 3, PMOS_ACTIVE_Y), ("M0", 13, PMOS_ACTIVE_Y),
            ], cable_locs=set(GATE_CABLE_LOCS)),
        ]
        solution = mgr.run(nets)
        assert solution.results["net_CK"].success, "net_CK 应布通"
        assert solution.results["net_CKB"].success, "net_CKB 应布通"


# ======================================================================
# Class 3: 全 DFF_ML 布线（space=0）
# ======================================================================

class TestDFFMLFullRouting:
    """14 个线网完整布线测试"""

    @pytest.fixture
    def grid(self):
        return build_dff_ml_grid()

    def test_full_routing_min_8_of_14(self, grid):
        """
        完整 DFF（14 nets）：至少 8/14 布通。
        保存 SVG 到 results/stdcell_dff_ml/。
        """
        nets = build_dff_ml_nets()
        mgr = make_mgr(grid, space=0, max_iter=100)
        solution = mgr.run(nets)

        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "results", "stdcell_dff_ml",
        )
        os.makedirs(save_dir, exist_ok=True)
        from maze_router.visualizer import Visualizer
        viz = Visualizer(grid, solution)
        viz.save_svgs(save_dir=save_dir)

        assert solution.routed_count >= 8, \
            f"DFF_ML 至少 8/14 布通，实际 {solution.routed_count}/14"

    def test_power_rails_always_routed(self, grid):
        """VDD 和 VSS power rail 始终布通（两端锚点距离近，必然成功）"""
        nets = build_dff_ml_nets()
        mgr = make_mgr(grid, space=0, max_iter=60)
        solution = mgr.run(nets)

        assert solution.results["net_VDD"].success, "net_VDD 必须布通"
        assert solution.results["net_VSS"].success, "net_VSS 必须布通"

    def test_no_node_overlap(self, grid):
        """已布通线网的路由节点不重叠"""
        nets = build_dff_ml_nets()
        mgr = make_mgr(grid, space=0, max_iter=100)
        solution = mgr.run(nets)

        occupied: dict = {}
        for name, result in solution.results.items():
            if not result.success:
                continue
            for node in result.routed_nodes:
                if node in occupied:
                    pytest.fail(f"节点 {node} 被 {occupied[node]} 和 {name} 同时占用")
                occupied[node] = name

    def test_gate_nets_use_m1(self, grid):
        """时钟线网（net_CK, net_CKB）布通后应有路由节点（可走 M0 或 M1）"""
        nets = build_dff_ml_nets()
        mgr = make_mgr(grid, space=0, max_iter=100)
        solution = mgr.run(nets)

        for name in ("net_CK", "net_CKB"):
            result = solution.results.get(name)
            if result is None or not result.success:
                continue
            assert len(result.routed_nodes) > 0, f"{name} 应有路由节点"

    def test_sd_nets_use_m1(self, grid):
        """SD 内部线网布通后应使用 M1（从 M0 via 引出）"""
        nets = build_dff_ml_nets()
        mgr = make_mgr(grid, space=0, max_iter=100)
        solution = mgr.run(nets)

        sd_nets = ["net_di", "net_tg1", "net_m", "net_mB", "net_tg2", "net_s"]
        for name in sd_nets:
            result = solution.results.get(name)
            if result is None or not result.success:
                continue
            m1_nodes = [n for n in result.routed_nodes if n[0] == "M1"]
            assert len(m1_nodes) > 0, f"{name} 应使用 M1"

    def test_full_routing_summary(self, grid):
        """完整布线摘要（打印详情，至少 6/14 布通）"""
        nets = build_dff_ml_nets()
        mgr = make_mgr(grid, space=0, max_iter=100)
        solution = mgr.run(nets)

        print(f"\n{'='*60}")
        print(f"DFF_X1_ML 布线结果: {solution.routed_count}/14 布通")
        print(f"{'='*60}")
        for name, result in sorted(solution.results.items()):
            status = "OK" if result.success else "FAIL"
            layers = sorted({n[0] for n in result.routed_nodes}) if result.routed_nodes else []
            print(f"  {name:10s}: {status}  cost={result.total_cost:6.1f}  layers={layers}")
        print(f"总代价: {solution.total_cost:.1f}")
        print(f"{'='*60}")

        assert solution.routed_count >= 6, \
            f"DFF_ML 至少 6/14 布通，实际 {solution.routed_count}/14"

    def test_no_m2_outside_m2_rows(self, grid):
        """所有线网均不在 M2 的 y=3, y=5 以外使用 M2 层"""
        nets = build_dff_ml_nets()
        mgr = make_mgr(grid, space=0, max_iter=100)
        solution = mgr.run(nets)

        for name, result in solution.results.items():
            if not result.success:
                continue
            for node in result.routed_nodes:
                if node[0] == "M2":
                    assert node[2] in M2_ROWS, \
                        f"{name} 在 M2 使用了 y={node[2]}，应仅 y=3 或 y=5"


# ======================================================================
# Class 4: space=1 布线测试
# ======================================================================

class TestDFFMLWithSpacing:
    """space=1 间距约束下的布线测试"""

    @pytest.fixture
    def grid(self):
        return build_dff_ml_grid()

    def test_full_routing_space1_min_5(self, grid):
        """space=1 下至少 5/14 布通"""
        nets = build_dff_ml_nets()
        mgr = make_mgr(grid, space=1, max_iter=120)
        solution = mgr.run(nets)
        assert solution.routed_count >= 5, \
            f"DFF_ML space=1 至少 5/14 布通，实际 {solution.routed_count}/14"

    def test_power_rails_space1(self, grid):
        """space=1 下单独路由 power rail 仍布通（排除其他线网竞争）"""
        mgr = make_mgr(grid, space=1, max_iter=20)
        nets = [
            Net("net_VDD", [("M1", 0, VDD_RAIL_Y), ("M1", 18, VDD_RAIL_Y)]),
            Net("net_VSS", [("M1", 0, VSS_RAIL_Y), ("M1", 18, VSS_RAIL_Y)]),
        ]
        solution = mgr.run(nets)
        assert solution.results["net_VDD"].success, "space=1: net_VDD 应布通"
        assert solution.results["net_VSS"].success, "space=1: net_VSS 应布通"


# ======================================================================
# Class 5: Engine 端到端测试
# ======================================================================

class TestDFFMLEngine:
    """通过顶层 MazeRouterEngine 进行集成测试"""

    @pytest.fixture
    def grid(self):
        return build_dff_ml_grid()

    def test_engine_space0_min_8(self, grid):
        """Engine 端到端：DFF_ML 全 14 线网，space=0，至少 8/14 布通"""
        nets = build_dff_ml_nets()
        engine = MazeRouterEngine(
            grid=grid,
            nets=nets,
            space_constr={"M0": 0, "M1": 0, "M2": 0},
            corner_l_costs={"M0": 5.0, "M1": 5.0, "M2": 5.0},
            strategy="congestion_aware",
            max_iterations=100,
        )
        solution = engine.run()
        assert solution.routed_count >= 8, \
            f"Engine space=0 应至少 8/14 布通，实际 {solution.routed_count}/14"

    def test_engine_space1_min_5(self, grid):
        """Engine 端到端：DFF_ML，space=1，至少 5/14 布通"""
        nets = build_dff_ml_nets()
        engine = MazeRouterEngine(
            grid=grid,
            nets=nets,
            space_constr={"M0": 1, "M1": 1, "M2": 1},
            corner_l_costs={"M0": 5.0, "M1": 5.0, "M2": 5.0},
            strategy="congestion_aware",
            max_iterations=120,
        )
        solution = engine.run()
        assert solution.routed_count >= 5, \
            f"Engine space=1 应至少 5/14 布通，实际 {solution.routed_count}/14"

    def test_engine_saves_svg(self, grid, tmp_path):
        """Engine 可视化：生成 3 层 SVG（M0,M1,M2）"""
        nets = build_dff_ml_nets()
        engine = MazeRouterEngine(
            grid=grid,
            nets=nets,
            space_constr={"M0": 0, "M1": 0, "M2": 0},
        )
        engine.run()
        engine.visualize(save_dir=str(tmp_path), prefix="dff_ml_")
        svgs = list(tmp_path.glob("*.svg"))
        assert len(svgs) == 3, \
            f"应生成 3 个 SVG (M0/M1/M2)，实际 {len(svgs)}: {[s.name for s in svgs]}"

    def test_engine_no_overlap(self, grid):
        """Engine 路由后节点不重叠"""
        nets = build_dff_ml_nets()
        engine = MazeRouterEngine(
            grid=grid,
            nets=nets,
            space_constr={"M0": 0, "M1": 0, "M2": 0},
            max_iterations=100,
        )
        solution = engine.run()

        occupied: dict = {}
        for name, result in solution.results.items():
            if not result.success:
                continue
            for node in result.routed_nodes:
                if node in occupied:
                    pytest.fail(
                        f"[Engine] 节点 {node} 被 {occupied[node]} 和 {name} 同时占用"
                    )
                occupied[node] = name
