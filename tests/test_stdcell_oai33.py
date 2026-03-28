"""
标准单元版图布线测试 — OAI33门（9行网格）

基于OAI33门（Y = ~((A+B+C)·(D+E+F))）的实际标准单元版图布线场景。
6个PMOS + 6个NMOS晶体管，严格遵守标准单元布线规则。

网格规格：13列(x=0..12) × 9行(y=0..8) × 3层(M0,M1,M2)

============================================================
电路说明：
  OAI33 = OR-AND-INVERT (3-3)
  Y = ~((A+B+C) · (D+E+F))
  NMOS下拉：(NA||NB||NC) 串联 (ND||NE||NF)
  PMOS上拉：(PA-PB-PC串联) 并联 (PD-PE-PF串联)
============================================================

列类型（6个晶体管/行）：
  偶数列(0,2,4,6,8,10,12) = Source/Drain 扩散列
  奇数列(1,3,5,7,9,11)    = Gate 多晶硅列

行含义（9行）：
  y=0: VSS轨道（下边界）
  y=1: NMOS有源区（S/D扩散）
  y=2: 下方布线通道1
  y=3: 下方布线通道2
  y=4: 单元中心
  y=5: 上方布线通道1
  y=6: 上方布线通道2
  y=7: PMOS有源区（S/D扩散）
  y=8: VDD轨道（上边界）

M0约束：
  - S/D列仅竖向布线，NMOS区(y=0↔1)和PMOS区(y=7↔8)分离，不可通过中心
  - Gate列全程竖向连通(y=0→8)，横向仅在Active外(y≠1,7)
  - S/D列无横向边

M2约束：仅第3行(y=3)和第5行(y=5)可以布线

============================================================
版图布局：

PMOS行 (y=7/8):
  VDD(x0)-PA(gA,x1)-p1(x2)-PB(gB,x3)-p2(x4)-PC(gC,x5)-Y(x6)-PD(gD,x7)-p3(x8)-PE(gE,x9)-p4(x10)-PF(gF,x11)-VDD(x12)

  串联路径1: VDD(x0)→PA→p1(x2)→PB→p2(x4)→PC→Y(x6)
  串联路径2: Y(x6)→PD→p3(x8)→PE→p4(x10)→PF→VDD(x12)
  两条串联路径并联，共同连接VDD与Y。

NMOS行 (y=0/1):（交叉布局，创造布线挑战）
  Y(x0)-NA(gA,x1)-mid(x2)-NB(gB,x3)-Y(x4)-NC(gC,x5)-mid(x6)-NE(gE,x7)-VSS(x8)-NF(gF,x9)-mid(x10)-ND(gD,x11)-VSS(x12)

  并联组1 (NA||NB||NC)连接Y与mid:
    NA: Y(x0)→mid(x2), NB: mid(x2)→Y(x4), NC: Y(x4)→mid(x6)
  并联组2 (ND||NE||NF)连接mid与VSS:
    NE: mid(x6)→VSS(x8), NF: VSS(x8)→mid(x10), ND: mid(x10)→VSS(x12)

============================================================
线网定义（10个）：

  Gate线网（6个）：
    net_A: 共栅 — PMOS(x=1) ↔ NMOS(x=1) 同列
    net_B: 共栅 — PMOS(x=3) ↔ NMOS(x=3) 同列
    net_C: 共栅 — PMOS(x=5) ↔ NMOS(x=5) 同列
    net_D: 交叉 — PMOS(x=7) ↔ NMOS(x=11) 跨4列
    net_E: 交叉 — PMOS(x=9) ↔ NMOS(x=7) 跨2列，与D交叉
    net_F: 交叉 — PMOS(x=11) ↔ NMOS(x=9) 跨2列，与D/E形成三路循环交叉

  S/D线网（4个）：
    net_Y:   output — PMOS(x=6,y=7) + NMOS(x=0,y=1) + NMOS(x=4,y=1)  [3端口]
    net_mid: internal — NMOS(x=2,y=1) + NMOS(x=6,y=1) + NMOS(x=10,y=1) [3端口]
    net_VDD: power — (x=0,y=8) + (x=12,y=8)
    net_VSS: ground — (x=8,y=0) + (x=12,y=0)
"""

import os
import pytest
import networkx as nx

from maze_router.net import Net
from maze_router.grid import RoutingGrid
from maze_router.spacing import SpacingManager
from maze_router.ripup import RipupManager
from maze_router.strategy import DefaultStrategy, CongestionAwareStrategy
from maze_router.visualizer import Visualizer


# === 网格常量 ===
COLS = 13       # x: 0..12  (6个晶体管 → 2*6+1=13列)
ROWS = 9        # y: 0..8

SD_COLS = {0, 2, 4, 6, 8, 10, 12}    # Source/Drain列
GATE_COLS = {1, 3, 5, 7, 9, 11}       # Gate列
ACTIVE_ROWS = {1, 7}                    # Active区（NMOS y=1, PMOS y=7）
NON_ACTIVE_ROWS = {0, 2, 3, 4, 5, 6, 8}
M2_ROWS = {3, 5}                        # M2仅在这两行布线

# cable_locs
GATE_CABLE_LOCS = frozenset(
    ("M0", x, y) for x in GATE_COLS for y in range(ROWS)
)
SD_CABLE_LOCS = frozenset(
    ("M0", x, y) for x in SD_COLS for y in range(ROWS)
)


def build_oai33_grid() -> RoutingGrid:
    """
    构建OAI33标准单元的三层布线网格（9行×13列）。

    M0: 受限连接（S/D竖向、Gate竖向+有限横向）
    M1: 完整矩形网格
    M2: 仅y=3和y=5行的水平轨道
    Via: M0↔M1全位置, M1↔M2仅y=3,5
    """
    G = nx.Graph()

    # ========== M0层 ==========
    for x in range(COLS):
        for y in range(ROWS):
            G.add_node(("M0", x, y))

    # M0竖向边
    for x in range(COLS):
        if x in SD_COLS:
            # S/D列：仅NMOS区(y=0↔1)和PMOS区(y=7↔8)，不可通过中心
            G.add_edge(("M0", x, 0), ("M0", x, 1), cost=1.0)
            G.add_edge(("M0", x, 7), ("M0", x, 8), cost=1.0)
        else:
            # Gate列：全程竖向连通 y=0→y=8
            for y in range(ROWS - 1):
                G.add_edge(("M0", x, y), ("M0", x, y + 1), cost=1.0)

    # M0横向边：仅Gate列之间，仅在非Active行
    gate_sorted = sorted(GATE_COLS)  # [1, 3, 5, 7, 9, 11]
    for i in range(len(gate_sorted) - 1):
        x1, x2 = gate_sorted[i], gate_sorted[i + 1]
        for y in NON_ACTIVE_ROWS:
            # 跨过中间的S/D列，代价=2
            G.add_edge(("M0", x1, y), ("M0", x2, y), cost=2.0)

    # ========== M1层：完整矩形网格 ==========
    for x in range(COLS):
        for y in range(ROWS):
            G.add_node(("M1", x, y))
    for x in range(COLS):
        for y in range(ROWS - 1):
            G.add_edge(("M1", x, y), ("M1", x, y + 1), cost=1.0)
    for y in range(ROWS):
        for x in range(COLS - 1):
            G.add_edge(("M1", x, y), ("M1", x + 1, y), cost=1.0)

    # ========== M2层：仅y=3和y=5行 ==========
    for x in range(COLS):
        for y in M2_ROWS:
            G.add_node(("M2", x, y))
    for y in M2_ROWS:
        for x in range(COLS - 1):
            G.add_edge(("M2", x, y), ("M2", x + 1, y), cost=1.0)
    # M2竖向：y=3↔y=5直连（跨2行，代价=2）
    for x in range(COLS):
        G.add_edge(("M2", x, 3), ("M2", x, 5), cost=2.0)

    # ========== Via连接 ==========
    # M0↔M1：所有位置
    for x in range(COLS):
        for y in range(ROWS):
            G.add_edge(("M0", x, y), ("M1", x, y), cost=2.0)
    # M1↔M2：仅M2存在的位置
    for x in range(COLS):
        for y in M2_ROWS:
            G.add_edge(("M1", x, y), ("M2", x, y), cost=2.0)

    return RoutingGrid(G)


def build_oai33_nets():
    """
    构建OAI33门的布线线网。

    OAI33: Y = ~((A+B+C) · (D+E+F))

    PMOS行(y=7/8):
      VDD(x0)-PA(gA,x1)-p1(x2)-PB(gB,x3)-p2(x4)-PC(gC,x5)-Y(x6)-
      PD(gD,x7)-p3(x8)-PE(gE,x9)-p4(x10)-PF(gF,x11)-VDD(x12)

    NMOS行(y=0/1): 交叉布局
      Y(x0)-NA(gA,x1)-mid(x2)-NB(gB,x3)-Y(x4)-NC(gC,x5)-mid(x6)-
      NE(gE,x7)-VSS(x8)-NF(gF,x9)-mid(x10)-ND(gD,x11)-VSS(x12)

    线网定义：
      Gate线网:
        net_A: 共栅 PMOS(x=1) ↔ NMOS(x=1)
        net_B: 共栅 PMOS(x=3) ↔ NMOS(x=3)
        net_C: 共栅 PMOS(x=5) ↔ NMOS(x=5)
        net_D: 交叉 PMOS(x=7) ↔ NMOS(x=11)
        net_E: 交叉 PMOS(x=9) ↔ NMOS(x=7)
        net_F: 交叉 PMOS(x=11) ↔ NMOS(x=9)
      S/D线网:
        net_Y:   PMOS(x=6,y=7) + NMOS(x=0,y=1) + NMOS(x=4,y=1)
        net_mid: NMOS(x=2,y=1) + NMOS(x=6,y=1) + NMOS(x=10,y=1)
        net_VDD: (x=0,y=8) + (x=12,y=8)
        net_VSS: (x=8,y=0) + (x=12,y=0)
    """
    gate_locs = set(GATE_CABLE_LOCS)
    sd_locs = set(SD_CABLE_LOCS)

    return [
        # === Gate线网 ===
        # 共栅: gate A — 同列(x=1)直接M0竖向连通
        Net("net_A", [("M0", 1, 7), ("M0", 1, 1)], cable_locs=gate_locs),
        # 共栅: gate B — 同列(x=3)
        Net("net_B", [("M0", 3, 7), ("M0", 3, 1)], cable_locs=gate_locs),
        # 共栅: gate C — 同列(x=5)
        Net("net_C", [("M0", 5, 7), ("M0", 5, 1)], cable_locs=gate_locs),
        # 交叉gate D — PMOS(x=7) ↔ NMOS(x=11)，跨4列
        Net("net_D", [("M0", 7, 7), ("M0", 11, 1)], cable_locs=gate_locs),
        # 交叉gate E — PMOS(x=9) ↔ NMOS(x=7)，与D交叉
        Net("net_E", [("M0", 9, 7), ("M0", 7, 1)], cable_locs=gate_locs),
        # 交叉gate F — PMOS(x=11) ↔ NMOS(x=9)，三路循环交叉
        Net("net_F", [("M0", 11, 7), ("M0", 9, 1)], cable_locs=gate_locs),

        # === S/D线网 ===
        # 输出Y: 3端口跨区域连接
        Net("net_Y", [("M0", 6, 7), ("M0", 0, 1), ("M0", 4, 1)],
            cable_locs=sd_locs),
        # 内部节点mid: 3端口，全在NMOS区
        Net("net_mid", [("M0", 2, 1), ("M0", 6, 1), ("M0", 10, 1)],
            cable_locs=sd_locs),
        # VDD电源
        Net("net_VDD", [("M0", 0, 8), ("M0", 12, 8)], cable_locs=sd_locs),
        # VSS地线
        Net("net_VSS", [("M0", 8, 0), ("M0", 12, 0)], cable_locs=sd_locs),
    ]


# ======================================================================
# 测试
# ======================================================================

class TestOAI33StandardCellRouting:
    """OAI33标准单元版图布线测试（9行网格）"""

    @pytest.fixture
    def grid(self):
        return build_oai33_grid()

    @pytest.fixture
    def spacing_mgr(self):
        return SpacingManager({"M0": 0, "M1": 0, "M2": 0})

    # ------------------------------------------------------------------
    # 网格结构验证
    # ------------------------------------------------------------------

    def test_grid_dimensions(self, grid):
        """验证网格尺寸: 13列 × 9行 × 3层"""
        m0_nodes = grid.get_nodes_on_layer("M0")
        m1_nodes = grid.get_nodes_on_layer("M1")
        m2_nodes = grid.get_nodes_on_layer("M2")

        assert len(m0_nodes) == COLS * ROWS, f"M0应有{COLS * ROWS}个节点"
        assert len(m1_nodes) == COLS * ROWS, f"M1应有{COLS * ROWS}个节点"
        assert len(m2_nodes) == COLS * len(M2_ROWS), \
            f"M2应有{COLS * len(M2_ROWS)}个节点"

    def test_grid_m0_sd_no_through_center(self, grid):
        """S/D列在M0上无法通过中心连通"""
        for x in SD_COLS:
            # NMOS区(y=0↔1)有边, y=1↔2无边
            assert grid.graph.has_edge(("M0", x, 0), ("M0", x, 1)), \
                f"S/D列 x={x} 应有 M0 y=0↔1 的边（NMOS区）"
            assert not grid.graph.has_edge(("M0", x, 1), ("M0", x, 2)), \
                f"S/D列 x={x} 不应有 M0 y=1↔2 的边"
            # PMOS区(y=7↔8)有边, y=6↔7无边
            assert grid.graph.has_edge(("M0", x, 7), ("M0", x, 8)), \
                f"S/D列 x={x} 应有 M0 y=7↔8 的边（PMOS区）"
            assert not grid.graph.has_edge(("M0", x, 6), ("M0", x, 7)), \
                f"S/D列 x={x} 不应有 M0 y=6↔7 的边"

    def test_grid_m0_gate_full_vertical(self, grid):
        """Gate列在M0上全程竖向连通(y=0→8)"""
        for x in GATE_COLS:
            for y in range(ROWS - 1):
                assert grid.graph.has_edge(("M0", x, y), ("M0", x, y + 1)), \
                    f"Gate列 x={x} 应有 M0 y={y}↔{y+1} 的边"

    def test_grid_m0_no_sd_horizontal(self, grid):
        """S/D列在M0上无横向边"""
        for x in SD_COLS:
            for y in range(ROWS):
                for dx in [-1, 1]:
                    nx_ = x + dx
                    if 0 <= nx_ < COLS:
                        assert not grid.graph.has_edge(
                            ("M0", x, y), ("M0", nx_, y)
                        ), f"S/D列 x={x} 不应有到 x={nx_} 的M0横向边"

    def test_grid_m0_gate_horizontal_only_outside_active(self, grid):
        """Gate列M0横向边仅在非Active行"""
        for y in range(ROWS):
            for i in range(len(sorted(GATE_COLS)) - 1):
                x1 = sorted(GATE_COLS)[i]
                x2 = sorted(GATE_COLS)[i + 1]
                has_edge = grid.graph.has_edge(("M0", x1, y), ("M0", x2, y))
                if y in ACTIVE_ROWS:
                    assert not has_edge, \
                        f"Active行 y={y} 不应有Gate横向边 x={x1}↔{x2}"
                else:
                    assert has_edge, \
                        f"非Active行 y={y} 应有Gate横向边 x={x1}↔{x2}"

    def test_grid_m2_only_rows_3_and_5(self, grid):
        """M2层只有y=3和y=5的节点"""
        m2_nodes = grid.get_nodes_on_layer("M2")
        for node in m2_nodes:
            assert node[2] in M2_ROWS, f"M2节点 {node} 不在允许的行(3或5)"

    def test_grid_via_m0_m1_all_positions(self, grid):
        """M0↔M1 via 存在于所有位置"""
        for x in range(COLS):
            for y in range(ROWS):
                assert grid.graph.has_edge(("M0", x, y), ("M1", x, y)), \
                    f"应有 M0↔M1 via 在 ({x},{y})"

    def test_grid_via_m1_m2_only_at_m2_rows(self, grid):
        """M1↔M2 via 仅在M2存在的行"""
        for x in range(COLS):
            for y in range(ROWS):
                has_via = grid.graph.has_edge(("M1", x, y), ("M2", x, y))
                if y in M2_ROWS:
                    assert has_via, \
                        f"应有 M1↔M2 via 在 ({x},{y})"
                else:
                    assert not has_via, \
                        f"不应有 M1↔M2 via 在 ({x},{y})"

    # ------------------------------------------------------------------
    # 共栅（Shared Gate）布线测试
    # ------------------------------------------------------------------

    def test_oai33_shared_gates(self, grid, spacing_mgr):
        """
        共栅测试：gate A/B/C 的PMOS/NMOS在同一Gate列，
        应该可以直接通过M0竖向连通，无需使用M1/M2。
        """
        strategy = DefaultStrategy(max_iterations=10)
        manager = RipupManager(grid, spacing_mgr, strategy)

        nets = [
            Net("net_A", [("M0", 1, 7), ("M0", 1, 1)],
                cable_locs=set(GATE_CABLE_LOCS)),
            Net("net_B", [("M0", 3, 7), ("M0", 3, 1)],
                cable_locs=set(GATE_CABLE_LOCS)),
            Net("net_C", [("M0", 5, 7), ("M0", 5, 1)],
                cable_locs=set(GATE_CABLE_LOCS)),
        ]

        solution = manager.run(nets)

        for net_name in ["net_A", "net_B", "net_C"]:
            assert solution.results[net_name].success, \
                f"共栅 {net_name} 应成功"
            # 验证仅使用M0
            for node in solution.results[net_name].routed_nodes:
                assert node[0] == "M0", \
                    f"共栅 {net_name} 应仅使用M0，但包含 {node}"

    # ------------------------------------------------------------------
    # 交叉Gate布线测试（三路循环交叉）
    # ------------------------------------------------------------------

    def test_oai33_cross_gates_pair(self, grid, spacing_mgr):
        """
        交叉Gate对测试：net_E(x=9→x=7)和net_F(x=11→x=9)。
        两者交叉，至少一个需要使用M1。
        """
        strategy = CongestionAwareStrategy(max_iterations=30)
        manager = RipupManager(grid, spacing_mgr, strategy)

        nets = [
            Net("net_E", [("M0", 9, 7), ("M0", 7, 1)],
                cable_locs=set(GATE_CABLE_LOCS)),
            Net("net_F", [("M0", 11, 7), ("M0", 9, 1)],
                cable_locs=set(GATE_CABLE_LOCS)),
        ]

        solution = manager.run(nets)
        assert solution.results["net_E"].success, "交叉gate net_E 应成功"
        assert solution.results["net_F"].success, "交叉gate net_F 应成功"

    def test_oai33_triple_cross_gates(self, grid, spacing_mgr):
        """
        三路循环交叉测试：D(x=7→11), E(x=9→7), F(x=11→9)。
        形成循环交叉 7→11, 9→7, 11→9，是高难度布线场景。
        至少需要M1甚至M2来解决交叉冲突。
        """
        strategy = CongestionAwareStrategy(max_iterations=50, congestion_weight=0.3)
        manager = RipupManager(grid, spacing_mgr, strategy)

        nets = [
            Net("net_D", [("M0", 7, 7), ("M0", 11, 1)],
                cable_locs=set(GATE_CABLE_LOCS)),
            Net("net_E", [("M0", 9, 7), ("M0", 7, 1)],
                cable_locs=set(GATE_CABLE_LOCS)),
            Net("net_F", [("M0", 11, 7), ("M0", 9, 1)],
                cable_locs=set(GATE_CABLE_LOCS)),
        ]

        solution = manager.run(nets)

        assert solution.results["net_D"].success, "交叉gate net_D 应成功"
        assert solution.results["net_E"].success, "交叉gate net_E 应成功"
        assert solution.results["net_F"].success, "交叉gate net_F 应成功"

        # 验证至少有一个线网使用了M1（M0无法解决循环交叉）
        any_uses_m1 = False
        for net_name in ["net_D", "net_E", "net_F"]:
            m1_nodes = [n for n in solution.results[net_name].routed_nodes
                        if n[0] == "M1"]
            if m1_nodes:
                any_uses_m1 = True
        assert any_uses_m1, "三路循环交叉中至少一个线网应使用M1"

    # ------------------------------------------------------------------
    # S/D多端口布线测试
    # ------------------------------------------------------------------

    def test_oai33_output_net(self, grid, spacing_mgr):
        """
        输出net_Y连接3个S/D端口：
        PMOS输出(x=6,y=7)和两个NMOS输出(x=0,y=1; x=4,y=1)。
        必须通过M1/M2进行跨区域连接。
        """
        strategy = CongestionAwareStrategy(max_iterations=30)
        manager = RipupManager(grid, spacing_mgr, strategy)

        nets = [
            Net("net_Y", [("M0", 6, 7), ("M0", 0, 1), ("M0", 4, 1)],
                cable_locs=set(SD_CABLE_LOCS)),
        ]

        solution = manager.run(nets)
        assert solution.results["net_Y"].success, "输出 net_Y 应成功"

        # 验证使用了M1（S/D在M0无法跨区域）
        m1_nodes = [n for n in solution.results["net_Y"].routed_nodes
                    if n[0] == "M1"]
        assert len(m1_nodes) > 0, "net_Y 应使用M1进行跨区域连接"

    def test_oai33_mid_net(self, grid, spacing_mgr):
        """
        内部节点net_mid连接3个NMOS S/D端口：
        (x=2,y=1), (x=6,y=1), (x=10,y=1)，全在NMOS区。
        S/D列M0无法横向连通，需通过M1连接。
        """
        strategy = CongestionAwareStrategy(max_iterations=30)
        manager = RipupManager(grid, spacing_mgr, strategy)

        nets = [
            Net("net_mid", [("M0", 2, 1), ("M0", 6, 1), ("M0", 10, 1)],
                cable_locs=set(SD_CABLE_LOCS)),
        ]

        solution = manager.run(nets)
        assert solution.results["net_mid"].success, "内部节点 net_mid 应成功"

        # 验证使用了M1
        m1_nodes = [n for n in solution.results["net_mid"].routed_nodes
                    if n[0] == "M1"]
        assert len(m1_nodes) > 0, "net_mid 应使用M1连接分散的S/D端口"

    def test_oai33_power_nets(self, grid, spacing_mgr):
        """VDD和VSS电源线网布线"""
        strategy = DefaultStrategy(max_iterations=20)
        manager = RipupManager(grid, spacing_mgr, strategy)

        nets = [
            Net("net_VDD", [("M0", 0, 8), ("M0", 12, 8)],
                cable_locs=set(SD_CABLE_LOCS)),
            Net("net_VSS", [("M0", 8, 0), ("M0", 12, 0)],
                cable_locs=set(SD_CABLE_LOCS)),
        ]

        solution = manager.run(nets)
        assert solution.results["net_VDD"].success, "VDD 应成功"
        assert solution.results["net_VSS"].success, "VSS 应成功"

    # ------------------------------------------------------------------
    # 完整OAI33布线测试
    # ------------------------------------------------------------------

    def test_oai33_full_routing(self, grid, spacing_mgr):
        """
        完整的OAI33标准单元布线：10个线网同时布线。
        保存SVG到 results/stdcell_oai33/。
        """
        strategy = CongestionAwareStrategy(max_iterations=80, congestion_weight=0.3)
        manager = RipupManager(grid, spacing_mgr, strategy)
        nets = build_oai33_nets()

        solution = manager.run(nets)

        # 保存SVG
        viz = Visualizer(grid)
        output_dir = os.path.join(
            os.path.dirname(__file__), "..", "results", "stdcell_oai33"
        )
        saved = viz.save_svg(
            solution, output_dir=output_dir, spacing_mgr=spacing_mgr,
            show_spacing=True, cell_size=50, padding=70
        )
        assert len(saved) == 3, f"应生成3个SVG文件，实际 {len(saved)}"

        # 共栅必须成功
        for net_name in ["net_A", "net_B", "net_C"]:
            assert solution.results[net_name].success, \
                f"共栅 {net_name} 必须成功"

        # 整体至少8/10布通
        assert solution.routed_count >= 8, \
            f"应至少布通8/10个线网，实际 {solution.routed_count}/10"

        # 打印摘要
        print(f"\n{'='*60}")
        print(f"OAI33 布线结果摘要: {solution.routed_count}/10 布通")
        print(f"{'='*60}")
        for name, result in sorted(solution.results.items()):
            status = "OK" if result.success else "FAIL"
            layers_used = sorted(set(n[0] for n in result.routed_nodes)) \
                if result.routed_nodes else []
            print(f"  {name:10s}: {status:4s}  cost={result.total_cost:6.1f}"
                  f"  layers={layers_used}")
        print(f"{'='*60}")
        print(f"总代价: {solution.total_cost:.1f}")

    # ------------------------------------------------------------------
    # 布线约束验证
    # ------------------------------------------------------------------

    def test_m0_constraints_satisfied(self, grid, spacing_mgr):
        """
        验证布线结果中M0层的约束：
        1. Gate线网在M0上只使用Gate列
        2. S/D线网在M0上只使用S/D列
        """
        strategy = CongestionAwareStrategy(max_iterations=80, congestion_weight=0.3)
        manager = RipupManager(grid, spacing_mgr, strategy)
        nets = build_oai33_nets()
        solution = manager.run(nets)

        gate_nets = {"net_A", "net_B", "net_C", "net_D", "net_E", "net_F"}
        sd_nets = {"net_Y", "net_mid", "net_VDD", "net_VSS"}

        for net_name, result in solution.results.items():
            if not result.success:
                continue
            for node in result.routed_nodes:
                if node[0] != "M0":
                    continue
                x = node[1]
                if net_name in gate_nets:
                    assert x in GATE_COLS, \
                        f"Gate线网 {net_name} 在M0使用了S/D列 x={x}: {node}"
                elif net_name in sd_nets:
                    assert x in SD_COLS, \
                        f"S/D线网 {net_name} 在M0使用了Gate列 x={x}: {node}"

    def test_m2_routing_constraint(self, grid, spacing_mgr):
        """验证布线结果中M2层只使用y=3和y=5"""
        strategy = CongestionAwareStrategy(max_iterations=80, congestion_weight=0.3)
        manager = RipupManager(grid, spacing_mgr, strategy)
        nets = build_oai33_nets()
        solution = manager.run(nets)

        for net_name, result in solution.results.items():
            if not result.success:
                continue
            for node in result.routed_nodes:
                if node[0] == "M2":
                    assert node[2] in M2_ROWS, \
                        f"线网 {net_name} 在M2使用了禁止行 y={node[2]}: {node}"

    def test_no_net_overlap(self, grid, spacing_mgr):
        """验证不同线网的布线路径不重叠（节点不共用）"""
        strategy = CongestionAwareStrategy(max_iterations=80, congestion_weight=0.3)
        manager = RipupManager(grid, spacing_mgr, strategy)
        nets = build_oai33_nets()
        solution = manager.run(nets)

        occupied = {}
        for net_name, result in solution.results.items():
            if not result.success:
                continue
            for node in result.routed_nodes:
                if node in occupied:
                    pytest.fail(
                        f"节点 {node} 被 {occupied[node]} 和 {net_name} 同时占用"
                    )
                occupied[node] = net_name


# ======================================================================
# Space=2 间距约束测试
# ======================================================================

class TestOAI33WithSpacing2:
    """
    OAI33标准单元布线测试 — 三层间距均为2。

    Space=2意味着每个已布线节点在Chebyshev距离2范围内（5×5区域）
    对其他线网形成禁区。在13×9的网格上，这极大压缩了可用布线资源，
    是对拆线重布和拥塞感知能力的严格考验。
    """

    @pytest.fixture
    def grid(self):
        return build_oai33_grid()

    @pytest.fixture
    def spacing_mgr_s2(self):
        return SpacingManager({"M0": 2, "M1": 2, "M2": 2})

    # ------------------------------------------------------------------
    # 共栅在spacing=2下的布线
    # ------------------------------------------------------------------

    def test_shared_gates_with_spacing2(self, grid, spacing_mgr_s2):
        """
        共栅测试（spacing=2）：gate A/B/C 同列直连。
        虽然共栅可M0直连，但spacing=2使得相邻Gate列之间会互相阻塞
        （Gate列间距=2，恰好等于space），布线顺序和策略至关重要。
        """
        strategy = CongestionAwareStrategy(max_iterations=30, congestion_weight=0.5)
        manager = RipupManager(grid, spacing_mgr_s2, strategy)

        nets = [
            Net("net_A", [("M0", 1, 7), ("M0", 1, 1)],
                cable_locs=set(GATE_CABLE_LOCS)),
            Net("net_B", [("M0", 3, 7), ("M0", 3, 1)],
                cable_locs=set(GATE_CABLE_LOCS)),
            Net("net_C", [("M0", 5, 7), ("M0", 5, 1)],
                cable_locs=set(GATE_CABLE_LOCS)),
        ]

        solution = manager.run(nets)

        routed = sum(1 for n in ["net_A", "net_B", "net_C"]
                     if solution.results[n].success)
        print(f"\n共栅 spacing=2: {routed}/3 布通")
        for name in ["net_A", "net_B", "net_C"]:
            r = solution.results[name]
            status = "OK" if r.success else "FAIL"
            layers = sorted(set(n[0] for n in r.routed_nodes)) \
                if r.routed_nodes else []
            print(f"  {name}: {status}  cost={r.total_cost:.1f}  layers={layers}")

        # spacing=2下Gate列间距=2，Chebyshev禁区5×5，共栅竖向路径互相阻塞严重
        # 至少1/3应布通
        assert routed >= 1, \
            f"共栅 spacing=2 至少应布通1/3，实际 {routed}/3"

    # ------------------------------------------------------------------
    # 三路交叉Gate在spacing=2下的布线
    # ------------------------------------------------------------------

    def test_cross_gates_with_spacing2(self, grid, spacing_mgr_s2):
        """
        三路循环交叉测试（spacing=2）。
        Spacing=2使得每条布线的禁区扩展到5×5，
        交叉线网几乎不可能在同一层共存，需要充分利用M1和M2。
        """
        strategy = CongestionAwareStrategy(max_iterations=80, congestion_weight=0.5)
        manager = RipupManager(grid, spacing_mgr_s2, strategy)

        nets = [
            Net("net_D", [("M0", 7, 7), ("M0", 11, 1)],
                cable_locs=set(GATE_CABLE_LOCS)),
            Net("net_E", [("M0", 9, 7), ("M0", 7, 1)],
                cable_locs=set(GATE_CABLE_LOCS)),
            Net("net_F", [("M0", 11, 7), ("M0", 9, 1)],
                cable_locs=set(GATE_CABLE_LOCS)),
        ]

        solution = manager.run(nets)

        routed = sum(1 for n in ["net_D", "net_E", "net_F"]
                     if solution.results[n].success)
        print(f"\n交叉gate spacing=2: {routed}/3 布通")
        for name in ["net_D", "net_E", "net_F"]:
            r = solution.results[name]
            status = "OK" if r.success else "FAIL"
            layers = sorted(set(n[0] for n in r.routed_nodes)) \
                if r.routed_nodes else []
            print(f"  {name}: {status}  cost={r.total_cost:.1f}  layers={layers}")

        # spacing=2下三路交叉极其困难，至少1个应布通
        assert routed >= 1, \
            f"交叉gate spacing=2 至少应布通1/3，实际 {routed}/3"

    # ------------------------------------------------------------------
    # 完整OAI33布线（spacing=2）
    # ------------------------------------------------------------------

    def test_oai33_full_routing_with_spacing2(self, grid, spacing_mgr_s2):
        """
        完整OAI33布线（spacing=2）：10个线网，三层间距均为2。
        这是极端压力测试，验证拆线重布引擎在高间距约束下的表现。
        保存SVG到 results/stdcell_oai33_s2/。
        """
        strategy = CongestionAwareStrategy(max_iterations=100, congestion_weight=0.5)
        manager = RipupManager(grid, spacing_mgr_s2, strategy)
        nets = build_oai33_nets()

        solution = manager.run(nets)

        # 保存SVG
        viz = Visualizer(grid)
        output_dir = os.path.join(
            os.path.dirname(__file__), "..", "results", "stdcell_oai33_s2"
        )
        saved = viz.save_svg(
            solution, output_dir=output_dir, spacing_mgr=spacing_mgr_s2,
            show_spacing=True, cell_size=50, padding=70
        )
        assert len(saved) == 3, f"应生成3个SVG文件，实际 {len(saved)}"

        # 打印详细摘要
        print(f"\n{'='*60}")
        print(f"OAI33 (spacing=2) 布线结果: {solution.routed_count}/10 布通")
        print(f"{'='*60}")
        for name, result in sorted(solution.results.items()):
            status = "OK" if result.success else "FAIL"
            layers_used = sorted(set(n[0] for n in result.routed_nodes)) \
                if result.routed_nodes else []
            node_count = len(result.routed_nodes)
            print(f"  {name:10s}: {status:4s}  cost={result.total_cost:6.1f}"
                  f"  nodes={node_count:3d}  layers={layers_used}")
        print(f"{'='*60}")
        print(f"总代价: {solution.total_cost:.1f}")
        print(f"布通率: {solution.routed_count}/10"
              f" ({solution.routed_count * 10:.0f}%)")
        if solution.failed_nets:
            print(f"未布通: {solution.failed_nets}")

        # spacing=2约束极其严格（Chebyshev 5×5禁区），13×9网格资源非常紧张
        # 完整10线网布通率预期较低，至少2/10布通即验证引擎基本功能
        assert solution.routed_count >= 2, \
            f"spacing=2 应至少布通2/10个线网，实际 {solution.routed_count}/10"

    # ------------------------------------------------------------------
    # 约束验证（spacing=2）
    # ------------------------------------------------------------------

    def test_spacing2_no_violation(self, grid, spacing_mgr_s2):
        """
        验证spacing=2下，已布通的不同线网之间的Chebyshev距离≥3
        （即任意两个不同线网的已布线节点在同层上的距离 > space=2）。
        """
        strategy = CongestionAwareStrategy(max_iterations=100, congestion_weight=0.5)
        manager = RipupManager(grid, spacing_mgr_s2, strategy)
        nets = build_oai33_nets()
        solution = manager.run(nets)

        # 按层收集每个已布通线网的节点
        layer_net_nodes = {}
        for net_name, result in solution.results.items():
            if not result.success:
                continue
            for node in result.routed_nodes:
                layer = node[0]
                if layer not in layer_net_nodes:
                    layer_net_nodes[layer] = {}
                if net_name not in layer_net_nodes[layer]:
                    layer_net_nodes[layer][net_name] = []
                layer_net_nodes[layer][net_name].append((node[1], node[2]))

        # 检查同层上不同线网间的Chebyshev距离
        space = 2
        violations = []
        for layer, net_nodes in layer_net_nodes.items():
            net_names = list(net_nodes.keys())
            for i in range(len(net_names)):
                for j in range(i + 1, len(net_names)):
                    n1, n2 = net_names[i], net_names[j]
                    for x1, y1 in net_nodes[n1]:
                        for x2, y2 in net_nodes[n2]:
                            chebyshev = max(abs(x1 - x2), abs(y1 - y2))
                            if chebyshev <= space:
                                violations.append(
                                    f"{layer}: {n1}({x1},{y1}) ↔ "
                                    f"{n2}({x2},{y2}) dist={chebyshev}"
                                )

        if violations:
            print(f"\nSpacing violations ({len(violations)}):")
            for v in violations[:10]:
                print(f"  {v}")
            if len(violations) > 10:
                print(f"  ... and {len(violations) - 10} more")

        assert len(violations) == 0, \
            f"发现 {len(violations)} 处spacing=2违规"
