"""
标准单元版图布线测试

基于AOI22门（Y = ~((A·B) + (C·D))）的实际标准单元版图布线场景。
4个PMOS + 4个NMOS晶体管，严格遵守标准单元布线规则。

网格规格：9列(x=0..8) × 7行(y=0..6) × 3层(M0,M1,M2)

列类型（4个晶体管/行）：
  偶数列(0,2,4,6,8) = Source/Drain 扩散列
  奇数列(1,3,5,7)   = Gate 多晶硅列

行含义：
  y=0: VSS轨道（下边界）
  y=1: NMOS有源区（S/D扩散）
  y=2: 下方布线通道
  y=3: 单元中心
  y=4: 上方布线通道
  y=5: PMOS有源区（S/D扩散）
  y=6: VDD轨道（上边界）

M0约束：
  - S/D列仅竖向布线，NMOS区(y=0↔1)和PMOS区(y=5↔6)分离，不可通过中心
  - Gate列全程竖向连通(y=0→6)，横向仅在Active外(y≠1,5)
  - S/D列无横向边

M2约束：仅第2行(y=2)和第4行(y=4)可以布线

AOI22布局：
  PMOS: VDD-PA(A)-p1-PB(B)-Y-PC(C)-p2-PD(D)-VDD
  NMOS: Y-NA(A)-n1-NB(B)-VSS-ND(D)-n2-NC(C)-Y
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
COLS = 9       # x: 0..8
ROWS = 7       # y: 0..6

SD_COLS = {0, 2, 4, 6, 8}      # Source/Drain列
GATE_COLS = {1, 3, 5, 7}        # Gate列
ACTIVE_ROWS = {1, 5}             # Active区（NMOS y=1, PMOS y=5）
NON_ACTIVE_ROWS = {0, 2, 3, 4, 6}
M2_ROWS = {2, 4}                 # M2仅在这两行布线

# cable_locs
GATE_CABLE_LOCS = frozenset(
    ("M0", x, y) for x in GATE_COLS for y in range(ROWS)
)
SD_CABLE_LOCS = frozenset(
    ("M0", x, y) for x in SD_COLS for y in range(ROWS)
)


def build_standard_cell_grid() -> RoutingGrid:
    """
    构建标准单元的三层布线网格。

    M0: 受限连接（S/D竖向、Gate竖向+有限横向）
    M1: 完整矩形网格
    M2: 仅y=2和y=4行的水平轨道
    Via: M0↔M1全位置, M1↔M2仅y=2,4
    """
    G = nx.Graph()

    # ========== M0层 ==========
    for x in range(COLS):
        for y in range(ROWS):
            G.add_node(("M0", x, y))

    # M0竖向边
    for x in range(COLS):
        if x in SD_COLS:
            # S/D列：仅NMOS区(y=0↔1)和PMOS区(y=5↔6)，不可通过中心
            G.add_edge(("M0", x, 0), ("M0", x, 1), cost=1.0)
            G.add_edge(("M0", x, 5), ("M0", x, 6), cost=1.0)
        else:
            # Gate列：全程竖向连通 y=0→y=6
            for y in range(ROWS - 1):
                G.add_edge(("M0", x, y), ("M0", x, y + 1), cost=1.0)

    # M0横向边：仅Gate列之间，仅在非Active行
    gate_sorted = sorted(GATE_COLS)  # [1, 3, 5, 7]
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

    # ========== M2层：仅y=2和y=4行 ==========
    for x in range(COLS):
        for y in M2_ROWS:
            G.add_node(("M2", x, y))
    for y in M2_ROWS:
        for x in range(COLS - 1):
            G.add_edge(("M2", x, y), ("M2", x + 1, y), cost=1.0)
    # M2竖向：y=2↔y=4直连（跨2行，代价=2）
    for x in range(COLS):
        G.add_edge(("M2", x, 2), ("M2", x, 4), cost=2.0)

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


def build_aoi22_nets():
    """
    构建AOI22门的布线线网。

    AOI22: Y = ~((A·B) + (C·D))

    PMOS行(y=5/6): VDD(x0)—PA(gA,x1)—p1(x2)—PB(gB,x3)—Y(x4)—PC(gC,x5)—p2(x6)—PD(gD,x7)—VDD(x8)
    NMOS行(y=0/1): Y(x0)—NA(gA,x1)—n1(x2)—NB(gB,x3)—VSS(x4)—ND(gD,x5)—n2(x6)—NC(gC,x7)—Y(x8)

    线网定义：
      net_A: gate A 共栅 (PMOS x=1, NMOS x=1 → 同列)
      net_B: gate B 共栅 (PMOS x=3, NMOS x=3 → 同列)
      net_C: gate C 交叉 (PMOS x=5, NMOS x=7 → 不同列)
      net_D: gate D 交叉 (PMOS x=7, NMOS x=5 → 不同列，与net_C交叉)
      net_Y: output   (PMOS x=4,y=5; NMOS x=0,y=1 和 x=8,y=1)
      net_VDD: power   (x=0,y=6 和 x=8,y=6)
    """
    gate_locs = set(GATE_CABLE_LOCS)
    sd_locs = set(SD_CABLE_LOCS)

    return [
        # 共栅: gate A — PMOS和NMOS在同一Gate列(x=1)，可直接M0竖向连通
        Net("net_A", [("M0", 1, 5), ("M0", 1, 1)], cable_locs=gate_locs),
        # 共栅: gate B — 同一Gate列(x=3)
        Net("net_B", [("M0", 3, 5), ("M0", 3, 1)], cable_locs=gate_locs),
        # 交叉gate: C — PMOS在x=5, NMOS在x=7，需横向布线
        Net("net_C", [("M0", 5, 5), ("M0", 7, 1)], cable_locs=gate_locs),
        # 交叉gate: D — PMOS在x=7, NMOS在x=5，与net_C交叉
        Net("net_D", [("M0", 7, 5), ("M0", 5, 1)], cable_locs=gate_locs),
        # 输出Y: S/D多端口连接（需M1/M2布线）
        Net("net_Y", [("M0", 4, 5), ("M0", 0, 1), ("M0", 8, 1)], cable_locs=sd_locs),
        # VDD电源: S/D水平连接（需M1）
        Net("net_VDD", [("M0", 0, 6), ("M0", 8, 6)], cable_locs=sd_locs),
    ]


# ======================================================================
# 测试
# ======================================================================

class TestStandardCellRouting:
    """标准单元版图布线测试"""

    @pytest.fixture
    def grid(self):
        return build_standard_cell_grid()

    @pytest.fixture
    def spacing_mgr(self):
        return SpacingManager({"M0": 0, "M1": 0, "M2": 0})

    # ------------------------------------------------------------------
    # 网格结构验证
    # ------------------------------------------------------------------

    def test_grid_m0_sd_no_through_center(self, grid):
        """S/D列在M0上无法通过中心连通"""
        for x in SD_COLS:
            # y=1和y=5之间不应有路径仅通过M0
            # 检查y=1和y=2之间无M0边
            assert not grid.graph.has_edge(("M0", x, 1), ("M0", x, 2)), \
                f"S/D列 x={x} 不应有 M0 y=1↔2 的边"

    def test_grid_m0_gate_full_vertical(self, grid):
        """Gate列在M0上全程竖向连通"""
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
                        assert not grid.graph.has_edge(("M0", x, y), ("M0", nx_, y)), \
                            f"S/D列 x={x} 不应有到 x={nx_} 的M0横向边"

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

    def test_grid_m2_only_rows_2_and_4(self, grid):
        """M2层只有y=2和y=4的节点"""
        m2_nodes = grid.get_nodes_on_layer("M2")
        for node in m2_nodes:
            assert node[2] in M2_ROWS, f"M2节点 {node} 不在允许的行(2或4)"

    # ------------------------------------------------------------------
    # 共栅（Shared Gate）布线测试
    # ------------------------------------------------------------------

    def test_aoi22_shared_gates(self, grid, spacing_mgr):
        """
        共栅测试：gate A和gate B的PMOS/NMOS在同一Gate列，
        应该可以直接通过M0竖向连通，无需使用M1/M2。
        """
        strategy = DefaultStrategy(max_iterations=10)
        manager = RipupManager(grid, spacing_mgr, strategy)

        nets = [
            Net("net_A", [("M0", 1, 5), ("M0", 1, 1)], cable_locs=set(GATE_CABLE_LOCS)),
            Net("net_B", [("M0", 3, 5), ("M0", 3, 1)], cable_locs=set(GATE_CABLE_LOCS)),
        ]

        solution = manager.run(nets)

        assert solution.results["net_A"].success, "共栅 net_A 应成功"
        assert solution.results["net_B"].success, "共栅 net_B 应成功"

        # 验证仅使用M0（无via到M1/M2）
        for net_name in ["net_A", "net_B"]:
            for node in solution.results[net_name].routed_nodes:
                assert node[0] == "M0", \
                    f"共栅 {net_name} 应仅使用M0，但包含 {node}"

    # ------------------------------------------------------------------
    # 交叉Gate布线测试
    # ------------------------------------------------------------------

    def test_aoi22_cross_gates(self, grid, spacing_mgr):
        """
        交叉Gate测试：net_C(x=5→x=7)和net_D(x=7→x=5)形成交叉，
        至少一个需要使用M1。两者都应布通。
        """
        strategy = CongestionAwareStrategy(max_iterations=30)
        manager = RipupManager(grid, spacing_mgr, strategy)

        nets = [
            Net("net_C", [("M0", 5, 5), ("M0", 7, 1)], cable_locs=set(GATE_CABLE_LOCS)),
            Net("net_D", [("M0", 7, 5), ("M0", 5, 1)], cable_locs=set(GATE_CABLE_LOCS)),
        ]

        solution = manager.run(nets)

        assert solution.results["net_C"].success, "交叉gate net_C 应成功"
        assert solution.results["net_D"].success, "交叉gate net_D 应成功"

    # ------------------------------------------------------------------
    # S/D多端口布线测试
    # ------------------------------------------------------------------

    def test_aoi22_output_net(self, grid, spacing_mgr):
        """
        输出net_Y连接3个S/D端口：PMOS输出(x=4,y=5)和两个NMOS输出(x=0,y=1; x=8,y=1)。
        必须通过M1/M2进行跨区域连接。
        """
        strategy = CongestionAwareStrategy(max_iterations=30)
        manager = RipupManager(grid, spacing_mgr, strategy)

        nets = [
            Net("net_Y", [("M0", 4, 5), ("M0", 0, 1), ("M0", 8, 1)],
                cable_locs=set(SD_CABLE_LOCS)),
        ]

        solution = manager.run(nets)
        assert solution.results["net_Y"].success, "输出 net_Y 应成功"

        # 验证使用了M1（S/D在M0无法跨区域）
        m1_nodes = [n for n in solution.results["net_Y"].routed_nodes if n[0] == "M1"]
        assert len(m1_nodes) > 0, "net_Y 应使用M1进行跨区域连接"

    # ------------------------------------------------------------------
    # 完整AOI22布线测试
    # ------------------------------------------------------------------

    def test_aoi22_full_routing(self, grid, spacing_mgr):
        """
        完整的AOI22标准单元布线：6个线网同时布线。
        保存SVG到 results/stdcell_aoi22/。
        """
        strategy = CongestionAwareStrategy(max_iterations=50, congestion_weight=0.3)
        manager = RipupManager(grid, spacing_mgr, strategy)
        nets = build_aoi22_nets()

        solution = manager.run(nets)

        # 保存SVG
        viz = Visualizer(grid)
        output_dir = os.path.join(os.path.dirname(__file__), "..", "results", "stdcell_aoi22")
        saved = viz.save_svg(solution, output_dir=output_dir, spacing_mgr=spacing_mgr,
                             show_spacing=True, cell_size=50, padding=70)
        assert len(saved) == 3, f"应生成3个SVG文件，实际 {len(saved)}"

        # 共栅必须成功
        assert solution.results["net_A"].success, "共栅 net_A 必须成功"
        assert solution.results["net_B"].success, "共栅 net_B 必须成功"

        # 整体至少5/6布通
        assert solution.routed_count >= 5, \
            f"应至少布通5/6个线网，实际 {solution.routed_count}/6"

        # 打印摘要
        for name, result in sorted(solution.results.items()):
            status = "OK" if result.success else "FAIL"
            layers_used = sorted(set(n[0] for n in result.routed_nodes)) if result.routed_nodes else []
            print(f"  {name}: {status} cost={result.total_cost:.1f} layers={layers_used}")

    # ------------------------------------------------------------------
    # 布线约束验证
    # ------------------------------------------------------------------

    def test_m0_constraints_satisfied(self, grid, spacing_mgr):
        """
        验证布线结果中M0层的约束：
        1. Gate线网在M0上只使用Gate列
        2. S/D线网在M0上只使用S/D列
        """
        strategy = CongestionAwareStrategy(max_iterations=50, congestion_weight=0.3)
        manager = RipupManager(grid, spacing_mgr, strategy)
        nets = build_aoi22_nets()
        solution = manager.run(nets)

        gate_nets = {"net_A", "net_B", "net_C", "net_D"}
        sd_nets = {"net_Y", "net_VDD"}

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
        """验证布线结果中M2层只使用y=2和y=4"""
        strategy = CongestionAwareStrategy(max_iterations=50, congestion_weight=0.3)
        manager = RipupManager(grid, spacing_mgr, strategy)
        nets = build_aoi22_nets()
        solution = manager.run(nets)

        for net_name, result in solution.results.items():
            if not result.success:
                continue
            for node in result.routed_nodes:
                if node[0] == "M2":
                    assert node[2] in M2_ROWS, \
                        f"线网 {net_name} 在M2使用了禁止行 y={node[2]}: {node}"
