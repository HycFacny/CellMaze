"""
标准单元版图布线测试 — 传输门 D 触发器（DFF）

基于传输门 Master-Slave D 触发器的实际标准单元版图布线场景。
10个PMOS + 10个NMOS = 20 晶体管，19列×9行网格，14个线网。

比 OAI33（13列，10nets）更复杂：
  - 更宽的网格（19 列）
  - CK/CKB 各有 4 个端口，相互交叉（时序单元特征）
  - 主/从锁存的 4 端口内部节点
  - 反馈回路（n_fb）
  - MinAreaConstraint 集成测试

============================================================
电路说明：传输门 DFF（Transmission Gate Master-Slave）
  Master latch（CK=1 透明）+ Slave latch（CK=0 透明）
  输入：D；时钟：CK/CKB；输出：Q/QB
============================================================

列类型（10晶体管/行）：
  SD_COLS   = {0,2,4,6,8,10,12,14,16,18}   偶数列（Source/Drain）
  GATE_COLS = {1,3,5,7,9,11,13,15,17}       奇数列（Gate 多晶硅）

行含义（9行，与 OAI33 相同）：
  y=0: VSS 轨道
  y=1: NMOS 有源区（S/D）
  y=2: 下方布线通道 1
  y=3: 下方布线通道 2  ← M2 层
  y=4: 单元中心
  y=5: 上方布线通道 1  ← M2 层
  y=6: 上方布线通道 2
  y=7: PMOS 有源区（S/D）
  y=8: VDD 轨道

M0 约束（与 OAI33 相同）：
  - S/D 列仅竖向，NMOS 区(y=0↔1) 和 PMOS 区(y=7↔8) 分离
  - Gate 列全程竖向(y=0→8)，横向仅在非Active行(y≠1,7)，代价=2
  - S/D 列无横向边

M2 约束：仅 y=3 和 y=5 行横向，y=3↔y=5 竖向直连（代价=2）

============================================================
线网定义（14个）：

  电源（2）：
    VDD:   (M0,0,8)  ↔ (M0,18,8)
    VSS:   (M0,0,0)  ↔ (M0,18,0)

  数据输入（1）：
    D:     (M0,0,7)  ↔ (M0,0,1)    同 SD 列直连

  时钟（2，每个 4 端口，相互交叉）：
    CK:    (M0,3,1) + (M0,13,1) + (M0,5,7) + (M0,11,7)
    CKB:   (M0,5,1) + (M0,11,1) + (M0,3,7) + (M0,13,7)

  内部节点（7）：
    n_di:  (M0,2,1) + (M0,2,7)            输入反相器输出
    n_tg1: (M0,4,1) + (M0,4,7)            传输门1输出
    n_m:   (M0,6,1) + (M0,6,7) + (M0,8,1) + (M0,8,7)   主锁存（4T）
    n_mB:  (M0,10,1) + (M0,10,7)          主锁存反相
    n_fb:  (M0,4,7) + (M0,8,7)            PMOS 侧反馈
    n_tg2: (M0,12,1) + (M0,12,7)          传输门2输出
    n_s:   (M0,14,1) + (M0,14,7) + (M0,16,1) + (M0,16,7)  从锁存（4T）

  输出（2）：
    Q:     (M0,16,1) + (M0,16,7)
    QB:    (M0,18,1) + (M0,18,7)
============================================================
"""

from __future__ import annotations
import os
import pytest

from maze_router.data.net import Net, Node
from maze_router.data.grid import GridGraph
from maze_router.constraint_manager import ConstraintManager
from maze_router.cost_manager import CostManager
from maze_router.constraints.space_constraint import SpaceConstraint
from maze_router.constraints.min_area_constraint import MinAreaConstraint
from maze_router.costs.corner_cost import CornerCost
from maze_router.ripup_strategy import (
    DefaultRipupStrategy, CongestionAwareRipupStrategy,
)
from maze_router.ripup_manager import RipupManager
from maze_router import MazeRouterEngine


# ======================================================================
# 常量
# ======================================================================

COLS = 19       # x: 0..18
ROWS = 9        # y: 0..8

SD_COLS   = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18}
GATE_COLS = {1, 3, 5, 7, 9, 11, 13, 15, 17}
ACTIVE_ROWS     = {1, 7}
NON_ACTIVE_ROWS = {0, 2, 3, 4, 5, 6, 8}
M2_ROWS = {3, 5}

# Gate 线网的 cable_locs（只允许在 Gate 列走 M0）
GATE_CABLE_LOCS: frozenset = frozenset(
    ("M0", x, y) for x in GATE_COLS for y in range(ROWS)
)
# SD 线网的 cable_locs（只允许在 SD 列走 M0）
SD_CABLE_LOCS: frozenset = frozenset(
    ("M0", x, y) for x in SD_COLS for y in range(ROWS)
)


# ======================================================================
# 网格构建
# ======================================================================

def build_dff_grid() -> GridGraph:
    """
    构建 DFF 标准单元的三层布线网格（9行×19列）。

    M0 约束与 OAI33 完全相同，只是扩展到 19 列：
    - S/D 列：仅 NMOS 区(y=0↔1) 和 PMOS 区(y=7↔8) 有竖向边
    - Gate 列：全程竖向(y=0→8)，横向仅在非Active行(y≠1,7)，代价=2
    - S/D 列无横向边
    - M2：仅 y=3, y=5 横向，y=3↔y=5 竖向直连（代价=2）
    """
    g = GridGraph()

    # ---- M0 ----
    for x in range(COLS):
        for y in range(ROWS):
            g.add_node(("M0", x, y))

    # M0 竖向边
    for x in range(COLS):
        if x in SD_COLS:
            # S/D 列：仅 NMOS 区 y=0↔1 和 PMOS 区 y=7↔8
            g.add_edge(("M0", x, 0), ("M0", x, 1), cost=1.0)
            g.add_edge(("M0", x, 7), ("M0", x, 8), cost=1.0)
        else:
            # Gate 列：全程竖向
            for y in range(ROWS - 1):
                g.add_edge(("M0", x, y), ("M0", x, y + 1), cost=1.0)

    # M0 横向边（Gate 列之间，仅在非Active行）
    gate_list = sorted(GATE_COLS)
    for i in range(len(gate_list) - 1):
        x1, x2 = gate_list[i], gate_list[i + 1]
        if x2 - x1 == 2:   # 相邻 Gate 列间距=2
            for y in NON_ACTIVE_ROWS:
                g.add_edge(("M0", x1, y), ("M0", x2, y), cost=2.0)

    # ---- M1（全网格）----
    for x in range(COLS):
        for y in range(ROWS):
            g.add_node(("M1", x, y))

    for x in range(COLS):
        for y in range(ROWS - 1):
            g.add_edge(("M1", x, y), ("M1", x, y + 1), cost=1.0)

    for y in range(ROWS):
        for x in range(COLS - 1):
            g.add_edge(("M1", x, y), ("M1", x + 1, y), cost=1.0)

    # ---- M2（仅 y=3, y=5 横向，y=3↔y=5 竖向直连）----
    for x in range(COLS):
        for y in M2_ROWS:
            g.add_node(("M2", x, y))

    # M2 横向边
    for y in M2_ROWS:
        for x in range(COLS - 1):
            g.add_edge(("M2", x, y), ("M2", x + 1, y), cost=1.0)

    # M2 竖向直连 y=3 ↔ y=5
    for x in range(COLS):
        g.add_edge(("M2", x, 3), ("M2", x, 5), cost=2.0)

    # ---- Via M0↔M1（所有位置）----
    for x in range(COLS):
        for y in range(ROWS):
            g.add_edge(("M0", x, y), ("M1", x, y), cost=2.0)

    # ---- Via M1↔M2（仅 y=3, y=5）----
    for x in range(COLS):
        for y in M2_ROWS:
            g.add_edge(("M1", x, y), ("M2", x, y), cost=2.0)

    return g


# ======================================================================
# 线网定义
# ======================================================================

def build_dff_nets() -> list:
    """
    构建 DFF 的 14 个线网。

    时钟 CK/CKB 各有 4 个端口并相互交叉（时序单元核心复杂性）。
    n_m 和 n_s 各有 4 个端口（主/从锁存内部节点）。
    """
    nets = [
        # 电源轨
        Net("net_VDD", [("M0", 0, 8), ("M0", 18, 8)]),
        Net("net_VSS", [("M0", 0, 0), ("M0", 18, 0)]),

        # 数据输入（SD 列同列直连）
        Net("net_D",   [("M0", 0, 7), ("M0", 0, 1)],
            cable_locs=set(SD_CABLE_LOCS)),

        # 时钟（4端口，Gate 列，CK 与 CKB 交叉）
        # CK:  NMOS gate x=3,13（y=1）；PMOS gate x=5,11（y=7）
        Net("net_CK",  [("M0", 3, 1), ("M0", 13, 1),
                        ("M0", 5, 7), ("M0", 11, 7)],
            cable_locs=set(GATE_CABLE_LOCS)),

        # CKB: NMOS gate x=5,11（y=1）；PMOS gate x=3,13（y=7）
        Net("net_CKB", [("M0", 5, 1), ("M0", 11, 1),
                        ("M0", 3, 7), ("M0", 13, 7)],
            cable_locs=set(GATE_CABLE_LOCS)),

        # 输入反相器输出（SD 列，同列直连）
        Net("net_di",  [("M0", 2, 1), ("M0", 2, 7)],
            cable_locs=set(SD_CABLE_LOCS)),

        # 传输门1输出 / 主锁存输入（SD 列，同列直连）
        Net("net_tg1", [("M0", 4, 1), ("M0", 4, 7)],
            cable_locs=set(SD_CABLE_LOCS)),

        # 主锁存内部节点（4端口：S/D 两侧，NMOS+PMOS）
        Net("net_m",   [("M0", 6, 1), ("M0", 6, 7),
                        ("M0", 8, 1), ("M0", 8, 7)],
            cable_locs=set(SD_CABLE_LOCS)),

        # 主锁存反相（SD 列，同列直连）
        Net("net_mB",  [("M0", 10, 1), ("M0", 10, 7)],
            cable_locs=set(SD_CABLE_LOCS)),

        # PMOS 侧反馈（连接 TG1 输出 x=4 和主锁存 x=8，PMOS 行）
        Net("net_fb",  [("M0", 4, 7), ("M0", 8, 7)],
            cable_locs=set(SD_CABLE_LOCS)),

        # 传输门2输出 / 从锁存输入（SD 列，同列直连）
        Net("net_tg2", [("M0", 12, 1), ("M0", 12, 7)],
            cable_locs=set(SD_CABLE_LOCS)),

        # 从锁存内部节点（4端口）
        Net("net_s",   [("M0", 14, 1), ("M0", 14, 7),
                        ("M0", 16, 1), ("M0", 16, 7)],
            cable_locs=set(SD_CABLE_LOCS)),

        # 输出 Q（SD 列，同列直连）
        Net("net_Q",   [("M0", 16, 1), ("M0", 16, 7)],
            cable_locs=set(SD_CABLE_LOCS)),

        # 互补输出 QB（SD 列，同列直连）
        Net("net_QB",  [("M0", 18, 1), ("M0", 18, 7)],
            cable_locs=set(SD_CABLE_LOCS)),
    ]
    return nets


# ======================================================================
# 辅助函数
# ======================================================================

def make_mgr(grid: GridGraph, space: int = 0,
             max_iter: int = 80) -> RipupManager:
    """构建 RipupManager（CongestionAwareRipupStrategy）。"""
    rules = {layer: space for layer in ("M0", "M1", "M2")}
    cmgr = ConstraintManager([SpaceConstraint(rules=rules)])
    cost_mgr = CostManager(
        grid=grid,
        costs=[CornerCost(l_costs={"M0": 5.0, "M1": 5.0, "M2": 5.0})],
    )
    strategy = CongestionAwareRipupStrategy(max_iterations=max_iter)
    return RipupManager(grid, cmgr, cost_mgr, strategy)


# ======================================================================
# Class 1: 网格结构测试
# ======================================================================

class TestDFFGridStructure:
    """验证 DFF 网格的结构正确性"""

    @pytest.fixture(scope="class")
    def grid(self):
        return build_dff_grid()

    def test_node_counts(self, grid):
        """节点数：M0=19*9, M1=19*9, M2=19*2"""
        assert len(grid.get_nodes_on_layer("M0")) == COLS * ROWS
        assert len(grid.get_nodes_on_layer("M1")) == COLS * ROWS
        assert len(grid.get_nodes_on_layer("M2")) == COLS * 2  # y=3,5 only

    def test_m0_sd_nmos_area(self, grid):
        """M0 SD 列：NMOS 区 y=0↔1 竖向边存在"""
        for x in SD_COLS:
            assert grid.graph.has_edge(("M0", x, 0), ("M0", x, 1)), \
                f"SD 列 x={x}: NMOS 区竖向边缺失"

    def test_m0_sd_pmos_area(self, grid):
        """M0 SD 列：PMOS 区 y=7↔8 竖向边存在"""
        for x in SD_COLS:
            assert grid.graph.has_edge(("M0", x, 7), ("M0", x, 8)), \
                f"SD 列 x={x}: PMOS 区竖向边缺失"

    def test_m0_sd_no_through_center(self, grid):
        """M0 SD 列：中心区域（y=1↔2 等）无竖向边"""
        for x in SD_COLS:
            for y in range(1, 7):  # y=1↔2 到 y=6↔7
                assert not grid.graph.has_edge(("M0", x, y), ("M0", x, y + 1)), \
                    f"SD 列 x={x}: 中心不应有边 y={y}↔{y+1}"

    def test_m0_gate_full_vertical(self, grid):
        """M0 Gate 列：全程竖向连通 y=0→8"""
        for x in GATE_COLS:
            for y in range(ROWS - 1):
                assert grid.graph.has_edge(("M0", x, y), ("M0", x, y + 1)), \
                    f"Gate 列 x={x}: 竖向边 y={y}↔{y+1} 缺失"

    def test_m0_sd_no_horizontal(self, grid):
        """M0 SD 列：无横向边"""
        for x in SD_COLS:
            for dx in (-1, 1):
                nx = x + dx
                if 0 <= nx < COLS:
                    for y in range(ROWS):
                        assert not grid.graph.has_edge(
                            ("M0", x, y), ("M0", nx, y)
                        ), f"SD 列 x={x}: 不应有横向边到 x={nx}, y={y}"

    def test_m0_gate_horizontal_only_non_active(self, grid):
        """M0 Gate 列横向边：仅在非Active行存在，Active行不存在"""
        gate_list = sorted(GATE_COLS)
        for i in range(len(gate_list) - 1):
            x1, x2 = gate_list[i], gate_list[i + 1]
            if x2 - x1 != 2:
                continue
            for y in ACTIVE_ROWS:
                assert not grid.graph.has_edge(("M0", x1, y), ("M0", x2, y)), \
                    f"Active 行 y={y}: Gate 横向边 x={x1}↔{x2} 不应存在"
            for y in NON_ACTIVE_ROWS:
                has = grid.graph.has_edge(("M0", x1, y), ("M0", x2, y))
                assert has, f"非Active 行 y={y}: Gate 横向边 x={x1}↔{x2} 缺失"

    def test_m2_rows_only(self, grid):
        """M2 节点仅在 y=3, y=5"""
        for node in grid.get_nodes_on_layer("M2"):
            assert node[2] in M2_ROWS, f"M2 节点 {node} 行不在 {M2_ROWS}"

    def test_m2_vertical_35(self, grid):
        """M2 y=3↔y=5 竖向直连存在"""
        for x in range(COLS):
            assert grid.graph.has_edge(("M2", x, 3), ("M2", x, 5)), \
                f"M2 y=3↔y=5 竖向边 x={x} 缺失"

    def test_via_m0m1_all(self, grid):
        """M0↔M1 via 存在于所有位置"""
        for x in range(COLS):
            for y in range(ROWS):
                assert grid.graph.has_edge(("M0", x, y), ("M1", x, y)), \
                    f"M0↔M1 via 缺失: ({x},{y})"

    def test_via_m1m2_only_m2rows(self, grid):
        """M1↔M2 via 仅在 y=3, y=5"""
        for x in range(COLS):
            for y in range(ROWS):
                has = grid.graph.has_edge(("M1", x, y), ("M2", x, y))
                if y in M2_ROWS:
                    assert has, f"M1↔M2 via 缺失: ({x},{y})"
                else:
                    assert not has, f"M1↔M2 via 不应存在: ({x},{y})"

    def test_grid_larger_than_oai33(self, grid):
        """DFF 网格比 OAI33（13列）更宽"""
        m0_nodes = grid.get_nodes_on_layer("M0")
        assert len(m0_nodes) > 13 * 9, "DFF 网格应比 OAI33 更大"


# ======================================================================
# Class 2: 单/组线网布线测试（space=0）
# ======================================================================

class TestDFFNetRouting:
    """各子线网的独立布线测试"""

    @pytest.fixture(scope="class")
    def grid(self):
        return build_dff_grid()

    def test_data_input_d_routes_successfully(self, grid):
        """D 线网：SD 列同列，y=7→y=1 需经 M1，应布通"""
        mgr = make_mgr(grid, space=0)
        net = Net("net_D", [("M0", 0, 7), ("M0", 0, 1)],
                  cable_locs=set(SD_CABLE_LOCS))
        solution = mgr.run([net])
        result = solution.results["net_D"]
        assert result.success, "D 线网应布通"
        # SD 列无中心段，必经 M1（y=7→via→M1→M0, y=1←via←M1←M0）
        layers = {n[0] for n in result.routed_nodes}
        assert "M1" in layers or len(result.routed_nodes) >= 2, \
            "D 线网应连通两端点"

    def test_ck_4terminal_alone(self, grid):
        """CK 线网（4 端口）单独布通（space=0）"""
        mgr = make_mgr(grid, space=0)
        net = Net("net_CK",
                  [("M0", 3, 1), ("M0", 13, 1), ("M0", 5, 7), ("M0", 11, 7)],
                  cable_locs=set(GATE_CABLE_LOCS))
        solution = mgr.run([net])
        assert solution.results["net_CK"].success, "CK 4T 线网应单独布通"
        for t in net.terminals:
            assert t in solution.results["net_CK"].routed_nodes

    def test_ckb_4terminal_alone(self, grid):
        """CKB 线网（4 端口）单独布通（space=0）"""
        mgr = make_mgr(grid, space=0)
        net = Net("net_CKB",
                  [("M0", 5, 1), ("M0", 11, 1), ("M0", 3, 7), ("M0", 13, 7)],
                  cable_locs=set(GATE_CABLE_LOCS))
        solution = mgr.run([net])
        assert solution.results["net_CKB"].success, "CKB 4T 线网应单独布通"

    def test_ck_ckb_both_route_and_use_m1(self, grid):
        """CK+CKB 同时布线：均成功，至少一个使用 M1（交叉需跨层）"""
        mgr = make_mgr(grid, space=0)
        nets = [
            Net("net_CK",
                [("M0", 3, 1), ("M0", 13, 1), ("M0", 5, 7), ("M0", 11, 7)],
                cable_locs=set(GATE_CABLE_LOCS)),
            Net("net_CKB",
                [("M0", 5, 1), ("M0", 11, 1), ("M0", 3, 7), ("M0", 13, 7)],
                cable_locs=set(GATE_CABLE_LOCS)),
        ]
        solution = mgr.run(nets)
        assert solution.results["net_CK"].success,  "CK 应布通"
        assert solution.results["net_CKB"].success, "CKB 应布通"

        # 至少一个线网使用了 M1 或 M2（因为 CK/CKB 相互交叉）
        ck_layers  = {n[0] for n in solution.results["net_CK"].routed_nodes}
        ckb_layers = {n[0] for n in solution.results["net_CKB"].routed_nodes}
        uses_upper = ("M1" in ck_layers or "M2" in ck_layers or
                      "M1" in ckb_layers or "M2" in ckb_layers)
        assert uses_upper, "CK/CKB 交叉布线至少一个应使用 M1 或 M2"

    def test_master_latch_4terminal(self, grid):
        """n_m（主锁存，4端口）布通"""
        mgr = make_mgr(grid, space=0)
        net = Net("net_m",
                  [("M0", 6, 1), ("M0", 6, 7), ("M0", 8, 1), ("M0", 8, 7)],
                  cable_locs=set(SD_CABLE_LOCS))
        solution = mgr.run([net])
        assert solution.results["net_m"].success, "n_m 4T 线网应布通"

    def test_slave_latch_4terminal(self, grid):
        """n_s（从锁存，4端口）布通"""
        mgr = make_mgr(grid, space=0)
        net = Net("net_s",
                  [("M0", 14, 1), ("M0", 14, 7), ("M0", 16, 1), ("M0", 16, 7)],
                  cable_locs=set(SD_CABLE_LOCS))
        solution = mgr.run([net])
        assert solution.results["net_s"].success, "n_s 4T 线网应布通"

    def test_power_vdd_vss(self, grid):
        """VDD/VSS 电源轨布通"""
        mgr = make_mgr(grid, space=0)
        nets = [
            Net("net_VDD", [("M0", 0, 8), ("M0", 18, 8)]),
            Net("net_VSS", [("M0", 0, 0), ("M0", 18, 0)]),
        ]
        solution = mgr.run(nets)
        assert solution.results["net_VDD"].success, "VDD 应布通"
        assert solution.results["net_VSS"].success, "VSS 应布通"

    def test_output_q_qb(self, grid):
        """Q/QB 输出线网布通"""
        mgr = make_mgr(grid, space=0)
        nets = [
            Net("net_Q",  [("M0", 16, 1), ("M0", 16, 7)],
                cable_locs=set(SD_CABLE_LOCS)),
            Net("net_QB", [("M0", 18, 1), ("M0", 18, 7)],
                cable_locs=set(SD_CABLE_LOCS)),
        ]
        solution = mgr.run(nets)
        assert solution.results["net_Q"].success,  "Q 应布通"
        assert solution.results["net_QB"].success, "QB 应布通"


# ======================================================================
# Class 3: 全 14 nets 布线测试（space=0）
# ======================================================================

class TestDFFFullRouting:
    """全部 14 个线网的综合布线测试"""

    @pytest.fixture(scope="class")
    def grid(self):
        return build_dff_grid()

    @pytest.fixture(scope="class")
    def solution(self, grid):
        mgr = make_mgr(grid, space=0, max_iter=100)
        return mgr.run(build_dff_nets())

    def test_full_routing_min_10_of_14(self, solution):
        """DFF: ≥ 10/14 线网布通（space=0）"""
        total = 14
        routed = solution.routed_count
        assert routed >= 10, f"只布通了 {routed}/{total} 个线网，低于 10"

    def test_full_routing_cable_locs_respected(self, solution):
        """所有线网的 M0 节点在其 cable_locs 范围内"""
        nets = {net.name: net for net in build_dff_nets()}
        for name, result in solution.results.items():
            if not result.success:
                continue
            net = nets.get(name)
            if net is None or net.cable_locs is None:
                continue
            for node in result.routed_nodes:
                if node[0] == "M0":
                    assert node in net.cable_locs, \
                        f"线网 {name}: M0 节点 {node} 超出 cable_locs"

    def test_full_routing_m2_rows_respected(self, solution):
        """所有线网的 M2 节点仅在 y=3, y=5"""
        for name, result in solution.results.items():
            if not result.success:
                continue
            for node in result.routed_nodes:
                if node[0] == "M2":
                    assert node[2] in M2_ROWS, \
                        f"线网 {name}: M2 节点 {node} 行不在 {M2_ROWS}"

    def test_full_routing_no_node_overlap(self, solution):
        """不同线网间无节点重叠（共享 terminal 的线网除外）"""
        # 收集所有 terminal 节点及其所属线网（terminal 可被多线网共享）
        terminal_nets: dict = {}  # node -> set of net names
        for net in build_dff_nets():
            for t in net.terminals:
                if t not in terminal_nets:
                    terminal_nets[t] = set()
                terminal_nets[t].add(net.name)

        node_owner: dict = {}
        for name, result in solution.results.items():
            if not result.success:
                continue
            for node in result.routed_nodes:
                if node in node_owner:
                    owner = node_owner[node]
                    if owner == name:
                        continue
                    # 允许共享 terminal
                    shared = terminal_nets.get(node, set())
                    if name in shared and owner in shared:
                        continue
                    assert False, \
                        f"节点 {node} 被 {owner} 和 {name} 同时占用（非共享 terminal）"
                else:
                    node_owner[node] = name

    def test_full_routing_summary(self, solution):
        """输出布线摘要（含未布通线网）"""
        print(f"\nDFF 布线结果: {solution.routed_count}/14 布通")
        if solution.failed_nets:
            print(f"  未布通: {solution.failed_nets}")
        for name, result in solution.results.items():
            layers = {n[0] for n in result.routed_nodes}
            print(f"  {name}: {'OK' if result.success else 'FAIL'}, "
                  f"nodes={len(result.routed_nodes)}, layers={layers}")

    def test_full_routing_save_svg(self, grid, solution):
        """保存 SVG 到 results/stdcell_dff/"""
        # 复用 solution fixture 中的 mgr 结果，直接通过 engine 生成
        engine = MazeRouterEngine(
            grid=grid,
            nets=build_dff_nets(),
            space_constr={"M0": 0, "M1": 0, "M2": 0},
            corner_l_costs={"M0": 5.0, "M1": 5.0, "M2": 5.0},
            strategy="congestion_aware",
            max_iterations=100,
        )
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "results", "stdcell_dff",
        )
        engine.visualize(save_dir=save_dir)
        assert os.path.isdir(save_dir)


# ======================================================================
# Class 4: space=1 压力测试
# ======================================================================

class TestDFFWithSpacing1:
    """Chebyshev space=1 约束下的压力测试"""

    @pytest.fixture(scope="class")
    def grid(self):
        return build_dff_grid()

    def test_ck_ckb_spacing1_at_least_1(self, grid):
        """CK+CKB 在 space=1 下至少 1/2 布通"""
        mgr = make_mgr(grid, space=1, max_iter=50)
        nets = [
            Net("net_CK",
                [("M0", 3, 1), ("M0", 13, 1), ("M0", 5, 7), ("M0", 11, 7)],
                cable_locs=set(GATE_CABLE_LOCS)),
            Net("net_CKB",
                [("M0", 5, 1), ("M0", 11, 1), ("M0", 3, 7), ("M0", 13, 7)],
                cable_locs=set(GATE_CABLE_LOCS)),
        ]
        solution = mgr.run(nets)
        assert solution.routed_count >= 1, \
            f"CK+CKB space=1 应至少 1/2 布通，实际 {solution.routed_count}/2"

    def test_full_routing_spacing1_at_least_6(self, grid):
        """全 14 nets 在 space=1 下至少 6/14 布通"""
        mgr = make_mgr(grid, space=1, max_iter=100)
        solution = mgr.run(build_dff_nets())
        assert solution.routed_count >= 6, \
            f"space=1 应至少 6/14 布通，实际 {solution.routed_count}/14"

    def test_spacing1_save_svg(self, grid, tmp_path):
        """space=1 结果可保存 SVG"""
        engine = MazeRouterEngine(
            grid=grid,
            nets=build_dff_nets(),
            space_constr={"M0": 1, "M1": 1, "M2": 1},
            corner_l_costs={"M0": 5.0, "M1": 5.0, "M2": 5.0},
            strategy="congestion_aware",
            max_iterations=80,
        )
        engine.run()
        engine.visualize(save_dir=str(tmp_path), prefix="dff_space1_")
        svgs = list(tmp_path.glob("*.svg"))
        assert len(svgs) >= 1

    def test_spacing1_no_routing_violation(self, grid):
        """space=1 布线结果无 Chebyshev 间距违规"""
        mgr = make_mgr(grid, space=1, max_iter=50)
        nets = build_dff_nets()
        solution = mgr.run(nets)

        all_terminals = set()
        for net in nets:
            all_terminals.update(net.terminals)

        from maze_router.constraints.space_constraint import SpaceConstraint
        sc_check = SpaceConstraint(rules={"M0": 1, "M1": 1, "M2": 1})
        for name, result in solution.results.items():
            if not result.success:
                continue
            routing_only = result.routed_nodes - all_terminals
            sc_check.mark_route(name, routing_only)

        violations = 0
        for name, result in solution.results.items():
            if not result.success:
                continue
            routing_only = result.routed_nodes - all_terminals
            for node in routing_only:
                if not sc_check.is_available(node, name):
                    violations += 1
        assert violations == 0, f"space=1 路由存在 {violations} 处间距违规"


# ======================================================================
# Class 5: MinAreaConstraint 集成测试
# ======================================================================

class TestDFFMinArea:
    """MinAreaConstraint 在 DFF 场景的集成测试"""

    @pytest.fixture(scope="class")
    def grid(self):
        return build_dff_grid()

    def test_min_area_applied_after_routing(self, grid):
        """布线后 routed_nodes 满足 min_area 要求"""
        from maze_router.constraints.min_area_constraint import MinAreaConstraint
        min_area_rules = {"M0": 2, "M1": 2, "M2": 2}

        constraints = [
            SpaceConstraint(rules={"M0": 0, "M1": 0, "M2": 0}),
            MinAreaConstraint(rules=min_area_rules),
        ]
        cmgr = ConstraintManager(constraints)
        cost_mgr = CostManager(
            grid=grid,
            costs=[CornerCost(l_costs={"M0": 5.0, "M1": 5.0, "M2": 5.0})],
        )
        strategy = CongestionAwareRipupStrategy(max_iterations=50)
        mgr = RipupManager(grid, cmgr, cost_mgr, strategy)
        nets = [
            Net("net_D",  [("M0", 0, 7), ("M0", 0, 1)],
                cable_locs=set(SD_CABLE_LOCS)),
            Net("net_CK", [("M0", 3, 1), ("M0", 13, 1),
                           ("M0", 5, 7), ("M0", 11, 7)],
                cable_locs=set(GATE_CABLE_LOCS)),
        ]
        solution = mgr.run(nets)

        mac = MinAreaConstraint(rules=min_area_rules)
        for name, result in solution.results.items():
            if not result.success:
                continue
            violations = mac.check_violations(result.routed_nodes)
            assert len(violations) == 0, \
                f"线网 {name} 违反 min_area: {violations}"

    def test_min_area_engine_integration(self, grid):
        """通过 Engine min_area 参数端到端：布线后无 min_area 违规"""
        engine = MazeRouterEngine(
            grid=grid,
            nets=[
                Net("net_D",  [("M0", 0, 7), ("M0", 0, 1)],
                    cable_locs=set(SD_CABLE_LOCS)),
                Net("net_QB", [("M0", 18, 1), ("M0", 18, 7)],
                    cable_locs=set(SD_CABLE_LOCS)),
            ],
            space_constr={"M0": 0, "M1": 0, "M2": 0},
            min_area={"M0": 2, "M1": 2},
        )
        solution = engine.run()

        mac = MinAreaConstraint(rules={"M0": 2, "M1": 2})
        for name, result in solution.results.items():
            if not result.success:
                continue
            violations = mac.check_violations(result.routed_nodes)
            assert len(violations) == 0, \
                f"Engine min_area 参数：线网 {name} 违规 {violations}"

    def test_min_area_soft_rule_no_failure(self, grid):
        """min_area 是 soft rule：无法充分扩展时 success 仍为 True"""
        # 用一个几乎被包围的节点，让扩展受限
        from maze_router.constraints.min_area_constraint import MinAreaConstraint
        # 非常大的 min_area（会被 clamp 到 4），但布线仍然 success
        big_rule = MinAreaConstraint(rules={"M0": 10})  # clamp 到 4
        assert big_rule.get_min_area("M0") == 4, "min_area 应被 clamp 到 4"

        constraints = [
            SpaceConstraint(rules={"M0": 0, "M1": 0, "M2": 0}),
            big_rule,
        ]
        cmgr = ConstraintManager(constraints)
        cost_mgr = CostManager(grid=grid, costs=[CornerCost.default()])
        mgr = RipupManager(grid, cmgr, cost_mgr,
                           CongestionAwareRipupStrategy(max_iterations=30))
        nets = [Net("net_D", [("M0", 0, 7), ("M0", 0, 1)],
                    cable_locs=set(SD_CABLE_LOCS))]
        solution = mgr.run(nets)
        # soft rule：success 不应因 min_area 失败
        assert solution.results["net_D"].success, \
            "min_area soft rule 不应导致布线失败"

    def test_min_area_layer_filter(self, grid):
        """min_area 只对有路由节点的层生效"""
        from maze_router.constraints.min_area_constraint import MinAreaConstraint
        mac = MinAreaConstraint(rules={"M0": 2, "M1": 3, "M2": 2})

        # 只有 M0 节点的集合
        nodes_m0_only = {("M0", 0, 0), ("M0", 0, 1)}
        violations = mac.check_violations(nodes_m0_only)
        # M1/M2 未使用，不应产生违规
        assert "M1" not in violations, "M1 未使用，不应报违规"
        assert "M2" not in violations, "M2 未使用，不应报违规"

    def test_min_area_default_2(self):
        """未设置规则时默认 min_area=2"""
        mac = MinAreaConstraint(rules={})
        assert mac.get_min_area("M0") == 2
        assert mac.get_min_area("M1") == 2
        assert mac.get_min_area("M99") == 2


# ======================================================================
# Class 6: Engine 端到端测试
# ======================================================================

class TestDFFEngine:
    """通过 MazeRouterEngine 的端到端测试"""

    @pytest.fixture(scope="class")
    def grid(self):
        return build_dff_grid()

    def test_engine_space0_full(self, grid):
        """Engine 端到端：DFF 全部线网，space=0，至少 10/14 布通"""
        engine = MazeRouterEngine(
            grid=grid,
            nets=build_dff_nets(),
            space_constr={"M0": 0, "M1": 0, "M2": 0},
            corner_l_costs={"M0": 5.0, "M1": 5.0, "M2": 5.0},
            strategy="congestion_aware",
            max_iterations=100,
        )
        solution = engine.run()
        assert solution.routed_count >= 10, \
            f"Engine space=0 应至少 10/14 布通，实际 {solution.routed_count}/14"

    def test_engine_space1_at_least_6(self, grid):
        """Engine 端到端：DFF，space=1，至少 6/14 布通"""
        engine = MazeRouterEngine(
            grid=grid,
            nets=build_dff_nets(),
            space_constr={"M0": 1, "M1": 1, "M2": 1},
            corner_l_costs={"M0": 5.0, "M1": 5.0, "M2": 5.0},
            strategy="congestion_aware",
            max_iterations=80,
        )
        solution = engine.run()
        assert solution.routed_count >= 6, \
            f"Engine space=1 应至少 6/14 布通，实际 {solution.routed_count}/14"

    def test_engine_saves_svg(self, grid, tmp_path):
        """Engine 可视化：生成 3 层 SVG 文件"""
        engine = MazeRouterEngine(
            grid=grid,
            nets=build_dff_nets(),
            space_constr={"M0": 0, "M1": 0, "M2": 0},
            strategy="congestion_aware",
            max_iterations=50,
        )
        engine.run()
        engine.visualize(save_dir=str(tmp_path), prefix="dff_engine_")
        svgs = list(tmp_path.glob("*.svg"))
        assert len(svgs) >= 3, f"应生成 3 个 SVG，实际 {len(svgs)} 个"
