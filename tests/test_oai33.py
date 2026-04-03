"""
标准单元版图布线测试 — OAI33 门（精确版）

基于 OAI33 门（Y = ~((A+B+C)·(D+E+F))）的实际标准单元版图布线场景。
6个PMOS + 6个NMOS晶体管，严格遵守标准单元布线规则。

网格规格：13列(x=0..12) × 9行(y=0..8) × 3层(M0,M1,M2)

============================================================
电路说明：OAI33 = OR-AND-INVERT(3-3)
  Y = ~((A+B+C) · (D+E+F))
  NMOS下拉：(NA||NB||NC) 串联 (ND||NE||NF)
  PMOS上拉：(PA-PB-PC串联) 并联 (PD-PE-PF串联)
============================================================

列类型（6晶体管/行）：
  偶数列 (0,2,4,6,8,10,12) = Source/Drain 扩散列
  奇数列 (1,3,5,7,9,11)    = Gate 多晶硅列

行含义（9行）：
  y=0: VSS 轨道（下边界）
  y=1: NMOS 有源区（S/D 扩散）
  y=2: 下方布线通道 1
  y=3: 下方布线通道 2
  y=4: 单元中心
  y=5: 上方布线通道 1
  y=6: 上方布线通道 2
  y=7: PMOS 有源区（S/D 扩散）
  y=8: VDD 轨道（上边界）

M0 约束：
  - S/D 列仅竖向布线，NMOS 区(y=0↔1)和 PMOS 区(y=7↔8)分离，不可通过中心
  - Gate 列全程竖向连通(y=0→8)，横向仅在非Active行(y≠1,7)
  - S/D 列无横向边
  - Gate 横向边跨 2 格（相邻 Gate 列间距=2，代价=2）

M2 约束：仅 y=3 和 y=5 行可横向布线，y=3↔y=5 竖向直连（代价=2）

============================================================
线网定义（10个）：

  Gate 线网（6个）：
    net_A: 共栅 — PMOS(x=1,y=7) ↔ NMOS(x=1,y=1)  同列直连
    net_B: 共栅 — PMOS(x=3,y=7) ↔ NMOS(x=3,y=1)
    net_C: 共栅 — PMOS(x=5,y=7) ↔ NMOS(x=5,y=1)
    net_D: 交叉 — PMOS(x=7,y=7) ↔ NMOS(x=11,y=1)  跨4列
    net_E: 交叉 — PMOS(x=9,y=7) ↔ NMOS(x=7,y=1)   与D交叉
    net_F: 交叉 — PMOS(x=11,y=7)↔ NMOS(x=9,y=1)   D/E/F三路循环交叉

  S/D 线网（4个）：
    net_Y:   output  — PMOS(x=6,y=7) + NMOS(x=0,y=1) + NMOS(x=4,y=1)
    net_mid: internal— NMOS(x=2,y=1) + NMOS(x=6,y=1) + NMOS(x=10,y=1)
    net_VDD: power   — (x=0,y=8) + (x=12,y=8)
    net_VSS: ground  — (x=8,y=0) + (x=12,y=0)
============================================================
"""

from __future__ import annotations
import os
import pytest

from maze_router.data.net import Net, Node, PinSpec
from maze_router.data.grid import GridGraph
from maze_router.constraint_manager import ConstraintManager
from maze_router.cost_manager import CostManager
from maze_router.constraints.space_constraint import SpaceConstraint
from maze_router.costs.corner_cost import CornerCost
from maze_router.ripup_strategy import (
    DefaultRipupStrategy, CongestionAwareRipupStrategy,
)
from maze_router.ripup_manager import RipupManager
from maze_router import MazeRouterEngine


# ======================================================================
# 常量
# ======================================================================

COLS = 13       # x: 0..12
ROWS = 9        # y: 0..8

SD_COLS   = {0, 2, 4, 6, 8, 10, 12}
GATE_COLS = {1, 3, 5, 7, 9, 11}
ACTIVE_ROWS     = {1, 7}
NON_ACTIVE_ROWS = {0, 2, 3, 4, 5, 6, 8}
M2_ROWS = {3, 5}

# 行类型 y 范围（单行标准单元 ROWS=9）
NMOS_ROW = 0   # i=0
PMOS_ROW = 1   # i=1
ROW_TYPE_Y_RANGES = {
    NMOS_ROW: (0, 3),   # y=0..3
    PMOS_ROW: (5, 8),   # y=5..8
}
# 每个 SD Terminal 的 active 覆盖节点数要求
SD_ACTIVE_N = 2   # N = active_size + 1，活跃行默认覆盖 2 个 M1 节点

# 虚拟起点层名（已注册为 virtual_node_layers）
EXT_LAYER = "EXT"

# M0 cable locs 集合（用于 cable_locs 约束）
GATE_CABLE_LOCS = frozenset(("M0", x, y) for x in GATE_COLS for y in range(ROWS))
SD_CABLE_LOCS   = frozenset(("M0", x, y) for x in SD_COLS   for y in range(ROWS))


def _sd_virtual_node(j: int, i: int) -> Node:
    """返回列 j、行类型 i 对应的 EXT 虚拟起点。"""
    return (EXT_LAYER, j, i)


def _add_sd_virtual_to_grid(
    grid: GridGraph,
    j: int,
    i: int,
    N: int = SD_ACTIVE_N,
) -> Node:
    """
    向网格添加 SD 虚拟起点并连边到 M0 可行起始位置，返回虚拟节点。

    可行 M0 起始位置：在行类型 i 的 y_range 内，令 [y_start, y_start+N-1] ⊆ y_range
    的所有 y_start 对应的 ("M0", j, y_start)。
    """
    y_min, y_max = ROW_TYPE_Y_RANGES[i]
    vnode = _sd_virtual_node(j, i)
    grid.add_node(vnode)
    for y_start in range(y_min, y_max - N + 2):
        m0_node = ("M0", j, y_start)
        if grid.is_valid_node(m0_node):
            grid.add_edge(vnode, m0_node, cost=0.0)
    return vnode


# ======================================================================
# 网格构建
# ======================================================================

def build_oai33_grid() -> GridGraph:
    """
    构建 OAI33 标准单元的三层布线网格（9行×13列）。

    S/D 列 M0 规则（新模型）：
    - Active 行（y=1 NMOS，y=7 PMOS）在 S/D 列上 **无** M0 竖向边，
      仅允许通过 via 连接到 M1（符合工艺 active coverage 规则）。
    - S/D 列在 active 行以外也无 M0 竖向边，整列通过 via 上 M1 后水平走线。
    - 虚拟起点（EXT 层）由 build_oai33_nets() 在创建线网时注入。
    Gate 列：全程竖向(y=0→8)，横向仅在非 Active 行(y≠1,7)。
    S/D 列无任何横向边。
    M2：仅 y=3, y=5 横向，y=3↔y=5 竖向直连。
    """
    g = GridGraph()

    # ---- M0 ----
    for x in range(COLS):
        for y in range(ROWS):
            g.add_node(("M0", x, y))

    # M0 竖向边
    for x in range(COLS):
        if x in SD_COLS:
            # S/D 列：无任何 M0 竖向边（active 行只能通过 via 连 M1）
            pass
        else:
            # Gate 列：全程竖向 y=0→8
            for y in range(ROWS - 1):
                g.add_edge(("M0", x, y), ("M0", x, y + 1), cost=1.0)

    # M0 横向边：仅相邻 Gate 列之间，仅在非 Active 行
    gate_sorted = sorted(GATE_COLS)   # [1, 3, 5, 7, 9, 11]
    for i in range(len(gate_sorted) - 1):
        x1, x2 = gate_sorted[i], gate_sorted[i + 1]
        for y in NON_ACTIVE_ROWS:
            g.add_edge(("M0", x1, y), ("M0", x2, y), cost=2.0)

    # ---- M1：完整矩形网格 ----
    for x in range(COLS):
        for y in range(ROWS):
            g.add_node(("M1", x, y))
    for x in range(COLS):
        for y in range(ROWS - 1):
            g.add_edge(("M1", x, y), ("M1", x, y + 1), cost=1.0)
    for y in range(ROWS):
        for x in range(COLS - 1):
            g.add_edge(("M1", x, y), ("M1", x + 1, y), cost=1.0)

    # ---- M2：仅 y=3 和 y=5 ----
    for x in range(COLS):
        for y in M2_ROWS:
            g.add_node(("M2", x, y))
    for y in M2_ROWS:
        for x in range(COLS - 1):
            g.add_edge(("M2", x, y), ("M2", x + 1, y), cost=1.0)
    # M2 y=3↔y=5 竖向直连（代价=2）
    for x in range(COLS):
        g.add_edge(("M2", x, 3), ("M2", x, 5), cost=2.0)

    # ---- Via M0↔M1：所有位置 ----
    for x in range(COLS):
        for y in range(ROWS):
            g.add_edge(("M0", x, y), ("M1", x, y), cost=2.0)

    # ---- Via M1↔M2：仅 M2 存在位置 ----
    for x in range(COLS):
        for y in M2_ROWS:
            g.add_edge(("M1", x, y), ("M2", x, y), cost=2.0)

    # 注册虚拟层（S/D 起点由 build_oai33_nets 注入）
    g.register_virtual_layer(EXT_LAYER)

    return g


# ======================================================================
# 线网构建
# ======================================================================

# SD 线网各 Terminal 的 active 覆盖规格：{(net_name, j, i): N}
OAI33_ACTIVE_RULES = {
    ("net_Y",   6,  PMOS_ROW): SD_ACTIVE_N,
    ("net_Y",   0,  NMOS_ROW): SD_ACTIVE_N,
    ("net_Y",   4,  NMOS_ROW): SD_ACTIVE_N,
    ("net_mid", 2,  NMOS_ROW): SD_ACTIVE_N,
    ("net_mid", 6,  NMOS_ROW): SD_ACTIVE_N,
    ("net_mid", 10, NMOS_ROW): SD_ACTIVE_N,
}


def build_oai33_nets(grid: GridGraph):
    """
    构建 OAI33 门的 10 个布线线网。

    S/D 线网中位于 active 行（y=1 NMOS，y=7 PMOS）的 Terminal 改为虚拟起点，
    虚拟节点及其到 M0 可行起始位置的边会就地注入 grid。

    参数:
        grid: 由 build_oai33_grid() 构建的网格（会被原地修改注入虚拟节点）

    返回:
        (nets, active_rules)
        active_rules: Dict[(net_name, j, i) → N]，传给 ActiveOccupancyConstraint
    """
    gate_locs = set(GATE_CABLE_LOCS)
    sd_locs   = set(SD_CABLE_LOCS)

    # 注入 SD 虚拟起点
    v_Y_pmos = _add_sd_virtual_to_grid(grid, j=6,  i=PMOS_ROW)
    v_Y_n0   = _add_sd_virtual_to_grid(grid, j=0,  i=NMOS_ROW)
    v_Y_n4   = _add_sd_virtual_to_grid(grid, j=4,  i=NMOS_ROW)
    v_mid_2  = _add_sd_virtual_to_grid(grid, j=2,  i=NMOS_ROW)
    v_mid_6  = _add_sd_virtual_to_grid(grid, j=6,  i=NMOS_ROW)
    v_mid_10 = _add_sd_virtual_to_grid(grid, j=10, i=NMOS_ROW)

    nets = [
        # === Gate 线网（直接 M0 端口，gate 列不受 active 规则约束）===
        Net("net_A", [("M0", 1,  7), ("M0", 1,  1)], cable_locs=gate_locs),
        Net("net_B", [("M0", 3,  7), ("M0", 3,  1)], cable_locs=gate_locs),
        Net("net_C", [("M0", 5,  7), ("M0", 5,  1)], cable_locs=gate_locs),
        Net("net_D", [("M0", 7,  7), ("M0", 11, 1)], cable_locs=gate_locs),   # 交叉
        Net("net_E", [("M0", 9,  7), ("M0", 7,  1)], cable_locs=gate_locs),   # 交叉
        Net("net_F", [("M0", 11, 7), ("M0", 9,  1)], cable_locs=gate_locs),   # 交叉

        # === S/D 线网 ===
        # net_Y: PMOS drain(6,7)→虚拟, NMOS drains (0,1)+(4,1)→虚拟
        Net("net_Y",   [v_Y_pmos, v_Y_n0, v_Y_n4],       cable_locs=sd_locs),
        # net_mid: NMOS 内部节点 (2,1)+(6,1)+(10,1)→虚拟
        Net("net_mid", [v_mid_2, v_mid_6, v_mid_10],       cable_locs=sd_locs),
        # net_VDD: PMOS 电源轨（y=8 非 active 行，保持直接端口）
        Net("net_VDD", [("M0", 0, 8), ("M0", 12, 8)],     cable_locs=sd_locs),
        # net_VSS: NMOS 地轨（y=0 非 active 行，保持直接端口）
        Net("net_VSS", [("M0", 8, 0), ("M0", 12, 0)],     cable_locs=sd_locs),
    ]

    return nets, OAI33_ACTIVE_RULES


def make_mgr(grid: GridGraph, space: int = 0, max_iter: int = 50):
    """构建 RipupManager（space=0 表示无间距约束）。"""
    constraint_mgr = ConstraintManager([
        SpaceConstraint({"M0": space, "M1": space, "M2": space})
    ])
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

class TestOAI33GridStructure:
    """严格验证 OAI33 网格的拓扑结构"""

    @pytest.fixture(scope="class")
    def grid(self):
        return build_oai33_grid()

    def test_node_counts(self, grid):
        """验证各层节点数量"""
        assert len(grid.get_nodes_on_layer("M0")) == COLS * ROWS
        assert len(grid.get_nodes_on_layer("M1")) == COLS * ROWS
        assert len(grid.get_nodes_on_layer("M2")) == COLS * len(M2_ROWS)

    def test_m0_sd_nmos_area(self, grid):
        """S/D 列在 active 行(y=0↔1)无 M0 竖向边（仅允许 via 到 M1）"""
        for x in SD_COLS:
            assert not grid.graph.has_edge(("M0", x, 0), ("M0", x, 1)), \
                f"S/D x={x}: M0 y=0↔1 边不应存在（active-only-via 规则）"

    def test_m0_sd_pmos_area(self, grid):
        """S/D 列在 active 行(y=7↔8)无 M0 竖向边（仅允许 via 到 M1）"""
        for x in SD_COLS:
            assert not grid.graph.has_edge(("M0", x, 7), ("M0", x, 8)), \
                f"S/D x={x}: M0 y=7↔8 边不应存在（active-only-via 规则）"

    def test_m0_sd_no_through_center(self, grid):
        """S/D 列 M0 无任何竖向边（active 行 via-only，通道行由 M1 走线）"""
        for x in SD_COLS:
            for y in range(ROWS - 1):
                assert not grid.graph.has_edge(("M0", x, y), ("M0", x, y + 1)), \
                    f"S/D x={x}: M0 y={y}↔{y+1} 边不应存在"

    def test_m0_gate_full_vertical(self, grid):
        """Gate 列 M0 全程竖向连通 (y=0→8)"""
        for x in GATE_COLS:
            for y in range(ROWS - 1):
                assert grid.graph.has_edge(("M0", x, y), ("M0", x, y + 1)), \
                    f"Gate x={x}: M0 y={y}↔{y+1} 边缺失"

    def test_m0_sd_no_horizontal(self, grid):
        """S/D 列 M0 无横向边"""
        for x in SD_COLS:
            for y in range(ROWS):
                for dx in (-1, +1):
                    nx_ = x + dx
                    if 0 <= nx_ < COLS:
                        assert not grid.graph.has_edge(("M0", x, y), ("M0", nx_, y)), \
                            f"S/D x={x} 不应有 M0 横向边到 x={nx_} (y={y})"

    def test_m0_gate_horizontal_only_non_active(self, grid):
        """Gate 列 M0 横向边仅在非 Active 行 (y≠1,7)"""
        gate_sorted = sorted(GATE_COLS)
        for i in range(len(gate_sorted) - 1):
            x1, x2 = gate_sorted[i], gate_sorted[i + 1]
            for y in range(ROWS):
                has = grid.graph.has_edge(("M0", x1, y), ("M0", x2, y))
                if y in ACTIVE_ROWS:
                    assert not has, f"Active 行 y={y}: Gate 横向边 x={x1}↔{x2} 不应存在"
                else:
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


# ======================================================================
# Class 2: 单线网/分组布线测试（space=0）
# ======================================================================

class TestOAI33NetRouting:
    """各子线网的独立布线测试"""

    @pytest.fixture(scope="class")
    def grid(self):
        return build_oai33_grid()

    def test_shared_gates_abc_m0_only(self, grid):
        """
        共栅 A/B/C：PMOS/NMOS 在同一 Gate 列，应在 M0 直接竖向连通，
        不需要跳 M1/M2。
        """
        mgr = make_mgr(grid, space=0)
        nets = [
            Net("net_A", [("M0", 1, 7), ("M0", 1, 1)], cable_locs=set(GATE_CABLE_LOCS)),
            Net("net_B", [("M0", 3, 7), ("M0", 3, 1)], cable_locs=set(GATE_CABLE_LOCS)),
            Net("net_C", [("M0", 5, 7), ("M0", 5, 1)], cable_locs=set(GATE_CABLE_LOCS)),
        ]
        solution = mgr.run(nets)

        for name in ["net_A", "net_B", "net_C"]:
            assert solution.results[name].success, f"共栅 {name} 应布通"
            # 共栅应仅使用 M0（直接竖向路径）
            for node in solution.results[name].routed_nodes:
                assert node[0] == "M0", \
                    f"共栅 {name} 不应使用 {node[0]} 层节点 {node}"

    def test_shared_gates_abc_straight_no_corner(self, grid):
        """共栅 A/B/C 路径为直线，折角数应为 0"""
        mgr = make_mgr(grid, space=0)
        nets = [
            Net("net_A", [("M0", 1, 7), ("M0", 1, 1)], cable_locs=set(GATE_CABLE_LOCS)),
            Net("net_B", [("M0", 3, 7), ("M0", 3, 1)], cable_locs=set(GATE_CABLE_LOCS)),
            Net("net_C", [("M0", 5, 7), ("M0", 5, 1)], cable_locs=set(GATE_CABLE_LOCS)),
        ]
        solution = mgr.run(nets)

        from maze_router.maze_router_algo import move_dir_code, DIR_NONE
        from collections import defaultdict
        for name in ["net_A", "net_B", "net_C"]:
            result = solution.results[name]
            if not result.success:
                continue
            adj = defaultdict(list)
            for u, v in result.routed_edges:
                adj[u].append(v)
                adj[v].append(u)
            corners = 0
            for u in adj:
                neighbors = adj[u]
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        v, w = neighbors[i], neighbors[j]
                        if u[0] != v[0] or u[0] != w[0]:
                            continue   # via 不算折角
                        d1 = move_dir_code(v, u)
                        d2 = move_dir_code(u, w)
                        if d1 != DIR_NONE and d2 != DIR_NONE and d1 != d2:
                            corners += 1
            assert corners == 0, f"共栅 {name} 折角数应为 0，实际 {corners}"

    def test_cross_gate_ef_both_routed(self, grid):
        """
        交叉 Gate E(x=9→x=7) 和 F(x=11→x=9)：两线网交叉，
        至少需要一条使用 M1 绕行。
        """
        mgr = make_mgr(grid, space=0, max_iter=30)
        nets = [
            Net("net_E", [("M0", 9, 7), ("M0", 7, 1)],  cable_locs=set(GATE_CABLE_LOCS)),
            Net("net_F", [("M0", 11, 7), ("M0", 9, 1)], cable_locs=set(GATE_CABLE_LOCS)),
        ]
        solution = mgr.run(nets)
        assert solution.results["net_E"].success, "交叉 net_E 应布通"
        assert solution.results["net_F"].success, "交叉 net_F 应布通"

    def test_triple_cross_gate_def_all_routed(self, grid):
        """
        三路循环交叉 D(x=7→11), E(x=9→7), F(x=11→9)：
        形成循环交叉，必须至少有一条线使用 M1/M2 绕行。
        """
        mgr = make_mgr(grid, space=0, max_iter=80)
        nets = [
            Net("net_D", [("M0", 7, 7), ("M0", 11, 1)], cable_locs=set(GATE_CABLE_LOCS)),
            Net("net_E", [("M0", 9, 7), ("M0", 7, 1)],  cable_locs=set(GATE_CABLE_LOCS)),
            Net("net_F", [("M0", 11, 7), ("M0", 9, 1)], cable_locs=set(GATE_CABLE_LOCS)),
        ]
        solution = mgr.run(nets)
        assert solution.results["net_D"].success, "三叉 net_D 应布通"
        assert solution.results["net_E"].success, "三叉 net_E 应布通"
        assert solution.results["net_F"].success, "三叉 net_F 应布通"

        # 验证循环交叉中至少有一个线网使用了 M1
        any_m1 = any(
            any(n[0] == "M1" for n in solution.results[name].routed_nodes)
            for name in ["net_D", "net_E", "net_F"]
        )
        assert any_m1, "三路循环交叉中至少一个线网应使用 M1"

    def test_output_net_y_three_terminals(self, grid):
        """
        输出线网 net_Y（3 端口）：PMOS(x=6,y=7) + NMOS(x=0,y=1) + NMOS(x=4,y=1)，
        S/D 在 M0 无法横向连通，必须借助 M1 跨区域。
        """
        mgr = make_mgr(grid, space=0, max_iter=30)
        nets = [
            Net("net_Y", [("M0", 6, 7), ("M0", 0, 1), ("M0", 4, 1)],
                cable_locs=set(SD_CABLE_LOCS)),
        ]
        solution = mgr.run(nets)
        assert solution.results["net_Y"].success, "输出 net_Y 应布通"
        m1_nodes = [n for n in solution.results["net_Y"].routed_nodes if n[0] == "M1"]
        assert len(m1_nodes) > 0, "net_Y 应使用 M1 进行跨区域连接"

    def test_mid_net_three_nmos_terminals(self, grid):
        """
        内部节点 net_mid（3端口，全在 NMOS 区）：
        (x=2,y=1), (x=6,y=1), (x=10,y=1)，S/D 列无横向，必须上 M1。
        """
        mgr = make_mgr(grid, space=0, max_iter=30)
        nets = [
            Net("net_mid", [("M0", 2, 1), ("M0", 6, 1), ("M0", 10, 1)],
                cable_locs=set(SD_CABLE_LOCS)),
        ]
        solution = mgr.run(nets)
        assert solution.results["net_mid"].success, "内部节点 net_mid 应布通"
        m1_nodes = [n for n in solution.results["net_mid"].routed_nodes if n[0] == "M1"]
        assert len(m1_nodes) > 0, "net_mid 应使用 M1 连接分散的 S/D 端口"

    def test_power_nets_vdd_vss(self, grid):
        """VDD 和 VSS 电源线网布线"""
        mgr = make_mgr(grid, space=0, max_iter=20)
        nets = [
            Net("net_VDD", [("M0", 0, 8), ("M0", 12, 8)], cable_locs=set(SD_CABLE_LOCS)),
            Net("net_VSS", [("M0", 8, 0), ("M0", 12, 0)], cable_locs=set(SD_CABLE_LOCS)),
        ]
        solution = mgr.run(nets)
        assert solution.results["net_VDD"].success, "VDD 应布通"
        assert solution.results["net_VSS"].success, "VSS 应布通"


# ======================================================================
# Class 3: 完整 OAI33 布线测试（space=0）
# ======================================================================

class TestOAI33FullRouting:
    """完整 OAI33 10 线网同时布线测试"""

    @pytest.fixture(scope="class")
    def grid(self):
        return build_oai33_grid()

    def test_full_routing_min_8_of_10(self, grid):
        """
        完整 OAI33：10 个线网同时布线，至少 8/10 布通。
        保存 SVG 到 results/stdcell_oai33/。
        """
        mgr = make_mgr(grid, space=0, max_iter=80)
        nets, _ = build_oai33_nets(grid)
        solution = mgr.run(nets)

        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "results", "stdcell_oai33",
        )
        from maze_router.visualizer import Visualizer
        from maze_router.data.grid import GridGraph as _GG
        # 直接使用 Visualizer
        sol_obj = solution
        viz = Visualizer(grid, sol_obj)
        viz.save_svgs(save_dir=save_dir)

        # 共栅必须成功
        for name in ["net_A", "net_B", "net_C"]:
            assert solution.results[name].success, f"共栅 {name} 必须布通"

        # 整体至少 8/10
        assert solution.routed_count >= 8, \
            f"应至少布通 8/10 个线网，实际 {solution.routed_count}/10"

    def test_full_routing_cable_locs_respected(self, grid):
        """验证各线网在 M0 层仅使用其 cable_locs 内的节点"""
        mgr = make_mgr(grid, space=0, max_iter=80)
        nets, _ = build_oai33_nets(grid)
        solution = mgr.run(nets)

        gate_nets = {"net_A", "net_B", "net_C", "net_D", "net_E", "net_F"}
        sd_nets   = {"net_Y", "net_mid", "net_VDD", "net_VSS"}

        for net_name, result in solution.results.items():
            if not result.success:
                continue
            for node in result.routed_nodes:
                if node[0] != "M0":
                    continue
                x = node[1]
                if net_name in gate_nets:
                    assert x in GATE_COLS, \
                        f"Gate 线网 {net_name} 在 M0 使用了 S/D 列 x={x}"
                elif net_name in sd_nets:
                    assert x in SD_COLS, \
                        f"S/D 线网 {net_name} 在 M0 使用了 Gate 列 x={x}"

    def test_full_routing_m2_rows_respected(self, grid):
        """验证 M2 层布线节点仅在 y=3, y=5"""
        mgr = make_mgr(grid, space=0, max_iter=80)
        nets, _ = build_oai33_nets(grid)
        solution = mgr.run(nets)

        for net_name, result in solution.results.items():
            if not result.success:
                continue
            for node in result.routed_nodes:
                if node[0] == "M2":
                    assert node[2] in M2_ROWS, \
                        f"线网 {net_name} 在 M2 使用了禁止行 y={node[2]}"

    def test_full_routing_no_node_overlap(self, grid):
        """验证不同线网的布线路径节点不重叠"""
        mgr = make_mgr(grid, space=0, max_iter=80)
        nets, _ = build_oai33_nets(grid)
        solution = mgr.run(nets)

        occupied: dict = {}
        for net_name, result in solution.results.items():
            if not result.success:
                continue
            for node in result.routed_nodes:
                if node in occupied:
                    pytest.fail(
                        f"节点 {node} 被 {occupied[node]} 和 {net_name} 同时占用"
                    )
                occupied[node] = net_name

    def test_full_routing_summary(self, grid, capsys):
        """完整布线并打印详细摘要"""
        mgr = make_mgr(grid, space=0, max_iter=80)
        nets, _ = build_oai33_nets(grid)
        solution = mgr.run(nets)

        print(f"\n{'='*60}")
        print(f"OAI33 布线结果: {solution.routed_count}/10 布通")
        print(f"{'='*60}")
        for name, result in sorted(solution.results.items()):
            status = "OK" if result.success else "FAIL"
            layers = sorted({n[0] for n in result.routed_nodes}) if result.routed_nodes else []
            print(f"  {name:10s}: {status}  cost={result.total_cost:6.1f}  layers={layers}")
        print(f"总代价: {solution.total_cost:.1f}")
        print(f"{'='*60}")

        assert solution.routed_count >= 6


# ======================================================================
# Class 4: space=2 高压力测试
# ======================================================================

class TestOAI33WithSpacing2:
    """
    OAI33 布线测试 — 三层间距均为 2。

    Spacing=2 意味着每个已布线节点在 Chebyshev 距离 2 范围内（5×5 区域）
    对其他线网形成禁区。在 13×9 的网格上极大压缩了可用布线资源，
    是对拆线重布和拥塞感知能力的严格考验。
    """

    @pytest.fixture(scope="class")
    def grid(self):
        return build_oai33_grid()

    def _make_mgr_s2(self, grid: GridGraph, max_iter: int = 100):
        constraint_mgr = ConstraintManager([
            SpaceConstraint({"M0": 2, "M1": 2, "M2": 2})
        ])
        cost_mgr = CostManager(
            grid=grid,
            costs=[CornerCost(
                l_costs={"M0": 5.0, "M1": 5.0, "M2": 5.0},
                t_costs={},
            )],
        )
        strategy = CongestionAwareRipupStrategy(
            max_iterations=max_iter,
            base_penalty=0.2,
            penalty_growth=0.5,
        )
        return RipupManager(grid, constraint_mgr, cost_mgr, strategy)

    def test_shared_gates_spacing2_at_least_1(self, grid):
        """
        共栅 A/B/C（spacing=2）：Gate 列间距=2，Chebyshev 禁区 5×5，
        相邻 Gate 列互相阻塞，至少 1/3 布通。
        """
        mgr = self._make_mgr_s2(grid, max_iter=30)
        nets = [
            Net("net_A", [("M0", 1, 7), ("M0", 1, 1)], cable_locs=set(GATE_CABLE_LOCS)),
            Net("net_B", [("M0", 3, 7), ("M0", 3, 1)], cable_locs=set(GATE_CABLE_LOCS)),
            Net("net_C", [("M0", 5, 7), ("M0", 5, 1)], cable_locs=set(GATE_CABLE_LOCS)),
        ]
        solution = mgr.run(nets)
        routed = sum(1 for n in ["net_A","net_B","net_C"] if solution.results[n].success)
        assert routed >= 1, f"共栅 spacing=2 至少布通 1/3，实际 {routed}/3"

    def test_cross_gates_spacing2_at_least_1(self, grid):
        """
        三路循环交叉 D/E/F（spacing=2）：极端困难，至少 1/3 布通。
        """
        mgr = self._make_mgr_s2(grid, max_iter=80)
        nets = [
            Net("net_D", [("M0", 7, 7), ("M0", 11, 1)], cable_locs=set(GATE_CABLE_LOCS)),
            Net("net_E", [("M0", 9, 7), ("M0", 7, 1)],  cable_locs=set(GATE_CABLE_LOCS)),
            Net("net_F", [("M0", 11, 7), ("M0", 9, 1)], cable_locs=set(GATE_CABLE_LOCS)),
        ]
        solution = mgr.run(nets)
        routed = sum(1 for n in ["net_D","net_E","net_F"] if solution.results[n].success)
        assert routed >= 1, f"交叉 gate spacing=2 至少布通 1/3，实际 {routed}/3"

    def test_full_routing_spacing2_at_least_2(self, grid):
        """
        完整 OAI33（spacing=2）：10 个线网，至少 2/10 布通，
        保存 SVG 到 results/stdcell_oai33_s2/。
        """
        mgr = self._make_mgr_s2(grid, max_iter=100)
        nets, _ = build_oai33_nets(grid)
        solution = mgr.run(nets)

        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "results", "stdcell_oai33_s2",
        )
        from maze_router.visualizer import Visualizer
        viz = Visualizer(grid, solution)
        viz.save_svgs(save_dir=save_dir)

        print(f"\n{'='*60}")
        print(f"OAI33(spacing=2) 布线结果: {solution.routed_count}/10 布通")
        print(f"{'='*60}")
        for name, result in sorted(solution.results.items()):
            status = "OK" if result.success else "FAIL"
            layers = sorted({n[0] for n in result.routed_nodes}) if result.routed_nodes else []
            nodes_n = len(result.routed_nodes)
            print(f"  {name:10s}: {status}  cost={result.total_cost:6.1f}"
                  f"  nodes={nodes_n:3d}  layers={layers}")
        print(f"总代价: {solution.total_cost:.1f}  布通率: {solution.routed_count}/10")
        print(f"{'='*60}")

        assert solution.routed_count >= 2, \
            f"spacing=2 应至少布通 2/10，实际 {solution.routed_count}/10"

    def test_spacing2_no_routing_violation(self, grid):
        """
        验证 spacing=2 下：已布通线网的非 terminal 路由节点间
        Chebyshev 距离 > 2（不同线网）。
        """
        mgr = self._make_mgr_s2(grid, max_iter=100)
        nets, _ = build_oai33_nets(grid)
        solution = mgr.run(nets)

        all_terminals = set()
        for net in nets:
            all_terminals.update(net.terminals)

        # 按层收集各线网路由节点（排除 terminal）
        layer_net_nodes: dict = {}
        for net_name, result in solution.results.items():
            if not result.success:
                continue
            for node in result.routed_nodes:
                if node in all_terminals:
                    continue
                layer = node[0]
                if layer not in layer_net_nodes:
                    layer_net_nodes[layer] = {}
                layer_net_nodes[layer].setdefault(net_name, []).append((node[1], node[2]))

        violations = []
        for layer, net_nodes in layer_net_nodes.items():
            names = list(net_nodes.keys())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    n1, n2 = names[i], names[j]
                    for x1, y1 in net_nodes[n1]:
                        for x2, y2 in net_nodes[n2]:
                            cheby = max(abs(x1 - x2), abs(y1 - y2))
                            if cheby <= 2:
                                violations.append(
                                    f"{layer}: {n1}({x1},{y1}) ↔ "
                                    f"{n2}({x2},{y2}) dist={cheby}"
                                )

        if violations:
            print(f"\nSpacing=2 violations ({len(violations)}):")
            for v in violations[:10]:
                print(f"  {v}")
        assert len(violations) == 0, f"发现 {len(violations)} 处 spacing=2 违规"


# ======================================================================
# Class 5: MazeRouterEngine 集成测试
# ======================================================================

class TestOAI33Engine:
    """通过顶层 MazeRouterEngine 进行集成测试"""

    @pytest.fixture(scope="class")
    def grid(self):
        return build_oai33_grid()

    def test_engine_space0_full(self, grid):
        """Engine 端到端：OAI33 全部线网，space=0，至少 8/10 布通"""
        nets, _ = build_oai33_nets(grid)
        engine = MazeRouterEngine(
            grid=grid,
            nets=nets,
            space_constr={"M0": 0, "M1": 0, "M2": 0},
            corner_l_costs={"M0": 5.0, "M1": 5.0, "M2": 5.0},
            strategy="congestion_aware",
            max_iterations=80,
        )
        solution = engine.run()
        assert solution.routed_count >= 8, \
            f"Engine space=0 应至少 8/10 布通，实际 {solution.routed_count}/10"

    def test_engine_space1_at_least_4(self, grid):
        """Engine 端到端：OAI33，space=1，至少 4/10 布通（13×9 网格 + Chebyshev space=1 限制较强）"""
        nets, _ = build_oai33_nets(grid)
        engine = MazeRouterEngine(
            grid=grid,
            nets=nets,
            space_constr={"M0": 1, "M1": 1, "M2": 1},
            corner_l_costs={"M0": 5.0, "M1": 5.0, "M2": 5.0},
            strategy="congestion_aware",
            max_iterations=80,
        )
        solution = engine.run()
        assert solution.routed_count >= 4, \
            f"Engine space=1 应至少 4/10 布通，实际 {solution.routed_count}/10"

    def test_engine_saves_svg(self, grid, tmp_path):
        """Engine 可视化：生成 3 层 SVG 文件"""
        nets, _ = build_oai33_nets(grid)
        engine = MazeRouterEngine(
            grid=grid,
            nets=nets,
            space_constr={"M0": 0, "M1": 0, "M2": 0},
        )
        engine.run()
        engine.visualize(save_dir=str(tmp_path), prefix="oai33_")
        svgs = list(tmp_path.glob("*.svg"))
        assert len(svgs) == 3, f"应生成 3 个 SVG，实际 {len(svgs)}"
