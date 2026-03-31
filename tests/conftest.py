"""
pytest 共享 fixtures 和标准单元测试用例生成器

生成一个接近真实标准单元版图的测试场景：
  - 两排晶体管（PMOS 上排、NMOS 下排），每排 n 个
  - M0 网格: COLS = 2*n-1 列 × ROWS 行
    - SD 列（偶数列 0,2,4,...）：只能纵向走线，上/下半区各自限制
    - Gate 列（奇数列 1,3,5,...）：横向+纵向，Gate 可直连上下
  - M1 网格: 全网格，横向+纵向
  - M2 网格: 只有 y=2, y=4 两行，只允许横向走线
  - Via M0-M1: 在 Gate 列（Active 区域外）允许 via，SD 列也可
  - Via M1-M2: 在 y=2, y=4 可打 via
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import pytest

from maze_router.data.net import Node, Net, PinSpec
from maze_router.data.grid import GridGraph
from maze_router.constraint_manager import ConstraintManager
from maze_router.cost_manager import CostManager
from maze_router.constraints.space_constraint import SpaceConstraint
from maze_router.costs.corner_cost import CornerCost
from maze_router.costs.space_cost import SpaceCost, SpaceType, SpaceCostRule


# -----------------------------------------------------------------------
# 标准单元测试用例生成器
# -----------------------------------------------------------------------

def make_stdcell_grid(
    n_transistors: int = 4,
    n_rows: int = 9,
    space: int = 1,
) -> GridGraph:
    """
    构建标准单元版图风格的三层网格。

    参数:
        n_transistors: 每排晶体管数量
        n_rows:        网格行数（建议 9）
        space:         层间 via 边代价（不是间距，是边cost）
    返回:
        GridGraph
    """
    COLS = 2 * n_transistors - 1
    ROWS = n_rows
    MID = ROWS // 2   # 中间行（区分上下半区）

    # SD 列（偶数）和 Gate 列（奇数）
    SD_COLS = set(range(0, COLS, 2))
    GATE_COLS = set(range(1, COLS, 2))

    layers = ["M0", "M1", "M2"]

    # M2 只有 y=2, y=4 行
    m2_allowed = {(x, y) for x in range(COLS) for y in (2, 4) if y < ROWS}

    # M0: SD 列上/下半区分离（通过移除中间节点实现）
    # 上半区 SD: y > MID；下半区 SD: y < MID；MID 行 SD 列节点允许
    m0_removed: Set[Node] = set()
    # （标准单元中 Active 区域在中间，SD 列在 Active 内不能跨中线）
    # 这里简化：M0 全保留，但通过 cable_locs 限制各线网

    via_connections = []

    # M0-M1 via: 所有 Gate 列节点 + SD 列节点都允许
    for x in range(COLS):
        for y in range(ROWS):
            src = ("M0", x, y)
            dst = ("M1", x, y)
            via_connections.append((src, dst, 2.0))

    # M1-M2 via: 只在 y=2, y=4
    for x in range(COLS):
        for y in (2, 4):
            if y < ROWS:
                src = ("M1", x, y)
                dst = ("M2", x, y)
                via_connections.append((src, dst, 2.0))

    g = GridGraph.build_grid(
        layers=layers,
        width=COLS,
        height=ROWS,
        edge_cost=1.0,
        via_cost=2.0,
        allowed_nodes={"M2": m2_allowed},
        horizontal_only_layers={"M2"},
        via_connections=via_connections,
        vertical_only_cols={"M0": SD_COLS},
    )

    return g


def make_stdcell_nets(
    n_transistors: int = 4,
    n_rows: int = 9,
    net_defs: Optional[List[Dict]] = None,
) -> Tuple[List[Net], Dict[str, Set[Node]]]:
    """
    生成标准单元线网。

    参数:
        n_transistors:  每排晶体管数量
        n_rows:         网格行数
        net_defs:       线网定义列表，每项:
                        {name, terminals: [(layer,x,y),...], sd_cols(可选), pin_layer(可选)}
    返回:
        (nets, cable_locs_map)
        cable_locs_map: {net_name: Set[Node]}  M0 可走节点集合
    """
    COLS = 2 * n_transistors - 1
    ROWS = n_rows
    MID = ROWS // 2

    SD_COLS = set(range(0, COLS, 2))
    GATE_COLS = set(range(1, COLS, 2))

    if net_defs is None:
        # 默认生成一套典型线网（Gate 和 SD 连接）
        net_defs = _default_net_defs(n_transistors, n_rows, COLS, MID)

    nets = []
    cable_locs_map: Dict[str, Set[Node]] = {}

    for nd in net_defs:
        name = nd["name"]
        terminals = [tuple(t) for t in nd["terminals"]]

        pin_spec = None
        if "pin_layer" in nd and nd["pin_layer"]:
            pin_spec = PinSpec(layer=nd["pin_layer"])

        # cable_locs: 该线网在 M0 上可走的节点
        if "sd_cols" in nd and nd["sd_cols"] is not None:
            # SD 线网：只能在指定 SD 列的上/下半区走纵向
            sd_c = set(nd["sd_cols"])
            cable = set()
            # 判断是上半区还是下半区
            term_ys = [t[2] for t in terminals if t[0] == "M0"]
            if term_ys:
                avg_y = sum(term_ys) / len(term_ys)
                if avg_y > MID:
                    # 上半区
                    cable = {("M0", x, y) for x in sd_c for y in range(MID, ROWS)}
                else:
                    # 下半区
                    cable = {("M0", x, y) for x in sd_c for y in range(0, MID + 1)}
            cable_locs_map[name] = cable
        elif "gate_cols" in nd and nd["gate_cols"] is not None:
            # Gate 线网：只能在指定 Gate 列走，但可以跨上下
            gc = set(nd["gate_cols"])
            cable = {("M0", x, y) for x in gc for y in range(ROWS)}
            cable_locs_map[name] = cable
        # 不设 cable_locs 的线网不限制 M0 走线

        net = Net(
            name=name,
            terminals=terminals,
            pin_spec=pin_spec,
        )
        nets.append(net)

    return nets, cable_locs_map


def _default_net_defs(
    n_transistors: int, n_rows: int, n_cols: int, mid: int
) -> List[Dict]:
    """
    生成默认线网定义（模拟 OAI 门）。

    线网布局：
      - Gate 线网：奇数列，上下两排都有端口（共栅）
      - SD 线网：偶数列，上排 VDD，下排 GND，中间为内部节点
    """
    defs = []
    top_row = n_rows - 1
    bot_row = 0

    # Gate 线网（共栅：上排 top_row 和下排 bot_row 都连接）
    for i, gc in enumerate(range(1, n_cols, 2)):
        defs.append({
            "name": f"G{i}",
            "terminals": [
                ("M0", gc, top_row),
                ("M0", gc, bot_row),
            ],
            "gate_cols": [gc],
        })

    # VDD 线网（上排 SD，连接到顶端）
    defs.append({
        "name": "VDD",
        "terminals": [("M0", x, top_row) for x in range(0, n_cols, 2)],
        "sd_cols": list(range(0, n_cols, 2)),
        "pin_layer": "M1",
    })

    # GND 线网（下排 SD，连接到底端）
    defs.append({
        "name": "GND",
        "terminals": [("M0", x, bot_row) for x in range(0, n_cols, 2)],
        "sd_cols": list(range(0, n_cols, 2)),
        "pin_layer": "M1",
    })

    return defs


def make_stdcell_testcase(
    n_transistors: int = 4,
    n_rows: int = 9,
    space: int = 1,
    soft_space_penalty: float = 0.0,
    corner_l_cost: float = 5.0,
    corner_t_cost: float = 0.0,
    net_defs: Optional[List[Dict]] = None,
) -> Tuple[GridGraph, List[Net], ConstraintManager, CostManager]:
    """
    生成标准单元版图测试用例。

    参数:
        n_transistors:      每排晶体管数
        n_rows:             网格行数（建议 9）
        space:              硬间距约束
        soft_space_penalty: 软间距代价（0=不启用）
        corner_l_cost:      L 型折角代价
        corner_t_cost:      T 型折角代价
        net_defs:           自定义线网定义（None=默认）
    返回:
        (grid, nets, constraint_mgr, cost_mgr)
    """
    grid = make_stdcell_grid(n_transistors, n_rows, space)
    nets, cable_locs_map = make_stdcell_nets(n_transistors, n_rows, net_defs)

    # 应用 cable_locs
    for net in nets:
        if net.name in cable_locs_map:
            net.cable_locs = cable_locs_map[net.name]

    # 约束
    constraint_mgr = ConstraintManager([
        SpaceConstraint(rules={"M0": space, "M1": space, "M2": space})
    ])

    # 代价
    costs = [CornerCost(
        l_costs={"M0": corner_l_cost, "M1": corner_l_cost, "M2": corner_l_cost},
        t_costs={"M0": corner_t_cost, "M1": corner_t_cost, "M2": corner_t_cost},
    )]
    if soft_space_penalty > 0:
        costs.append(SpaceCost([
            SpaceCostRule("M0", SpaceType.S2S, space + 1, soft_space_penalty),
            SpaceCostRule("M1", SpaceType.S2S, space + 1, soft_space_penalty),
        ]))
    cost_mgr = CostManager(grid=grid, costs=costs)

    return grid, nets, constraint_mgr, cost_mgr


# -----------------------------------------------------------------------
# OAI33 标准测试用例
# -----------------------------------------------------------------------

def make_oai33_testcase(
    space: int = 1,
    corner_l_cost: float = 5.0,
) -> Tuple[GridGraph, List[Net], ConstraintManager, CostManager]:
    """
    OAI33 标准单元测试用例（3+3 输入，6 个晶体管/排）。

    线网拓扑（OAI33: (A1+A2+A3)·(B1+B2+B3) 的补）：
      - 3 个并联 PMOS + 3 个并联 PMOS（上排）
      - 3 个串联 NMOS + 3 个串联 NMOS（下排）
      - Gate 线网：A1/A2/A3, B1/B2/B3
      - 内部节点：NMOS 串联点
      - VDD/GND
    """
    n = 6   # 每排 6 个晶体管
    COLS = 2 * n - 1   # 11 列
    ROWS = 9
    MID = 4

    net_defs = [
        # Gate 线网（共栅）
        {"name": "A1", "terminals": [("M0", 1, 8), ("M0", 1, 0)], "gate_cols": [1]},
        {"name": "A2", "terminals": [("M0", 3, 8), ("M0", 3, 0)], "gate_cols": [3]},
        {"name": "A3", "terminals": [("M0", 5, 8), ("M0", 5, 0)], "gate_cols": [5]},
        {"name": "B1", "terminals": [("M0", 7, 8), ("M0", 7, 0)], "gate_cols": [7]},
        {"name": "B2", "terminals": [("M0", 9, 8), ("M0", 9, 0)], "gate_cols": [9]},
        # VDD: 上排所有 SD 列
        {
            "name": "VDD",
            "terminals": [("M0", x, 8) for x in range(0, COLS, 2)],
            "sd_cols": list(range(0, COLS, 2)),
            "pin_layer": "M1",
        },
        # GND: 下排两端 SD
        {
            "name": "GND",
            "terminals": [("M0", 0, 0), ("M0", COLS - 1, 0)],
            "sd_cols": [0, COLS - 1],
            "pin_layer": "M1",
        },
        # 输出 Y: 连接 PMOS/NMOS 漏极
        {
            "name": "Y",
            "terminals": [("M0", COLS // 2, MID)],
            "pin_layer": "M1",
        },
        # NMOS 串联内部节点（A 组 3 个串联的中间点）
        {
            "name": "INT_A",
            "terminals": [("M0", 4, 2), ("M0", 4, 3)],
            "sd_cols": [4],
        },
        # NMOS 串联内部节点（B 组 3 个串联的中间点）
        {
            "name": "INT_B",
            "terminals": [("M0", 8, 2), ("M0", 8, 3)],
            "sd_cols": [8],
        },
    ]

    return make_stdcell_testcase(
        n_transistors=n,
        n_rows=ROWS,
        space=space,
        corner_l_cost=corner_l_cost,
        net_defs=net_defs,
    )


# -----------------------------------------------------------------------
# pytest fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def small_grid():
    """5x5 的简单三层网格（无约束）。"""
    return GridGraph.build_grid(
        layers=["M0", "M1"],
        width=5,
        height=5,
        edge_cost=1.0,
        via_cost=2.0,
    )


@pytest.fixture
def simple_constraint_mgr():
    """间距=1 的约束管理器。"""
    return ConstraintManager([
        SpaceConstraint(rules={"M0": 1, "M1": 1, "M2": 1})
    ])


@pytest.fixture
def default_cost_mgr(small_grid):
    """默认代价管理器（L折角=5，T折角=0）。"""
    return CostManager(grid=small_grid, costs=[CornerCost.default()])


@pytest.fixture
def stdcell_4t():
    """4 晶体管/排标准单元测试用例。"""
    return make_stdcell_testcase(n_transistors=4, n_rows=9, space=1)


@pytest.fixture
def oai33():
    """OAI33 标准单元测试用例。"""
    return make_oai33_testcase(space=1)
