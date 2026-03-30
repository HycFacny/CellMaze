"""
CellMaze — 标准单元布线主入口

演示案例：OAI33 标准单元（Y = ~((A+B+C)·(D+E+F))）
  - 6 PMOS + 6 NMOS 晶体管，共 10 个布线线网
  - 网格：13列(x=0..12) × 9行(y=0..8) × 3层(M0/M1/M2)
  - M2 仅第二行(y=2)和第四行(y=4)可横向布线（遵照 proposal 规定）
  - 三层间距 space = 2（Chebyshev 距离，严格压力测试）

用法：
    python main.py

输出：
    终端打印布线摘要，SVG 保存到 results/main_oai33/
"""

import os
import sys
import time
from typing import Dict
import networkx as nx

from maze_router.net import Net
from maze_router.grid import RoutingGrid
from maze_router.spacing import SpacingManager
from maze_router.corner import CornerManager
from maze_router.ripup import RipupManager
from maze_router.strategy import CongestionAwareStrategy
from maze_router.visualizer import Visualizer


# ======================================================================
# 网格参数
# ======================================================================

COLS = 13        # x: 0..12，6晶体管/行 → 2*6+1=13列
ROWS = 9         # y: 0..8，9行布线网格

#  行语义（9行）:
#   y=0  VSS 轨道（下边界）
#   y=1  NMOS 有源区（S/D 扩散）
#   y=2  下方布线通道 ← M2 轨道 1（proposal 规定"第二行"）
#   y=3  下方布线通道
#   y=4  单元中心    ← M2 轨道 2（proposal 规定"第四行"）
#   y=5  上方布线通道
#   y=6  上方布线通道
#   y=7  PMOS 有源区（S/D 扩散）
#   y=8  VDD 轨道（上边界）

SD_COLS      = {0, 2, 4, 6, 8, 10, 12}   # Source/Drain 列（偶数列）
GATE_COLS    = {1, 3, 5, 7, 9, 11}        # Gate 列（奇数列）
ACTIVE_ROWS  = {1, 7}                      # NMOS(y=1) / PMOS(y=7) 有源区
M2_ROWS      = {2, 4}                      # M2 仅在 y=2 和 y=4（proposal 规定）

# M0 cable 约束集合
GATE_CABLE_LOCS = frozenset(
    ("M0", x, y) for x in GATE_COLS for y in range(ROWS)
)
SD_CABLE_LOCS = frozenset(
    ("M0", x, y) for x in SD_COLS for y in range(ROWS)
)


# ======================================================================
# 网格构建
# ======================================================================

def build_grid() -> RoutingGrid:
    """
    构建 OAI33 标准单元三层布线网格。

    M0 规则：
      - S/D 列：仅 NMOS 区(y=0↔1) 和 PMOS 区(y=7↔8) 有竖向边，无横向边
      - Gate 列：全程竖向连通(y=0→8)，横向仅在非 Active 行
    M1 规则：完整矩形网格（全连通）
    M2 规则：仅 y=2、y=4 存在节点，仅横向布线；y=2↔y=4 有竖向直连边
    Via：M0↔M1 全位置；M1↔M2 仅 y=2,4
    """
    G = nx.Graph()

    # ---------- M0 ----------
    for x in range(COLS):
        for y in range(ROWS):
            G.add_node(("M0", x, y))

    for x in range(COLS):
        if x in SD_COLS:
            # S/D 列：仅 NMOS 区和 PMOS 区竖向连通
            G.add_edge(("M0", x, 0), ("M0", x, 1), cost=1.0)
            G.add_edge(("M0", x, 7), ("M0", x, 8), cost=1.0)
        else:
            # Gate 列：全程竖向
            for y in range(ROWS - 1):
                G.add_edge(("M0", x, y), ("M0", x, y + 1), cost=1.0)

    # Gate 列横向边：仅非 Active 行，相邻 Gate 列之间（跨中间 S/D 列，cost=2）
    gate_sorted = sorted(GATE_COLS)
    for i in range(len(gate_sorted) - 1):
        x1, x2 = gate_sorted[i], gate_sorted[i + 1]
        for y in range(ROWS):
            if y not in ACTIVE_ROWS:
                G.add_edge(("M0", x1, y), ("M0", x2, y), cost=2.0)

    # ---------- M1 ----------
    for x in range(COLS):
        for y in range(ROWS):
            G.add_node(("M1", x, y))
    for x in range(COLS):
        for y in range(ROWS - 1):
            G.add_edge(("M1", x, y), ("M1", x, y + 1), cost=1.0)
    for y in range(ROWS):
        for x in range(COLS - 1):
            G.add_edge(("M1", x, y), ("M1", x + 1, y), cost=1.0)

    # ---------- M2：仅 y=2 和 y=4 ----------
    for x in range(COLS):
        for y in M2_ROWS:
            G.add_node(("M2", x, y))
    for y in M2_ROWS:
        for x in range(COLS - 1):
            G.add_edge(("M2", x, y), ("M2", x + 1, y), cost=1.0)
    # y=2 ↔ y=4 竖向直连（跨 2 行，cost=2）
    for x in range(COLS):
        G.add_edge(("M2", x, 2), ("M2", x, 4), cost=2.0)

    # ---------- Via ----------
    for x in range(COLS):
        for y in range(ROWS):
            G.add_edge(("M0", x, y), ("M1", x, y), cost=2.0)
    for x in range(COLS):
        for y in M2_ROWS:
            G.add_edge(("M1", x, y), ("M2", x, y), cost=2.0)

    return RoutingGrid(G)


# ======================================================================
# 线网定义
# ======================================================================

def build_nets():
    """
    构建 OAI33 的 10 个布线线网。

    电路：Y = ~((A+B+C) · (D+E+F))

    PMOS 布局（y=7/8）：
      VDD(x0)-PA(A,x1)-p1(x2)-PB(B,x3)-p2(x4)-PC(C,x5)-Y(x6)
             -PD(D,x7)-p3(x8)-PE(E,x9)-p4(x10)-PF(F,x11)-VDD(x12)
      串联路径1: VDD→PA→PB→PC→Y（A/B/C全为0时上拉）
      串联路径2: Y→PD→PE→PF→VDD（D/E/F全为0时上拉）

    NMOS 布局（y=0/1，交叉布局）：
      Y(x0)-NA(A,x1)-mid(x2)-NB(B,x3)-Y(x4)-NC(C,x5)-mid(x6)
            -NE(E,x7)-VSS(x8)-NF(F,x9)-mid(x10)-ND(D,x11)-VSS(x12)
      并联组1(NA||NB||NC)：连接 Y ↔ mid
      并联组2(ND||NE||NF)：连接 mid ↔ VSS

    Gate 映射（PMOS列 → NMOS列）：
      A: x=1 → x=1  共栅（同列）
      B: x=3 → x=3  共栅（同列）
      C: x=5 → x=5  共栅（同列）
      D: x=7 → x=11 交叉（跨4列）
      E: x=9 → x=7  交叉（与D交叉）
      F: x=11→ x=9  交叉（三路循环交叉）
    """
    gate_locs = set(GATE_CABLE_LOCS)
    sd_locs   = set(SD_CABLE_LOCS)

    nets = [
        # ---- Gate 线网 ----
        Net("net_A",   [("M0", 1, 7), ("M0",  1, 1)], cable_locs=gate_locs),
        Net("net_B",   [("M0", 3, 7), ("M0",  3, 1)], cable_locs=gate_locs),
        Net("net_C",   [("M0", 5, 7), ("M0",  5, 1)], cable_locs=gate_locs),
        Net("net_D",   [("M0", 7, 7), ("M0", 11, 1)], cable_locs=gate_locs),
        Net("net_E",   [("M0", 9, 7), ("M0",  7, 1)], cable_locs=gate_locs),
        Net("net_F",   [("M0",11, 7), ("M0",  9, 1)], cable_locs=gate_locs),
        # ---- S/D 线网 ----
        Net("net_Y",   [("M0", 6, 7), ("M0",  0, 1), ("M0", 4, 1)],
            cable_locs=sd_locs),
        Net("net_mid", [("M0", 2, 1), ("M0",  6, 1), ("M0",10, 1)],
            cable_locs=sd_locs),
        Net("net_VDD", [("M0", 0, 8), ("M0", 12, 8)], cable_locs=sd_locs),
        Net("net_VSS", [("M0", 8, 0), ("M0", 12, 0)], cable_locs=sd_locs),
    ]
    return nets


# ======================================================================
# 打印工具
# ======================================================================

def print_separator(char="=", width=62):
    print(char * width)

def print_grid_info(grid: RoutingGrid):
    m0 = len(grid.get_nodes_on_layer("M0"))
    m1 = len(grid.get_nodes_on_layer("M1"))
    m2 = len(grid.get_nodes_on_layer("M2"))
    total_edges = grid.graph.number_of_edges()
    print_separator()
    print("网格信息")
    print_separator("-")
    print(f"  尺寸        : {COLS} 列 × {ROWS} 行  ({COLS}×{ROWS}={COLS*ROWS} 单元/层)")
    print(f"  M0 节点数   : {m0}  (全部 {COLS}×{ROWS}，含 S/D 列 + Gate 列)")
    print(f"  M1 节点数   : {m1}  (完整矩形网格)")
    print(f"  M2 节点数   : {m2}  (仅 y=2 和 y=4，共 {COLS}×2 节点)")
    print(f"  总边数      : {total_edges}")
    print(f"  M2 布线行   : y={sorted(M2_ROWS)}  (proposal 规定：第2行和第4行)")


def print_net_info(nets):
    print_separator()
    print(f"线网定义（共 {len(nets)} 个）")
    print_separator("-")
    gate_nets = [n for n in nets if n.name.startswith("net_") and
                 n.name[4] in "ABCDEF"]
    sd_nets   = [n for n in nets if n.name not in {g.name for g in gate_nets}]

    print("  Gate 线网（6个）：")
    for net in gate_nets:
        t = net.terminals
        shared = "共栅" if t[0][1] == t[1][1] else "交叉"
        arrow = f"PMOS(x={t[0][1]}) ↔ NMOS(x={t[1][1]})"
        print(f"    {net.name:8s}: {shared}  {arrow}")

    print("  S/D 线网（4个）：")
    for net in sd_nets:
        pts = ", ".join(f"({n[1]},{n[2]})" for n in net.terminals)
        print(f"    {net.name:8s}: {len(net.terminals)} 端口  [{pts}]")


def print_solution(solution, nets, elapsed: float, space: int):
    total  = len(nets)
    routed = solution.routed_count
    failed = solution.failed_nets

    print_separator()
    print(f"布线结果  (space={space}, 用时 {elapsed:.2f}s)")
    print_separator("-")
    print(f"  布通率: {routed}/{total}  ({routed/total*100:.0f}%)")
    print(f"  总代价: {solution.total_cost:.1f}")
    if failed:
        print(f"  未布通: {failed}")
    print()
    print(f"  {'线网':<10} {'状态':<6} {'代价':>7}  {'节点数':>5}  层")
    print(f"  {'-'*9} {'-'*5} {'-'*7}  {'-'*5}  ---")

    for name, result in sorted(solution.results.items()):
        status = "OK  ✓" if result.success else "FAIL"
        nodes  = len(result.routed_nodes)
        layers = sorted(set(n[0] for n in result.routed_nodes)) \
                 if result.routed_nodes else []
        layers_str = "+".join(l.replace("M","") for l in layers) \
                     if layers else "-"
        print(f"  {name:<10} {status:<6} {result.total_cost:>7.1f}  "
              f"{nodes:>5}  M{layers_str}")


def print_spacing_analysis(solution, nets, space: int):
    """
    对已布通线网做同层 Chebyshev 间距检查，输出统计。

    注意：终端(terminal)节点是电路固定连接点（如 Gate 列紧邻 S/D 列，
    Chebyshev=1 是版图设计固有的），不计入违规检查。
    只检查布线过程中新增的中间节点（non-terminal routing nodes）。
    """
    # 收集所有线网的终端节点集合（固定点，排除在外）
    all_terminals = set()
    for net in nets:
        all_terminals.update(net.terminals)

    layer_net_nodes = {}
    for net_name, result in solution.results.items():
        if not result.success:
            continue
        for node in result.routed_nodes:
            if node in all_terminals:   # 终端是固定位置，跳过
                continue
            layer = node[0]
            layer_net_nodes.setdefault(layer, {}).setdefault(
                net_name, []).append((node[1], node[2]))

    violations = 0
    min_dist_by_layer = {}
    for layer, net_nodes in layer_net_nodes.items():
        net_list = list(net_nodes.keys())
        layer_min = float('inf')
        for i in range(len(net_list)):
            for j in range(i + 1, len(net_list)):
                n1, n2 = net_list[i], net_list[j]
                for x1, y1 in net_nodes[n1]:
                    for x2, y2 in net_nodes[n2]:
                        d = max(abs(x1 - x2), abs(y1 - y2))
                        layer_min = min(layer_min, d)
                        if d <= space:
                            violations += 1
        if layer_min < float('inf'):
            min_dist_by_layer[layer] = layer_min

    print_separator("-")
    print(f"  间距检查 (space={space}, Chebyshev, 已排除 terminal 固定点):")
    for layer in ["M0", "M1", "M2"]:
        if layer in min_dist_by_layer:
            md = min_dist_by_layer[layer]
            ok = "OK ✓" if md > space else f"VIOLATION(min={md}) ✗"
            print(f"    {layer}: 布线节点最小间距={md}  {ok}")
        else:
            print(f"    {layer}: 无布线中间节点")
    if violations == 0:
        print("  → 布线节点间距约束全部满足 ✓")
    else:
        print(f"  → 发现 {violations} 处间距违规 ✗")


def print_corner_stats(solution, nets):
    """
    统计已布通线网的折角数量（同层方向改变次数）。

    遍历每个线网的 routed_edges，比较相邻边的方向，
    累计同层折角次数，按层分别输出。
    """
    from maze_router.router import _move_dir_code, _DIR_NONE

    # 收集所有终端，用于识别路径起点（方向不确定，不计折角）
    all_terminals = set()
    for net in nets:
        all_terminals.update(net.terminals)

    total_corners = 0
    print_separator("-")
    print("  折角统计（同层方向改变次数）:")

    for name, result in sorted(solution.results.items()):
        if not result.success or not result.routed_edges:
            continue

        # 构建节点邻接关系，重建每个节点的邻居列表（用于找路径）
        # 直接分析 edges：相邻两条边在同节点处方向改变则为折角
        # 建图：node -> list of neighbors（有向边）
        from collections import defaultdict
        adj_out = defaultdict(list)
        for (u, v) in result.routed_edges:
            adj_out[u].append(v)
            adj_out[v].append(u)

        # 在 Steiner 树的每个内部节点，枚举所有边对，统计折角
        corners_by_layer: Dict[str, int] = {}
        counted_pairs = set()

        for (u, v) in result.routed_edges:
            # 在节点 u 处：检查所有其他邻居 w，看 w→u→v 是否折角
            for w in adj_out[u]:
                if w == v:
                    continue
                pair = (min(id(w), id(v)), max(id(w), id(v)), u)
                if pair in counted_pairs:
                    continue
                counted_pairs.add(pair)

                d_in  = _move_dir_code(w, u)
                d_out = _move_dir_code(u, v)
                if (d_in != _DIR_NONE
                        and d_out != _DIR_NONE
                        and d_in != d_out
                        and u[0] == v[0] == w[0]):   # 同层
                    layer = u[0]
                    corners_by_layer[layer] = corners_by_layer.get(layer, 0) + 1
                    total_corners += 1

        net_corners = sum(corners_by_layer.values())
        if net_corners > 0:
            detail = ", ".join(f"{l}:{c}" for l, c in sorted(corners_by_layer.items()))
            print(f"    {name:<10}: {net_corners:3d} 个折角  [{detail}]")
        else:
            print(f"    {name:<10}: {net_corners:3d} 个折角")

    print_separator("-")
    print(f"  总折角数: {total_corners}")



def main():
    print_separator()
    print("  CellMaze — OAI33 标准单元布线演示")
    print("  Y = ~((A+B+C) · (D+E+F))")
    print_separator()

    # 1. 构建网格
    print("\n[1/4] 构建布线网格...")
    grid = build_grid()
    print_grid_info(grid)

    # 2. 定义线网
    print("\n[2/4] 定义布线线网...")
    nets = build_nets()
    print_net_info(nets)

    # 3. 配置布线策略并运行
    SPACE = 1
    CORNER = CornerManager(l_costs={"M0": 5.0, "M1": 5.0, "M2": 5.0})
    print(f"\n[3/4] 启动布线引擎  (space={SPACE}, corner={CORNER})...")
    print_separator("-")

    spacing_mgr = SpacingManager({"M0": SPACE, "M1": SPACE, "M2": SPACE})

    strategy = CongestionAwareStrategy(
        max_iterations=100,
        congestion_weight=0.5,
    )

    manager = RipupManager(grid, spacing_mgr, strategy, corner_mgr=CORNER)

    t0 = time.time()
    solution = manager.run(nets)
    elapsed = time.time() - t0

    print(f"  布线完成，耗时 {elapsed:.2f}s")

    # 4. 打印结果
    print(f"\n[4/4] 布线结果摘要")
    print_solution(solution, nets, elapsed, SPACE)
    print_corner_stats(solution, nets)

    # 5. 保存 SVG
    output_dir = os.path.join(os.path.dirname(__file__), "results", "main_oai33")
    print_separator("-")
    print(f"  保存 SVG 到: {output_dir}/")
    viz = Visualizer(grid)
    saved = viz.save_svg(
        solution,
        output_dir=output_dir,
        spacing_mgr=spacing_mgr,
        show_spacing=True,
        cell_size=50,
        padding=70,
    )
    for path in saved:
        print(f"    ✓ {os.path.basename(path)}")

    print_separator()
    routed = solution.routed_count
    total  = len(nets)
    print(f"  完成：{routed}/{total} 线网布通  总代价 {solution.total_cost:.1f}")
    print_separator()

    return 0 if solution.routed_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
