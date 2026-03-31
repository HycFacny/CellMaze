# CellMaze

基于迷宫算法的标准单元布线引擎，实现精确 Steiner 树路由与拆线重布。

## 特性

- **三层金属网格** — M0（局部层）/ M1（中间层）/ M2（全局层），Via 跨层连接
- **精确 Steiner 树** — Dreyfus-Wagner DP（≤ 8 端口），方向感知折角代价
- **增量贪心路由** — 多源 A* 迷宫搜索，适用于大端口线网
- **拆线重布（Ripup & Reroute）** — PathFinder 风格拥塞感知迭代优化
- **插件化约束 & 代价** — 硬间距（SpaceConstraint）、软间距（SpaceCost）、折角（CornerCost）、最小面积（MinAreaConstraint）
- **标准单元测试用例** — OAI33 组合逻辑门（13×9，10 nets）+ D 触发器（19×9，14 nets）
- **SVG 可视化** — 每层独立 SVG 输出，含布线路径、terminal、图例

## 安装

```bash
pip install networkx pytest
```

无其他外部依赖。

## 快速上手

### 最简示例

```python
from maze_router import MazeRouterEngine, GridGraph, Net

# 1. 构建网格（3层，5×5）
grid = GridGraph.build_grid(
    layers=["M0", "M1", "M2"],
    width=5, height=5,
    edge_cost=1.0, via_cost=2.0,
)

# 2. 定义线网
nets = [
    Net("VDD", [("M0", 0, 4), ("M0", 4, 4)]),
    Net("VSS", [("M0", 0, 0), ("M0", 4, 0)]),
    Net("CLK", [("M0", 0, 2), ("M0", 2, 2), ("M0", 4, 2)]),
]

# 3. 运行布线
engine = MazeRouterEngine(
    grid=grid,
    nets=nets,
    space_constr={"M0": 1, "M1": 1, "M2": 1},
)
solution = engine.run()

print(f"布通: {solution.routed_count}/{len(nets)}")
print(f"总代价: {solution.total_cost:.2f}")

# 4. 可视化
engine.visualize(save_dir="results/demo")
```

### 标准单元完整示例

```python
from maze_router import MazeRouterEngine
from maze_router.costs.corner_cost import CornerCost
from maze_router.ripup_strategy import CongestionAwareRipupStrategy
from tests.conftest import make_oai33_testcase

# 生成 OAI33 测试用例
grid, nets, _, _ = make_oai33_testcase(space=1, corner_l_cost=3.0)

engine = MazeRouterEngine(
    grid=grid,
    nets=nets,
    space_constr={"M0": 1, "M1": 1, "M2": 1},
    corner_cost=CornerCost(
        l_costs={"M0": 3.0, "M1": 3.0, "M2": 0.0},
        t_costs={"M0": 1.0, "M1": 1.0},
    ),
    min_area={"M0": 2, "M1": 2},                          # 最小面积约束
    strategy=CongestionAwareRipupStrategy(max_iterations=20),
)

solution = engine.run()
engine.visualize(save_dir="results/stdcell_oai33")
```

## 项目结构

```
CellMaze/
├── maze_router/
│   ├── __init__.py                  # 包入口，导出所有公开类
│   ├── engine.py                    # MazeRouterEngine — 顶层调度
│   ├── router.py                    # Router — 算法派发 + Pin 引出
│   ├── ripup_manager.py             # RipupManager — 拆线重布主循环
│   ├── ripup_strategy.py            # RipupStrategy — 策略 ABC + 两个内置实现
│   ├── maze_router_algo.py          # MazeRouter (A*) + build_steiner_greedy
│   ├── steiner_router_algo.py       # SteinerRouter — Dreyfus-Wagner DP
│   ├── cost_manager.py              # CostManager + RoutingContext
│   ├── constraint_manager.py        # ConstraintManager
│   ├── visualizer.py                # Visualizer — SVG 输出
│   ├── data/
│   │   ├── net.py                   # Node, Net, PinSpec, RoutingResult, RoutingSolution
│   │   ├── grid.py                  # GridGraph — nx.Graph 封装
│   │   └── region.py                # Region — 区域拆线辅助
│   ├── constraints/
│   │   ├── base.py                  # BaseConstraint ABC
│   │   ├── space_constraint.py      # SpaceConstraint — Chebyshev 硬间距
│   │   └── min_area_constraint.py   # MinAreaConstraint — 最小面积（soft rule）
│   └── costs/
│       ├── base.py                  # BaseCost ABC
│       ├── corner_cost.py           # CornerCost — L/T 型折角代价
│       └── space_cost.py            # SpaceCost — 软间距代价（S2S/T2S/T2T）
├── tests/
│   ├── conftest.py                  # 共享 fixture，标准单元测试用例生成器
│   ├── test_basic.py                # 基础单元测试（网格/约束/代价/路由器）
│   ├── test_steiner.py              # Steiner 树专项测试
│   ├── test_ripup.py                # 拆线重布测试
│   ├── test_stdcell.py              # 标准单元通用测试（OAI33 / 6T）
│   ├── test_oai33.py                # OAI33 精确版布局测试（30 tests）
│   └── test_dff.py                  # DFF 时序单元测试（38 tests）
├── results/                         # SVG 输出目录
│   ├── stdcell_oai33/
│   ├── stdcell_dff/
│   └── stdcell_6t/
├── main.py                          # 演示脚本
└── proposal.txt                     # 设计规格文档
```

## 运行测试

```bash
# 运行全套测试
pytest tests/ -q

# 分模块运行
pytest tests/test_basic.py -v        # 基础单元测试（~35 tests）
pytest tests/test_steiner.py -v      # Steiner 树专项（7 tests）
pytest tests/test_ripup.py -v        # 拆线重布（13 tests）
pytest tests/test_oai33.py -v        # OAI33 精确测试（30 tests）
pytest tests/test_dff.py -v          # DFF 时序单元（38 tests）
pytest tests/ -q                     # 全部（133 tests）
```

## 核心 API

### MazeRouterEngine（顶层入口）

```python
engine = MazeRouterEngine(
    grid=grid,                          # GridGraph
    nets=nets,                          # List[Net]
    space_constr={"M0": 1, "M1": 1},    # 硬间距约束 {layer: min_space}
    corner_cost=CornerCost.default(),   # 折角代价（None=默认）
    space_cost_rules=[...],             # 软间距代价规则列表（可选）
    min_area={"M0": 2, "M1": 2},        # 最小面积约束（可选，soft rule）
    cable_locs={"net_A": {...}},        # M0 可走节点 {net_name: Set[Node]}
    pin_layers={"net_Q": "M1"},         # Pin 引出层 {net_name: layer}
    strategy=CongestionAwareRipupStrategy(max_iterations=20),
    congestion_weight=1.0,
)
solution = engine.run()
engine.visualize(save_dir="results/", prefix="demo_")
```

### GridGraph

```python
# 方式1：工厂方法构建标准网格
grid = GridGraph.build_grid(
    layers=["M0", "M1", "M2"],
    width=13, height=9,
    edge_cost=1.0, via_cost=2.0,
)

# 方式2：手动构建精确版图网格
grid = GridGraph()
grid.add_node(("M0", 0, 0))
grid.add_edge(("M0", 0, 0), ("M0", 1, 0), cost=1.0)
grid.add_edge(("M0", 0, 0), ("M1", 0, 0), cost=2.0)  # via
```

### Net

```python
net = Net(
    name="net_CK",
    terminals=[("M0", 3, 1), ("M0", 13, 1), ("M0", 5, 7), ("M0", 11, 7)],
    cable_locs=gate_allowed_nodes,   # M0 层可走节点（None=不限制）
    pin_spec=PinSpec(layer="M1"),    # Pin 引出规格（None=不需要）
    priority=0,                      # 布线优先级（越小越先）
)
```

### CornerCost

```python
# 默认：L型折角=5.0，T型折角=0.0（默认不惩罚分支）
cc = CornerCost.default()

# 按层自定义
cc = CornerCost(
    l_costs={"M0": 5.0, "M1": 5.0, "M2": 0.0},
    t_costs={"M0": 10.0},
)

# 完全禁用
cc = CornerCost.disabled()
```

### RipupStrategy

```python
# 默认策略（10 轮）
strategy = DefaultRipupStrategy(max_iterations=10)

# PathFinder 风格拥塞感知（推荐用于复杂场景）
strategy = CongestionAwareRipupStrategy(
    max_iterations=80,
    base_penalty=0.1,
    penalty_growth=0.5,
)

# 自定义策略
class MyStrategy(RipupStrategy):
    def order_nets(self, nets): ...
    def get_router_type(self, net, iteration): ...
    def order_terminals(self, net, routed_nodes): ...
    def decide_ripup(self, ...): ...
    def get_max_iterations(self): ...
```

### RoutingSolution

```python
solution.routed_count           # 布通线网数
solution.total_cost             # 总代价
solution.failed_nets            # 未布通线网名列表
solution.results["net_CK"]      # RoutingResult

result = solution.results["net_CK"]
result.success                  # bool
result.routed_nodes             # Set[Node]
result.routed_edges             # List[Edge]
result.pin_point                # Optional[Node]（Pin 引出点）
result.total_cost               # float
```

## 演示

```bash
python main.py
# 输出 SVG 到 results/stdcell_oai33/ 和 results/stdcell_6t/
```
