# 使用指南

## 1. 安装

```bash
pip install networkx
```

项目无其他外部依赖。运行测试还需要：

```bash
pip install pytest
```

## 2. 快速上手

```python
import logging
from maze_router import (
    RoutingGrid, Net, SpacingManager,
    RipupManager, DefaultStrategy, Visualizer
)

# 开启日志查看布线过程
logging.basicConfig(level=logging.INFO)

# 1. 构建三层网格
vias = []
for x in range(10):
    for y in range(10):
        vias.append((("M0", x, y), ("M1", x, y), 2.0))
        vias.append((("M1", x, y), ("M2", x, y), 2.0))

grid = RoutingGrid.build_grid(
    layers=["M0", "M1", "M2"],
    width=10,
    height=10,
    via_connections=vias,
)

# 2. 定义线网
nets = [
    Net("VDD", [("M0", 0, 5), ("M0", 9, 5)]),
    Net("GND", [("M0", 0, 0), ("M0", 9, 0), ("M0", 5, 9)]),
    Net("CLK", [("M0", 3, 0), ("M0", 3, 9), ("M0", 7, 5)]),
]

# 3. 设置间距约束
spacing_mgr = SpacingManager({"M0": 1, "M1": 1, "M2": 1})

# 4. 选择策略并执行布线
strategy = DefaultStrategy(max_iterations=50)
manager = RipupManager(grid, spacing_mgr, strategy)
solution = manager.run(nets)

# 5. 查看结果
print(f"布通率: {solution.routed_count}/{len(nets)}")
print(f"总代价: {solution.total_cost:.2f}")

# 6. 可视化
vis = Visualizer(grid)
vis.print_solution(solution, spacing_mgr, show_spacing=True)
```

## 3. 模块结构

```
maze_router/
├── __init__.py         # 包入口，导出所有公开类
├── net.py              # 数据结构：Net, RoutingResult, RoutingSolution
├── grid.py             # RoutingGrid - 网格封装
├── spacing.py          # SpacingManager - 间距约束管理
├── router.py           # MazeRouter - 核心迷宫路由引擎
├── steiner.py          # SteinerTreeBuilder - 多端口Steiner树构建
├── strategy.py         # RoutingStrategy, DefaultStrategy, CongestionAwareStrategy
├── ripup.py            # RipupManager - 拆线重布管理
└── visualizer.py       # Visualizer - 文本可视化
```

## 4. 核心API

### 4.1 RoutingGrid

```python
# 方式一：从已有的nx.Graph创建
grid = RoutingGrid(graph)

# 方式二：使用便捷方法构建标准网格
grid = RoutingGrid.build_grid(
    layers=["M0", "M1", "M2"],   # 层名称
    width=10,                     # x方向尺寸
    height=10,                    # y方向尺寸
    via_connections=vias,          # [(node1, node2, cost), ...]
    removed_nodes=None,            # 不存在的节点集合
    removed_edges=None,            # 不存在的边集合
    default_cost=1.0,             # 同层边默认代价
    default_via_cost=2.0,         # via默认代价
)
```

### 4.2 Net

```python
# 基本线网
net = Net("net_name", terminals=[("M0", 0, 0), ("M0", 5, 5)])

# 带cable_locs约束的线网
allowed_m0 = {("M0", x, 0) for x in range(10)}
net = Net("net_name", terminals=[("M0", 0, 0), ("M0", 9, 0)],
          cable_locs=allowed_m0)
```

### 4.3 SpacingManager

```python
# 创建：指定每层最小间距
spacing_mgr = SpacingManager({"M0": 1, "M1": 1, "M2": 2})

# 标记/解除标记（通常由RipupManager自动管理）
spacing_mgr.mark_route("net1", routed_nodes)
spacing_mgr.unmark_route("net1")

# 查询
spacing_mgr.is_available(node, "net2")     # 节点对net2是否可用
spacing_mgr.get_blocking_nets(node, "net2") # 阻塞节点的线网
```

### 4.4 策略

```python
# 使用默认策略
strategy = DefaultStrategy(max_iterations=50)

# 使用拥塞感知策略
strategy = CongestionAwareStrategy(
    max_iterations=50,
    congestion_weight=0.5,  # 拥塞代价权重
)

# 自定义策略：继承RoutingStrategy
from maze_router.strategy import RoutingStrategy, RipupDecision, RipupAction

class MyStrategy(RoutingStrategy):
    def order_nets(self, nets):
        # 按包围盒面积从大到小排序（关键线网优先）
        return sorted(nets, key=lambda n: -n.bounding_box_area())

    def order_terminals(self, net, tree_nodes):
        return list(net.terminals)  # 保持原始顺序

    def get_cost_multiplier(self, node, net_name):
        # 鼓励使用M1层
        if node[0] == "M1":
            return 0.8
        return 1.0

    def decide_ripup(self, failed_net, blocking_nets, net_results, iteration):
        if blocking_nets:
            return RipupDecision(
                action=RipupAction.RIPUP_SINGLE,
                nets_to_ripup=[list(blocking_nets)[0]],
            )
        return RipupDecision(action=RipupAction.SKIP, nets_to_ripup=[])

    def get_max_iterations(self):
        return 100
```

### 4.5 执行布线

```python
manager = RipupManager(grid, spacing_mgr, strategy)
solution = manager.run(nets)

# 检查结果
solution.all_routed        # bool: 是否全部布通
solution.routed_count      # int: 布通线网数量
solution.total_cost        # float: 总代价
solution.failed_nets       # List[str]: 未布通的线网名称

# 获取单个线网的结果
result = solution.get_result("net1")
result.success             # bool
result.routed_nodes        # Set[Node]: 布线节点
result.routed_edges        # List[Edge]: 布线边
result.total_cost          # float
```

## 5. 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试
python -m pytest tests/test_basic.py -v          # 基础测试
python -m pytest tests/test_spacing.py -v         # 间距约束测试
python -m pytest tests/test_multi_terminal.py -v  # 多端口测试
python -m pytest tests/test_ripup.py -v           # 拆线重布测试
python -m pytest tests/test_complex.py -v         # 复杂场景测试
```

## 6. 使用外部nx.Graph

如果版图前处理已经生成了nx.Graph，可以直接使用：

```python
import networkx as nx
from maze_router import RoutingGrid

# 假设 G 是前处理得到的图
G = nx.Graph()
G.add_node(("M0", 0, 0))
G.add_node(("M0", 1, 0))
G.add_edge(("M0", 0, 0), ("M0", 1, 0), cost=1.0)
# ... 添加更多节点和边

# 直接包装
grid = RoutingGrid(G)
```

注意：G中的节点必须为 `(layer, x, y)` 三元组格式，边必须有 `cost` 属性。
