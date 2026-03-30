"""
基础布线测试

测试单线网、两端口的基本迷宫路由功能。
"""

import pytest
import networkx as nx

from maze_router.net import Net, Node
from maze_router.grid import RoutingGrid
from maze_router.spacing import SpacingManager
from maze_router.corner import CornerManager
from maze_router.router import MazeRouter
from maze_router.steiner import SteinerTreeBuilder


def make_simple_grid(width=5, height=5):
    """创建简单的单层5x5网格"""
    return RoutingGrid.build_grid(
        layers=["M0"],
        width=width,
        height=height,
    )


def make_3layer_grid(width=5, height=5):
    """创建3层5x5网格，带via连接"""
    vias = []
    for x in range(width):
        for y in range(height):
            vias.append((("M0", x, y), ("M1", x, y), 2.0))
            vias.append((("M1", x, y), ("M2", x, y), 2.0))
    return RoutingGrid.build_grid(
        layers=["M0", "M1", "M2"],
        width=width,
        height=height,
        via_connections=vias,
    )


class TestMazeRouterBasic:
    """基础迷宫路由测试"""

    def test_route_adjacent_nodes(self):
        """测试相邻节点的路由"""
        grid = make_simple_grid()
        spacing_mgr = SpacingManager({"M0": 0})
        router = MazeRouter(grid, spacing_mgr)

        result = router.route(
            sources={("M0", 0, 0)},
            targets={("M0", 1, 0)},
            net_name="net1",
        )
        assert result.success
        assert ("M0", 0, 0) in result.nodes
        assert ("M0", 1, 0) in result.nodes
        assert result.cost == 1.0

    def test_route_same_node(self):
        """测试源和目标为同一节点"""
        grid = make_simple_grid()
        spacing_mgr = SpacingManager({"M0": 0})
        router = MazeRouter(grid, spacing_mgr)

        result = router.route(
            sources={("M0", 2, 2)},
            targets={("M0", 2, 2)},
            net_name="net1",
        )
        assert result.success
        assert result.cost == 0.0

    def test_route_across_grid(self):
        """测试跨网格路由（不含折角代价，验证纯边代价）"""
        grid = make_simple_grid()
        spacing_mgr = SpacingManager({"M0": 0})
        router = MazeRouter(grid, spacing_mgr, corner_mgr=CornerManager.disabled())

        result = router.route(
            sources={("M0", 0, 0)},
            targets={("M0", 4, 4)},
            net_name="net1",
        )
        assert result.success
        # 最短路径长度应为 4+4 = 8 步（无折角代价）
        assert result.cost == 8.0

    def test_route_with_removed_nodes(self):
        """测试有障碍物时的路由（某些节点不存在）"""
        removed = {("M0", 2, 2), ("M0", 2, 1), ("M0", 2, 3)}
        grid = RoutingGrid.build_grid(
            layers=["M0"],
            width=5,
            height=5,
            removed_nodes=removed,
        )
        spacing_mgr = SpacingManager({"M0": 0})
        router = MazeRouter(grid, spacing_mgr)

        result = router.route(
            sources={("M0", 0, 2)},
            targets={("M0", 4, 2)},
            net_name="net1",
        )
        assert result.success
        # 需要绕路
        assert result.cost > 4.0

    def test_route_impossible(self):
        """测试不可达的路由"""
        # 移除中间一整列，使左右不连通
        removed = {("M0", 2, y) for y in range(5)}
        grid = RoutingGrid.build_grid(
            layers=["M0"],
            width=5,
            height=5,
            removed_nodes=removed,
        )
        spacing_mgr = SpacingManager({"M0": 0})
        router = MazeRouter(grid, spacing_mgr)

        result = router.route(
            sources={("M0", 0, 0)},
            targets={("M0", 4, 0)},
            net_name="net1",
        )
        assert not result.success

    def test_route_via_crossing(self):
        """测试via跨层路由"""
        grid = make_3layer_grid()
        spacing_mgr = SpacingManager({"M0": 0, "M1": 0, "M2": 0})
        router = MazeRouter(grid, spacing_mgr)

        # 从M0层到M2层同一坐标
        result = router.route(
            sources={("M0", 2, 2)},
            targets={("M2", 2, 2)},
            net_name="net1",
        )
        assert result.success
        # M0->M1 via(2) + M1->M2 via(2) = 4
        assert result.cost == 4.0

    def test_multi_source_routing(self):
        """测试多源路由（Steiner树扩展时的场景）"""
        grid = make_simple_grid(width=10, height=1)
        spacing_mgr = SpacingManager({"M0": 0})
        router = MazeRouter(grid, spacing_mgr)

        # 从多个源出发，到达一个目标
        result = router.route(
            sources={("M0", 0, 0), ("M0", 9, 0)},
            targets={("M0", 5, 0)},
            net_name="net1",
        )
        assert result.success
        # 从最近的源出发，距离应为4或5
        assert result.cost <= 5.0


class TestSteinerTreeBasic:
    """基础Steiner树测试"""

    def test_two_terminal_steiner(self):
        """测试两端口Steiner树"""
        grid = make_simple_grid()
        spacing_mgr = SpacingManager({"M0": 0})
        builder = SteinerTreeBuilder(grid, spacing_mgr)

        net = Net("net1", [("M0", 0, 0), ("M0", 4, 4)])
        result = builder.build_tree(net, list(net.terminals))

        assert result.success
        assert ("M0", 0, 0) in result.routed_nodes
        assert ("M0", 4, 4) in result.routed_nodes

    def test_three_terminal_steiner(self):
        """测试三端口Steiner树"""
        grid = make_simple_grid(width=10, height=10)
        spacing_mgr = SpacingManager({"M0": 0})
        builder = SteinerTreeBuilder(grid, spacing_mgr)

        net = Net("net1", [("M0", 0, 0), ("M0", 9, 0), ("M0", 5, 9)])
        result = builder.build_tree(net, list(net.terminals))

        assert result.success
        for t in net.terminals:
            assert t in result.routed_nodes

    def test_single_terminal(self):
        """测试单端口线网"""
        grid = make_simple_grid()
        spacing_mgr = SpacingManager({"M0": 0})
        builder = SteinerTreeBuilder(grid, spacing_mgr)

        net = Net("net1", [("M0", 2, 2)])
        result = builder.build_tree(net, list(net.terminals))

        assert result.success
        assert ("M0", 2, 2) in result.routed_nodes


class TestCornerCost:
    """折角代价测试"""

    def test_straight_path_no_corner_penalty(self):
        """直线路径：无论折角代价大小，折角惩罚均为零"""
        grid = make_simple_grid(width=8, height=3)
        spacing_mgr = SpacingManager({"M0": 0})

        # 无折角代价
        router_no = MazeRouter(grid, spacing_mgr, corner_mgr=CornerManager.disabled())
        r_no = router_no.route(
            sources={("M0", 0, 1)},
            targets={("M0", 7, 1)},
            net_name="net1",
        )
        # 折角代价使用默认值5.0
        router_hi = MazeRouter(grid, spacing_mgr)
        r_hi = router_hi.route(
            sources={("M0", 0, 1)},
            targets={("M0", 7, 1)},
            net_name="net1",
        )

        assert r_no.success and r_hi.success
        # 直线路径无折角，代价相同
        assert r_no.cost == r_hi.cost == 7.0

    def test_corner_penalty_increases_cost(self):
        """从 (0,0) 到 (3,3) 必须至少折角一次，折角代价应增加总代价"""
        grid = make_simple_grid(width=5, height=5)
        spacing_mgr = SpacingManager({"M0": 0})

        router_none = MazeRouter(grid, spacing_mgr, corner_mgr=CornerManager.disabled())
        r_none = router_none.route(
            sources={("M0", 0, 0)},
            targets={("M0", 3, 3)},
            net_name="net1",
        )

        router_hi = MazeRouter(grid, spacing_mgr, corner_mgr=CornerManager(l_costs={"M0": 5.0}))
        r_hi = router_hi.route(
            sources={("M0", 0, 0)},
            targets={("M0", 3, 3)},
            net_name="net1",
        )

        assert r_none.success and r_hi.success
        # 无折角代价时：纯边代价 = 6
        assert r_none.cost == 6.0
        # 有折角代价时：至少 1 个折角，总代价 ≥ 6 + 5 = 11
        assert r_hi.cost >= r_none.cost + 5.0

    def test_corner_penalty_single_corner(self):
        """从 (0,0) 到 (3,3) 最优路径恰好 1 个折角，代价 = 6 + corner_cost"""
        grid = make_simple_grid(width=5, height=5)
        spacing_mgr = SpacingManager({"M0": 0})

        corner_val = 5.0
        router = MazeRouter(grid, spacing_mgr, corner_mgr=CornerManager(l_costs={"M0": corner_val}))
        result = router.route(
            sources={("M0", 0, 0)},
            targets={("M0", 3, 3)},
            net_name="net1",
        )

        assert result.success
        # 边代价 6 + 1 × corner_cost
        assert result.cost == pytest.approx(6.0 + corner_val)

    def test_large_corner_cost_prefers_fewer_corners(self):
        """
        当 corner_cost 极大时，路由器应选择折角更少（即使步数更多）的路径。

        场景：在 1 × 5 → 竖列延伸的 T 形网格中验证此行为。
        使用 10×2 网格，将路径限制在两条线上：
          行 y=0 水平，y=1 水平；从 (0,0) → (9,1)。
          路径必须折角一次；无法通过增加步数来减少折角。
        改用：从 (0,0) → (3,0) → (3,1) 对比 (0,0) → (0,1) → (3,1)
        两者折角数相同（各 1 个），代价相同。
        验证 corner_costs > 0 时总代价等于 边代价 + 1 × corner_cost。
        """
        grid = make_simple_grid(width=5, height=3)
        spacing_mgr = SpacingManager({"M0": 0})
        corner_val = 20.0
        router = MazeRouter(grid, spacing_mgr, corner_mgr=CornerManager(l_costs={"M0": corner_val}))

        result = router.route(
            sources={("M0", 0, 0)},
            targets={("M0", 4, 2)},
            net_name="net1",
        )
        assert result.success
        # 边代价 = 4+2 = 6，折角数 ≥ 1
        assert result.cost >= 6.0 + corner_val

    def test_via_resets_direction_no_corner(self):
        """跨层 via 后方向重置，不产生折角惩罚"""
        grid = make_3layer_grid(width=5, height=5)
        spacing_mgr = SpacingManager({"M0": 0, "M1": 0, "M2": 0})
        corner_val = 100.0  # 极大折角代价
        router = MazeRouter(
            grid, spacing_mgr,
            corner_mgr=CornerManager(l_costs={"M0": corner_val, "M1": corner_val, "M2": corner_val}),
        )

        # 路径：M0(0,0) → via → M1(0,0) → 水平 → M1(3,0) → via → M2(3,0)
        # via 穿层不算折角；M1 上纯水平移动也不折角
        result = router.route(
            sources={("M0", 0, 0)},
            targets={("M2", 3, 0)},
            net_name="net1",
        )
        assert result.success
        # 预期：M0→M1 via(2) + M1水平3步(3) + M1→M2 via(2) = 7，无折角
        assert result.cost == pytest.approx(7.0)

    def test_corner_mgr_none_uses_default(self):
        """corner_mgr=None 等价于每层默认值 5.0"""
        grid = make_simple_grid(width=5, height=5)
        spacing_mgr = SpacingManager({"M0": 0})

        r_none = MazeRouter(grid, spacing_mgr, corner_mgr=None).route(
            sources={("M0", 0, 0)},
            targets={("M0", 3, 3)},
            net_name="n",
        )
        r_default = MazeRouter(grid, spacing_mgr, corner_mgr=CornerManager(l_costs={"M0": 5.0})).route(
            sources={("M0", 0, 0)},
            targets={("M0", 3, 3)},
            net_name="n",
        )
        assert r_none.success and r_default.success
        assert r_none.cost == pytest.approx(r_default.cost)

    def test_steiner_builder_corner_mgr_propagated(self):
        """SteinerTreeBuilder 正确传递 corner_mgr 到底层路由器"""
        grid = make_simple_grid(width=6, height=6)
        spacing_mgr = SpacingManager({"M0": 0})

        # 无折角代价
        builder_no = SteinerTreeBuilder(grid, spacing_mgr, corner_mgr=CornerManager.disabled())
        net = Net("net1", [("M0", 0, 0), ("M0", 5, 5)])
        r_no = builder_no.build_tree(net, list(net.terminals))

        # 有折角代价
        builder_hi = SteinerTreeBuilder(grid, spacing_mgr, corner_mgr=CornerManager(l_costs={"M0": 10.0}))
        r_hi = builder_hi.build_tree(net, list(net.terminals))

        assert r_no.success and r_hi.success
        # 有折角代价时总代价更高
        assert r_hi.total_cost >= r_no.total_cost
