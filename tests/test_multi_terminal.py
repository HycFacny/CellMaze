"""
多端口Steiner树测试

测试多端口（3个以上）线网的布线和跨层路由。
"""

import pytest
from maze_router.net import Net
from maze_router.grid import RoutingGrid
from maze_router.spacing import SpacingManager
from maze_router.steiner import SteinerTreeBuilder
from maze_router.strategy import DefaultStrategy


def make_3layer_grid(width=10, height=10):
    """创建3层网格，带完整via"""
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


class TestMultiTerminal:
    """多端口Steiner树测试"""

    def test_four_terminal_steiner(self):
        """测试四端口Steiner树"""
        grid = RoutingGrid.build_grid(layers=["M0"], width=10, height=10)
        spacing_mgr = SpacingManager({"M0": 0})
        builder = SteinerTreeBuilder(grid, spacing_mgr)

        terminals = [("M0", 0, 0), ("M0", 9, 0), ("M0", 0, 9), ("M0", 9, 9)]
        net = Net("net1", terminals)
        strategy = DefaultStrategy()
        order = strategy.order_terminals(net, set())

        result = builder.build_tree(net, order)
        assert result.success
        for t in terminals:
            assert t in result.routed_nodes

    def test_five_terminal_cross_layer(self):
        """测试五端口跨层Steiner树"""
        grid = make_3layer_grid()
        spacing_mgr = SpacingManager({"M0": 0, "M1": 0, "M2": 0})
        builder = SteinerTreeBuilder(grid, spacing_mgr)

        # 端口分布在不同层
        terminals = [
            ("M0", 0, 0),
            ("M0", 9, 0),
            ("M1", 5, 5),
            ("M2", 0, 9),
            ("M2", 9, 9),
        ]
        net = Net("net1", terminals)
        strategy = DefaultStrategy()
        order = strategy.order_terminals(net, set())

        result = builder.build_tree(net, order)
        assert result.success
        for t in terminals:
            assert t in result.routed_nodes

    def test_cable_locs_constraint(self):
        """测试M0层cable_locs约束"""
        grid = RoutingGrid.build_grid(layers=["M0"], width=10, height=5)
        spacing_mgr = SpacingManager({"M0": 0})
        builder = SteinerTreeBuilder(grid, spacing_mgr)

        # 只允许在y=0和y=4行走线
        cable_locs = set()
        for x in range(10):
            cable_locs.add(("M0", x, 0))
            cable_locs.add(("M0", x, 4))
        # 加上x=0和x=9列
        for y in range(5):
            cable_locs.add(("M0", 0, y))
            cable_locs.add(("M0", 9, y))

        terminals = [("M0", 0, 0), ("M0", 9, 4)]
        net = Net("net1", terminals, cable_locs=cable_locs)

        result = builder.build_tree(net, list(net.terminals))
        assert result.success

        # 验证路径上M0层节点都在cable_locs中（端口除外）
        for node in result.routed_nodes:
            if node[0] == "M0":
                assert node in cable_locs or node in set(terminals), \
                    f"节点 {node} 不在cable_locs中"

    def test_via_cost_affects_routing(self):
        """测试via代价对路由的影响"""
        # 创建via代价很高的网格
        vias = []
        for x in range(5):
            for y in range(5):
                vias.append((("M0", x, y), ("M1", x, y), 100.0))

        grid = RoutingGrid.build_grid(
            layers=["M0", "M1"],
            width=5,
            height=5,
            via_connections=vias,
        )
        spacing_mgr = SpacingManager({"M0": 0, "M1": 0})

        from maze_router.router import MazeRouter
        router = MazeRouter(grid, spacing_mgr)

        # 同层路由应该比跨层路由便宜
        # M0上两点距离4
        same_layer = router.route(
            sources={("M0", 0, 0)},
            targets={("M0", 4, 0)},
            net_name="net1",
        )
        # 跨层后在M1上走再跨回来
        cross_layer = router.route(
            sources={("M0", 0, 0)},
            targets={("M1", 4, 0)},
            net_name="net1",
        )

        assert same_layer.success
        assert cross_layer.success
        assert same_layer.cost < cross_layer.cost


class TestSteinerTreeDP:
    """Dreyfus-Wagner动态规划Steiner树测试"""

    def test_three_terminal_dp(self):
        """测试三端口DP Steiner树"""
        grid = RoutingGrid.build_grid(layers=["M0"], width=10, height=10)
        spacing_mgr = SpacingManager({"M0": 0})
        builder = SteinerTreeBuilder(grid, spacing_mgr)

        terminals = [("M0", 0, 0), ("M0", 9, 0), ("M0", 5, 9)]
        net = Net("net1", terminals)

        result = builder.build_tree_dp(net)
        assert result.success
        for t in terminals:
            assert t in result.routed_nodes

    def test_four_terminal_dp(self):
        """测试四端口DP Steiner树——四角情况"""
        grid = RoutingGrid.build_grid(layers=["M0"], width=10, height=10)
        spacing_mgr = SpacingManager({"M0": 0})
        builder = SteinerTreeBuilder(grid, spacing_mgr)

        terminals = [("M0", 0, 0), ("M0", 9, 0), ("M0", 0, 9), ("M0", 9, 9)]
        net = Net("net1", terminals)

        result = builder.build_tree_dp(net)
        assert result.success
        for t in terminals:
            assert t in result.routed_nodes

    def test_dp_vs_greedy_optimality(self):
        """验证DP结果代价 ≤ 贪心结果代价（DP应不劣于贪心）"""
        grid = RoutingGrid.build_grid(layers=["M0"], width=8, height=8)
        spacing_mgr = SpacingManager({"M0": 0})
        builder = SteinerTreeBuilder(grid, spacing_mgr)

        # 选择一个贪心容易走弯路的端口布局
        terminals = [("M0", 0, 0), ("M0", 7, 0), ("M0", 4, 7)]
        net = Net("net1", terminals)

        strategy = DefaultStrategy()
        order = strategy.order_terminals(net, set())
        greedy_result = builder.build_tree(net, order)

        dp_result = builder.build_tree_dp(net)

        assert greedy_result.success
        assert dp_result.success
        # DP最优解代价不应超过贪心近似解
        assert dp_result.total_cost <= greedy_result.total_cost + 1e-9, (
            f"DP代价 {dp_result.total_cost} > 贪心代价 {greedy_result.total_cost}"
        )

    def test_dp_five_terminal_cross_layer(self):
        """测试五端口跨层DP Steiner树"""
        grid = make_3layer_grid(width=8, height=8)
        spacing_mgr = SpacingManager({"M0": 0, "M1": 0, "M2": 0})
        builder = SteinerTreeBuilder(grid, spacing_mgr)

        terminals = [
            ("M0", 0, 0),
            ("M0", 7, 0),
            ("M1", 4, 4),
            ("M2", 0, 7),
            ("M2", 7, 7),
        ]
        net = Net("net1", terminals)

        result = builder.build_tree_dp(net)
        assert result.success
        for t in terminals:
            assert t in result.routed_nodes

    def test_dp_with_spacing_constraint(self):
        """测试DP方法遵守间距约束"""
        grid = RoutingGrid.build_grid(layers=["M0"], width=10, height=5)
        spacing_mgr = SpacingManager({"M0": 1})

        # 先标记另一条线网的占用，制造障碍
        spacing_mgr.mark_route(
            "other_net",
            {("M0", x, 2) for x in range(10)}
        )

        builder = SteinerTreeBuilder(grid, spacing_mgr)

        terminals = [("M0", 0, 0), ("M0", 5, 0), ("M0", 9, 0)]
        net = Net("net1", terminals)

        result = builder.build_tree_dp(net)
        assert result.success
        # 验证所有节点都满足间距
        for node in result.routed_nodes:
            assert spacing_mgr.is_available(node, "net1"), \
                f"节点 {node} 违反间距约束"

    def test_dp_with_cable_locs(self):
        """测试DP方法遵守cable_locs约束"""
        grid = RoutingGrid.build_grid(layers=["M0"], width=8, height=8)
        spacing_mgr = SpacingManager({"M0": 0})
        builder = SteinerTreeBuilder(grid, spacing_mgr)

        # 限定只能走边框
        cable_locs = set()
        for x in range(8):
            cable_locs.add(("M0", x, 0))
            cable_locs.add(("M0", x, 7))
        for y in range(8):
            cable_locs.add(("M0", 0, y))
            cable_locs.add(("M0", 7, y))

        terminals = [("M0", 0, 0), ("M0", 7, 0), ("M0", 7, 7)]
        net = Net("net1", terminals, cable_locs=cable_locs)

        result = builder.build_tree_dp(net)
        assert result.success
        for node in result.routed_nodes:
            if node[0] == "M0":
                assert node in cable_locs, f"节点 {node} 不在cable_locs中"

    def test_dp_two_terminal_fallback(self):
        """测试两端口时DP退化为最短路径（走build_tree快速路径）"""
        grid = RoutingGrid.build_grid(layers=["M0"], width=5, height=5)
        spacing_mgr = SpacingManager({"M0": 0})
        builder = SteinerTreeBuilder(grid, spacing_mgr)

        net = Net("net1", [("M0", 0, 0), ("M0", 4, 4)])
        result = builder.build_tree_dp(net)
        assert result.success
        assert result.total_cost == 8.0

    def test_dp_impossible(self):
        """测试端口不可达时DP优雅失败"""
        removed = {("M0", 2, y) for y in range(5)}
        grid = RoutingGrid.build_grid(
            layers=["M0"], width=5, height=5,
            removed_nodes=removed,
        )
        spacing_mgr = SpacingManager({"M0": 0})
        builder = SteinerTreeBuilder(grid, spacing_mgr)

        terminals = [("M0", 0, 0), ("M0", 4, 0), ("M0", 0, 4)]
        net = Net("net1", terminals)

        result = builder.build_tree_dp(net)
        assert not result.success
