"""
基础布线测试

测试单线网、两端口的基本迷宫路由功能。
"""

import pytest
import networkx as nx

from maze_router.net import Net, Node
from maze_router.grid import RoutingGrid
from maze_router.spacing import SpacingManager
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
        """测试跨网格路由"""
        grid = make_simple_grid()
        spacing_mgr = SpacingManager({"M0": 0})
        router = MazeRouter(grid, spacing_mgr)

        result = router.route(
            sources={("M0", 0, 0)},
            targets={("M0", 4, 4)},
            net_name="net1",
        )
        assert result.success
        # 最短路径长度应为 4+4 = 8 步
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
