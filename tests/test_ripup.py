"""
拆线重布测试

测试当多个线网发生冲突时的拆线重布机制。
"""

import logging
import pytest

from maze_router.net import Net
from maze_router.grid import RoutingGrid
from maze_router.spacing import SpacingManager
from maze_router.ripup import RipupManager
from maze_router.strategy import DefaultStrategy, CongestionAwareStrategy


def make_grid(width=10, height=5):
    """创建单层网格"""
    return RoutingGrid.build_grid(layers=["M0"], width=width, height=height)


class TestRipupReroute:
    """拆线重布测试"""

    def test_two_nets_no_conflict(self):
        """测试两个不冲突的线网"""
        grid = make_grid(width=10, height=10)
        spacing_mgr = SpacingManager({"M0": 1})
        strategy = DefaultStrategy()
        manager = RipupManager(grid, spacing_mgr, strategy)

        nets = [
            Net("netA", [("M0", 0, 0), ("M0", 9, 0)]),
            Net("netB", [("M0", 0, 9), ("M0", 9, 9)]),
        ]

        solution = manager.run(nets)
        assert solution.all_routed

    def test_two_nets_crossing(self):
        """测试两个必须交叉的线网，需要利用可用空间绕路"""
        grid = make_grid(width=10, height=10)
        spacing_mgr = SpacingManager({"M0": 1})
        strategy = DefaultStrategy()
        manager = RipupManager(grid, spacing_mgr, strategy)

        # 两条线网需要交叉
        nets = [
            Net("netA", [("M0", 0, 5), ("M0", 9, 5)]),
            Net("netB", [("M0", 5, 0), ("M0", 5, 9)]),
        ]

        solution = manager.run(nets)
        # 在10x10网格上，间距为1，两条交叉线网应该能通过绕路布通
        assert solution.routed_count >= 1  # 至少一个能布通

    def test_ripup_improves_routing(self):
        """测试多层网格下拆线重布能改善结果"""
        # 使用三层网格，给拆线重布留出跨层绕行空间
        vias = []
        for x in range(15):
            for y in range(15):
                vias.append((("M0", x, y), ("M1", x, y), 2.0))
                vias.append((("M1", x, y), ("M2", x, y), 2.0))
        grid = RoutingGrid.build_grid(
            layers=["M0", "M1", "M2"],
            width=15,
            height=15,
            via_connections=vias,
        )
        spacing_mgr = SpacingManager({"M0": 1, "M1": 1, "M2": 1})
        strategy = DefaultStrategy(max_iterations=30)
        manager = RipupManager(grid, spacing_mgr, strategy)

        nets = [
            Net("netA", [("M0", 0, 7), ("M0", 14, 7)]),
            Net("netB", [("M0", 7, 0), ("M0", 7, 14)]),
            Net("netC", [("M0", 0, 0), ("M0", 14, 14)]),
        ]

        solution = manager.run(nets)
        assert solution.routed_count >= 2  # 至少两个布通

    def test_congestion_aware_strategy(self):
        """测试拥塞感知策略"""
        grid = make_grid(width=15, height=15)
        spacing_mgr = SpacingManager({"M0": 1})
        strategy = CongestionAwareStrategy(max_iterations=30, congestion_weight=0.3)
        manager = RipupManager(grid, spacing_mgr, strategy)

        nets = [
            Net("net1", [("M0", 0, 7), ("M0", 14, 7)]),
            Net("net2", [("M0", 7, 0), ("M0", 7, 14)]),
        ]

        solution = manager.run(nets)
        assert solution.routed_count >= 1

    def test_graceful_failure(self):
        """测试不可能布通时的优雅失败"""
        # 极小网格，间距约束很紧
        grid = make_grid(width=3, height=3)
        spacing_mgr = SpacingManager({"M0": 1})
        strategy = DefaultStrategy(max_iterations=5)
        manager = RipupManager(grid, spacing_mgr, strategy)

        # 在3x3网格间距为1时，不可能布通两个都需要跨越网格的线网
        nets = [
            Net("netA", [("M0", 0, 0), ("M0", 2, 2)]),
            Net("netB", [("M0", 2, 0), ("M0", 0, 2)]),
        ]

        solution = manager.run(nets)
        # 不要求全部布通，但不应崩溃
        assert len(solution.results) == 2

    def test_multi_layer_ripup(self):
        """测试多层网格的拆线重布"""
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
        spacing_mgr = SpacingManager({"M0": 1, "M1": 1, "M2": 1})
        strategy = DefaultStrategy(max_iterations=30)
        manager = RipupManager(grid, spacing_mgr, strategy)

        nets = [
            Net("netA", [("M0", 0, 5), ("M0", 9, 5)]),
            Net("netB", [("M0", 5, 0), ("M0", 5, 9)]),
            Net("netC", [("M0", 0, 0), ("M0", 9, 9)]),
        ]

        solution = manager.run(nets)
        # 三层网格有更多空间，应该能布通更多线网
        assert solution.routed_count >= 2
