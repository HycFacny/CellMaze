"""
复杂测试场景

包含8个线网、多层网格、各种终端数量的综合测试。
"""

import logging
import pytest

from maze_router.net import Net
from maze_router.grid import RoutingGrid
from maze_router.spacing import SpacingManager
from maze_router.corner import CornerManager
from maze_router.ripup import RipupManager
from maze_router.strategy import DefaultStrategy, CongestionAwareStrategy
from maze_router.visualizer import Visualizer


def make_complex_grid(width=15, height=15):
    """创建15x15的三层网格，带完整via连接"""
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


class TestComplexScenario:
    """复杂场景综合测试"""

    def test_8_nets_basic(self):
        """8个线网，每个2个端口，在15x15x3网格上"""
        grid = make_complex_grid()
        spacing_mgr = SpacingManager({"M0": 1, "M1": 1, "M2": 1})
        strategy = DefaultStrategy(max_iterations=50)
        manager = RipupManager(
            grid, spacing_mgr, strategy,
            corner_mgr=CornerManager(l_costs={"M0": 5.0, "M1": 5.0, "M2": 5.0}),
        )

        nets = [
            Net("net1", [("M0", 0, 0), ("M0", 14, 0)]),
            Net("net2", [("M0", 0, 2), ("M0", 14, 2)]),
            Net("net3", [("M0", 0, 4), ("M0", 14, 4)]),
            Net("net4", [("M0", 0, 7), ("M0", 14, 7)]),
            Net("net5", [("M0", 0, 10), ("M0", 14, 10)]),
            Net("net6", [("M0", 0, 12), ("M0", 14, 12)]),
            Net("net7", [("M0", 0, 14), ("M0", 14, 14)]),
            Net("net8", [("M0", 7, 0), ("M0", 7, 14)]),
        ]

        solution = manager.run(nets)
        assert solution.routed_count >= 6, f"仅布通 {solution.routed_count}/8"

    def test_8_nets_multi_terminal(self):
        """8个线网，端口数量从2到5不等"""
        grid = make_complex_grid(width=20, height=20)
        spacing_mgr = SpacingManager({"M0": 1, "M1": 1, "M2": 1})
        strategy = DefaultStrategy(max_iterations=50)
        manager = RipupManager(grid, spacing_mgr, strategy)

        nets = [
            # 2端口
            Net("net1", [("M0", 0, 0), ("M0", 19, 0)]),
            Net("net2", [("M0", 0, 19), ("M0", 19, 19)]),
            # 3端口
            Net("net3", [("M0", 0, 5), ("M0", 10, 5), ("M0", 19, 5)]),
            Net("net4", [("M0", 0, 14), ("M0", 10, 14), ("M0", 19, 14)]),
            # 4端口
            Net("net5", [("M0", 5, 0), ("M0", 5, 7), ("M0", 5, 13), ("M0", 5, 19)]),
            Net("net6", [("M0", 14, 0), ("M0", 14, 7), ("M0", 14, 13), ("M0", 14, 19)]),
            # 5端口
            Net("net7", [
                ("M0", 3, 3), ("M0", 3, 10), ("M0", 10, 10),
                ("M0", 16, 3), ("M0", 16, 16),
            ]),
            Net("net8", [
                ("M0", 7, 2), ("M0", 7, 8), ("M0", 12, 8),
                ("M0", 12, 17), ("M0", 17, 10),
            ]),
        ]

        solution = manager.run(nets)
        assert solution.routed_count >= 3, f"仅布通 {solution.routed_count}/8"

    def test_8_nets_with_cable_locs(self):
        """8个线网，部分有cable_locs约束"""
        grid = make_complex_grid()
        spacing_mgr = SpacingManager({"M0": 1, "M1": 1, "M2": 1})
        strategy = DefaultStrategy(max_iterations=30)
        manager = RipupManager(grid, spacing_mgr, strategy)

        # 为部分线网设定cable_locs
        even_row_locs = set()
        for x in range(15):
            for y in range(0, 15, 2):
                even_row_locs.add(("M0", x, y))

        odd_row_locs = set()
        for x in range(15):
            for y in range(1, 15, 2):
                odd_row_locs.add(("M0", x, y))

        nets = [
            Net("net1", [("M0", 0, 0), ("M0", 14, 0)], cable_locs=even_row_locs),
            Net("net2", [("M0", 0, 2), ("M0", 14, 2)], cable_locs=even_row_locs),
            Net("net3", [("M0", 0, 1), ("M0", 14, 1)], cable_locs=odd_row_locs),
            Net("net4", [("M0", 0, 3), ("M0", 14, 3)], cable_locs=odd_row_locs),
            Net("net5", [("M0", 0, 4), ("M0", 14, 8)]),
            Net("net6", [("M0", 0, 10), ("M0", 14, 14)]),
            Net("net7", [("M0", 7, 0), ("M0", 7, 14)]),
            Net("net8", [("M0", 0, 7), ("M0", 14, 7)]),
        ]

        solution = manager.run(nets)
        # 有cable_locs约束会让布线更困难，区域拆线策略变化可能影响结果
        assert solution.routed_count >= 2

    def test_congestion_aware_8_nets(self):
        """使用拥塞感知策略的8线网测试"""
        grid = make_complex_grid(width=20, height=20)
        spacing_mgr = SpacingManager({"M0": 1, "M1": 1, "M2": 1})
        strategy = CongestionAwareStrategy(max_iterations=50, congestion_weight=0.5)
        manager = RipupManager(
            grid, spacing_mgr, strategy,
            corner_mgr=CornerManager(l_costs={"M0": 5.0, "M1": 5.0, "M2": 5.0}),
        )

        nets = [
            Net("net1", [("M0", 0, 0), ("M0", 19, 0)]),
            Net("net2", [("M0", 0, 4), ("M0", 19, 4)]),
            Net("net3", [("M0", 0, 8), ("M0", 19, 8)]),
            Net("net4", [("M0", 0, 12), ("M0", 19, 12)]),
            Net("net5", [("M0", 0, 16), ("M0", 19, 16)]),
            Net("net6", [("M0", 0, 19), ("M0", 19, 19)]),
            Net("net7", [("M0", 10, 0), ("M0", 10, 19)]),
            Net("net8", [("M0", 5, 0), ("M0", 5, 10), ("M0", 15, 10), ("M0", 15, 19)]),
        ]

        solution = manager.run(nets)
        assert solution.routed_count >= 5

    def test_visualization_output(self):
        """测试可视化输出不崩溃"""
        grid = make_complex_grid(width=8, height=8)
        spacing_mgr = SpacingManager({"M0": 1, "M1": 1, "M2": 1})
        strategy = DefaultStrategy(max_iterations=20)
        manager = RipupManager(grid, spacing_mgr, strategy)

        nets = [
            Net("net1", [("M0", 0, 0), ("M0", 7, 0)]),
            Net("net2", [("M0", 0, 7), ("M0", 7, 7)]),
            Net("net3", [("M0", 3, 0), ("M0", 3, 7)]),
        ]

        solution = manager.run(nets)

        vis = Visualizer(grid)
        text = vis.visualize_text(solution, spacing_mgr, show_spacing=True, width=8, height=8)
        assert "布线可视化" in text
        assert "摘要" in text
        assert len(text) > 100

    def test_dense_via_grid(self):
        """测试密集via连接的网格"""
        # 创建via代价不同的网格
        vias = []
        for x in range(10):
            for y in range(10):
                # 中心区域via代价低，边缘代价高
                center_dist = abs(x - 5) + abs(y - 5)
                cost = 1.0 + center_dist * 0.5
                vias.append((("M0", x, y), ("M1", x, y), cost))
                vias.append((("M1", x, y), ("M2", x, y), cost))

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
            Net("net1", [("M0", 0, 5), ("M0", 9, 5)]),
            Net("net2", [("M0", 5, 0), ("M0", 5, 9)]),
            Net("net3", [("M0", 0, 0), ("M0", 9, 9)]),
            Net("net4", [("M0", 9, 0), ("M0", 0, 9)]),
        ]

        solution = manager.run(nets)
        assert solution.routed_count >= 2
