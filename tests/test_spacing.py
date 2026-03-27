"""
间距约束测试

测试不同线网之间的最小间距约束是否正确执行。
"""

import pytest
from maze_router.net import Net
from maze_router.grid import RoutingGrid
from maze_router.spacing import SpacingManager
from maze_router.router import MazeRouter
from maze_router.steiner import SteinerTreeBuilder


class TestSpacingConstraint:
    """间距约束测试"""

    def test_spacing_blocks_adjacent(self):
        """测试间距约束阻止相邻布线"""
        spacing_mgr = SpacingManager({"M0": 1})

        # 线网A占用(M0, 2, 2)
        spacing_mgr.mark_route("netA", {("M0", 2, 2)})

        # 间距为1时，相邻格子应对其他线网不可用
        assert not spacing_mgr.is_available(("M0", 3, 2), "netB")
        assert not spacing_mgr.is_available(("M0", 1, 2), "netB")
        assert not spacing_mgr.is_available(("M0", 2, 3), "netB")
        assert not spacing_mgr.is_available(("M0", 2, 1), "netB")
        # 对角线也不可用（Chebyshev距离=1）
        assert not spacing_mgr.is_available(("M0", 3, 3), "netB")
        assert not spacing_mgr.is_available(("M0", 1, 1), "netB")

        # 距离为2的格子应该可用
        assert spacing_mgr.is_available(("M0", 4, 2), "netB")
        assert spacing_mgr.is_available(("M0", 0, 2), "netB")

        # 自身线网不受限制
        assert spacing_mgr.is_available(("M0", 3, 2), "netA")

    def test_spacing_chebyshev_distance_2(self):
        """测试Chebyshev距离为2的间距约束"""
        spacing_mgr = SpacingManager({"M0": 2})
        spacing_mgr.mark_route("netA", {("M0", 5, 5)})

        # Chebyshev距离 <= 2 的都应该被阻塞
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                assert not spacing_mgr.is_available(
                    ("M0", 5 + dx, 5 + dy), "netB"
                ), f"({5+dx},{5+dy}) should be blocked"

        # Chebyshev距离 = 3 应该可用
        assert spacing_mgr.is_available(("M0", 8, 5), "netB")
        assert spacing_mgr.is_available(("M0", 5, 8), "netB")

    def test_unmark_clears_spacing(self):
        """测试解除标记后禁区被清除"""
        spacing_mgr = SpacingManager({"M0": 1})
        spacing_mgr.mark_route("netA", {("M0", 2, 2)})

        assert not spacing_mgr.is_available(("M0", 3, 2), "netB")

        spacing_mgr.unmark_route("netA")

        assert spacing_mgr.is_available(("M0", 3, 2), "netB")
        assert spacing_mgr.is_available(("M0", 2, 2), "netB")

    def test_spacing_per_layer(self):
        """测试每层独立的间距约束"""
        spacing_mgr = SpacingManager({"M0": 1, "M1": 2})

        spacing_mgr.mark_route("netA", {("M0", 5, 5), ("M1", 5, 5)})

        # M0层间距1
        assert not spacing_mgr.is_available(("M0", 6, 5), "netB")
        assert spacing_mgr.is_available(("M0", 7, 5), "netB")

        # M1层间距2
        assert not spacing_mgr.is_available(("M1", 7, 5), "netB")
        assert spacing_mgr.is_available(("M1", 8, 5), "netB")

    def test_routing_respects_spacing(self):
        """测试布线器实际遵守间距约束"""
        grid = RoutingGrid.build_grid(
            layers=["M0"],
            width=10,
            height=5,
        )
        spacing_mgr = SpacingManager({"M0": 1})

        # 先布线netA横跨中间
        spacing_mgr.mark_route(
            "netA",
            {("M0", x, 2) for x in range(10)}
        )

        router = MazeRouter(grid, spacing_mgr)

        # netB 尝试从左下到右下，必须绕过netA的禁区
        result = router.route(
            sources={("M0", 0, 0)},
            targets={("M0", 9, 0)},
            net_name="netB",
        )
        assert result.success
        # 验证路径上所有节点都不违反间距
        for node in result.nodes:
            assert spacing_mgr.is_available(node, "netB")

    def test_blocking_nets_detection(self):
        """测试阻塞线网检测"""
        spacing_mgr = SpacingManager({"M0": 1})
        spacing_mgr.mark_route("netA", {("M0", 2, 2)})
        spacing_mgr.mark_route("netB", {("M0", 5, 5)})

        # 检查被谁阻塞
        blockers = spacing_mgr.get_blocking_nets(("M0", 3, 2), "netC")
        assert "netA" in blockers
        assert "netB" not in blockers
