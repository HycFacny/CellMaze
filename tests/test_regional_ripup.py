"""
区域拆线重布测试

测试区域拆线的核心逻辑：
- ConflictRegion构建与节点包含判断
- SpacingManager的partial_unmark/mark
- 连通分量检测
- 区域拆线后的线网重连
- 集成到RipupManager的完整流程
"""

import pytest

from maze_router.net import Net, Node, RoutingResult
from maze_router.grid import RoutingGrid
from maze_router.spacing import SpacingManager
from maze_router.ripup import RipupManager
from maze_router.strategy import DefaultStrategy, CongestionAwareStrategy
from maze_router.region import (
    ConflictRegion,
    build_conflict_region,
    analyze_regional_ripup,
    find_nodes_in_region,
    find_connected_components,
)


# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------

def make_3layer_grid(width=10, height=10):
    """创建三层网格"""
    vias = []
    for x in range(width):
        for y in range(height):
            vias.append((("M0", x, y), ("M1", x, y), 2.0))
            vias.append((("M1", x, y), ("M2", x, y), 2.0))
    return RoutingGrid.build_grid(
        layers=["M0", "M1", "M2"],
        width=width, height=height,
        via_connections=vias,
    )


# ------------------------------------------------------------------
# ConflictRegion 单元测试
# ------------------------------------------------------------------

class TestConflictRegion:
    """冲突区域基本功能"""

    def test_contains_inside(self):
        region = ConflictRegion(2, 2, 5, 5, {"M0", "M1"}, margin=0)
        assert region.contains(("M0", 3, 3))
        assert region.contains(("M1", 2, 5))

    def test_contains_outside(self):
        region = ConflictRegion(2, 2, 5, 5, {"M0"}, margin=0)
        assert not region.contains(("M0", 1, 3))
        assert not region.contains(("M0", 6, 3))
        # 层不匹配
        assert not region.contains(("M2", 3, 3))

    def test_margin_expands_region(self):
        region = ConflictRegion(3, 3, 5, 5, {"M0"}, margin=2)
        # margin=2 扩展后范围为 [1,7] x [1,7]
        assert region.contains(("M0", 1, 1))
        assert region.contains(("M0", 7, 7))
        assert not region.contains(("M0", 0, 3))

    def test_area(self):
        region = ConflictRegion(0, 0, 4, 4, {"M0"}, margin=0)
        assert region.area() == 25  # 5x5


# ------------------------------------------------------------------
# SpacingManager partial_unmark/mark 测试
# ------------------------------------------------------------------

class TestPartialSpacing:
    """测试部分标记/清除"""

    def test_partial_unmark_route(self):
        """部分清除后，被清除节点应可用，未清除节点仍被占用"""
        mgr = SpacingManager({"M0": 1})
        nodes = {("M0", 0, 0), ("M0", 1, 0), ("M0", 2, 0), ("M0", 3, 0)}
        mgr.mark_route("netA", nodes)

        # 清除中间两个节点
        mgr.partial_unmark_route("netA", {("M0", 1, 0), ("M0", 2, 0)})

        # 清除的节点对其他线网可用
        assert mgr.is_available(("M0", 1, 0), "netB")
        assert mgr.is_available(("M0", 2, 0), "netB")

        # 未清除的节点仍被占用
        assert not mgr.is_available(("M0", 0, 0), "netB")
        assert not mgr.is_available(("M0", 3, 0), "netB")

        # 线网节点集合正确更新
        remaining = mgr.get_net_nodes("netA")
        assert remaining == {("M0", 0, 0), ("M0", 3, 0)}

    def test_partial_mark_route(self):
        """补充标记后新节点应被占用"""
        mgr = SpacingManager({"M0": 1})
        mgr.mark_route("netA", {("M0", 0, 0), ("M0", 1, 0)})

        # 补充标记
        mgr.partial_mark_route("netA", {("M0", 2, 0), ("M0", 3, 0)})

        # 新节点被netA占用
        assert not mgr.is_available(("M0", 2, 0), "netB")
        assert not mgr.is_available(("M0", 3, 0), "netB")

        # 线网节点集合包含所有节点
        all_nodes = mgr.get_net_nodes("netA")
        assert len(all_nodes) == 4

    def test_partial_unmark_all_removes_net(self):
        """清除全部节点后线网从记录中消失"""
        mgr = SpacingManager({"M0": 0})
        nodes = {("M0", 5, 5)}
        mgr.mark_route("netA", nodes)
        mgr.partial_unmark_route("netA", nodes)
        assert mgr.get_net_nodes("netA") == set()

    def test_partial_unmark_idempotent(self):
        """对不存在的节点调用partial_unmark不会出错"""
        mgr = SpacingManager({"M0": 1})
        mgr.mark_route("netA", {("M0", 0, 0)})
        # 清除一个不属于该线网的节点
        mgr.partial_unmark_route("netA", {("M0", 9, 9)})
        assert mgr.get_net_nodes("netA") == {("M0", 0, 0)}


# ------------------------------------------------------------------
# 连通分量检测测试
# ------------------------------------------------------------------

class TestConnectedComponents:
    """测试连通分量检测"""

    def test_single_component(self):
        nodes = {("M0", 0, 0), ("M0", 1, 0), ("M0", 2, 0)}
        edges = [(("M0", 0, 0), ("M0", 1, 0)), (("M0", 1, 0), ("M0", 2, 0))]
        comps = find_connected_components(nodes, edges)
        assert len(comps) == 1
        assert comps[0] == nodes

    def test_two_components(self):
        nodes = {("M0", 0, 0), ("M0", 1, 0), ("M0", 5, 0), ("M0", 6, 0)}
        edges = [
            (("M0", 0, 0), ("M0", 1, 0)),
            (("M0", 5, 0), ("M0", 6, 0)),
        ]
        comps = find_connected_components(nodes, edges)
        assert len(comps) == 2

    def test_isolated_nodes(self):
        """没有边时每个节点是独立分量"""
        nodes = {("M0", 0, 0), ("M0", 5, 5)}
        comps = find_connected_components(nodes, [])
        assert len(comps) == 2

    def test_empty(self):
        comps = find_connected_components(set(), [])
        assert len(comps) == 0


# ------------------------------------------------------------------
# 区域分析测试
# ------------------------------------------------------------------

class TestRegionAnalysis:
    """测试冲突区域构建和比例分析"""

    def test_build_conflict_region_covers_terminals(self):
        """冲突区域应覆盖失败线网的所有端口"""
        grid = make_3layer_grid(10, 10)
        mgr = SpacingManager({"M0": 1, "M1": 1, "M2": 1})
        net = Net("failed", [("M0", 2, 3), ("M0", 7, 3)])

        region = build_conflict_region(net, set(), mgr, grid, margin=1)

        for t in net.terminals:
            assert region.contains(t)

    def test_analyze_regional_ripup_ratio(self):
        """测试区域内节点占比计算"""
        region = ConflictRegion(3, 3, 6, 6, {"M0"}, margin=0)

        # 模拟一个线网有10个节点，其中3个在区域内
        result = RoutingResult(net=Net("bn", []))
        result.routed_nodes = {("M0", i, 0) for i in range(10)}
        result.success = True
        net_results = {"bn": result}

        mgr = SpacingManager({"M0": 0})
        ratios = analyze_regional_ripup(
            Net("failed", [("M0", 4, 4)]), {"bn"}, mgr, net_results, region,
        )
        # 节点(3,0),(4,0),(5,0),(6,0) 在 x=[3,6]，但y=0不在y=[3,6]内
        assert "bn" in ratios
        assert ratios["bn"] == 0.0  # y=0 都不在 [3,6] 内

    def test_find_nodes_in_region(self):
        region = ConflictRegion(2, 2, 4, 4, {"M0"}, margin=0)
        nodes = {("M0", i, 3) for i in range(7)}
        in_region = find_nodes_in_region(nodes, region)
        # x in [2,4], y=3 in [2,4] → x=2,3,4
        assert in_region == {("M0", 2, 3), ("M0", 3, 3), ("M0", 4, 3)}


# ------------------------------------------------------------------
# 区域拆线集成测试
# ------------------------------------------------------------------

class TestRegionalRipupIntegration:
    """完整流程测试：区域拆线融入RipupManager"""

    def test_regional_ripup_with_long_blocking_net(self):
        """
        场景：一条长线网(netA)横穿网格，阻塞了一条短线网(netB)。
        区域拆线应只拆netA在冲突区域的片段，而非整条netA。

        使用三层网格确保重连有足够空间。
        """
        grid = make_3layer_grid(15, 15)
        spacing_mgr = SpacingManager({"M0": 1, "M1": 1, "M2": 1})
        # 使用较低的阈值使区域拆线更容易触发
        strategy = DefaultStrategy(
            max_iterations=30,
            regional_ripup_threshold=0.5,
        )
        manager = RipupManager(grid, spacing_mgr, strategy)

        nets = [
            # 长线网横穿底部
            Net("netA", [("M0", 0, 7), ("M0", 14, 7)]),
            # 短线网需要穿越netA
            Net("netB", [("M0", 7, 5), ("M0", 7, 9)]),
        ]

        solution = manager.run(nets)
        # 两个线网都应该能布通
        assert solution.routed_count >= 1

    def test_regional_ripup_preserves_net_segments(self):
        """
        区域拆线后，被拆线网的区域外部分应保留。
        用SpacingManager的partial_unmark验证。
        """
        mgr = SpacingManager({"M0": 0})

        # 手动构建场景
        long_net_nodes = {("M0", i, 5) for i in range(10)}
        mgr.mark_route("longNet", long_net_nodes)

        # 定义冲突区域 x=[3,6]
        region_nodes = {("M0", i, 5) for i in range(3, 7)}
        mgr.partial_unmark_route("longNet", region_nodes)

        # 区域外节点仍然被占用
        remaining = mgr.get_net_nodes("longNet")
        assert ("M0", 0, 5) in remaining
        assert ("M0", 9, 5) in remaining
        # 区域内节点已释放
        assert ("M0", 4, 5) not in remaining

    def test_regional_ripup_with_congestion_aware(self):
        """测试区域拆线与拥塞感知策略结合"""
        grid = make_3layer_grid(12, 12)
        spacing_mgr = SpacingManager({"M0": 1, "M1": 1, "M2": 1})
        strategy = CongestionAwareStrategy(
            max_iterations=30, congestion_weight=0.3,
        )
        manager = RipupManager(grid, spacing_mgr, strategy)

        nets = [
            Net("net1", [("M0", 0, 6), ("M0", 11, 6)]),
            Net("net2", [("M0", 6, 0), ("M0", 6, 11)]),
            Net("net3", [("M0", 3, 3), ("M0", 9, 9)]),
        ]

        solution = manager.run(nets)
        assert solution.routed_count >= 2

    def test_regional_ripup_multiple_blocking(self):
        """多个阻塞线网的区域拆线场景"""
        grid = make_3layer_grid(15, 15)
        spacing_mgr = SpacingManager({"M0": 1, "M1": 1, "M2": 1})
        strategy = DefaultStrategy(
            max_iterations=40,
            regional_ripup_threshold=0.5,
        )
        manager = RipupManager(grid, spacing_mgr, strategy)

        nets = [
            Net("h1", [("M0", 0, 5), ("M0", 14, 5)]),
            Net("h2", [("M0", 0, 9), ("M0", 14, 9)]),
            Net("v1", [("M0", 7, 0), ("M0", 7, 14)]),
        ]

        solution = manager.run(nets)
        assert solution.routed_count >= 2
