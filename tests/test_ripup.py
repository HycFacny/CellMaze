"""
拆线重布测试：RipupManager, RipupStrategy, 区域拆线, 拥塞感知
"""

import pytest
from maze_router.data.net import Net, PinSpec
from maze_router.data.grid import GridGraph
from maze_router.constraint_manager import ConstraintManager
from maze_router.cost_manager import CostManager
from maze_router.constraints.space_constraint import SpaceConstraint
from maze_router.costs.corner_cost import CornerCost
from maze_router.ripup_strategy import (
    DefaultRipupStrategy, CongestionAwareRipupStrategy, RouterType,
)
from maze_router.ripup_manager import RipupManager
from maze_router.engine import MazeRouterEngine


def make_base(width=9, height=9, space=1):
    grid = GridGraph.build_grid(
        ["M0", "M1", "M2"],
        width, height,
        edge_cost=1.0, via_cost=2.0,
        allowed_nodes={"M2": {(x, y) for x in range(width) for y in (2, 4)}},
        horizontal_only_layers={"M2"},
    )
    cmgr = ConstraintManager([SpaceConstraint({"M0": space, "M1": space, "M2": space})])
    cost_mgr = CostManager(grid=grid, costs=[CornerCost.disabled()])
    return grid, cmgr, cost_mgr


class TestDefaultStrategy:

    def test_order_nets_by_terminal_count(self):
        strat = DefaultRipupStrategy()
        nets = [
            Net("n1", [("M0", 0, 0)]),
            Net("n4", [("M0", 0, 0)] * 4),
            Net("n2", [("M0", 0, 0)] * 2),
        ]
        ordered = strat.order_nets(nets)
        assert ordered[0].name == "n4"
        assert ordered[-1].name == "n1"

    def test_router_type_dp_for_small(self):
        strat = DefaultRipupStrategy()
        net_small = Net("s", [("M0", i, 0) for i in range(5)])
        net_large = Net("l", [("M0", i, 0) for i in range(12)])
        assert strat.get_router_type(net_small, 1) == RouterType.STEINER_DP
        assert strat.get_router_type(net_large, 1) == RouterType.MAZE_GREEDY

    def test_order_terminals_nearest_first(self):
        strat = DefaultRipupStrategy()
        net = Net("n", [("M0", 0, 0), ("M0", 4, 0), ("M0", 8, 0)])
        ordered = strat.order_terminals(net, set())
        # First should be terminal[0], next should be nearest
        assert ordered[0] == ("M0", 0, 0)


class TestRipupManagerBasic:

    def test_two_non_conflicting_nets(self):
        grid, cmgr, cost_mgr = make_base(width=9, height=3)
        strategy = DefaultRipupStrategy()
        mgr = RipupManager(grid, cmgr, cost_mgr, strategy)

        nets = [
            Net("n1", [("M0", 0, 0), ("M0", 8, 0)]),
            Net("n2", [("M0", 0, 2), ("M0", 8, 2)]),
        ]
        solution = mgr.run(nets)
        assert solution.routed_count == 2

    def test_conflicting_nets_ripup(self):
        """两个线网互相冲突时，至少一个能成功布通"""
        grid, cmgr, cost_mgr = make_base(width=5, height=5)
        strategy = DefaultRipupStrategy(max_iterations=5)
        mgr = RipupManager(grid, cmgr, cost_mgr, strategy)

        # Two nets that must cross the same area
        nets = [
            Net("h", [("M0", 0, 2), ("M0", 4, 2)]),   # horizontal
            Net("v", [("M0", 2, 0), ("M0", 2, 4)]),   # vertical
        ]
        solution = mgr.run(nets)
        assert solution.routed_count >= 1

    def test_three_nets_all_routed(self):
        grid, cmgr, cost_mgr = make_base(width=11, height=5)
        strategy = DefaultRipupStrategy()
        mgr = RipupManager(grid, cmgr, cost_mgr, strategy)

        nets = [
            Net("a", [("M0", 0, 0), ("M0", 10, 0)]),
            Net("b", [("M0", 0, 2), ("M0", 10, 2)]),
            Net("c", [("M0", 0, 4), ("M0", 10, 4)]),
        ]
        solution = mgr.run(nets)
        assert solution.routed_count == 3


class TestCongestionAwareStrategy:

    def test_record_congestion(self):
        strat = CongestionAwareRipupStrategy()
        node = ("M0", 3, 3)
        strat.record_congestion(node)
        strat.record_congestion(node)
        cmap = strat.get_congestion_map()
        assert cmap[node] == pytest.approx(2.0)

    def test_cost_multiplier_increases(self):
        strat = CongestionAwareRipupStrategy()
        node = ("M1", 2, 2)
        mult_before = strat.get_cost_multiplier(node, "net")
        strat.record_congestion(node)
        mult_after = strat.get_cost_multiplier(node, "net")
        assert mult_after > mult_before

    def test_congestion_aware_routing(self):
        grid, cmgr, cost_mgr = make_base(width=9, height=9)
        strategy = CongestionAwareRipupStrategy(max_iterations=8)
        mgr = RipupManager(grid, cmgr, cost_mgr, strategy)

        nets = [
            Net("a", [("M0", 0, 0), ("M0", 8, 0)]),
            Net("b", [("M0", 0, 8), ("M0", 8, 8)]),
        ]
        solution = mgr.run(nets)
        assert solution.routed_count >= 1


class TestPinExtraction:

    def test_pin_extracted_to_m1(self):
        grid, cmgr, cost_mgr = make_base()
        strategy = DefaultRipupStrategy()
        mgr = RipupManager(grid, cmgr, cost_mgr, strategy)

        net = Net(
            "pnet",
            [("M0", 4, 0), ("M0", 4, 8)],
            pin_spec=PinSpec(layer="M1"),
        )
        solution = mgr.run([net])
        result = solution.results.get("pnet")
        assert result is not None
        assert result.success
        assert result.pin_point is not None
        assert result.pin_point[0] == "M1"

    def test_pin_extracted_to_any(self):
        grid, cmgr, cost_mgr = make_base()
        strategy = DefaultRipupStrategy()
        mgr = RipupManager(grid, cmgr, cost_mgr, strategy)

        net = Net(
            "pnet2",
            [("M0", 3, 0), ("M0", 3, 8)],
            pin_spec=PinSpec(layer="Any"),
        )
        solution = mgr.run([net])
        result = solution.results.get("pnet2")
        assert result is not None
        assert result.success
        if result.pin_point is not None:
            assert result.pin_point[0] in ("M1", "M2")


class TestMazeRouterEngine:

    def test_engine_basic(self):
        grid = GridGraph.build_grid(
            ["M0", "M1"], 7, 7, edge_cost=1.0, via_cost=2.0
        )
        nets = [
            Net("a", [("M0", 0, 0), ("M0", 6, 0)]),
            Net("b", [("M0", 0, 6), ("M0", 6, 6)]),
        ]
        engine = MazeRouterEngine(
            grid=grid,
            nets=nets,
            space_constr={"M0": 1, "M1": 1},
        )
        solution = engine.run()
        assert solution.routed_count >= 1

    def test_engine_with_corner_cost(self):
        grid = GridGraph.build_grid(["M0", "M1"], 7, 7, edge_cost=1.0)
        nets = [Net("n", [("M0", 0, 3), ("M0", 6, 3)])]
        engine = MazeRouterEngine(
            grid=grid,
            nets=nets,
            space_constr={"M0": 1, "M1": 1},
            corner_cost=CornerCost(l_costs={"M0": 5.0}),
        )
        solution = engine.run()
        assert solution.routed_count == 1
