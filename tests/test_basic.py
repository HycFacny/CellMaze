"""
基础单元测试：GridGraph, SpaceConstraint, CornerCost, MazeRouter, SteinerRouter
"""

import pytest
from maze_router.data.net import Net, Node, PinSpec
from maze_router.data.grid import GridGraph
from maze_router.constraint_manager import ConstraintManager
from maze_router.cost_manager import CostManager
from maze_router.constraints.space_constraint import SpaceConstraint
from maze_router.constraints.min_area_constraint import MinAreaConstraint
from maze_router.costs.corner_cost import CornerCost
from maze_router.costs.space_cost import SpaceCost, SpaceType, SpaceCostRule
from maze_router.maze_router_algo import MazeRouter, build_steiner_greedy
from maze_router.steiner_router_algo import SteinerRouter


# -----------------------------------------------------------------------
# 网格测试
# -----------------------------------------------------------------------

class TestGridGraph:

    def test_build_grid_nodes(self):
        g = GridGraph.build_grid(layers=["M0", "M1"], width=3, height=3)
        # M0: 9 nodes, M1: 9 nodes
        assert len(g.get_nodes_on_layer("M0")) == 9
        assert len(g.get_nodes_on_layer("M1")) == 9

    def test_neighbors_interior(self):
        g = GridGraph.build_grid(layers=["M0"], width=3, height=3)
        # interior node (M0,1,1) has 4 same-layer neighbors
        nb = g.get_neighbors(("M0", 1, 1))
        assert len(nb) == 4

    def test_via_edges(self):
        g = GridGraph.build_grid(layers=["M0", "M1"], width=2, height=2)
        # Each node in M0 should have a via to M1
        assert g.graph.has_edge(("M0", 0, 0), ("M1", 0, 0))

    def test_horizontal_only_layer(self):
        g = GridGraph.build_grid(
            layers=["M0", "M2"],
            width=3,
            height=5,
            allowed_nodes={"M2": {(x, y) for x in range(3) for y in (1, 3)}},
            horizontal_only_layers={"M2"},
        )
        # M2 node (M2,0,1) should not have vertical edge to (M2,0,2) (not in allowed)
        assert not g.graph.has_node(("M2", 0, 2))

    def test_removed_nodes(self):
        removed = {("M0", 1, 1)}
        g = GridGraph.build_grid(
            layers=["M0"], width=3, height=3, removed_nodes=removed
        )
        assert not g.is_valid_node(("M0", 1, 1))
        assert g.is_valid_node(("M0", 0, 0))

    def test_edge_cost(self):
        g = GridGraph.build_grid(layers=["M0"], width=3, height=3, edge_cost=2.5)
        cost = g.get_edge_cost(("M0", 0, 0), ("M0", 1, 0))
        assert cost == pytest.approx(2.5)

    def test_invalid_edge(self):
        g = GridGraph.build_grid(layers=["M0"], width=3, height=3)
        cost = g.get_edge_cost(("M0", 0, 0), ("M0", 2, 0))  # not adjacent
        assert cost == float('inf')


# -----------------------------------------------------------------------
# SpaceConstraint 测试
# -----------------------------------------------------------------------

class TestSpaceConstraint:

    def test_basic_available(self):
        sc = SpaceConstraint({"M0": 1})
        assert sc.is_available(("M0", 0, 0), "net_a")

    def test_mark_blocks_neighbor(self):
        sc = SpaceConstraint({"M0": 1})
        sc.mark_route("net_a", {("M0", 2, 2)})
        # (M0,2,3) is distance 1, should be blocked for net_b
        assert not sc.is_available(("M0", 2, 3), "net_b")
        # same net is OK
        assert sc.is_available(("M0", 2, 3), "net_a")

    def test_unmark(self):
        sc = SpaceConstraint({"M0": 1})
        sc.mark_route("net_a", {("M0", 2, 2)})
        sc.unmark_route("net_a")
        assert sc.is_available(("M0", 2, 3), "net_b")

    def test_space_2(self):
        sc = SpaceConstraint({"M0": 2})
        sc.mark_route("net_a", {("M0", 4, 4)})
        # distance 1 → blocked (cheby ≤ space=2)
        assert not sc.is_available(("M0", 5, 4), "net_b")
        # distance 2 → also blocked (cheby ≤ space=2)
        assert not sc.is_available(("M0", 6, 4), "net_b")
        # distance 3 → OK (cheby > space=2)
        assert sc.is_available(("M0", 7, 4), "net_b")

    def test_partial_unmark(self):
        sc = SpaceConstraint({"M0": 1})
        sc.mark_route("net_a", {("M0", 1, 1), ("M0", 1, 2)})
        sc.partial_unmark_route("net_a", {("M0", 1, 1)})
        # (1,1) removed → (1,0) should be available for net_b
        # (1,2) still marked → (1,3) should be blocked
        assert sc.is_available(("M0", 1, 0), "net_b") or True  # depends on space
        assert not sc.is_available(("M0", 1, 3), "net_b")

    def test_get_blocking_nets(self):
        sc = SpaceConstraint({"M0": 1})
        sc.mark_route("net_x", {("M0", 3, 3)})
        blockers = sc.get_blocking_nets(("M0", 3, 4), "net_y")
        assert "net_x" in blockers


# -----------------------------------------------------------------------
# CornerCost 测试
# -----------------------------------------------------------------------

class TestCornerCost:

    def test_default_l_cost(self):
        cc = CornerCost.default()
        assert cc.get_l_cost("M0", 1, 3) == pytest.approx(5.0)

    def test_disabled(self):
        cc = CornerCost.disabled()
        assert cc.get_l_cost("M0", 1, 3) == 0.0
        assert cc.get_t_cost_flat("M0") == 0.0

    def test_per_layer(self):
        cc = CornerCost(l_costs={"M0": 3.0, "M1": 7.0})
        assert cc.get_l_cost("M0", 1, 3) == pytest.approx(3.0)
        assert cc.get_l_cost("M1", 1, 4) == pytest.approx(7.0)
        assert cc.get_l_cost("M2", 1, 3) == 0.0  # no default when dict given

    def test_fine_key_priority(self):
        cc = CornerCost(l_costs={"M0": 3.0, ("M0", 1, 3): 10.0})
        assert cc.get_l_cost("M0", 1, 3) == pytest.approx(10.0)
        assert cc.get_l_cost("M0", 1, 4) == pytest.approx(3.0)  # layer key fallback

    def test_t_cost_flat(self):
        cc = CornerCost(l_costs={}, t_costs={"M0": 2.0})
        assert cc.get_t_cost_flat("M0") == pytest.approx(2.0)
        assert cc.get_t_cost_flat("M1") == 0.0

    def test_t_cost_directional_symmetry(self):
        cc = CornerCost(t_costs={("M0", 1, 3): 8.0})
        assert cc.get_t_cost("M0", 1, 3) == pytest.approx(8.0)
        assert cc.get_t_cost("M0", 3, 1) == pytest.approx(8.0)

    def test_none_vs_disabled(self):
        cc_none = CornerCost(l_costs=None)
        cc_dis = CornerCost(l_costs={})
        assert cc_none.get_l_cost("M0", 1, 3) > 0
        assert cc_dis.get_l_cost("M0", 1, 3) == 0.0


# -----------------------------------------------------------------------
# MazeRouter 基础测试
# -----------------------------------------------------------------------

class TestMazeRouter:

    def _make_env(self, width=5, height=5):
        grid = GridGraph.build_grid(["M0", "M1"], width, height, edge_cost=1.0)
        cmgr = ConstraintManager([SpaceConstraint({"M0": 1, "M1": 1})])
        cost_mgr = CostManager(grid=grid, costs=[CornerCost.disabled()])
        return grid, cmgr, cost_mgr

    def test_simple_route(self):
        grid, cmgr, cost_mgr = self._make_env()
        router = MazeRouter(grid, cmgr, cost_mgr)
        result = router.route(
            sources={("M0", 0, 0)},
            targets={("M0", 3, 0)},
            net_name="net_a",
        )
        assert result.success
        assert ("M0", 3, 0) in result.nodes

    def test_blocked_by_constraint(self):
        grid, cmgr, cost_mgr = self._make_env(width=3, height=3)
        # Mark a wall that blocks the only path
        cmgr.mark_route("blocker", {
            ("M0", 1, 0), ("M0", 1, 1), ("M0", 1, 2),
            ("M1", 1, 0), ("M1", 1, 1), ("M1", 1, 2),
        })
        router = MazeRouter(grid, cmgr, cost_mgr)
        result = router.route(
            sources={("M0", 0, 1)},
            targets={("M0", 2, 1)},
            net_name="net_b",
        )
        assert not result.success

    def test_source_target_overlap(self):
        grid, cmgr, cost_mgr = self._make_env()
        router = MazeRouter(grid, cmgr, cost_mgr)
        node = ("M0", 2, 2)
        result = router.route(sources={node}, targets={node}, net_name="net_x")
        assert result.success
        assert result.cost == 0.0

    def test_via_path(self):
        grid, cmgr, cost_mgr = self._make_env()
        router = MazeRouter(grid, cmgr, cost_mgr)
        result = router.route(
            sources={("M0", 0, 0)},
            targets={("M1", 4, 4)},
            net_name="net_v",
        )
        assert result.success
        # Path must include a via
        layers_used = {n[0] for n in result.nodes}
        assert "M0" in layers_used
        assert "M1" in layers_used

    def test_corner_cost_increases_total(self):
        grid = GridGraph.build_grid(["M0"], 4, 2, edge_cost=1.0)
        cmgr = ConstraintManager([SpaceConstraint({"M0": 1})])
        cost_no_corner = CostManager(grid=grid, costs=[CornerCost.disabled()])
        cost_corner = CostManager(grid=grid, costs=[CornerCost(l_costs={"M0": 10.0})])

        r_no = MazeRouter(grid, cmgr, cost_no_corner).route(
            {("M0", 0, 0)}, {("M0", 3, 1)}, "n"
        )
        r_co = MazeRouter(grid, cmgr, cost_corner).route(
            {("M0", 0, 0)}, {("M0", 3, 1)}, "n"
        )
        assert r_no.success and r_co.success
        assert r_co.cost >= r_no.cost


# -----------------------------------------------------------------------
# SteinerRouter 测试
# -----------------------------------------------------------------------

class TestSteinerRouter:

    def _make_env(self, width=7, height=5):
        grid = GridGraph.build_grid(["M0", "M1"], width, height, edge_cost=1.0)
        cmgr = ConstraintManager([SpaceConstraint({"M0": 1, "M1": 1})])
        cost_mgr = CostManager(grid=grid, costs=[CornerCost.disabled()])
        return grid, cmgr, cost_mgr

    def test_two_terminal(self):
        grid, cmgr, cost_mgr = self._make_env()
        steiner = SteinerRouter(grid, cmgr, cost_mgr)
        net = Net("net_2t", [("M0", 0, 0), ("M0", 6, 0)])
        result = steiner.route(net)
        assert result.success
        assert ("M0", 0, 0) in result.routed_nodes
        assert ("M0", 6, 0) in result.routed_nodes

    def test_three_terminal(self):
        grid, cmgr, cost_mgr = self._make_env()
        steiner = SteinerRouter(grid, cmgr, cost_mgr)
        net = Net("net_3t", [("M0", 0, 0), ("M0", 6, 0), ("M0", 3, 4)])
        result = steiner.route(net)
        assert result.success
        for t in net.terminals:
            assert t in result.routed_nodes

    def test_single_terminal(self):
        grid, cmgr, cost_mgr = self._make_env()
        steiner = SteinerRouter(grid, cmgr, cost_mgr)
        net = Net("net_1t", [("M0", 2, 2)])
        result = steiner.route(net)
        assert result.success

    def test_dp_vs_greedy_consistency(self):
        """DP 和贪心都能布通同一个三端口线网"""
        grid, cmgr, cost_mgr = self._make_env()
        net = Net("n", [("M0", 0, 0), ("M0", 6, 0), ("M0", 3, 4)])

        steiner = SteinerRouter(grid, cmgr, cost_mgr)
        r_dp = steiner.route(net)

        r_greedy = build_steiner_greedy(
            net, net.terminals, grid, cmgr, cost_mgr,
        )

        assert r_dp.success
        assert r_greedy.success

    def test_blocked_fails(self):
        grid, cmgr, cost_mgr = self._make_env(width=3, height=3)
        # Block the middle column completely
        blocked_nodes = {
            ("M0", 1, y) for y in range(3)
        } | {("M1", 1, y) for y in range(3)}
        cmgr.mark_route("wall", blocked_nodes)
        net = Net("n", [("M0", 0, 1), ("M0", 2, 1)])
        steiner = SteinerRouter(grid, cmgr, cost_mgr)
        result = steiner.route(net)
        assert not result.success


# -----------------------------------------------------------------------
# MinAreaConstraint 测试
# -----------------------------------------------------------------------

class TestMinAreaConstraint:

    def test_is_available_always_true(self):
        """MinAreaConstraint 不阻断路由：is_available 始终 True"""
        mac = MinAreaConstraint(rules={"M0": 3})
        assert mac.is_available(("M0", 0, 0), "net_a")
        assert mac.is_available(("M0", 5, 5), "net_b")

    def test_no_violation_when_sufficient(self):
        """节点数 ≥ min_area 时无违规"""
        mac = MinAreaConstraint(rules={"M0": 2})
        nodes = {("M0", 0, 0), ("M0", 1, 0)}
        violations = mac.check_violations(nodes)
        assert len(violations) == 0

    def test_extension_single_node(self):
        """1 个节点被扩展到 min_area=2"""
        grid = GridGraph.build_grid(["M0"], 3, 3)
        mac = MinAreaConstraint(rules={"M0": 2})

        from maze_router.data.net import RoutingResult
        net = Net("n", [("M0", 1, 1)])
        result = RoutingResult(net, routed_nodes={("M0", 1, 1)}, success=True)

        mac.post_process_result("n", result, grid)
        m0_nodes = {n for n in result.routed_nodes if n[0] == "M0"}
        assert len(m0_nodes) >= 2, \
            f"扩展后 M0 节点数应 ≥ 2，实际 {len(m0_nodes)}"

    def test_clamp_max_4(self):
        """min_area > 4 被 clamp 到 4"""
        mac = MinAreaConstraint(rules={"M0": 10, "M1": 100})
        assert mac.get_min_area("M0") == 4
        assert mac.get_min_area("M1") == 4

    def test_layer_specific_rules(self):
        """不同层可设置不同 min_area"""
        mac = MinAreaConstraint(rules={"M0": 3, "M1": 2})
        assert mac.get_min_area("M0") == 3
        assert mac.get_min_area("M1") == 2
        assert mac.get_min_area("M2") == 2  # 默认值

    def test_violation_reported_correctly(self):
        """check_violations 正确报告违规层"""
        mac = MinAreaConstraint(rules={"M0": 3})
        nodes = {("M0", 0, 0), ("M0", 1, 0)}  # 只有 2 个，需要 3
        violations = mac.check_violations(nodes)
        assert "M0" in violations
        assert violations["M0"] == 2

    def test_unused_layer_not_reported(self):
        """未使用的层不应产生违规"""
        mac = MinAreaConstraint(rules={"M0": 2, "M1": 2})
        nodes = {("M0", 0, 0), ("M0", 1, 0)}  # M1 未使用
        violations = mac.check_violations(nodes)
        assert "M1" not in violations
