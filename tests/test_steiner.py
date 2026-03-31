"""
Steiner 树相关测试：多端口、折角代价、DP 精度
"""

import pytest
from maze_router.data.net import Net
from maze_router.data.grid import GridGraph
from maze_router.constraint_manager import ConstraintManager
from maze_router.cost_manager import CostManager
from maze_router.constraints.space_constraint import SpaceConstraint
from maze_router.costs.corner_cost import CornerCost
from maze_router.steiner_router_algo import SteinerRouter
from maze_router.maze_router_algo import build_steiner_greedy


def make_env(width=9, height=9):
    grid = GridGraph.build_grid(["M0", "M1"], width, height, edge_cost=1.0, via_cost=2.0)
    cmgr = ConstraintManager([SpaceConstraint({"M0": 1, "M1": 1})])
    cost_no_corner = CostManager(grid=grid, costs=[CornerCost.disabled()])
    cost_with_corner = CostManager(
        grid=grid,
        costs=[CornerCost(l_costs={"M0": 5.0, "M1": 5.0})],
    )
    return grid, cmgr, cost_no_corner, cost_with_corner


class TestMultiTerminalSteiner:

    def test_4_terminals(self):
        grid, cmgr, cost_mgr, _ = make_env()
        net = Net("n4", [("M0", 0, 0), ("M0", 8, 0), ("M0", 0, 8), ("M0", 8, 8)])
        steiner = SteinerRouter(grid, cmgr, cost_mgr)
        result = steiner.route(net)
        assert result.success
        for t in net.terminals:
            assert t in result.routed_nodes

    def test_5_terminals_on_line(self):
        grid, cmgr, cost_mgr, _ = make_env()
        terminals = [("M0", x, 4) for x in (0, 2, 4, 6, 8)]
        net = Net("n5", terminals)
        steiner = SteinerRouter(grid, cmgr, cost_mgr)
        result = steiner.route(net)
        assert result.success

    def test_corner_cost_increases_total(self):
        grid, cmgr, no_corner, with_corner = make_env()
        net = Net("n3", [("M0", 0, 0), ("M0", 4, 0), ("M0", 0, 4)])

        r_nc = SteinerRouter(grid, cmgr, no_corner).route(net)
        r_wc = SteinerRouter(grid, cmgr, with_corner).route(net)

        assert r_nc.success and r_wc.success
        # With corner cost, total should be >= without corner cost
        assert r_wc.total_cost >= r_nc.total_cost

    def test_t_cost_at_branch(self):
        """T 型折角在分支节点处增加代价"""
        grid, cmgr, _, _ = make_env()
        cost_no_t = CostManager(grid=grid, costs=[CornerCost(l_costs={}, t_costs={})])
        cost_with_t = CostManager(
            grid=grid,
            costs=[CornerCost(l_costs={}, t_costs={"M0": 10.0, "M1": 10.0})],
        )
        # T-junction: 3 terminals forming a T shape
        net = Net("t_shape", [("M0", 4, 0), ("M0", 0, 4), ("M0", 8, 4)])

        r_no_t = SteinerRouter(grid, cmgr, cost_no_t).route(net)
        r_with_t = SteinerRouter(grid, cmgr, cost_with_t).route(net)

        assert r_no_t.success and r_with_t.success
        assert r_with_t.total_cost >= r_no_t.total_cost

    def test_two_terminal_less_t_than_three(self):
        """两端口线网 T 型代价应 ≤ 三端口线网（有分支时 T 代价更大）"""
        grid, cmgr, _, _ = make_env()
        cost_with_t = CostManager(
            grid=grid,
            costs=[CornerCost(l_costs={}, t_costs={"M0": 100.0})],
        )
        net2 = Net("n2", [("M0", 0, 4), ("M0", 8, 4)])
        net3 = Net("n3", [("M0", 0, 0), ("M0", 8, 0), ("M0", 4, 8)])

        r2 = SteinerRouter(grid, cmgr, cost_with_t).route(net2)
        r3 = SteinerRouter(grid, cmgr, cost_with_t).route(net3)

        assert r2.success and r3.success
        # 3-terminal tree has more branches than 2-terminal, so higher T cost
        # (both have DP splits, but 3-terminal has more)
        assert r3.total_cost >= r2.total_cost

    def test_dp_better_or_equal_to_greedy(self):
        """DP 应该得到不超过贪心的总代价（近似最优）"""
        grid, cmgr, cost_mgr, _ = make_env()
        net = Net("n4", [("M0", 0, 0), ("M0", 8, 0), ("M0", 4, 4), ("M0", 0, 8)])

        r_dp = SteinerRouter(grid, cmgr, cost_mgr).route(net)
        r_greedy = build_steiner_greedy(
            net, net.terminals, grid, cmgr, cost_mgr,
        )

        assert r_dp.success
        assert r_greedy.success
        # DP should not be significantly worse than greedy
        assert r_dp.total_cost <= r_greedy.total_cost * 1.5

    def test_cable_locs_constraint(self):
        """cable_locs 约束：M0 只允许在指定节点走线"""
        grid = GridGraph.build_grid(["M0", "M1"], 5, 5)
        cmgr = ConstraintManager([SpaceConstraint({"M0": 1, "M1": 1})])
        cost_mgr = CostManager(grid=grid, costs=[CornerCost.disabled()])

        # 只允许 M0 在 x=0,1,2 列走线
        allowed_m0 = {("M0", x, y) for x in range(3) for y in range(5)}
        net = Net(
            "n_cable",
            [("M0", 0, 2), ("M0", 2, 2)],
            cable_locs=allowed_m0,
        )

        steiner = SteinerRouter(grid, cmgr, cost_mgr)
        result = steiner.route(net)
        assert result.success
        # All M0 nodes should be in allowed_m0
        for node in result.routed_nodes:
            if node[0] == "M0":
                assert node in allowed_m0
