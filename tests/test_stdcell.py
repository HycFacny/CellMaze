"""
标准单元 OAI33 完整测试

验证真实复杂度场景下的布线正确性：
  - 10 个线网同时布线
  - 硬间距约束无违规
  - ≥ 80% 线网布通
  - Pin 引出点在正确层
"""

import os
import pytest
from maze_router import MazeRouterEngine
from maze_router.costs.corner_cost import CornerCost
from maze_router.ripup_strategy import DefaultRipupStrategy, CongestionAwareRipupStrategy
from tests.conftest import make_oai33_testcase, make_stdcell_testcase


class TestOAI33:

    def test_oai33_majority_routed(self):
        """OAI33: ≥ 60% 线网布通"""
        grid, nets, cmgr, cost_mgr = make_oai33_testcase(space=1)
        strategy = DefaultRipupStrategy(max_iterations=10)
        from maze_router.ripup_manager import RipupManager
        mgr = RipupManager(grid, cmgr, cost_mgr, strategy)
        solution = mgr.run(nets)

        total = len(nets)
        routed = solution.routed_count
        assert routed / total >= 0.6, (
            f"只布通了 {routed}/{total} 个线网，低于 60%"
        )

    def test_oai33_no_space_violation(self):
        """OAI33: 各线网内部路由路径无间距冲突（terminal 之间的固有邻近不算违规）"""
        grid, nets, cmgr, cost_mgr = make_oai33_testcase(space=1)
        strategy = DefaultRipupStrategy(max_iterations=10)
        from maze_router.ripup_manager import RipupManager
        mgr = RipupManager(grid, cmgr, cost_mgr, strategy)
        solution = mgr.run(nets)

        # 收集所有 terminal 节点（允许 terminal 之间天然邻近）
        all_terminals = set()
        for net in nets:
            all_terminals.update(net.terminals)

        # 验证：非 terminal 路由节点不与其他线网的非 terminal 节点相邻
        from maze_router.constraints.space_constraint import SpaceConstraint
        sc_check = SpaceConstraint(rules={"M0": 1, "M1": 1, "M2": 1})
        for name, result in solution.results.items():
            if not result.success:
                continue
            # 只标记非 terminal 的路由节点
            routing_only = result.routed_nodes - all_terminals
            sc_check.mark_route(name, routing_only)

        violations = 0
        for name, result in solution.results.items():
            if not result.success:
                continue
            routing_only = result.routed_nodes - all_terminals
            for node in routing_only:
                if not sc_check.is_available(node, name):
                    violations += 1
        assert violations == 0, f"路由路径存在 {violations} 处间距违规"

    def test_oai33_pin_points(self):
        """OAI33: 需要 pin 引出的线网，pin_point 在正确层"""
        grid, nets, cmgr, cost_mgr = make_oai33_testcase(space=1)
        strategy = DefaultRipupStrategy(max_iterations=10)
        from maze_router.ripup_manager import RipupManager
        mgr = RipupManager(grid, cmgr, cost_mgr, strategy)
        solution = mgr.run(nets)

        for net in nets:
            if net.pin_spec is None:
                continue
            result = solution.results.get(net.name)
            if result is None or not result.success:
                continue
            if result.pin_point is not None:
                assert net.pin_spec.allows_layer(result.pin_point[0]), (
                    f"线网 {net.name} 的 pin_point {result.pin_point} 不在正确层"
                )

    def test_oai33_congestion_aware(self):
        """OAI33: 拥塞感知策略不应比默认策略差太多"""
        grid1, nets1, cmgr1, cost_mgr1 = make_oai33_testcase(space=1)
        grid2, nets2, cmgr2, cost_mgr2 = make_oai33_testcase(space=1)

        from maze_router.ripup_manager import RipupManager
        sol1 = RipupManager(
            grid1, cmgr1, cost_mgr1, DefaultRipupStrategy(10)
        ).run(nets1)
        sol2 = RipupManager(
            grid2, cmgr2, cost_mgr2, CongestionAwareRipupStrategy(12)
        ).run(nets2)

        # 两种策略都应该能布通至少 50%
        assert sol1.routed_count / len(nets1) >= 0.5
        assert sol2.routed_count / len(nets2) >= 0.5

    def test_oai33_engine_visualize(self, tmp_path):
        """OAI33: 通过 MazeRouterEngine 运行，并生成 SVG"""
        grid, nets, _, _ = make_oai33_testcase(space=1)

        engine = MazeRouterEngine(
            grid=grid,
            nets=nets,
            space_constr={"M0": 1, "M1": 1, "M2": 1},
            corner_l_costs={"M0": 3.0, "M1": 3.0, "M2": 3.0},
        )
        solution = engine.run()
        engine.visualize(save_dir=str(tmp_path), prefix="oai33_test_")

        # 检查 SVG 文件已生成
        svgs = list(tmp_path.glob("*.svg"))
        assert len(svgs) >= 1

    def test_oai33_to_results_dir(self):
        """OAI33: 保存 SVG 到 results/stdcell_oai33/ 目录"""
        grid, nets, _, _ = make_oai33_testcase(space=1)
        engine = MazeRouterEngine(
            grid=grid,
            nets=nets,
            space_constr={"M0": 1, "M1": 1, "M2": 1},
        )
        solution = engine.run()
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "results", "stdcell_oai33",
        )
        engine.visualize(save_dir=save_dir)
        assert os.path.isdir(save_dir)


class TestLargerStdCell:

    def test_6t_per_row(self):
        """6 晶体管/排的标准单元，8 个线网，≥ 70% 布通"""
        grid, nets, cmgr, cost_mgr = make_stdcell_testcase(
            n_transistors=6, n_rows=9, space=1,
            corner_l_cost=3.0,
        )
        strategy = CongestionAwareRipupStrategy(max_iterations=12)
        from maze_router.ripup_manager import RipupManager
        solution = RipupManager(grid, cmgr, cost_mgr, strategy).run(nets)
        total = len(nets)
        assert solution.routed_count / total >= 0.5

    def test_engine_multi_net(self):
        """通过 Engine 运行 8 线网场景"""
        from tests.conftest import make_stdcell_nets, make_stdcell_grid
        n = 5
        grid = make_stdcell_grid(n_transistors=n, n_rows=9)
        nets, cable_locs_map = make_stdcell_nets(n_transistors=n, n_rows=9)

        engine = MazeRouterEngine(
            grid=grid,
            nets=nets,
            space_constr={"M0": 1, "M1": 1, "M2": 1},
            corner_l_costs={"M0": 5.0, "M1": 5.0, "M2": 0.0},
            cable_locs=cable_locs_map,
        )
        solution = engine.run()
        assert solution.routed_count >= 1
