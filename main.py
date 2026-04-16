"""
main.py — CellMaze 布线器演示

演示 OAI33 标准单元的完整布线流程，并保存各层 SVG 到 results/stdcell_oai33/。
"""

import logging
import os

from maze_router import MazeRouterEngine
from tests.conftest import make_oai33_testcase, make_stdcell_testcase
from tests.test_mux2_x1_ml import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

def run_mux2():
    """Engine 端到端：MUX2 全 10 线网，space=0，至少 10/10 布通"""
    grid = build_mux2_grid()
    nets, active_rules = build_and2_nets(grid)
    engine = MazeRouterEngine(
        grid=grid,
        nets=nets,
        space_constr={"M0": 0, "M1": 0, "M2": 0},
        corner_l_costs={"M0": 1.0, "M1": 5.0, "M2": 0.0},
        corner_t_costs={"M0": 2.0, "M1": 10.0, "M2": 0.0},
        strategy="congestion_aware",
        max_iterations=80,
        net_active_must_occupy_num=active_rules,
        row_type_y_ranges=ROW_TYPE_Y_RANGES,
    )
    solution = engine.run()
    save_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "results", "stdcell_mux2",
    )
    from maze_router.visualizer import Visualizer
    viz = Visualizer(grid, solution)
    viz.save_svgs(save_dir=save_dir)

    for name in [net.name for net in nets]:
        assert solution.results[name].success, f"共栅 {name} 必须布通"

    assert solution.routed_count == 10, \
        f"AND2 布通实际 {solution.routed_count}/10"
    assert solution.routed_count >= 4, \
        f"Engine space=1 应至少 10/10 布通，实际 {solution.routed_count}/10"

if __name__ == "__main__":
    sol = run_mux2()
