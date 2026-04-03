"""
main.py — CellMaze 布线器演示

演示 OAI33 标准单元的完整布线流程，并保存各层 SVG 到 results/stdcell_oai33/。
"""

import logging
import os

from maze_router import MazeRouterEngine
from tests.conftest import make_oai33_testcase, make_stdcell_testcase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_oai33():
    """OAI33 演示。"""
    logger.info("=== OAI33 标准单元布线演示 ===")
    grid, nets, _, _ = make_oai33_testcase(space=1, corner_l_cost=3.0)

    engine = MazeRouterEngine(
        grid=grid,
        nets=nets,
        space_constr={"M0": 1, "M1": 1, "M2": 1},
        corner_l_costs={"M0": 3.0, "M1": 3.0, "M2": 0.0},
        corner_t_costs={"M0": 1.0, "M1": 1.0},
        strategy="congestion_aware",
        max_iterations=12,
    )

    solution = engine.run()

    save_dir = os.path.join("results", "stdcell_oai33")
    engine.visualize(save_dir=save_dir)

    logger.info(f"布线结果: {solution}")
    logger.info(f"SVG 保存到: {save_dir}/")

    failed = solution.failed_nets
    if failed:
        logger.warning(f"未布通线网: {failed}")
    else:
        logger.info("所有线网布通！")

    return solution


def run_larger():
    """更大规模的标准单元演示（6 晶体管/排）。"""
    logger.info("=== 6T 标准单元布线演示 ===")
    grid, nets, _, _ = make_stdcell_testcase(
        n_transistors=6,
        n_rows=9,
        space=1,
        corner_l_cost=5.0,
        corner_t_cost=1.0,
        soft_space_penalty=0.5,
    )

    engine = MazeRouterEngine(
        grid=grid,
        nets=nets,
        space_constr={"M0": 1, "M1": 1, "M2": 1},
        corner_l_costs={"M0": 5.0, "M1": 5.0, "M2": 0.0},
        corner_t_costs={"M0": 1.0},
        space_cost_rules=[
            ("M0", "S2S", 2, 0.5),
            ("M1", "T2S", 2, 0.3),
        ],
        strategy="congestion_aware",
        max_iterations=15,
    )

    solution = engine.run()

    save_dir = os.path.join("results", "stdcell_6t")
    engine.visualize(save_dir=save_dir)

    logger.info(f"布线结果: {solution}")
    logger.info(f"SVG 保存到: {save_dir}/")
    return solution


if __name__ == "__main__":
    sol1 = run_oai33()
    sol2 = run_larger()

    print("\n" + "=" * 60)
    print("OAI33 结果:", sol1)
    print("6T 结果:  ", sol2)
    print("=" * 60)
