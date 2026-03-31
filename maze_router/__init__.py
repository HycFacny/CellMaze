"""
CellMaze 布线器包

主要入口：
    from maze_router import MazeRouterEngine
    engine = MazeRouterEngine(grid, nets, space_constr={"M0":1,"M1":1,"M2":1})
    solution = engine.run()
"""

from maze_router.data.net import Node, Edge, PinSpec, Net, RoutingResult, RoutingSolution
from maze_router.data.grid import GridGraph
from maze_router.data.region import Region

from maze_router.constraints.space_constraint import SpaceConstraint
from maze_router.costs.corner_cost import CornerCost
from maze_router.costs.space_cost import SpaceCost, SpaceType, SpaceCostRule

from maze_router.constraint_manager import ConstraintManager
from maze_router.cost_manager import CostManager, RoutingContext
from maze_router.ripup_strategy import (
    RipupStrategy, DefaultRipupStrategy, CongestionAwareRipupStrategy,
    RouterType, RipupAction, RipupDecision,
)
from maze_router.engine import MazeRouterEngine
from maze_router.visualizer import Visualizer

__all__ = [
    # data
    "Node", "Edge", "PinSpec", "Net", "RoutingResult", "RoutingSolution",
    "GridGraph", "Region",
    # constraints
    "SpaceConstraint", "ConstraintManager",
    # costs
    "CornerCost", "SpaceCost", "SpaceType", "SpaceCostRule", "CostManager", "RoutingContext",
    # strategy
    "RipupStrategy", "DefaultRipupStrategy", "CongestionAwareRipupStrategy",
    "RouterType", "RipupAction", "RipupDecision",
    # top-level
    "MazeRouterEngine", "Visualizer",
]
