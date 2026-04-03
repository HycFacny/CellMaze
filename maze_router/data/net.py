"""
数据类定义：Net, PinSpec, RoutingResult, RoutingSolution

Node = (layer, x, y)
Edge = (Node, Node)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# -----------------------------------------------------------------------
# 基础类型
# -----------------------------------------------------------------------

Node = Tuple[str, int, int]   # (layer, x, y)
Edge = Tuple[Node, Node]


# -----------------------------------------------------------------------
# Pin 引出规格
# -----------------------------------------------------------------------

@dataclass
class PinSpec:
    """Pin 引出规格。layer = "M0" | "M1" | "M2" | "Any"。"""
    layer: str   # "M0" | "M1" | "M2" | "Any"

    def needs_extraction(self) -> bool:
        return self.layer != "M0"   # M0 terminal 本身就在 M0，无需引出

    def allows_layer(self, layer: str) -> bool:
        if self.layer == "Any":
            return layer in ("M1", "M2")
        return layer == self.layer


# -----------------------------------------------------------------------
# 线网
# -----------------------------------------------------------------------

@dataclass
class Net:
    """
    线网对象。

    Attributes:
        name:       线网名称
        terminals:  所有端口节点列表（通常位于 M0 层边界）
        cable_locs: M0 层上该线网可走的节点集合（None=不限制）
        pin_spec:   Pin 引出规格（None=不需要引出）
        priority:   布线优先级（越小越先布），用于 strategy 排序
    """
    name: str
    terminals: List[Node]
    cable_locs: Optional[Set[Node]] = None
    pin_spec: Optional[PinSpec] = None
    priority: int = 0
    active_must_occupy_num: Optional[Dict[Tuple[int, int], int]] = None
    """per-SD-terminal active 占用约束：{(j, i): N}，j=列，i=行类型，N=M1最少节点数。"""

    def terminal_set(self) -> Set[Node]:
        return set(self.terminals)


# -----------------------------------------------------------------------
# 布线结果
# -----------------------------------------------------------------------

class RoutingResult:
    """
    单个线网的布线结果。

    Attributes:
        net:          对应线网
        routed_nodes: 所有已布线节点
        routed_edges: 所有已布线边
        pin_point:    Pin 引出点（仅在 pin_spec 不为 None 时设置）
        total_cost:   总代价
        success:      是否布通
    """

    def __init__(
        self,
        net: Net,
        routed_nodes: Optional[Set[Node]] = None,
        routed_edges: Optional[List[Edge]] = None,
        pin_point: Optional[Node] = None,
        total_cost: float = 0.0,
        success: bool = False,
    ):
        self.net = net
        self.routed_nodes: Set[Node] = routed_nodes if routed_nodes is not None else set()
        self.routed_edges: List[Edge] = routed_edges if routed_edges is not None else []
        self.pin_point: Optional[Node] = pin_point
        self.total_cost: float = total_cost
        self.success: bool = success

    def __repr__(self) -> str:
        status = "OK" if self.success else "FAIL"
        return (
            f"RoutingResult({self.net.name}, {status}, "
            f"nodes={len(self.routed_nodes)}, cost={self.total_cost:.2f})"
        )


# -----------------------------------------------------------------------
# 布线方案
# -----------------------------------------------------------------------

class RoutingSolution:
    """所有线网的布线结果集合。"""

    def __init__(self):
        self.results: Dict[str, RoutingResult] = {}

    def add_result(self, result: RoutingResult):
        self.results[result.net.name] = result

    @property
    def routed_count(self) -> int:
        return sum(1 for r in self.results.values() if r.success)

    @property
    def total_cost(self) -> float:
        return sum(r.total_cost for r in self.results.values() if r.success)

    @property
    def failed_nets(self) -> List[str]:
        return [name for name, r in self.results.items() if not r.success]

    def __repr__(self) -> str:
        total = len(self.results)
        routed = self.routed_count
        return f"RoutingSolution({routed}/{total} routed, cost={self.total_cost:.2f})"
