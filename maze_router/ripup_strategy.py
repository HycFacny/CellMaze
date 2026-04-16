"""
拆线重布策略 (RipupStrategy)

封装布线顺序、路由器类型选择、拆线方案决策等策略逻辑，
与布线流程解耦，便于扩展不同的策略。
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple

from maze_router.data.net import Net, Node
from maze_router.data.region import Region


# -----------------------------------------------------------------------
# 枚举
# -----------------------------------------------------------------------

class RouterType(Enum):
    """路由器类型"""
    STEINER_DP = "steiner_dp"       # Dreyfus-Wagner 精确 DP（≤10 terminals）
    MAZE_GREEDY = "maze_greedy"     # 增量贪心 A*


class RipupAction(Enum):
    """拆线动作"""
    SKIP = "skip"                   # 跳过（本轮不处理，后续再试）
    FAIL = "fail"                   # 标记为失败，不再尝试
    RIPUP_SINGLE = "ripup_single"   # 拆除单个阻塞线网
    RIPUP_MULTIPLE = "ripup_multiple"  # 拆除多个阻塞线网
    RIPUP_REGION = "ripup_region"   # 区域拆线（只拆区域内片段）


# -----------------------------------------------------------------------
# 拆线决策
# -----------------------------------------------------------------------

@dataclass
class RipupDecision:
    """拆线决策结果。"""
    action: RipupAction
    nets_to_ripup: List[str] = field(default_factory=list)
    reason: str = ""
    region: Optional[Region] = None


# -----------------------------------------------------------------------
# 策略基类
# -----------------------------------------------------------------------

class RipupStrategy(ABC):
    """
    拆线重布策略基类。

    子类实现以下方法来控制布线行为：
    - order_nets:          线网排序
    - get_router_type:     每个线网使用哪种路由器
    - order_terminals:     线网内端口排序（用于贪心路由）
    - decide_ripup:        拆线方案决策
    - get_max_iterations:  最大迭代次数
    """

    @abstractmethod
    def order_nets(self, nets: List[Net]) -> List[Net]:
        """对线网排序，返回按优先级排列的线网列表。"""
        ...

    @abstractmethod
    def get_router_type(self, net: Net, iteration: int) -> RouterType:
        """决定使用哪种路由器。"""
        ...

    @abstractmethod
    def order_terminals(self, net: Net, routed_nodes: Set[Node]) -> List[Node]:
        """对线网端口排序，供贪心路由逐一连接。"""
        ...

    @abstractmethod
    def decide_ripup(
        self,
        failed_net: Net,
        blocking_nets: Set[str],
        net_results: dict,
        iteration: int,
        region_ratios: Dict[str, float],
        region: Optional[Region],
    ) -> RipupDecision:
        """根据当前状态决定拆线方案。"""
        ...

    @abstractmethod
    def get_max_iterations(self) -> int:
        """最大迭代次数。"""
        ...

    def record_congestion(self, node: Node):
        """记录节点拥塞（可选实现）。"""
        pass

    def get_congestion_map(self) -> Dict[Node, float]:
        """返回拥塞地图（可选实现）。"""
        return {}

    def get_cost_multiplier(self, node: Node, net_name: str) -> float:
        """返回节点代价乘数（可选，供向后兼容）。"""
        return 1.0


# -----------------------------------------------------------------------
# 默认策略
# -----------------------------------------------------------------------

class DefaultRipupStrategy(RipupStrategy):
    """
    默认拆线重布策略。

    - 线网排序：按端口数降序（端口多的先布，更复杂的优先）
    - 路由器选择：端口数 ≤ 8 使用 SteinerDP，否则贪心
    - 拆线决策：
        1. 若阻塞线网中有优先级较低的，优先选择拆除
        2. 若区域比例 > 0.8，优先区域拆线
        3. 否则拆优先级最低的单个线网
    - 最大迭代：10
    """

    DP_TERMINAL_LIMIT = 8

    def __init__(self, max_iterations: int = 10):
        self._max_iterations = max_iterations

    def order_nets(self, nets: List[Net]) -> List[Net]:
        """按 priority 升序优先（priority 小的先布），priority 相同则按端口数降序。"""
        return sorted(nets, key=lambda n: (n.priority, -len(n.terminals)))

    def get_router_type(self, net: Net, iteration: int) -> RouterType:
        """端口数 ≤ DP_TERMINAL_LIMIT 时使用 DP，否则贪心。"""
        if len(net.terminals) <= self.DP_TERMINAL_LIMIT:
            return RouterType.STEINER_DP
        return RouterType.MAZE_GREEDY

    def order_terminals(self, net: Net, routed_nodes: Set[Node]) -> List[Node]:
        """
        贪心端口顺序：
        先选第一个端口，然后每次选与当前树最近（Manhattan距离）的未布端口。
        """
        terminals = list(net.terminals)
        if len(terminals) <= 1:
            return terminals

        ordered = [terminals[0]]
        remaining = list(terminals[1:])

        while remaining:
            tree_nodes = routed_nodes | set(ordered)
            best = min(
                remaining,
                key=lambda t: min(
                    abs(t[1] - s[1]) + abs(t[2] - s[2])
                    for s in tree_nodes
                )
            )
            ordered.append(best)
            remaining.remove(best)

        return ordered

    def decide_ripup(
        self,
        failed_net: Net,
        blocking_nets: Set[str],
        net_results: dict,
        iteration: int,
        region_ratios: Dict[str, float],
        region: Optional[Region],
    ) -> RipupDecision:
        """拆线决策逻辑。"""
        if not blocking_nets:
            return RipupDecision(
                action=RipupAction.FAIL,
                reason="无阻塞线网，布线空间不足"
            )

        # 在最后几轮迭代中优先选区域拆线
        high_ratio_nets = [
            n for n, r in region_ratios.items()
            if r > 0.5 and n in net_results and net_results[n].success
        ]
        if high_ratio_nets and region is not None and iteration <= self._max_iterations - 2:
            return RipupDecision(
                action=RipupAction.RIPUP_REGION,
                nets_to_ripup=high_ratio_nets[:2],
                reason=f"区域拆线：{high_ratio_nets[:2]}",
                region=region,
            )

        # 选优先级最低的阻塞线网拆除
        candidates = [
            n for n in blocking_nets
            if n in net_results and net_results[n].success
        ]
        if not candidates:
            return RipupDecision(
                action=RipupAction.SKIP,
                reason="阻塞线网均未布通，跳过"
            )

        # 找优先级（端口数越少 → 优先级越低 → 越容易被拆）
        def ripup_priority(name: str) -> int:
            result = net_results.get(name)
            if result is None:
                return 0
            return len(result.net.terminals)

        target = min(candidates, key=ripup_priority)
        return RipupDecision(
            action=RipupAction.RIPUP_SINGLE,
            nets_to_ripup=[target],
            reason=f"拆除阻塞线网 {target}",
        )

    def get_max_iterations(self) -> int:
        return self._max_iterations


# -----------------------------------------------------------------------
# 拥塞感知策略（PathFinder 风格）
# -----------------------------------------------------------------------

class CongestionAwareRipupStrategy(DefaultRipupStrategy):
    """
    拥塞感知拆线重布策略（PathFinder 风格）。

    在默认策略基础上增加：
    - 拥塞计数：拆线时记录节点拥塞
    - 代价乘数：拥塞节点的边代价随迭代轮次增加
    """

    def __init__(
        self,
        max_iterations: int = 15,
        base_penalty: float = 0.1,
        penalty_growth: float = 0.5,
    ):
        """
        参数:
            max_iterations: 最大迭代次数
            base_penalty:   基础拥塞惩罚
            penalty_growth: 每轮迭代的惩罚增长系数
        """
        super().__init__(max_iterations)
        self.base_penalty = base_penalty
        self.penalty_growth = penalty_growth
        self._congestion: Dict[Node, float] = {}

    def record_congestion(self, node: Node):
        self._congestion[node] = self._congestion.get(node, 0.0) + 1.0

    def get_congestion_map(self) -> Dict[Node, float]:
        return dict(self._congestion)

    def get_cost_multiplier(self, node: Node, net_name: str) -> float:
        """拥塞越高，代价乘数越大。"""
        cong = self._congestion.get(node, 0.0)
        if cong <= 0:
            return 1.0
        return 1.0 + self.base_penalty * (1 + self.penalty_growth) ** cong
