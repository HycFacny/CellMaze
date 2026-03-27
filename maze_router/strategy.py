"""
布线策略模块

将布线顺序、端口排序、代价偏好、拆线决策、Steiner方法选择等策略逻辑
从布线流程中解耦，以策略模式管理。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Set, Dict, Optional

from maze_router.net import Net, Node


class SteinerMethod(Enum):
    """Steiner树构建方法"""
    GREEDY = "greedy"   # 增量贪心法，O(k·N·logN)，近似比约2
    DP = "dp"           # Dreyfus-Wagner DP，O(3^k·N + 2^k·N·logN)，精确最优


class RipupAction(Enum):
    """拆线动作类型"""
    RIPUP_SINGLE = "ripup_single"       # 拆除单个阻塞线网
    RIPUP_MULTIPLE = "ripup_multiple"   # 拆除多个阻塞线网
    RIPUP_REGION = "ripup_region"       # 拆除区域内所有线网
    SKIP = "skip"                        # 跳过当前线网，稍后重试
    FAIL = "fail"                        # 放弃当前线网


@dataclass
class RipupDecision:
    """
    拆线决策结果。

    当action为RIPUP_REGION时，region字段描述冲突区域，
    只拆除阻塞线网在该区域内的片段。
    """
    action: RipupAction
    nets_to_ripup: List[str]    # 需要拆除的线网名称列表
    reason: str = ""
    region: object = None       # 可选的ConflictRegion（避免循环导入用object）


class RoutingStrategy(ABC):
    """
    布线策略抽象基类。

    定义布线过程中所有策略性决策的接口，
    包括线网排序、端口排序、代价计算和拆线决策。
    """

    @abstractmethod
    def order_nets(self, nets: List[Net]) -> List[Net]:
        """
        决定线网的布线顺序。

        参数:
            nets: 所有需要布线的线网
        返回:
            排序后的线网列表
        """
        ...

    @abstractmethod
    def order_terminals(self, net: Net, tree_nodes: Set[Node]) -> List[Node]:
        """
        决定一个线网内端口的连接顺序。

        参数:
            net: 当前线网
            tree_nodes: 当前已在树中的节点集合（初始为空）
        返回:
            排序后的端口列表
        """
        ...

    @abstractmethod
    def get_cost_multiplier(self, node: Node, net_name: str) -> float:
        """
        为节点返回代价乘数，用于引导布线偏好。

        参数:
            node: 当前节点
            net_name: 当前线网名称
        返回:
            代价乘数（1.0 表示无偏好）
        """
        ...

    @abstractmethod
    def decide_ripup(
        self,
        failed_net: Net,
        blocking_nets: Set[str],
        net_results: dict,
        iteration: int,
        region_ratios: Optional[Dict[str, float]] = None,
        region: object = None,
    ) -> RipupDecision:
        """
        当布线失败时决定拆线策略。

        参数:
            failed_net: 布线失败的线网
            blocking_nets: 阻塞路径的线网名称集合
            net_results: 当前各线网的布线结果
            iteration: 当前迭代轮次
            region_ratios: 各阻塞线网在冲突区域内的节点占比（可选）
            region: 冲突区域对象（可选）
        返回:
            拆线决策
        """
        ...

    @abstractmethod
    def get_max_iterations(self) -> int:
        """返回拆线重布的最大迭代次数"""
        ...

    @abstractmethod
    def get_steiner_method(self, net: Net, iteration: int) -> SteinerMethod:
        """
        决定对指定线网使用哪种Steiner树构建方法。

        可根据线网端口数量、当前迭代轮次等因素动态选择：
        - GREEDY: 速度快，适用于端口多或首轮快速布线
        - DP: 精确最优，适用于端口少或拆线重布后期需要更优解

        参数:
            net: 当前线网
            iteration: 当前拆线重布迭代轮次（首轮为1）
        返回:
            SteinerMethod 枚举值
        """
        ...


class DefaultStrategy(RoutingStrategy):
    """
    默认布线策略。

    线网排序: 端口少的优先，端口数相同时包围盒面积小的优先
    端口排序: 离重心最近的优先作为起始点，后续按离已有树的距离排序
    代价乘数: 无偏好（始终为1.0）
    拆线决策: 拆除占用节点最少的阻塞线网
    Steiner方法: 端口数≤dp_terminal_threshold时用DP，否则用贪心；
                 重布迭代后期强制用DP争取更优解
    """

    def __init__(
        self,
        max_iterations: int = 50,
        dp_terminal_threshold: int = 6,
        force_dp_after_ratio: float = 0.5,
        regional_ripup_threshold: float = 0.4,
    ):
        """
        参数:
            max_iterations: 最大拆线重布迭代次数
            dp_terminal_threshold: 端口数≤此值时使用DP方法
            force_dp_after_ratio: 迭代进度超过此比例后，对所有可行线网强制使用DP
            regional_ripup_threshold: 区域内节点占比低于此值时选择区域拆线
        """
        self._max_iterations = max_iterations
        self._dp_terminal_threshold = dp_terminal_threshold
        self._force_dp_after_ratio = force_dp_after_ratio
        self._regional_ripup_threshold = regional_ripup_threshold

    def order_nets(self, nets: List[Net]) -> List[Net]:
        """端口少的优先，包围盒小的优先"""
        return sorted(nets, key=lambda n: (n.terminal_count, n.bounding_box_area()))

    def order_terminals(self, net: Net, tree_nodes: Set[Node]) -> List[Node]:
        """
        第一个端口选离重心最近的，后续由Steiner树构建器动态排序。
        这里返回初始排序：离重心最近的排在第一位。
        """
        terminals = list(net.terminals)
        if not terminals:
            return []

        # 计算重心
        cx = sum(t[1] for t in terminals) / len(terminals)
        cy = sum(t[2] for t in terminals) / len(terminals)

        # 按到重心的曼哈顿距离排序
        terminals.sort(key=lambda t: abs(t[1] - cx) + abs(t[2] - cy))
        return terminals

    def get_cost_multiplier(self, node: Node, net_name: str) -> float:
        return 1.0

    def decide_ripup(
        self,
        failed_net: Net,
        blocking_nets: Set[str],
        net_results: dict,
        iteration: int,
        region_ratios: Optional[Dict[str, float]] = None,
        region: object = None,
    ) -> RipupDecision:
        """
        拆线策略：
        - 如果阻塞线网只有一个，直接拆除
        - 如果阻塞线网在冲突区域内占比低，使用区域拆线
        - 如果有多个阻塞线网，拆除其中占用节点最少的
        - 迭代次数过多时，尝试拆除所有阻塞线网
        - 没有阻塞线网信息时，跳过

        参数:
            failed_net: 布线失败的线网
            blocking_nets: 阻塞线网名称集合
            net_results: 当前各线网的布线结果
            iteration: 当前迭代轮次
            region_ratios: 各阻塞线网在冲突区域内的节点占比（可选）
            region: 冲突区域对象（可选）
        """
        if not blocking_nets:
            return RipupDecision(
                action=RipupAction.SKIP,
                nets_to_ripup=[],
                reason="未找到阻塞线网信息",
            )

        # 尝试区域拆线：当所有阻塞线网在区域内的占比都低于阈值时
        if region_ratios and region:
            candidates = [
                bn for bn, ratio in region_ratios.items()
                if 0 < ratio < self._regional_ripup_threshold
            ]
            if candidates:
                return RipupDecision(
                    action=RipupAction.RIPUP_REGION,
                    nets_to_ripup=candidates,
                    reason=(
                        f"区域拆线 {len(candidates)} 个线网"
                        f"（区域内占比均<{self._regional_ripup_threshold:.0%}）"
                    ),
                    region=region,
                )

        if len(blocking_nets) == 1:
            return RipupDecision(
                action=RipupAction.RIPUP_SINGLE,
                nets_to_ripup=list(blocking_nets),
                reason=f"拆除唯一的阻塞线网",
            )

        # 多个阻塞线网：迭代前期拆最小的，后期拆所有
        if iteration < self._max_iterations // 2:
            # 选择占用节点最少的线网拆除
            min_net = None
            min_count = float('inf')
            for bn in blocking_nets:
                if bn in net_results and net_results[bn].success:
                    count = len(net_results[bn].routed_nodes)
                    if count < min_count:
                        min_count = count
                        min_net = bn
            if min_net:
                return RipupDecision(
                    action=RipupAction.RIPUP_SINGLE,
                    nets_to_ripup=[min_net],
                    reason=f"拆除最小阻塞线网 {min_net}（{min_count}个节点）",
                )

        # 后期策略：拆除所有阻塞线网
        return RipupDecision(
            action=RipupAction.RIPUP_MULTIPLE,
            nets_to_ripup=list(blocking_nets),
            reason=f"拆除所有 {len(blocking_nets)} 个阻塞线网",
        )

    def get_max_iterations(self) -> int:
        return self._max_iterations

    def get_steiner_method(self, net: Net, iteration: int) -> SteinerMethod:
        """
        策略：
        - 首轮布线时，端口数≤阈值用DP，否则用贪心
        - 迭代后期（重布阶段），对端口数≤8的线网强制用DP争取更优解
        - 端口数>8时始终用贪心（DP的3^k开销过大）
        """
        k = net.terminal_count
        # 硬上限：端口数>8时DP内存/时间不可接受
        if k > 8:
            return SteinerMethod.GREEDY
        # 迭代后期强制DP
        if iteration > self._max_iterations * self._force_dp_after_ratio:
            return SteinerMethod.DP
        # 正常阈值判断
        if k <= self._dp_terminal_threshold:
            return SteinerMethod.DP
        return SteinerMethod.GREEDY


class CongestionAwareStrategy(DefaultStrategy):
    """
    拥塞感知策略。

    在默认策略的基础上，根据拥塞历史调整代价乘数，
    使布线倾向于绕过拥塞区域。类似PathFinder算法的思想。
    """

    def __init__(self, max_iterations: int = 50, congestion_weight: float = 0.5):
        super().__init__(max_iterations)
        self.congestion_weight = congestion_weight
        # 节点的历史拥塞计数
        self._congestion_history: Dict[Node, int] = {}

    def record_congestion(self, node: Node):
        """记录节点的拥塞事件"""
        self._congestion_history[node] = self._congestion_history.get(node, 0) + 1

    def get_cost_multiplier(self, node: Node, net_name: str) -> float:
        """根据拥塞历史增加代价"""
        history = self._congestion_history.get(node, 0)
        return 1.0 + self.congestion_weight * history

    def get_congestion_map(self) -> Dict[Node, float]:
        """获取拥塞代价映射，可传入MazeRouter"""
        return {
            node: self.congestion_weight * count
            for node, count in self._congestion_history.items()
        }
