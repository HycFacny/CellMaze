"""
CornerManager 测试

测试 CornerManager 的 L型折角（方向感知）和 T型折角（Steiner分支）代价
的配置、查询及对布线结果的影响。
"""

import pytest
from maze_router.net import Net
from maze_router.grid import RoutingGrid
from maze_router.spacing import SpacingManager
from maze_router.corner import CornerManager
from maze_router.router import MazeRouter
from maze_router.steiner import SteinerTreeBuilder


# ======================================================================
# CornerManager API 单元测试
# ======================================================================

class TestCornerManagerAPI:
    """CornerManager 接口与查询逻辑测试"""

    def test_default_l_cost(self):
        """默认配置：L型折角代价为5.0"""
        cm = CornerManager()
        assert cm.get_l_cost("M0", 1, 3) == 5.0
        assert cm.get_l_cost("M1", 2, 4) == 5.0

    def test_disabled_l_cost(self):
        """l_costs={} 禁用所有L型折角代价"""
        cm = CornerManager(l_costs={})
        assert cm.get_l_cost("M0", 1, 3) == 0.0
        assert cm.get_l_cost("M2", 3, 2) == 0.0

    def test_layer_l_cost(self):
        """按层配置L型折角代价"""
        cm = CornerManager(l_costs={"M0": 3.0, "M1": 7.0})
        assert cm.get_l_cost("M0", 1, 3) == 3.0
        assert cm.get_l_cost("M1", 2, 4) == 7.0
        assert cm.get_l_cost("M2", 1, 3) == 0.0   # 未配置层返回0

    def test_directional_l_cost_priority(self):
        """精细方向键优先于层键"""
        cm = CornerManager(l_costs={
            "M0": 5.0,
            ("M0", 1, 3): 10.0,   # right→up
            ("M0", 3, 1): 8.0,    # up→right
        })
        assert cm.get_l_cost("M0", 1, 3) == 10.0   # 精细键
        assert cm.get_l_cost("M0", 3, 1) == 8.0    # 精细键
        assert cm.get_l_cost("M0", 1, 4) == 5.0    # 回退到层键（无对应精细键）

    def test_directional_l_cost_fallback(self):
        """无精细键时回退到层键，无层键时回退到0"""
        cm = CornerManager(l_costs={("M0", 1, 3): 10.0})
        assert cm.get_l_cost("M0", 1, 3) == 10.0   # 精细键命中
        assert cm.get_l_cost("M0", 2, 4) == 0.0    # 无精细键，无层键，回退0

    def test_default_t_cost_is_zero(self):
        """默认配置：T型折角代价为0.0（关闭）"""
        cm = CornerManager()
        assert cm.get_t_cost_flat("M0") == 0.0
        assert cm.get_t_cost("M0", 1, 3) == 0.0

    def test_layer_t_cost(self):
        """按层配置T型折角代价"""
        cm = CornerManager(t_costs={"M0": 5.0, "M1": 3.0})
        assert cm.get_t_cost_flat("M0") == 5.0
        assert cm.get_t_cost_flat("M1") == 3.0
        assert cm.get_t_cost_flat("M2") == 0.0   # 未配置层返回0

    def test_directional_t_cost(self):
        """按方向配置T型折角代价"""
        cm = CornerManager(t_costs={
            "M0": 3.0,
            ("M0", 1, 3): 7.0,   # 右+上 T型
        })
        # 方向对无序：(1,3) == (3,1)
        assert cm.get_t_cost("M0", 1, 3) == 7.0
        assert cm.get_t_cost("M0", 3, 1) == 7.0   # 对称
        assert cm.get_t_cost("M0", 1, 4) == 3.0   # 回退到层键

    def test_disabled_t_cost(self):
        """t_costs={} 禁用所有T型折角代价"""
        cm = CornerManager(t_costs={})
        assert cm.get_t_cost_flat("M0") == 0.0
        assert cm.get_t_cost("M0", 1, 3) == 0.0

    def test_factory_default(self):
        """CornerManager.default() 等价于 CornerManager()"""
        cm1 = CornerManager.default()
        cm2 = CornerManager()
        assert cm1.get_l_cost("M0", 1, 3) == cm2.get_l_cost("M0", 1, 3)
        assert cm1.get_t_cost_flat("M0") == cm2.get_t_cost_flat("M0")

    def test_factory_disabled(self):
        """CornerManager.disabled() 关闭所有代价"""
        cm = CornerManager.disabled()
        assert cm.get_l_cost("M0", 1, 3) == 0.0
        assert cm.get_t_cost_flat("M0") == 0.0

    def test_from_layer_costs(self):
        """from_layer_costs() 向后兼容辅助方法"""
        cm = CornerManager.from_layer_costs(
            l_layer_costs={"M0": 5.0, "M1": 5.0},
            t_layer_costs={"M0": 2.0},
        )
        assert cm.get_l_cost("M0", 1, 3) == 5.0
        assert cm.get_t_cost_flat("M0") == 2.0


# ======================================================================
# L型折角与路由集成测试
# ======================================================================

class TestLCornerRouting:
    """L型折角代价对路由影响的集成测试"""

    def test_directional_l_cost_asymmetry(self):
        """
        方向非对称L代价影响路由选择。

        设置 right→up 代价极高（100），up→right 代价低（0）；
        从(0,0)到(3,3)，最优路径应选择先上后右（避免right→up折角）。
        """
        grid = RoutingGrid.build_grid(layers=["M0"], width=5, height=5)
        sm = SpacingManager({"M0": 0})

        # right→up极高代价，up→right零代价
        cm = CornerManager(l_costs={
            ("M0", 1, 3): 100.0,  # right→up
            ("M0", 3, 1): 0.0,    # up→right
        })
        router = MazeRouter(grid, sm, corner_mgr=cm)
        result = router.route(
            sources={("M0", 0, 0)},
            targets={("M0", 3, 3)},
            net_name="n",
        )
        assert result.success
        # up→right路径代价=6（无折角惩罚）
        assert result.cost == pytest.approx(6.0)

    def test_uniform_l_cost_both_directions(self):
        """
        uniform L代价（无方向区分）应使两方向折角代价相同。
        """
        grid = RoutingGrid.build_grid(layers=["M0"], width=5, height=5)
        sm = SpacingManager({"M0": 0})

        cm = CornerManager(l_costs={"M0": 5.0})
        router = MazeRouter(grid, sm, corner_mgr=cm)
        result = router.route(
            sources={("M0", 0, 0)},
            targets={("M0", 3, 3)},
            net_name="n",
        )
        assert result.success
        # 最优：边代价6 + 1×折角5 = 11
        assert result.cost == pytest.approx(11.0)

    def test_none_corner_mgr_uses_default(self):
        """corner_mgr=None 等价于默认5.0每层L代价"""
        grid = RoutingGrid.build_grid(layers=["M0"], width=5, height=5)
        sm = SpacingManager({"M0": 0})

        r1 = MazeRouter(grid, sm, corner_mgr=None).route(
            sources={("M0", 0, 0)}, targets={("M0", 3, 3)}, net_name="n"
        )
        r2 = MazeRouter(grid, sm, corner_mgr=CornerManager()).route(
            sources={("M0", 0, 0)}, targets={("M0", 3, 3)}, net_name="n"
        )
        assert r1.success and r2.success
        assert r1.cost == pytest.approx(r2.cost)


# ======================================================================
# T型折角与Steiner树集成测试
# ======================================================================

class TestTCornerSteiner:
    """T型折角代价对Steiner树构建的影响测试"""

    def test_t_cost_increases_steiner_cost(self):
        """
        T型折角代价使Steiner树总代价增加（3端口树必然有分支节点）。
        """
        grid = RoutingGrid.build_grid(layers=["M0"], width=8, height=8)
        sm = SpacingManager({"M0": 0})

        cm_no = CornerManager(l_costs={}, t_costs={})
        cm_hi = CornerManager(l_costs={}, t_costs={"M0": 50.0})

        net = Net("n", [("M0", 0, 0), ("M0", 7, 0), ("M0", 4, 7)])
        r_no = SteinerTreeBuilder(grid, sm, corner_mgr=cm_no).build_tree_dp(net)
        r_hi = SteinerTreeBuilder(grid, sm, corner_mgr=cm_hi).build_tree_dp(net)

        assert r_no.success and r_hi.success
        assert r_hi.total_cost > r_no.total_cost

    def test_t_cost_zero_no_effect(self):
        """
        T型折角代价为0时不影响Steiner树代价。
        """
        grid = RoutingGrid.build_grid(layers=["M0"], width=8, height=8)
        sm = SpacingManager({"M0": 0})

        cm1 = CornerManager(l_costs={}, t_costs={})
        cm2 = CornerManager(l_costs={}, t_costs={"M0": 0.0})

        net = Net("n", [("M0", 0, 0), ("M0", 7, 0), ("M0", 4, 7)])
        r1 = SteinerTreeBuilder(grid, sm, corner_mgr=cm1).build_tree_dp(net)
        r2 = SteinerTreeBuilder(grid, sm, corner_mgr=cm2).build_tree_dp(net)

        assert r1.success and r2.success
        assert r1.total_cost == pytest.approx(r2.total_cost)

    def test_t_cost_two_terminal_no_branch(self):
        """
        两端口线网无分支节点，T型折角代价不影响代价。
        """
        grid = RoutingGrid.build_grid(layers=["M0"], width=8, height=1)
        sm = SpacingManager({"M0": 0})

        cm_no = CornerManager(l_costs={}, t_costs={})
        cm_hi = CornerManager(l_costs={}, t_costs={"M0": 100.0})

        net = Net("n", [("M0", 0, 0), ("M0", 7, 0)])
        r_no = SteinerTreeBuilder(grid, sm, corner_mgr=cm_no).build_tree_dp(net)
        r_hi = SteinerTreeBuilder(grid, sm, corner_mgr=cm_hi).build_tree_dp(net)

        assert r_no.success and r_hi.success
        # 两端口退化为最短路径，无分支，T代价不生效
        assert r_no.total_cost == pytest.approx(r_hi.total_cost)

    def test_t_cost_combined_with_l_cost(self):
        """
        L型和T型折角代价可以同时配置，各自独立作用。
        """
        grid = RoutingGrid.build_grid(layers=["M0"], width=8, height=8)
        sm = SpacingManager({"M0": 0})

        # 只有L代价
        cm_l = CornerManager(l_costs={"M0": 5.0}, t_costs={})
        # 只有T代价
        cm_t = CornerManager(l_costs={}, t_costs={"M0": 50.0})
        # 两者都有
        cm_lt = CornerManager(l_costs={"M0": 5.0}, t_costs={"M0": 50.0})
        # 两者都无
        cm_none = CornerManager(l_costs={}, t_costs={})

        net = Net("n", [("M0", 0, 0), ("M0", 7, 0), ("M0", 4, 7)])
        r_l = SteinerTreeBuilder(grid, sm, corner_mgr=cm_l).build_tree_dp(net)
        r_t = SteinerTreeBuilder(grid, sm, corner_mgr=cm_t).build_tree_dp(net)
        r_lt = SteinerTreeBuilder(grid, sm, corner_mgr=cm_lt).build_tree_dp(net)
        r_none = SteinerTreeBuilder(grid, sm, corner_mgr=cm_none).build_tree_dp(net)

        assert r_l.success and r_t.success and r_lt.success and r_none.success

        # 有代价时总代价不小于无代价
        assert r_l.total_cost >= r_none.total_cost
        assert r_t.total_cost >= r_none.total_cost
        assert r_lt.total_cost >= r_none.total_cost
