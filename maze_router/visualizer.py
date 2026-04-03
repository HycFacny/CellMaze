"""
Visualizer: 布线结果 SVG 可视化

将各层的布线结果保存为 SVG 文件（layer_m0.svg, layer_m1.svg, layer_m2.svg）。

SVG 内容：
  - 网格节点（灰色小圆点）
  - 布线路径（按线网着色）
  - Terminal 端口（大圆点，标注线网名）
  - Pin 引出点（星形标记，标注 "PIN"）
"""

from __future__ import annotations
import os
import logging
from typing import Dict, List, Optional, Set, Tuple

from maze_router.data.net import Node, RoutingSolution
from maze_router.data.grid import GridGraph

logger = logging.getLogger(__name__)

# 颜色列表（循环使用）
_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075",
]

_LAYER_BG = {
    "M0": "#f0f0ff",
    "M1": "#f0fff0",
    "M2": "#fff0f0",
}


class Visualizer:
    """
    布线结果可视化器。

    将 RoutingSolution 中的布线结果绘制为分层 SVG 图像。
    """

    def __init__(
        self,
        grid: GridGraph,
        solution: RoutingSolution,
        cell_size: int = 40,
        margin: int = 40,
    ):
        """
        参数:
            grid:      布线网格
            solution:  布线方案
            cell_size: 每个网格单元的像素大小
            margin:    图像边距
        """
        self.grid = grid
        self.solution = solution
        self.cell_size = cell_size
        self.margin = margin

        # 计算网格范围
        all_nodes = grid.get_all_nodes()
        if all_nodes:
            self._xs = [n[1] for n in all_nodes]
            self._ys = [n[2] for n in all_nodes]
            self._x_min = min(self._xs)
            self._x_max = max(self._xs)
            self._y_min = min(self._ys)
            self._y_max = max(self._ys)
        else:
            self._x_min = self._x_max = self._y_min = self._y_max = 0

        # 为每个线网分配颜色
        net_names = sorted(solution.results.keys())
        self._net_colors: Dict[str, str] = {
            name: _COLORS[i % len(_COLORS)]
            for i, name in enumerate(net_names)
        }

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def save_svgs(
        self,
        save_dir: str = "results",
        prefix: str = "",
        layers: Optional[List[str]] = None,
    ):
        """
        保存各层 SVG 文件。

        参数:
            save_dir: 保存目录（不存在则创建）
            prefix:   文件名前缀
            layers:   要保存的层（None=所有出现的层）
        """
        os.makedirs(save_dir, exist_ok=True)

        if layers is None:
            all_layers = set(n[0] for n in self.grid.get_all_nodes())
            # 过滤掉虚拟层（虚拟节点不在物理布线图中渲染）
            layers = sorted(l for l in all_layers if not self.grid.is_virtual_node((l, 0, 0)))

        for layer in layers:
            filename = f"{prefix}layer_{layer.lower()}.svg"
            filepath = os.path.join(save_dir, filename)
            svg = self._render_layer(layer)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(svg)
            logger.info(f"已保存 {filepath}")

    # ------------------------------------------------------------------
    # 内部渲染
    # ------------------------------------------------------------------

    def _coord_to_px(self, x: int, y: int) -> Tuple[float, float]:
        """网格坐标 → SVG 像素坐标（y 轴翻转使 y=0 在底部）。"""
        px = self.margin + (x - self._x_min) * self.cell_size
        # SVG y 轴向下，网格 y 轴向上，因此翻转
        py = self.margin + (self._y_max - y) * self.cell_size
        return px, py

    def _render_layer(self, layer: str) -> str:
        """生成指定层的 SVG 字符串。"""
        cs = self.cell_size
        w = self.margin * 2 + (self._x_max - self._x_min) * cs + cs
        h = self.margin * 2 + (self._y_max - self._y_min) * cs + cs

        lines = []
        lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">')
        lines.append(f'  <rect width="{w}" height="{h}" fill="{_LAYER_BG.get(layer, "#ffffff")}"/>')
        lines.append(f'  <text x="10" y="20" font-size="16" fill="#333">{layer} Layer</text>')

        # 网格节点（灰点）
        layer_nodes = self.grid.get_nodes_on_layer(layer)
        for node in layer_nodes:
            px, py = self._coord_to_px(node[1], node[2])
            lines.append(
                f'  <circle cx="{px}" cy="{py}" r="3" fill="#cccccc" opacity="0.6"/>'
            )

        # 网格同层边（浅灰虚线）
        drawn_edges: Set[Tuple[Node, Node]] = set()
        for node in layer_nodes:
            for nb in self.grid.get_neighbors(node):
                if nb[0] != layer:
                    continue
                edge_key = (min(node, nb), max(node, nb))
                if edge_key in drawn_edges:
                    continue
                drawn_edges.add(edge_key)
                x1, y1 = self._coord_to_px(node[1], node[2])
                x2, y2 = self._coord_to_px(nb[1], nb[2])
                lines.append(
                    f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                    f'stroke="#e0e0e0" stroke-width="1" stroke-dasharray="4,4"/>'
                )

        # 布线路径
        for net_name, result in self.solution.results.items():
            if not result.success:
                continue
            color = self._net_colors.get(net_name, "#000000")

            for edge in result.routed_edges:
                src, dst = edge
                if src[0] != layer and dst[0] != layer:
                    continue
                # via 边在两层都画
                if src[0] == layer:
                    x1, y1 = self._coord_to_px(src[1], src[2])
                else:
                    x1, y1 = self._coord_to_px(src[1], src[2])
                if dst[0] == layer:
                    x2, y2 = self._coord_to_px(dst[1], dst[2])
                else:
                    x2, y2 = self._coord_to_px(dst[1], dst[2])

                is_via = src[0] != dst[0]
                stroke_w = 2 if not is_via else 1
                dash = "" if not is_via else 'stroke-dasharray="3,3"'
                lines.append(
                    f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                    f'stroke="{color}" stroke-width="{stroke_w}" {dash} opacity="0.85"/>'
                )

        # Terminal 端口（大圆点）
        for net_name, result in self.solution.results.items():
            if not result.success:
                continue
            color = self._net_colors.get(net_name, "#000000")
            for terminal in result.net.terminals:
                if terminal[0] != layer:
                    continue
                px, py = self._coord_to_px(terminal[1], terminal[2])
                lines.append(
                    f'  <circle cx="{px}" cy="{py}" r="7" fill="{color}" '
                    f'stroke="white" stroke-width="1.5"/>'
                )
                lines.append(
                    f'  <text x="{px+9}" y="{py+4}" font-size="9" fill="{color}">'
                    f'{net_name}</text>'
                )

        # Pin 引出点（星形 / 菱形标记）
        for net_name, result in self.solution.results.items():
            if result.pin_point is None:
                continue
            pin = result.pin_point
            if pin[0] != layer:
                continue
            color = self._net_colors.get(net_name, "#000000")
            px, py = self._coord_to_px(pin[1], pin[2])
            s = 8
            # 绘制菱形
            pts = (
                f"{px},{py-s} {px+s},{py} {px},{py+s} {px-s},{py}"
            )
            lines.append(
                f'  <polygon points="{pts}" fill="{color}" stroke="black" stroke-width="1"/>'
            )
            lines.append(
                f'  <text x="{px+10}" y="{py+4}" font-size="9" fill="black">PIN</text>'
            )

        # 图例
        legend_y = h - self.margin + 10
        for i, (net_name, color) in enumerate(self._net_colors.items()):
            lx = self.margin + i * 80
            if lx + 70 > w:
                break
            lines.append(
                f'  <rect x="{lx}" y="{legend_y}" width="12" height="12" fill="{color}"/>'
            )
            lines.append(
                f'  <text x="{lx+15}" y="{legend_y+11}" font-size="10" fill="#333">'
                f'{net_name}</text>'
            )

        lines.append('</svg>')
        return "\n".join(lines)
