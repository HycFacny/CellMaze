"""
可视化模块

提供文本形式的网格可视化和SVG图片导出，显示每层的布线路径和禁区。
"""

import os
from typing import Dict, Set, Optional, List, Tuple
from maze_router.net import Node, Edge, RoutingSolution
from maze_router.grid import RoutingGrid
from maze_router.spacing import SpacingManager


class Visualizer:
    """
    布线结果可视化器。

    支持文本模式输出，在终端显示每层的网格布线情况。
    """

    # 线网显示字符映射（最多支持26个线网，用字母标识）
    NET_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    def __init__(self, grid: RoutingGrid):
        self.grid = grid

    def visualize_text(
        self,
        solution: RoutingSolution,
        spacing_mgr: Optional[SpacingManager] = None,
        show_spacing: bool = False,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> str:
        """
        生成文本形式的可视化输出。

        参数:
            solution: 布线方案
            spacing_mgr: 间距管理器（用于显示禁区）
            show_spacing: 是否显示禁区
            width: 网格宽度（自动检测如果为None）
            height: 网格高度（自动检测如果为None）
        返回:
            格式化的字符串
        """
        # 自动检测网格尺寸
        if width is None or height is None:
            all_nodes = self.grid.get_all_nodes()
            if not all_nodes:
                return "空网格"
            if width is None:
                width = max(n[1] for n in all_nodes) + 1
            if height is None:
                height = max(n[2] for n in all_nodes) + 1

        # 为每个线网分配显示字符
        net_char_map: Dict[str, str] = {}
        for i, net_name in enumerate(sorted(solution.results.keys())):
            if i < len(self.NET_CHARS):
                net_char_map[net_name] = self.NET_CHARS[i]
            else:
                net_char_map[net_name] = "#"

        # 构建每层的节点到线网映射
        node_net_map: Dict[Node, str] = {}
        terminal_set: Set[Node] = set()
        for net_name, result in solution.results.items():
            if result.success:
                for node in result.routed_nodes:
                    node_net_map[node] = net_name
                for terminal in result.net.terminals:
                    terminal_set.add(terminal)

        output_lines = []

        # 图例
        output_lines.append("=== 布线可视化 ===")
        output_lines.append("图例:")
        for net_name, char in net_char_map.items():
            result = solution.results[net_name]
            status = "成功" if result.success else "失败"
            output_lines.append(
                f"  {char}/{char.lower() if char.isupper() else char.upper()}: "
                f"{net_name} ({status}, 代价={result.total_cost:.1f})"
            )
        output_lines.append(f"  *: 端口(terminal)  .: 空节点  x: 不存在的节点")
        if show_spacing:
            output_lines.append(f"  ~: 禁区(spacing zone)")
        output_lines.append("")

        # 逐层显示
        for layer in sorted(self.grid.layers):
            output_lines.append(f"--- {layer} 层 ---")

            # 列标题
            header = "    " + "".join(f"{x:2d}" for x in range(width))
            output_lines.append(header)

            # 从上到下（y从大到小）显示
            for y in range(height - 1, -1, -1):
                row = f"{y:3d} "
                for x in range(width):
                    node = (layer, x, y)
                    if not self.grid.is_valid_node(node):
                        row += " x"
                    elif node in node_net_map:
                        net_name = node_net_map[node]
                        char = net_char_map.get(net_name, "#")
                        # 端口用大写/特殊标记显示
                        if node in terminal_set:
                            row += f" *"
                        else:
                            row += f" {char}"
                    elif show_spacing and spacing_mgr:
                        if spacing_mgr.is_occupied(node):
                            row += " #"
                        elif not spacing_mgr.is_available(node, "__check__"):
                            row += " ~"
                        else:
                            row += " ."
                    else:
                        row += " ."

                output_lines.append(row)
            output_lines.append("")

        # 摘要
        output_lines.append(f"=== 摘要 ===")
        output_lines.append(
            f"布通率: {solution.routed_count}/{len(solution.results)} "
            f"({100*solution.routed_count/max(1,len(solution.results)):.1f}%)"
        )
        output_lines.append(f"总代价: {solution.total_cost:.2f}")
        if solution.failed_nets:
            output_lines.append(f"未布通: {', '.join(solution.failed_nets)}")

        return "\n".join(output_lines)

    def print_solution(
        self,
        solution: RoutingSolution,
        spacing_mgr: Optional[SpacingManager] = None,
        show_spacing: bool = False,
    ):
        """直接打印可视化结果"""
        print(self.visualize_text(solution, spacing_mgr, show_spacing))


    # 线网颜色调色板（高对比度，适合EDA布线图）
    NET_COLORS = [
        "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
        "#264653", "#A8DADC", "#6A4C93", "#1982C4", "#8AC926",
        "#FF595E", "#FFCA3A", "#6A0572", "#AB83A1", "#36827F",
        "#C1666B", "#4ECDC4", "#556270", "#FF6B6B", "#FFA07A",
    ]

    def save_svg(
        self,
        solution: RoutingSolution,
        output_dir: Optional[str] = None,
        spacing_mgr: Optional[SpacingManager] = None,
        show_spacing: bool = False,
        cell_size: int = 40,
        padding: int = 60,
    ) -> List[str]:
        """
        将每层的布线结果导出为SVG文件。

        文件命名: layer_m0.svg, layer_m1.svg, layer_m2.svg
        默认保存到项目根目录下的 results/ 目录。

        参数:
            solution: 布线方案
            output_dir: 保存目录（默认为项目根目录下的results/）
            spacing_mgr: 间距管理器（用于显示禁区）
            show_spacing: 是否显示禁区
            cell_size: 每个网格单元的像素大小
            padding: 图片四周的留白
        返回:
            生成的SVG文件路径列表
        """
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(output_dir, exist_ok=True)

        # 自动检测网格尺寸
        all_nodes = self.grid.get_all_nodes()
        if not all_nodes:
            return []
        width = max(n[1] for n in all_nodes) + 1
        height = max(n[2] for n in all_nodes) + 1

        # 为每个线网分配颜色
        net_names = sorted(solution.results.keys())
        net_color_map: Dict[str, str] = {}
        for i, name in enumerate(net_names):
            net_color_map[name] = self.NET_COLORS[i % len(self.NET_COLORS)]

        # 构建每层的节点到线网映射
        node_net_map: Dict[Node, str] = {}
        terminal_set: Set[Node] = set()
        edge_set: Dict[str, List[Edge]] = {}
        for net_name, result in solution.results.items():
            if result.success:
                for node in result.routed_nodes:
                    node_net_map[node] = net_name
                for terminal in result.net.terminals:
                    terminal_set.add(terminal)
                edge_set[net_name] = list(result.routed_edges)

        saved_files: List[str] = []

        for layer in sorted(self.grid.layers):
            svg_content = self._render_layer_svg(
                layer, width, height, cell_size, padding,
                solution, net_color_map, node_net_map,
                terminal_set, edge_set, spacing_mgr, show_spacing,
            )

            filename = f"layer_{layer.lower()}.svg"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(svg_content)
            saved_files.append(filepath)

        return saved_files

    def _render_layer_svg(
        self,
        layer: str,
        width: int,
        height: int,
        cell_size: int,
        padding: int,
        solution: RoutingSolution,
        net_color_map: Dict[str, str],
        node_net_map: Dict[Node, str],
        terminal_set: Set[Node],
        edge_set: Dict[str, List[Edge]],
        spacing_mgr: Optional[SpacingManager],
        show_spacing: bool,
    ) -> str:
        """渲染单层的SVG内容"""
        svg_w = width * cell_size + padding * 2
        svg_h = height * cell_size + padding * 2

        parts: List[str] = []
        parts.append(
            f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{svg_w}" height="{svg_h}" '
            f'viewBox="0 0 {svg_w} {svg_h}">\n'
        )

        # 背景
        parts.append(f'<rect width="{svg_w}" height="{svg_h}" fill="white"/>\n')

        # 标题
        parts.append(
            f'<text x="{svg_w // 2}" y="25" text-anchor="middle" '
            f'font-family="monospace" font-size="18" font-weight="bold" '
            f'fill="#333">{layer} 层布线结果</text>\n'
        )

        def cx(x: int) -> float:
            return padding + x * cell_size + cell_size / 2

        def cy(y: int) -> float:
            # SVG y轴向下，布线网格y轴向上，翻转
            return padding + (height - 1 - y) * cell_size + cell_size / 2

        # 网格线
        for x in range(width):
            x_pos = cx(x)
            parts.append(
                f'<line x1="{x_pos}" y1="{padding}" '
                f'x2="{x_pos}" y2="{padding + (height - 1) * cell_size + cell_size}" '
                f'stroke="#E0E0E0" stroke-width="0.5"/>\n'
            )
        for y in range(height):
            y_pos = cy(y)
            parts.append(
                f'<line x1="{padding}" y1="{y_pos}" '
                f'x2="{padding + (width - 1) * cell_size + cell_size}" y2="{y_pos}" '
                f'stroke="#E0E0E0" stroke-width="0.5"/>\n'
            )

        # 坐标轴标注
        for x in range(width):
            parts.append(
                f'<text x="{cx(x)}" y="{padding - 8}" text-anchor="middle" '
                f'font-family="monospace" font-size="10" fill="#888">{x}</text>\n'
            )
        for y in range(height):
            parts.append(
                f'<text x="{padding - 12}" y="{cy(y) + 4}" text-anchor="end" '
                f'font-family="monospace" font-size="10" fill="#888">{y}</text>\n'
            )

        # 禁区显示
        if show_spacing and spacing_mgr:
            for y in range(height):
                for x in range(width):
                    node = (layer, x, y)
                    if node not in node_net_map and not spacing_mgr.is_available(node, "__check__"):
                        parts.append(
                            f'<rect x="{cx(x) - cell_size / 2 + 1}" '
                            f'y="{cy(y) - cell_size / 2 + 1}" '
                            f'width="{cell_size - 2}" height="{cell_size - 2}" '
                            f'fill="#FEE2E2" opacity="0.5"/>\n'
                        )

        # 布线边（线段）
        line_width = max(3, cell_size // 10)
        for net_name, edges in edge_set.items():
            color = net_color_map.get(net_name, "#999")
            for u, v in edges:
                if u[0] != layer or v[0] != layer:
                    continue
                parts.append(
                    f'<line x1="{cx(u[1])}" y1="{cy(u[2])}" '
                    f'x2="{cx(v[1])}" y2="{cy(v[2])}" '
                    f'stroke="{color}" stroke-width="{line_width}" '
                    f'stroke-linecap="round"/>\n'
                )

        # 布线节点（小圆点）——只画没有边连接到的孤立占用节点
        dot_r = max(2, cell_size // 8)
        for y in range(height):
            for x in range(width):
                node = (layer, x, y)
                if node in node_net_map:
                    net_name = node_net_map[node]
                    color = net_color_map.get(net_name, "#999")
                    parts.append(
                        f'<circle cx="{cx(x)}" cy="{cy(y)}" r="{dot_r}" '
                        f'fill="{color}"/>\n'
                    )

        # 端口（大圆环 + 标签）
        terminal_r = max(5, cell_size // 4)
        for net_name, result in solution.results.items():
            color = net_color_map.get(net_name, "#999")
            for terminal in result.net.terminals:
                if terminal[0] != layer:
                    continue
                tx, ty = terminal[1], terminal[2]
                parts.append(
                    f'<circle cx="{cx(tx)}" cy="{cy(ty)}" r="{terminal_r}" '
                    f'fill="{color}" stroke="white" stroke-width="2"/>\n'
                )
                # Via标记：如果该端口连接到其他层
                parts.append(
                    f'<text x="{cx(tx)}" y="{cy(ty) + 3}" text-anchor="middle" '
                    f'font-family="monospace" font-size="{max(8, cell_size // 5)}" '
                    f'font-weight="bold" fill="white">'
                    f'{net_name[:3]}</text>\n'
                )

        # 图例
        legend_y = svg_h - 30
        legend_x = padding
        for i, net_name in enumerate(sorted(solution.results.keys())):
            result = solution.results[net_name]
            color = net_color_map.get(net_name, "#999")
            status = "OK" if result.success else "FAIL"
            lx = legend_x + i * 120
            if lx + 110 > svg_w:
                break
            parts.append(
                f'<rect x="{lx}" y="{legend_y - 8}" width="12" height="12" '
                f'fill="{color}" rx="2"/>\n'
            )
            parts.append(
                f'<text x="{lx + 16}" y="{legend_y + 2}" '
                f'font-family="monospace" font-size="11" fill="#333">'
                f'{net_name}({status})</text>\n'
            )

        parts.append("</svg>\n")
        return "".join(parts)
