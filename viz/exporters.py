"""
图表导出器
提供图表导出为多种格式的功能（PNG、PDF、HTML、JSON等）
"""

import os
import io
import base64
import json
from typing import Dict, Any, Optional, Union, BinaryIO
from pathlib import Path

try:
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import kaleido
    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False


class ChartExporter:
    """
    图表导出器，支持多种格式导出
    """

    def __init__(self, backend: str = "plotly"):
        """
        初始化导出器

        Args:
            backend: 后端类型，"plotly"或"matplotlib"
        """
        self.backend = backend.lower()

        if self.backend == "plotly" and not PLOTLY_AVAILABLE:
            raise ImportError("Plotly not installed. Please install with: pip install plotly")
        if self.backend == "matplotlib" and not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib not installed. Please install with: pip install matplotlib")

    def export_chart(self,
                    chart,
                    output_path: str,
                    format: str = "auto",
                    width: int = 800,
                    height: int = 600,
                    scale: float = 2.0,
                    **kwargs) -> bool:
        """
        导出图表到文件

        Args:
            chart: 图表对象
            output_path: 输出文件路径
            format: 输出格式，"png"、"pdf"、"svg"、"jpg"、"html"、"json"，或"auto"（根据扩展名自动检测）
            width: 图像宽度（像素）
            height: 图像高度（像素）
            scale: 缩放因子
            **kwargs: 额外参数传递给后端导出函数

        Returns:
            成功返回True，失败返回False
        """
        if format == "auto":
            # 根据扩展名自动检测格式
            ext = Path(output_path).suffix.lower()
            if ext in ['.png']:
                format = "png"
            elif ext in ['.pdf']:
                format = "pdf"
            elif ext in ['.svg']:
                format = "svg"
            elif ext in ['.jpg', '.jpeg']:
                format = "jpg"
            elif ext in ['.html', '.htm']:
                format = "html"
            elif ext in ['.json']:
                format = "json"
            else:
                format = "png"  # 默认

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        try:
            if self.backend == "plotly":
                return self._export_plotly_chart(chart, output_path, format, width, height, scale, **kwargs)
            elif self.backend == "matplotlib":
                return self._export_mpl_chart(chart, output_path, format, width, height, scale, **kwargs)
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
        except Exception as e:
            print(f"导出图表失败: {e}")
            return False

    def save_chart_as_image(self,
                           chart,
                           output_path: str,
                           format: str = "png",
                           **kwargs) -> bool:
        """
        保存图表为图像文件

        Args:
            chart: 图表对象
            output_path: 输出文件路径
            format: 图像格式，"png"、"pdf"、"svg"、"jpg"
            **kwargs: 额外参数

        Returns:
            成功返回True，失败返回False
        """
        return self.export_chart(chart, output_path, format=format, **kwargs)

    def save_chart_as_html(self,
                          chart,
                          output_path: str,
                          include_plotlyjs: bool = True,
                          full_html: bool = True,
                          **kwargs) -> bool:
        """
        保存图表为HTML文件

        Args:
            chart: 图表对象
            output_path: 输出文件路径
            include_plotlyjs: 是否包含Plotly.js库
            full_html: 是否生成完整的HTML文档
            **kwargs: 额外参数

        Returns:
            成功返回True，失败返回False
        """
        if self.backend != "plotly":
            raise ValueError("HTML导出仅支持Plotly后端")

        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # 导出HTML
            pio.write_html(
                chart,
                file=output_path,
                include_plotlyjs=include_plotlyjs,
                full_html=full_html,
                **kwargs
            )
            print(f"图表已保存为HTML: {output_path}")
            return True
        except Exception as e:
            print(f"保存HTML图表失败: {e}")
            return False

    def chart_to_base64(self,
                       chart,
                       format: str = "png",
                       **kwargs) -> Optional[str]:
        """
        将图表转换为Base64编码字符串

        Args:
            chart: 图表对象
            format: 图像格式，"png"、"pdf"、"svg"、"jpg"
            **kwargs: 额外参数

        Returns:
            Base64编码字符串，失败返回None
        """
        try:
            if self.backend == "plotly":
                img_bytes = pio.to_image(chart, format=format, **kwargs)
            elif self.backend == "matplotlib":
                # 对于matplotlib，先保存到内存缓冲区
                buf = io.BytesIO()
                chart.savefig(buf, format=format, **kwargs)
                buf.seek(0)
                img_bytes = buf.read()
            else:
                return None

            # 转换为Base64
            base64_str = base64.b64encode(img_bytes).decode('utf-8')
            return base64_str
        except Exception as e:
            print(f"转换为Base64失败: {e}")
            return None

    def chart_to_bytes(self,
                      chart,
                      format: str = "png",
                      **kwargs) -> Optional[bytes]:
        """
        将图表转换为字节数据

        Args:
            chart: 图表对象
            format: 图像格式
            **kwargs: 额外参数

        Returns:
            字节数据，失败返回None
        """
        try:
            if self.backend == "plotly":
                return pio.to_image(chart, format=format, **kwargs)
            elif self.backend == "matplotlib":
                buf = io.BytesIO()
                chart.savefig(buf, format=format, **kwargs)
                buf.seek(0)
                return buf.read()
            else:
                return None
        except Exception as e:
            print(f"转换为字节数据失败: {e}")
            return None

    def chart_to_json(self,
                     chart,
                     output_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        将图表转换为JSON数据

        Args:
            chart: 图表对象
            output_path: 输出文件路径（可选）

        Returns:
            JSON字典，失败返回None
        """
        if self.backend != "plotly":
            raise ValueError("JSON导出仅支持Plotly后端")

        try:
            # 转换为JSON字典
            chart_dict = pio.to_json(chart)

            if output_path:
                # 确保输出目录存在
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)

                # 保存到文件
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(chart_dict, f, ensure_ascii=False, indent=2)
                print(f"图表已保存为JSON: {output_path}")

            return chart_dict
        except Exception as e:
            print(f"转换为JSON失败: {e}")
            return None

    def _export_plotly_chart(self,
                            chart,
                            output_path: str,
                            format: str,
                            width: int,
                            height: int,
                            scale: float,
                            **kwargs) -> bool:
        """导出Plotly图表"""
        try:
            if format == "html":
                pio.write_html(
                    chart,
                    file=output_path,
                    **kwargs
                )
            elif format == "json":
                chart_dict = pio.to_json(chart)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(chart_dict, f, ensure_ascii=False, indent=2)
            else:
                # 图像格式
                pio.write_image(
                    chart,
                    file=output_path,
                    format=format,
                    width=width,
                    height=height,
                    scale=scale,
                    **kwargs
                )

            print(f"Plotly图表已导出: {output_path} ({format})")
            return True
        except Exception as e:
            print(f"导出Plotly图表失败: {e}")
            return False

    def _export_mpl_chart(self,
                         chart,
                         output_path: str,
                         format: str,
                         width: int,
                         height: int,
                         scale: float,
                         **kwargs) -> bool:
        """导出Matplotlib图表"""
        try:
            # 设置图表尺寸（英寸）
            dpi = 100
            width_inch = width / dpi
            height_inch = height / dpi

            # 调整图表尺寸
            chart.set_size_inches(width_inch, height_inch)

            # 保存图表
            chart.savefig(
                output_path,
                format=format,
                dpi=dpi * scale,
                bbox_inches='tight',
                **kwargs
            )

            print(f"Matplotlib图表已导出: {output_path} ({format})")
            return True
        except Exception as e:
            print(f"导出Matplotlib图表失败: {e}")
            return False


# 便捷函数
def export_chart(chart,
                output_path: str,
                backend: str = "plotly",
                **kwargs) -> bool:
    """
    便捷函数：导出图表

    Args:
        chart: 图表对象
        output_path: 输出文件路径
        backend: 后端类型
        **kwargs: 额外参数

    Returns:
        成功返回True，失败返回False
    """
    exporter = ChartExporter(backend=backend)
    return exporter.export_chart(chart, output_path, **kwargs)


def save_chart_as_image(chart,
                       output_path: str,
                       backend: str = "plotly",
                       **kwargs) -> bool:
    """
    便捷函数：保存图表为图像

    Args:
        chart: 图表对象
        output_path: 输出文件路径
        backend: 后端类型
        **kwargs: 额外参数

    Returns:
        成功返回True，失败返回False
    """
    exporter = ChartExporter(backend=backend)
    return exporter.save_chart_as_image(chart, output_path, **kwargs)


def save_chart_as_html(chart,
                      output_path: str,
                      backend: str = "plotly",
                      **kwargs) -> bool:
    """
    便捷函数：保存图表为HTML

    Args:
        chart: 图表对象
        output_path: 输出文件路径
        backend: 后端类型
        **kwargs: 额外参数

    Returns:
        成功返回True，失败返回False
    """
    if backend != "plotly":
        raise ValueError("HTML导出仅支持Plotly后端")

    exporter = ChartExporter(backend=backend)
    return exporter.save_chart_as_html(chart, output_path, **kwargs)


def chart_to_base64(chart,
                   backend: str = "plotly",
                   **kwargs) -> Optional[str]:
    """
    便捷函数：图表转换为Base64

    Args:
        chart: 图表对象
        backend: 后端类型
        **kwargs: 额外参数

    Returns:
        Base64字符串，失败返回None
    """
    exporter = ChartExporter(backend=backend)
    return exporter.chart_to_base64(chart, **kwargs)


def chart_to_bytes(chart,
                  backend: str = "plotly",
                  **kwargs) -> Optional[bytes]:
    """
    便捷函数：图表转换为字节数据

    Args:
        chart: 图表对象
        backend: 后端类型
        **kwargs: 额外参数

    Returns:
        字节数据，失败返回None
    """
    exporter = ChartExporter(backend=backend)
    return exporter.chart_to_bytes(chart, **kwargs)


# 批量导出功能
def batch_export_charts(charts_dict: Dict[str, Any],
                       output_dir: str,
                       backend: str = "plotly",
                       format: str = "png",
                       **kwargs) -> Dict[str, bool]:
    """
    批量导出多个图表

    Args:
        charts_dict: 图表字典，{图表名称: 图表对象}
        output_dir: 输出目录
        backend: 后端类型
        format: 输出格式
        **kwargs: 额外参数

    Returns:
        导出结果字典，{图表名称: 成功与否}
    """
    results = {}
    exporter = ChartExporter(backend=backend)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    for chart_name, chart in charts_dict.items():
        # 生成输出文件名
        if format == "html":
            ext = ".html"
        elif format == "json":
            ext = ".json"
        else:
            ext = f".{format}"

        output_path = os.path.join(output_dir, f"{chart_name}{ext}")

        try:
            success = exporter.export_chart(
                chart,
                output_path,
                format=format,
                **kwargs
            )
            results[chart_name] = success
        except Exception as e:
            print(f"批量导出图表 '{chart_name}' 失败: {e}")
            results[chart_name] = False

    return results