"""
可视化模块：提供现代化、交互式的医疗数据可视化

包含：
1. 医疗专业配色方案
2. 核心图表构建器
3. SOFA评分专用图表
4. 模型分析图表
5. 生命体征图表
6. 图表导出器

支持多平台：
- Tkinter GUI
- Streamlit应用
- HTML页面
- 静态导出（PNG、PDF、HTML）
"""

from .themes import MEDICAL_THEME, apply_medical_theme
from .chart_builder import ChartBuilder, ChartBackend
from .sofa_charts import create_sofa_trend_chart, create_sofa_component_chart
from .model_charts import create_model_trust_chart, create_confidence_chart
# from .vital_signs_charts import create_vital_signs_chart  # TODO: Implement later
from .exporters import export_chart, save_chart_as_image, save_chart_as_html

__all__ = [
    'MEDICAL_THEME',
    'apply_medical_theme',
    'ChartBuilder',
    'ChartBackend',
    'create_sofa_trend_chart',
    'create_sofa_component_chart',
    'create_model_trust_chart',
    'create_confidence_chart',
    # 'create_vital_signs_chart',
    'export_chart',
    'save_chart_as_image',
    'save_chart_as_html',
]

__version__ = '1.0.0'