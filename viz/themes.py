"""
医疗专业配色方案
为医疗数据可视化设计的现代化配色方案
"""

MEDICAL_THEME = {
    # 主色调 - 专业蓝色系
    "primary": ["#1E88E5", "#1565C0", "#0D47A1"],  # 渐进蓝
    "secondary": ["#43A047", "#2E7D32", "#1B5E20"],  # 生命绿
    "accent": ["#FF7043", "#F4511E", "#D84315"],  # 警报橙
    "neutral": ["#78909C", "#546E7A", "#37474F"],  # 中性灰

    # SOFA组件专用配色
    "sofa_components": [
        "#4FC3F7",  # 呼吸系统 - 浅蓝
        "#7986CB",  # 凝血功能 - 紫蓝
        "#4DB6AC",  # 肝脏功能 - 青绿
        "#FFB74D",  # 心血管系统 - 橙黄
        "#BA68C8",  # 中枢神经系统 - 紫色
        "#81C784",  # 肾脏功能 - 绿色
    ],

    # 模型信任度梯度
    "trust_gradient": ["#FF6B6B", "#FFD166", "#06D6A0"],  # 低-中-高：红-黄-绿

    # 生命体征配色
    "vital_signs": {
        "heart_rate": "#EF5350",      # 心率 - 红色
        "blood_pressure": "#AB47BC",  # 血压 - 紫色
        "respiratory_rate": "#29B6F6", # 呼吸频率 - 蓝色
        "temperature": "#FFA726",     # 体温 - 橙色
        "oxygen_saturation": "#66BB6A" # 血氧饱和度 - 绿色
    },

    # 风险等级
    "risk_levels": {
        "low": "#4CAF50",      # 低风险 - 绿色
        "moderate": "#FFC107", # 中等风险 - 黄色
        "high": "#FF9800",     # 高风险 - 橙色
        "critical": "#F44336"  # 极高风险 - 红色
    },

    # 图表元素
    "grid_color": "#E0E0E0",
    "background_color": "#FFFFFF",
    "text_color": "#212121",
    "axis_color": "#757575",

    # 字体配置
    "font_family": "Arial, Helvetica, sans-serif",
    "title_font_size": 16,
    "axis_font_size": 12,
    "legend_font_size": 11,
}

def get_sofa_component_color(system_name: str) -> str:
    """
    根据SOFA系统名称获取对应的颜色

    Args:
        system_name: 系统名称，如"respiration", "coagulation", "liver", "cardiovascular", "cns", "renal"

    Returns:
        对应的颜色代码
    """
    system_map = {
        "respiration": 0,
        "coagulation": 1,
        "liver": 2,
        "cardiovascular": 3,
        "cns": 4,
        "renal": 5,
        "呼吸系统": 0,
        "凝血功能": 1,
        "肝脏功能": 2,
        "心血管系统": 3,
        "中枢神经系统": 4,
        "肾脏功能": 5,
    }

    idx = system_map.get(system_name.lower(), 0)
    return MEDICAL_THEME["sofa_components"][idx % len(MEDICAL_THEME["sofa_components"])]

def get_risk_level_color(risk_level: str) -> str:
    """
    根据风险等级获取对应的颜色

    Args:
        risk_level: 风险等级，如"low", "moderate", "high", "critical"

    Returns:
        对应的颜色代码，默认为中性灰
    """
    return MEDICAL_THEME["risk_levels"].get(risk_level.lower(), MEDICAL_THEME["neutral"][0])

def apply_medical_theme(fig=None, backend="plotly"):
    """
    应用医疗主题到图表

    Args:
        fig: 图表对象（Plotly Figure或matplotlib Figure）
        backend: 后端类型，"plotly"或"matplotlib"

    Returns:
        应用主题后的图表对象
    """
    if backend == "plotly":
        import plotly.graph_objects as go
        if fig is None:
            fig = go.Figure()

        # 更新布局
        fig.update_layout(
            font_family=MEDICAL_THEME["font_family"],
            font_color=MEDICAL_THEME["text_color"],
            title_font_size=MEDICAL_THEME["title_font_size"],
            plot_bgcolor=MEDICAL_THEME["background_color"],
            paper_bgcolor=MEDICAL_THEME["background_color"],
        )

        # 更新坐标轴
        fig.update_xaxes(
            gridcolor=MEDICAL_THEME["grid_color"],
            linecolor=MEDICAL_THEME["axis_color"],
            tickfont_size=MEDICAL_THEME["axis_font_size"],
        )
        fig.update_yaxes(
            gridcolor=MEDICAL_THEME["grid_color"],
            linecolor=MEDICAL_THEME["axis_color"],
            tickfont_size=MEDICAL_THEME["axis_font_size"],
        )

    elif backend == "matplotlib":
        import matplotlib.pyplot as plt

        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.gca()

        # 设置颜色
        ax.set_facecolor(MEDICAL_THEME["background_color"])
        fig.patch.set_facecolor(MEDICAL_THEME["background_color"])

        # 设置网格
        ax.grid(True, color=MEDICAL_THEME["grid_color"], linestyle='-', alpha=0.3)

        # 设置坐标轴颜色
        ax.spines['bottom'].set_color(MEDICAL_THEME["axis_color"])
        ax.spines['top'].set_color(MEDICAL_THEME["axis_color"])
        ax.spines['right'].set_color(MEDICAL_THEME["axis_color"])
        ax.spines['left'].set_color(MEDICAL_THEME["axis_color"])

        # 设置刻度颜色
        ax.tick_params(colors=MEDICAL_THEME["axis_color"])

        # 设置标签颜色
        ax.xaxis.label.set_color(MEDICAL_THEME["text_color"])
        ax.yaxis.label.set_color(MEDICAL_THEME["text_color"])
        ax.title.set_color(MEDICAL_THEME["text_color"])

    return fig

def create_color_scale(colors: list, n_steps: int = 10):
    """
    创建颜色渐变

    Args:
        colors: 颜色列表
        n_steps: 渐变步数

    Returns:
        渐变颜色列表
    """
    import plotly.express as px
    import numpy as np

    if len(colors) == 1:
        return colors * n_steps

    # 使用Plotly的color scale
    scale = px.colors.sample_colorscale(colors, np.linspace(0, 1, n_steps))
    return scale