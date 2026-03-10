"""
SOFA评分专用图表
提供SOFA评分相关的高级可视化功能
"""

from typing import Dict, Any, Optional, List, Union
import numpy as np

from .chart_builder import ChartBuilder, ChartBackend
from .themes import MEDICAL_THEME, get_sofa_component_color, get_risk_level_color


def create_sofa_trend_chart(hourly_totals: Dict[str, Union[int, float]],
                           baseline_totals: Optional[Dict[str, Union[int, float]]] = None,
                           title: str = "SOFA总分随时间变化",
                           backend: str = "plotly",
                           **kwargs) -> Any:
    """
    创建SOFA总分趋势图

    Args:
        hourly_totals: 干预组SOFA总分，{时间点: 总分}
        baseline_totals: 基线组SOFA总分（可选）
        title: 图表标题
        backend: 后端类型，"plotly"或"matplotlib"
        **kwargs: 额外参数传递给图表构建器

    Returns:
        图表对象
    """
    builder = ChartBuilder(backend=backend)
    return builder.create_sofa_trend_chart(
        hourly_totals=hourly_totals,
        baseline_totals=baseline_totals,
        title=title,
        **kwargs
    )


def create_sofa_component_chart(sofa_scores: Dict[str, Union[int, float]],
                               title: str = "SOFA各组件得分",
                               backend: str = "plotly",
                               **kwargs) -> Any:
    """
    创建SOFA各组件得分柱状图

    Args:
        sofa_scores: SOFA组件得分，如{"sofa_respiration": 2, "sofa_coagulation": 1, ...}
        title: 图表标题
        backend: 后端类型
        **kwargs: 额外参数

    Returns:
        图表对象
    """
    builder = ChartBuilder(backend=backend)
    return builder.create_sofa_component_chart(
        sofa_scores=sofa_scores,
        title=title,
        **kwargs
    )


def create_sofa_radar_chart(sofa_scores: Dict[str, Union[int, float]],
                           title: str = "SOFA评分雷达图",
                           max_score: int = 4,
                           **kwargs) -> Any:
    """
    创建SOFA评分雷达图（仅支持Plotly后端）

    Args:
        sofa_scores: SOFA组件得分
        title: 图表标题
        max_score: 最大得分（用于标准化）
        **kwargs: 额外参数

    Returns:
        Plotly雷达图
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        raise ImportError("雷达图需要Plotly库，请安装: pip install plotly")

    # 准备数据
    categories = []
    values = []
    colors = []

    system_names = {
        "sofa_respiration": "呼吸系统",
        "sofa_coagulation": "凝血功能",
        "sofa_liver": "肝脏功能",
        "sofa_cardiovascular": "心血管系统",
        "sofa_cns": "中枢神经系统",
        "sofa_renal": "肾脏功能",
    }

    for key, value in sofa_scores.items():
        if key.startswith("sofa_") and key != "sofa_total":
            system_key = key.replace("sofa_", "")
            categories.append(system_names.get(key, system_key))
            values.append(float(value) / max_score)  # 标准化到0-1
            colors.append(get_sofa_component_color(system_key))

    # 闭合雷达图
    categories = categories + [categories[0]]
    values = values + [values[0]]

    # 创建雷达图
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(30, 136, 229, 0.3)',  # 半透明蓝色
        line=dict(color=MEDICAL_THEME["primary"][0], width=2),
        marker=dict(size=8, color=MEDICAL_THEME["primary"][0]),
        name='SOFA评分',
        hovertemplate='系统: %{theta}<br>标准化得分: %{r:.2f}<extra></extra>'
    ))

    # 应用主题
    from .themes import apply_medical_theme
    fig = apply_medical_theme(fig, backend="plotly")

    # 更新布局
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=MEDICAL_THEME["title_font_size"]),
            x=0.5,
            xanchor='center'
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont_size=10,
                gridcolor=MEDICAL_THEME["grid_color"],
                linecolor=MEDICAL_THEME["axis_color"],
            ),
            angularaxis=dict(
                gridcolor=MEDICAL_THEME["grid_color"],
                linecolor=MEDICAL_THEME["axis_color"],
                rotation=90,  # 旋转角度
                direction="clockwise"
            ),
            bgcolor=MEDICAL_THEME["background_color"]
        ),
        showlegend=False,
        hovermode='closest'
    )

    return fig


def create_sofa_heatmap(sofa_scores_series: Dict[str, List[Union[int, float]]],
                       time_points: Optional[List[str]] = None,
                       title: str = "SOFA评分热力图",
                       **kwargs) -> Any:
    """
    创建SOFA评分热力图（时间序列）

    Args:
        sofa_scores_series: SOFA评分时间序列，{组件名称: [得分列表]}
        time_points: 时间点标签（可选）
        title: 图表标题
        **kwargs: 额外参数

    Returns:
        Plotly热力图
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        raise ImportError("热力图需要Plotly库，请安装: pip install plotly")

    # 准备数据
    systems = []
    data_matrix = []

    system_names = {
        "sofa_respiration": "呼吸",
        "sofa_coagulation": "凝血",
        "sofa_liver": "肝脏",
        "sofa_cardiovascular": "心血管",
        "sofa_cns": "中枢神经",
        "sofa_renal": "肾脏",
    }

    for key, values in sofa_scores_series.items():
        if key.startswith("sofa_") and key != "sofa_total":
            systems.append(system_names.get(key, key.replace("sofa_", "")))
            data_matrix.append(values)

    # 转置矩阵：行=时间点，列=系统
    if data_matrix:
        data_matrix = np.array(data_matrix).T.tolist()
    else:
        data_matrix = []

    if time_points is None and data_matrix:
        time_points = [f"T+{i}" for i in range(len(data_matrix))]

    # 创建热力图
    fig = go.Figure(data=go.Heatmap(
        z=data_matrix,
        x=systems,
        y=time_points if time_points else [],
        colorscale='Blues',  # 蓝色系
        colorbar=dict(title="SOFA得分"),
        hovertemplate='时间: %{y}<br>系统: %{x}<br>得分: %{z}<extra></extra>',
        showscale=True
    ))

    # 应用主题
    from .themes import apply_medical_theme
    fig = apply_medical_theme(fig, backend="plotly")

    # 更新布局
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=MEDICAL_THEME["title_font_size"]),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="系统",
        yaxis_title="时间点",
        xaxis=dict(tickangle=45),
        yaxis=dict(autorange="reversed"),  # 时间从上到下
    )

    return fig


def create_sofa_risk_assessment_chart(sofa_total: Union[int, float],
                                     risk_level: str,
                                     previous_scores: Optional[List[Union[int, float]]] = None,
                                     title: str = "SOFA风险评估",
                                     **kwargs) -> Any:
    """
    创建SOFA风险评估图表

    Args:
        sofa_total: 当前SOFA总分
        risk_level: 风险等级（low/moderate/high/critical）
        previous_scores: 历史SOFA总分（可选，用于趋势）
        title: 图表标题
        **kwargs: 额外参数

    Returns:
        图表对象
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("风险评估图表需要Plotly库")

    # 创建子图
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        subplot_titles=('SOFA总分趋势', '风险评估'),
        specs=[[{"type": "scatter"}, {"type": "indicator"}]]
    )

    # 左侧：趋势图
    if previous_scores:
        time_points = [f"T-{len(previous_scores)-i}" for i in range(len(previous_scores))]
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=previous_scores,
                mode='lines+markers',
                name='历史SOFA总分',
                line=dict(color=MEDICAL_THEME["neutral"][0], width=2),
                marker=dict(size=8),
                hovertemplate='时间: %{x}<br>SOFA总分: %{y}<extra></extra>'
            ),
            row=1, col=1
        )

        # 添加当前点
        fig.add_trace(
            go.Scatter(
                x=[f"当前"],
                y=[sofa_total],
                mode='markers',
                name='当前SOFA总分',
                marker=dict(
                    size=12,
                    color=get_risk_level_color(risk_level),
                    line=dict(width=2, color='white')
                ),
                hovertemplate='当前SOFA总分: %{y}<br>风险等级: %{text}<extra></extra>',
                text=[risk_level]
            ),
            row=1, col=1
        )

    # 右侧：指示器图
    risk_colors = {
        "low": "green",
        "moderate": "yellow",
        "high": "orange",
        "critical": "red"
    }

    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=sofa_total,
            title=dict(text="SOFA总分"),
            domain=dict(row=0, column=1),
            gauge=dict(
                axis=dict(range=[0, 24], tickwidth=1, tickcolor="darkblue"),
                bar=dict(color=get_risk_level_color(risk_level)),
                bgcolor="white",
                borderwidth=2,
                bordercolor="gray",
                steps=[
                    dict(range=[0, 6], color="lightgreen"),
                    dict(range=[6, 10], color="lightyellow"),
                    dict(range=[10, 13], color="lightorange"),
                    dict(range=[13, 24], color="lightcoral")
                ],
                threshold=dict(
                    line=dict(color="red", width=4),
                    thickness=0.75,
                    value=sofa_total
                )
            )
        ),
        row=1, col=2
    )

    # 应用主题
    from .themes import apply_medical_theme
    fig = apply_medical_theme(fig, backend="plotly")

    # 更新布局
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=MEDICAL_THEME["title_font_size"]),
            x=0.5,
            xanchor='center'
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig


def extract_sofa_data_from_result(result_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    从结果数据中提取SOFA相关数据

    Args:
        result_data: 预测结果数据

    Returns:
        SOFA数据字典，包含hourly_totals, sofa_scores等
    """
    sofa_data = {
        "hourly_totals": {},
        "sofa_scores": {},
        "sofa_scores_series": {},
        "baseline_totals": None,
    }

    if not isinstance(result_data, dict):
        return sofa_data

    # 提取hourly_sofa_totals
    if "hourly_sofa_totals" in result_data:
        sofa_data["hourly_totals"] = result_data["hourly_sofa_totals"]
    elif "prediction_data" in result_data:
        pred_data = result_data.get("prediction_data", {})
        if isinstance(pred_data, dict):
            sofa_data["hourly_totals"] = pred_data.get("hourly_sofa_totals", {})

    # 提取sofa_scores
    if "predicted_sofa_scores" in result_data:
        sofa_data["sofa_scores"] = result_data["predicted_sofa_scores"]
    elif "computed_sofa_scores" in result_data:
        sofa_data["sofa_scores"] = result_data["computed_sofa_scores"]
    elif "prediction_data" in result_data:
        pred_data = result_data.get("prediction_data", {})
        if isinstance(pred_data, dict):
            pred = pred_data.get("prediction", {})
            if isinstance(pred, dict):
                sofa_data["sofa_scores"] = pred.get("sofa_scores", {})

    # 提取sofa_scores_series
    if "predicted_sofa_scores_series" in result_data:
        sofa_data["sofa_scores_series"] = result_data["predicted_sofa_scores_series"]

    # 尝试提取基线数据
    if "baseline" in result_data:
        baseline = result_data.get("baseline", {})
        if isinstance(baseline, dict) and "hourly_sofa_totals" in baseline:
            sofa_data["baseline_totals"] = baseline["hourly_sofa_totals"]

    return sofa_data