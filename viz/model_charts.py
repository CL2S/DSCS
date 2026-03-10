"""
模型分析图表
提供模型信任度、置信度、性能比较等可视化功能
"""

from typing import Dict, Any, Optional, List, Union
import json
import os

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.subplots as sp
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .themes import MEDICAL_THEME, apply_medical_theme
from .chart_builder import ChartBuilder, ChartBackend


def create_model_trust_chart(model_data: List[Dict[str, Any]],
                           title: str = "模型信任度分析",
                           backend: str = "plotly",
                           **kwargs) -> Any:
    """
    创建模型信任度分析图

    Args:
        model_data: 模型数据列表，每个字典应包含：
            - model_name: 模型名称
            - avg_score: 平均信任分数
            - count: 样本数量
            - param_size: 参数大小（可选）
        title: 图表标题
        backend: 后端类型
        **kwargs: 额外参数

    Returns:
        图表对象
    """
    builder = ChartBuilder(backend=backend)
    return builder.create_model_trust_chart(
        model_data=model_data,
        title=title,
        **kwargs
    )


def create_confidence_chart(confidence: float,
                          title: str = "模型置信度",
                          backend: str = "plotly",
                          **kwargs) -> Any:
    """
    创建模型置信度图表

    Args:
        confidence: 置信度值（0-1）
        title: 图表标题
        backend: 后端类型
        **kwargs: 额外参数

    Returns:
        图表对象
    """
    builder = ChartBuilder(backend=backend)
    return builder.create_confidence_chart(
        confidence=confidence,
        title=title,
        **kwargs
    )


def create_model_comparison_radar(models_performance: Dict[str, Dict[str, float]],
                                metrics: List[str],
                                title: str = "模型性能雷达图",
                                **kwargs) -> Any:
    """
    创建模型性能比较雷达图（仅支持Plotly）

    Args:
        models_performance: 模型性能数据，{模型名: {指标: 得分}}
        metrics: 要显示的指标列表
        title: 图表标题
        **kwargs: 额外参数

    Returns:
        Plotly雷达图
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("雷达图需要Plotly库，请安装: pip install plotly")

    # 准备数据
    fig = go.Figure()

    # 颜色列表
    colors = MEDICAL_THEME["primary"] + MEDICAL_THEME["secondary"] + MEDICAL_THEME["accent"]

    # 为每个模型添加雷达图
    for i, (model_name, performance) in enumerate(models_performance.items()):
        # 提取指标值，缺失值用0填充
        values = [performance.get(metric, 0.0) for metric in metrics]
        # 归一化到0-1（假设所有指标都是越高越好）
        max_val = max(values) if values else 1.0
        if max_val > 0:
            normalized_values = [v / max_val for v in values]
        else:
            normalized_values = values

        # 闭合雷达图
        radar_values = normalized_values + [normalized_values[0]]
        radar_metrics = metrics + [metrics[0]]

        # 添加雷达图
        fig.add_trace(go.Scatterpolar(
            r=radar_values,
            theta=radar_metrics,
            fill='toself',
            fillcolor=f'rgba{(*tuple(int(colors[i % len(colors)][j:j+2], 16) for j in (1, 3, 5)), 0.2)}',
            line=dict(color=colors[i % len(colors)], width=2),
            name=model_name,
            hovertemplate='模型: %{text}<br>指标: %{theta}<br>得分: %{r:.2f}<extra></extra>',
            text=[model_name] * len(radar_metrics)
        ))

    # 应用主题
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
                rotation=90,
                direction="clockwise"
            ),
            bgcolor=MEDICAL_THEME["background_color"]
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='closest'
    )

    return fig


def create_model_heatmap(models_data: Dict[str, Dict[str, float]],
                        title: str = "模型性能热力图",
                        **kwargs) -> Any:
    """
    创建模型性能热力图

    Args:
        models_data: 模型数据，{模型名: {指标: 得分}}
        title: 图表标题
        **kwargs: 额外参数

    Returns:
        Plotly热力图
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("热力图需要Plotly库，请安装: pip install plotly")

    # 准备数据
    models = list(models_data.keys())
    if not models:
        raise ValueError("没有模型数据")

    # 收集所有指标
    all_metrics = set()
    for performance in models_data.values():
        all_metrics.update(performance.keys())
    metrics = sorted(list(all_metrics))

    # 构建数据矩阵
    data_matrix = []
    for model in models:
        row = []
        for metric in metrics:
            row.append(models_data[model].get(metric, 0.0))
        data_matrix.append(row)

    # 创建热力图
    fig = go.Figure(data=go.Heatmap(
        z=data_matrix,
        x=metrics,
        y=models,
        colorscale='Viridis',  # 使用Viridis颜色方案
        colorbar=dict(title="得分"),
        hovertemplate='模型: %{y}<br>指标: %{x}<br>得分: %{z:.3f}<extra></extra>',
        showscale=True
    ))

    # 应用主题
    fig = apply_medical_theme(fig, backend="plotly")

    # 更新布局
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=MEDICAL_THEME["title_font_size"]),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="性能指标",
        yaxis_title="模型",
        xaxis=dict(tickangle=45),
    )

    return fig


def create_dynamic_trust_evolution(trust_history: Dict[str, List[float]],
                                 time_points: Optional[List[str]] = None,
                                 title: str = "动态信任度演化",
                                 **kwargs) -> Any:
    """
    创建动态信任度演化图

    Args:
        trust_history: 信任度历史数据，{模型名: [信任度列表]}
        time_points: 时间点标签（可选）
        title: 图表标题
        **kwargs: 额外参数

    Returns:
        图表对象
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("动态图需要Plotly库，请安装: pip install plotly")

    # 准备数据
    fig = go.Figure()

    # 颜色列表
    colors = MEDICAL_THEME["primary"] + MEDICAL_THEME["secondary"]

    # 为每个模型添加线
    for i, (model_name, trust_values) in enumerate(trust_history.items()):
        if time_points is None or len(time_points) != len(trust_values):
            x_values = list(range(len(trust_values)))
        else:
            x_values = time_points

        fig.add_trace(go.Scatter(
            x=x_values,
            y=trust_values,
            mode='lines+markers',
            name=model_name,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=6),
            hovertemplate='时间: %{x}<br>模型: %{text}<br>信任度: %{y:.3f}<extra></extra>',
            text=[model_name] * len(trust_values)
        ))

    # 应用主题
    fig = apply_medical_theme(fig, backend="plotly")

    # 更新布局
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=MEDICAL_THEME["title_font_size"]),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="时间/轮次",
        yaxis_title="信任度",
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    return fig


def create_model_selection_breakdown(eval_breakdown: List[Dict[str, Any]],
                                   title: str = "模型选择得分分解",
                                   **kwargs) -> Any:
    """
    创建模型选择得分分解图

    Args:
        eval_breakdown: 评估分解数据，每个字典包含：
            - evaluator: 评估者模型
            - score: 评分
            - trust_weight: 信任权重
            - weighted_score: 加权得分
        title: 图表标题
        **kwargs: 额外参数

    Returns:
        图表对象
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("分解图需要Plotly库，请安装: pip install plotly")

    # 准备数据
    evaluators = [item.get("evaluator", f"Evaluator {i}") for i, item in enumerate(eval_breakdown)]
    scores = [item.get("score", 0.0) for item in eval_breakdown]
    trust_weights = [item.get("trust_weight", 1.0) for item in eval_breakdown]
    weighted_scores = [item.get("weighted_score", 0.0) for item in eval_breakdown]

    # 创建分组柱状图
    fig = go.Figure()

    # 原始得分
    fig.add_trace(go.Bar(
        x=evaluators,
        y=scores,
        name='原始得分',
        marker_color=MEDICAL_THEME["primary"][0],
        hovertemplate='评估者: %{x}<br>原始得分: %{y:.3f}<extra></extra>'
    ))

    # 信任权重
    fig.add_trace(go.Bar(
        x=evaluators,
        y=trust_weights,
        name='信任权重',
        marker_color=MEDICAL_THEME["secondary"][0],
        hovertemplate='评估者: %{x}<br>信任权重: %{y:.3f}<extra></extra>'
    ))

    # 加权得分
    fig.add_trace(go.Bar(
        x=evaluators,
        y=weighted_scores,
        name='加权得分',
        marker_color=MEDICAL_THEME["accent"][0],
        hovertemplate='评估者: %{x}<br>加权得分: %{y:.3f}<extra></extra>'
    ))

    # 应用主题
    fig = apply_medical_theme(fig, backend="plotly")

    # 更新布局
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=MEDICAL_THEME["title_font_size"]),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="评估者模型",
        yaxis_title="得分",
        barmode='group',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    return fig


def load_model_trust_data(file_path: str = "/data/wzx/output/Model_Trust_Score.json") -> List[Dict[str, Any]]:
    """
    从Model_Trust_Score.json文件加载模型信任数据

    Args:
        file_path: 数据文件路径

    Returns:
        模型数据列表
    """
    if not os.path.exists(file_path):
        print(f"模型信任数据文件不存在: {file_path}")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"加载模型信任数据失败: {e}")
        return []

    model_data = []
    for model_name, metrics in data.items():
        if model_name == "_meta":
            continue
        if not isinstance(metrics, dict):
            continue

        # 提取参数量
        param_size = None
        if '4b' in model_name.lower():
            param_size = 4
        elif '7b' in model_name.lower():
            param_size = 7
        elif '12b' in model_name.lower():
            param_size = 12
        elif '30b' in model_name.lower():
            param_size = 30
        elif '32b' in model_name.lower():
            param_size = 32

        model_data.append({
            "model_name": model_name,
            "avg_score": metrics.get("average_score", 0.0),
            "count": metrics.get("count", 0),
            "param_size": param_size,
        })

    return model_data


def extract_model_data_from_result(result_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    从结果数据中提取模型相关数据

    Args:
        result_data: 预测结果数据

    Returns:
        模型数据字典
    """
    model_data = {
        "total_confidence": 0.0,
        "prediction_model": "",
        "evaluation_models": [],
        "eval_breakdown": [],
        "trust_a": 1.0,
    }

    if not isinstance(result_data, dict):
        return model_data

    # 提取置信度
    if "total_confidence" in result_data:
        model_data["total_confidence"] = result_data["total_confidence"]

    # 提取预测模型
    if "prediction_model" in result_data:
        model_data["prediction_model"] = result_data["prediction_model"]

    # 提取评估模型
    if "evaluation_models" in result_data:
        model_data["evaluation_models"] = result_data["evaluation_models"]

    # 提取评估分解
    if "eval_breakdown" in result_data:
        model_data["eval_breakdown"] = result_data["eval_breakdown"]

    # 提取信任权重a
    if "trust_a" in result_data:
        model_data["trust_a"] = result_data["trust_a"]

    return model_data