"""
核心图表构建器
支持多后端渲染（Plotly、Matplotlib、HTML）
"""

import json
import base64
import io
from typing import Dict, Any, Optional, Union, List, Tuple
from enum import Enum

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from .themes import MEDICAL_THEME, apply_medical_theme, get_sofa_component_color, get_risk_level_color


class ChartBackend(Enum):
    """图表后端类型"""
    PLOTLY = "plotly"
    MATPLOTLIB = "matplotlib"
    HTML = "html"  # 生成HTML/JS代码


class ChartTarget(Enum):
    """图表渲染目标"""
    TKINTER = "tkinter"
    STREAMLIT = "streamlit"
    HTML = "html"
    STATIC = "static"


class ChartBuilder:
    """
    统一图表构建器，支持多后端渲染和多平台适配
    """

    def __init__(self, backend: Union[str, ChartBackend] = "plotly", theme: Optional[Dict] = None):
        """
        初始化图表构建器

        Args:
            backend: 后端类型，"plotly"、"matplotlib"或"html"
            theme: 自定义主题，默认为医疗专业主题
        """
        if isinstance(backend, str):
            backend = ChartBackend(backend.lower())

        self.backend = backend
        self.theme = theme or MEDICAL_THEME

        # 验证后端可用性
        if self.backend == ChartBackend.PLOTLY and not PLOTLY_AVAILABLE:
            raise ImportError("Plotly not installed. Please install with: pip install plotly")
        if self.backend == ChartBackend.MATPLOTLIB and not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib not installed. Please install with: pip install matplotlib")

    def create_sofa_trend_chart(self,
                               hourly_totals: Dict[str, Union[int, float]],
                               baseline_totals: Optional[Dict[str, Union[int, float]]] = None,
                               title: str = "SOFA总分随时间变化",
                               **kwargs) -> Any:
        """
        创建SOFA总分趋势图

        Args:
            hourly_totals: 干预组SOFA总分，{时间点: 总分}
            baseline_totals: 基线组SOFA总分（可选）
            title: 图表标题
            **kwargs: 额外参数传递给后端实现

        Returns:
            图表对象（Plotly Figure或matplotlib Figure）
        """
        if self.backend == ChartBackend.PLOTLY:
            return self._create_plotly_sofa_trend(hourly_totals, baseline_totals, title, **kwargs)
        elif self.backend == ChartBackend.MATPLOTLIB:
            return self._create_mpl_sofa_trend(hourly_totals, baseline_totals, title, **kwargs)
        else:
            raise ValueError(f"Unsupported backend for sofa trend chart: {self.backend}")

    def create_sofa_component_chart(self,
                                   sofa_scores: Dict[str, Union[int, float]],
                                   title: str = "SOFA各组件得分",
                                   **kwargs) -> Any:
        """
        创建SOFA各组件得分柱状图

        Args:
            sofa_scores: SOFA组件得分，如{"sofa_respiration": 2, "sofa_coagulation": 1, ...}
            title: 图表标题
            **kwargs: 额外参数

        Returns:
            图表对象
        """
        if self.backend == ChartBackend.PLOTLY:
            return self._create_plotly_sofa_components(sofa_scores, title, **kwargs)
        elif self.backend == ChartBackend.MATPLOTLIB:
            return self._create_mpl_sofa_components(sofa_scores, title, **kwargs)
        else:
            raise ValueError(f"Unsupported backend for sofa component chart: {self.backend}")

    def create_model_trust_chart(self,
                                model_data: List[Dict[str, Any]],
                                title: str = "模型信任度分析",
                                **kwargs) -> Any:
        """
        创建模型信任度分析图

        Args:
            model_data: 模型数据列表，每个字典包含model_name, avg_score, count等字段
            title: 图表标题
            **kwargs: 额外参数

        Returns:
            图表对象
        """
        if self.backend == ChartBackend.PLOTLY:
            return self._create_plotly_model_trust(model_data, title, **kwargs)
        elif self.backend == ChartBackend.MATPLOTLIB:
            return self._create_mpl_model_trust(model_data, title, **kwargs)
        else:
            raise ValueError(f"Unsupported backend for model trust chart: {self.backend}")

    def create_confidence_chart(self,
                              confidence: float,
                              title: str = "模型置信度",
                              **kwargs) -> Any:
        """
        创建模型置信度图表

        Args:
            confidence: 置信度值（0-1）
            title: 图表标题
            **kwargs: 额外参数

        Returns:
            图表对象
        """
        if self.backend == ChartBackend.PLOTLY:
            return self._create_plotly_confidence(confidence, title, **kwargs)
        elif self.backend == ChartBackend.MATPLOTLIB:
            return self._create_mpl_confidence(confidence, title, **kwargs)
        else:
            raise ValueError(f"Unsupported backend for confidence chart: {self.backend}")

    def create_vital_signs_chart(self,
                                vital_data: Dict[str, List[Union[int, float]]],
                                time_points: Optional[List[str]] = None,
                                title: str = "生命体征趋势",
                                **kwargs) -> Any:
        """
        创建生命体征趋势图

        Args:
            vital_data: 生命体征数据，{指标名称: [数值列表]}
            time_points: 时间点标签（可选）
            title: 图表标题
            **kwargs: 额外参数

        Returns:
            图表对象
        """
        if self.backend == ChartBackend.PLOTLY:
            return self._create_plotly_vital_signs(vital_data, time_points, title, **kwargs)
        elif self.backend == ChartBackend.MATPLOTLIB:
            return self._create_mpl_vital_signs(vital_data, time_points, title, **kwargs)
        else:
            raise ValueError(f"Unsupported backend for vital signs chart: {self.backend}")

    def render(self, chart, target: Union[str, ChartTarget] = "auto", **kwargs) -> Any:
        """
        渲染图表到目标环境

        Args:
            chart: 图表对象
            target: 目标环境，"tkinter"、"streamlit"、"html"、"static"或"auto"
            **kwargs: 渲染参数

        Returns:
            渲染后的对象（图像数据、HTML代码等）
        """
        if isinstance(target, str):
            if target == "auto":
                # 自动检测
                try:
                    import streamlit as st
                    target = ChartTarget.STREAMLIT
                except ImportError:
                    try:
                        import tkinter as tk
                        target = ChartTarget.TKINTER
                    except ImportError:
                        target = ChartTarget.STATIC
            else:
                target = ChartTarget(target.lower())

        if target == ChartTarget.TKINTER:
            return self._render_to_tkinter(chart, **kwargs)
        elif target == ChartTarget.STREAMLIT:
            return self._render_to_streamlit(chart, **kwargs)
        elif target == ChartTarget.HTML:
            return self._render_to_html(chart, **kwargs)
        elif target == ChartTarget.STATIC:
            return self._render_to_static(chart, **kwargs)
        else:
            raise ValueError(f"Unsupported target: {target}")

    # Plotly实现
    def _create_plotly_sofa_trend(self, hourly_totals, baseline_totals, title, **kwargs):
        """Plotly实现：SOFA总分趋势图"""
        # 准备数据
        hours = list(hourly_totals.keys())
        totals = list(hourly_totals.values())

        # 创建图表
        fig = go.Figure()

        # 干预组线
        fig.add_trace(go.Scatter(
            x=hours,
            y=totals,
            mode='lines+markers',
            name='干预组',
            line=dict(color=self.theme["primary"][0], width=3),
            marker=dict(size=8),
            hovertemplate='时间: %{x}<br>SOFA总分: %{y}<extra></extra>'
        ))

        # 基线组线（如果存在）
        if baseline_totals:
            baseline_hours = list(baseline_totals.keys())
            baseline_values = list(baseline_totals.values())
            fig.add_trace(go.Scatter(
                x=baseline_hours,
                y=baseline_values,
                mode='lines',
                name='基线组',
                line=dict(color=self.theme["neutral"][0], width=2, dash='dash'),
                hovertemplate='时间: %{x}<br>基线SOFA: %{y}<extra></extra>'
            ))

        # 应用主题
        fig = apply_medical_theme(fig, backend="plotly")

        # 更新布局
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=self.theme["title_font_size"]),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="时间点",
            yaxis_title="SOFA总分",
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        return fig

    def _create_plotly_sofa_components(self, sofa_scores, title, **kwargs):
        """Plotly实现：SOFA组件柱状图"""
        # 提取组件数据
        components = []
        scores = []
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
                components.append(system_names.get(key, system_key))
                scores.append(float(value))
                colors.append(get_sofa_component_color(system_key))

        # 创建图表
        fig = go.Figure(data=[
            go.Bar(
                x=components,
                y=scores,
                marker_color=colors,
                hovertemplate='系统: %{x}<br>得分: %{y}<extra></extra>',
                text=scores,
                textposition='auto',
            )
        ])

        # 应用主题
        fig = apply_medical_theme(fig, backend="plotly")

        # 更新布局
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=self.theme["title_font_size"]),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="系统",
            yaxis_title="SOFA得分",
            yaxis=dict(range=[0, max(scores) * 1.2 if scores else 4]),
            showlegend=False
        )

        return fig

    def _create_plotly_model_trust(self, model_data, title, **kwargs):
        """Plotly实现：模型信任度分析图"""
        # 提取数据
        model_names = [item.get("model_name", f"Model {i}") for i, item in enumerate(model_data)]
        avg_scores = [item.get("avg_score", 0.0) for item in model_data]
        counts = [item.get("count", 0) for item in model_data]

        # 创建双Y轴图表
        fig = go.Figure()

        # 平均分数柱状图
        fig.add_trace(go.Bar(
            x=model_names,
            y=avg_scores,
            name='平均信任分数',
            marker_color=self.theme["primary"][0],
            yaxis='y',
            hovertemplate='模型: %{x}<br>平均分数: %{y:.3f}<extra></extra>'
        ))

        # 样本数量折线图
        fig.add_trace(go.Scatter(
            x=model_names,
            y=counts,
            name='样本数量',
            marker_color=self.theme["accent"][0],
            yaxis='y2',
            mode='lines+markers',
            line=dict(width=2),
            hovertemplate='模型: %{x}<br>样本数量: %{y}<extra></extra>'
        ))

        # 应用主题
        fig = apply_medical_theme(fig, backend="plotly")

        # 更新布局
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=self.theme["title_font_size"]),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="模型",
            yaxis=dict(
                title="平均信任分数",
                titlefont=dict(color=self.theme["primary"][0]),
                tickfont=dict(color=self.theme["primary"][0]),
                range=[0.6, 1.0]  # 信任分数通常在此范围
            ),
            yaxis2=dict(
                title="样本数量",
                titlefont=dict(color=self.theme["accent"][0]),
                tickfont=dict(color=self.theme["accent"][0]),
                overlaying='y',
                side='right',
                range=[0, max(counts) * 1.2 if counts else 10]
            ),
            legend=dict(
                x=0.01,
                y=0.99,
                yanchor="top"
            )
        )

        return fig

    def _create_plotly_confidence(self, confidence, title, **kwargs):
        """Plotly实现：模型置信度图表"""
        # 创建环形图
        fig = go.Figure(data=[
            go.Pie(
                values=[confidence * 100, (1 - confidence) * 100],
                labels=['置信度', '不确定性'],
                hole=0.5,
                marker=dict(colors=[self.theme["secondary"][0], self.theme["neutral"][0]]),
                hovertemplate='%{label}: %{value:.1f}%<extra></extra>',
                textinfo='none'
            )
        ])

        # 应用主题
        fig = apply_medical_theme(fig, backend="plotly")

        # 更新布局
        fig.update_layout(
            title=dict(
                text=f"{title}: {confidence:.1%}",
                font=dict(size=self.theme["title_font_size"]),
                x=0.5,
                xanchor='center'
            ),
            showlegend=True,
            annotations=[
                dict(
                    text=f'{confidence:.1%}',
                    x=0.5, y=0.5,
                    font=dict(size=24, color=self.theme["text_color"]),
                    showarrow=False
                )
            ]
        )

        return fig

    def _create_plotly_vital_signs(self, vital_data, time_points, title, **kwargs):
        """Plotly实现：生命体征趋势图"""
        # 准备数据
        if time_points is None:
            max_len = max(len(vals) for vals in vital_data.values()) if vital_data else 0
            time_points = [f"T+{i}" for i in range(max_len)]

        # 创建图表
        fig = go.Figure()

        # 为每个生命体征添加线
        colors = self.theme["vital_signs"]
        for i, (sign_name, values) in enumerate(vital_data.items()):
            color = colors.get(sign_name, self.theme["primary"][i % len(self.theme["primary"])])

            fig.add_trace(go.Scatter(
                x=time_points[:len(values)],
                y=values,
                mode='lines+markers',
                name=sign_name,
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate='时间: %{x}<br>%{text}: %{y}<extra></extra>',
                text=[sign_name] * len(values)
            ))

        # 应用主题
        fig = apply_medical_theme(fig, backend="plotly")

        # 更新布局
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=self.theme["title_font_size"]),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="时间",
            yaxis_title="数值",
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )

        return fig

    # Matplotlib实现（兼容模式）
    def _create_mpl_sofa_trend(self, hourly_totals, baseline_totals, title, **kwargs):
        """Matplotlib实现：SOFA总分趋势图"""
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))

        # 准备数据
        hours = list(hourly_totals.keys())
        totals = list(hourly_totals.values())

        # 干预组线
        ax.plot(hours, totals, 'o-', label='干预组',
                color=self.theme["primary"][0], linewidth=3, markersize=8)

        # 基线组线（如果存在）
        if baseline_totals:
            baseline_hours = list(baseline_totals.keys())
            baseline_values = list(baseline_totals.values())
            ax.plot(baseline_hours, baseline_values, '--', label='基线组',
                    color=self.theme["neutral"][0], linewidth=2)

        # 应用主题
        fig = apply_medical_theme(fig, backend="matplotlib")

        # 设置标题和标签
        ax.set_title(title, fontsize=self.theme["title_font_size"])
        ax.set_xlabel("时间点")
        ax.set_ylabel("SOFA总分")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def _create_mpl_sofa_components(self, sofa_scores, title, **kwargs):
        """Matplotlib实现：SOFA组件柱状图"""
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))

        # 提取组件数据
        components = []
        scores = []
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
                components.append(system_names.get(key, system_key))
                scores.append(float(value))
                colors.append(get_sofa_component_color(system_key))

        # 创建柱状图
        bars = ax.bar(components, scores, color=colors)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=10)

        # 应用主题
        fig = apply_medical_theme(fig, backend="matplotlib")

        # 设置标题和标签
        ax.set_title(title, fontsize=self.theme["title_font_size"])
        ax.set_xlabel("系统")
        ax.set_ylabel("SOFA得分")
        ax.set_ylim(0, max(scores) * 1.2 if scores else 4)
        plt.xticks(rotation=45, ha='right')

        return fig

    def _create_mpl_model_trust(self, model_data, title, **kwargs):
        """Matplotlib实现：模型信任度分析图"""
        fig, ax1 = plt.subplots(figsize=kwargs.get('figsize', (12, 8)))

        # 提取数据
        model_names = [item.get("model_name", f"Model {i}") for i, item in enumerate(model_data)]
        avg_scores = [item.get("avg_score", 0.0) for item in model_data]
        counts = [item.get("count", 0) for item in model_data]

        # 清理模型名称格式，移除冒号
        cleaned_models = [name.replace(':', '-') for name in model_names]

        # 设置柱状图的宽度
        width = 0.35
        x = range(len(cleaned_models))

        # 绘制平均分数柱状图（左侧）
        bars1 = ax1.bar([i - width/2 for i in x], avg_scores, width,
                       label='平均信任分数', color=self.theme["primary"][0], alpha=0.8)
        ax1.set_xlabel('模型', fontsize=self.theme["axis_font_size"])
        ax1.set_ylabel('平均信任分数', fontsize=self.theme["axis_font_size"], color=self.theme["primary"][0])
        ax1.set_ylim(0.6, 1.0)  # 信任分数范围
        ax1.tick_params(axis='y', labelcolor=self.theme["primary"][0])

        # 创建第二个y轴用于展示计数
        ax2 = ax1.twinx()
        bars2 = ax2.bar([i + width/2 for i in x], counts, width,
                       label='样本数量', color=self.theme["accent"][0], alpha=0.8)
        ax2.set_ylabel('样本数量', fontsize=self.theme["axis_font_size"], color=self.theme["accent"][0])
        ax2.set_ylim(0, max(counts) * 1.2 if counts else 10)
        ax2.tick_params(axis='y', labelcolor=self.theme["accent"][0])

        # 设置x轴标签
        ax1.set_xticks(x)
        ax1.set_xticklabels(cleaned_models, rotation=0, ha='center',
                           fontsize=self.theme["axis_font_size"] - 2)

        # 添加标题
        ax1.set_title(title, fontsize=self.theme["title_font_size"], pad=20)

        # 添加图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
                  fontsize=self.theme["legend_font_size"])

        # 在柱状图上方添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom',
                    fontsize=self.theme["legend_font_size"] - 1, fontweight='bold')

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom',
                    fontsize=self.theme["legend_font_size"] - 1, fontweight='bold')

        # 应用主题
        fig = apply_medical_theme(fig, backend="matplotlib")

        # 添加网格线
        ax1.grid(axis='y', linestyle='--', alpha=0.3)

        return fig

    def _create_mpl_confidence(self, confidence, title, **kwargs):
        """Matplotlib实现：模型置信度图表"""
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 8)))

        # 创建饼图
        labels = ['置信度', '不确定性']
        sizes = [confidence * 100, (1 - confidence) * 100]
        colors = [self.theme["secondary"][0], self.theme["neutral"][0]]
        explode = (0.1, 0)  # 突出显示置信度部分

        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                         autopct='%1.1f%%', startangle=90,
                                         textprops=dict(fontsize=self.theme["legend_font_size"]))

        # 设置中心文本
        center_text = f'{confidence:.1%}'
        ax.text(0, 0, center_text, ha='center', va='center',
               fontsize=24, fontweight='bold', color=self.theme["text_color"])

        # 设置标题
        ax.set_title(f"{title}: {confidence:.1%}", fontsize=self.theme["title_font_size"], pad=20)

        # 应用主题
        fig = apply_medical_theme(fig, backend="matplotlib")

        return fig

    def _create_mpl_vital_signs(self, vital_data, time_points, title, **kwargs):
        """Matplotlib实现：生命体征趋势图"""
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (12, 8)))

        # 准备数据
        if time_points is None:
            max_len = max(len(vals) for vals in vital_data.values()) if vital_data else 0
            time_points = [f"T+{i}" for i in range(max_len)]

        # 为每个生命体征添加线
        colors = self.theme["vital_signs"]
        for i, (sign_name, values) in enumerate(vital_data.items()):
            color = colors.get(sign_name, self.theme["primary"][i % len(self.theme["primary"])])

            x_vals = time_points[:len(values)]
            ax.plot(x_vals, values, 'o-', label=sign_name,
                   color=color, linewidth=2, markersize=6)

        # 应用主题
        fig = apply_medical_theme(fig, backend="matplotlib")

        # 设置标题和标签
        ax.set_title(title, fontsize=self.theme["title_font_size"])
        ax.set_xlabel("时间", fontsize=self.theme["axis_font_size"])
        ax.set_ylabel("数值", fontsize=self.theme["axis_font_size"])
        ax.legend(fontsize=self.theme["legend_font_size"])
        ax.grid(True, alpha=0.3)

        return fig

    # 渲染适配器
    def _render_to_tkinter(self, chart, **kwargs):
        """渲染到Tkinter Canvas"""
        if self.backend == ChartBackend.PLOTLY:
            # Plotly图表转换为静态图像
            import plotly.io as pio
            img_bytes = pio.to_image(chart, format='png', **kwargs)
            return img_bytes
        elif self.backend == ChartBackend.MATPLOTLIB:
            # Matplotlib图表转换为图像
            import io
            buf = io.BytesIO()
            chart.savefig(buf, format='png', dpi=kwargs.get('dpi', 100))
            buf.seek(0)
            return buf.read()
        else:
            raise ValueError(f"Cannot render {self.backend} to Tkinter")

    def _render_to_streamlit(self, chart, **kwargs):
        """渲染到Streamlit"""
        if self.backend == ChartBackend.PLOTLY:
            # Streamlit原生支持Plotly
            import streamlit as st
            return st.plotly_chart(chart, use_container_width=True, **kwargs)
        elif self.backend == ChartBackend.MATPLOTLIB:
            # Streamlit也支持matplotlib
            import streamlit as st
            return st.pyplot(chart, **kwargs)
        else:
            raise ValueError(f"Cannot render {self.backend} to Streamlit")

    def _render_to_html(self, chart, **kwargs):
        """渲染为HTML代码"""
        if self.backend == ChartBackend.PLOTLY:
            # 生成Plotly HTML
            import plotly.io as pio
            return pio.to_html(chart, full_html=False, **kwargs)
        else:
            raise ValueError(f"Cannot render {self.backend} to HTML")

    def _render_to_static(self, chart, **kwargs):
        """渲染为静态图像数据"""
        return self._render_to_tkinter(chart, **kwargs)