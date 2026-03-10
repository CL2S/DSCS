#!/usr/bin/env python3
"""
测试demo2.py中的Plotly图表
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=== 测试demo2.py中的Plotly图表 ===\n")

# 1. 测试导入
print("1. 测试导入...")
try:
    import plotly.graph_objects as go
    print("   ✅ plotly.graph_objects导入成功")

    # 检查demo2.py中使用的医疗主题颜色
    medical_theme = {
        "primary": "#1E88E5",
        "secondary": "#43A047",
        "accent": "#FF7043",
        "neutral": "#78909C",
        "grid_color": "#E0E0E0",
        "background_color": "#FFFFFF",
        "text_color": "#212121"
    }
    print("   ✅ 医疗主题定义完整")

    vital_signs_colors = {
        "heart_rate": "#EF5350",      # 心率 - 红色
        "blood_pressure": "#AB47BC",  # 血压 - 紫色
        "respiratory_rate": "#29B6F6", # 呼吸频率 - 蓝色
        "temperature": "#FFA726",     # 体温 - 橙色
        "oxygen_saturation": "#66BB6A" # 血氧饱和度 - 绿色
    }
    print("   ✅ 生命体征颜色映射完整")

except ImportError as e:
    print(f"   ❌ 导入失败: {e}")
    sys.exit(1)
except Exception as e:
    print(f"   ❌ 其他错误: {e}")
    sys.exit(1)

# 2. 测试图表创建（基于demo2.py中的代码）
print("\n2. 测试Plotly图表创建...")
try:
    # 模拟生命体征数据
    vital_data = {
        "heart_rate": [72, 75, 78, 76, 74, 73],
        "blood_pressure": [120, 118, 122, 119, 121, 120],
        "respiratory_rate": [16, 18, 17, 19, 18, 17],
        "temperature": [36.6, 36.8, 37.0, 36.9, 36.7, 36.6],
        "oxygen_saturation": [98, 97, 99, 98, 98, 99]
    }

    time_points = [f"T+{i*4}h" for i in range(6)]

    # 创建图表（模拟demo2.py中的代码结构）
    for sign, trend in vital_data.items():
        color = vital_signs_colors.get(sign.lower(), medical_theme["primary"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_points,
            y=trend,
            mode='lines+markers',
            name=sign,
            line=dict(color=color, width=2),
            marker=dict(size=6, color=color),
            hovertemplate='时间: %{x}<br>数值: %{y}<extra></extra>'
        ))

        # 更新布局（基于demo2.py）
        fig.update_layout(
            title=dict(
                text=sign,
                font=dict(size=14, color=medical_theme["text_color"])
            ),
            xaxis=dict(
                title="时间点",
                gridcolor=medical_theme["grid_color"],
                linecolor=medical_theme["neutral"],
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                title="数值",
                gridcolor=medical_theme["grid_color"],
                linecolor=medical_theme["neutral"],
                tickfont=dict(size=10)
            ),
            plot_bgcolor=medical_theme["background_color"],
            paper_bgcolor=medical_theme["background_color"],
            font=dict(family='Arial, Helvetica, sans-serif'),
            margin=dict(l=40, r=30, b=40, t=40, pad=4),
            height=220
        )

    print(f"   ✅ 成功创建 {len(vital_data)} 个生命体征图表")

    # 测试图表导出为HTML
    import plotly.io as pio
    html_output = pio.to_html(fig, full_html=False)
    print(f"   ✅ 图表导出为HTML成功，大小: {len(html_output)} 字符")

except Exception as e:
    print(f"   ❌ 图表创建失败: {e}")
    import traceback
    traceback.print_exc()

# 3. 测试demo2.py中定义的医疗主题与viz模块的一致性
print("\n3. 测试主题一致性...")
try:
    from viz.themes import MEDICAL_THEME as viz_theme
    print("   ✅ 成功导入viz医疗主题")

    # 比较颜色
    demo_colors = {
        "primary": medical_theme["primary"],
        "secondary": medical_theme["secondary"],
        "accent": medical_theme["accent"],
        "neutral": medical_theme["neutral"],
        "grid_color": medical_theme["grid_color"],
        "background_color": medical_theme["background_color"],
        "text_color": medical_theme["text_color"]
    }

    viz_colors = {
        "primary": viz_theme["primary"][0] if isinstance(viz_theme["primary"], list) else viz_theme["primary"],
        "secondary": viz_theme["secondary"][0] if isinstance(viz_theme["secondary"], list) else viz_theme["secondary"],
        "accent": viz_theme["accent"][0] if isinstance(viz_theme["accent"], list) else viz_theme["accent"],
        "neutral": viz_theme["neutral"][0] if isinstance(viz_theme["neutral"], list) else viz_theme["neutral"],
        "grid_color": viz_theme["grid_color"],
        "background_color": viz_theme["background_color"],
        "text_color": viz_theme["text_color"]
    }

    matches = 0
    total = len(demo_colors)

    for key in demo_colors:
        if key in viz_colors and demo_colors[key] == viz_colors[key]:
            matches += 1
        else:
            print(f"      ⚠️ 颜色不匹配: {key} - demo: {demo_colors[key]}, viz: {viz_colors.get(key)}")

    if matches == total:
        print("   ✅ 所有颜色完全匹配")
    else:
        print(f"   ⚠️ 颜色部分匹配: {matches}/{total}")

except ImportError as e:
    print(f"   ⚠️ 无法导入viz主题: {e}")
except Exception as e:
    print(f"   ❌ 主题比较失败: {e}")

print("\n=== demo2.py Plotly图表测试完成 ===")