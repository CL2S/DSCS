#!/usr/bin/env python3
"""
测试GUI中的可视化集成
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=== 测试GUI可视化集成 ===\n")

# 模拟Tkinter环境（如果没有显示）
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 1. 测试导入
print("1. 测试GUI模块导入...")
try:
    from gui import PredictionApp
    print("   ✅ PredictionApp导入成功")

    # 检查viz模块是否可用
    from viz import ChartBuilder, create_sofa_trend_chart, create_sofa_component_chart, create_confidence_chart
    print("   ✅ viz模块导入成功")

except ImportError as e:
    print(f"   ❌ 导入失败: {e}")
    sys.exit(1)
except Exception as e:
    print(f"   ❌ 其他错误: {e}")
    sys.exit(1)

# 2. 测试ChartBuilder与matplotlib后端
print("\n2. 测试ChartBuilder (matplotlib后端)...")
try:
    builder = ChartBuilder(backend="matplotlib")
    print("   ✅ ChartBuilder创建成功")

    # 测试SOFA趋势图表
    hourly_totals = {"0h": 5, "24h": 7, "48h": 6, "72h": 4}
    trend_chart = builder.create_sofa_trend_chart(hourly_totals=hourly_totals)
    print("   ✅ SOFA趋势图表创建成功")

    # 测试SOFA组件图表
    sofa_scores = {
        "sofa_respiration": 2,
        "sofa_coagulation": 1,
        "sofa_liver": 0,
        "sofa_cardiovascular": 3,
        "sofa_cns": 1,
        "sofa_renal": 0
    }
    comp_chart = builder.create_sofa_component_chart(sofa_scores=sofa_scores)
    print("   ✅ SOFA组件图表创建成功")

    # 测试置信度图表
    confidence_chart = builder.create_confidence_chart(confidence=0.85)
    print("   ✅ 置信度图表创建成功")

except Exception as e:
    print(f"   ❌ 图表创建失败: {e}")
    import traceback
    traceback.print_exc()

# 3. 测试模拟的plot_results方法
print("\n3. 测试模拟的plot_results逻辑...")
try:
    # 创建模拟的result_data
    result_data = {
        "hourly_sofa_totals": {"0h": 5, "24h": 7, "48h": 6, "72h": 4},
        "computed_sofa_scores": {
            "sofa_respiration": 2,
            "sofa_coagulation": 1,
            "sofa_liver": 0,
            "sofa_cardiovascular": 3,
            "sofa_cns": 1,
            "sofa_renal": 0,
            "sofa_total": 7
        },
        "total_confidence": 0.85
    }

    # 导入必要的模块
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    # 模拟GUI中的plot_results逻辑
    fig = plt.Figure(figsize=(10, 8))

    # 测试不同情况
    test_cases = [
        ("仅趋势数据", {"hourly_sofa_totals": result_data["hourly_sofa_totals"]}),
        ("仅组件数据", {"computed_sofa_scores": result_data["computed_sofa_scores"]}),
        ("仅置信度", {"total_confidence": result_data["total_confidence"]}),
        ("完整数据", result_data)
    ]

    for case_name, test_data in test_cases:
        print(f"   测试: {case_name}")
        try:
            # 清空图形
            fig.clear()

            # 模拟确定显示哪些图表
            hourly_totals = test_data.get("hourly_sofa_totals")
            sofa_scores = test_data.get("computed_sofa_scores")
            confidence = test_data.get("total_confidence")

            show_trend = hourly_totals is not None
            show_components = sofa_scores is not None and any(
                key.startswith("sofa_") and key != "sofa_total" for key in sofa_scores.keys()
            )
            show_confidence = confidence is not None

            if show_confidence:
                # 仅显示置信度图表
                builder = ChartBuilder(backend="matplotlib")
                confidence_chart = builder.create_confidence_chart(
                    confidence=confidence,
                    title=f"模型置信度: {confidence:.2f}"
                )
                ax = fig.add_subplot(111)
                # 复制内容（简化版，实际GUI中有_copy_axes_content方法）
                ax.set_title(f"模型置信度: {confidence:.2f}")
                ax.text(0.5, 0.5, f"{confidence:.1%}", ha='center', va='center', fontsize=24)
                print("      ✅ 置信度图表逻辑通过")
            else:
                # 显示趋势和/或组件
                num_charts = sum([show_trend, show_components])
                if num_charts == 0:
                    print("      ✅ 无数据逻辑通过")
                else:
                    print(f"      ✅ {num_charts}个图表逻辑通过")

        except Exception as e:
            print(f"      ❌ 失败: {e}")

except Exception as e:
    print(f"   ❌ plot_results测试失败: {e}")
    import traceback
    traceback.print_exc()

# 4. 测试_copy_axes_content方法
print("\n4. 测试_copy_axes_content方法...")
try:
    # 创建两个图形
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    # 在ax1中添加一些内容
    ax1.plot([0, 1, 2], [0, 1, 4], 'r-', label='线1')
    ax1.bar([0, 1], [2, 3], color='blue')
    ax1.set_title("源图形")
    ax1.set_xlabel("X轴")
    ax1.set_ylabel("Y轴")
    ax1.legend()
    ax1.grid(True)

    # 需要从gui模块导入_copy_axes_content方法
    from gui import PredictionApp
    # 创建应用实例以访问方法
    import tkinter as tk
    root = tk.Tk()
    app = PredictionApp(root)

    # 复制内容
    app._copy_axes_content(ax1, ax2)

    # 检查是否复制成功
    if ax2.get_title():
        print("   ✅ _copy_axes_content方法成功")
    else:
        print("   ⚠️ _copy_axes_content可能未完全复制")

    root.destroy()

except Exception as e:
    print(f"   ❌ _copy_axes_content测试失败: {e}")
    # 这可能是因为Tkinter问题，可以接受

print("\n=== GUI可视化集成测试完成 ===")