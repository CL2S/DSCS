#!/usr/bin/env python3
"""
测试新的可视化模块
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=== 测试可视化模块 ===\n")

# 1. 测试导入
print("1. 测试模块导入...")
try:
    from viz import (
        MEDICAL_THEME, apply_medical_theme,
        ChartBuilder, ChartBackend,
        create_sofa_trend_chart, create_sofa_component_chart,
        create_model_trust_chart, create_confidence_chart,
        export_chart, save_chart_as_image, save_chart_as_html
    )
    print("   ✅ 成功导入viz模块")

    # 检查ChartBuilder
    builder = ChartBuilder(backend="plotly")
    print("   ✅ ChartBuilder创建成功")

except ImportError as e:
    print(f"   ❌ 导入失败: {e}")
    sys.exit(1)
except Exception as e:
    print(f"   ❌ 其他错误: {e}")
    sys.exit(1)

# 2. 测试主题
print("\n2. 测试医疗主题...")
try:
    print(f"   主题包含: {len(MEDICAL_THEME.keys())}个键")
    print(f"   主要颜色: {MEDICAL_THEME['primary'][:2]}")
    print(f"   SOFA组件颜色: {len(MEDICAL_THEME['sofa_components'])}个")
    print("   ✅ 主题数据完整")
except Exception as e:
    print(f"   ❌ 主题错误: {e}")

# 3. 测试模型信任数据加载
print("\n3. 测试模型信任数据加载...")
try:
    from viz.model_charts import load_model_trust_data
    model_data = load_model_trust_data("/data/wzx/output/Model_Trust_Score.json")
    print(f"   加载了 {len(model_data)} 个模型的数据")
    if model_data:
        print(f"   第一个模型: {model_data[0]['model_name']}, 分数: {model_data[0]['avg_score']:.3f}")
    print("   ✅ 数据加载成功")
except Exception as e:
    print(f"   ❌ 数据加载失败: {e}")

# 4. 测试图表创建
print("\n4. 测试图表创建...")
try:
    # 测试模型信任图表
    if model_data:
        chart = create_model_trust_chart(
            model_data=model_data,
            title="测试图表",
            backend="plotly"
        )
        print("   ✅ 模型信任图表创建成功")

    # 测试SOFA趋势图表
    hourly_totals = {"0h": 5, "24h": 7, "48h": 6, "72h": 4}
    trend_chart = builder.create_sofa_trend_chart(
        hourly_totals=hourly_totals,
        title="测试SOFA趋势"
    )
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
    comp_chart = builder.create_sofa_component_chart(
        sofa_scores=sofa_scores,
        title="测试SOFA组件"
    )
    print("   ✅ SOFA组件图表创建成功")

    # 测试置信度图表
    confidence_chart = builder.create_confidence_chart(
        confidence=0.85,
        title="测试置信度"
    )
    print("   ✅ 置信度图表创建成功")

    # 测试生命体征图表
    vital_data = {
        "heart_rate": [72, 75, 78, 76, 74],
        "blood_pressure": [120, 118, 122, 119, 121]
    }
    vital_chart = builder.create_vital_signs_chart(
        vital_data=vital_data,
        title="测试生命体征"
    )
    print("   ✅ 生命体征图表创建成功")

except Exception as e:
    print(f"   ❌ 图表创建失败: {e}")
    import traceback
    traceback.print_exc()

# 5. 测试导出功能
print("\n5. 测试导出功能...")
try:
    # 创建输出目录
    test_output_dir = "/data/wzx/test_output"
    os.makedirs(test_output_dir, exist_ok=True)

    if model_data:
        chart = create_model_trust_chart(
            model_data=model_data[:3],  # 只测试前3个模型
            title="测试导出",
            backend="plotly"
        )

        # 测试PNG导出
        png_path = os.path.join(test_output_dir, "test_chart.png")
        success = export_chart(chart, png_path, backend="plotly", format="png")
        print(f"   PNG导出: {'✅ 成功' if success else '❌ 失败'}")

        # 测试HTML导出
        html_path = os.path.join(test_output_dir, "test_chart.html")
        success = save_chart_as_html(chart, html_path, backend="plotly")
        print(f"   HTML导出: {'✅ 成功' if success else '❌ 失败'}")

        # 检查文件是否存在
        if os.path.exists(png_path):
            print(f"   PNG文件大小: {os.path.getsize(png_path)} 字节")
        if os.path.exists(html_path):
            print(f"   HTML文件大小: {os.path.getsize(html_path)} 字节")

except Exception as e:
    print(f"   ❌ 导出测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 测试完成 ===")