import json
import os

# 尝试导入新的可视化模块
try:
    from viz import create_model_trust_chart, export_chart, save_chart_as_image
    from viz.model_charts import load_model_trust_data
    VIZ_AVAILABLE = True
except ImportError as e:
    print(f"警告: 可视化模块不可用，使用备用方案: {e}")
    VIZ_AVAILABLE = False
    import matplotlib.pyplot as plt
    import numpy as np

def generate_model_trust_chart():
    """生成模型信任分析图表"""
    # 使用局部变量跟踪可视化模块可用性
    viz_available = VIZ_AVAILABLE

    # 读取Model_Trust_Score.json数据
    data_file = '/data/wzx/output/Model_Trust_Score.json'
    if not os.path.exists(data_file):
        print(f"错误: 数据文件不存在: {data_file}")
        return False

    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取数据文件失败: {e}")
        return False

    # 如果新可视化模块可用，使用现代化图表
    if viz_available:
        try:
            # 使用load_model_trust_data函数加载数据
            model_data = load_model_trust_data(data_file)

            if not model_data:
                print("错误: 无法加载模型信任数据")
                return False

            # 创建模型信任图表
            chart = create_model_trust_chart(
                model_data=model_data,
                title="模型信任度分析",
                backend="plotly"  # 使用Plotly生成交互式图表
            )

            # 确保输出目录存在
            output_dir = '/data/wzx/output'
            os.makedirs(output_dir, exist_ok=True)

            # 导出为多种格式
            export_results = {}

            # PNG格式
            png_path = os.path.join(output_dir, 'model_trust_analysis.png')
            export_results['png'] = export_chart(
                chart, png_path,
                backend="plotly", format="png",
                width=1000, height=600, scale=2.0
            )

            # PDF格式
            pdf_path = os.path.join(output_dir, 'model_trust_analysis.pdf')
            export_results['pdf'] = export_chart(
                chart, pdf_path,
                backend="plotly", format="pdf",
                width=1000, height=600, scale=2.0
            )

            # HTML格式 (交互式)
            html_path = os.path.join(output_dir, 'model_trust_analysis.html')
            export_results['html'] = export_chart(
                chart, html_path,
                backend="plotly", format="html"
            )

            # 打印结果
            print("现代化图表已生成并保存到以下位置：")
            for format_name, success in export_results.items():
                path = os.path.join(output_dir, f'model_trust_analysis.{format_name}')
                status = "✓ 成功" if success else "✗ 失败"
                print(f"- {path} ({status})")

            # 检查关键格式（PNG和PDF）是否成功
            critical_formats = ['png', 'pdf']
            critical_success = all(export_results.get(fmt, False) for fmt in critical_formats)

            if critical_success:
                return True
            else:
                print("关键格式（PNG/PDF）导出失败，回退到matplotlib方法...")
                return generate_matplotlib_fallback(data)

        except Exception as e:
            print(f"生成现代化图表失败: {e}")
            print("回退到原始matplotlib方法...")
            return generate_matplotlib_fallback(data)

    # 回退到原始matplotlib方法
    if not viz_available:
        return generate_matplotlib_fallback(data)

    return True

def generate_matplotlib_fallback(data):
    """回退到原始matplotlib方法"""
    import matplotlib.pyplot as plt
    import numpy as np

    # 准备数据并按参数量排序
    models_data = []
    for model_name, metrics in data.items():
        if model_name == "_meta":
            continue
        if not isinstance(metrics, dict):
            continue
        if "average_score" not in metrics or "count" not in metrics:
            continue
        # 提取参数量（假设模型名称中包含参数量信息）
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

        models_data.append((model_name, metrics['average_score'], metrics['count'], param_size))

    # 按参数量从小到大排序，None值视为0
    models_data.sort(key=lambda x: x[3] if x[3] is not None else 0)

    # 提取排序后的数据
    models = [item[0] for item in models_data]
    avg_scores = [item[1] for item in models_data]
    counts = [item[2] for item in models_data]

    # 清理模型名称格式，移除冒号
    cleaned_models = [model.replace(':', '-') for model in models]

    # 创建图表
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 设置柱状图的宽度
    width = 0.35

    # 设置x轴位置
    x = np.arange(len(cleaned_models))

    # 绘制平均分数柱状图（左侧）
    bar1 = ax1.bar(x - width/2, avg_scores, width, label='Average Trust Score', color='#4472C4', alpha=0.8)
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Trust Score', fontsize=12, fontweight='bold')
    ax1.set_ylim(0.65, 0.9)  # 设置合适的y轴范围以突出差异
    ax1.tick_params(axis='y')

    # 创建第二个y轴用于展示计数
    ax2 = ax1.twinx()
    bar2 = ax2.bar(x + width/2, counts, width, label='Sample Count', color='#1E3A8A', alpha=0.8)  # 改为深蓝色
    ax2.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(counts) * 1.2)  # 设置计数的y轴范围
    ax2.tick_params(axis='y')

    # 设置x轴标签
    ax1.set_xticks(x)
    ax1.set_xticklabels(cleaned_models, rotation=0, ha='center', fontsize=10, fontweight='bold')

    # 添加标题
    plt.title('Model Trust Analysis: Scores and Sample Counts', fontsize=14, fontweight='bold', pad=20)

    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    # 在柱状图上方添加数值标签
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if bar in bar1:  # 为平均分数添加标签
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            else:  # 为计数添加标签
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    add_labels(bar1)
    add_labels(bar2)

    # 添加网格线使图表更易读
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # 调整布局
    plt.tight_layout()

    # 保存图表为多种格式
    plt.savefig('/data/wzx/output/model_trust_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('/data/wzx/output/model_trust_analysis.pdf', format='pdf', dpi=300, bbox_inches='tight')

    print("回退方案图表已生成并保存到以下位置：")
    print("- /data/wzx/output/model_trust_analysis.png")
    print("- /data/wzx/output/model_trust_analysis.pdf")

    return True

if __name__ == "__main__":
    success = generate_model_trust_chart()
    if success:
        print("✅ 模型信任分析图表生成完成！")
    else:
        print("❌ 图表生成失败！")
        exit(1)
