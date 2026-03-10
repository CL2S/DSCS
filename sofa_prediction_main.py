#!/usr/bin/env python3
import sys
import os
import argparse

# 添加项目路径到sys.path，确保能导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sofa_prediction_input import main as input_main
from sofa_prediction_evaluator import (
    calculate_vital_signs_confidence,
    evaluate_with_ollama,
    save_evaluation_report,
    main as evaluator_main
)


def main():
    """
    主函数，支持两种模式：交互式输入和从JSON文件读取
    """
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='SOFA预测评估工具')
    parser.add_argument('--mode', choices=['interactive', 'batch'], default='interactive',
                        help='运行模式：interactive（交互式）或 batch（批量处理JSON文件）')
    
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        # 交互式模式
        # 获取输入数据
        model_name, input_description, intervention, predicted_vital_signs = input_main()
        
        # 检查数据是否有效
        if not all([model_name, input_description, intervention, predicted_vital_signs]):
            print("输入数据不完整，无法进行评估")
            return
        
        # 计算置信度评分
        confidence = calculate_vital_signs_confidence(predicted_vital_signs)
        
        # 使用Ollama大模型进行评估
        ollama_result = evaluate_with_ollama(model_name, input_description, intervention, predicted_vital_signs)
        
        # 输出简化结果
        print(f"预测置信度评分: {confidence:.4f}")
        
        if ollama_result["success"]:
            print(f"模型名称: {model_name}")
            print(f"模型输出:\n{ollama_result['model_output']}")
        else:
            print(f"评估失败: {ollama_result['error']}")
        
        # 保存评估报告
        save_evaluation_report(model_name, input_description, intervention, predicted_vital_signs, confidence, ollama_result)
    else:
        # 批量处理模式
        evaluator_main()


if __name__ == "__main__":
    main()