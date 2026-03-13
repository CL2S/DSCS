#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""调试干预措施参数传递的测试脚本"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiment import AdaptiveExperimentAgent


def test_intervention_parameter():
    """测试intervention_and_risk参数的传递"""
    print("开始测试intervention_and_risk参数传递...")
    
    # 创建测试数据
    patient_sepsis_summary = """ICU住院编号 30000831，对应患者编号 15726459，78岁男性，体重nankg。【生命体征】心率变化：[108.0, 81.5]（单位：次/分）收缩压变化：[86.67, 70.25]（单位：mmHg）舒张压变化：[76.0, 58.75]（单位：mmHg）体温变化：[98.6, 98.6]（单位：℃）呼吸频率变化：[28.17, 25.5]（单位：次/分）血氧饱和度变化：[95.5, 95.5]（单位：%）平均动脉压变化：[79.56, 62.58]（单位：mmHg）动脉血pH值 = [[7.45]（单位：无量纲）动脉氧分压 = [[77.0]（单位：mmHg）动脉二氧化碳分压 = [[38.0]（单位：mmHg）【疾病评分】SOFA总分变化：[0, 0, 0, 0, 5, 7]（单位：分）SIRS评分变化：[0, 0, 0, 0, 4, 2]（单位：分）"""
    
    intervention_and_risk = "立即服用抗生素，评估8小时内发生感染性休克的风险。"
    actual_vital_signs = {}
    
    print(f"患者摘要: {patient_sepsis_summary[:100]}...")
    print(f"干预措施: {intervention_and_risk}")
    print(f"实际体征: {actual_vital_signs}")
    
    try:
        # 创建代理
        agent = AdaptiveExperimentAgent()
        
        # 运行预测
        prediction = agent(
            patient_sepsis_summary=patient_sepsis_summary,
            intervention_and_risk=intervention_and_risk,
            actual_vital_signs=actual_vital_signs
        )
        
        print("预测成功完成")
        print(f"风险评估: {prediction.risk_assessment.risk_level}")
        print(f"干预分析: {prediction.intervention_analysis.predicted_outcome[:100]}...")
        
    except Exception as e:
        print(f"预测过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    test_intervention_parameter()


if __name__ == "__main__":
    main()