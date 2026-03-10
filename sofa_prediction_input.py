import json
import argparse
from datetime import datetime
import re


def extract_patient_id(input_description, patient_data=None):
    """
    从input_description或patient_data中提取患者编号
    """
    # 优先从patient_data中直接获取
    if patient_data and isinstance(patient_data, dict):
        if 'stay_id' in patient_data:
            return str(patient_data['stay_id'])
        if 'subject_id' in patient_data:
            return str(patient_data['subject_id'])

    # 如果input_description本身就是纯数字字符串，直接作为ID返回
    if input_description and str(input_description).strip().isdigit():
        return str(input_description).strip()

    # 尝试从ICU住院编号中提取
    icu_id_match = re.search(r'ICU住院编号\s*(\d+)', input_description)
    if icu_id_match:
        return icu_id_match.group(1)
    
    # 尝试从患者编号中提取
    patient_id_match = re.search(r'患者编号\s*(\d+)', input_description)
    if patient_id_match:
        return patient_id_match.group(1)
    
    # 尝试从stay_id提取（支持英文格式）
    stay_id_match = re.search(r'stay_id\s*[:=]\s*(\d+)', input_description, re.IGNORECASE)
    if stay_id_match:
        return stay_id_match.group(1)
        
    # 尝试从subject_id提取（支持英文格式）
    subject_id_match = re.search(r'subject_id\s*[:=]\s*(\d+)', input_description, re.IGNORECASE)
    if subject_id_match:
        return subject_id_match.group(1)
    
    # 如果都没有找到，返回默认值
    raise ValueError("无法从 input_description 提取患者编号")


def get_user_input():
    """
    获取用户输入的数据
    """
    print("生命体征预测评估工具 - 输入模式")
    print("=" * 40)
    
    # 在交互模式下，模型名称已在程序启动时输入
    # 只需要获取患者信息
    print("\n请输入患者信息 (input_description):")
    input_description = input().strip()
    if not input_description:
        raise ValueError("必须输入患者信息 input_description")
    
    # 获取干预措施
    print("\n请输入干预措施 (intervention):")
    intervention = input().strip()
    if not intervention:
        raise ValueError("必须输入干预措施 intervention")
    
    # 手动模式下不需要用户输入预测生命体征数据
    # 这些将在后续处理中自动生成或从预测结果中获取
    predicted_vital_signs = {}
    
    return "", input_description, intervention, predicted_vital_signs


def load_sample_data():
    """
    加载示例数据
    """
    input_description = '''ICU住院编号 30000831，对应患者编号 15726459，78岁男性，体重nankg。【生命体征】心率变化：[108.0, 81.5]（单位：次/分）收缩压变化：[86.67, 70.25]（单位：mmHg）舒张压变化：[76.0, 58.75]（单位：mmHg）体温变化：[98.6, 98.6]（单位：℃）呼吸频率变化：[28.17, 25.5]（单位：次/分）血氧饱和度变化：[95.5, 95.5]（单位：%）平均动脉压变化：[79.56, 62.58]（单位：mmHg）动脉血pH值 = [[7.45]（单位：无量纲）动脉氧分压 = [[77.0]（单位：mmHg）动脉二氧化碳分压 = [[38.0]（单位：mmHg）【疾病评分】SOFA总分变化：[0, 0, 0, 0, 5, 7]（单位：分）SIRS评分变化：[0, 0, 0, 0, 4, 2]（单位：分）'''
    
    intervention = "立即服用抗生素，评估8小时内发生感染性休克的风险。"
    predicted_vital_signs = {
        "blood_pressure": [100, 105, 110, 115, 120, 125, 130, 135],
        "heart_rate": [100, 95, 90.0, 85.0, 80.0, 78.0, 76.0, 75.0],
        "respiratory_rate": [24.0, 23.0, 22.0, 21.0, 20.0, 18.0, 18, 18],
        "temperature": [36.5, 36.6, 36.7, 36.8, 36.9, 37.0, 37.0, 37.0]
    }
    model_name = "gemma3:12b"  # 默认模型名称
    
    return model_name, input_description, intervention, predicted_vital_signs


def main():
    """
    主函数，处理命令行参数并决定使用哪种输入模式
    """
    parser = argparse.ArgumentParser(description='生命体征预测评估工具输入模块')
    parser.add_argument('--interactive', action='store_true', 
                        help='启用交互模式手动输入数据')
    parser.add_argument('--sample', action='store_true',
                        help='使用示例数据')
    
    args = parser.parse_args()
    
    if args.interactive:
        return get_user_input()
    elif args.sample:
        return load_sample_data()
    else:
        # 默认使用示例数据
        return load_sample_data()


def cli_main():
    """
    命令行接口主函数
    """
    model_name, input_description, intervention, predicted_vital_signs = main()
    if model_name and input_description and intervention and predicted_vital_signs:
        # 这里可以调用评估模块
        print("数据加载成功，可以传递给评估模块")
        print(f"模型名称: {model_name}")
        print(f"患者信息: {extract_patient_id(input_description)}")
        print(f"干预措施: {intervention}")
        print(f"预测生命体征: {json.dumps(predicted_vital_signs, indent=2, ensure_ascii=False)}")
        return model_name, input_description, intervention, predicted_vital_signs
    return None, None, None, None
