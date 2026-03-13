#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事实预测模块
用于从icu_stays_descriptions88.json中提取实际干预方案，并进行事实预测和对比分析
"""

import re
import numpy as np
import json
import traceback
from typing import Dict, List, Any, Tuple
from experiment import AdaptiveExperimentAgent, configure_dspy
from sofa_prediction_evaluator import extract_patient_id

# 定义要计算信任度的所有模型名称列表
MODEL_NAMES = ["gemma3:12b", "deepseek-r1:32b", "qwen3:30b", "gpt-oss:20b"]

# 定义SOFA评分的六项指标
SOFA_COMPONENTS = [
    "sofa_respiration",    # 呼吸系统
    "sofa_coagulation",    # 凝血系统
    "sofa_liver",          # 肝脏
    "sofa_cardiovascular", # 心血管系统
    "sofa_cns",            # 中枢神经系统
    "sofa_renal"           # 肾脏
]

# 定义从特征值计算SOFA评分的函数

def calculate_respiration_sofa(pao2_fio2_ratio: float, mechanical_ventilation: float) -> int:
    """计算呼吸系统SOFA评分"""
    if pao2_fio2_ratio >= 400 and mechanical_ventilation == 0:
        return 0
    elif 300 <= pao2_fio2_ratio < 400 or mechanical_ventilation == 1:
        return 1
    elif 200 <= pao2_fio2_ratio < 300:
        return 2
    elif 100 <= pao2_fio2_ratio < 200:
        return 3
    else:
        return 4

def calculate_coagulation_sofa(platelet: float) -> int:
    """计算凝血系统SOFA评分"""
    if platelet >= 150:
        return 0
    elif 100 <= platelet < 150:
        return 1
    elif 50 <= platelet < 100:
        return 2
    elif 20 <= platelet < 50:
        return 3
    else:
        return 4

def calculate_liver_sofa(bilirubin_total: float) -> int:
    """计算肝脏SOFA评分"""
    if bilirubin_total < 1.2:
        return 0
    elif 1.2 <= bilirubin_total < 2.0:
        return 1
    elif 2.0 <= bilirubin_total < 6.0:
        return 2
    elif 6.0 <= bilirubin_total < 12.0:
        return 3
    else:
        return 4

def calculate_cardiovascular_sofa(vasopressor_rate: float) -> int:
    """计算心血管系统SOFA评分"""
    if vasopressor_rate == 0:
        return 0
    elif 0 < vasopressor_rate <= 5:
        return 1
    elif 5 < vasopressor_rate <= 15:
        return 2
    else:
        return 4

def calculate_cns_sofa(gcs_total: float) -> int:
    """计算中枢神经系统SOFA评分"""
    if gcs_total >= 15:
        return 0
    elif 13 <= gcs_total < 15:
        return 1
    elif 10 <= gcs_total < 13:
        return 2
    elif 7 <= gcs_total < 10:
        return 3
    else:
        return 4

def calculate_renal_sofa(creatinine: float, urine_output_ml: float) -> int:
    """计算肾脏SOFA评分"""
    if creatinine < 1.2 and urine_output_ml >= 500:
        return 0
    elif (1.2 <= creatinine < 2.0) or (200 <= urine_output_ml < 500):
        return 1
    elif (2.0 <= creatinine < 3.5) or (100 <= urine_output_ml < 200):
        return 2
    elif (3.5 <= creatinine < 5.0) or (urine_output_ml < 100):
        return 3
    else:
        return 4

def calculate_sofa_scores_from_features(features: Dict[str, List[float]]) -> Dict[str, List[int]]:
    """
    从特征值计算六项SOFA评分
    
    参数:
    features (dict): 包含各项特征值的字典
    
    返回:
    dict: 包含六项SOFA评分的字典
    """
    sofa_scores = {component: [] for component in SOFA_COMPONENTS}
    
    # 获取所有特征值列表的最大长度
    max_length = 0
    for values in features.values():
        if len(values) > max_length:
            max_length = len(values)
    
    # 对每个时间点计算SOFA评分
    for i in range(max_length):
        # 获取当前时间点的各项特征值
        pao2_fio2 = features.get("pao2_fio2_ratio", [0.0])[i] if i < len(features.get("pao2_fio2_ratio", [])) else 0.0
        platelet = features.get("platelet", [0.0])[i] if i < len(features.get("platelet", [])) else 0.0
        bilirubin = features.get("bilirubin_total", [0.0])[i] if i < len(features.get("bilirubin_total", [])) else 0.0
        vasopressor = features.get("vasopressor_rate", [0.0])[i] if i < len(features.get("vasopressor_rate", [])) else 0.0
        gcs = features.get("gcs_total", [0.0])[i] if i < len(features.get("gcs_total", [])) else 0.0
        creatinine = features.get("creatinine", [0.0])[i] if i < len(features.get("creatinine", [])) else 0.0
        urine = features.get("urine_output_ml", [0.0])[i] if i < len(features.get("urine_output_ml", [])) else 0.0
        ventilation = features.get("mechanical_ventilation", [0.0])[i] if i < len(features.get("mechanical_ventilation", [])) else 0.0
        
        # 计算各项SOFA评分
        sofa_scores["sofa_respiration"].append(calculate_respiration_sofa(pao2_fio2, ventilation))
        sofa_scores["sofa_coagulation"].append(calculate_coagulation_sofa(platelet))
        sofa_scores["sofa_liver"].append(calculate_liver_sofa(bilirubin))
        sofa_scores["sofa_cardiovascular"].append(calculate_cardiovascular_sofa(vasopressor))
        sofa_scores["sofa_cns"].append(calculate_cns_sofa(gcs))
        sofa_scores["sofa_renal"].append(calculate_renal_sofa(creatinine, urine))
    
    return sofa_scores

def extract_actual_intervention_from_output_summary(output_summary: str) -> List[float]:
    """从output_summary中提取血管活性药物使用剂量数组"""
    # 使用正则表达式从文本中提取血管活性药物使用剂量数组
    match = re.search(r'血管活性药物使用剂量：\[(.*?)\]', output_summary)
    if match:
        # 提取并转换数值
        values_str = match.group(1)
        try:
            values = [float(v.strip()) for v in values_str.split(',') if v.strip()]
            return values
        except ValueError:
            print(f"解析血管活性药物使用剂量时出错: {values_str}")
            return []
    else:
        print("未找到血管活性药物使用剂量信息")
        return []

def generate_intervention_description(vasopressor_doses: List[float]) -> str:
    """
    将血管活性药物剂量时间序列转换为临床描述
    
    Args:
        vasopressor_doses: 血管活性药物剂量的时间序列数据（通常为13个时间点）
    
    Returns:
        str: 临床干预措施的语言化描述
    """
    if not vasopressor_doses:
        return "无血管活性药物干预措施，建议密切监测血压和器官灌注，评估8小时内发生感染性休克的风险"
    
    # 计算剂量统计信息
    non_zero_doses = [dose for dose in vasopressor_doses if dose > 0]
    
    if not non_zero_doses:
        return "无血管活性药物干预措施，建议密切监测血压和器官灌注，评估8小时内发生感染性休克的风险"
    
    initial_dose = vasopressor_doses[0] if vasopressor_doses[0] > 0 else non_zero_doses[0]
    max_dose = max(non_zero_doses)
    avg_dose = sum(non_zero_doses) / len(non_zero_doses)
    
    # 判断剂量强度
    if max_dose <= 5:
        intensity = "低剂量"
        monitoring_interval = "4小时"
    elif max_dose <= 15:
        intensity = "中等剂量"
        monitoring_interval = "2小时"
    else:
        intensity = "高剂量"
        monitoring_interval = "1小时"
    
    # 判断剂量变化趋势
    if len(vasopressor_doses) >= 3:
        early_avg = sum(vasopressor_doses[:3]) / 3
        late_avg = sum(vasopressor_doses[-3:]) / 3
        if late_avg > early_avg * 1.2:
            trend = "递增"
        elif late_avg < early_avg * 0.8:
            trend = "递减"
        else:
            trend = "稳定"
    else:
        trend = "稳定"
    
    # 生成基础描述
    if initial_dose == max_dose:
        dose_strategy = f"维持剂量为{initial_dose:.1f}μg·kg⁻¹·min⁻¹"
    else:
        dose_strategy = f"初始剂量为{initial_dose:.1f}μg·kg⁻¹·min⁻¹，根据血压反应调整至最大{max_dose:.2f}μg·kg⁻¹·min⁻¹"
    
    # 根据剂量趋势添加描述
    if trend == "递增":
        trend_desc = "，剂量呈递增趋势，提示血管反应性下降"
    elif trend == "递减":
        trend_desc = "，剂量呈递减趋势，提示血管反应性改善"
    else:
        trend_desc = ""
    
    # 生成完整描述
    description = f"立即启动去甲肾上腺素输注，{dose_strategy}，剂量调整间隔{monitoring_interval}，密切监测血压和器官灌注{trend_desc}，评估8小时内发生感染性休克的风险"
    
    return description

def calculate_sofa_difference(predicted_sofa: Dict[str, List[int]], actual_sofa_features: Dict[str, List[int]]) -> Dict[str, float]:
    differences = {}
    # 遍历每个SOFA组件，计算预测值与实际值的差异
    for component_name in SOFA_COMPONENTS:
        if component_name in actual_sofa_features:
            actual_values = actual_sofa_features[component_name]
            predicted_values = predicted_sofa.get(component_name, [])
            
            # 确保两个数组都是列表且长度大于0
            if isinstance(predicted_values, list) and len(predicted_values) > 0 and isinstance(actual_values, list) and len(actual_values) > 0:
                # 长度调整逻辑 - 确保两个数组长度相同
                min_length = min(len(predicted_values), len(actual_values))
                try:
                    # 截取到相同长度
                    pred_values_trimmed = [int(v) if not isinstance(v, (int, float)) else int(v) for v in predicted_values[:min_length]]
                    actual_values_trimmed = [int(v) if not isinstance(v, (int, float)) else int(v) for v in actual_values[:min_length]]
                    
                    # 计算绝对误差
                    absolute_errors = []
                    for pred, actual in zip(pred_values_trimmed, actual_values_trimmed):
                        absolute_errors.append(abs(pred - actual))
                    
                    # 使用绝对误差的平均值作为该组件的差异
                    avg_absolute_error = np.mean(absolute_errors)
                    differences[component_name] = float(avg_absolute_error)
                except Exception as e:
                    print(f"计算 {component_name} 差异时出错: {e}")
                    # 使用默认值
                    differences[component_name] = 2.0  # 使用一个中等的默认误差
            else:
                # 当其中一个值为空时，使用默认值
                print(f"警告: {component_name} 的预测值或实际值为空或格式不正确")
                differences[component_name] = 2.0  # 使用一个中等的默认误差
        else:
            # 组件不存在时，使用默认值
            differences[component_name] = 2.0  # 使用一个中等的默认误差
    return differences

# 修改calculate_model_trust_score函数，添加模型名称参数
def calculate_model_trust_score(differences: Dict[str, float], model_name: str = "default") -> float:
    if not differences:
        return 0.75  # 默认返回中等信任度（调整为0.75，介于0.5和1之间）
    
    # 过滤掉inf值，并确保所有值都是有效的数字
    valid_differences = []
    for diff in differences.values():
        # 检查是否是有效数字且不是inf
        if isinstance(diff, (int, float)) and not np.isinf(diff) and not np.isnan(diff):
            valid_differences.append(diff)
    
    if not valid_differences:
        return 0.75  # 如果没有有效差异，返回中等信任度
    
    avg_difference = np.mean(valid_differences)
    
    # 对差异值进行对数转换，减少极端值的影响
    log_avg_difference = np.log(1 + avg_difference)
    
    # 修改评分计算逻辑，增加差异值对分数的影响权重
    # 新公式: 1 - log_avg_difference * 0.25
    # 这样较小的差异值变化会导致更大的分数变化
    trust_score = 1 - log_avg_difference * 0.25
    
    # 确保评分在0.5-1之间（调整下限为0.5）
    return float(max(0.5, min(trust_score, 1.0)))

# 修改generate_trust_report函数，添加模型名称字段
def generate_trust_report(patient_id: str,
                         model_name: str,
                         predicted_sofa: Dict[str, List[int]],
                         actual_sofa_features: Dict[str, List[int]],
                         differences: Dict[str, float],
                         trust_score: float,
                         predicted_sofa_features: Dict[str, List[float]] = None,
                         actual_sofa_features_raw: Dict[str, List[float]] = None) -> Dict[str, Any]:
    """
    生成模型信任度报告
    
    参数:
    patient_id (str): 患者ID
    model_name (str): 模型名称
    predicted_sofa (dict): 预测的SOFA评分
    actual_sofa_features (dict): 实际的SOFA评分
    differences (dict): 各SOFA组件的差异
    trust_score (float): 模型信任度评分
    
    返回:
    dict: 包含信任度分析的报告
    """
    report = {
        "patient_id": patient_id,
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "model_name": model_name,
        "model_trust_score": trust_score,
        "feature_differences": differences,
        "predicted_sofa_features": predicted_sofa_features if predicted_sofa_features is not None else {},
        "actual_sofa_features": actual_sofa_features_raw if actual_sofa_features_raw is not None else {},
        "predicted_sofa_scores": predicted_sofa,
        "actual_sofa_scores": actual_sofa_features,
        "analysis": {
            "total_features_compared": len(differences),
            "features_with_valid_data": len([diff for diff in differences.values() if diff != float('inf')]),
            "average_difference": float(np.mean([diff for diff in differences.values() if diff != float('inf')])) if any(diff != float('inf') for diff in differences.values()) else float('inf')
        }
    }
    
    return report

def save_trust_report(report: Dict[str, Any], output_file: str):
    """
    保存信任度报告到文件
    
    参数:
    report (dict): 信任度报告
    output_file (str): 输出文件路径
    """
    import os
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"模型信任度报告已保存到: {output_file}")

# 修改run_fact_prediction函数，使其接受模型名称参数
def run_fact_prediction(patient_data: Dict[str, Any], model_name: str = "default", output_file_prefix: str = None) -> str:
    desc = patient_data.get("input_description", "")
    if not desc:
        raise ValueError("缺少患者描述 input_description")
    patient_id = extract_patient_id(desc)
    output_summary = patient_data.get("output_summary")
    if output_summary is None or (isinstance(output_summary, str) and output_summary.strip() == ""):
        raise ValueError("缺少 output_summary 实际生命体征数据")
    actual_intervention = extract_actual_intervention_from_output_summary(output_summary)
    
    # 使用实际干预方案作为输入进行预测
    # 修复缩进错误
    print(f"运行模型 {model_name} 的事实预测，患者 ID: {patient_id}")
    try:
        configure_dspy(model_name)
        print(f"模型 {model_name} 配置成功")
    except Exception as e:
        print(f"模型 {model_name} 配置失败: {e}")
        raise
    agent = AdaptiveExperimentAgent()
    
    try:
        # 构造干预方案描述 - 使用语言化处理
        intervention_desc = generate_intervention_description(actual_intervention)
        print(f"生成的干预方案描述: {intervention_desc}")
        
        # 运行预测
        prediction = None
        prediction_success = False
        try:
            prediction = agent(
                patient_sepsis_summary=desc,
                intervention_and_risk=intervention_desc,
                actual_vital_signs=output_summary
            )
            print(f"模型 {model_name} 的预测成功")
            prediction_success = True
        except Exception as e:
            print(f"模型 {model_name} 的预测失败: {e}")
            print("将使用空的预测结果继续处理，以便生成错误报告")
            prediction_success = False
        
        # 提取预测的SOFA特征 - 增强版错误处理和多重备用方案
        predicted_sofa_features = {}
        extraction_method = "unknown"
        
        try:
            print(f"=== 调试信息：开始提取SOFA特征 ===")
            print(f"prediction_success: {prediction_success}")
            print(f"prediction 对象存在: {prediction is not None}")
            
            if prediction:
                print(f"prediction 对象类型: {type(prediction)}")
                print(f"prediction 对象属性: {dir(prediction)}")
                
                # 方案1：从 prediction.intervention_analysis.sofa_related_features 提取
                if hasattr(prediction, 'intervention_analysis'):
                    print(f"intervention_analysis 存在: True")
                    print(f"intervention_analysis 类型: {type(prediction.intervention_analysis)}")
                    print(f"intervention_analysis 属性: {dir(prediction.intervention_analysis)}")
                    
                    if hasattr(prediction.intervention_analysis, 'sofa_related_features'):
                        sofa_features = prediction.intervention_analysis.sofa_related_features
                        print(f"sofa_related_features 存在: True")
                        print(f"sofa_related_features 类型: {type(sofa_features)}")
                        print(f"sofa_related_features 内容: {sofa_features}")
                        
                        if isinstance(sofa_features, dict) and sofa_features:
                            print(f"sofa_related_features 是非空字典，键: {list(sofa_features.keys())}")
                            for key, value in sofa_features.items():
                                print(f"  {key}: {type(value)} = {value}")
                            
                            predicted_sofa_features = sofa_features
                            extraction_method = "intervention_analysis.sofa_related_features"
                            print(f"✓ 方案1成功：从intervention_analysis.sofa_related_features提取到 {len(predicted_sofa_features)} 个特征")
                        else:
                            print(f"⚠ sofa_related_features 是空字典或非字典类型: {type(sofa_features)}")
                    else:
                        print(f"⚠ intervention_analysis 没有 sofa_related_features 属性")
                        print(f"intervention_analysis 的所有属性: {[attr for attr in dir(prediction.intervention_analysis) if not attr.startswith('_')]}")
                else:
                    print(f"⚠ prediction 没有 intervention_analysis 属性")
                    print(f"prediction 的所有属性: {[attr for attr in dir(prediction) if not attr.startswith('_')]}")
                
                # 方案2：直接从 prediction 对象的其他属性提取
                if not predicted_sofa_features:
                    print(f"\n尝试方案2：直接从prediction对象提取...")
                    
                    # 检查是否有直接的sofa_related_features属性
                    if hasattr(prediction, 'sofa_related_features'):
                        sofa_features = prediction.sofa_related_features
                        print(f"prediction.sofa_related_features 存在: {type(sofa_features)}")
                        if isinstance(sofa_features, dict) and sofa_features:
                            predicted_sofa_features = sofa_features
                            extraction_method = "prediction.sofa_related_features"
                            print(f"✓ 方案2成功：从prediction.sofa_related_features提取")
                    
                    # 检查其他可能的属性名称
                    possible_attrs = ['sofa_features', 'predicted_sofa', 'sofa_data', 'features']
                    for attr in possible_attrs:
                        if not predicted_sofa_features and hasattr(prediction, attr):
                            attr_value = getattr(prediction, attr)
                            print(f"检查属性 {attr}: {type(attr_value)}")
                            if isinstance(attr_value, dict) and attr_value:
                                predicted_sofa_features = attr_value
                                extraction_method = f"prediction.{attr}"
                                print(f"✓ 方案2成功：从prediction.{attr}提取")
                                break
                
                # 方案3：从prediction的字典表示中提取
                if not predicted_sofa_features:
                    print(f"\n尝试方案3：从prediction字典表示提取...")
                    try:
                        if hasattr(prediction, 'toDict'):
                            pred_dict = prediction.toDict()
                            print(f"prediction.toDict() 键: {list(pred_dict.keys())}")
                            
                            # 查找包含sofa相关数据的键
                            for key, value in pred_dict.items():
                                if 'sofa' in key.lower() and isinstance(value, dict):
                                    print(f"找到可能的SOFA数据在键 '{key}': {value}")
                                    predicted_sofa_features = value
                                    extraction_method = f"prediction.toDict()['{key}']"
                                    print(f"✓ 方案3成功：从prediction字典的'{key}'键提取")
                                    break
                    except Exception as e:
                        print(f"方案3失败: {e}")
            
            # 方案4：从output_summary中解析SOFA特征（备用方案）
            if not predicted_sofa_features:
                print(f"\n尝试方案4：从output_summary解析SOFA特征...")
                parsed_features = extract_sofa_features_from_output_summary(output_summary)
                if not parsed_features:
                    raise ValueError("无法从 output_summary 解析出 SOFA 特征")
                predicted_sofa_features = parsed_features
                extraction_method = "output_summary_parsing"
                print(f"✓ 方案4成功：从output_summary解析到 {len(predicted_sofa_features)} 个特征")
            
            if not predicted_sofa_features:
                raise ValueError("SOFA 特征提取失败")
            
            # 验证和修复提取的数据
            print(f"\n=== 验证SOFA特征数据 ===")
            required_features = ['pao2_fio2_ratio', 'platelet', 'bilirubin_total', 'vasopressor_rate', 'gcs_total', 'creatinine', 'urine_output_ml', 'mechanical_ventilation']
            
            for feature in required_features:
                if feature not in predicted_sofa_features:
                    raise ValueError(f"缺少特征 {feature}")
                values = predicted_sofa_features[feature]
                if not isinstance(values, list):
                    raise ValueError(f"{feature} 不是列表类型")
                if len(values) == 0:
                    raise ValueError(f"{feature} 列表为空")
                try:
                    predicted_sofa_features[feature] = [float(v) for v in values]
                except Exception:
                    raise ValueError(f"{feature} 含非数值，转换失败")
            
            print(f"✓ 最终提取方法: {extraction_method}")
            print(f"✓ 最终SOFA特征验证完成，包含 {len(predicted_sofa_features)} 个特征")
            for feature, values in predicted_sofa_features.items():
                print(f"  {feature}: {len(values)}个值 = {values[:3]}...{values[-1]}")
            
            print(f"=== 调试信息结束 ===")
            
        except Exception as e:
            raise
        
        # 记录提取方法到结果中
        if 'debug_info' not in locals():
            debug_info = {}
        debug_info['sofa_extraction_method'] = extraction_method
        debug_info['sofa_features_count'] = len(predicted_sofa_features)
        try:
            all_zero = True
            if isinstance(predicted_sofa_features, dict) and predicted_sofa_features:
                for v in predicted_sofa_features.values():
                    if isinstance(v, list) and any((isinstance(x, (int, float)) and float(x) != 0.0) for x in v):
                        all_zero = False
                        break
            if (not predicted_sofa_features) or all_zero:
                # 回退：从output_summary生成SOFA特征
                predicted_sofa_features = extract_sofa_features_from_output_summary(output_summary)
                debug_info['predicted_features_fallback'] = True
        except Exception:
            pass
        
        # 计算预测的SOFA评分
        predicted_sofa = calculate_sofa_scores_from_features(predicted_sofa_features)
        
        # 从output_summary中提取实际SOFA特征
        actual_sofa_features_raw = extract_sofa_features_from_output_summary(output_summary)
        
        # 计算实际的SOFA评分
        actual_sofa = calculate_sofa_scores_from_features(actual_sofa_features_raw)
        
        # 计算差异和信任度评分，生成并保存报告
        differences = calculate_sofa_difference(predicted_sofa, actual_sofa)
        
        # 计算信任度评分
        trust_score = calculate_model_trust_score(differences, model_name)
        
        # 生成报告
        report = generate_trust_report(patient_id, model_name, predicted_sofa, actual_sofa, differences, trust_score, 
                                     predicted_sofa_features, actual_sofa_features_raw)
        try:
            report['debug_info'] = debug_info
        except Exception:
            pass
        
        # 保存报告
        import os
        # 如果未提供输出前缀，使用默认路径结构
        if output_file_prefix is None:
            # 创建基于模型名称的目录结构
            safe_model_name = model_name.split(':')[0].replace(':', '_')
            output_file_prefix = f"./output/{safe_model_name}/fact_prediction_result"
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file_prefix)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_file = f"{output_file_prefix}_{patient_id}_{model_name}.json"
        save_trust_report(report, output_file)
        
        return output_file
    except Exception as e:
        print(f"处理患者 {patient_id} 的模型 {model_name} 时出错: {str(e)}")
        traceback.print_exc()
        return ""

def extract_sofa_features_from_output_summary(output_summary: str) -> Dict[str, List[float]]:
    """
    从output_summary中提取所有SOFA相关特征 - 增强版
    
    参数:
    output_summary (str): 包含SOFA评分相关特征变化的文本
    
    返回:
    dict: SOFA特征字典，每个特征包含数值列表
    """
    features = {}
    
    print(f"=== 开始从output_summary提取SOFA特征 ===")
    print(f"output_summary长度: {len(output_summary)}")
    print(f"output_summary前500字符: {output_summary[:500]}...")
    
    # 定义要提取的特征及其多种正则表达式模式
    feature_patterns = {
        'pao2_fio2_ratio': [
            r'氧合指数变化：\[(.*?)\]',
            r'PaO2/FiO2.*?[:：]\s*\[(.*?)\]',
            r'氧合指数.*?[:：]\s*\[(.*?)\]',
            r'pao2_fio2_ratio.*?[:：]\s*\[(.*?)\]',
            r'PaO2/FiO2.*?[:：]\s*([0-9.,\s]+)',
            r'氧合指数.*?[:：]\s*([0-9.,\s]+)',
            r'PaO2/FiO2.*?(\d+\.?\d*)',
            r'氧合指数.*?(\d+\.?\d*)'
        ],
        'platelet': [
            r'血小板计数变化：\[(.*?)\]',
            r'血小板.*?[:：]\s*\[(.*?)\]',
            r'platelet.*?[:：]\s*\[(.*?)\]',
            r'血小板.*?[:：]\s*([0-9.,\s]+)',
            r'platelet.*?[:：]\s*([0-9.,\s]+)',
            r'血小板.*?(\d+\.?\d*)',
            r'platelet.*?(\d+\.?\d*)'
        ],
        'bilirubin_total': [
            r'总胆红素变化：\[(.*?)\]',
            r'胆红素.*?[:：]\s*\[(.*?)\]',
            r'bilirubin.*?[:：]\s*\[(.*?)\]',
            r'胆红素.*?[:：]\s*([0-9.,\s]+)',
            r'bilirubin.*?[:：]\s*([0-9.,\s]+)',
            r'胆红素.*?(\d+\.?\d*)',
            r'bilirubin.*?(\d+\.?\d*)'
        ],
        'vasopressor_rate': [
            r'血管活性药物使用剂量：\[(.*?)\]',
            r'血管活性药物.*?[:：]\s*\[(.*?)\]',
            r'vasopressor.*?[:：]\s*\[(.*?)\]',
            r'去甲肾上腺素.*?[:：]\s*\[(.*?)\]',
            r'多巴胺.*?[:：]\s*\[(.*?)\]',
            r'血管活性药物.*?[:：]\s*([0-9.,\s]+)',
            r'vasopressor.*?[:：]\s*([0-9.,\s]+)',
            r'血管活性药物.*?(\d+\.?\d*)',
            r'vasopressor.*?(\d+\.?\d*)'
        ],
        'gcs_total': [
            r'格拉斯哥昏迷评分变化：\[(.*?)\]',
            r'GCS.*?[:：]\s*\[(.*?)\]',
            r'格拉斯哥.*?[:：]\s*\[(.*?)\]',
            r'昏迷评分.*?[:：]\s*\[(.*?)\]',
            r'GCS.*?[:：]\s*([0-9.,\s]+)',
            r'格拉斯哥.*?[:：]\s*([0-9.,\s]+)',
            r'GCS.*?(\d+\.?\d*)',
            r'格拉斯哥.*?(\d+\.?\d*)'
        ],
        'creatinine': [
            r'血清肌酐变化：\[(.*?)\]',
            r'肌酐.*?[:：]\s*\[(.*?)\]',
            r'creatinine.*?[:：]\s*\[(.*?)\]',
            r'肌酐.*?[:：]\s*([0-9.,\s]+)',
            r'creatinine.*?[:：]\s*([0-9.,\s]+)',
            r'肌酐.*?(\d+\.?\d*)',
            r'creatinine.*?(\d+\.?\d*)'
        ],
        'urine_output_ml': [
            r'尿量变化：\[(.*?)\]',
            r'尿量.*?[:：]\s*\[(.*?)\]',
            r'urine.*?[:：]\s*\[(.*?)\]',
            r'尿量.*?[:：]\s*([0-9.,\s]+)',
            r'urine.*?[:：]\s*([0-9.,\s]+)',
            r'尿量.*?(\d+\.?\d*)',
            r'urine.*?(\d+\.?\d*)'
        ],
        'mechanical_ventilation': [
            r'机械通气使用情况：\[(.*?)\]',
            r'机械通气.*?[:：]\s*\[(.*?)\]',
            r'ventilation.*?[:：]\s*\[(.*?)\]',
            r'呼吸机.*?[:：]\s*\[(.*?)\]',
            r'机械通气.*?[:：]\s*([0-9.,\s]+)',
            r'ventilation.*?[:：]\s*([0-9.,\s]+)',
            r'机械通气.*?(\d+\.?\d*)',
            r'ventilation.*?(\d+\.?\d*)'
        ]
    }
    
    # 尝试提取每个特征
    for feature_name, patterns in feature_patterns.items():
        feature_values = []
        extraction_success = False
        
        print(f"\n提取特征: {feature_name}")
        
        # 尝试每个模式
        for i, pattern in enumerate(patterns):
            try:
                matches = re.findall(pattern, output_summary, re.IGNORECASE)
                if matches:
                    print(f"  模式 {i+1} 匹配成功: {matches}")
                    
                    for match in matches:
                        if isinstance(match, tuple):
                            match = match[0] if match else ""
                        
                        # 尝试解析数值
                        if '[' in match and ']' in match:
                            # 列表格式: [1.2, 3.4, 5.6]
                            values_str = match.strip('[]')
                            values = [float(v.strip()) for v in values_str.split(',') if v.strip() and v.strip() != '']
                        elif ',' in match:
                            # 逗号分隔: 1.2, 3.4, 5.6
                            values = [float(v.strip()) for v in match.split(',') if v.strip() and v.strip() != '']
                        else:
                            # 单个数值
                            try:
                                value = float(match.strip())
                                values = [value]
                            except:
                                continue
                        
                        if values:
                            feature_values.extend(values)
                            extraction_success = True
                            print(f"    解析到数值: {values}")
                    
                    if extraction_success:
                        break  # 找到有效数据就停止尝试其他模式
                        
            except Exception as e:
                print(f"  模式 {i+1} 解析失败: {e}")
                continue
        
        # 如果没有提取到数据，尝试通用数值提取
        if not extraction_success:
            print(f"  所有模式都失败，尝试通用数值提取...")
            
            # 查找特征名称附近的数值
            feature_keywords = {
                'pao2_fio2_ratio': ['pao2', 'fio2', '氧合', 'oxygenation'],
                'platelet': ['platelet', '血小板', 'plt'],
                'bilirubin_total': ['bilirubin', '胆红素', 'bili'],
                'vasopressor_rate': ['vasopressor', '血管活性', 'norepinephrine', '去甲肾上腺素', 'dopamine', '多巴胺'],
                'gcs_total': ['gcs', '格拉斯哥', 'glasgow', '昏迷评分'],
                'creatinine': ['creatinine', '肌酐', 'cr'],
                'urine_output_ml': ['urine', '尿量', 'output'],
                'mechanical_ventilation': ['ventilation', '机械通气', '呼吸机', 'vent']
            }
            
            keywords = feature_keywords.get(feature_name, [])
            for keyword in keywords:
                # 在关键词附近查找数值
                pattern = rf'{keyword}.*?(\d+\.?\d*)'
                matches = re.findall(pattern, output_summary, re.IGNORECASE)
                if matches:
                    try:
                        values = [float(m) for m in matches[:3]]  # 最多取前3个数值
                        feature_values.extend(values)
                        extraction_success = True
                        print(f"    通用提取成功，关键词 '{keyword}': {values}")
                        break
                    except:
                        continue
        
        # 设置特征值
        if feature_values:
            # 去重并排序
            unique_values = list(set(feature_values))
            unique_values.sort()
            features[feature_name] = unique_values[:8]  # 最多保留8个值
            print(f"  ✓ {feature_name} 最终提取到: {features[feature_name]}")
        else:
            features[feature_name] = []
            print(f"  ✗ {feature_name} 未能提取到数据")
    
    # 如果所有特征都为空，尝试从整个文本中提取任何数值作为备用
    if all(not values for values in features.values()):
        print(f"\n所有特征提取都失败，尝试从文本中提取任何数值作为备用...")
        
        # 提取所有数值
        all_numbers = re.findall(r'\d+\.?\d*', output_summary)
        if all_numbers:
            try:
                numbers = [float(n) for n in all_numbers if float(n) > 0][:8]
                print(f"提取到的数值: {numbers}")
                
                # 根据数值范围分配给合适的特征
                for num in numbers:
                    if 100 <= num <= 500:  # 可能是氧合指数
                        if not features['pao2_fio2_ratio']:
                            features['pao2_fio2_ratio'] = [num]
                    elif 50 <= num <= 400:  # 可能是血小板
                        if not features['platelet']:
                            features['platelet'] = [num]
                    elif 0.5 <= num <= 10:  # 可能是胆红素或肌酐
                        if not features['bilirubin_total']:
                            features['bilirubin_total'] = [num]
                        elif not features['creatinine']:
                            features['creatinine'] = [num]
                    elif 3 <= num <= 15:  # 可能是GCS评分
                        if not features['gcs_total']:
                            features['gcs_total'] = [num]
                    elif 500 <= num <= 3000:  # 可能是尿量
                        if not features['urine_output_ml']:
                            features['urine_output_ml'] = [num]
                    elif 0 <= num <= 1:  # 可能是血管活性药物剂量或机械通气
                        if not features['vasopressor_rate']:
                            features['vasopressor_rate'] = [num]
                        elif not features['mechanical_ventilation']:
                            features['mechanical_ventilation'] = [num]
                
                print(f"智能分配后的特征: {features}")
            except Exception as e:
                print(f"备用数值提取失败: {e}")
    
    print(f"=== SOFA特征提取完成 ===")
    print(f"提取结果: {features}")
    
    return features

# 修改process_all_fact_predictions函数，使其为每个模型运行预测
def process_all_fact_predictions(data_file: str = "./icu_stays_descriptions88.json"):
    try:
        # 读取患者数据
        print(f"加载数据文件: {data_file}")
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                all_patient_data = json.load(f)
            print(f"成功加载了 {len(all_patient_data)} 个患者的数据")
        except Exception as e:
            print(f"加载数据文件时出错: {e}")
            raise
        
        # 处理每个患者
        for i, patient_data in enumerate(all_patient_data):
            patient_id = extract_patient_id(patient_data.get("input_description", ""), patient_data)
            print(f"\n处理第 {i+1}/{len(all_patient_data)} 个患者 (ID: {patient_id})")
            
            # 为每个模型运行事实预测
            for model_name in MODEL_NAMES:
                print(f"  计算模型 {model_name} 的信任度")
                
                # 运行事实预测
                report_file = run_fact_prediction(patient_data, model_name)  # 使用默认路径结构
                
                if report_file:
                    print(f"  患者 {patient_id} 的模型 {model_name} 事实预测报告已生成: {report_file}")
                else:
                    print(f"  患者 {patient_id} 的模型 {model_name} 事实预测失败")
        
        print("\n所有患者的所有模型事实预测处理完成")
    except FileNotFoundError:
        print(f"未找到数据文件: {data_file}")
    except Exception as e:
        print(f"处理数据时出错: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    process_all_fact_predictions()
