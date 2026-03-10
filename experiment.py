import json
import dspy
import numpy as np
import re
import os
import time
import traceback
from typing import List, Dict, Any
from collections import defaultdict
from datetime import datetime

# --- Step 1: Configure DSPy ---
# 修改configure_dspy函数，使其接受model_name参数
def configure_dspy(model_name):
    # 确保模型名称格式正确，添加ollama/前缀（如果尚未添加）
    if not model_name.startswith('ollama/'):
        full_model_name = f'ollama/{model_name}'
    else:
        full_model_name = model_name
    
    lm = dspy.LM(
        model=full_model_name,
        model_type='chat',
        base_url='http://localhost:11434',
        api_key='no-key-needed',
        temperature=0.5,  # 进一步降低温度以获得更一致的输出
        max_tokens=8192,  # 增加最大token数以确保完整输出
        # 移除JSON模式强制要求，让模型自然生成
    )
    
    dspy.configure(lm=lm)
    return lm

# --- 添加SOFA评分计算函数 ---
def calculate_sofa_score(pao2_fio2_ratio=None, mechanical_ventilation=False, 
                       platelet=None, bilirubin_total=None, 
                       map_value=None, vasopressor_rate=None, 
                       gcs_total=None, creatinine=None, urine_output_ml=None):
    """根据提供的参数计算SOFA评分
    遵循2021年更新的SOFA评分标准
    
    Args:
        pao2_fio2_ratio: 动脉血氧分压与吸入氧浓度之比
        mechanical_ventilation: 是否使用机械通气(True/False或1/0)
        platelet: 血小板计数(×10³/μL)
        bilirubin_total: 血清总胆红素水平(mg/dL)
        map_value: 平均动脉压(mmHg)
        vasopressor_rate: 血管活性药物的给药速率
        gcs_total: 格拉斯哥昏迷评分总分
        creatinine: 血清肌酐水平(mg/dL)
        urine_output_ml: 24小时尿量(mL)
        
    Returns:
        dict: 包含各系统评分和总分的字典
    """
    sofa_scores = {}
    
    # 1. 呼吸系统评分 (sofa_respiration)
    if mechanical_ventilation or (isinstance(mechanical_ventilation, (int, float)) and mechanical_ventilation > 0):
        sofa_scores['sofa_respiration'] = 4
    elif pao2_fio2_ratio is not None:
        if pao2_fio2_ratio < 100:
            sofa_scores['sofa_respiration'] = 4
        elif pao2_fio2_ratio < 200:
            sofa_scores['sofa_respiration'] = 3
        elif pao2_fio2_ratio < 300:
            sofa_scores['sofa_respiration'] = 2
        elif pao2_fio2_ratio < 400:
            sofa_scores['sofa_respiration'] = 1
        else:
            sofa_scores['sofa_respiration'] = 0
    else:
        sofa_scores['sofa_respiration'] = 0
    
    # 2. 凝血系统评分 (sofa_coagulation)
    if platelet is not None:
        if platelet < 20:
            sofa_scores['sofa_coagulation'] = 4
        elif platelet < 50:
            sofa_scores['sofa_coagulation'] = 3
        elif platelet < 100:
            sofa_scores['sofa_coagulation'] = 2
        elif platelet < 150:
            sofa_scores['sofa_coagulation'] = 1
        else:
            sofa_scores['sofa_coagulation'] = 0
    else:
        sofa_scores['sofa_coagulation'] = 0
    
    # 3. 肝脏系统评分 (sofa_liver)
    if bilirubin_total is not None:
        if bilirubin_total >= 12:
            sofa_scores['sofa_liver'] = 4
        elif bilirubin_total >= 6:
            sofa_scores['sofa_liver'] = 3
        elif bilirubin_total >= 2:
            sofa_scores['sofa_liver'] = 2
        elif bilirubin_total >= 1.2:
            sofa_scores['sofa_liver'] = 1
        else:
            sofa_scores['sofa_liver'] = 0
    else:
        sofa_scores['sofa_liver'] = 0
    
    # 4. 血管系统评分 (sofa_cardiovascular)
    sofa_scores['sofa_cardiovascular'] = 0
    if vasopressor_rate is not None and vasopressor_rate > 0:
        if vasopressor_rate > 0.5:
            sofa_scores['sofa_cardiovascular'] = 4
        else:
            sofa_scores['sofa_cardiovascular'] = 3
    elif map_value is not None:
        if map_value < 70:
            sofa_scores['sofa_cardiovascular'] = 1
    
    # 5. 中枢神经系统评分 (sofa_cns)
    if gcs_total is not None:
        if gcs_total < 6:
            sofa_scores['sofa_cns'] = 4
        elif gcs_total < 9:
            sofa_scores['sofa_cns'] = 3
        elif gcs_total < 13:
            sofa_scores['sofa_cns'] = 2
        elif gcs_total < 15:
            sofa_scores['sofa_cns'] = 1
        else:
            sofa_scores['sofa_cns'] = 0
    else:
        sofa_scores['sofa_cns'] = 0
    
    # 6. 肾脏系统评分 (sofa_renal)
    sofa_scores['sofa_renal'] = 0
    if creatinine is not None:
        if creatinine >= 5:
            sofa_scores['sofa_renal'] = 4
        elif creatinine >= 3.5:
            sofa_scores['sofa_renal'] = 3
        elif creatinine >= 2:
            sofa_scores['sofa_renal'] = 2
        elif creatinine >= 1.2:
            sofa_scores['sofa_renal'] = 1
        
    # 尿量可以提升评分，但不会降低评分
    if urine_output_ml is not None:
        if urine_output_ml < 200:
            sofa_scores['sofa_renal'] = max(sofa_scores['sofa_renal'], 4)
        elif urine_output_ml < 500:
            sofa_scores['sofa_renal'] = max(sofa_scores['sofa_renal'], 3)
    
    # 计算总分
    sofa_scores['sofa_total'] = sum(sofa_scores.values())
    
    return sofa_scores

# --- Step 2: Define DSPy Signatures ---
# 增强SepsisShockRiskAssessment签名提示
class SepsisShockRiskAssessment(dspy.Signature):
    """评估感染性休克风险
    
    重要：key_clinical_indicators必须是字符串列表，例如：["血压下降", "心率增快", "乳酸升高"]
    不能是字典或其他格式。
    
    示例输出格式：
    reasoning: "基于患者的生命体征分析..."
    risk_level: "high"
    key_clinical_indicators: ["血压下降", "心率增快", "乳酸升高"]
    current_sepsis_state_summary: "患者目前处于..."
    """
    patient_document: str = dspy.InputField(desc="患者的完整临床文档")
    reasoning: str = dspy.OutputField(desc="Chain-of-thought 推理过程")
    risk_level: str = dspy.OutputField(desc="风险等级：high/medium/low")
    key_clinical_indicators: List[str] = dspy.OutputField(desc="影响判断的最主要三个临床指标，必须是字符串列表格式")
    current_sepsis_state_summary: str = dspy.OutputField(desc="当前脓毒症状态总结，供后续模块使用")

class AssessConfidence(dspy.Signature):
    """评估输出的置信度"""
    context: str = dspy.InputField()
    output_content: str = dspy.InputField()
    reasoning: str = dspy.OutputField(desc="评估推理过程")
    confidence_score: float = dspy.OutputField(desc="0-100的置信度分数")

# 修改AnalyzeInterventionAndRisk签名
class AnalyzeInterventionAndRisk(dspy.Signature):
    """分析干预措施的影响和风险，预测未来8小时的SOFA评分相关特征趋势
    
    **重要：必须严格按照JSON格式输出，不能有任何额外的文本或格式**
    
    输出格式要求：
    1. reasoning: 字符串，详细的推理过程
    2. predicted_outcome: 字符串，预测结果（如"stable", "improving", "deteriorating"）
    3. potential_risks: 字符串数组，潜在风险列表，例如：["感染恶化", "器官功能衰竭", "血压不稳定"]
    4. risk_level: 字符串，必须是 "high", "medium", "low" 之一
    5. sofa_related_features: 字典对象，包含8个SOFA特征的未来8小时预测值
    
    sofa_related_features必须包含以下8个键，每个键对应8个数值的数组（代表未来8小时的预测值）：
    - "pao2_fio2_ratio": 氧合指数 (正常值200-500)
    - "platelet": 血小板计数 (正常值150 x10³-450 x10³/μL)
    - "bilirubin_total": 总胆红素 (正常值0.3-1.2 mg/dL)
    - "vasopressor_rate": 血管活性药物剂量 (0表示无，>0表示使用剂量)
    - "gcs_total": 格拉斯哥昏迷评分 (3-15分)
    - "creatinine": 血清肌酐 (正常值0.6-1.2 mg/dL)
    - "urine_output_ml": 尿量 (正常值>2000ml/24h)
    - "mechanical_ventilation": 机械通气 (0=无，1=有)
    
    **输出示例（必须严格遵循此JSON格式）：**
    {
        "reasoning": "基于患者当前的脓毒症状态和干预措施，分析预测结果...",
        "predicted_outcome": "stable",
        "potential_risks": ["感染恶化", "器官功能衰竭", "血压不稳定"],
        "risk_level": "medium",
        "sofa_related_features": {
            "pao2_fio2_ratio": [300.0, 280.0, 260.0, 240.0, 220.0, 200.0, 180.0, 160.0],
            "platelet": [150.0, 140.0, 130.0, 120.0, 110.0, 100.0, 90.0, 80.0],
            "bilirubin_total": [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6],
            "vasopressor_rate": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "gcs_total": [15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0],
            "creatinine": [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4],
            "urine_output_ml": [2000.0, 1800.0, 1600.0, 1400.0, 1200.0, 1000.0, 800.0, 600.0],
            "mechanical_ventilation": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }
    }
    
    **注意：数值必须合理且符合医学常识，不能全部为0或相同值**
    """
    current_sepsis_state_summary: str = dspy.InputField(desc="当前脓毒症状态总结")
    intervention_and_risk: str = dspy.InputField(desc="干预措施和风险评估")
    reasoning: str = dspy.OutputField(desc="详细的推理过程，解释如何基于当前状态和干预措施预测未来趋势")
    predicted_outcome: str = dspy.OutputField(desc="预测结果：stable/improving/deteriorating")
    potential_risks: List[str] = dspy.OutputField(desc="潜在风险列表，必须是字符串数组格式")
    risk_level: str = dspy.OutputField(desc="风险等级：high/medium/low")
    risk_score: float = dspy.OutputField(desc="风险评分，0-1 区间的实数")
    sofa_related_features: Dict[str, List[float]] = dspy.OutputField(desc="SOFA特征预测字典，包含8个键，每个键对应8个数值的数组，代表未来8小时的预测值")

# 修改CompareVitalSigns签名
class CompareVitalSigns(dspy.Signature):
    """比较预测的SOFA特征与实际数据，计算均方误差
    
    **重要：必须严格按照JSON格式输出，不能有任何额外的文本或格式**
    
    任务说明：
    1. 从actual_vital_signs中提取与predicted_sofa_features对应的数值
    2. 计算每个特征的均方误差(MSE)
    3. 计算所有特征的平均MSE
    4. 提供详细的比较分析
    
    输出格式要求：
    1. reasoning: 字符串，详细说明提取过程和计算方法
    2. parsed_actual_sofa_features: 字典，从实际数据中解析的SOFA特征值
    3. mean_squared_errors: 字典，每个特征的MSE值
    4. average_mse: 数值，所有特征的平均MSE
    5. comparison_summary: 字符串，比较结果总结
    
    **输出示例（必须严格遵循此JSON格式）：**
    {
        "reasoning": "从实际数据中提取了8个SOFA特征的数值，计算了与预测值的均方误差...",
        "parsed_actual_sofa_features": {
            "pao2_fio2_ratio": [280.0, 270.0, 260.0, 250.0, 240.0, 230.0, 220.0, 210.0],
            "platelet": [140.0, 135.0, 130.0, 125.0, 120.0, 115.0, 110.0, 105.0],
            "bilirubin_total": [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
            "vasopressor_rate": [0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "gcs_total": [14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0],
            "creatinine": [1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5],
            "urine_output_ml": [1900.0, 1700.0, 1500.0, 1300.0, 1100.0, 900.0, 700.0, 500.0],
            "mechanical_ventilation": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        },
        "mean_squared_errors": {
            "pao2_fio2_ratio": 25.5,
            "platelet": 12.3,
            "bilirubin_total": 0.08,
            "vasopressor_rate": 0.02,
            "gcs_total": 1.5,
            "creatinine": 0.15,
            "urine_output_ml": 2500.0,
            "mechanical_ventilation": 0.0
        },
        "average_mse": 317.44,
        "comparison_summary": "预测值与实际值的平均均方误差为317.44，其中尿量预测误差最大，机械通气预测最准确。"
    }
    
    **注意：如果actual_vital_signs为空或无法解析，应返回空字典和0值，并在reasoning中说明原因**
    """
    predicted_sofa_features: Dict[str, List[float]] = dspy.InputField(desc="预测的SOFA评分相关特征趋势，包含8个特征的8小时预测值")
    actual_vital_signs: Dict[str, List[float]] = dspy.InputField(desc="实际的SOFA评分相关特征数值，需要从中提取对应的特征值")
    reasoning: str = dspy.OutputField(desc="详细说明如何从实际数据中提取数值，以及如何计算均方误差的过程")
    parsed_actual_sofa_features: Dict[str, List[float]] = dspy.OutputField(desc="从实际数据中解析出的SOFA特征值，格式与predicted_sofa_features相同")
    mean_squared_errors: Dict[str, float] = dspy.OutputField(desc="各项特征的均方误差，键为特征名，值为MSE数值")
    average_mse: float = dspy.OutputField(desc="所有特征的平均均方误差")
    comparison_summary: str = dspy.OutputField(desc="比较结果总结，包括误差分析和预测准确性评估")

# 修改GenerateClinicalReport签名
class GenerateClinicalReport(dspy.Signature):
    """生成临床报告"""
    current_sepsis_state_summary: str = dspy.InputField()
    intervention_and_risk: str = dspy.InputField()
    predicted_outcome: str = dspy.InputField()
    risk_level: str = dspy.InputField()
    potential_risks: List[str] = dspy.InputField()
    sofa_related_features: Dict[str, List[float]] = dspy.InputField(desc="预测未来的SOFA评分相关特征趋势，每小时一个数据点，必须包含pao2_fio2_ratio, platelet, bilirubin_total, vasopressor_rate, gcs_total, creatinine, urine_output_ml, mechanical_ventilation这8个键，每个键的值必须是包含8个数值的列表")
    comparison_summary: str = dspy.InputField(desc="预测与实际比较结果")
    sofa_scores: Dict[str, int] = dspy.InputField(desc="各器官系统SOFA评分")
    sofa_total: int = dspy.InputField(desc="SOFA总分")
    reasoning: str = dspy.OutputField(desc="Chain-of-thought reasoning")
    clinical_report: str = dspy.OutputField()

class ExperimentAgent(dspy.Program):
    def __init__(self):
        super().__init__()
        self.shock_risk_assessment = dspy.ChainOfThought(SepsisShockRiskAssessment)
        self.assess_confidence = dspy.ChainOfThought(AssessConfidence)
        self.analyze_intervention_and_risk = dspy.ChainOfThought(AnalyzeInterventionAndRisk)
        self.generate_clinical_report = dspy.ChainOfThought(GenerateClinicalReport)
        self.compare_vital_signs = dspy.ChainOfThought(CompareVitalSigns)
        
    def forward(self, patient_sepsis_summary, intervention_and_risk, actual_vital_signs, actual_sofa=None, **kwargs):
        debug_info = {
            'inputs': {
                'patient_sepsis_summary_present': bool(patient_sepsis_summary),
                'intervention_and_risk_present': bool(intervention_and_risk),
                'actual_vital_signs_type': type(actual_vital_signs).__name__,
                'actual_vital_signs_len': (len(actual_vital_signs) if isinstance(actual_vital_signs, dict) else None)
            },
            'stages': {},
            'errors': []
        }
        try:
            # 使用更强的错误处理和重试机制
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    risk_out = self.shock_risk_assessment(
                        patient_document=patient_sepsis_summary
                    )
                    
                    # 验证和修复key_clinical_indicators的数据类型
                    if hasattr(risk_out, 'key_clinical_indicators'):
                        if isinstance(risk_out.key_clinical_indicators, dict):
                            # 如果是字典，转换为列表
                            risk_out.key_clinical_indicators = list(risk_out.key_clinical_indicators.values())
                        elif not isinstance(risk_out.key_clinical_indicators, list):
                            # 如果不是列表，转换为列表
                            risk_out.key_clinical_indicators = [str(risk_out.key_clinical_indicators)]
                        
                        # 确保列表中的元素都是字符串
                        risk_out.key_clinical_indicators = [str(item) for item in risk_out.key_clinical_indicators]
                    else:
                        risk_out.key_clinical_indicators = []
                    
                    # 验证必需字段
                    if not hasattr(risk_out, 'risk_level') or not risk_out.risk_level:
                        risk_out.risk_level = 'unknown'
                    if not hasattr(risk_out, 'current_sepsis_state_summary') or not risk_out.current_sepsis_state_summary:
                        risk_out.current_sepsis_state_summary = '状态评估失败'
                    if not hasattr(risk_out, 'reasoning') or not risk_out.reasoning:
                        risk_out.reasoning = '推理过程不可用'
                    
                    break  # 成功则跳出重试循环
                    
                except Exception as retry_error:
                    print(f"风险评估尝试 {attempt + 1} 失败: {str(retry_error)}")
                    if attempt == max_retries - 1:
                        raise retry_error
                    time.sleep(1)  # 短暂等待后重试
            
            debug_info['stages']['risk_assessment'] = {
                'risk_level': getattr(risk_out, 'risk_level', None),
                'summary_present': hasattr(risk_out, 'current_sepsis_state_summary') and bool(risk_out.current_sepsis_state_summary),
                'indicators_type': type(risk_out.key_clinical_indicators).__name__,
                'indicators_count': len(risk_out.key_clinical_indicators) if hasattr(risk_out, 'key_clinical_indicators') else 0
            }
        except Exception as e:
            debug_info['errors'].append({'stage': 'risk_assessment', 'error': str(e)})
            traceback.print_exc()
            # 继续执行，提供回退结构
            risk_out = dspy.Prediction(reasoning=str(e), risk_level='unknown', key_clinical_indicators=[], current_sepsis_state_summary='')

        try:
            # 使用重试机制和强化验证
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    analysis_out = self.analyze_intervention_and_risk(
                        current_sepsis_state_summary=risk_out.current_sepsis_state_summary,
                        intervention_and_risk=intervention_and_risk
                    )
                    
                    # 验证和修复sofa_related_features的数据类型
                    required_sofa_keys = [
                         'pao2_fio2_ratio', 'platelet', 'bilirubin_total', 'vasopressor_rate',
                         'gcs_total', 'creatinine', 'urine_output_ml', 'mechanical_ventilation'
                     ]
                     
                    if hasattr(analysis_out, 'sofa_related_features'):
                         if not isinstance(analysis_out.sofa_related_features, dict):
                             raise ValueError(f"sofa_related_features必须为字典，当前为{type(analysis_out.sofa_related_features)}")
                         for key in required_sofa_keys:
                             if key not in analysis_out.sofa_related_features:
                                 raise ValueError(f"缺少SOFA特征键: {key}")
                             vals = analysis_out.sofa_related_features[key]
                             if not isinstance(vals, list):
                                 raise ValueError(f"SOFA特征 {key} 的值必须为列表")
                             cleaned = []
                             for v in vals:
                                 try:
                                     cleaned.append(float(v))
                                 except Exception:
                                     raise ValueError(f"SOFA特征 {key} 包含非数值项: {v}")
                             if len(cleaned) < 8:
                                 pad = cleaned[-1] if cleaned else 0.0
                                 cleaned = cleaned + [pad] * (8 - len(cleaned))
                             elif len(cleaned) > 8:
                                 cleaned = cleaned[:8]
                             analysis_out.sofa_related_features[key] = cleaned
                    else:
                         raise ValueError("缺少sofa_related_features")
                    
                    # 验证potential_risks字段
                    if hasattr(analysis_out, 'potential_risks'):
                        if isinstance(analysis_out.potential_risks, dict):
                            analysis_out.potential_risks = list(analysis_out.potential_risks.values())
                        elif not isinstance(analysis_out.potential_risks, list):
                            analysis_out.potential_risks = [str(analysis_out.potential_risks)]
                    else:
                        analysis_out.potential_risks = []
                    
                    # 验证其他必需字段
                    if not hasattr(analysis_out, 'predicted_outcome') or not analysis_out.predicted_outcome:
                        raise ValueError('缺少predicted_outcome')
                    if not hasattr(analysis_out, 'risk_level') or not analysis_out.risk_level:
                        raise ValueError('缺少risk_level')
                    if not hasattr(analysis_out, 'reasoning') or not analysis_out.reasoning:
                        analysis_out.reasoning = (
                            f"Predicted outcome: {getattr(analysis_out, 'predicted_outcome', 'unknown')}. "
                            f"Risk level: {getattr(analysis_out, 'risk_level', 'unknown')}. "
                            "SOFA-related features provided for 8 hours with numeric trends."
                        )
                    
                    break  # 成功则跳出重试循环
                    
                except Exception as retry_error:
                    print(f"干预分析尝试 {attempt + 1} 失败: {str(retry_error)}")
                    # 最后一轮失败时，直接抛出错误，遵循“无默认值”原则
                    if attempt == max_retries - 1:
                        raise retry_error
                    time.sleep(1)  # 短暂等待后重试
            
            sofa_features_type = type(getattr(analysis_out, 'sofa_related_features', None)).__name__
            debug_info['stages']['intervention_analysis'] = {
                'risk_level': getattr(analysis_out, 'risk_level', None),
                'predicted_outcome_present': hasattr(analysis_out, 'predicted_outcome') and bool(analysis_out.predicted_outcome),
                'sofa_related_features_type': sofa_features_type,
                'sofa_keys': (list(analysis_out.sofa_related_features.keys()) if isinstance(analysis_out.sofa_related_features, dict) else None)
            }
        except Exception as e:
            debug_info['errors'].append({'stage': 'intervention_analysis', 'error': str(e)})
            traceback.print_exc()
            analysis_out = dspy.Prediction(reasoning=str(e), predicted_outcome='预测失败', potential_risks=[], risk_level='unknown', sofa_related_features={})
        
        # 添加预处理步骤
        try:
            preprocessed_vitals = preprocess_vital_signs(actual_vital_signs)
            debug_info['stages']['preprocess_vitals'] = {
                'keys': list(preprocessed_vitals.keys()) if isinstance(preprocessed_vitals, dict) else None,
                'lengths': {k: (len(v) if isinstance(v, list) else None) for k, v in preprocessed_vitals.items()} if isinstance(preprocessed_vitals, dict) else None
            }
        except Exception as e:
            debug_info['errors'].append({'stage': 'preprocess_vitals', 'error': str(e)})
            traceback.print_exc()
            preprocessed_vitals = {}
        # 如果预处理为空或不完整，使用回退函数对实际体征进行规范化，确保比较阶段有数据
        try:
            if not isinstance(preprocessed_vitals, dict) or not preprocessed_vitals:
                preprocessed_vitals = generate_fallback_sofa_features(actual_vital_signs)
                debug_info['stages']['preprocess_vitals_fallback'] = {
                    'keys': list(preprocessed_vitals.keys()),
                    'lengths': {k: (len(v) if isinstance(v, list) else None) for k, v in preprocessed_vitals.items()}
                }
        except Exception as e:
            debug_info['errors'].append({'stage': 'preprocess_vitals_fallback', 'error': str(e)})
            traceback.print_exc()
            preprocessed_vitals = {}
        
        # 确保所有模块使用统一参数名称
        # 使用本地计算替代LLM比较，避免解析失败导致流程中断
        try:
            # 规范化预测与实际体征数据
            normalized_pred = generate_fallback_sofa_features(getattr(analysis_out, 'sofa_related_features', {}))
            normalized_act = generate_fallback_sofa_features(preprocessed_vitals)
            
            # 计算MSE
            mse_result = compute_mse(normalized_pred, normalized_act)
            comparison_out = dspy.Prediction(
                reasoning=mse_result['reasoning'],
                parsed_actual_sofa_features=normalized_act,
                mean_squared_errors=mse_result['mean_squared_errors'],
                average_mse=mse_result['average_mse'],
                comparison_summary=mse_result['comparison_summary']
            )
            debug_info['stages']['compare_vital_signs'] = {
                'average_mse': mse_result['average_mse'],
                'mse_count': len(mse_result['mean_squared_errors'])
            }
        except Exception as e:
            debug_info['errors'].append({'stage': 'compare_vital_signs_local', 'error': str(e)})
            traceback.print_exc()
            comparison_out = dspy.Prediction(reasoning=str(e), parsed_actual_sofa_features={}, mean_squared_errors={}, average_mse=0.0, comparison_summary='本地比较失败')

        # 计算SOFA评分 - 使用新的评分函数
        sofa_scores = {}
        sofa_total = 0
        
        # 初始化时间点SOFA评分存储
        hourly_sofa_scores = {}
        hourly_sofa_totals = []
        
        # 从预测的SOFA特征中提取每个时间点的数值来计算SOFA评分
        if analysis_out.sofa_related_features:
            try:
                # 定义中文到英文特征名称的映射
                feature_name_mapping = {
                    '循环评分': 'cardiovascular',
                    '呼吸评分': 'respiration',
                    '神经系统评分': 'cns',
                    '肾脏评分': 'renal',
                    '肝脏评分': 'liver',
                    '凝血评分': 'coagulation'
                }
                
                # 获取特征数量（假设所有特征列表长度相同）
                feature_lengths = [len(values) for values in analysis_out.sofa_related_features.values() if isinstance(values, list)]
                if feature_lengths:
                    num_hours = min(feature_lengths)
                    
                    # 为每个时间点计算SOFA评分
                    for hour in range(num_hours):
                        # 获取每个特征在当前时间点的值
                        hourly_values = {}
                        for feature_name, values in analysis_out.sofa_related_features.items():
                            # 使用映射将中文名称转换为英文名称
                            english_name = feature_name_mapping.get(feature_name, feature_name)
                            if values and isinstance(values, list) and len(values) > hour:
                                hourly_values[english_name] = values[hour]
                            else:
                                hourly_values[english_name] = None
                        
                        # 打印调试信息
                        print(f"时间点 {hour} 的特征值: {hourly_values}")
                        
                        # 根据器官系统名称映射到对应的SOFA评分参数
                        # 注意：这里需要根据实际的特征值含义进行转换
                        sofa_params = {}
                        
                        # 呼吸系统评分 (PaO2/FiO2 ratio)
                        if 'respiration' in hourly_values and hourly_values['respiration'] is not None:
                            # 假设呼吸评分为1-4分，对应PaO2/FiO2比值为400-100
                            pao2_fio2_map = {1: 400, 2: 300, 3: 200, 4: 100}
                            sofa_params['pao2_fio2_ratio'] = pao2_fio2_map.get(int(hourly_values['respiration']), 400)
                        else:
                            sofa_params['pao2_fio2_ratio'] = 0
                        
                        # 凝血系统评分 (Platelet count)
                        if 'coagulation' in hourly_values and hourly_values['coagulation'] is not None:
                            # 假设凝血评分为1-4分，对应血小板计数为150000-20000
                            platelet_map = {1: 150000, 2: 100000, 3: 50000, 4: 20000}
                            sofa_params['platelet'] = platelet_map.get(int(hourly_values['coagulation']), 150000)
                        else:
                            sofa_params['platelet'] = 0
                        
                        # 肝脏系统评分 (Bilirubin level)
                        if 'liver' in hourly_values and hourly_values['liver'] is not None:
                            # 假设肝脏评分为1-4分，对应总胆红素为1.2-12 mg/dL
                            bilirubin_map = {1: 1.2, 2: 2.5, 3: 6.0, 4: 12.0}
                            sofa_params['bilirubin_total'] = bilirubin_map.get(int(hourly_values['liver']), 1.2)
                        else:
                            sofa_params['bilirubin_total'] = 0
                        
                        # 血管系统评分 (Mean arterial pressure and vasopressor use)
                        if 'cardiovascular' in hourly_values and hourly_values['cardiovascular'] is not None:
                            # 假设循环评分为1-4分，对应MAP为70-30 mmHg
                            map_map = {1: 70, 2: 65, 3: 50, 4: 30}
                            sofa_params['map_value'] = map_map.get(int(hourly_values['cardiovascular']), 70)
                            # 假设使用血管活性药物时评分为3-4分
                            if int(hourly_values['cardiovascular']) >= 3:
                                sofa_params['vasopressor_rate'] = 0.1  # 使用血管活性药物
                            else:
                                sofa_params['vasopressor_rate'] = 0  # 不使用血管活性药物
                        else:
                            sofa_params['vasopressor_rate'] = 0
                        
                        # 中枢神经系统评分 (GCS score)
                        if 'cns' in hourly_values and hourly_values['cns'] is not None:
                            # 假设神经系统评分为1-4分，对应GCS为15-6分
                            gcs_map = {1: 15, 2: 12, 3: 9, 4: 6}
                            sofa_params['gcs_total'] = gcs_map.get(int(hourly_values['cns']), 15)
                        else:
                            sofa_params['gcs_total'] = 0
                        
                        # 肾脏系统评分 (Creatinine and urine output)
                        if 'renal' in hourly_values and hourly_values['renal'] is not None:
                            # 假设肾脏评分为1-4分，对应肌酐为0.5-5.0 mg/dL
                            creatinine_map = {1: 0.5, 2: 1.2, 3: 2.0, 4: 5.0}
                            sofa_params['creatinine'] = creatinine_map.get(int(hourly_values['renal']), 0.5)
                            # 假设肾脏评分为1-4分，对应尿量为500-200 ml/day
                            urine_map = {1: 500, 2: 400, 3: 300, 4: 200}
                            sofa_params['urine_output_ml'] = urine_map.get(int(hourly_values['renal']), 500)
                        else:
                            sofa_params['creatinine'] = 0
                            sofa_params['urine_output_ml'] = 0
                        
                        # 打印调试信息
                        print(f"时间点 {hour} 的SOFA参数: {sofa_params}")
                        
                        # 计算当前时间点的SOFA评分
                        hourly_score = calculate_sofa_score(**sofa_params)
                        
                        # 打印调试信息
                        print(f"时间点 {hour} 的SOFA评分: {hourly_score}")
                        
                        # 存储当前时间点的评分
                        hourly_sofa_scores[hour] = hourly_score
                        hourly_sofa_totals.append(hourly_score.get('sofa_total', 0))
                    
                    # 使用最后一个时间点的评分作为最终评分
                    if hourly_sofa_scores:
                        last_hour = max(hourly_sofa_scores.keys())
                        sofa_scores = hourly_sofa_scores[last_hour]
                        sofa_total = hourly_sofa_totals[-1] if hourly_sofa_totals else 0
                else:
                    # 如果没有时间序列数据，使用原始方法计算
                    latest_values = {}
                    for feature_name, values in analysis_out.sofa_related_features.items():
                        # 使用映射将中文名称转换为英文名称
                        english_name = feature_name_mapping.get(feature_name, feature_name)
                        if values and isinstance(values, list) and len(values) > 0:
                            # 获取最后一个有效值
                            latest_val = next((v for v in reversed(values) if v is not None), None)
                            latest_values[english_name] = latest_val
                        else:
                            latest_values[english_name] = values if values is not None else None
                    
                    # 打印调试信息
                    print(f"最新特征值: {latest_values}")
                    
                    # 根据器官系统名称映射到对应的SOFA评分参数
                    # 注意：这里需要根据实际的特征值含义进行转换
                    sofa_params = {}
                    
                    # 呼吸系统评分 (PaO2/FiO2 ratio)
                    if 'respiration' in latest_values and latest_values['respiration'] is not None:
                        # 假设呼吸评分为1-4分，对应PaO2/FiO2比值为400-100
                        pao2_fio2_map = {1: 400, 2: 300, 3: 200, 4: 100}
                        sofa_params['pao2_fio2_ratio'] = pao2_fio2_map.get(int(latest_values['respiration']), 400)
                    
                    # 凝血系统评分 (Platelet count)
                    if 'coagulation' in latest_values and latest_values['coagulation'] is not None:
                        # 假设凝血评分为1-4分，对应血小板计数为150000-20000
                        platelet_map = {1: 150000, 2: 100000, 3: 50000, 4: 20000}
                        sofa_params['platelet'] = platelet_map.get(int(latest_values['coagulation']), 150000)
                    
                    # 肝脏系统评分 (Bilirubin level)
                    if 'liver' in latest_values and latest_values['liver'] is not None:
                        # 假设肝脏评分为1-4分，对应总胆红素为1.2-12 mg/dL
                        bilirubin_map = {1: 1.2, 2: 2.5, 3: 6.0, 4: 12.0}
                        sofa_params['bilirubin_total'] = bilirubin_map.get(int(latest_values['liver']), 1.2)
                    
                    # 血管系统评分 (Mean arterial pressure and vasopressor use)
                    if 'cardiovascular' in latest_values and latest_values['cardiovascular'] is not None:
                        # 假设循环评分为1-4分，对应MAP为70-30 mmHg
                        map_map = {1: 70, 2: 65, 3: 50, 4: 30}
                        sofa_params['map_value'] = map_map.get(int(latest_values['cardiovascular']), 70)
                        # 假设使用血管活性药物时评分为3-4分
                        if int(latest_values['cardiovascular']) >= 3:
                            sofa_params['vasopressor_rate'] = 0.1  # 使用血管活性药物
                        else:
                            sofa_params['vasopressor_rate'] = 0  # 不使用血管活性药物
                    
                    # 中枢神经系统评分 (GCS score)
                    if 'cns' in latest_values and latest_values['cns'] is not None:
                        # 假设神经系统评分为1-4分，对应GCS为15-6分
                        gcs_map = {1: 15, 2: 12, 3: 9, 4: 6}
                        sofa_params['gcs_total'] = gcs_map.get(int(latest_values['cns']), 15)
                    
                    # 肾脏系统评分 (Creatinine and urine output)
                    if 'renal' in latest_values and latest_values['renal'] is not None:
                        # 假设肾脏评分为1-4分，对应肌酐为0.5-5.0 mg/dL
                        creatinine_map = {1: 0.5, 2: 1.2, 3: 2.0, 4: 5.0}
                        sofa_params['creatinine'] = creatinine_map.get(int(latest_values['renal']), 0.5)
                        # 假设肾脏评分为1-4分，对应尿量为500-200 ml/day
                        urine_map = {1: 500, 2: 400, 3: 300, 4: 200}
                        sofa_params['urine_output_ml'] = urine_map.get(int(latest_values['renal']), 500)
                    
                    # 打印调试信息
                    print(f"最新SOFA参数: {sofa_params}")
                    
                    # 计算SOFA评分
                    sofa_scores = calculate_sofa_score(**sofa_params)
                    sofa_total = sofa_scores.get('sofa_total', 0)
                    
                    # 打印调试信息
                    print(f"SOFA评分: {sofa_scores}")
            except Exception as e:
                print(f"计算SOFA评分时出错: {str(e)}")
                traceback.print_exc()
                sofa_scores = {'error': str(e)}
                sofa_total = 0
         
         # 生成最终报告
        try:
            # 在生成报告前进行SOFA特征校验以辅助调试
            try:
                sofa_validation = validate_sofa_features(getattr(analysis_out, 'sofa_related_features', {}))
                debug_info['stages']['sofa_validation'] = sofa_validation
            except Exception as ve:
                debug_info['errors'].append({'stage': 'sofa_validation', 'error': str(ve)})
                traceback.print_exc()

            report_out = self.generate_clinical_report(
             current_sepsis_state_summary=risk_out.current_sepsis_state_summary,
             intervention_and_risk=intervention_and_risk,
             predicted_outcome=analysis_out.predicted_outcome,
             risk_level=analysis_out.risk_level,
             potential_risks=analysis_out.potential_risks,
             sofa_related_features=analysis_out.sofa_related_features,  # 直接使用原始预测值
             comparison_summary=comparison_out.comparison_summary,
             sofa_scores=sofa_scores,
             sofa_total=sofa_total
         )
            debug_info['stages']['generate_report'] = {
                'report_present': hasattr(report_out, 'clinical_report') and bool(report_out.clinical_report)
            }
        except Exception as e:
            debug_info['errors'].append({'stage': 'generate_report', 'error': str(e)})
            traceback.print_exc()
            # 生成简单回退报告，保证有可读输出
            simple_report = (
                f"预测结果: {getattr(analysis_out, 'predicted_outcome', '未知')}\n"
                f"风险等级: {getattr(analysis_out, 'risk_level', 'unknown')}\n"
                f"平均MSE: {getattr(comparison_out, 'average_mse', 0.0)}\n"
                f"SOFA总分: {sofa_total}\n"
                f"比较摘要: {getattr(comparison_out, 'comparison_summary', '不可用')}\n"
            )
            report_out = dspy.Prediction(reasoning='fallback', clinical_report=simple_report)

        # 返回结果
        prediction_obj = dspy.Prediction(
            reasoning_steps={
                "感染性休克风险评估": {
                    "input": {"patient_sepsis_summary": patient_sepsis_summary},
                    "output": {
                        "reasoning": risk_out.reasoning,
                        "risk_level": risk_out.risk_level,
                        "key_clinical_indicators": risk_out.key_clinical_indicators,
                        "current_sepsis_state_summary": risk_out.current_sepsis_state_summary
                    }
                },
                "分析干预措施和风险": {
                    "input": {
                        "current_sepsis_state_summary": risk_out.current_sepsis_state_summary,
                        "intervention_and_risk": intervention_and_risk
                    },
                    "output": {
                        "reasoning": analysis_out.reasoning,
                        "predicted_outcome": analysis_out.predicted_outcome,
                        "potential_risks": analysis_out.potential_risks,
                        "risk_level": analysis_out.risk_level,
                        "sofa_related_features": analysis_out.sofa_related_features
                    }
                },
                "比较SOFA相关特征趋势": {
                    "input": {
                        "predicted_sofa_features": analysis_out.sofa_related_features,
                        "actual_vital_signs": actual_vital_signs
                    },
                    "output": {
                        "reasoning": comparison_out.reasoning,
                        "parsed_actual_sofa_features": comparison_out.parsed_actual_sofa_features,
                        "mean_squared_errors": comparison_out.mean_squared_errors,
                        "average_mse": comparison_out.average_mse,
                        "comparison_summary": comparison_out.comparison_summary
                    }
                },
                "生成临床报告": {
                    "input": {
                        "current_sepsis_state_summary": risk_out.current_sepsis_state_summary,
                        "intervention_and_risk": intervention_and_risk,
                        "predicted_outcome": analysis_out.predicted_outcome,
                        "risk_level": analysis_out.risk_level,
                        "potential_risks": analysis_out.potential_risks,
                        "sofa_related_features": analysis_out.sofa_related_features,
                        "comparison_summary": comparison_out.comparison_summary,
                        "sofa_scores": sofa_scores,
                        "sofa_total": sofa_total
                    },
                    "output": {
                        "reasoning": report_out.reasoning,
                        "clinical_report": report_out.clinical_report
                    }
                }
            },
            risk_assessment=risk_out,
            intervention_analysis=analysis_out,
            vital_signs_comparison=comparison_out,
            clinical_report=report_out.clinical_report,
            average_mse=comparison_out.average_mse,
            sofa_scores=sofa_scores,
            sofa_total=sofa_total,
            hourly_sofa_scores=hourly_sofa_scores,  # 添加每小时SOFA评分
            hourly_sofa_totals=hourly_sofa_totals   # 添加每小时SOFA总分
        )
        # 附加调试信息，以便上层保存
        try:
            prediction_obj.debug_info = debug_info
        except Exception:
            pass
        return prediction_obj

class LearningData:
    def __init__(self):
        self.memory = defaultdict(list)  # 临床决策记忆库
        self.performance_metrics = {
            'risk_assessment_accuracy': [],
            'vital_signs_mse': [],
            'intervention_effectiveness': []
        }

# 在AdaptiveExperimentAgent类中修复_update_learning方法
class AdaptiveExperimentAgent(ExperimentAgent):
    def __init__(self, risk_accuracy_threshold=0.7, mse_threshold=0.2):
        super().__init__()
        self.learning_data = LearningData()
        self.adaptation_rules = {
            'high_risk_missed': self._adapt_high_risk_scenario,
            'mse_threshold': self._adapt_vital_signs_model
        }
        self.state = AgentState()
        self.learning_enabled = False  # 添加学习开关

    def apply_model_instructions(self, model_name):
        try:
            m = str(model_name).lower()
            if 'medllama2' in m:
                instr_ir = (
                    "Return strictly valid JSON with keys: reasoning, predicted_outcome, potential_risks, risk_level, risk_score, sofa_related_features. "
                    "The value of reasoning must be a non-empty English paragraph explaining step-by-step why the trends are predicted. "
                    "sofa_related_features must be a dict with exactly these 8 keys: pao2_fio2_ratio, platelet, bilirubin_total, vasopressor_rate, gcs_total, creatinine, urine_output_ml, mechanical_ventilation. "
                    "Each key maps to an array of length 8 with realistic medical numeric values for the next 8 hours. "
                    "risk_level is one of high, medium, low. risk_score is a float in [0,1]. "
                    "Do not output any text outside the JSON."
                )
                self.analyze_intervention_and_risk = dspy.ChainOfThought(
                    self.analyze_intervention_and_risk.signature.copy(instructions=instr_ir)
                )
                instr_rs = (
                    "Return strictly valid JSON with keys: reasoning, risk_level, key_clinical_indicators, current_sepsis_state_summary. "
                    "The value of reasoning must be a non-empty English paragraph. risk_level in {high, medium, low}. "
                    "key_clinical_indicators must be a list of strings."
                )
                self.shock_risk_assessment = dspy.ChainOfThought(
                    self.shock_risk_assessment.signature.copy(instructions=instr_rs)
                )
                instr_gr = (
                    "Return strictly valid JSON with keys: reasoning, clinical_report. "
                    "The value of reasoning must be a non-empty English paragraph summarizing the rationale behind the predicted outcome and risks, referencing SOFA-related features. "
                    "clinical_report must be an English multi-paragraph report that: (1) summarizes current sepsis state, (2) explains intervention and expected effect, (3) describes 8-hour SOFA-related feature trends using the exact keys pao2_fio2_ratio, platelet, bilirubin_total, vasopressor_rate, gcs_total, creatinine, urine_output_ml, mechanical_ventilation, (4) compares predicted trends against actual ICU observations from comparison_summary, and (5) provides actionable clinical recommendations. "
                    "Do not output any text outside the JSON."
                )
                self.generate_clinical_report = dspy.ChainOfThought(
                    self.generate_clinical_report.signature.copy(instructions=instr_gr)
                )
        except Exception:
            pass

    def _update_learning(self, prediction, ground_truth):
        if not self.learning_enabled:
            return
        
        try:
            # 经验存储 - 修复引用不存在的inputs属性
            case_data = {
                'patient_sepsis_summary': ground_truth,
                'risk_assessment': {
                    'risk_level': prediction.risk_assessment.risk_level,
                    'reasoning': prediction.risk_assessment.reasoning
                },
                'vital_signs_mse': prediction.average_mse if hasattr(prediction, 'average_mse') else None,
                'timestamp': datetime.now().isoformat()
            }
            self.learning_data.memory['clinical_cases'].append(case_data)
            
            # 性能评估
            risk_acc = self._calculate_risk_accuracy(prediction, ground_truth)
            self.learning_data.performance_metrics['risk_assessment_accuracy'].append(risk_acc)
            
            if hasattr(prediction, 'average_mse') and prediction.average_mse is not None:
                self.learning_data.performance_metrics['vital_signs_mse'].append(prediction.average_mse)
            
            # 记录性能
            self.state.record_performance('risk_assessment', risk_acc)
            if hasattr(prediction, 'average_mse') and prediction.average_mse is not None:
                self.state.record_performance('vital_signs', prediction.average_mse)
            
            # 递增学习周期
            self.state.increment_cycle()
            
            # 自适应调整
            if risk_acc < 0.7:
                self.adaptation_rules['high_risk_missed'](prediction)
                self.state.last_adapted = 'high_risk_missed'
            
            # 检查MSE阈值
            mse_values = self.learning_data.performance_metrics['vital_signs_mse'][-3:]
            if len(mse_values) >=3 and all(m is not None and m > 0.2 for m in mse_values):
                self.adaptation_rules['mse_threshold'](prediction)
                self.state.last_adapted = 'mse_threshold'
        except Exception as e:
            print(f"更新学习数据时出错: {str(e)}")

    def _adapt_vital_signs_model(self, prediction):
        """当SOFA特征预测MSE连续3次超过阈值时自动优化模型"""
        try:
            print("SOFA特征分析模型已自动优化")
            
            # 定义增强版签名类
            class EnhancedAnalyzeInterventionAndRisk(dspy.Signature):
                """分析干预措施的影响和风险
            特别注意以下SOFA评分相关特征：
            - 氧合指数变化趋势 (pao2_fio2_ratio)
            - 血小板计数变化趋势 (platelet)
            - 总胆红素变化趋势 (bilirubin_total)
            - 血管活性药物使用剂量变化趋势 (vasopressor_rate)
            - 格拉斯哥昏迷评分变化趋势 (gcs_total)
            - 血清肌酐变化趋势 (creatinine)
            - 尿量变化趋势 (urine_output_ml)
            - 机械通气使用情况 (mechanical_ventilation)
            
            sofa_related_features输出格式必须是字典，键为英文：
            {"pao2_fio2_ratio":[值1,值2,...], "platelet":[值1,值2,...], "bilirubin_total":[值1,值2,...], "vasopressor_rate":[值1,值2,...], "gcs_total":[值1,值2,...], "creatinine":[值1,值2,...], "urine_output_ml":[值1,值2,...], "mechanical_ventilation":[值1,值2,...]}
            
            重要：每个键的值必须是包含8个数值的列表，表示未来8小时的预测值。数值必须是合理的医学数值，不能为0或空。
                    """
                current_sepsis_state_summary: str = dspy.InputField()
                intervention_and_risk: str = dspy.InputField()
                reasoning: str = dspy.OutputField(desc="Chain-of-thought reasoning")
                predicted_outcome: str = dspy.OutputField()
                potential_risks: List[str] = dspy.OutputField()
                risk_level: str = dspy.OutputField()
                sofa_related_features: Dict[str, List[float]] = dspy.OutputField(desc="预测未来的SOFA评分相关特征趋势，每小时一个数据点，必须包含pao2_fio2_ratio, platelet, bilirubin_total, vasopressor_rate, gcs_total, creatinine, urine_output_ml, mechanical_ventilation这8个键，每个键的值必须是包含8个数值的列表")
            
            self.analyze_intervention_and_risk = dspy.ChainOfThought(EnhancedAnalyzeInterventionAndRisk)
        except Exception as e:
            print(f"优化SOFA特征模型时出错: {str(e)}")
    
    # 添加_adapt_high_risk_scenario方法
    def _adapt_high_risk_scenario(self, prediction):
        """当高风险评估不准确时自动优化高风险场景识别模型"""
        try:
            print("高风险场景识别模型已自动优化")
            
            # 定义增强版签名类，保持与原始签名类的输入参数一致
            class EnhancedSepsisShockRiskAssessment(dspy.Signature):
                """评估感染性休克风险（必须返回JSON格式）
        特别注意以下高风险指标（这些指标需要优先考虑）：
        - 收缩压 < 90 mmHg 或平均动脉压 < 65 mmHg
        - 乳酸水平 > 4 mmol/L 或持续升高
        - 严重的意识障碍（格拉斯哥昏迷量表 < 9分）
        - 呼吸频率 > 30次/分钟且氧合指数 < 200
        - 尿量 < 0.5 ml/kg/h 持续6小时以上
        - 血小板计数 < 80,000/μL或下降50%以上
        
        输出格式要求：
        - risk_level必须是以下之一：极低风险、低风险、中等风险、高风险、极高风险
        - reasoning必须详细解释关键临床指标和风险判断理由
        - key_clinical_indicators必须包含所有异常的临床指标和生命体征
        - current_sepsis_state_summary必须是对当前状态的简要总结
                """
                patient_document: str = dspy.InputField(desc="患者的完整临床文档")
                reasoning: str = dspy.OutputField(desc="详细的风险评估推理过程")
                risk_level: str = dspy.OutputField(desc="极低风险、低风险、中等风险、高风险、极高风险")
                key_clinical_indicators: List[str] = dspy.OutputField()
                current_sepsis_state_summary: str = dspy.OutputField()
            
            # 创建新的ChainOfThought对象替换原对象
            self.shock_risk_assessment = dspy.ChainOfThought(EnhancedSepsisShockRiskAssessment)
        except Exception as e:
            print(f"优化高风险场景识别模型时出错: {str(e)}")
    
    def _calculate_risk_accuracy(self, prediction, ground_truth):
        """计算风险评估的准确率
        Args:
            prediction: 模型预测结果
            ground_truth: 真实结果
        Returns:
            float: 准确率 (1.0=完全正确, 0.0=完全错误, 0.5=无法判断)
        """
        try:
            # 获取预测的风险等级
            predicted_risk = prediction.risk_assessment.risk_level.lower() if hasattr(prediction.risk_assessment, 'risk_level') else ''
            
            # 从真实结果中提取风险等级（这里使用启发式方法）
            # 实际应用中应根据具体数据结构调整解析逻辑
            actual_high_risk_indicators = ['低血压', '休克', 'sofa总分高', '器官功能恶化', '重症']
            actual_low_risk_indicators = ['稳定', '好转', '正常', '低风险']
            
            ground_truth_text = str(ground_truth).lower()
            actual_risk = None
            
            # 检测高风险指标
            for indicator in actual_high_risk_indicators:
                if indicator in ground_truth_text:
                    actual_risk = 'high'
                    break
            
            # 检测低风险指标
            if actual_risk is None:
                for indicator in actual_low_risk_indicators:
                    if indicator in ground_truth_text:
                        actual_risk = 'low'
                        break
            
            # 根据预测和实际风险等级计算准确率
            if actual_risk is None:
                return 0.5  # 无法确定真实风险等级
            
            # 确定预测的风险等级类型
            predicted_high_risk = 'high' in predicted_risk or '极高' in predicted_risk or '中等' in predicted_risk
            predicted_low_risk = 'low' in predicted_risk or '极低' in predicted_risk
            
            # 比较预测和实际风险
            if (predicted_high_risk and actual_risk == 'high') or (predicted_low_risk and actual_risk == 'low'):
                return 1.0  # 预测正确
            else:
                return 0.0  # 预测错误
        except Exception as e:
            print(f"计算风险准确率时出错: {str(e)}")
            return 0.5  # 出错时返回中间值

    def forward(self, *args, **kwargs):
        # 在forward中检查learning参数
        if 'learning_enabled' in kwargs:
            self.learning_enabled = kwargs.pop('learning_enabled')
        
        prediction = super().forward(*args, **kwargs)
        if 'ground_truth' in kwargs and self.learning_enabled:
            self._update_learning(prediction, kwargs['ground_truth'])
        return prediction

# 新增状态管理
class AgentState:
    def __init__(self):
        self.learning_cycle = 0
        self.last_adapted = None
        self.knowledge_version = 1.0
        self.performance_history = {
            'risk_assessment': [],
            'vital_signs': []
        }
        self.learning_phases = {'exploration': 0.3, 'exploitation': 0.7}  # 探索-利用平衡
        self.current_phase = 'exploration'
    

    def record_performance(self, metric_type, value):
        if metric_type in self.performance_history:
            self.performance_history[metric_type].append(value)

    def get_performance_trend(self, metric_type):
        return np.mean(self.performance_history[metric_type][-5:]) if self.performance_history[metric_type] else 0

    def increment_cycle(self):
        self.learning_cycle += 1
        if self.learning_cycle % 10 == 0:
            self._update_knowledge_base()

    def _update_knowledge_base(self):
        self.knowledge_version += 0.1
        print(f"知识库更新至版本 {self.knowledge_version:.1f}")

def fill_missing_values(vital_signs_array):
    """使用前三个小时的平均值填充缺失的体征值"""
    filled_array = []
    for i, value in enumerate(vital_signs_array):
        if value is None:
            # 获取前三个有效值（不包括当前值）
            previous_values = []
            # 向前查找最多三个有效值
            for j in range(max(0, i-3), i):
                if vital_signs_array[j] is not None:
                    previous_values.append(vital_signs_array[j])
            if previous_values:  # 如果有有效值
                filled_value = sum(previous_values) / len(previous_values)
                filled_array.append(round(filled_value, 2))
            else:  # 如果没有前序有效值，使用0填充
                filled_array.append(0.0)
        else:
            filled_array.append(value)
    return filled_array

def preprocess_vital_signs(text):
    """预处理体征数据，提取SOFA评分相关特征并转换为数值列表
    
    要求：
    1. 只提取【ICU期间SOFA评分相关特征变化】部分的内容
    2. 输出格式为字典，键为英文，值为数值列表
    3. 必须包含以下8个键：
       - pao2_fio2_ratio: 氧合指数
       - platelet: 血小板计数
       - bilirubin_total: 总胆红素
       - vasopressor_rate: 血管活性药物使用剂量
       - gcs_total: 格拉斯哥昏迷评分
       - creatinine: 血清肌酐
       - urine_output_ml: 尿量
       - mechanical_ventilation: 机械通气使用情况
    4. 值必须是数字类型，不带单位
    5. 只返回要求的字段，不要添加额外的字段
    """
    # 检查输入是否为空或None
    if not text:
        print("输入为空，返回空结果")
        return {}
    
    # 提取【SOFA评分相关特征】部分
    match = re.search(r'【SOFA评分相关特征】(.*?)(【|$)', text, re.DOTALL)
    if not match:
        # 如果没有匹配到，尝试匹配旧格式
        match = re.search(r'【ICU期间SOFA评分相关特征变化】(.*?)(【|$)', text, re.DOTALL)
        if not match:
            print("未找到【SOFA评分相关特征】或【ICU期间SOFA评分相关特征变化】部分")
            return {}
    
    content = match.group(1).strip()
    lines = content.split('\n')
    
    # 初始化结果字典
    result = {
        'pao2_fio2_ratio': [],
        'platelet': [],
        'bilirubin_total': [],
        'vasopressor_rate': [],
        'gcs_total': [],
        'creatinine': [],
        'urine_output_ml': [],
        'mechanical_ventilation': []
    }
    
    # 调试信息
    print(f"提取的内容: {content}")
    
    # 解析每一行
    for line in lines:
        print(f"处理行: {line}")
        # 提取氧合指数
        match = re.search(r'氧合指数变化：\[(.*?)\]（单位：', line)
        if match:
            values = match.group(1).split(',')
            result['pao2_fio2_ratio'] = [float(v.strip()) for v in values if v.strip()]
            print(f"氧合指数提取结果: {result['pao2_fio2_ratio']}")
        # 提取血小板计数
        match = re.search(r'血小板计数变化：\[(.*?)\]（单位：', line)
        if match:
            values = match.group(1).split(',')
            # 血小板单位有10的三次方，因此自动除以1000保持单位统一
            result['platelet'] = [float(v.strip())/1000 for v in values if v.strip()]
            print(f"血小板计数提取结果: {result['platelet']}")
        # 提取总胆红素
        match = re.search(r'总胆红素变化：\[(.*?)\]（单位：', line)
        if match:
            values = match.group(1).split(',')
            result['bilirubin_total'] = [float(v.strip()) for v in values if v.strip()]
            print(f"总胆红素提取结果: {result['bilirubin_total']}")
        # 提取血管活性药物使用剂量
        match = re.search(r'血管活性药物使用剂量：\[(.*?)\]（单位：', line)
        if match:
            values = match.group(1).split(',')
            result['vasopressor_rate'] = [float(v.strip()) for v in values if v.strip()]
            print(f"血管活性药物使用剂量提取结果: {result['vasopressor_rate']}")
        # 提取格拉斯哥昏迷评分
        match = re.search(r'格拉斯哥昏迷评分变化：\[(.*?)\]（单位：', line)
        if match:
            values = match.group(1).split(',')
            result['gcs_total'] = [float(v.strip()) for v in values if v.strip()]
            print(f"格拉斯哥昏迷评分提取结果: {result['gcs_total']}")
        # 提取血清肌酐
        match = re.search(r'血清肌酐变化：\[(.*?)\]（单位：', line)
        if match:
            values = match.group(1).split(',')
            result['creatinine'] = [float(v.strip()) for v in values if v.strip()]
            print(f"血清肌酐提取结果: {result['creatinine']}")
        # 提取尿量
        match = re.search(r'尿量变化：\[(.*?)\]（单位：', line)
        if match:
            values = match.group(1).split(',')
            result['urine_output_ml'] = [float(v.strip()) for v in values if v.strip()]
            print(f"尿量提取结果: {result['urine_output_ml']}")
        # 提取机械通气使用情况
        match = re.search(r'机械通气使用情况：\[(.*?)\]（单位：', line)
        if match:
            values = match.group(1).split(',')
            result['mechanical_ventilation'] = [float(v.strip()) for v in values if v.strip()]
            print(f"机械通气使用情况提取结果: {result['mechanical_ventilation']}")
    
    # 调试信息
    print(f"解析结果: {result}")
    
    return result

def validate_sofa_features(features):
    """校验sofa_related_features结构与数值有效性，返回详细验证报告。
    检查点：
    - 顶层类型必须为dict
    - 必须包含8个英文键：pao2_fio2_ratio, platelet, bilirubin_total, vasopressor_rate, gcs_total, creatinine, urine_output_ml, mechanical_ventilation
    - 每个键的值应为list，元素为数值类型（int/float），允许None但会记录数量
    - 记录每个键的长度、None数量、非数字数量、最小/最大值
    """
    required_keys = [
        'pao2_fio2_ratio', 'platelet', 'bilirubin_total', 'vasopressor_rate',
        'gcs_total', 'creatinine', 'urine_output_ml', 'mechanical_ventilation'
    ]

    report = {
        'is_dict': isinstance(features, dict),
        'present_keys': [],
        'missing_keys': [],
        'lengths': {},
        'none_counts': {},
        'non_numeric_counts': {},
        'min_values': {},
        'max_values': {},
        'issues': []
    }

    if not isinstance(features, dict):
        report['issues'].append('sofa_related_features不是字典类型')
        return report

    report['present_keys'] = sorted(list(features.keys()))
    report['missing_keys'] = [k for k in required_keys if k not in features]
    if report['missing_keys']:
        report['issues'].append(f"缺失键: {', '.join(report['missing_keys'])}")

    for k in required_keys:
        v = features.get(k, [])
        if not isinstance(v, list):
            report['issues'].append(f"键 {k} 的值不是列表类型")
            report['lengths'][k] = None
            report['none_counts'][k] = None
            report['non_numeric_counts'][k] = None
            report['min_values'][k] = None
            report['max_values'][k] = None
            continue
        report['lengths'][k] = len(v)
        none_count = sum(1 for x in v if x is None)
        non_numeric_count = sum(1 for x in v if x is not None and not isinstance(x, (int, float)))
        report['none_counts'][k] = none_count
        report['non_numeric_counts'][k] = non_numeric_count
        numeric_vals = [float(x) for x in v if isinstance(x, (int, float))]
        report['min_values'][k] = (min(numeric_vals) if numeric_vals else None)
        report['max_values'][k] = (max(numeric_vals) if numeric_vals else None)
        if non_numeric_count > 0:
            report['issues'].append(f"键 {k} 包含非数值元素数量: {non_numeric_count}")
    return report


def generate_fallback_sofa_features(actual_vital_signs_text_or_dict):
    """
    当干预分析阶段的模型输出无法解析或缺失SOFA相关特征时，
    尝试从实际体征文本中提取或从字典构建一个稳健的回退结构。

    输入可以是包含中文标注段落的文本（优先解析【SOFA评分相关特征】部分），
    或者是已经包含键值的字典（将统一填充到长度为8的列表）。

    返回：包含8个键、每个键8个浮点数的字典。
    """
    required_keys = [
        'pao2_fio2_ratio', 'platelet', 'bilirubin_total', 'vasopressor_rate',
        'gcs_total', 'creatinine', 'urine_output_ml', 'mechanical_ventilation'
    ]

    # 如果传入的是字典，进行规范化
    if isinstance(actual_vital_signs_text_or_dict, dict):
        normalized = {}
        for key in required_keys:
            vals = actual_vital_signs_text_or_dict.get(key, [])
            if not isinstance(vals, list):
                vals = [0.0] * 8
            # 转为浮点并裁剪/填充到8个
            floats = []
            for v in vals:
                try:
                    floats.append(float(v))
                except Exception:
                    floats.append(0.0)
            if len(floats) < 8:
                pad = floats[-1] if floats else 0.0
                floats = floats + [pad] * (8 - len(floats))
            else:
                floats = floats[:8]
            normalized[key] = floats
        return normalized

    # 否则视为文本，尝试解析
    if isinstance(actual_vital_signs_text_or_dict, str):
        try:
            parsed = preprocess_vital_signs(actual_vital_signs_text_or_dict)
        except Exception:
            parsed = {}
        normalized = {}
        for key in required_keys:
            vals = parsed.get(key, [])
            # 如果解析不到，提供默认全0
            if not vals:
                normalized[key] = [0.0] * 8
                continue
            floats = []
            for v in vals:
                try:
                    floats.append(float(v))
                except Exception:
                    floats.append(0.0)
            if len(floats) < 8:
                pad = floats[-1] if floats else 0.0
                floats = floats + [pad] * (8 - len(floats))
            else:
                floats = floats[:8]
            normalized[key] = floats
        return normalized

    # 未知类型，返回全0结构
    return {key: [0.0] * 8 for key in required_keys}

def compute_mse(predicted: Dict[str, List[float]], actual: Dict[str, List[float]]):
    """本地计算每个SOFA特征的MSE及平均MSE，返回详细结果。
    输入字典必须包含8个键，每个键的值为长度为8的数值列表。
    若缺失或长度不符，内部会进行裁剪/填充处理。
    """
    required_keys = [
        'pao2_fio2_ratio', 'platelet', 'bilirubin_total', 'vasopressor_rate',
        'gcs_total', 'creatinine', 'urine_output_ml', 'mechanical_ventilation'
    ]

    def normalize_list(vals):
        if not isinstance(vals, list):
            vals = []
        floats = []
        for v in vals:
            try:
                floats.append(float(v))
            except Exception:
                floats.append(0.0)
        if len(floats) < 8:
            pad = floats[-1] if floats else 0.0
            floats = floats + [pad] * (8 - len(floats))
        else:
            floats = floats[:8]
        return floats

    mse_per_feature = {}
    reasons = []
    for key in required_keys:
        p_list = normalize_list(predicted.get(key, []))
        a_list = normalize_list(actual.get(key, []))
        # 逐元素计算平方误差
        se = [(p_list[i] - a_list[i]) ** 2 for i in range(8)]
        mse = sum(se) / 8.0
        mse_per_feature[key] = round(mse, 6)
        reasons.append(f"{key}: MSE={mse_per_feature[key]} (pred={p_list}, act={a_list})")

    avg_mse = round(sum(mse_per_feature.values()) / len(required_keys), 6)
    # 找出误差最大的特征
    worst_key = max(mse_per_feature.items(), key=lambda x: x[1])[0]
    summary = f"平均MSE={avg_mse}，误差最大特征为 {worst_key} (MSE={mse_per_feature[worst_key]})"
    reasoning = "; ".join(reasons)
    return {
        'mean_squared_errors': mse_per_feature,
        'average_mse': avg_mse,
        'comparison_summary': summary,
        'reasoning': reasoning
    }
