import numpy as np
from sklearn.metrics import mean_squared_error
import subprocess
import os
import json
from datetime import datetime
import re
import glob

def calculate_vital_signs_confidence(predicted_vital_signs):
    """
    基于预测的生命体征数据计算置信度评分
    
    参数:
    predicted_vital_signs (dict): 包含预测生命体征的字典
    
    返回:
    float: 0-1范围的置信度评分
    """
    
    # 定义正常生命体征范围
    normal_ranges = {
        "blood_pressure": (80, 120),  # 收缩压
        "heart_rate": (60, 100),      # 心率
        "respiratory_rate": (12, 20), # 呼吸频率
        "temperature": (36.1, 37.2)   # 体温
    }
    
    # 计算每个生命体征在正常范围内的比例
    scores = []
    
    for vital_sign, (min_val, max_val) in normal_ranges.items():
        if vital_sign in predicted_vital_signs:
            values = predicted_vital_signs[vital_sign]
            if isinstance(values, list) and len(values) > 0:
                # 计算在正常范围内的值的比例
                in_range_count = sum(1 for v in values if min_val <= v <= max_val)
                in_range_ratio = in_range_count / len(values)
                scores.append(in_range_ratio)
    
    # 如果没有有效的生命体征数据，返回较低的置信度
    if not scores:
        return 0.0
    
    # 计算平均置信度
    avg_confidence = sum(scores) / len(scores)
    
    # 考虑趋势因素：评估是否有改善趋势
    trend_score = 0.0
    if "blood_pressure" in predicted_vital_signs and len(predicted_vital_signs["blood_pressure"]) > 1:
        bp_values = predicted_vital_signs["blood_pressure"]
        # 检查是否有上升趋势
        if bp_values[-1] > bp_values[0]:
            trend_score += 0.1  # 血压改善加分
    
    if "heart_rate" in predicted_vital_signs and len(predicted_vital_signs["heart_rate"]) > 1:
        hr_values = predicted_vital_signs["heart_rate"]
        # 检查是否有下降趋势（心率下降通常是好的）
        if hr_values[-1] < hr_values[0]:
            trend_score += 0.1  # 心率改善加分
    
    if "respiratory_rate" in predicted_vital_signs and len(predicted_vital_signs["respiratory_rate"]) > 1:
        rr_values = predicted_vital_signs["respiratory_rate"]
        # 检查是否有下降趋势（呼吸频率下降通常是好的）
        if rr_values[-1] < rr_values[0]:
            trend_score += 0.1  # 呼吸频率改善加分
    
    # 综合评分（限制在0-1之间）
    final_confidence = min(1.0, avg_confidence + trend_score)
    
    return final_confidence

def evaluate_with_ollama(model_name, input_description, intervention, predicted_sofa_features):
    """
    使用Ollama框架的大模型对预测的SOFA相关指标进行评估
    
    参数:
    model_name (str): 大模型名称
    input_description (str): 患者信息描述
    intervention (str): 干预措施
    predicted_sofa_features (dict): 预测的SOFA相关指标数据
    
    返回:
    dict: 包含评估结果的字典
    """
    # 构造提示词的基础部分
    prompt_parts = [
        "请根据以下信息评估预测的SOFA相关指标数据的合理性：\n",
        f"患者ICU入院前信息：\n{input_description}\n"
    ]
    
    # 只有当干预措施非空时才添加
    if intervention:
        prompt_parts.append(f"计划实施的干预措施：\n{intervention}\n")
    
    # 只有当预测的SOFA特征非空时才添加
    if predicted_sofa_features:
        prompt_parts.append(f"预测的干预后SOFA相关指标变化：\n{json.dumps(predicted_sofa_features, indent=2, ensure_ascii=False) if isinstance(predicted_sofa_features, dict) else predicted_sofa_features}\n")
    
    # 添加分析要求
    if "medllama2" in model_name.lower():
        prompt_parts.append("""
    Please analyze the rationality of the predicted results considering the following factors:
    1. Whether each SOFA-related indicator value is within a reasonable medical range.
    2. The logical consistency between the expected effect of the intervention and the trend of SOFA-related indicators.
    3. Evaluate the logical consistency between SOFA score components, such as the coordination of respiratory, circulatory, and renal functions.
    4. Whether the overall trend meets the expectation of clinical improvement.
    5. These are counterfactual prediction data and do not need to be compared with pre-admission data.
    6. Please pay special attention to the unity of units, ensuring all indicators use correct medical units (e.g., mmHg for blood pressure, bpm for heart rate).
    7. Consider the rationality of time-series data, checking if the magnitude of change between adjacent time points is clinically realistic (avoiding unreasonable mutations).
    8. Evaluate if the numerical precision is reasonable (e.g., temperature to 1 decimal place, blood pressure as integers).
    9. Please provide a confidence score in the range of 0-1, along with detailed reasoning. 
    
    IMPORTANT: You MUST output the confidence score strictly in this format at the very beginning:
    Confidence Score: [value between 0 and 1]
    
    Example:
    Confidence Score: 0.85
    
    Analysis Reasoning:
    ...
    """)
    else:
        prompt_parts.append("""
    请分析预测结果的合理性，考虑以下因素：
    1. 各项SOFA相关指标数值是否在合理的医学范围内
    2. 干预措施的预期效果与SOFA相关指标变化趋势的逻辑一致性
    3. 评估SOFA评分各组件之间的逻辑一致性，例如呼吸功能、循环功能、肾功能等各系统指标的协调性
    4. 整体趋势是否符合临床改善的预期
    5. 这些是预测的反事实数据，不需要与入院前数据进行比较
    6. 请特别注意数据的单位统一性，确保所有指标使用正确的医学单位（如血压为mmHg，心率为bpm等）
    7. 考虑时间序列数据的合理性，检查相邻时间点之间的变化幅度是否符合临床实际（避免不合理的突变）
    8. 评估数值的精确度是否合理（例如，体温精确到小数点后1位，血压整数即可）
    9. 请提供一个0-1范围的置信度评分，以及详细的分析理由。格式为：置信度评分：[0-1之间的数值]，请将置信度评分放在最前面，并确保数值在0-1之间
    """)
    
    prompt = "".join(prompt_parts)
    
    # 构造Ollama命令
    try:
        # 使用ollama run命令
        cmd = ["ollama", "run", model_name, prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5分钟超时
        
        if result.returncode == 0:
            # 返回模型的输出
            return {
                "success": True,
                "model_output": result.stdout.strip(),
                "error": None
            }
        else:
            return {
                "success": False,
                "model_output": None,
                "error": f"Ollama命令执行失败: {result.stderr}"
            }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "model_output": None,
            "error": "Ollama命令执行超时"
        }
    except Exception as e:
        return {
            "success": False,
            "model_output": None,
            "error": f"执行Ollama命令时出错: {str(e)}"
        }

def extract_patient_id(input_description, patient_data=None):
    """
    从input_description中提取患者编号
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

    m = re.search(r'ICU住院编号\s*(\d+)', input_description)
    if m:
        return m.group(1)
    m = re.search(r'患者编号\s*(\d+)', input_description)
    if m:
        return m.group(1)
    m = re.search(r'hos-number\s*(\d+)', input_description)
    if m:
        return m.group(1)
    
    # 尝试从stay_id提取（支持英文格式）
    stay_id_match = re.search(r'stay_id\s*[:=]\s*(\d+)', input_description, re.IGNORECASE)
    if stay_id_match:
        return stay_id_match.group(1)
        
    # 尝试从subject_id提取（支持英文格式）
    subject_id_match = re.search(r'subject_id\s*[:=]\s*(\d+)', input_description, re.IGNORECASE)
    if subject_id_match:
        return subject_id_match.group(1)
        
    raise ValueError("无法从 input_description 提取患者编号")


def extract_model_confidence(model_output):
    """
    从大模型的输出中提取置信度评分
    """
    # 规范化为字符串文本，兼容字典/列表/None
    if isinstance(model_output, dict):
        if 'model_output' in model_output:
            model_output = model_output['model_output']
        else:
            try:
                model_output = json.dumps(model_output, ensure_ascii=False)
            except Exception:
                model_output = str(model_output)
    elif isinstance(model_output, (list, tuple)):
        try:
            model_output = "\n".join(str(x) for x in model_output)
        except Exception:
            model_output = str(model_output)
    elif model_output is None:
        return 0.0

    if not isinstance(model_output, str):
        try:
            model_output = str(model_output)
        except Exception:
            return 0.0

    # 查找置信度评分，要求在0-1范围内（支持多种标签）
    patterns = [
        r'置信度评分[：:]\s*([0-1]\.\d*)',
        r'置信度[：:]\s*([0-1]\.\d*)',
        r'confidence[：: ]\s*([0-1]\.\d*)',
        r'confidence_score[：: ]\s*([0-1]\.\d*)',
        r'Confidence Score: \s*([0-1]\.\d*)',
        r'Your Confidence Score: \s*([0-1]\.\d*)'
    ]
    for pat in patterns:
        m = re.search(pat, model_output, flags=re.IGNORECASE)
        if m:
            try:
                confidence = float(m.group(1))
                return max(0.0, min(1.0, confidence))
            except Exception:
                continue

    # 如果没有找到或解析失败，返回默认值
    return 0.0


def extract_key_reasons(model_output):
    """
    从大模型的输出中提取最核心的三点评分理由
    """
    reasons = []
    
    # 按段落分割模型输出
    paragraphs = model_output.split('\n')
    
    # 寻找包含分析理由的部分
    in_reasons_section = False
    for paragraph in paragraphs:
        # 检查是否有明确的分析理由标记
        if '分析理由：' in paragraph or 'Analysis Reasoning:' in paragraph:
            in_reasons_section = True
            continue
        
        # 如果在理由部分或者段落包含分析内容
        if (in_reasons_section or 
            (paragraph.strip() and len(paragraph.strip()) > 30 and 
            ('原因' in paragraph or '因为' in paragraph or '合理性' in paragraph or 
             '置信度' in paragraph or '评分' in paragraph or 
             'reason' in paragraph.lower() or 'because' in paragraph.lower() or 
             'rationality' in paragraph.lower() or 'confidence' in paragraph.lower() or 
             'score' in paragraph.lower()))):
            
            # 忽略标题行
            if not (paragraph.startswith('**') and paragraph.endswith('**')):
                # 清理段落格式
                clean_paragraph = re.sub(r'\*+', '', paragraph).strip()
                if clean_paragraph and len(clean_paragraph) > 10:
                    reasons.append(clean_paragraph)
                    # 只取前三个理由
                    if len(reasons) >= 3:
                        break
    
    # 如果没有找到足够的理由，尝试提取看起来像理由的段落
    if not reasons:
        for paragraph in paragraphs:
            if paragraph.strip() and len(paragraph.strip()) > 50:
                clean_paragraph = re.sub(r'\*+', '', paragraph).strip()
                if clean_paragraph:
                    reasons.append(clean_paragraph)
                    if len(reasons) >= 3:
                        break
    
    return reasons[:3]

def load_experiment_data(data_dir):
    """
    从指定目录加载实验数据
    
    参数:
    data_dir (str): 包含JSON文件的目录路径
    
    返回:
    list: 包含所有实验数据的列表
    """
    experiment_data = []
    
    # 查找目录中的所有JSON文件
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 如果数据是列表，添加所有元素
                if isinstance(data, list):
                    experiment_data.extend(data)
                # 如果数据是字典，直接添加
                else:
                    experiment_data.append(data)
        except Exception as e:
            print(f"加载文件 {json_file} 时出错: {e}")
    
    return experiment_data

def get_sofa_features_from_data(data_entry):
    """
    从数据条目中提取SOFA相关特征
    
    参数:
    data_entry (dict): 单个实验数据条目
    
    返回:
    dict: 包含SOFA相关特征的字典
    """
    if "预测指标数值" in data_entry:
        return data_entry["预测指标数值"]
    return {}

def select_model():
    """
    选择评估模型
    
    返回:
    str: 模型名称
    """
    print("请选择评估模型：")
    print("1. deepseek-r1:32b (默认)")
    print("2. gemma3:12b")
    print("3. 其他模型")
    
    choice = input("请输入选项编号 (默认为1): ").strip()
    
    if choice == "2":
        return "gemma3:12b"
    elif choice == "3":
        model_name = input("请输入模型名称: ").strip()
        return model_name if model_name else "deepseek-r1:32b"
    elif choice == "1" or choice == "":
        return "deepseek-r1:32b"
    else:
        # 如果用户直接输入了模型名称
        if choice:
            return choice
        else:
            return "deepseek-r1:32b"

def save_evaluation_report(
    model_name,
    input_description,
    intervention,
    predicted_vital_signs,
    confidence,
    ollama_result,
    prediction_model_name=None,
    evaluator_trust=None,
    prediction_trust=None,
):
    """
    保存评估报告到JSON文件
    """
    # 确保output目录存在
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 提取患者ID
    patient_id = extract_patient_id(input_description)
    
    # 创建模型特定的子目录
    model_dir = os.path.join(output_dir, model_name.split(':')[0])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 构造文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluator_{patient_id}_{model_name.replace(':', '_')}_{timestamp}.json"
    filepath = os.path.join(model_dir, filename)
    
    # 确保model_output是字符串类型
    model_output = ollama_result["model_output"] if ollama_result["success"] else ""
    print(f"调试信息 - save_evaluation_report函数中的model_output类型: {type(model_output)}")
    print(f"调试信息 - save_evaluation_report函数中的model_output内容预览: {repr(model_output)[:100]}")
    
    if not isinstance(model_output, str):
        model_output = str(model_output)
    
    # 从大模型输出中提取置信度评分
    model_confidence = extract_model_confidence(model_output)
    
    # 构造报告内容
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "prediction_model_name": prediction_model_name,
        "patient_id": patient_id,
        "input_description": input_description,
        "intervention": intervention,
        "evaluator_trust": evaluator_trust,
        "prediction_trust": prediction_trust,
        "predicted_vital_signs": predicted_vital_signs,
        "confidence_score": float(model_confidence),  # 使用从大模型输出中提取的置信度评分
        "evaluation": ollama_result,
        "key_reasons": extract_key_reasons(model_output)
    }
    
    # 保存到文件
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"评估报告已保存到: {filepath}")

def main():
    """
    主函数：处理从JSON文件读取数据并进行评估的流程
    """
    # 默认数据目录
    data_dir = "experiment_results_gemma"
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误：数据目录 {data_dir} 不存在")
        return
    
    # 加载实验数据
    experiment_data = load_experiment_data(data_dir)
    
    if not experiment_data:
        print("错误：未找到实验数据")
        return
    
    # 选择评估模型
    model_name = select_model()
    
    # 处理每个数据条目
    for i, data_entry in enumerate(experiment_data):
        print(f"\n处理第 {i+1}/{len(experiment_data)} 个数据条目")
        
        try:
            # 提取输入描述和干预措施
            input_description = data_entry.get("input_description", "")
            intervention = data_entry.get("干预措施", "")
            
            # 提取SOFA相关特征
            sofa_features = get_sofa_features_from_data(data_entry)
            
            if not sofa_features:
                print("警告：未找到SOFA相关特征，跳过该条目")
                continue
            
            # 计算置信度
            confidence = calculate_vital_signs_confidence(sofa_features)
            
            # 使用Ollama进行评估
            ollama_result = evaluate_with_ollama(model_name, input_description, intervention, sofa_features)
            
            # 保存评估报告
            save_evaluation_report(model_name, input_description, intervention, sofa_features, confidence, ollama_result)
            
            print(f"已完成第 {i+1} 个数据条目的评估")
        except Exception as e:
            print(f"处理第 {i+1} 个数据条目时出错: {str(e)}")
            print("跳过该条目并继续处理下一个")
            continue
    
    print(f"\n所有 {len(experiment_data)} 个数据条目处理完成")
