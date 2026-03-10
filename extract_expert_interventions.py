import json
import re
import os

def extract_actual_intervention_from_output_summary(output_summary: str):
    """从output_summary中提取血管活性药物使用剂量数组"""
    match = re.search(r'血管活性药物使用剂量：\[(.*?)\]', output_summary)
    if match:
        values_str = match.group(1)
        try:
            values = [float(v.strip()) for v in values_str.split(',') if v.strip()]
            return values
        except ValueError:
            return []
    else:
        return []

def generate_intervention_description(vasopressor_doses):
    """
    将血管活性药物剂量时间序列转换为临床描述
    """
    if not vasopressor_doses:
        return "无血管活性药物干预措施，建议密切监测血压和器官灌注，评估8小时内发生感染性休克的风险"
    
    non_zero_doses = [dose for dose in vasopressor_doses if dose > 0]
    
    if not non_zero_doses:
        return "无血管活性药物干预措施，建议密切监测血压和器官灌注，评估8小时内发生感染性休克的风险"
    
    initial_dose = vasopressor_doses[0] if vasopressor_doses[0] > 0 else non_zero_doses[0]
    max_dose = max(non_zero_doses)
    
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

def extract_patient_info(description):
    info = {}
    # Extract age, gender, weight
    age_match = re.search(r'(\d+)岁', description)
    gender_match = re.search(r'(男性|女性)', description)
    weight_match = re.search(r'体重(\d+\.?\d*)kg', description)
    
    if age_match:
        info['age'] = int(age_match.group(1))
    if gender_match:
        info['gender'] = gender_match.group(1)
    if weight_match:
        info['weight'] = float(weight_match.group(1))
        
    return info

def main():
    input_file = '/data/wzx/icu_stays_descriptions88.json'
    output_file = '/data/wzx/extracted_expert_interventions.json'
    
    print(f"Reading from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return

    extracted_data = []
    
    print(f"Processing {len(data)} records...")
    for entry in data:
        stay_id = entry.get('stay_id')
        subject_id = entry.get('subject_id')
        input_desc = entry.get('input_description', '')
        output_summary = entry.get('output_summary', '')
        sofa_scores_list = entry.get('sofa_scores', [])
        
        # 1. Basic Information
        patient_info = extract_patient_info(input_desc)
        patient_info['stay_id'] = stay_id
        patient_info['subject_id'] = subject_id
        
        # 2. Expert Intervention
        vasopressor_doses = extract_actual_intervention_from_output_summary(output_summary)
        intervention_desc = generate_intervention_description(vasopressor_doses)
        
        # 3. SOFA Score Changes (Total SOFA)
        sofa_totals = [s.get('sofa_total') for s in sofa_scores_list if isinstance(s, dict)]
        
        extracted_entry = {
            "patient_info": patient_info,
            "expert_intervention": {
                "description": intervention_desc,
                "vasopressor_doses": vasopressor_doses
            },
            "sofa_score_changes": {
                "sofa_totals": sofa_totals
            }
        }
        extracted_data.append(extracted_entry)

    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=2)
    
    print("Done.")

if __name__ == "__main__":
    main()
