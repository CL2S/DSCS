import pandas as pd
import json
from datetime import datetime

def load_data(file_path):
    """加载CSV数据文件"""
    return pd.read_csv(file_path)

def format_list(data, unit="", prefix=""):
    if not data:
        return ""
    return f"{prefix}[{', '.join(data)}]（单位：{unit}）"

def pad_feature_values(feature_dict):
    """补齐特征值数量，增加空列表处理"""
    # 1. 获取非空特征的最大长度
    valid_lengths = [len(lst) for lst in feature_dict.values() 
                    if isinstance(lst, list) and len(lst) > 0]
    
    if not valid_lengths:  # 所有特征都为空
        return feature_dict
    
    max_len = max(valid_lengths)
    
    # 2. 安全处理空列表
    for key, values in feature_dict.items():
        if not isinstance(values, list):
            continue
            
        current_len = len(values)
        
        # 情况1: 空列表处理
        if current_len == 0:
            feature_dict[key] = ["N/A"] * max_len  # 使用占位符
            continue
            
        # 情况2: 长度不足补齐
        if current_len < max_len:
            pad_count = max_len - current_len
            last_value = values[-1]
            feature_dict[key] = values + [last_value] * pad_count
    
    return feature_dict

# 添加SOFA各系统评分计算函数
def calculate_sofa_respiration(pao2_fio2_ratio, mechanical_ventilation):
    """计算呼吸系统SOFA评分"""
    if mechanical_ventilation == 1:
        return 4
    elif pd.isna(pao2_fio2_ratio):
        return 0  # 或根据实际情况决定默认值
    elif pao2_fio2_ratio < 100:
        return 4
    elif pao2_fio2_ratio < 200:
        return 3
    elif pao2_fio2_ratio < 300:
        return 2
    elif pao2_fio2_ratio < 400:
        return 1
    else:
        return 0

def calculate_sofa_coagulation(platelet):
    """计算凝血系统SOFA评分"""
    if pd.isna(platelet):
        return 0
    elif platelet < 20:
        return 4
    elif platelet < 50:
        return 3
    elif platelet < 100:
        return 2
    elif platelet < 150:
        return 1
    else:
        return 0

def calculate_sofa_liver(bilirubin_total):
    """计算肝脏系统SOFA评分"""
    if pd.isna(bilirubin_total):
        return 0
    elif bilirubin_total >= 12.0:
        return 4
    elif bilirubin_total >= 6.0:
        return 3
    elif bilirubin_total >= 2.0:
        return 2
    elif bilirubin_total >= 1.2:
        return 1
    else:
        return 0

def calculate_sofa_cardiovascular(map, vasopressor_rate):
    """计算心血管系统SOFA评分"""
    if not pd.isna(vasopressor_rate) and vasopressor_rate > 0:
        return 4  # 简化处理，实际可能需要根据具体药物剂量细分
    elif pd.isna(map):
        return 0
    elif map < 70:
        return 1
    else:
        return 0

def calculate_sofa_cns(gcs_total):
    """计算中枢神经系统SOFA评分"""
    if pd.isna(gcs_total):
        return 0
    elif gcs_total <= 3:
        return 4
    elif gcs_total <= 6:
        return 3
    elif gcs_total <= 9:
        return 2
    elif gcs_total <= 12:
        return 1
    else:
        return 0

def calculate_sofa_renal(creatinine, urine_output_ml):
    """计算肾脏系统SOFA评分"""
    # 简化处理，假设urine_output_ml是24小时尿量
    if not pd.isna(urine_output_ml) and urine_output_ml < 200:
        return 4
    elif not pd.isna(creatinine):
        if creatinine >= 5.0:
            return 4
        elif creatinine >= 3.5:
            return 3
        elif creatinine >= 2.0:
            return 2
        elif creatinine >= 1.2:
            return 1
    return 0

def calculate_total_sofa(respiration_score, coagulation_score, liver_score, 
                         cardiovascular_score, cns_score, renal_score):
    """计算SOFA总评分"""
    return sum([respiration_score, coagulation_score, liver_score, 
                cardiovascular_score, cns_score, renal_score])

def fill_missing_values(data, time_steps):
    """填补缺失值，使用其他时间点的数据
    只有当一个时间点所有SOFA相关特征都不为空时，才可以用它填补另一个时间点
    特殊处理：尿量如果为0则视为缺失
    """
    # 定义SOFA相关的特征列表
    sofa_related_features = [
        'pao2_fio2_ratio', 'mechanical_ventilation', 
        'platelet', 'bilirubin_total', 
        'map', 'vasopressor_rate', 
        'gcs_total', 'creatinine', 'urine_output_ml'
    ]
    
    # 首先检查是否需要处理
    minus1_data = data[data['time_step'] == -1].copy()  # 创建副本以避免修改原始数据
    zero_data = data[data['time_step'] == 0].copy()     # 创建副本以避免修改原始数据
    
    # 特殊处理：将尿量为0的值视为缺失值（NaN）
    if 'urine_output_ml' in minus1_data.columns:
        minus1_data.loc[minus1_data['urine_output_ml'] == 0, 'urine_output_ml'] = float('nan')
    if 'urine_output_ml' in zero_data.columns:
        zero_data.loc[zero_data['urine_output_ml'] == 0, 'urine_output_ml'] = float('nan')
    
    has_minus1_data = not minus1_data.empty
    has_zero_data = not zero_data.empty
    
    # 如果-1和0时间点都没有数据，返回None表示放弃该数据
    if not has_minus1_data and not has_zero_data:
        return None
    
    # 检查-1时间点是否所有SOFA相关特征都不为空
    minus1_valid = False
    if has_minus1_data:
        # 确保只检查数据中存在的SOFA相关特征
        available_sofa_features = [col for col in sofa_related_features if col in minus1_data.columns]
        minus1_valid = not minus1_data[available_sofa_features].isna().any().any()
    
    # 检查0时间点是否所有SOFA相关特征都不为空
    zero_valid = False
    if has_zero_data:
        # 确保只检查数据中存在的SOFA相关特征
        available_sofa_features = [col for col in sofa_related_features if col in zero_data.columns]
        zero_valid = not zero_data[available_sofa_features].isna().any().any()
    
    # 如果两个时间点都不满足所有SOFA相关特征不为空的条件，返回None
    if not minus1_valid and not zero_valid:
        return None
    
    # 创建数据的副本以避免修改原始数据
    data_copy = data.copy()
    
    # 如果-1时间点有效而0时间点缺失，用-1的值填充0时间点
    if minus1_valid and not data[data['time_step'] == 0].empty:
        # 先找到原始数据中的0时间点行
        zero_indices = data[data['time_step'] == 0].index
        # 然后用-1时间点的数据填充，但保持原始数据结构
        for idx in zero_indices:
            data_copy.loc[idx] = data[data['time_step'] == -1].iloc[0].copy()
            data_copy.loc[idx, 'time_step'] = 0
    elif minus1_valid and data[data['time_step'] == 0].empty:
        # 如果原始数据中没有0时间点，则添加一个
        zero_row = data[data['time_step'] == -1].iloc[0].copy()
        zero_row['time_step'] = 0
        data_copy = pd.concat([data_copy, pd.DataFrame([zero_row])], ignore_index=True)
    
    # 如果0时间点有效而-1时间点缺失，用0的值填充-1时间点
    if zero_valid and not data[data['time_step'] == -1].empty:
        # 先找到原始数据中的-1时间点行
        minus1_indices = data[data['time_step'] == -1].index
        # 然后用0时间点的数据填充，但保持原始数据结构
        for idx in minus1_indices:
            data_copy.loc[idx] = data[data['time_step'] == 0].iloc[0].copy()
            data_copy.loc[idx, 'time_step'] = -1
    elif zero_valid and data[data['time_step'] == -1].empty:
        # 如果原始数据中没有-1时间点，则添加一个
        minus1_row = data[data['time_step'] == 0].iloc[0].copy()
        minus1_row['time_step'] = -1
        data_copy = pd.concat([data_copy, pd.DataFrame([minus1_row])], ignore_index=True)
    
    return data_copy

def generate_stay_description(stay_data):
    """为单个 stay_id 生成自然语言描述（ICU之前的数据）"""
    latest_record = stay_data.iloc[0]

    # 基础信息
    stay_id = latest_record['stay_id']
    subject_id = latest_record['subject_id']
    age = latest_record['age']
    gender = "男性" if latest_record['gender_male'] == 1 else "女性"
    weight = latest_record.get('weight', '未记录')

    # 时间范围
    time_steps = stay_data['time_step'].tolist()
    duration_hours = max(time_steps) * 4  # 每个 step 是 4 小时
    num_measurements = len(time_steps)

    # 提取SOFA相关特征
    sofa_features = {
        'pao2_fio2_ratio': stay_data['pao2_fio2_ratio'].dropna().round(2).astype(str).tolist(),
        'mechanical_ventilation': stay_data['mechanical_ventilation'].dropna().round(2).astype(str).tolist(),  # 修改为直接输出数字
        'platelet': stay_data['platelet'].dropna().round(2).astype(str).tolist(),
        'bilirubin_total': stay_data['bilirubin_total'].dropna().round(2).astype(str).tolist(),
        'map': stay_data['map'].dropna().round(2).astype(str).tolist(),
        'vasopressor_rate': stay_data['vasopressor_rate'].dropna().round(2).astype(str).tolist(),
        'gcs_total': stay_data.get('gcs_total', pd.Series()).dropna().round(2).astype(str).tolist(),
        'creatinine': stay_data['creatinine'].dropna().round(2).astype(str).tolist(),
        'urine_output_ml': stay_data['urine_output_ml'].dropna().round(2).astype(str).tolist()
    }

    # 生命体征时间序列
    feature_dict = {
        'heart_rate': stay_data['heart_rate'].dropna().round(2).astype(str).tolist(),
        'sbp': stay_data['sbp'].dropna().round(2).astype(str).tolist(),
        'dbp': stay_data['dbp'].dropna().round(2).astype(str).tolist(),
        'temperature': stay_data['temperature'].dropna().round(2).astype(str).tolist(),
        'resp_rate': stay_data['resp_rate'].dropna().round(2).astype(str).tolist(),
        'spo2': stay_data['spo2'].dropna().round(2).astype(str).tolist(),
        'map': stay_data['map'].dropna().round(2).astype(str).tolist(),
    }

    padded_features = pad_feature_values(feature_dict)
    padded_sofa_features = pad_feature_values(sofa_features)

    # 实验室检查
    wbc = stay_data['wbc'].dropna().round(2).astype(str).tolist()
    ph = stay_data.get('ph', pd.Series()).dropna().round(2).astype(str).tolist()
    po2 = stay_data.get('po2', pd.Series()).dropna().round(2).astype(str).tolist()
    pco2 = stay_data.get('pco2', pd.Series()).dropna().round(2).astype(str).tolist()
    lactate = stay_data.get('lactate', pd.Series()).dropna().round(2).astype(str).tolist()

    # 疾病评分
    sofa_totals = stay_data['sofa_total'].dropna().round(2).astype(str).tolist()
    sirs_scores = stay_data.get('sirs_score', pd.Series()).dropna().round(2).astype(str).tolist()

    # 治疗情况
    fio2 = stay_data.get('fio2', pd.Series()).dropna().round(2).astype(str).tolist()
    iv_fluids_ml = stay_data.get('iv_fluids_ml', pd.Series()).dropna().round(2).astype(str).tolist()
    cumulative_fluid_balance = stay_data.get('cumulative_fluid_balance', pd.Series()).dropna().round(2).astype(str).tolist()

    # 生成描述文本
    description = f"""
ICU住院编号 {stay_id}，对应患者编号 {subject_id}，{age}岁{gender}，体重{weight}kg。

监测持续时间为 {duration_hours} 小时（共 {num_measurements} 次测量，每4小时一次）：

【生命体征】
{format_list(padded_features['heart_rate'], "次/分", "心率变化：")}
{format_list(padded_features['sbp'], "mmHg", "收缩压变化：")}
{format_list(padded_features['dbp'], "mmHg", "舒张压变化：")}
{format_list(padded_features['temperature'], "℃", "体温变化：")}
{format_list(padded_features['resp_rate'], "次/分", "呼吸频率变化：")}
{format_list(padded_features['spo2'], "%", "血氧饱和度变化：")}
{format_list(padded_features['map'], "mmHg", "平均动脉压变化：")}

【SOFA评分相关特征】
{format_list(padded_sofa_features['pao2_fio2_ratio'], "mmHg", "氧合指数变化：")}
{format_list(padded_sofa_features['mechanical_ventilation'], "", "机械通气使用情况：")}
{format_list(padded_sofa_features['platelet'], "×10³/μL", "血小板计数变化：")}
{format_list(padded_sofa_features['bilirubin_total'], "mg/dL", "总胆红素变化：")}
{format_list(padded_sofa_features['vasopressor_rate'], "μg/kg/min", "血管活性药物使用剂量：")}
{format_list(padded_sofa_features['gcs_total'], "分", "格拉斯哥昏迷评分变化：")}
{format_list(padded_sofa_features['creatinine'], "mg/dL", "血清肌酐变化：")}
{format_list(padded_sofa_features['urine_output_ml'], "mL", "尿量变化：")}

【疾病评分】
{format_list(sofa_totals, "分", "SOFA总分变化：")}
{format_list(sirs_scores, "分", "SIRS评分变化：")}
    """.strip()

    # 提取SOFA评分（数值类型，非字符串）
    sofa_totals_numeric = stay_data['sofa_total'].dropna().round(2).tolist()

    # 计算各系统SOFA评分
    sofa_scores = []
    for _, row in stay_data.iterrows():
        resp_score = calculate_sofa_respiration(
            row.get('pao2_fio2_ratio'), row.get('mechanical_ventilation', 0))
        coag_score = calculate_sofa_coagulation(row.get('platelet'))
        liver_score = calculate_sofa_liver(row.get('bilirubin_total'))
        cardio_score = calculate_sofa_cardiovascular(
            row.get('map'), row.get('vasopressor_rate'))
        cns_score = calculate_sofa_cns(row.get('gcs_total'))
        renal_score = calculate_sofa_renal(
            row.get('creatinine'), row.get('urine_output_ml'))
        
        total = calculate_total_sofa(
            resp_score, coag_score, liver_score, cardio_score, cns_score, renal_score)
        
        sofa_scores.append({
            'sofa_respiration': resp_score,
            'sofa_coagulation': coag_score,
            'sofa_liver': liver_score,
            'sofa_cardiovascular': cardio_score,
            'sofa_cns': cns_score,
            'sofa_renal': renal_score,
            'sofa_total': total
        })

    # 增强数据预处理
    return {
        'stay_id': stay_id,
        'subject_id': subject_id,
        'timestamp': datetime.now().isoformat(),
        'description': description,
        'sofa_total': sofa_totals_numeric,  # 确保数值类型
        'sofa_scores': sofa_scores  # 各系统SOFA评分
    }


def generate_output_summary(stay_data_output):
    # 新增：记录实际使用的时间步范围
    used_steps = sorted(stay_data_output['time_step'].unique())
    latest_record = stay_data_output.iloc[0]
    stay_id = latest_record['stay_id']
    subject_id = latest_record['subject_id']

    # 提取SOFA评分（数值类型）
    sofa_totals_numeric = stay_data_output['sofa_total'].dropna().round(2).tolist()

    # 计算ICU后各系统SOFA评分
    sofa_scores_post_icu = []
    for _, row in stay_data_output.iterrows():
        resp_score = calculate_sofa_respiration(
            row.get('pao2_fio2_ratio'), row.get('mechanical_ventilation', 0))
        coag_score = calculate_sofa_coagulation(row.get('platelet'))
        liver_score = calculate_sofa_liver(row.get('bilirubin_total'))
        cardio_score = calculate_sofa_cardiovascular(
            row.get('map'), row.get('vasopressor_rate'))
        cns_score = calculate_sofa_cns(row.get('gcs_total'))
        renal_score = calculate_sofa_renal(
            row.get('creatinine'), row.get('urine_output_ml'))
        
        total = calculate_total_sofa(
            resp_score, coag_score, liver_score, cardio_score, cns_score, renal_score)
        
        sofa_scores_post_icu.append({
            'sofa_respiration': resp_score,
            'sofa_coagulation': coag_score,
            'sofa_liver': liver_score,
            'sofa_cardiovascular': cardio_score,
            'sofa_cns': cns_score,
            'sofa_renal': renal_score,
            'sofa_total': total
        })

    # 提取SOFA相关特征
    sofa_features = {
        'pao2_fio2_ratio': stay_data_output['pao2_fio2_ratio'].dropna().round(2).astype(str).tolist(),
        'mechanical_ventilation': stay_data_output['mechanical_ventilation'].dropna().round(2).astype(str).tolist(),  # 修改为直接输出数字
        'platelet': stay_data_output['platelet'].dropna().round(2).astype(str).tolist(),
        'bilirubin_total': stay_data_output['bilirubin_total'].dropna().round(2).astype(str).tolist(),
        'map': stay_data_output['map'].dropna().round(2).astype(str).tolist(),
        'vasopressor_rate': stay_data_output['vasopressor_rate'].dropna().round(2).astype(str).tolist(),
        'gcs_total': stay_data_output.get('gcs_total', pd.Series()).dropna().round(2).astype(str).tolist(),
        'creatinine': stay_data_output['creatinine'].dropna().round(2).astype(str).tolist(),
        'urine_output_ml': stay_data_output['urine_output_ml'].dropna().round(2).astype(str).tolist()
    }

    # 生命体征时间序列（提取ICU后的记录）
    feature_dict = {
        'heart_rate': stay_data_output['heart_rate'].dropna().round(2).astype(str).tolist(),
        'sbp': stay_data_output['sbp'].dropna().round(2).astype(str).tolist(),
        'dbp': stay_data_output['dbp'].dropna().round(2).astype(str).tolist(),
        'temperature': stay_data_output['temperature'].dropna().round(2).astype(str).tolist(),
        'resp_rate': stay_data_output['resp_rate'].dropna().round(2).astype(str).tolist(),
        'spo2': stay_data_output['spo2'].dropna().round(2).astype(str).tolist(),
        'map': stay_data_output['map'].dropna().round(2).astype(str).tolist()
    }

    padded_features = pad_feature_values(feature_dict)
    padded_sofa_features = pad_feature_values(sofa_features)
    
    # 构建描述文本
    description = f"""
【ICU期间生命体征变化】
{format_list(padded_features['heart_rate'], "次/分", "心率变化：")}
{format_list(padded_features['sbp'], "mmHg", "收缩压变化：")}
{format_list(padded_features['dbp'], "mmHg", "舒张压变化：")}
{format_list(padded_features['temperature'], "℃", "体温变化：")}
{format_list(padded_features['resp_rate'], "次/分", "呼吸频率变化：")}
{format_list(padded_features['spo2'], "%", "血氧饱和度变化：")}
{format_list(padded_features['map'], "mmHg", "平均动脉压变化：")}

【ICU期间SOFA评分相关特征变化】
{format_list(padded_sofa_features['pao2_fio2_ratio'], "mmHg", "氧合指数变化：")}
{format_list(padded_sofa_features['mechanical_ventilation'], "", "机械通气使用情况：")}
{format_list(padded_sofa_features['platelet'], "×10³/μL", "血小板计数变化：")}
{format_list(padded_sofa_features['bilirubin_total'], "mg/dL", "总胆红素变化：")}
{format_list(padded_sofa_features['vasopressor_rate'], "μg/kg/min", "血管活性药物使用剂量：")}
{format_list(padded_sofa_features['gcs_total'], "分", "格拉斯哥昏迷评分变化：")}
{format_list(padded_sofa_features['creatinine'], "mg/dL", "血清肌酐变化：")}
{format_list(padded_sofa_features['urine_output_ml'], "mL", "尿量变化：")}
"""

    # 提取ICU后SOFA评分（假设数据存在）
    sofa_totals_post_icu = stay_data_output['sofa_total'].dropna().round(2).tolist()

    # 修改返回数据结构中的SOFA字段注释
    return {
        'stay_id': stay_id,
        'subject_id': subject_id,
        'timestamp': datetime.now().isoformat(),
        'description': description,
        #'sofa_total': sofa_totals_numeric,  # 使用正确定义的变量
        #'sofa_total_post_icu': sofa_totals_post_icu,  # 保持数据一致性
        'sofa_scores_post_icu': sofa_scores_post_icu  # 添加ICU后各系统SOFA评分
    }

def find_complete_time_steps(stay_data_post_icu, vital_signs_columns):
    """找到最大连续完整时间步区间（从0开始）"""
    max_complete_step = -1
    # 按时间步升序排列（假设time_step为0,1,2...）
    sorted_steps = sorted(stay_data_post_icu['time_step'].unique())
    for ts in sorted_steps:
        ts_data = stay_data_post_icu[stay_data_post_icu['time_step'] == ts]
        if not ts_data[vital_signs_columns].isna().any().any():
            max_complete_step = ts  # 当前时间步完整，更新最大值
        else:
            break  # 遇到第一个缺失时间步，停止检查
    # 返回0到max_complete_step的完整时间步数据
    if max_complete_step == -1:
        return None  # 无完整时间步
    return stay_data_post_icu[stay_data_post_icu['time_step'] <= max_complete_step]

def process_dataset(csv_path, output_path):
    """处理整个数据集并生成 stay_id 描述"""
    df = load_data(csv_path)
    
    # 按 stay_id 分组
    stay_groups = df.groupby('stay_id')
    
    all_pairs = []

    for stay_id, stay_data in stay_groups:
        # 按时间排序
        stay_data_sorted = stay_data.sort_values('time_step', ascending=True)

        # 获取 ICU 前所有数据
        stay_data_pre_icu = stay_data_sorted[stay_data_sorted['time_step'] < 0]

        # 获取 ICU 后数据
        stay_data_post_icu = stay_data_sorted[stay_data_sorted['time_step'] >= 0]

        # 检查输入或输出数据是否为空
        if stay_data_pre_icu.empty or stay_data_post_icu.empty:
            print(f"[跳过] stay_id={stay_id}：输入或输出数据为空")
            continue

        # 使用修改后的fill_missing_values函数
        # 传入-1作为pre_icu的时间点，0作为post_icu的时间点
        stay_data_pre_icu_filled = fill_missing_values(stay_data_pre_icu, [-1])
        stay_data_post_icu_filled = fill_missing_values(stay_data_post_icu, [0])
        
        # 如果任一填补结果为None，表示该患者数据应被放弃
        if stay_data_pre_icu_filled is None or stay_data_post_icu_filled is None:
            #print(f"[跳过] stay_id={stay_id}：-1和0时间点数据都缺失")
            continue

        try:
            input_desc = generate_stay_description(stay_data_pre_icu_filled)
            output_desc = generate_output_summary(stay_data_post_icu_filled)

            combined = {
                'stay_id': stay_id,
                'subject_id': input_desc['subject_id'],
                'timestamp': input_desc['timestamp'],
                'input_description': input_desc['description'],
                'output_summary': output_desc['description'],
                #'sofa_total': input_desc['sofa_total'],  # 新增
                #'sofa_total_post_icu': output_desc.get('sofa_total_post_icu', []),  # 可选
                'sofa_scores': input_desc['sofa_scores'],  # 各系统SOFA评分
                'sofa_scores_post_icu': output_desc.get('sofa_scores_post_icu', [])  # ICU后各系统SOFA评分
            }
            all_pairs.append(combined)
            print(f"[成功] stay_id={stay_id} 处理完成")
        except Exception as e:
            print(f"[错误] stay_id={stay_id} 处理失败: {e}")

    # 保存到JSON文件
    def convert_to_serializable(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        return obj

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2, default=convert_to_serializable)

    return len(all_pairs)    

def main():
    csv_path = 'ai_clinician_dataset.csv'
    output_path = 'icu_stays_descriptions88.json'

    num_stays = process_dataset(csv_path, output_path)
    print(f"成功处理了 {num_stays} 条 ICU 住院记录")
    print(f"结果已保存到 {output_path}")

if __name__ == "__main__":
    main()
