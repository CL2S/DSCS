患者标识与时间信息
​​subject_id​​: 患者唯一标识符。
​​stay_id​​: 重症监护室（ICU）住院的唯一标识符，同一患者可能多次住院。
​​hadm_id​​: 医院住院的唯一标识符。
​​time_step​​: 时间步骤编号（相对于败血症发病时间点）。负值表示败血症发病前，0为发病时刻，正值表示发病后。每个时间步长为4小时。
​​step_start_time​​: 当前时间步的起始时间。
​​step_end_time​​: 当前时间步的结束时间。
​​sepsis_onset_time​​: 败血症发病时间点（定义为怀疑感染的时间）。
人口统计学和基线特征
​​age​​: 患者年龄（岁）。
​​gender_male​​: 性别是否为男性（1表示男性，0表示女性）。
​​weight​​: 患者体重（千克），取自住院第一天。
​​icu_readmission​​: 是否ICU再入院（1表示是，0表示否）。
​​elixhauser_score​​: Elixhauser合并症指数得分（由以下合并症构成：高血压、糖尿病、充血性心力衰竭、癌症、慢性阻塞性肺病/限制性肺病、慢性肾病。每个存在为1分，不存在为0分，然后求和）。
生命体征
​​heart_rate​​: 心率（次/分钟）。
​​sbp​​: 收缩压（mmHg）。
​​dbp​​: 舒张压（mmHg）。
​​map​​: 平均动脉压（由收缩压和舒张压计算：(收缩压 + 2 * 舒张压) / 3, 单位：mmHg）。
​​resp_rate​​: 呼吸频率（次/分钟）。
​​temperature​​: 体温（摄氏度）。
​​spo2​​: 血氧饱和度（%）。
​​gcs_total​​: 格拉斯哥昏迷评分总分（范围3-15分，数值越低表示意识障碍越重）。
实验室指标
​​sodium​​: 血清钠（mmol/L）。
​​potassium​​: 血清钾（mmol/L）。
​​chloride​​: 血清氯（mmol/L）。
​​bicarbonate​​: 血清碳酸氢盐（mmol/L）。
​​calcium​​: 血清钙（mg/dL）。
​​magnesium​​: 血清镁（mg/dL）。
​​glucose​​: 血糖（mg/dL）。
​​bun​​: 尿素氮（mg/dL）。
​​creatinine​​: 血清肌酐（mg/dL）。
​​ast​​: 天门冬氨酸氨基转移酶（U/L）。
​​alt​​: 丙氨酸氨基转移酶（U/L）。
​​bilirubin_total​​: 总胆红素（mg/dL）。
​​albumin​​: 血清白蛋白（g/dL）。
​​hemoglobin​​: 血红蛋白（g/dL）。
​​wbc​​: 白细胞计数（×10³/μL）。
​​platelet​​: 血小板计数（×10³/μL）。
​​ptt​​: 部分凝血活酶时间（秒）。
​​pt​​: 凝血酶原时间（秒）。
​​inr​​: 国际标准化比值。
​​ph​​: 动脉血pH值。
​​po2​​: 动脉氧分压（mmHg）。
​​pco2​​: 动脉二氧化碳分压（mmHg）。
​​lactate​​: 乳酸（mmol/L）。
​​pao2_fio2_ratio​​: 氧合指数（PaO₂/FiO₂，单位：mmHg）。
治疗与干预
​​mechanical_ventilation​​: 是否机械通气（1表示是，0表示否）。
​​fio2​​: 吸入氧浓度（%）。
​​iv_fluids_ml​​: 4小时内静脉输液总量（毫升）。
​​vasopressor_rate​​: 血管活性药物最大使用剂量（换算为去甲肾上腺素当量，单位：μg/kg/min）。
​​urine_output_ml​​: 4小时内尿量（毫升）。
​​cumulative_fluid_balance​​: 累积液体平衡（从ICU入院到当前时间步结束时的总入量减总出量，单位：毫升）。
疾病严重程度评分
​​shock_index​​: 休克指数（心率/收缩压，无单位）。
​​sirs_score​​: 全身炎症反应综合征评分（满足以下每项1分：心率>90、呼吸>20或PaCO2<32、体温异常、WBC异常。范围0-4分）。
​​sofa_total​​: 序贯器官衰竭评估总分（由以下6个系统的子分数相加：呼吸、凝血、肝脏、心血管、中枢神经系统、肾脏。范围0-24分，分数越高表示器官功能障碍越严重）。
​​sofa_respiration​​: SOFA呼吸系统评分（0-4分）。
​​sofa_coagulation​​: SOFA凝血系统评分（0-4分）。
​​sofa_liver​​: SOFA肝脏系统评分（0-4分）。
​​sofa_cardiovascular​​: SOFA心血管系统评分（0-4分）。
​​sofa_cns​​: SOFA中枢神经系统评分（0-4分）。
​​sofa_renal​​: SOFA肾脏系统评分（0-4分）。
结局指标
​​hospital_expire_flag​​: 院内死亡标志（1表示死亡，0表示存活）。
​​mortality_90day​​: 90天死亡率（1表示在入院后90天内死亡，0表示存活）。
备注
所有特征值均为当前4小时时间窗（time_step所定义的时间段）内的平均值或汇总值，除非另有说明（如累积液体平衡是从ICU入院开始累积）。
所有实验室指标和生命体征均为数值型，可能包含空值（表示该时间段内无测量）。
治疗干预（如机械通气、血管活性药物使用）在时间窗内存在即为1，否则为0；药物剂量或液体量则为具体数值，未使用为0。