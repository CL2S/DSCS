import re

def translate_patient(text):
    text = re.sub(r'ICU住院编号', 'ICU Stay ID', text)
    text = re.sub(r'对应患者编号', 'Patient ID', text)
    text = re.sub(r'(\d+)岁男性', r'\1-year-old male', text)
    text = re.sub(r'(\d+)岁女性', r'\1-year-old female', text)
    text = re.sub(r'体重([\d.]+)kg', r'weight \1kg', text)
    text = re.sub(r'【生命体征】', '\n[Vital Signs]', text)
    text = re.sub(r'心率变化', 'Heart Rate', text)
    text = re.sub(r'呼吸频率变化', 'Respiratory Rate', text)
    text = re.sub(r'血氧饱和度变化', 'SpO2', text)
    text = re.sub(r'平均动脉压变化', 'MAP', text)
    text = re.sub(r'收缩压变化', 'SBP', text)
    text = re.sub(r'舒张压变化', 'DBP', text)
    text = re.sub(r'体温变化', 'Temperature', text)
    text = re.sub(r'【SOFA评分相关特征】', '\n[SOFA Related Features]', text)
    text = re.sub(r'氧合指数变化', 'PaO2/FiO2 Ratio', text)
    text = re.sub(r'机械通气使用情况', 'Mechanical Ventilation', text)
    text = re.sub(r'血小板计数变化', 'Platelets', text)
    text = re.sub(r'总胆红素变化', 'Total Bilirubin', text)
    text = re.sub(r'格拉斯哥昏迷评分变化', 'GCS Score', text)
    text = re.sub(r'血清肌酐变化', 'Creatinine', text)
    text = re.sub(r'尿量变化', 'Urine Output', text)
    text = re.sub(r'血管活性药物使用剂量', 'Vasopressor Dose', text)
    text = re.sub(r'【疾病评分】', '\n[Disease Scores]', text)
    text = re.sub(r'SOFA总分变化', 'Total SOFA Score', text)
    text = re.sub(r'SIRS评分变化', 'SIRS Score', text)
    text = re.sub(r'（单位：(.+?)）', r'(Unit: \1)', text)
    text = re.sub(r'次/分', 'bpm', text)
    text = re.sub(r'分', 'points', text)
    text = re.sub(r'监测持续时间为 0 小时（共 7 次测量，每4小时一次）：', 'Monitoring duration 0 hours (7 measurements total, every 4 hours):', text)
    return text

def translate_intervention(text):
    replacements = [
        ("立即启动去甲肾上腺素输注", "Immediate initiation of norepinephrine infusion"),
        ("初始剂量为", "initial dose of "),
        ("维持剂量为", "maintenance dose of "),
        ("根据血压反应调整至最大", "titrated to a maximum of "),
        ("剂量调整间隔", "dose adjustment interval "),
        ("密切监测血压和器官灌注", "closely monitor blood pressure and organ perfusion"),
        ("剂量呈递增趋势", "increasing dose trend"),
        ("剂量呈递减趋势", "decreasing dose trend"),
        ("提示血管反应性下降", "indicating decreased vascular reactivity"),
        ("提示血管反应性改善", "indicating improved vascular reactivity"),
        ("评估8小时内发生感染性休克的风险", "assess risk of septic shock within 8 hours"),
        ("无血管活性药物干预措施", "No vasopressor intervention"),
        ("建议密切监测血压和器官灌注", "recommend close monitoring of blood pressure and organ perfusion"),
        ("小时", "hours"),
        ("分钟", "minutes"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text

def translate_reasoning(text):
    # Split into sentences or clauses for better matching
    replacements = [
        ("患者当前SOFA评分", "Patient's current SOFA score is "),
        ("提示多器官功能障碍", "indicating multiple organ dysfunction"),
        ("SIRS评分", "SIRS score "),
        ("显示炎症反应强烈", "shows strong inflammatory response"),
        ("存在意识障碍", "presence of consciousness disturbance"),
        ("心率呼吸增快", "increased heart and respiratory rates"),
        ("符合脓毒症高风险状态", "consistent with high risk of sepsis"),
        ("干预措施为", "Intervention is "),
        ("去甲肾上腺素初始剂量", "norepinephrine initial dose "),
        ("目标血压维持", "maintaining target blood pressure"),
        ("但剂量呈递增趋势", "but dose shows increasing trend"),
        ("提示血管活性药物需求增加", "indicating increased vasopressor requirement"),
        ("血管反应性下降", "decreased vascular reactivity"),
        ("根据临床经验", "Based on clinical experience"),
        ("血管活性药物剂量持续增加", "continued increase in vasopressor dose"),
        ("往往预示脓毒症休克进展", "often predicts progression to septic shock"),
        ("需警惕器官功能进一步恶化", "need to be alert for further organ function deterioration"),
        ("未来8小时预测需考虑", "Future 8-hour prediction needs to consider "),
        ("血管活性药物使用增加对循环稳定的影响", "impact of increased vasopressor use on circulatory stability"),
        ("以及脓毒症对多器官的持续损害", "and continuous damage of sepsis to multiple organs"),
        ("患者当前处于感染性休克状态", "Patient is currently in septic shock"),
        ("表现为持续低血压", "manifested as persistent hypotension"),
        ("心动过速", "tachycardia"),
        ("依赖血管活性药物", "dependence on vasopressors"),
        ("维持循环", "to maintain circulation"),
        ("需紧急干预", "requires urgent intervention"),
        ("基于当前干预措施", "Based on current intervention"),
        ("预计未来8小时血压和循环稳定性将逐步改善", "expect blood pressure and circulatory stability to gradually improve in next 8 hours"),
        ("SOFA评分可能下降", "SOFA score may decrease"),
        ("预测为改善趋势", "predicted as improving trend"),
        ("然而，仍需警惕", "However, still need to be alert for"),
        ("感染源控制不足", "insufficient source control"),
        ("严重心动过速", "severe tachycardia"),
        ("需血管活性药物支持", "requiring vasopressor support"),
        ("氧合指数显著降低", "significantly reduced oxygenation index"),
        ("由于", "Due to"),
        ("可能预示病情恶化", "may predict condition deterioration"),
        ("氧合指数低", "low oxygenation index"),
        ("和", "and"),
        ("表明患者处于危重状态", "indicates patient is in critical condition"),
        ("需持续监测和干预", "requires continuous monitoring and intervention"),
        ("若", "if"),
        ("但血压未明显改善", "but blood pressure does not significantly improve"),
        ("可能导致器官灌注不足", "may lead to insufficient organ perfusion"),
        ("进一步加重多器官功能障碍", "further aggravating multiple organ dysfunction"),
        ("患者当前存在低血压", "Patient currently has hypotension"),
        ("血小板减少", "thrombocytopenia"),
        ("提示感染性休克高风险", "indicating high risk of septic shock"),
        ("仅建议密切监测", "only recommend close monitoring"),
        ("未采取积极治疗手段", "no active treatment measures taken"),
        ("根据脓毒症管理指南", "According to sepsis management guidelines"),
        ("未及时使用血管活性药物", "failure to use vasopressors in time"),
        ("可能导致血压进一步下降", "may lead to further blood pressure drop"),
        ("进而", "consequently"),
        ("SOFA评分12分已属严重", "SOFA score of 12 is severe"),
        ("且无干预措施", "and with no intervention"),
        ("未来8小时风险较高", "risk is high in next 8 hours"),
        ("表现为循环衰竭", "manifested as circulatory failure"),
        ("神经系统功能受损", "nervous system impairment"),
        ("但未实施血管活性药物干预", "but no vasopressor intervention implemented"),
        ("缺乏血管活性药物支持将导致血压难以维持", "Lack of vasopressor support will make blood pressure difficult to maintain"),
        ("监测建议表明当前风险较高", "Monitoring recommendations indicate high current risk"),
        ("会显著增加病情恶化风险", "will significantly increase risk of deterioration"),
        ("预计将持续恶化", "expected to continue deteriorating"),
        ("提示未来8小时病情可能进一步加重", "suggesting condition may worsen further in next 8 hours"),
        ("意识障碍", "consciousness disturbance"),
        ("凝血功能异常", "coagulation abnormalities"),
        ("及需要血管活性药物支持", "and requiring vasopressor support"),
        ("意识障碍可能减轻", "consciousness disturbance may alleviate"),
        ("但需警惕感染持续存在", "but need to be alert for persistent infection"),
        ("器官功能进一步受损的风险", "risk of further organ function damage"),
        ("SOFA评分相关特征将呈现改善趋势", "SOFA related features will show improving trend"),
        ("但部分指标如凝血功能和尿量可能仍需时间恢复", "but some indicators like coagulation and urine output may still need time to recover"),
        ("表现为低血压", "manifested as hypotension"),
        ("SOFA评分升高", "increased SOFA score"),
        ("虽然血管活性药物剂量增加", "Although vasopressor dose increases"),
        ("但血压控制可能仍不稳定", "blood pressure control may still be unstable"),
        ("器官灌注风险高", "high risk of organ perfusion"),
        ("需密切监测", "Close monitoring required"),
        ("预测未来8小时可能继续恶化", "predicting possible continued deterioration in next 8 hours"),
        ("高剂量血管活性药物依赖", "dependence on high-dose vasopressors"),
        ("器官功能有望改善", "organ function expected to improve"),
        ("未完全控制", "not fully controlled"),
        ("分", " points"),
        ("（", "("),
        ("）", ")"),
        ("、", ", "),
        ("。", ". "),
        ("，", ", "),
        ("及", " and "),
        ("但", " but "),
        ("等", " etc."),
        ("需", " need "),
        ("此", " this "),
        ("部", " part "),
        ("相关指标", " related indicators "),
        ("如", " such as "),
        ("预期", " expected "),
        ("持续", " continuous "),
        ("恶化", " deterioration "),
        ("改善", " improvement "),
        ("稳定", " stable "),
        ("依赖性", " dependence "),
        ("风险", " risk "),
        ("可能", " may "),
        ("包括", " include "),
        ("缺乏", " lack of "),
        ("难以", " difficult to "),
        ("显著", " significantly "),
        ("增加", " increase "),
        ("减少", " decrease "),
        ("低血压", " hypotension "),
        ("感染性休克", " septic shock "),
        ("脓毒症", " sepsis "),
        ("循环衰竭", " circulatory failure "),
        ("神经系统功能受损", " nervous system impairment "),
        ("实施", " implement "),
        ("采取", " take "),
        ("积极", " active "),
        ("手段", " measures "),
        ("建议", " suggest "),
        ("表明", " indicate "),
        ("当前", " current "),
        ("状态", " status "),
        ("维持", " maintain "),
        ("依赖", " dependence "),
        ("高剂量", " high dose "),
        ("有望", " hopeful to "),
        ("时间", " time "),
        ("恢复", " recover "),
        ("由于", " due to "),
        ("导致", " lead to "),
        ("进而", " thereby "),
        ("加重", " aggravate "),
        ("不足", " insufficiency "),
        ("指南", " guidelines "),
        ("管理", " management "),
        ("及时", " timely "),
        ("使用", " use "),
        ("显著", " significant "),
        ("评估", " assess "),
        ("发生", " occurrence "),
        ("内", " within "),
        ("监测", " monitor "),
        ("反应性", " reactivity "),
        ("下降", " decrease "),
        ("呈", " show "),
        ("趋势", " trend "),
        ("密切", " closely "),
        ("间隔", " interval "),
        ("调整", " adjust "),
        ("至", " to "),
        ("最大", " maximum "),
        ("启动", " start "),
        ("输注", " infusion "),
        ("收缩压", " SBP "),
        ("心率", " HR "),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text

def main():
    with open("dumped_text.txt", "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.split("-" * 40)
    
    latex_output = []
    
    for block in blocks:
        if not block.strip():
            continue
            
        lines = block.strip().split('\n')
        data = {}
        current_key = None
        
        for line in lines:
            if line.startswith("FILE:"):
                data['FILE'] = line.replace("FILE:", "").strip()
            elif line.startswith("PATIENT:"):
                data['PATIENT'] = line.replace("PATIENT:", "").strip()
            elif line.startswith("INTERVENTION:"):
                data['INTERVENTION'] = line.replace("INTERVENTION:", "").strip()
            elif line.startswith("REASONING:"):
                data['REASONING'] = line.replace("REASONING:", "").strip()
            elif line.startswith("RISK:"):
                data['RISK'] = line.replace("RISK:", "").strip()
        
        if not data:
            continue
            
        patient_id = data.get('FILE', '').replace('result_', '').replace('.json', '')
        
        latex_block = f"\\begin{{lstlisting}}[caption={{Result for Patient {patient_id}}}, label={{lst:{patient_id}}}]\n"
        
        # Translate and append sections
        latex_block += "Patient Information:\n"
        latex_block += translate_patient(data.get('PATIENT', '')) + "\n\n"
        
        latex_block += "Intervention Plan:\n"
        latex_block += translate_intervention(data.get('INTERVENTION', '')) + "\n\n"
        
        latex_block += "Reasoning:\n"
        latex_block += translate_reasoning(data.get('REASONING', '')) + "\n\n"
        
        latex_block += "Risk Level: " + data.get('RISK', '') + "\n"
        latex_block += "\\end{lstlisting}\n\n"
        
        latex_output.append(latex_block)
        
    with open("best_result_summary.tex", "w", encoding="utf-8") as f:
        f.write("% Generated Summary of Best Results\n")
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{listings}\n")
        f.write("\\usepackage{caption}\n")
        f.write("\\begin{document}\n\n")
        f.write("\n".join(latex_output))
        f.write("\\end{document}\n")

if __name__ == "__main__":
    main()
