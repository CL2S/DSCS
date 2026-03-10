import json
import os
import time
import traceback
import numpy as np
from datetime import datetime
from experiment import AdaptiveExperimentAgent, configure_dspy, preprocess_vital_signs

# --- 实验执行相关函数 ---
def run_experiment(patient_data, intervention_text, enable_learning=False):
    """运行单个患者的实验"""
    try:
        print(f"  执行干预措施: {intervention_text}")
        
        # 提取必要的患者数据
        patient_sepsis_summary = patient_data['input_description']
        actual_vital_signs = patient_data['output_summary']
        
        # 创建自适应代理
        agent = AdaptiveExperimentAgent()
        
        # 运行实验
        prediction = agent.forward(
            patient_sepsis_summary=patient_sepsis_summary,
            intervention_and_risk=intervention_text,
            actual_vital_signs=actual_vital_signs,
            learning_enabled=enable_learning,
            ground_truth=patient_data  # 提供真实数据用于学习
        )
        
        # 构建结果 - 精简输出，只包含预测部分和评估报告
        result = {
            "患者ID": patient_data['stay_id'],
            "患者编号": patient_data['subject_id'],
            "干预措施": intervention_text,
            "风险等级": prediction.intervention_analysis.risk_level,
            "预测结果": prediction.intervention_analysis.predicted_outcome,
            "潜在风险": prediction.intervention_analysis.potential_risks,
            # 确保sofa_related_features包含所有必需的键
            # 格式: {"pao2_fio2_ratio":[值1,值2,...], "platelet":[值1,值2,...], "bilirubin_total":[值1,值2,...], 
            #       "vasopressor_rate":[值1,值2,...], "gcs_total":[值1,值2,...], "creatinine":[值1,值2,...], 
            #       "urine_output_ml":[值1,值2,...], "mechanical_ventilation":[值1,值2,...]}
            "预测指标数值": preprocess_vital_signs(patient_data.get('input_description', '')),
        "SOFA评分": {
            "呼吸系统": prediction.sofa_scores.get('sofa_respiration', 0),
            "凝血系统": prediction.sofa_scores.get('sofa_coagulation', 0),
            "肝脏系统": prediction.sofa_scores.get('sofa_liver', 0),
            "心血管系统": prediction.sofa_scores.get('sofa_cardiovascular', 0),
            "中枢神经系统": prediction.sofa_scores.get('sofa_cns', 0),
            "肾脏系统": prediction.sofa_scores.get('sofa_renal', 0),
            "总分": prediction.sofa_total
        },
        "每小时SOFA评分": {},  # 添加每小时SOFA评分
        "比较总结": prediction.vital_signs_comparison.comparison_summary,
        "平均均方误差": prediction.average_mse,
        "临床报告": prediction.clinical_report,
        "input_description": patient_data.get('input_description', '')  # 添加input_description字段
    }
        
        # 添加每小时SOFA评分到结果中
        if hasattr(prediction, 'hourly_sofa_scores') and prediction.hourly_sofa_scores:
            for hour, scores in prediction.hourly_sofa_scores.items():
                result["每小时SOFA评分"][hour] = {
                    "呼吸系统": scores.get('sofa_respiration', 0),
                    "凝血系统": scores.get('sofa_coagulation', 0),
                    "肝脏系统": scores.get('sofa_liver', 0),
                    "心血管系统": scores.get('sofa_cardiovascular', 0),
                    "中枢神经系统": scores.get('sofa_cns', 0),
                    "肾脏系统": scores.get('sofa_renal', 0),
                    "总分": scores.get('sofa_total', 0)
                }
        
        return result
    except Exception as e:
        print(f"  运行实验时出错: {str(e)}")
        traceback.print_exc()
        return None

def save_experiment_results(results, output_file="experiment_results.json", append=False):
    """保存实验结果到JSON文件，支持追加模式"""
    if append and os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            # 如果是列表则合并，否则创建新列表
            if isinstance(existing_results, list):
                existing_results.extend(results)
            else:
                existing_results = [existing_results] + results
            results = existing_results
        except Exception as e:
            print(f"读取现有结果失败，将覆盖文件: {str(e)}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"实验结果已保存到 {output_file}")

# --- 主函数 ---
def main():
    try:
        # 读取患者数据
        input_file = "icu_stays_descriptions88.json"
        print(f"正在读取患者数据文件: {input_file}")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                patient_data_list = json.load(f)
            print(f"成功读取患者数据，共 {len(patient_data_list)} 条记录")
        except Exception as e:
            print(f"读取患者数据文件时出错: {str(e)}")
            return
        
        # 添加学习模式开关
        enable_learning = True  # 设置为True启用自主学习
        if enable_learning:
            print("自主学习模式已启用")
        else:
            print("自主学习模式已禁用")
        
        # 存储所有实验结果
        all_results = []
        
        # 创建结果文件夹
        results_dir = "experiment_results_gemma"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        auto_mode = False
        # 对每个患者运行实验
        for i, patient_data in enumerate(patient_data_list):
            patient_results = []
            
            try:
                print(f"\n[{i+1}/{len(patient_data_list)}] 开始处理患者 {patient_data['stay_id']}")
                
                # 检查患者数据是否包含必要的字段
                required_fields = ['stay_id', 'subject_id', 'input_description', 'output_summary']
                missing_fields = [field for field in required_fields if field not in patient_data]
                if missing_fields:
                    print(f"患者 {patient_data.get('stay_id', 'unknown')} 数据缺少必要字段: {', '.join(missing_fields)}")
                    continue
                
                # 多轮干预循环
                interventions = [
                    "立即启动去甲肾上腺素输注，评估8小时内发生感染性休克的风险",
                    "立即服用抗生素，评估8小时内发生感染性休克的风险"
                ]
                while True:
                    # 获取用户输入的干预措施
                    if not auto_mode:
                        intervention_text = input("\n请输入干预措施 (输入'next'处理下一个患者，输入'auto'启用自动模式): ").strip()
                        if intervention_text.lower() == 'next':
                            break
                        if intervention_text.lower() == 'auto':
                            auto_mode = True
                            print("已启用自动模式，将对当前及后续患者执行预设干预措施")
                    
                    # auto模式下自动处理所有剩余患者
                    if auto_mode:
                        # 处理当前患者
                        for idx, intervention_text in enumerate(interventions):
                            print(f"\n[自动模式] 执行干预措施 {idx+1}: {intervention_text}")
                            # 运行实验，传递learning_enabled参数
                            result = run_experiment(patient_data, intervention_text, enable_learning)
                            if result is None:
                                print(f"患者 {patient_data['stay_id']} 实验结果为空，跳过本次干预")
                                continue
                            
                            patient_results.append(result)
                            print(f"完成患者 {patient_data['stay_id']} 的第 {idx+1} 条干预措施实验")
                            
                            # 保存单轮干预结果到文件夹
                            save_experiment_results([result], os.path.join(results_dir, f"experiment_result_{patient_data['stay_id']}.json"), append=True)
                        
                        # 处理下一个患者
                        break
                    
                    if not intervention_text:
                        print("干预措施不能为空，请重新输入")
                        continue
                    
                    # 运行实验，传递learning_enabled参数
                    result = run_experiment(patient_data, intervention_text, enable_learning)
                    if result is None:
                        print(f"患者 {patient_data['stay_id']} 实验结果为空，跳过本次干预")
                        continue
                    
                    patient_results.append(result)
                    print(f"完成患者 {patient_data['stay_id']} 的干预实验")
                    
                    # 保存单轮干预结果到文件夹
                    save_experiment_results([result], os.path.join(results_dir, f"experiment_result_{patient_data['stay_id']}.json"), append=True)
                    
                    # 打印关键结果
                    if '风险等级' in result:
                        print(f"  风险等级: {result['风险等级']}")
                    if '平均均方误差' in result:
                        print(f"  平均MSE: {result['平均均方误差']:.4f}")
                
                # 添加所有干预结果到总结果列表
                all_results.extend(patient_results)
                print(f"完成患者 {patient_data['stay_id']} 的所有干预实验")
                
                # 在处理下一个患者之前等待，避免请求过于频繁
                if i < len(patient_data_list) - 1:
                    print("  等待5秒后处理下一个患者...")
                    time.sleep(5)
                
            except Exception as e:
                print(f"处理患者 {patient_data.get('stay_id', 'unknown')} 时出错:")
                traceback.print_exc()
        
        # 保存所有实验结果
        if all_results:
            save_experiment_results(all_results, os.path.join(results_dir, "all_experiment_results.json"))
            
            # 计算平均MSE（仅包含成功解析的结果）
            valid_mses = [result['平均均方误差'] for result in all_results if '平均均方误差' in result and result['平均均方误差'] is not None]
            if valid_mses:
                avg_mse = np.mean(valid_mses)
                print(f"\n所有患者的平均MSE: {avg_mse:.4f}")
            print(f"成功处理 {len(all_results)}/{len(patient_data_list)} 条患者记录")
        else:
            print("没有成功处理任何患者记录")
    
    except Exception as e:
        print(f"运行实验时出错:")
        traceback.print_exc()

if __name__ == "__main__":
    main()