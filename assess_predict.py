import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def classify_sepsis_severity(sofa_score):
    """
    根据SOFA评分进行脓毒症严重程度分级
    0-1为low，2-3为medium，大于4为high
    """
    if sofa_score <= 1:
        return "low"
    elif 2 <= sofa_score <= 3:
        return "medium"
    else:  # sofa_score > 3
        return "high"

def map_risk_level(risk_level):
    """将risk_level映射到严重程度等级"""
    return risk_level

def process_file(file_path):
    """处理单个JSON文件，提取SOFA评分和risk_level进行比较"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    for item in data:
        input_description = item.get('input_description', '')
        
        sofa_scores = []
        if 'SOFA总分变化：' in input_description:
            start_idx = input_description.find('SOFA总分变化：') + len('SOFA总分变化：')
            end_idx = input_description.find('（单位：分）', start_idx)
            if end_idx == -1:
                end_idx = input_description.find('\n', start_idx)
            
            if start_idx < end_idx:
                sofa_str = input_description[start_idx:end_idx].strip()
                try:
                    sofa_str = sofa_str.replace('[', '').replace(']', '')
                    sofa_scores = [int(x.strip()) for x in sofa_str.split(',') if x.strip()]
                except:
                    print(f"解析SOFA评分时出错: {sofa_str}")
                    continue
        
        if not sofa_scores:
            print(f"未能提取到SOFA评分: {item.get('stay_id', 'N/A')}")
            continue
        
        last_sofa_score = sofa_scores[-1]
        predicted_severity = classify_sepsis_severity(last_sofa_score)
        actual_severity = item.get('risk_level', '').lower()
        
        if not actual_severity:
            print(f"缺少risk_level: {item.get('stay_id', 'N/A')}")
            continue
        
        match = 1 if predicted_severity == actual_severity else 0
        
        results.append({
            'stay_id': item.get('stay_id', 'N/A'),
            'subject_id': item.get('subject_id', 'N/A'),
            'last_sofa_score': last_sofa_score,
            'predicted_severity': predicted_severity,
            'actual_severity': actual_severity,
            'match': match
        })
    
    return results

def main():
    folder_path = '/data/wzx/experiment_results_gemma'
    all_results = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.json') and not filename.startswith('.'):
            file_path = os.path.join(folder_path, filename)
            print(f"处理文件: {file_path}")
            results = process_file(file_path)
            all_results.extend(results)
    
    if not all_results:
        print("没有提取到任何结果")
        return
    
    matches = [r['match'] for r in all_results]
    predicted_severities = [r['predicted_severity'] for r in all_results]
    actual_severities = [r['actual_severity'] for r in all_results]
    
    try:
        accuracy = accuracy_score(actual_severities, predicted_severities)
        f1 = f1_score(actual_severities, predicted_severities, average='macro', zero_division=0)
        precision = precision_score(actual_severities, predicted_severities, average='macro', zero_division=0)
        recall = recall_score(actual_severities, predicted_severities, average='macro', zero_division=0)
        f1_weighted = f1_score(actual_severities, predicted_severities, average='weighted', zero_division=0)
        
        # 准备评估指标字典
        evaluation_metrics = {
            "总患者数": len(all_results),
            "预测正确的患者数": sum(matches),
            "准确率 (Accuracy)": round(accuracy, 4),
            "F1分数 (Macro)": round(f1, 4),
            "精确率 (Precision, Macro)": round(precision, 4),
            "召回率 (Recall, Macro)": round(recall, 4),
            "F1分数 (Weighted)": round(f1_weighted, 4)
        }
        
        print("\n脓毒症严重程度预测评估结果:")
        print("=" * 50)
        for key, value in evaluation_metrics.items():
            print(f"{key}: {value}")
        
        print("\n各类别预测详情:")
        print("-" * 30)
        for r in all_results:
            match_result = '是' if r['match'] else '否'
            print(f"住院号: {r['stay_id']}, SOFA评分: {r['last_sofa_score']}, 预测等级: {r['predicted_severity']}, 实际等级: {r['actual_severity']}, 匹配: {match_result}")
        
        # 保存结果到文件，包括详细结果和评估指标
        output_file = '/data/wzx/sepsis_severity_evaluation_results_gemma.json'
        output_data = {
            "evaluation_metrics": evaluation_metrics,
            "detailed_results": all_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"计算指标时出错: {e}")
        accuracy = np.mean(matches)
        print("\n脓毒症严重程度预测评估结果:")
        print("=" * 50)
        print(f"总患者数: {len(all_results)}")
        print(f"预测正确的患者数: {sum(matches)}")
        print(f"准确率 (Accuracy): {accuracy:.4f}")

if __name__ == "__main__":
    main()