import os
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def transform_sofa_score(score):
    """根据规则转换SOFA评分
    若sofa值为0或1，则记为0
    若sofa值为2,3,4，则记为1
    大于4记为2
    """
    if score <= 1:
        return 0
    elif 2 <= score < 4:
        return 1
    else:
        return 2

def process_file(file_path):
    """处理单个JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    # 遍历每个干预措施
    for item in data:
        stay_id = item.get('stay_id', 'N/A')
        subject_id = item.get('subject_id', 'N/A')
        intervention = item.get('intervention', 'N/A')
        
        # 获取预测的SOFA评分
        predicted_sofa = item.get('sofa_total', [])
        
        # 获取实际的SOFA评分（在实际数据中应该有sofa_total_post_icu字段）
        actual_sofa = item.get('sofa_total_post_icu', [])
        
        # 如果任一数组为空，跳过
        if not predicted_sofa or not actual_sofa:
            continue
        
        # 确保两个数组长度相同
        min_length = min(len(predicted_sofa), len(actual_sofa))
        predicted_sofa = predicted_sofa[:min_length]
        actual_sofa = actual_sofa[:min_length]
        
        # 转换SOFA评分
        predicted_transformed = [transform_sofa_score(score) for score in predicted_sofa]
        actual_transformed = [transform_sofa_score(score) for score in actual_sofa]
        
        # 计算MAE和RMSE
        mae = mean_absolute_error(actual_transformed, predicted_transformed)
        rmse = np.sqrt(mean_squared_error(actual_transformed, predicted_transformed))
        
        results.append({
            'stay_id': stay_id,
            'subject_id': subject_id,
            'intervention': intervention,
            'predicted_sofa': predicted_sofa,
            'actual_sofa': actual_sofa,
            'predicted_transformed': predicted_transformed,
            'actual_transformed': actual_transformed,
            'mae': mae,
            'rmse': rmse
        })
    
    return results

def main():
    # 指定文件夹路径
    folder_path = '/data/wzx/experiment_results_gemma'
    
    # 存储所有结果
    all_results = []
    
    # 存储每种干预措施的结果
    intervention_results = {}
    
    # 遍历文件夹中的所有JSON文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json') and not filename.startswith('.'):
            file_path = os.path.join(folder_path, filename)
            print(f"处理文件: {file_path}")
            results = process_file(file_path)
            all_results.extend(results)
            
            # 按干预措施分类结果
            for result in results:
                intervention = result['intervention']
                if intervention not in intervention_results:
                    intervention_results[intervention] = []
                intervention_results[intervention].append(result)
    
    # 输出每种干预措施的结果
    print("\n按干预措施分类的结果汇总:")
    print("=" * 80)
    for intervention, results in intervention_results.items():
        print(f"\n干预措施: {intervention}")
        print("-" * 80)
        for result in results:
            print(f"住院编号: {result['stay_id']}, 患者编号: {result['subject_id']}")
            print(f"预测SOFA评分: {result['predicted_sofa']}")
            print(f"实际SOFA评分: {result['actual_sofa']}")
            print(f"转换后的预测评分: {result['predicted_transformed']}")
            print(f"转换后的实际评分: {result['actual_transformed']}")
            print(f"MAE: {result['mae']:.4f}, RMSE: {result['rmse']:.4f}")
            print("-" * 40)
        
        # 计算该干预措施的总体指标
        if results:
            overall_mae = np.mean([r['mae'] for r in results])
            overall_rmse = np.mean([r['rmse'] for r in results])
            print(f"\n{intervention} 的总体指标:")
            print(f"平均MAE: {overall_mae:.4f}")
            print(f"平均RMSE: {overall_rmse:.4f}")
    
    # 计算所有结果的总体指标
    if all_results:
        overall_mae = np.mean([r['mae'] for r in all_results])
        overall_rmse = np.mean([r['rmse'] for r in all_results])
        print(f"\n所有干预措施的总体指标:")
        print(f"平均MAE: {overall_mae:.4f}")
        print(f"平均RMSE: {overall_rmse:.4f}")
    
    # 将结果保存到文件
    output_file = '/data/wzx/sofa_metrics_results_gemma.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_file}")

if __name__ == "__main__":
    main()