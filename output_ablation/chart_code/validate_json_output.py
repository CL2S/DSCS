#!/usr/bin/env python3
"""
JSON输出格式和数据结构完整性验证脚本
验证预测结果文件的结构、数据类型和完整性
"""

import json
import os
from typing import Dict, List, Any

def validate_sofa_features(features: Dict[str, List[float]], feature_name: str) -> bool:
    """验证SOFA特征数据结构"""
    required_keys = [
        'pao2_fio2_ratio', 'platelet', 'bilirubin_total', 'vasopressor_rate',
        'gcs_total', 'creatinine', 'urine_output_ml', 'mechanical_ventilation'
    ]
    
    print(f"\n=== 验证 {feature_name} ===")
    
    # 检查是否为字典
    if not isinstance(features, dict):
        print(f"❌ {feature_name} 不是字典类型: {type(features)}")
        return False
    
    # 检查必需的键
    missing_keys = [key for key in required_keys if key not in features]
    if missing_keys:
        print(f"❌ {feature_name} 缺少必需的键: {missing_keys}")
        return False
    
    # 检查每个键的值
    for key in required_keys:
        value = features[key]
        if not isinstance(value, list):
            print(f"❌ {feature_name}[{key}] 不是列表类型: {type(value)}")
            return False
        
        if len(value) == 0:
            print(f"❌ {feature_name}[{key}] 是空列表")
            return False
        
        # 检查列表中的数值类型
        for i, item in enumerate(value):
            if not isinstance(item, (int, float)):
                print(f"❌ {feature_name}[{key}][{i}] 不是数值类型: {type(item)}")
                return False
        
        print(f"✓ {key}: {len(value)} 个数值")
    
    print(f"✓ {feature_name} 验证通过")
    return True

def validate_sofa_scores(scores: Dict[str, List[int]], score_name: str) -> bool:
    """验证SOFA评分数据结构"""
    required_keys = [
        'sofa_respiration', 'sofa_coagulation', 'sofa_liver', 
        'sofa_cardiovascular', 'sofa_cns', 'sofa_renal'
    ]
    
    print(f"\n=== 验证 {score_name} ===")
    
    if not isinstance(scores, dict):
        print(f"❌ {score_name} 不是字典类型: {type(scores)}")
        return False
    
    missing_keys = [key for key in required_keys if key not in scores]
    if missing_keys:
        print(f"❌ {score_name} 缺少必需的键: {missing_keys}")
        return False
    
    for key in required_keys:
        value = scores[key]
        if not isinstance(value, list):
            print(f"❌ {score_name}[{key}] 不是列表类型: {type(value)}")
            return False
        
        if len(value) == 0:
            print(f"❌ {score_name}[{key}] 是空列表")
            return False
        
        # 检查SOFA评分范围 (0-4)
        for i, score in enumerate(value):
            if not isinstance(score, int) or score < 0 or score > 4:
                print(f"❌ {score_name}[{key}][{i}] SOFA评分无效: {score}")
                return False
        
        print(f"✓ {key}: {len(value)} 个评分")
    
    print(f"✓ {score_name} 验证通过")
    return True

def validate_prediction_result(file_path: str) -> bool:
    """验证预测结果文件"""
    print(f"\n{'='*60}")
    print(f"验证文件: {os.path.basename(file_path)}")
    print(f"{'='*60}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 无法读取JSON文件: {e}")
        return False
    
    # 验证基本字段
    required_fields = [
        'patient_id', 'timestamp', 'model_name', 'model_trust_score',
        'predicted_sofa_features', 'actual_sofa_features',
        'predicted_sofa_scores', 'actual_sofa_scores'
    ]
    
    print("\n=== 验证基本字段 ===")
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        print(f"❌ 缺少必需字段: {missing_fields}")
        return False
    
    for field in required_fields:
        print(f"✓ {field}: {type(data[field])}")
    
    # 验证模型信任评分
    trust_score = data['model_trust_score']
    if not isinstance(trust_score, (int, float)) or trust_score < 0 or trust_score > 1:
        print(f"❌ 模型信任评分无效: {trust_score}")
        return False
    print(f"✓ 模型信任评分: {trust_score:.4f}")
    
    # 验证SOFA特征
    if not validate_sofa_features(data['predicted_sofa_features'], 'predicted_sofa_features'):
        return False
    
    if not validate_sofa_features(data['actual_sofa_features'], 'actual_sofa_features'):
        return False
    
    # 验证SOFA评分
    if not validate_sofa_scores(data['predicted_sofa_scores'], 'predicted_sofa_scores'):
        return False
    
    if not validate_sofa_scores(data['actual_sofa_scores'], 'actual_sofa_scores'):
        return False
    
    # 验证分析部分
    if 'analysis' in data:
        analysis = data['analysis']
        print(f"\n=== 验证分析结果 ===")
        print(f"✓ 比较特征总数: {analysis.get('total_features_compared', 'N/A')}")
        print(f"✓ 有效数据特征数: {analysis.get('features_with_valid_data', 'N/A')}")
        print(f"✓ 平均差异: {analysis.get('average_difference', 'N/A')}")
    
    print(f"\n🎉 文件 {os.path.basename(file_path)} 验证通过！")
    return True

def main():
    """主函数"""
    print("JSON输出格式和数据结构完整性验证")
    print("="*60)
    
    # 验证最新的预测结果文件
    test_files = [
        "/data/wzx/output/meditron/fact_prediction_result_30717105_meditron:7b.json",
        "/data/wzx/output/meditron/fact_prediction_result_30117609_meditron:7b.json"
    ]
    
    all_passed = True
    
    for file_path in test_files:
        if os.path.exists(file_path):
            if not validate_prediction_result(file_path):
                all_passed = False
        else:
            print(f"❌ 文件不存在: {file_path}")
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 所有JSON输出格式验证通过！")
    else:
        print("❌ 部分验证失败，请检查输出格式")
    print(f"{'='*60}")
    
    return all_passed

if __name__ == "__main__":
    main()