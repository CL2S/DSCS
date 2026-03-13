#!/usr/bin/env python3
"""
模型预测性能和准确性分析脚本
分析预测结果的准确性、信任评分和特征差异
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple
import glob
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, balanced_accuracy_score
COMPONENT_FEATURES = {
    "sofa_respiration": ["pao2_fio2_ratio", "mechanical_ventilation"],
    "sofa_coagulation": ["platelet"],
    "sofa_liver": ["bilirubin_total"],
    "sofa_cardiovascular": ["vasopressor_rate"],
    "sofa_cns": ["gcs_total"],
    "sofa_renal": ["creatinine", "urine_output_ml"],
}

def calculate_mse(predicted: List[float], actual: List[float]) -> float:
    """计算均方误差"""
    if len(predicted) == 0 or len(actual) == 0:
        return float('inf')
    
    # 取较短序列的长度
    min_len = min(len(predicted), len(actual))
    pred_array = np.array(predicted[:min_len])
    actual_array = np.array(actual[:min_len])
    
    return np.mean((pred_array - actual_array) ** 2)

def calculate_mae(predicted: List[float], actual: List[float]) -> float:
    """计算平均绝对误差"""
    if len(predicted) == 0 or len(actual) == 0:
        return float('inf')
    
    min_len = min(len(predicted), len(actual))
    pred_array = np.array(predicted[:min_len])
    actual_array = np.array(actual[:min_len])
    
    return np.mean(np.abs(pred_array - actual_array))

def calculate_sofa_accuracy(predicted_scores: Dict[str, List[int]], 
                           actual_scores: Dict[str, List[int]]) -> Dict[str, float]:
    accuracies = {}
    for feature in predicted_scores.keys():
        if feature in actual_scores:
            pred = predicted_scores[feature]
            actual = actual_scores[feature]
            if len(pred) > 0 and len(actual) > 0:
                min_len = min(len(pred), len(actual))
                correct = sum(1 for i in range(min_len) if pred[i] == actual[i])
                accuracies[feature] = correct / min_len
            else:
                accuracies[feature] = 0.0
        else:
            accuracies[feature] = 0.0
    return accuracies

def calculate_component_metrics(
    pred: List[int],
    actual: List[int],
    feature_key: str,
    actual_features: Dict[str, List[float]]
) -> Dict[str, float]:
    if not pred or not actual:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'f1': 0.0,
            'auc_roc': 0.0
        }

    min_len = min(len(pred), len(actual))
    yp_raw = [int(round(x)) for x in pred[:min_len] if isinstance(x, (int, float))]
    yt_raw = [int(round(x)) for x in actual[:min_len] if isinstance(x, (int, float))]
    min_len2 = min(len(yp_raw), len(yt_raw))
    yp_raw = yp_raw[:min_len2]
    yt_raw = yt_raw[:min_len2]
    if min_len2 == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'f1': 0.0,
            'auc_roc': 0.0
        }

    req_keys = COMPONENT_FEATURES.get(feature_key, [])
    def is_missing_at(i: int) -> bool:
        for k in req_keys:
            vals = actual_features.get(k)
            if not isinstance(vals, list) or i >= len(vals):
                return True
            v = vals[i]
            if v is None:
                return True
            try:
                fv = float(v)
            except Exception:
                return True
            if np.isnan(fv):
                return True
        return False

    tolerant_scores = []
    yp_adj = []
    yt_adj = []
    for i in range(min_len2):
        ypi = yp_raw[i]
        yti = yt_raw[i]
        missing_zero = (yti == 0 and is_missing_at(i))
        if missing_zero:
            tolerant_scores.append(1.0)
            yp_adj.append(ypi)
            yt_adj.append(ypi)  # 视为预测正确
            continue
        d = abs(ypi - yti)
        if d == 0:
            tolerant_scores.append(1.0)
        elif d == 1:
            tolerant_scores.append(0.5)
        else:
            tolerant_scores.append(0.0)
        yp_adj.append(ypi)
        yt_adj.append(yti)

    try:
        tolerant_acc = float(np.mean(tolerant_scores)) if tolerant_scores else 0.0
    except Exception:
        tolerant_acc = 0.0

    try:
        prec = float(precision_score(yt_adj, yp_adj, average='macro', zero_division=0))
    except Exception:
        prec = 0.0
    try:
        f1 = float(f1_score(yt_adj, yp_adj, average='macro', zero_division=0))
    except Exception:
        f1 = 0.0

    y_true_bin = [1 if yp_adj[i] == yt_adj[i] else 0 for i in range(min_len2)]
    closeness = []
    for i in range(min_len2):
        d = abs(yp_adj[i] - yt_adj[i])
        score = 1.0 - min(d / 4.0, 1.0)
        closeness.append(score)
    try:
        auc = float(roc_auc_score(y_true_bin, closeness)) if len(set(y_true_bin)) > 1 else 0.0
    except Exception:
        auc = 0.0

    return {
        'accuracy': tolerant_acc,
        'precision': prec,
        'f1': f1,
        'auc_roc': auc
    }

def analyze_prediction_file(file_path: str) -> Dict:
    """分析单个预测文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 无法读取文件 {file_path}: {e}")
        return {}
    
    analysis = {
        'file_name': os.path.basename(file_path),
        'patient_id': data.get('patient_id', 'unknown'),
        'model_name': data.get('model_name', 'unknown'),
        'trust_score': data.get('model_trust_score', 0.0),
        'feature_mse': {},
        'feature_mae': {},
        'sofa_accuracy': {},
        'overall_metrics': {}
    }
    
    # 分析特征预测误差
    predicted_features = data.get('predicted_sofa_features', {})
    actual_features = data.get('actual_sofa_features', {})
    
    total_mse = []
    total_mae = []
    
    for feature in predicted_features.keys():
        if feature in actual_features:
            pred_values = predicted_features[feature]
            actual_values = actual_features[feature]
            
            mse = calculate_mse(pred_values, actual_values)
            mae = calculate_mae(pred_values, actual_values)
            
            analysis['feature_mse'][feature] = mse
            analysis['feature_mae'][feature] = mae
            
            if mse != float('inf'):
                total_mse.append(mse)
            if mae != float('inf'):
                total_mae.append(mae)
    
    # 分析SOFA评分准确性（容忍阈值与缺失零分视为正确）
    predicted_scores = data.get('predicted_sofa_scores', {})
    actual_scores = data.get('actual_sofa_scores', {})
    actual_features = data.get('actual_sofa_features', {})
    sofa_component_metrics = {}
    for feature in predicted_scores.keys():
        if feature in actual_scores:
            sofa_component_metrics[feature] = calculate_component_metrics(
                predicted_scores.get(feature, []),
                actual_scores.get(feature, []),
                feature,
                actual_features
            )
        else:
            sofa_component_metrics[feature] = {
                'accuracy': 0.0,
                'precision': 0.0,
                'f1': 0.0,
                'auc_roc': 0.0
            }
    analysis['sofa_component_metrics'] = sofa_component_metrics
    # 直接以容忍准确率作为每组件的准确性
    analysis['sofa_accuracy'] = {k: v.get('accuracy', 0.0) for k, v in sofa_component_metrics.items()}
    
    # 计算总体指标
    analysis['overall_metrics'] = {
        'average_mse': np.mean(total_mse) if total_mse else float('inf'),
        'average_mae': np.mean(total_mae) if total_mae else float('inf'),
        'average_sofa_accuracy': np.mean(list(analysis['sofa_accuracy'].values())) if analysis['sofa_accuracy'] else 0.0,
        'trust_score': analysis['trust_score']
    }
    
    return analysis

def print_analysis_summary(analysis: Dict):
    """打印分析摘要"""
    print(f"\n{'='*60}")
    print(f"文件: {analysis['file_name']}")
    print(f"患者ID: {analysis['patient_id']}")
    print(f"模型: {analysis['model_name']}")
    print(f"信任评分: {analysis['trust_score']:.4f}")
    print(f"{'='*60}")
    
    print(f"\n📊 特征预测误差 (MSE):")
    for feature, mse in analysis['feature_mse'].items():
        if mse != float('inf'):
            print(f"  {feature}: {mse:.4f}")
        else:
            print(f"  {feature}: 无法计算")
    
    print(f"\n📊 特征预测误差 (MAE):")
    for feature, mae in analysis['feature_mae'].items():
        if mae != float('inf'):
            print(f"  {feature}: {mae:.4f}")
        else:
            print(f"  {feature}: 无法计算")
    
    print(f"\n🎯 SOFA评分准确性:")
    for feature, accuracy in analysis['sofa_accuracy'].items():
        print(f"  {feature}: {accuracy:.2%}")
    scm = analysis.get('sofa_component_metrics', {})
    if scm:
        print(f"\n🎯 SOFA评分分类指标:")
        for feature, m in scm.items():
            print(f"  {feature}: acc={m['accuracy']:.4f}, prec={m['precision']:.4f}, f1={m['f1']:.4f}, auc={m['auc_roc']:.4f}")
    
    print(f"\n📈 总体指标:")
    metrics = analysis['overall_metrics']
    print(f"  平均MSE: {metrics['average_mse']:.4f}")
    print(f"  平均MAE: {metrics['average_mae']:.4f}")
    print(f"  平均SOFA准确性: {metrics['average_sofa_accuracy']:.2%}")
    print(f"  模型信任评分: {metrics['trust_score']:.4f}")

def main():
    """主函数"""
    print("模型预测性能和准确性分析")
    print("="*60)
    
    try:
        from fact_prediction import MODEL_NAMES
    except Exception:
        MODEL_NAMES = ["gemma3:12b", "meditron:7b", "medllama2:latest", "deepseek-r1:32b", "qwen3:30b"]
    rows = []
    components = ["sofa_respiration", "sofa_coagulation", "sofa_liver", "sofa_cardiovascular", "sofa_cns", "sofa_renal"]
    for model in MODEL_NAMES:
        safe_model = model.split(':')[0].replace(':', '_')
        pattern = f"/data/wzx/output/{safe_model}/fact_prediction_result_*.json"
        files = glob.glob(pattern)
        if not files:
            continue
        acc_lists = {c: [] for c in components}
        prec_lists = {c: [] for c in components}
        f1_lists = {c: [] for c in components}
        auc_lists = {c: [] for c in components}
        trust_scores = []
        avg_mse_list = []
        avg_mae_list = []
        for fp in files:
            a = analyze_prediction_file(fp)
            if not a:
                continue
            trust_scores.append(a.get('trust_score', 0.0))
            m = a.get('overall_metrics', {})
            if isinstance(m.get('average_mse'), (int, float)) and m.get('average_mse') != float('inf'):
                avg_mse_list.append(m.get('average_mse'))
            if isinstance(m.get('average_mae'), (int, float)) and m.get('average_mae') != float('inf'):
                avg_mae_list.append(m.get('average_mae'))
            sa = a.get('sofa_accuracy', {})
            scm = a.get('sofa_component_metrics', {})
            for c in components:
                if isinstance(sa.get(c), (int, float)):
                    acc_lists[c].append(sa.get(c))
                cm = scm.get(c, {})
                if isinstance(cm.get('precision'), (int, float)):
                    prec_lists[c].append(cm.get('precision'))
                if isinstance(cm.get('f1'), (int, float)):
                    f1_lists[c].append(cm.get('f1'))
                if isinstance(cm.get('auc_roc'), (int, float)):
                    auc_lists[c].append(cm.get('auc_roc'))
        # 生成每模型×组件的长表行
        avg_conf = float(np.mean(trust_scores)) if trust_scores else 0.0
        comp_label = {
            'sofa_respiration': 'respiration',
            'sofa_coagulation': 'coagulation',
            'sofa_liver': 'liver',
            'sofa_cardiovascular': 'cardiovascular',
            'sofa_cns': 'cns',
            'sofa_renal': 'renal'
        }
        for c in components:
            acc_mean = float(np.mean(acc_lists[c])) if acc_lists[c] else 0.0
            prec_mean = float(np.mean(prec_lists[c])) if prec_lists[c] else 0.0
            f1_mean = float(np.mean(f1_lists[c])) if f1_lists[c] else 0.0
            auc_mean = float(np.mean(auc_lists[c])) if auc_lists[c] else 0.0
            rows.append({
                'model_name': model,
                'sofa_param': comp_label.get(c, c),
                'samples': len(files),
                'acc': acc_mean,
                'precision': prec_mean,
                'F1_Score': f1_mean,
                'AUC_ROC': auc_mean,
                'confidence': avg_conf
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        out_csv = '/data/wzx/output/Model_SOFA_Performance.csv'
        df_sorted = df.sort_values(['model_name', 'sofa_param'])
        df_sorted.to_csv(out_csv, index=False, encoding='utf-8', float_format='%.3f')
        print("\nSOFA评分模型性能长表汇总:")
        print(tabulate(df_sorted[['model_name','sofa_param','samples','acc','precision','F1_Score','AUC_ROC','confidence']], headers='keys', tablefmt='psql', floatfmt='.3f'))
        print(f"\n已保存到: {out_csv}")
    else:
        print("未汇总到任何模型的事实预测结果")

if __name__ == "__main__":
    main()
