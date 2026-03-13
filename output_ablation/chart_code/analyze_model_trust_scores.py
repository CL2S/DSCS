#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, roc_auc_score, confusion_matrix
import pandas as pd
from tabulate import tabulate

# 定义输入和输出目录
# 现在从output目录下的各个模型子目录读取文件
BASE_OUTPUT_DIR = './output'
OUTPUT_FILE = '/data/wzx/output/Model_Trust_Score.json'
METRICS_OUTPUT_FILE = '/data/wzx/output/Model_Feature_Metrics.csv'


def analyze_model_trust_scores():
    """
    分析fact_prediction_results中的所有文件，提取model_name和model_trust_score，
    计算每个模型的平均信任分数，并将结果保存到output目录下的文件中。
    """
    # 创建用于存储模型分数的字典
    model_scores = defaultdict(list)
    
    # 检查基础输出目录是否存在
    if not os.path.exists(BASE_OUTPUT_DIR):
        print(f"错误: 基础输出目录 {BASE_OUTPUT_DIR} 不存在")
        return
    
    # 确保输出目录存在
    output_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历output目录下的所有模型子目录
    file_count = 0
    for model_dir in os.listdir(BASE_OUTPUT_DIR):
        model_path = os.path.join(BASE_OUTPUT_DIR, model_dir)
        if os.path.isdir(model_path):
            for filename in os.listdir(model_path):
                if filename.endswith('.json') and 'fact_prediction' in filename:
                    file_path = os.path.join(model_path, filename)
                    file_count += 1
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        model_name = data.get('model_name')
                        trust_score = data.get('model_trust_score')
                        if model_name is not None and isinstance(trust_score, (int, float)):
                            model_scores[model_name].append(float(trust_score))
                        else:
                            print(f"警告: 文件 {filename} 缺少必要的字段")
                    except json.JSONDecodeError as e:
                        print(f"错误: 解析文件 {filename} 时出错: {e}")
                    except Exception as e:
                        print(f"错误: 处理文件 {filename} 时出错: {e}")
    
    # 计算每个模型的平均信任分数
    model_avg_scores = {}
    for model_name, scores in model_scores.items():
        avg_score = sum(scores) / len(scores)
        model_avg_scores[model_name] = {
            'average_score': avg_score,
            'count': len(scores)
        }
    
    # 按平均分数降序排序
    sorted_model_scores = dict(sorted(
        model_avg_scores.items(), 
        key=lambda item: item[1]['average_score'], 
        reverse=True
    ))
    
    # 保存结果到文件
    try:
        existing = {}
        if os.path.exists(OUTPUT_FILE):
            try:
                with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        if not isinstance(existing, dict):
            existing = {}
        merged_output = {}
        if isinstance(existing.get("_meta"), dict):
            merged_output["_meta"] = existing.get("_meta")
        for model_name, info in sorted_model_scores.items():
            model_existing = existing.get(model_name) if isinstance(existing.get(model_name), dict) else {}
            model_existing = model_existing or {}
            merged = dict(info)
            for k in ("dynamic_sum", "dynamic_count", "dynamic_confidence"):
                if k in model_existing:
                    merged[k] = model_existing.get(k)
            merged_output[model_name] = merged
        for model_name, info in existing.items():
            if model_name == "_meta":
                continue
            if model_name not in merged_output and isinstance(info, dict) and "average_score" in info:
                merged_output[model_name] = info
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(merged_output, f, ensure_ascii=False, indent=2)
        
        print(f"成功处理了 {file_count} 个文件")
        print(f"模型平均信任分数已保存到: {OUTPUT_FILE}")
        print(f"共分析了 {len(model_scores)} 个不同的模型")
        for i, (model_name, info) in enumerate(sorted_model_scores.items()):
            print(f"{model_name}: 平均分数 = {info['average_score']:.6f} (样本数: {info['count']})")
            
    except Exception as e:
        print(f"错误: 保存结果时出错: {e}")


def calculate_feature_metrics():
    """
    计算feature_differences中所有参数的F1 score、AUC-ROC、准确率和精确率，并以表格形式输出。
    """
    # 创建用于存储每个模型的每个特征的预测和实际值的字典
    model_feature_data = defaultdict(lambda: defaultdict(lambda: {'predictions': [], 'actuals': [], 'differences': []}))
    
    # 遍历output目录下的所有模型子目录
    file_count = 0
    for model_dir in os.listdir(BASE_OUTPUT_DIR):
        model_path = os.path.join(BASE_OUTPUT_DIR, model_dir)
        if os.path.isdir(model_path):
            # 遍历模型子目录中的所有JSON文件
            for filename in os.listdir(model_path):
                if filename.endswith('.json') and 'fact_prediction' in filename:
                    file_path = os.path.join(model_path, filename)
                    file_count += 1
            
            try:
                # 读取JSON文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 提取model_name和相关特征
                model_name = data.get('model_name')
                feature_differences = data.get('feature_differences', {})
                predicted_features = data.get('predicted_sofa_features', {})
                actual_features = data.get('actual_sofa_features', {})
                
                if model_name:
                    # 对每个特征进行处理
                    for feature_name in feature_differences:
                        # 获取差异值
                        difference = feature_differences[feature_name]
                        
                        # 如果有预测和实际值，使用它们来计算二元分类结果
                        if feature_name in predicted_features and feature_name in actual_features:
                            # 取预测和实际值的最新值（最后一个）
                            if predicted_features[feature_name] and actual_features[feature_name]:
                                # 简化处理：如果有预测值和实际值，将差异值与阈值比较作为预测结果
                                # 这里使用差异值的绝对值大于0.5作为分类阈值
                                y_pred = 1 if abs(difference) > 0.5 else 0
                                # 实际情况：如果预测值和实际值不相等，则标记为错误
                                predicted_value = predicted_features[feature_name][-1] if isinstance(predicted_features[feature_name], list) else predicted_features[feature_name]
                                actual_value = actual_features[feature_name][-1] if isinstance(actual_features[feature_name], list) else actual_features[feature_name]
                                
                                # 根据特征类型确定实际标签
                                if isinstance(predicted_value, (int, float)) and isinstance(actual_value, (int, float)):
                                    # 对于连续值，判断是否在合理范围内（例如10%误差内）
                                    if max(abs(predicted_value - actual_value) / (max(abs(actual_value), 1e-6)), 0) > 0.1:
                                        y_true = 1  # 预测错误
                                    else:
                                        y_true = 0  # 预测正确
                                else:
                                    # 对于离散值，直接比较是否相等
                                    y_true = 0 if predicted_value == actual_value else 1
                                
                                # 保存预测结果、实际结果和差异值
                                model_feature_data[model_name][feature_name]['predictions'].append(y_pred)
                                model_feature_data[model_name][feature_name]['actuals'].append(y_true)
                                model_feature_data[model_name][feature_name]['differences'].append(difference)
                
            except json.JSONDecodeError as e:
                print(f"错误: 解析文件 {filename} 时出错: {e}")
            except Exception as e:
                print(f"错误: 处理文件 {filename} 时出错: {e}")

    # 计算每个模型的每个特征的评估指标
    metrics_data = []
    
    for model_name in model_feature_data:
        for feature_name in model_feature_data[model_name]:
            predictions = model_feature_data[model_name][feature_name]['predictions']
            actuals = model_feature_data[model_name][feature_name]['actuals']
            differences = model_feature_data[model_name][feature_name]['differences']
            
            # 确保有足够的数据进行计算
            if len(predictions) > 1 and len(set(actuals)) > 1:
                try:
                    # 计算混淆矩阵
                    cm = confusion_matrix(actuals, predictions)
                    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                    
                    # 计算指标
                    accuracy = accuracy_score(actuals, predictions)
                    precision = precision_score(actuals, predictions, zero_division=0)
                    f1 = f1_score(actuals, predictions, zero_division=0)
                    
                    # 计算AUC-ROC（使用差异值作为预测分数）
                    try:
                        auc_roc = roc_auc_score(actuals, [abs(d) for d in differences])
                    except ValueError:
                        auc_roc = 0.0
                    
                    # 添加到结果数据
                    metrics_data.append({
                        'model_name': model_name,
                        'feature_name': feature_name,
                        'samples': len(predictions),
                        'accuracy': accuracy,
                        'precision': precision,
                        'f1_score': f1,
                        'auc_roc': auc_roc,
                        'true_positives': tp,
                        'true_negatives': tn,
                        'false_positives': fp,
                        'false_negatives': fn
                    })
                except Exception as e:
                    print(f"错误: 计算模型 {model_name} 的特征 {feature_name} 指标时出错: {e}")
    
    # 创建DataFrame并按模型和特征排序
    df = pd.DataFrame(metrics_data)
    if not df.empty:
        # 保存到CSV文件
        df.to_csv(METRICS_OUTPUT_FILE, index=False, encoding='utf-8')
        
        # 打印表格
        print(f"\n特征指标计算结果已保存到: {METRICS_OUTPUT_FILE}")
        print("\n特征评估指标汇总:")
        
        # 按模型和特征名排序
        df_sorted = df.sort_values(['model_name', 'feature_name'])
        
        # 使用tabulate库打印美观的表格
        print(tabulate(df_sorted[['model_name', 'feature_name', 'samples', 'accuracy', 'precision', 'f1_score', 'auc_roc']], 
                      headers='keys', tablefmt='psql', floatfmt='.4f'))
    else:
        print("\n没有足够的数据来计算特征指标")


def compute_confidence_metrics(models_limit=5, threshold=0.75):
    model_scores = defaultdict(lambda: {'y_true': [], 'y_score': []})
    selected_models = None
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            trust_data = json.load(f)
        sorted_items = sorted(trust_data.items(), key=lambda x: x[1].get('average_score', 0), reverse=True)
        filtered = [m for m, info in sorted_items if info.get('count', 0) >= 10]
        selected_models = filtered[:models_limit] if filtered else [m for m, _ in sorted_items[:models_limit]]
    except Exception:
        selected_models = None

    for model_dir in os.listdir(BASE_OUTPUT_DIR):
        model_path = os.path.join(BASE_OUTPUT_DIR, model_dir)
        if os.path.isdir(model_path):
            for filename in os.listdir(model_path):
                if filename.endswith('.json') and 'fact_prediction' in filename:
                    file_path = os.path.join(model_path, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        mname = data.get('model_name')
                        if selected_models and mname not in selected_models:
                            continue
                        trust_score = data.get('model_trust_score')
                        analysis = data.get('analysis', {})
                        avg_diff = analysis.get('average_difference')
                        if isinstance(trust_score, (int, float)) and isinstance(avg_diff, (int, float)) and not np.isinf(avg_diff) and not np.isnan(avg_diff):
                            y_true = 1 if float(avg_diff) <= 0.5 else 0
                            model_scores[mname]['y_true'].append(y_true)
                            model_scores[mname]['y_score'].append(float(trust_score))
                    except Exception:
                        pass

    rows = []
    for mname, arrs in model_scores.items():
        y_true = arrs['y_true']
        y_score = arrs['y_score']
        if len(set(y_true)) < 2 or len(y_true) < 2:
            continue
        y_pred = [1 if s >= threshold else 0 for s in y_score]
        try:
            acc = float(__import__('sklearn.metrics').metrics.accuracy_score(y_true, y_pred))
        except Exception:
            acc = 0.0
        try:
            pre = float(__import__('sklearn.metrics').metrics.precision_score(y_true, y_pred, zero_division=0))
        except Exception:
            pre = 0.0
        try:
            f1 = float(__import__('sklearn.metrics').metrics.f1_score(y_true, y_pred, zero_division=0))
        except Exception:
            f1 = 0.0
        try:
            auc = float(__import__('sklearn.metrics').metrics.roc_auc_score(y_true, y_score))
        except Exception:
            auc = 0.0
        rows.append({'model_name': mname, 'samples': len(y_true), 'accuracy': acc, 'precision': pre, 'f1_score': f1, 'auc_roc': auc})

    df = __import__('pandas').DataFrame(rows)
    if not df.empty:
        df_sorted = df.sort_values(['model_name'])
        print('\n置信度评分指标汇总:')
        print(tabulate(df_sorted[['model_name', 'samples', 'accuracy', 'precision', 'f1_score', 'auc_roc']], headers='keys', tablefmt='psql', floatfmt='.4f'))
    else:
        print('\n没有足够的数据来计算置信度评分指标')


if __name__ == "__main__":
    print("开始分析模型信任分数...")
    analyze_model_trust_scores()
    
    print("\n开始计算特征指标...")
    calculate_feature_metrics()
    
    print("\n基于置信度评分计算分类指标...")
    compute_confidence_metrics(models_limit=5, threshold=0.75)
    
    print("\n所有分析完成")
