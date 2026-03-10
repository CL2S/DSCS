#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
经验知识库模块：存储和检索预测评估经验，提升后续预测性能

基于技术路线文档中的记忆演化机制设计：
1. 患者信息存储：存储患者特征、SOFA评分、预测结果
2. 经验规则提取：从成功/失败案例中提取规则
3. 相似性匹配：基于患者特征找到相似历史案例
4. 经验集成：在预测过程中提供参考案例和调整建议
"""

import json
import os
import re
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import uuid

# 导入现有记忆系统组件
try:
    from memory_representation import PatientRecord, PatientState, MedicalKnowledgeBase
    from integrate_memory import extract_basic_info, parse_values_from_text
except ImportError:
    # 定义简单的备用函数
    def extract_basic_info(text: str) -> Dict[str, Any]:
        """提取基本信息备用实现"""
        info = {}
        age_gender_match = re.search(r'(\d+)岁(男性|女性)', text)
        if age_gender_match:
            info['age'] = int(age_gender_match.group(1))
            info['gender'] = 'M' if age_gender_match.group(2) == '男性' else 'F'
        weight_match = re.search(r'体重([\d\.]+)kg', text)
        if weight_match:
            info['weight'] = float(weight_match.group(1))
        stay_id_match = re.search(r'ICU住院编号 (\d+)', text)
        if stay_id_match:
            info['stay_id'] = stay_id_match.group(1)
        return info

    def parse_values_from_text(text: str, key: str) -> List[float]:
        """解析数值列表备用实现"""
        pattern = f"{key}.*?\\[(.*?)\\]"
        match = re.search(pattern, text)
        if match:
            content = match.group(1)
            if not content.strip():
                return []
            try:
                return [float(x.strip()) for x in content.split(',') if x.strip() and x.strip() != 'nan']
            except ValueError:
                return []
        return []


class ExperienceCase:
    """
    单个经验案例，包含完整的预测评估结果
    基于技术路线文档中的患者信息存储和反馈事件设计
    """

    def __init__(self,
                 case_id: str,
                 patient_id: str,
                 input_description: str,
                 intervention: str,
                 prediction_result: Dict[str, Any],
                 evaluation_result: Dict[str, Any]):
        """
        初始化经验案例

        Args:
            case_id: 案例唯一标识
            patient_id: 患者ID
            input_description: 患者描述文本
            intervention: 干预措施
            prediction_result: 预测结果（包含SOFA特征、评分等）
            evaluation_result: 评估结果（包含置信度、评估模型等）
        """
        self.case_id = case_id
        self.patient_id = patient_id
        self.input_description = input_description
        self.intervention = intervention
        self.prediction_result = prediction_result
        self.evaluation_result = evaluation_result
        self.timestamp = datetime.now()

        # 提取关键特征
        self.basic_info = self._extract_basic_info()
        self.initial_features = self._extract_initial_features()
        self.predicted_features = self._extract_predicted_features()
        self.sofa_scores = self._extract_sofa_scores()

        # 计算经验质量分数
        self.quality_score = self._calculate_quality_score()

        # 提取关键标签
        self.tags = self._extract_tags()

    def _extract_basic_info(self) -> Dict[str, Any]:
        """从患者描述中提取基本信息"""
        return extract_basic_info(self.input_description)

    def _extract_initial_features(self) -> Dict[str, List[float]]:
        """从患者描述中提取初始SOFA相关特征"""
        # 解析input_description中的SOFA特征
        features = {}

        # SOFA相关特征键列表
        feature_keys = [
            "氧合指数变化",  # pao2_fio2_ratio
            "机械通气使用情况",  # mechanical_ventilation
            "血小板计数变化",  # platelet
            "总胆红素变化",  # bilirubin_total
            "血管活性药物使用剂量",  # vasopressor_rate
            "格拉斯哥昏迷评分变化",  # gcs_total
            "血清肌酐变化",  # creatinine
            "尿量变化"  # urine_output_ml
        ]

        english_keys = [
            "pao2_fio2_ratio",
            "mechanical_ventilation",
            "platelet",
            "bilirubin_total",
            "vasopressor_rate",
            "gcs_total",
            "creatinine",
            "urine_output_ml"
        ]

        for ch_key, en_key in zip(feature_keys, english_keys):
            values = parse_values_from_text(self.input_description, ch_key)
            if values:
                features[en_key] = values

        return features

    def _extract_predicted_features(self) -> Dict[str, List[float]]:
        """提取预测的SOFA特征（未来8小时）"""
        # 优先从predicted_sofa_features获取
        pred_features = self.prediction_result.get("predicted_sofa_features", {})
        if pred_features:
            return pred_features

        # 尝试从prediction_data中获取
        prediction_data = self.prediction_result.get("prediction_data", {})
        if prediction_data:
            pred = prediction_data.get("prediction", {})
            if pred:
                # 尝试多个路径
                sofa_features = (
                    pred.get("intervention_analysis", {}).get("sofa_related_features") or
                    pred.get("stages", {}).get("分析干预措施和风险", {}).get("output", {}).get("sofa_related_features") or
                    pred.get("_store", {}).get("reasoning_steps", {}).get("分析干预措施和风险", {}).get("output", {}).get("sofa_related_features") or
                    {}
                )
                return sofa_features

        return {}

    def _extract_sofa_scores(self) -> Dict[str, Any]:
        """提取SOFA评分信息"""
        scores = {}

        # 预测的SOFA评分序列
        scores["predicted_series"] = self.prediction_result.get("predicted_sofa_scores_series", {})

        # 预测的最终SOFA评分
        scores["predicted_final"] = self.prediction_result.get("predicted_sofa_scores", {})

        # 每小时SOFA总分
        scores["hourly_totals"] = self.prediction_result.get("hourly_sofa_totals", {})

        # 风险等级
        scores["risk_level"] = self.prediction_result.get("risk_level", "unknown")

        return scores

    def _calculate_quality_score(self) -> float:
        """计算经验质量分数（基于置信度和评估结果）"""
        # 总置信度
        total_confidence = self.evaluation_result.get("total_confidence", 0.0)

        # 评估模型数量
        eval_models = self.evaluation_result.get("evaluation_models", [])
        num_evaluators = len(eval_models) if eval_models else 1

        # 评估模型平均信任度
        eval_trust = self.evaluation_result.get("evaluation_models_trust", {})
        avg_trust = np.mean([v.get("total_confidence", 0.0) for v in eval_trust.values()]) if eval_trust else 0.5

        # 综合质量分数
        quality = 0.5 * total_confidence + 0.3 * (num_evaluators / 3.0) + 0.2 * avg_trust

        return min(1.0, max(0.0, quality))

    def _extract_tags(self) -> List[str]:
        """提取案例标签用于分类检索"""
        tags = []

        # 患者特征标签
        basic = self.basic_info
        if basic.get("age"):
            age = basic["age"]
            if age < 40:
                tags.append("young")
            elif age < 60:
                tags.append("middle_aged")
            else:
                tags.append("elderly")

        if basic.get("gender"):
            tags.append(basic["gender"].lower())

        # 干预措施标签
        intervention = self.intervention.lower()
        if "去甲肾上腺素" in intervention or "norepinephrine" in intervention:
            tags.append("norepinephrine")
        if "多巴胺" in intervention or "dopamine" in intervention:
            tags.append("dopamine")
        if "血管活性" in intervention or "vasopressor" in intervention:
            tags.append("vasopressor")

        # 风险等级标签
        risk_level = self.prediction_result.get("risk_level", "").lower()
        if risk_level:
            tags.append(f"risk_{risk_level}")

        # SOFA总分标签
        sofa_total = self.sofa_scores.get("predicted_final", {}).get("sofa_total", 0)
        if sofa_total < 6:
            tags.append("sofa_low")
        elif sofa_total < 9:
            tags.append("sofa_moderate")
        elif sofa_total < 12:
            tags.append("sofa_high")
        else:
            tags.append("sofa_critical")

        # 模型性能标签
        quality = self.quality_score
        if quality > 0.8:
            tags.append("high_quality")
        elif quality > 0.6:
            tags.append("medium_quality")
        else:
            tags.append("low_quality")

        return tags

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式用于序列化"""
        return {
            "case_id": self.case_id,
            "patient_id": self.patient_id,
            "timestamp": self.timestamp.isoformat(),
            "basic_info": self.basic_info,
            "input_description": self.input_description[:500] + "..." if len(self.input_description) > 500 else self.input_description,
            "intervention": self.intervention,
            "initial_features": self.initial_features,
            "predicted_features": self.predicted_features,
            "sofa_scores": self.sofa_scores,
            "prediction_model": self.prediction_result.get("prediction_model"),
            "evaluation_models": self.evaluation_result.get("evaluation_models", []),
            "total_confidence": self.evaluation_result.get("total_confidence"),
            "risk_level": self.prediction_result.get("risk_level"),
            "quality_score": self.quality_score,
            "tags": self.tags
        }

    def get_feature_vector(self) -> np.ndarray:
        """
        获取特征向量用于相似性计算
        包含：年龄、初始SOFA特征均值、预测SOFA特征趋势等
        """
        features = []

        # 基本信息
        basic = self.basic_info
        features.append(basic.get("age", 50) / 100.0)  # 归一化年龄
        features.append(1.0 if basic.get("gender") == "M" else 0.0)  # 性别

        # 初始SOFA特征均值（归一化）
        initial = self.initial_features
        for key in ["pao2_fio2_ratio", "platelet", "bilirubin_total", "gcs_total", "creatinine"]:
            values = initial.get(key, [])
            if values:
                # 归一化到[0,1]
                if key == "pao2_fio2_ratio":
                    norm_val = min(1.0, np.mean(values) / 500.0)
                elif key == "platelet":
                    norm_val = min(1.0, np.mean(values) / 300.0)
                elif key == "bilirubin_total":
                    norm_val = min(1.0, np.mean(values) / 10.0)
                elif key == "gcs_total":
                    norm_val = np.mean(values) / 15.0
                elif key == "creatinine":
                    norm_val = min(1.0, np.mean(values) / 5.0)
                else:
                    norm_val = 0.0
                features.append(norm_val)
            else:
                features.append(0.0)

        # 预测SOFA特征趋势（斜率）
        predicted = self.predicted_features
        for key in ["pao2_fio2_ratio", "platelet", "bilirubin_total", "vasopressor_rate", "gcs_total", "creatinine", "urine_output_ml"]:
            values = predicted.get(key, [])
            if len(values) >= 2:
                # 计算线性趋势
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                # 归一化斜率
                if key in ["pao2_fio2_ratio", "platelet", "gcs_total", "urine_output_ml"]:
                    # 上升为好
                    norm_slope = (slope + 10) / 20.0  # 假设最大变化±10
                elif key in ["bilirubin_total", "creatinine", "vasopressor_rate"]:
                    # 下降为好
                    norm_slope = (-slope + 10) / 20.0
                else:
                    norm_slope = 0.5
                features.append(min(1.0, max(0.0, norm_slope)))
            else:
                features.append(0.5)  # 中性

        return np.array(features)


class SimilarityMatcher:
    """
    相似性匹配器：基于患者特征找到相似历史案例
    基于技术路线文档中的神经符号相似性度量设计
    """

    def __init__(self):
        self.case_vectors = {}  # case_id -> 特征向量
        self.case_metadata = {}  # case_id -> 元数据

    def add_case(self, case: ExperienceCase):
        """添加案例到匹配器"""
        vector = case.get_feature_vector()
        self.case_vectors[case.case_id] = vector
        self.case_metadata[case.case_id] = {
            "patient_id": case.patient_id,
            "quality_score": case.quality_score,
            "tags": case.tags,
            "risk_level": case.sofa_scores.get("risk_level", "unknown")
        }

    def find_similar_cases(self,
                          query_vector: np.ndarray,
                          top_k: int = 5,
                          min_similarity: float = 0.6,
                          filter_tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        查找相似案例

        Args:
            query_vector: 查询特征向量
            top_k: 返回最相似的前k个案例
            min_similarity: 最小相似度阈值
            filter_tags: 过滤标签列表，只返回包含指定标签的案例

        Returns:
            相似案例列表，每个元素包含case_id、相似度、元数据
        """
        if not self.case_vectors:
            return []

        similarities = []
        for case_id, vector in self.case_vectors.items():
            # 计算余弦相似度
            similarity = self._cosine_similarity(query_vector, vector)

            # 应用标签过滤
            metadata = self.case_metadata[case_id]
            if filter_tags and not any(tag in metadata["tags"] for tag in filter_tags):
                continue

            # 应用质量权重：高质量案例更受信任
            weighted_similarity = similarity * (0.7 + 0.3 * metadata["quality_score"])

            if weighted_similarity >= min_similarity:
                similarities.append({
                    "case_id": case_id,
                    "similarity": weighted_similarity,
                    "raw_similarity": similarity,
                    "metadata": metadata
                })

        # 按相似度排序
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        return similarities[:top_k]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    def find_cases_by_tags(self, tags: List[str], min_quality: float = 0.0) -> List[str]:
        """根据标签查找案例"""
        result = []
        for case_id, metadata in self.case_metadata.items():
            if metadata["quality_score"] >= min_quality:
                if any(tag in metadata["tags"] for tag in tags):
                    result.append(case_id)
        return result


class ExperienceKnowledgeBase:
    """
    经验知识库主类：管理经验案例的存储、检索和更新
    基于技术路线文档中的知识存储和记忆演化机制
    """

    def __init__(self, storage_path: str = "./output/experience_knowledge_base.json"):
        self.storage_path = storage_path
        self.cases: Dict[str, ExperienceCase] = {}  # case_id -> ExperienceCase
        self.similarity_matcher = SimilarityMatcher()
        self.knowledge_base = MedicalKnowledgeBase()  # 用于存储提取的规则

        # 加载已有知识
        self._load()

    def _load(self):
        """从文件加载经验库"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 目前只存储元数据，案例需要从原始结果文件重建
                # 这里简化处理，实际应用中需要更完善的序列化
                print(f"Loaded experience knowledge base metadata from {self.storage_path}")

            except Exception as e:
                print(f"Error loading experience knowledge base: {e}")

    def _save(self):
        """保存经验库到文件"""
        try:
            # 收集案例摘要信息
            summary = {
                "total_cases": len(self.cases),
                "cases_summary": [],
                "knowledge_rules": self.knowledge_base.clinical_rules,
                "last_updated": datetime.now().isoformat()
            }

            for case_id, case in self.cases.items():
                summary["cases_summary"].append({
                    "case_id": case_id,
                    "patient_id": case.patient_id,
                    "quality_score": case.quality_score,
                    "tags": case.tags,
                    "risk_level": case.sofa_scores.get("risk_level", "unknown"),
                    "prediction_model": case.prediction_result.get("prediction_model"),
                    "total_confidence": case.evaluation_result.get("total_confidence")
                })

            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            print(f"Experience knowledge base saved to {self.storage_path}")

        except Exception as e:
            print(f"Error saving experience knowledge base: {e}")

    def add_experience_from_result(self, result_file: str):
        """
        从结果文件添加经验案例

        Args:
            result_file: 结果文件路径（如 result_*.json）
        """
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)

            # 提取患者ID
            patient_id = self._extract_patient_id(result_data.get("input_description", ""))
            if not patient_id:
                print(f"Cannot extract patient ID from result file: {result_file}")
                return None

            # 生成案例ID
            case_id = f"exp_{patient_id}_{int(datetime.now().timestamp())}"

            # 分离预测结果和评估结果
            prediction_result = {
                k: v for k, v in result_data.items()
                if k not in ["total_confidence", "evaluation_models", "evaluation_models_trust", "selected_evaluators", "evaluator_scores"]
            }

            evaluation_result = {
                "total_confidence": result_data.get("total_confidence", 0.0),
                "evaluation_models": result_data.get("evaluation_models", []),
                "evaluation_models_trust": result_data.get("evaluation_models_trust", {}),
                "selected_evaluators": result_data.get("selected_evaluators", []),
                "evaluator_scores": result_data.get("evaluator_scores", {})
            }

            # 创建经验案例
            case = ExperienceCase(
                case_id=case_id,
                patient_id=patient_id,
                input_description=result_data.get("input_description", ""),
                intervention=result_data.get("intervention", ""),
                prediction_result=prediction_result,
                evaluation_result=evaluation_result
            )

            # 添加到知识库
            self.cases[case_id] = case
            self.similarity_matcher.add_case(case)

            # 提取潜在规则
            self._extract_rules_from_case(case)

            # 保存更新
            self._save()

            print(f"Added experience case {case_id} (quality: {case.quality_score:.3f})")
            return case_id

        except Exception as e:
            print(f"Error adding experience from result file {result_file}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_patient_id(self, input_description: str) -> str:
        """从患者描述中提取患者ID"""
        # 使用现有函数
        basic_info = extract_basic_info(input_description)
        stay_id = basic_info.get("stay_id")
        if stay_id:
            return str(stay_id)

        # 尝试从文本中直接匹配
        match = re.search(r'ICU住院编号\s*(\d+)', input_description)
        if match:
            return match.group(1)

        # 生成默认ID
        return f"unknown_{hash(input_description) % 10000:04d}"

    def _extract_rules_from_case(self, case: ExperienceCase):
        """
        从案例中提取潜在规则
        基于技术路线文档中的规则范式提取
        """
        # 提取干预-效果规则
        intervention = case.intervention
        initial_state = case.initial_features
        predicted_state = case.predicted_features

        # 计算关键指标变化
        key_metrics = ["pao2_fio2_ratio", "platelet", "bilirubin_total", "vasopressor_rate", "gcs_total", "creatinine"]
        changes = {}

        for metric in key_metrics:
            init_vals = initial_state.get(metric, [])
            pred_vals = predicted_state.get(metric, [])

            if init_vals and pred_vals:
                init_mean = np.mean(init_vals[-3:]) if len(init_vals) >= 3 else np.mean(init_vals)
                pred_mean = np.mean(pred_vals[-3:]) if len(pred_vals) >= 3 else np.mean(pred_vals)

                if metric in ["pao2_fio2_ratio", "platelet", "gcs_total"]:
                    # 上升为好
                    change = "improve" if pred_mean > init_mean * 1.1 else "deteriorate" if pred_mean < init_mean * 0.9 else "stable"
                else:
                    # 下降为好
                    change = "improve" if pred_mean < init_mean * 0.9 else "deteriorate" if pred_mean > init_mean * 1.1 else "stable"

                changes[metric] = change

        # 创建规则
        if changes and case.quality_score > 0.7:
            rule = {
                "rule_id": f"rule_{case.case_id}",
                "intervention_type": self._classify_intervention(intervention),
                "initial_state": self._summarize_state(initial_state),
                "predicted_changes": changes,
                "confidence": case.quality_score,
                "source_case": case.case_id
            }

            # 添加到知识库
            self.knowledge_base.clinical_rules.append(rule)

    def _classify_intervention(self, intervention: str) -> str:
        """分类干预措施类型"""
        intervention_lower = intervention.lower()
        if "去甲肾上腺素" in intervention_lower or "norepinephrine" in intervention_lower:
            return "norepinephrine"
        elif "多巴胺" in intervention_lower or "dopamine" in intervention_lower:
            return "dopamine"
        elif "血管活性" in intervention_lower or "vasopressor" in intervention_lower:
            return "vasopressor"
        elif "抗生素" in intervention_lower or "antibiotic" in intervention_lower:
            return "antibiotic"
        else:
            return "other"

    def _summarize_state(self, state_features: Dict[str, List[float]]) -> str:
        """总结患者状态"""
        if not state_features:
            return "unknown"

        # 计算SOFA相关特征严重程度
        severity_scores = []

        # 氧合指数
        pao2 = state_features.get("pao2_fio2_ratio", [])
        if pao2:
            avg_pao2 = np.mean(pao2)
            if avg_pao2 < 100:
                severity_scores.append(4)
            elif avg_pao2 < 200:
                severity_scores.append(3)
            elif avg_pao2 < 300:
                severity_scores.append(2)
            elif avg_pao2 < 400:
                severity_scores.append(1)
            else:
                severity_scores.append(0)

        # 血小板
        platelet = state_features.get("platelet", [])
        if platelet:
            avg_platelet = np.mean(platelet)
            if avg_platelet < 20:
                severity_scores.append(4)
            elif avg_platelet < 50:
                severity_scores.append(3)
            elif avg_platelet < 100:
                severity_scores.append(2)
            elif avg_platelet < 150:
                severity_scores.append(1)
            else:
                severity_scores.append(0)

        # 胆红素
        bilirubin = state_features.get("bilirubin_total", [])
        if bilirubin:
            avg_bilirubin = np.mean(bilirubin)
            if avg_bilirubin >= 12:
                severity_scores.append(4)
            elif avg_bilirubin >= 6:
                severity_scores.append(3)
            elif avg_bilirubin >= 2:
                severity_scores.append(2)
            elif avg_bilirubin >= 1.2:
                severity_scores.append(1)
            else:
                severity_scores.append(0)

        if severity_scores:
            avg_severity = np.mean(severity_scores)
            if avg_severity < 1:
                return "mild"
            elif avg_severity < 2:
                return "moderate"
            elif avg_severity < 3:
                return "severe"
            else:
                return "critical"
        else:
            return "unknown"

    def find_similar_experiences(self,
                                input_description: str,
                                intervention: str,
                                initial_features: Optional[Dict[str, List[float]]] = None,
                                top_k: int = 3) -> List[Dict[str, Any]]:
        """
        查找相似经验案例

        Args:
            input_description: 患者描述
            intervention: 干预措施
            initial_features: 初始SOFA特征（可选）
            top_k: 返回相似案例数量

        Returns:
            相似案例列表，包含案例详情和相似度
        """
        # 从输入创建查询案例（不保存）
        query_case = self._create_query_case(input_description, intervention, initial_features)
        query_vector = query_case.get_feature_vector()

        # 查找相似案例
        similar_cases = self.similarity_matcher.find_similar_cases(
            query_vector,
            top_k=top_k,
            min_similarity=0.5
        )

        # 丰富返回信息
        results = []
        for sim_info in similar_cases:
            case_id = sim_info["case_id"]
            case = self.cases.get(case_id)
            if case:
                results.append({
                    "case_id": case_id,
                    "similarity": sim_info["similarity"],
                    "raw_similarity": sim_info["raw_similarity"],
                    "patient_id": case.patient_id,
                    "intervention": case.intervention[:100] + "..." if len(case.intervention) > 100 else case.intervention,
                    "prediction_model": case.prediction_result.get("prediction_model"),
                    "total_confidence": case.evaluation_result.get("total_confidence"),
                    "risk_level": case.sofa_scores.get("risk_level", "unknown"),
                    "quality_score": case.quality_score,
                    "tags": case.tags,
                    "predicted_sofa_total": case.sofa_scores.get("predicted_final", {}).get("sofa_total", 0),
                    "reasoning": case.prediction_result.get("reasoning", "")[:200] + "..." if len(case.prediction_result.get("reasoning", "")) > 200 else case.prediction_result.get("reasoning", "")
                })

        return results

    def _create_query_case(self,
                          input_description: str,
                          intervention: str,
                          initial_features: Optional[Dict[str, List[float]]] = None) -> ExperienceCase:
        """创建查询用临时案例（不保存到知识库）"""
        # 提取患者ID
        patient_id = self._extract_patient_id(input_description)

        # 创建模拟结果数据
        prediction_result = {
            "input_description": input_description,
            "intervention": intervention,
            "risk_level": "unknown"
        }

        evaluation_result = {
            "total_confidence": 0.5,
            "evaluation_models": [],
            "evaluation_models_trust": {}
        }

        # 如果提供了初始特征，添加到预测结果中
        if initial_features:
            prediction_result["predicted_sofa_features"] = initial_features

        # 创建临时案例
        case = ExperienceCase(
            case_id=f"query_{patient_id}_{int(datetime.now().timestamp())}",
            patient_id=patient_id,
            input_description=input_description,
            intervention=intervention,
            prediction_result=prediction_result,
            evaluation_result=evaluation_result
        )

        # 如果提供了初始特征，覆盖案例的初始特征
        if initial_features:
            case.initial_features = initial_features

        return case

    def get_recommendations(self,
                           input_description: str,
                           intervention: str,
                           initial_features: Optional[Dict[str, List[float]]] = None) -> Dict[str, Any]:
        """
        基于经验库提供推荐

        Args:
            input_description: 患者描述
            intervention: 计划干预措施
            initial_features: 初始SOFA特征（可选）

        Returns:
            推荐结果，包含相似案例、规则建议等
        """
        # 查找相似经验
        similar_experiences = self.find_similar_experiences(
            input_description, intervention, initial_features, top_k=5
        )

        # 提取相关规则
        relevant_rules = []
        for rule in self.knowledge_base.clinical_rules:
            # 简单匹配干预类型
            intervention_lower = intervention.lower()
            rule_intervention = rule.get("intervention_type", "").lower()

            if rule_intervention and rule_intervention in intervention_lower:
                relevant_rules.append(rule)

        # 生成建议
        suggestions = []

        if similar_experiences:
            # 分析相似案例的趋势
            confidences = [exp["total_confidence"] for exp in similar_experiences if exp["total_confidence"]]
            avg_confidence = np.mean(confidences) if confidences else 0.5

            risk_levels = [exp["risk_level"] for exp in similar_experiences if exp["risk_level"] != "unknown"]
            if risk_levels:
                most_common_risk = max(set(risk_levels), key=risk_levels.count)
            else:
                most_common_risk = "unknown"

            if avg_confidence > 0.7:
                suggestions.append(f"基于{len(similar_experiences)}个相似案例，该干预措施预期效果良好（平均置信度{avg_confidence:.2f}）")
            elif avg_confidence > 0.5:
                suggestions.append(f"基于{len(similar_experiences)}个相似案例，该干预措施效果中等（平均置信度{avg_confidence:.2f}）")
            else:
                suggestions.append(f"基于{len(similar_experiences)}个相似案例，该干预措施效果存疑（平均置信度{avg_confidence:.2f}）")

            if most_common_risk != "unknown":
                suggestions.append(f"相似案例中常见风险等级为'{most_common_risk}'")

        if relevant_rules:
            suggestions.append(f"找到{len(relevant_rules)}条相关临床规则")

        return {
            "similar_experiences": similar_experiences,
            "relevant_rules": relevant_rules[:3],  # 只返回前3条
            "suggestions": suggestions,
            "total_experience_cases": len(self.cases)
        }

    def batch_import_from_directory(self, directory: str, pattern: str = "result_*.json"):
        """
        批量从目录导入结果文件

        Args:
            directory: 目录路径
            pattern: 文件匹配模式
        """
        import glob

        result_files = glob.glob(os.path.join(directory, pattern))
        print(f"Found {len(result_files)} result files in {directory}")

        imported_count = 0
        for result_file in result_files:
            case_id = self.add_experience_from_result(result_file)
            if case_id:
                imported_count += 1

        print(f"Imported {imported_count} experience cases from {directory}")

    def get_statistics(self) -> Dict[str, Any]:
        """获取经验库统计信息"""
        if not self.cases:
            return {"total_cases": 0}

        # 按质量分数分类
        quality_scores = [case.quality_score for case in self.cases.values()]

        # 按风险等级统计
        risk_levels = {}
        for case in self.cases.values():
            risk = case.sofa_scores.get("risk_level", "unknown")
            risk_levels[risk] = risk_levels.get(risk, 0) + 1

        # 按模型统计
        models = {}
        for case in self.cases.values():
            model = case.prediction_result.get("prediction_model", "unknown")
            models[model] = models.get(model, 0) + 1

        # 按标签统计
        tag_counts = defaultdict(int)
        for case in self.cases.values():
            for tag in case.tags:
                tag_counts[tag] += 1

        return {
            "total_cases": len(self.cases),
            "quality_score": {
                "mean": np.mean(quality_scores),
                "std": np.std(quality_scores),
                "min": min(quality_scores),
                "max": max(quality_scores)
            },
            "risk_level_distribution": risk_levels,
            "model_distribution": models,
            "top_tags": dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            "knowledge_rules": len(self.knowledge_base.clinical_rules)
        }


# 工具函数：集成到现有预测流程
def integrate_with_prediction(ekb: ExperienceKnowledgeBase,
                             patient_data: Dict[str, Any],
                             model_name: str) -> Dict[str, Any]:
    """
    在预测过程中集成经验知识库

    Args:
        ekb: 经验知识库实例
        patient_data: 患者数据（包含input_description, intervention等）
        model_name: 预测模型名称

    Returns:
        增强的预测上下文，包含相似经验和推荐
    """
    input_description = patient_data.get("input_description", "")
    intervention = patient_data.get("intervention", "")

    # 获取推荐
    recommendations = ekb.get_recommendations(input_description, intervention)

    # 构建增强上下文
    enhanced_context = {
        "patient_data": patient_data,
        "model_name": model_name,
        "experience_based_recommendations": recommendations,
        "has_experience_support": len(recommendations["similar_experiences"]) > 0
    }

    # 如果有相似经验，可以提供参考预测
    if recommendations["similar_experiences"]:
        best_experience = recommendations["similar_experiences"][0]
        enhanced_context["reference_prediction"] = {
            "similarity": best_experience["similarity"],
            "predicted_sofa_total": best_experience["predicted_sofa_total"],
            "risk_level": best_experience["risk_level"],
            "confidence": best_experience["total_confidence"]
        }

    return enhanced_context


# 使用示例
if __name__ == "__main__":
    # 初始化经验知识库
    ekb = ExperienceKnowledgeBase()

    # 从现有结果文件导入经验
    print("正在导入现有经验案例...")
    ekb.batch_import_from_directory("./output/best_result/", "result_*.json")

    # 查看统计信息
    stats = ekb.get_statistics()
    print(f"\n经验库统计信息:")
    print(f"总案例数: {stats['total_cases']}")
    print(f"质量分数均值: {stats['quality_score']['mean']:.3f}")
    print(f"风险等级分布: {stats['risk_level_distribution']}")
    print(f"知识规则数: {stats['knowledge_rules']}")

    # 示例查询
    print("\n示例查询:")
    sample_description = "ICU住院编号 30315020，对应患者编号 11992186，56岁男性，体重87.05kg。监测持续时间为 0 小时（共 7 次测量，每4小时一次）..."
    sample_intervention = "立即启动去甲肾上腺素输注，维持剂量为5.0μg·kg⁻¹·min⁻¹"

    recommendations = ekb.get_recommendations(sample_description, sample_intervention)
    print(f"找到 {len(recommendations['similar_experiences'])} 个相似经验案例")
    print(f"生成 {len(recommendations['suggestions'])} 条建议")

    if recommendations["similar_experiences"]:
        print("\n最相似的经验案例:")
        best_exp = recommendations["similar_experiences"][0]
        print(f"  相似度: {best_exp['similarity']:.3f}")
        print(f"  预测SOFA总分: {best_exp['predicted_sofa_total']}")
        print(f"  风险等级: {best_exp['risk_level']}")
        print(f"  置信度: {best_exp['total_confidence']:.3f}")