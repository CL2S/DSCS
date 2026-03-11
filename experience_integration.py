#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
经验知识库集成模块：将经验知识库集成到现有预测流程中
"""

import json
import os
import re
from typing import Dict, Any, Optional, List
from datetime import datetime

from experience_knowledge_base import ExperienceKnowledgeBase, integrate_with_prediction

try:
    from advanced_experience_memory import AdvancedExperienceMemoryBank
except Exception:
    AdvancedExperienceMemoryBank = None




def _clinical_risk_gate(input_description: str, intervention: str) -> Dict[str, Any]:
    """基础经验阈值门控：用于限制高危场景下经验置信度过度上调。"""
    text = (input_description or "") + "\n" + (intervention or "")

    def _series_last(key: str):
        m = re.search(rf"{re.escape(key)}.*?\[(.*?)\]", text)
        if not m:
            return None
        vals = []
        for p in m.group(1).split(','):
            p = p.strip()
            if not p or p.lower() == 'nan':
                continue
            try:
                vals.append(float(p))
            except Exception:
                continue
        return vals[-1] if vals else None

    rr = _series_last("呼吸频率变化")
    sbp = _series_last("收缩压变化")
    map_v = _series_last("平均动脉压变化")
    gcs = _series_last("格拉斯哥昏迷评分变化")
    lactate = _series_last("乳酸变化")
    ne = _series_last("去甲肾上腺素")
    if ne is None:
        ne = _series_last("血管活性药物使用剂量")

    qsofa = 0
    qsofa += 1 if rr is not None and rr >= 22 else 0
    qsofa += 1 if sbp is not None and sbp <= 100 else 0
    qsofa += 1 if gcs is not None and gcs < 15 else 0

    high_risk = False
    triggers = []
    if qsofa >= 2:
        high_risk = True
        triggers.append("qSOFA>=2")
    if map_v is not None and map_v < 65:
        high_risk = True
        triggers.append("MAP<65")
    if lactate is not None and lactate > 4:
        high_risk = True
        triggers.append("lactate>4")
    if ne is not None and ne > 0.1:
        high_risk = True
        triggers.append("norepinephrine>0.1")

    return {"high_risk_gate": high_risk, "triggers": triggers}

class ExperienceIntegration:
    """
    经验知识库与预测系统的集成管理器
    """

    def __init__(self, ekb_path: str = "./output/experience_knowledge_base.json"):
        """
        初始化集成管理器

        Args:
            ekb_path: 经验知识库存储路径
        """
        # 优先使用前沿混合记忆经验库；失败时回退到原有实现
        advanced_path = ekb_path.replace("experience_knowledge_base.json", "advanced_experience_memory.json")
        if AdvancedExperienceMemoryBank is not None:
            try:
                self.ekb = AdvancedExperienceMemoryBank(storage_path=advanced_path)
            except Exception:
                self.ekb = ExperienceKnowledgeBase(storage_path=ekb_path)
        else:
            self.ekb = ExperienceKnowledgeBase(storage_path=ekb_path)
        self.integration_enabled = True

        # 加载现有经验（从best_result目录）
        self._load_existing_experiences()

    def _load_existing_experiences(self):
        """加载现有的预测结果作为经验"""
        best_result_dir = "./output/best_result/"
        if os.path.exists(best_result_dir):
            try:
                self.ekb.batch_import_from_directory(best_result_dir, "result_*.json")
                print(f"已从 {best_result_dir} 加载现有经验案例")
            except Exception as e:
                print(f"加载现有经验时出错: {e}")
                self.integration_enabled = False

    def enhance_prediction_context(self,
                                  patient_data: Dict[str, Any],
                                  model_name: str) -> Dict[str, Any]:
        """
        增强预测上下文：基于经验知识库提供参考信息

        Args:
            patient_data: 患者数据
            model_name: 预测模型名称

        Returns:
            增强的预测上下文
        """
        if not self.integration_enabled:
            return {"experience_integration": False}

        try:
            input_description = patient_data.get("input_description", "")
            intervention = patient_data.get("intervention", "")

            # 获取经验推荐
            recommendations = self.ekb.get_recommendations(
                input_description, intervention
            )

            # 构建增强上下文
            enhanced_context = {
                "experience_integration": True,
                "similar_experiences_count": len(recommendations["similar_experiences"]),
                "recommendations": recommendations["suggestions"],
                "reference_experiences": recommendations["similar_experiences"][:3] if recommendations["similar_experiences"] else [],
                "relevant_rules_count": len(recommendations["relevant_rules"])
            }

            # 如果有相似经验，提取关键信息供预测参考
            if recommendations["similar_experiences"]:
                best_exp = recommendations["similar_experiences"][0]
                enhanced_context["best_reference"] = {
                    "similarity": best_exp["similarity"],
                    "predicted_sofa_total": best_exp.get("predicted_sofa_total", 0),
                    "risk_level": best_exp.get("risk_level", "unknown"),
                    "confidence": best_exp.get("total_confidence", 0.0),
                    "quality_score": best_exp.get("quality_score", 0.0)
                }

                # 生成提示文本，可用于调整DSPy提示
                prompt_addition = self._generate_prompt_addition(best_exp, recommendations["suggestions"])
                enhanced_context["prompt_addition"] = prompt_addition

            return enhanced_context

        except Exception as e:
            print(f"增强预测上下文时出错: {e}")
            return {"experience_integration": False, "error": str(e)}

    def _generate_prompt_addition(self,
                                 best_experience: Dict[str, Any],
                                 suggestions: List[str]) -> str:
        """
        生成添加到DSPy提示中的经验信息

        Args:
            best_experience: 最相似的经验案例
            suggestions: 建议列表

        Returns:
            提示文本
        """
        addition = "\n\n【经验知识库参考信息】\n"

        # 添加相似案例信息
        addition += f"发现相似历史案例（相似度{best_experience['similarity']:.2%}）：\n"
        addition += f"- 患者情况相似，曾使用干预：{best_experience['intervention'][:80]}...\n"
        addition += f"- 预测SOFA总分：{best_experience.get('predicted_sofa_total', '未知')}\n"
        addition += f"- 风险评估：{best_experience.get('risk_level', '未知')}\n"
        addition += f"- 预测置信度：{best_experience.get('total_confidence', 0.0):.2f}\n"

        # 添加建议
        if suggestions:
            addition += f"\n经验建议：\n"
            for i, suggestion in enumerate(suggestions[:3], 1):
                addition += f"{i}. {suggestion}\n"

        addition += "\n请参考上述经验信息，但注意当前患者可能存在的个体差异。"
        return addition

    def update_from_prediction_result(self,
                                     result_file: str,
                                     patient_data: Dict[str, Any]) -> Optional[str]:
        """
        从预测结果更新经验知识库

        Args:
            result_file: 结果文件路径
            patient_data: 患者数据

        Returns:
            添加的经验案例ID，失败时返回None
        """
        if not self.integration_enabled:
            return None

        try:
            # 检查结果文件是否存在
            if not os.path.exists(result_file):
                print(f"结果文件不存在: {result_file}")
                return None

            # 添加经验案例
            case_id = self.ekb.add_experience_from_result(result_file)

            if case_id:
                print(f"已将预测结果添加到经验知识库: {case_id}")

                # 尝试提取规则
                self._extract_and_update_rules(case_id)

            return case_id

        except Exception as e:
            print(f"更新经验知识库时出错: {e}")
            return None

    def _extract_and_update_rules(self, case_id: str):
        """提取并更新规则"""
        # 规则提取已在ExperienceCase中实现
        # 这里可以添加额外的规则学习逻辑
        pass

    def adjust_model_confidence(self,
                               model_name: str,
                               base_confidence: float,
                               patient_data: Dict[str, Any]) -> float:
        """
        基于经验调整模型置信度

        Args:
            model_name: 模型名称
            base_confidence: 基础置信度
            patient_data: 患者数据

        Returns:
            调整后的置信度
        """
        if not self.integration_enabled:
            return base_confidence

        try:
            input_description = patient_data.get("input_description", "")
            intervention = patient_data.get("intervention", "")

            # 查找相似经验
            recommendations = self.ekb.get_recommendations(
                input_description, intervention
            )

            if not recommendations["similar_experiences"]:
                return base_confidence

            # 计算相似经验的平均表现
            similar_exps = recommendations["similar_experiences"]
            relevant_exps = [exp for exp in similar_exps
                            if exp.get("prediction_model") == model_name]

            if not relevant_exps:
                # 使用所有相似经验
                relevant_exps = similar_exps

            # 计算加权置信度
            total_weight = 0.0
            weighted_conf = 0.0

            for exp in relevant_exps[:5]:  # 最多使用5个最相似的经验
                similarity = exp.get("similarity", 0.5)
                exp_confidence = exp.get("total_confidence", 0.5)
                quality = exp.get("quality_score", 0.5)

                weight = similarity * quality
                total_weight += weight
                weighted_conf += exp_confidence * weight

            if total_weight > 0:
                exp_based_confidence = weighted_conf / total_weight
                # 混合基础置信度和经验置信度（权重可调）
                adjusted = 0.7 * base_confidence + 0.3 * exp_based_confidence

                # 临床高危门控：高风险时限制经验导致的上调幅度
                gate = _clinical_risk_gate(input_description, intervention)
                if gate.get("high_risk_gate"):
                    max_uplift = 0.08
                    adjusted = min(adjusted, base_confidence + max_uplift)

                return min(1.0, max(0.0, adjusted))

            return base_confidence

        except Exception as e:
            print(f"调整模型置信度时出错: {e}")
            return base_confidence

    def provide_decision_support(self,
                                prediction_result: Dict[str, Any],
                                patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        提供决策支持：基于经验验证预测结果

        Args:
            prediction_result: 预测结果
            patient_data: 患者数据

        Returns:
            决策支持信息
        """
        if not self.integration_enabled:
            return {"decision_support": False}

        try:
            input_description = patient_data.get("input_description", "")
            intervention = patient_data.get("intervention", "")

            # 获取相似经验
            recommendations = self.ekb.get_recommendations(
                input_description, intervention
            )

            support_info = {
                "decision_support": True,
                "similar_experiences_count": len(recommendations["similar_experiences"]),
                "recommendations": recommendations["suggestions"]
            }

            # 对比预测结果与历史经验
            if recommendations["similar_experiences"]:
                pred_sofa_total = prediction_result.get("predicted_sofa_scores", {}).get("sofa_total", 0)
                pred_risk_level = prediction_result.get("risk_level", "unknown")

                exp_sofa_totals = []
                exp_risk_levels = []

                for exp in recommendations["similar_experiences"][:5]:
                    exp_sofa = exp.get("predicted_sofa_total", 0)
                    exp_risk = exp.get("risk_level", "unknown")

                    if exp_sofa:
                        exp_sofa_totals.append(exp_sofa)
                    if exp_risk != "unknown":
                        exp_risk_levels.append(exp_risk)

                if exp_sofa_totals:
                    avg_exp_sofa = sum(exp_sofa_totals) / len(exp_sofa_totals)
                    support_info["sofa_comparison"] = {
                        "predicted": pred_sofa_total,
                        "average_similar_cases": avg_exp_sofa,
                        "difference": pred_sofa_total - avg_exp_sofa
                    }

                if exp_risk_levels:
                    most_common_risk = max(set(exp_risk_levels), key=exp_risk_levels.count)
                    support_info["risk_comparison"] = {
                        "predicted": pred_risk_level,
                        "most_common_similar": most_common_risk,
                        "match": pred_risk_level == most_common_risk
                    }

            return support_info

        except Exception as e:
            print(f"提供决策支持时出错: {e}")
            return {"decision_support": False, "error": str(e)}

    def get_statistics(self) -> Dict[str, Any]:
        """获取集成统计信息"""
        if not self.integration_enabled:
            return {"integration_enabled": False}

        try:
            ekb_stats = self.ekb.get_statistics()
            return {
                "integration_enabled": True,
                "experience_knowledge_base": ekb_stats
            }
        except Exception as e:
            return {
                "integration_enabled": True,
                "error": str(e)
            }


# 全局集成实例（单例模式）
_experience_integration_instance = None


def get_experience_integration() -> ExperienceIntegration:
    """获取全局经验集成实例"""
    global _experience_integration_instance
    if _experience_integration_instance is None:
        _experience_integration_instance = ExperienceIntegration()
    return _experience_integration_instance


def enable_experience_integration(enable: bool = True):
    """启用或禁用经验集成"""
    integration = get_experience_integration()
    integration.integration_enabled = enable
    print(f"经验集成已{'启用' if enable else '禁用'}")


# 主要集成函数
def integrate_experience_into_prediction(patient_data: Dict[str, Any],
                                        model_name: str) -> Dict[str, Any]:
    """
    将经验知识集成到预测中（主入口函数）

    Args:
        patient_data: 患者数据
        model_name: 模型名称

    Returns:
        增强的预测上下文
    """
    integration = get_experience_integration()
    return integration.enhance_prediction_context(patient_data, model_name)


def update_experience_from_result(result_file: str,
                                 patient_data: Dict[str, Any]) -> Optional[str]:
    """
    从预测结果更新经验知识库

    Args:
        result_file: 结果文件路径
        patient_data: 患者数据

    Returns:
        案例ID
    """
    integration = get_experience_integration()
    return integration.update_from_prediction_result(result_file, patient_data)


def get_experience_based_decision_support(prediction_result: Dict[str, Any],
                                         patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取基于经验的决策支持

    Args:
        prediction_result: 预测结果
        patient_data: 患者数据

    Returns:
        决策支持信息
    """
    integration = get_experience_integration()
    return integration.provide_decision_support(prediction_result, patient_data)


# CLI工具函数
def experience_cli_tool():
    """经验知识库CLI工具"""
    import argparse

    parser = argparse.ArgumentParser(description='经验知识库管理工具')
    parser.add_argument('--import', dest='import_dir',
                       help='从目录导入结果文件作为经验案例')
    parser.add_argument('--stats', action='store_true',
                       help='显示经验库统计信息')
    parser.add_argument('--query', type=str,
                       help='查询相似案例（提供患者描述文件）')
    parser.add_argument('--test', action='store_true',
                       help='测试经验集成功能')

    args = parser.parse_args()

    integration = get_experience_integration()

    if args.import_dir:
        print(f"从目录导入经验案例: {args.import_dir}")
        integration.ekb.batch_import_from_directory(args.import_dir, "result_*.json")

    if args.stats:
        stats = integration.get_statistics()
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.query:
        try:
            with open(args.query, 'r', encoding='utf-8') as f:
                query_data = json.load(f)

            input_desc = query_data.get("input_description", "")
            intervention = query_data.get("intervention", "")

            recommendations = integration.ekb.get_recommendations(input_desc, intervention)
            print(f"查询结果:")
            print(f"- 找到 {len(recommendations['similar_experiences'])} 个相似经验案例")
            print(f"- 生成 {len(recommendations['suggestions'])} 条建议")

            if recommendations["similar_experiences"]:
                print(f"\n最相似的3个案例:")
                for i, exp in enumerate(recommendations["similar_experiences"][:3], 1):
                    print(f"{i}. 相似度: {exp['similarity']:.3f}, "
                          f"SOFA总分: {exp.get('predicted_sofa_total', '未知')}, "
                          f"风险等级: {exp.get('risk_level', '未知')}")

        except Exception as e:
            print(f"查询出错: {e}")

    if args.test:
        print("测试经验集成功能...")

        # 测试数据
        test_description = "ICU住院编号 30315020，对应患者编号 11992186，56岁男性，体重87.05kg。"
        test_intervention = "立即启动去甲肾上腺素输注"

        # 测试增强上下文
        test_patient_data = {
            "input_description": test_description,
            "intervention": test_intervention
        }

        context = integrate_experience_into_prediction(test_patient_data, "test_model")
        print(f"增强上下文: {json.dumps(context, ensure_ascii=False, indent=2)[:500]}...")

        # 测试决策支持
        test_prediction = {
            "predicted_sofa_scores": {"sofa_total": 6},
            "risk_level": "moderate"
        }

        support = get_experience_based_decision_support(test_prediction, test_patient_data)
        print(f"决策支持: {json.dumps(support, ensure_ascii=False, indent=2)}")

        print("测试完成")


if __name__ == "__main__":
    experience_cli_tool()
