#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core prediction and evaluation functions
Separated to avoid circular imports between main.py and gui.py
"""

import json
import os
import time

# Import dependencies
try:
    from experiment import AdaptiveExperimentAgent, configure_dspy, validate_sofa_features, generate_fallback_sofa_features
except Exception:
    AdaptiveExperimentAgent = None
    def configure_dspy(*args, **kwargs):
        return None
    def validate_sofa_features(*args, **kwargs):
        return True
    def generate_fallback_sofa_features(*args, **kwargs):
        return {}

from sofa_prediction_evaluator import (
    evaluate_with_ollama,
    extract_patient_id,
    extract_model_confidence,
    save_evaluation_report
)

# Experience knowledge base integration
try:
    from experience_integration import (
        update_experience_from_result,
        integrate_experience_into_prediction,
        get_experience_based_decision_support,
        get_experience_integration
    )
except ImportError as e:
    print(f"Warning: Experience integration module not available: {e}")
    update_experience_from_result = None
    integrate_experience_into_prediction = None
    get_experience_based_decision_support = None
    get_experience_integration = None

INITIAL_TRUST_PATH = "/data/wzx/output/Model_Trust_Score.json"
DYNAMIC_TRUST_PATH = "/data/wzx/output/Dynamic_Trust_State.json"

def _load_json_dict(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _save_json_dict(path: str, data: dict):
    try:
        out_dir = os.path.dirname(path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

def _load_initial_trust_scores(path: str = INITIAL_TRUST_PATH):
    raw = _load_json_dict(path)
    trust = {}
    for model, info in raw.items():
        if model == "_meta":
            continue
        if not isinstance(info, dict):
            continue
        avg = info.get("average_score")
        if isinstance(avg, (int, float)):
            trust[model] = float(avg)
    return trust

def _load_dynamic_trust_state(path: str = DYNAMIC_TRUST_PATH):
    raw = _load_json_dict(path)
    meta = raw.get("_meta") if isinstance(raw.get("_meta"), dict) else {}
    models = raw.get("models") if isinstance(raw.get("models"), dict) else {}
    schema_version = meta.get("schema_version")
    if not isinstance(schema_version, int) or schema_version <= 0:
        schema_version = 1
    dynamic_method = meta.get("dynamic_method")
    if not isinstance(dynamic_method, str) or not dynamic_method:
        dynamic_method = "max_normalized_v1"
    a_weight = meta.get("a", 1.0)
    if not isinstance(a_weight, (int, float)):
        a_weight = 1.0
    a_weight = float(a_weight)
    if a_weight < 0.2:
        a_weight = 0.2
    if a_weight > 1.0:
        a_weight = 1.0
    runs = meta.get("counterfactual_runs", 0)
    if not isinstance(runs, int) or runs < 0:
        runs = 0
    state = {
        "_meta": {
            "schema_version": int(schema_version),
            "dynamic_method": str(dynamic_method),
            "a": a_weight,
            "counterfactual_runs": runs,
        },
        "models": {},
    }
    for model, info in models.items():
        if not isinstance(info, dict):
            continue
        dynamic_sum = info.get("dynamic_sum", 0.0)
        dynamic_count = info.get("dynamic_count", 0)
        completed_predictions = info.get("completed_predictions", None)
        legacy_dynamic_sum = info.get("legacy_dynamic_sum", None)
        legacy_dynamic_count = info.get("legacy_dynamic_count", None)
        if not isinstance(dynamic_sum, (int, float)):
            dynamic_sum = 0.0
        if not isinstance(dynamic_count, int) or dynamic_count < 0:
            dynamic_count = 0
        if not isinstance(completed_predictions, int) or completed_predictions < 0:
            completed_predictions = None
        if not isinstance(legacy_dynamic_sum, (int, float)):
            legacy_dynamic_sum = None
        if not isinstance(legacy_dynamic_count, int) or legacy_dynamic_count < 0:
            legacy_dynamic_count = None
        entry = {"dynamic_sum": float(dynamic_sum), "dynamic_count": int(dynamic_count)}
        if completed_predictions is not None:
            entry["completed_predictions"] = int(completed_predictions)
        if legacy_dynamic_sum is not None:
            entry["legacy_dynamic_sum"] = float(legacy_dynamic_sum)
        if legacy_dynamic_count is not None:
            entry["legacy_dynamic_count"] = int(legacy_dynamic_count)
        state["models"][model] = entry

    if state["_meta"]["dynamic_method"] != "weighted_avg_v2":
        migrated = {"_meta": dict(state["_meta"]), "models": {}}
        migrated["_meta"]["schema_version"] = 2
        migrated["_meta"]["dynamic_method"] = "weighted_avg_v2"
        for model, info in state["models"].items():
            migrated["models"][model] = {
                "dynamic_sum": 0.0,
                "dynamic_count": 0,
                "completed_predictions": int(info.get("dynamic_count", 0) or 0),
                "legacy_dynamic_sum": float(info.get("dynamic_sum", 0.0) or 0.0),
                "legacy_dynamic_count": int(info.get("dynamic_count", 0) or 0),
            }
        _save_dynamic_trust_state(migrated, path)
        return {"_meta": migrated["_meta"], "models": {m: {"dynamic_sum": 0.0, "dynamic_count": 0, "completed_predictions": int(migrated["models"][m].get("completed_predictions", 0) or 0)} for m in migrated["models"].keys()}}

    return state

def _save_dynamic_trust_state(state: dict, path: str = DYNAMIC_TRUST_PATH):
    if not isinstance(state, dict):
        state = {}
    meta = state.get("_meta") if isinstance(state.get("_meta"), dict) else {}
    models = state.get("models") if isinstance(state.get("models"), dict) else {}
    schema_version = meta.get("schema_version")
    if not isinstance(schema_version, int) or schema_version <= 0:
        schema_version = 2
    dynamic_method = meta.get("dynamic_method")
    if not isinstance(dynamic_method, str) or not dynamic_method:
        dynamic_method = "weighted_avg_v2"
    a_weight = meta.get("a", 1.0)
    if not isinstance(a_weight, (int, float)):
        a_weight = 1.0
    a_weight = float(a_weight)
    if a_weight < 0.2:
        a_weight = 0.2
    if a_weight > 1.0:
        a_weight = 1.0
    runs = meta.get("counterfactual_runs", 0)
    if not isinstance(runs, int) or runs < 0:
        runs = 0
    output = {
        "_meta": {
            "schema_version": int(schema_version),
            "dynamic_method": str(dynamic_method),
            "a": a_weight,
            "counterfactual_runs": runs,
            "updated_at": float(time.time()),
        },
        "models": {},
    }
    for model, info in models.items():
        if not isinstance(info, dict):
            continue
        dynamic_sum = info.get("dynamic_sum", 0.0)
        dynamic_count = info.get("dynamic_count", 0)
        completed_predictions = info.get("completed_predictions", None)
        legacy_dynamic_sum = info.get("legacy_dynamic_sum", None)
        legacy_dynamic_count = info.get("legacy_dynamic_count", None)
        if not isinstance(dynamic_sum, (int, float)):
            dynamic_sum = 0.0
        if not isinstance(dynamic_count, int) or dynamic_count < 0:
            dynamic_count = 0
        if not isinstance(completed_predictions, int) or completed_predictions < 0:
            completed_predictions = None
        if not isinstance(legacy_dynamic_sum, (int, float)):
            legacy_dynamic_sum = None
        if not isinstance(legacy_dynamic_count, int) or legacy_dynamic_count < 0:
            legacy_dynamic_count = None
        dynamic_confidence = (float(dynamic_sum) / int(dynamic_count)) if int(dynamic_count) > 0 else 0.0
        out_model = {
            "dynamic_sum": float(dynamic_sum),
            "dynamic_count": int(dynamic_count),
            "dynamic_confidence": float(dynamic_confidence),
        }
        if completed_predictions is not None:
            out_model["completed_predictions"] = int(completed_predictions)
        if legacy_dynamic_sum is not None:
            out_model["legacy_dynamic_sum"] = float(legacy_dynamic_sum)
        if legacy_dynamic_count is not None:
            out_model["legacy_dynamic_count"] = int(legacy_dynamic_count)
        output["models"][model] = out_model
    return _save_json_dict(path, output)

def _bootstrap_dynamic_trust_state(model_names=None, path: str = DYNAMIC_TRUST_PATH):
    state = _load_dynamic_trust_state(path)
    meta = state.get("_meta") if isinstance(state.get("_meta"), dict) else {}
    state["_meta"] = meta
    models = state.get("models") if isinstance(state.get("models"), dict) else {}
    state["models"] = models

    initial_trust = _load_initial_trust_scores()
    if model_names and isinstance(model_names, (list, tuple)):
        names = [m for m in model_names if isinstance(m, str) and m]
    else:
        names = [m for m in initial_trust.keys() if isinstance(m, str) and m]

    changed = False
    for m in names:
        info = models.get(m)
        if not isinstance(info, dict):
            models[m] = {"dynamic_sum": 0.0, "dynamic_count": 0, "completed_predictions": 0}
            changed = True
            continue
        if "dynamic_sum" not in info or not isinstance(info.get("dynamic_sum"), (int, float)):
            info["dynamic_sum"] = 0.0
            changed = True
        if "dynamic_count" not in info or not isinstance(info.get("dynamic_count"), int) or int(info.get("dynamic_count")) < 0:
            info["dynamic_count"] = 0
            changed = True
        if "completed_predictions" not in info or not isinstance(info.get("completed_predictions"), int) or int(info.get("completed_predictions")) < 0:
            inferred = meta.get("counterfactual_runs", 0)
            if isinstance(inferred, int) and inferred > 0:
                info["completed_predictions"] = int(inferred)
            else:
                info["completed_predictions"] = 0
            changed = True

    if changed:
        _save_dynamic_trust_state(state, path)
    return state

def _get_model_trust_info(model_name: str):
    if not model_name:
        return {"a": 1.0, "dynamic_confidence": 0.0, "total_confidence": 1.0, "counterfactual_runs": 0}

    initial_trust = _load_initial_trust_scores()
    dynamic_state = _load_dynamic_trust_state()

    a_weight = float(dynamic_state.get("_meta", {}).get("a", 1.0))
    runs = dynamic_state.get("_meta", {}).get("counterfactual_runs", 0)
    if not isinstance(runs, int) or runs < 0:
        runs = 0

    initial = float(initial_trust.get(model_name, 1.0))
    st = dynamic_state.get("models", {}).get(model_name, {}) if isinstance(dynamic_state.get("models"), dict) else {}
    dynamic_sum = st.get("dynamic_sum", 0.0)
    dynamic_count = st.get("dynamic_count", 0)
    if not isinstance(dynamic_sum, (int, float)):
        dynamic_sum = 0.0
    if not isinstance(dynamic_count, int) or dynamic_count < 0:
        dynamic_count = 0
    dynamic_avg = (float(dynamic_sum) / int(dynamic_count)) if int(dynamic_count) > 0 else 0.0
    total_confidence = float(a_weight) * float(initial) + (1.0 - float(a_weight)) * float(dynamic_avg)
    return {
        "a": float(a_weight),
        "counterfactual_runs": int(runs),
        "dynamic_confidence": float(dynamic_avg),
        "total_confidence": float(total_confidence),
    }


 

def safe_get(d, path):
    cur = d
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur

try:
    from fact_prediction import (
        process_all_fact_predictions,
        MODEL_NAMES,
        calculate_sofa_scores_from_features,
    )
except Exception:
    process_all_fact_predictions = None
    MODEL_NAMES = [
        "gemma3:12b",
        "mistral:7b", 
        "qwen3:4b",
        "qwen3:30b",
        "deepseek-r1:32b",
        "medllama2:latest"
    ]
    def calculate_sofa_scores_from_features(features):
        return {}


def run_prediction(model_name, patient_data, output_dir="./output"):
    """Run prediction with specified model and patient data"""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if not AdaptiveExperimentAgent:
            desc = patient_data.get("input_description", "")
            if not desc:
                raise ValueError("缺少患者描述 input_description")
            patient_id = extract_patient_id(desc, patient_data)
            if not patient_id or patient_id == "unknown":
                raise ValueError("无法从患者描述提取患者编号")
            intervention = patient_data.get("intervention")
            if not intervention:
                raise ValueError("缺少干预措施 intervention")
            actual_vital_signs = patient_data.get("output_summary")
            if actual_vital_signs is None:
                raise ValueError("缺少 output_summary 实际生命体征数据")
            fallback_features = generate_fallback_sofa_features(actual_vital_signs) or {}
            sofa_scores_series = {}
            hourly_sofa_totals = {}
            if isinstance(fallback_features, dict) and fallback_features:
                try:
                    sofa_scores_series = calculate_sofa_scores_from_features(fallback_features) or {}
                    length = 0
                    for values in sofa_scores_series.values():
                        if isinstance(values, list) and len(values) > length:
                            length = len(values)
                    if length > 0:
                        for i in range(length):
                            total_i = 0
                            total_i += int(sofa_scores_series.get("sofa_respiration", [0]*length)[i]) if i < len(sofa_scores_series.get("sofa_respiration", [])) else 0
                            total_i += int(sofa_scores_series.get("sofa_coagulation", [0]*length)[i]) if i < len(sofa_scores_series.get("sofa_coagulation", [])) else 0
                            total_i += int(sofa_scores_series.get("sofa_liver", [0]*length)[i]) if i < len(sofa_scores_series.get("sofa_liver", [])) else 0
                            total_i += int(sofa_scores_series.get("sofa_cardiovascular", [0]*length)[i]) if i < len(sofa_scores_series.get("sofa_cardiovascular", [])) else 0
                            total_i += int(sofa_scores_series.get("sofa_cns", [0]*length)[i]) if i < len(sofa_scores_series.get("sofa_cns", [])) else 0
                            total_i += int(sofa_scores_series.get("sofa_renal", [0]*length)[i]) if i < len(sofa_scores_series.get("sofa_renal", [])) else 0
                            hourly_sofa_totals[str(i)] = int(total_i)
                except Exception:
                    sofa_scores_series = {}
                    hourly_sofa_totals = {}
            static_sofa_scores = None
            if sofa_scores_series:
                try:
                    last_idx = max(len(v) for v in sofa_scores_series.values() if isinstance(v, list)) - 1
                    static_sofa_scores = {
                        "sofa_respiration": sofa_scores_series.get("sofa_respiration", [0])[last_idx] if sofa_scores_series.get("sofa_respiration") else 0,
                        "sofa_coagulation": sofa_scores_series.get("sofa_coagulation", [0])[last_idx] if sofa_scores_series.get("sofa_coagulation") else 0,
                        "sofa_liver": sofa_scores_series.get("sofa_liver", [0])[last_idx] if sofa_scores_series.get("sofa_liver") else 0,
                        "sofa_cardiovascular": sofa_scores_series.get("sofa_cardiovascular", [0])[last_idx] if sofa_scores_series.get("sofa_cardiovascular") else 0,
                        "sofa_cns": sofa_scores_series.get("sofa_cns", [0])[last_idx] if sofa_scores_series.get("sofa_cns") else 0,
                        "sofa_renal": sofa_scores_series.get("sofa_renal", [0])[last_idx] if sofa_scores_series.get("sofa_renal") else 0,
                        "sofa_total": hourly_sofa_totals.get(str(last_idx), 0)
                    }
                except Exception:
                    static_sofa_scores = None
            def _map_risk_level_from_sofa_total(total):
                try:
                    t = float(total)
                except Exception:
                    return "unknown"
                if t < 6:
                    return "low"
                if t <= 9:
                    return "moderate"
                if t <= 12:
                    return "high"
                return "critical"
            ia = {
                "risk_level": _map_risk_level_from_sofa_total(static_sofa_scores.get("sofa_total") if isinstance(static_sofa_scores, dict) else None),
                "sofa_related_features": fallback_features,
                "reasoning": ""
            }
            trimmed_prediction = {
                "intervention_analysis": ia
            }
            if isinstance(sofa_scores_series, dict) and sofa_scores_series:
                trimmed_prediction["sofa_scores_series"] = sofa_scores_series
            if isinstance(hourly_sofa_totals, dict) and hourly_sofa_totals:
                trimmed_prediction["hourly_sofa_totals"] = hourly_sofa_totals
            if isinstance(static_sofa_scores, dict) and static_sofa_scores:
                trimmed_prediction["sofa_scores"] = static_sofa_scores
            safe_model_dir_name = model_name.replace(':', '_')
            model_dir = os.path.join(output_dir, safe_model_dir_name)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            output_file = os.path.join(model_dir, f"predict_{patient_id}_{safe_model_dir_name}.json")
            trust_info = _get_model_trust_info(model_name)
            prediction_data = {
                "patient_id": patient_id,
                "model_name": model_name,
                "input_description": patient_data.get("input_description", ""),
                "intervention": patient_data.get("intervention", ""),
                "dynamic_confidence": trust_info.get("dynamic_confidence", 0.0),
                "total_confidence": trust_info.get("total_confidence", 1.0),
                "prediction": trimmed_prediction,
                "timestamp": time.time()
            }
            if isinstance(hourly_sofa_totals, dict) and hourly_sofa_totals:
                prediction_data["hourly_sofa_totals"] = hourly_sofa_totals
            if isinstance(static_sofa_scores, dict) and static_sofa_scores:
                prediction_data["predicted_sofa_scores"] = static_sofa_scores
            if isinstance(sofa_scores_series, dict) and sofa_scores_series:
                prediction_data["predicted_sofa_scores_series"] = sofa_scores_series
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(prediction_data, f, ensure_ascii=False, indent=2)
            print(f"Prediction saved to: {output_file}")
            return output_file
            
        configure_dspy(model_name)
        agent = AdaptiveExperimentAgent()
        try:
            agent.apply_model_instructions(model_name)
        except Exception:
            pass
        try:
            agent.current_model = model_name
        except Exception:
            pass
        
        # Extract patient ID for filename
        desc = patient_data.get("input_description", "")
        if not desc:
            raise ValueError("缺少患者描述 input_description")
        patient_id = extract_patient_id(desc, patient_data)
        if not patient_id or patient_id == "unknown":
            raise ValueError("无法从患者描述提取患者编号")
        
        # Generate prediction
        print(f"Running prediction for patient {patient_id} with model {model_name}")
        intervention = patient_data.get("intervention")
        if not intervention:
            raise ValueError("缺少干预措施 intervention")
        output_summary = patient_data.get("output_summary")
        if output_summary is None:
            raise ValueError("缺少 output_summary 实际生命体征数据")
        prediction = agent(
            patient_sepsis_summary=desc,
            intervention_and_risk=intervention,
            actual_vital_signs=output_summary
        )

        if prediction:
            # 序列化预测对象，保证可写入JSON
            def _to_serializable(obj):
                try:
                    if obj is None:
                        return None
                    if isinstance(obj, (str, int, float, bool)):
                        return obj
                    if isinstance(obj, (list, tuple, set)):
                        return [_to_serializable(x) for x in obj]
                    if isinstance(obj, dict):
                        return {str(k): _to_serializable(v) for k, v in obj.items()}
                    # 处理含有 __dict__ 的对象（如 dspy.Prediction）
                    if hasattr(obj, "__dict__"):
                        return {str(k): _to_serializable(v) for k, v in obj.__dict__.items()}
                    # 处理numpy类型
                    try:
                        import numpy as _np
                        if isinstance(obj, (_np.integer,)):
                            return int(obj)
                        if isinstance(obj, (_np.floating,)):
                            return float(obj)
                        if isinstance(obj, (_np.ndarray,)):
                            return _to_serializable(obj.tolist())
                    except Exception:
                        pass
                    # 兜底：字符串表示
                    return str(obj)
                except Exception:
                    return str(obj)

            serialized_prediction = _to_serializable(prediction)

            # 仅保留必要字段，裁剪冗余结构

        # 提取核心：SOFA相关特征
        sofa_related_features = (
            safe_get(serialized_prediction, ["intervention_analysis", "sofa_related_features"])
            or safe_get(serialized_prediction, ["stages", "分析干预措施和风险", "output", "sofa_related_features"])
            or safe_get(serialized_prediction, ["_store", "reasoning_steps", "分析干预措施和风险", "output", "sofa_related_features"])
            or safe_get(serialized_prediction, ["_store", "reasoning_steps", "AnalyzeInterventionAndRisk", "output", "sofa_related_features"])
            or {}
        )

        # 提取结果摘要（可选）
        predicted_outcome = (
            safe_get(serialized_prediction, ["intervention_analysis", "predicted_outcome"])
            or safe_get(serialized_prediction, ["stages", "分析干预措施和风险", "output", "predicted_outcome"])
            or "unknown"
        )
        risk_level = (
            safe_get(serialized_prediction, ["intervention_analysis", "risk_level"])
            or safe_get(serialized_prediction, ["stages", "分析干预措施和风险", "output", "risk_level"])
            or "unknown"
        )
        # 额外提取推理文本（reasoning），优先来自干预分析阶段，其次来自生成临床报告阶段
        analysis_reasoning = (
            safe_get(serialized_prediction, ["intervention_analysis", "reasoning"]) or
            safe_get(serialized_prediction, ["stages", "分析干预措施和风险", "output", "reasoning"]) or
            safe_get(serialized_prediction, ["_store", "reasoning_steps", "分析干预措施和风险", "output", "reasoning"]) or
            safe_get(serialized_prediction, ["_store", "reasoning_steps", "AnalyzeInterventionAndRisk", "output", "reasoning"]) or
            safe_get(serialized_prediction, ["stages", "生成临床报告", "output", "reasoning"]) or
            safe_get(serialized_prediction, ["_store", "reasoning_steps", "生成临床报告", "output", "reasoning"]) or
            ""
        )
        potential_risks = (
            safe_get(serialized_prediction, ["intervention_analysis", "potential_risks"])
            or safe_get(serialized_prediction, ["stages", "分析干预措施和风险", "output", "potential_risks"])
            or []
        )
        if not isinstance(potential_risks, list):
            potential_risks = [potential_risks] if potential_risks else []

        # 计算SOFA评分的时间序列（从特征派生）
        sofa_scores_series = {}
        hourly_sofa_totals = {}
        if isinstance(sofa_related_features, dict) and sofa_related_features:
            try:
                sofa_scores_series = calculate_sofa_scores_from_features(sofa_related_features) or {}
                # 计算总分的时间序列
                # 统一长度
                length = 0
                for values in sofa_scores_series.values():
                    if isinstance(values, list) and len(values) > length:
                        length = len(values)
                if length > 0:
                    for i in range(length):
                        total_i = 0
                        total_i += int(sofa_scores_series.get("sofa_respiration", [0]*length)[i]) if i < len(sofa_scores_series.get("sofa_respiration", [])) else 0
                        total_i += int(sofa_scores_series.get("sofa_coagulation", [0]*length)[i]) if i < len(sofa_scores_series.get("sofa_coagulation", [])) else 0
                        total_i += int(sofa_scores_series.get("sofa_liver", [0]*length)[i]) if i < len(sofa_scores_series.get("sofa_liver", [])) else 0
                        total_i += int(sofa_scores_series.get("sofa_cardiovascular", [0]*length)[i]) if i < len(sofa_scores_series.get("sofa_cardiovascular", [])) else 0
                        total_i += int(sofa_scores_series.get("sofa_cns", [0]*length)[i]) if i < len(sofa_scores_series.get("sofa_cns", [])) else 0
                        total_i += int(sofa_scores_series.get("sofa_renal", [0]*length)[i]) if i < len(sofa_scores_series.get("sofa_renal", [])) else 0
                        # 使用字符串索引以保证JSON键类型
                        hourly_sofa_totals[str(i)] = int(total_i)
            except Exception:
                sofa_scores_series = {}
                hourly_sofa_totals = {}

        # 同时保留一个用于柱状图展示的静态截面（最后一个时间点）
        static_sofa_scores = None
        if sofa_scores_series:
            try:
                # 选取最后一个时间点
                last_idx = max(len(v) for v in sofa_scores_series.values() if isinstance(v, list)) - 1
                static_sofa_scores = {
                    "sofa_respiration": sofa_scores_series.get("sofa_respiration", [0])[last_idx] if sofa_scores_series.get("sofa_respiration") else 0,
                    "sofa_coagulation": sofa_scores_series.get("sofa_coagulation", [0])[last_idx] if sofa_scores_series.get("sofa_coagulation") else 0,
                    "sofa_liver": sofa_scores_series.get("sofa_liver", [0])[last_idx] if sofa_scores_series.get("sofa_liver") else 0,
                    "sofa_cardiovascular": sofa_scores_series.get("sofa_cardiovascular", [0])[last_idx] if sofa_scores_series.get("sofa_cardiovascular") else 0,
                    "sofa_cns": sofa_scores_series.get("sofa_cns", [0])[last_idx] if sofa_scores_series.get("sofa_cns") else 0,
                    "sofa_renal": sofa_scores_series.get("sofa_renal", [0])[last_idx] if sofa_scores_series.get("sofa_renal") else 0,
                    "sofa_total": hourly_sofa_totals.get(str(last_idx), 0)
                }
            except Exception:
                static_sofa_scores = None

        # 若 risk_level 仍为 unknown，基于最后时刻 SOFA 总分进行映射作为回退
        def _map_risk_level_from_sofa_total(total):
            try:
                t = float(total)
            except Exception:
                return "unknown"
            # 简单分级：<6 低风险；6-9 中等；10-12 高；>=13 极高
            if t < 6:
                return "low"
            if t <= 9:
                return "moderate"
            if t <= 12:
                return "high"
            return "critical"

        # 写入精简结构
        # 当缺少推理文本时，基于SOFA评分与风险等级合成一个简要推理，避免下游出现"unknown"
        def _compose_reasoning(static_scores, features, level, outcome, risks):
            try:
                parts = []
                if isinstance(static_scores, dict) and static_scores:
                    total = static_scores.get("sofa_total")
                    comps = [
                        f"呼吸{static_scores.get('sofa_respiration', 0)}",
                        f"凝血{static_scores.get('sofa_coagulation', 0)}",
                        f"肝功能{static_scores.get('sofa_liver', 0)}",
                        f"循环{static_scores.get('sofa_cardiovascular', 0)}",
                        f"中枢神经{static_scores.get('sofa_cns', 0)}",
                        f"肾脏{static_scores.get('sofa_renal', 0)}",
                    ]
                    parts.append(f"SOFA总分{total}（" + "，".join(comps) + "）")
                if isinstance(features, dict) and features:
                    keys = list(features.keys())
                    # 仅列出核心特征以免过长
                    core = [k for k in keys if k in (
                        "pao2_fio2_ratio","platelet","bilirubin_total","vasopressor_rate","gcs_total","creatinine","urine_output_ml","mechanical_ventilation"
                    )]
                    if core:
                        parts.append("关键SOFA相关特征: " + ", ".join(core))
                if outcome and outcome != "unknown":
                    parts.append(f"预测结果: {outcome}")
                if isinstance(risks, list) and risks:
                    parts.append("潜在风险: " + ", ".join(map(str, risks)))
                if level and level != "unknown":
                    parts.append(f"综合判定风险等级为 {level}")
                text = "；".join(parts)
                return text if text else None
            except Exception:
                return None

        if not analysis_reasoning:
            analysis_reasoning = _compose_reasoning(static_sofa_scores, sofa_related_features, risk_level, predicted_outcome, potential_risks) or ""

        # 构建干预分析结构，剔除空值/未知值
        ia = {}
        if predicted_outcome and predicted_outcome != "unknown":
            ia["predicted_outcome"] = predicted_outcome
        if risk_level:
            ia["risk_level"] = risk_level
        if isinstance(potential_risks, list) and potential_risks:
            ia["potential_risks"] = potential_risks
        if isinstance(sofa_related_features, dict) and sofa_related_features:
            ia["sofa_related_features"] = sofa_related_features
        if analysis_reasoning:
            ia["reasoning"] = analysis_reasoning

        trimmed_prediction = {
            "intervention_analysis": ia
        }
        # 写入时间序列和静态截面
        if isinstance(sofa_scores_series, dict) and sofa_scores_series:
            trimmed_prediction["sofa_scores_series"] = sofa_scores_series
        if isinstance(hourly_sofa_totals, dict) and hourly_sofa_totals:
            trimmed_prediction["hourly_sofa_totals"] = hourly_sofa_totals
        if isinstance(static_sofa_scores, dict) and static_sofa_scores:
            trimmed_prediction["sofa_scores"] = static_sofa_scores

        # 保存精简后的预测结果到对应模型名称的子文件夹
        safe_model_dir_name = model_name.replace(':', '_')
        model_dir = os.path.join(output_dir, safe_model_dir_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        output_file = os.path.join(model_dir, f"predict_{patient_id}_{safe_model_dir_name}.json")
        trust_info = _get_model_trust_info(model_name)
        prediction_data = {
            "patient_id": patient_id,
            "model_name": model_name,
            "input_description": patient_data.get("input_description", ""),
            "intervention": patient_data.get("intervention", ""),
            "dynamic_confidence": trust_info.get("dynamic_confidence", 0.0),
            "total_confidence": trust_info.get("total_confidence", 1.0),
            "prediction": trimmed_prediction,
            "timestamp": time.time()
        }
        # 为GUI展示与下游分析提供顶层的时间序列与静态SOFA评分
        if isinstance(hourly_sofa_totals, dict) and hourly_sofa_totals:
            prediction_data["hourly_sofa_totals"] = hourly_sofa_totals
        if isinstance(static_sofa_scores, dict) and static_sofa_scores:
            prediction_data["predicted_sofa_scores"] = static_sofa_scores
            # 如果 risk_level 仍为 unknown，则根据静态总分回退映射
            try:
                if isinstance(trimmed_prediction, dict):
                    ia = trimmed_prediction.get("intervention_analysis", {}) or {}
                    if isinstance(ia, dict) and ia.get("risk_level") in (None, "", "unknown"):
                        mapped = _map_risk_level_from_sofa_total(static_sofa_scores.get("sofa_total"))
                        if mapped and mapped != "unknown":
                            ia["risk_level"] = mapped
                            trimmed_prediction["intervention_analysis"] = ia
            except Exception:
                pass
        if isinstance(sofa_scores_series, dict) and sofa_scores_series:
            prediction_data["predicted_sofa_scores_series"] = sofa_scores_series
        
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(prediction_data, f, ensure_ascii=False, indent=2)

                print(f"Prediction saved to: {output_file}")
                return output_file
        else:
            print("Prediction failed")
            return None
            
    except Exception as e:
        print(f"Error in run_prediction: {str(e)}")
        return None


def run_evaluation(model_name, prediction_file):
    """Run evaluation on prediction file and persist evaluation report"""
    try:
        if not os.path.exists(prediction_file):
            print(f"Prediction file not found: {prediction_file}")
            return None
            
        with open(prediction_file, 'r', encoding='utf-8') as f:
            prediction_data = json.load(f)
        
        patient_id = prediction_data.get("patient_id", "unknown")
        prediction_model_name = prediction_data.get("model_name")
        print(f"Running evaluation for patient {patient_id} with model {model_name}")

        # Prepare inputs for evaluator
        input_description = prediction_data.get("input_description", "")
        intervention = prediction_data.get("intervention", "")
        predicted = prediction_data.get("prediction", {})

        # 提取预测的SOFA相关特征用于评估报告

        

        # Try multiple paths to locate predicted SOFA features
        predicted_sofa_features = None
        # 1) Top-level intervention_analysis
        predicted_sofa_features = safe_get(predicted, ["intervention_analysis", "sofa_related_features"])
        # 2) Stages (中文键)
        if not isinstance(predicted_sofa_features, dict) or not predicted_sofa_features:
            predicted_sofa_features = safe_get(predicted, ["stages", "分析干预措施和风险", "output", "sofa_related_features"])
        # 3) Internal store.reasoning_steps (中文键)
        if not isinstance(predicted_sofa_features, dict) or not predicted_sofa_features:
            predicted_sofa_features = safe_get(predicted, ["_store", "reasoning_steps", "分析干预措施和风险", "output", "sofa_related_features"])
        # 4) Internal store.reasoning_steps (英文键)
        if not isinstance(predicted_sofa_features, dict) or not predicted_sofa_features:
            predicted_sofa_features = safe_get(predicted, ["_store", "reasoning_steps", "AnalyzeInterventionAndRisk", "output", "sofa_related_features"])
        # Fallback to empty dict
        if not isinstance(predicted_sofa_features, dict):
            predicted_sofa_features = {}

        # Run evaluation with correct signature
        evaluation_result = evaluate_with_ollama(
            model_name,
            input_description,
            intervention,
            predicted_sofa_features
        )
        
        if evaluation_result:
            # 兼容字典/字符串输出，避免类型错误
            model_output_text = (
                evaluation_result.get("model_output")
                if isinstance(evaluation_result, dict) else str(evaluation_result)
            )
            confidence = extract_model_confidence(model_output_text)
            # 保存评估报告到对应模型目录
            try:
                evaluator_trust = _get_model_trust_info(model_name)
                prediction_trust = _get_model_trust_info(prediction_model_name) if prediction_model_name else None
                save_evaluation_report(
                    model_name,
                    input_description,
                    intervention,
                    predicted_sofa_features,
                    confidence,
                    evaluation_result,
                    prediction_model_name=prediction_model_name,
                    evaluator_trust=evaluator_trust,
                    prediction_trust=prediction_trust,
                )
            except Exception as _e:
                print(f"Warning: failed to save evaluation report: {_e}")
            print(f"Evaluation completed. Confidence: {confidence}")
            return confidence
        else:
            print("Evaluation failed")
            return None
            
    except Exception as e:
        print(f"Error in run_evaluation: {str(e)}")
        return None


def select_best_prediction(model_names, patient_data):
    try:
        if len(model_names) < 3:
            print("Need at least 3 models for best selection")
            return None
        print(f"Running predictions with models: {model_names}")

        initial_trust = _load_initial_trust_scores()
        dynamic_state = _bootstrap_dynamic_trust_state(model_names)
        a_weight = float(dynamic_state.get("_meta", {}).get("a", 1.0))
        runs = dynamic_state.get("_meta", {}).get("counterfactual_runs", 0)
        if not isinstance(runs, int) or runs < 0:
            runs = 0

        model_trust_state = {}
        models_dyn = dynamic_state.get("models") if isinstance(dynamic_state.get("models"), dict) else {}

        # Get experience integration instance if available
        experience_integration = None
        if get_experience_integration is not None:
            try:
                experience_integration = get_experience_integration()
            except Exception as e:
                print(f"Warning: Failed to get experience integration instance: {e}")

        for m in model_names:
            init = float(initial_trust.get(m, 1.0))

            # Adjust initial confidence based on experience knowledge base
            adjusted_init = init
            if experience_integration is not None and experience_integration.integration_enabled:
                try:
                    adjusted_init = experience_integration.adjust_model_confidence(
                        model_name=m,
                        base_confidence=init,
                        patient_data=patient_data
                    )
                    print(f"Model {m}: initial confidence {init:.3f} -> experience-adjusted {adjusted_init:.3f}")
                except Exception as e:
                    print(f"Warning: Failed to adjust model confidence using experience: {e}")

            d = models_dyn.get(m) if isinstance(models_dyn.get(m), dict) else {}
            dynamic_sum = d.get("dynamic_sum", 0.0)
            dynamic_count = d.get("dynamic_count", 0)
            completed_predictions = d.get("completed_predictions", 0)
            if not isinstance(dynamic_sum, (int, float)):
                dynamic_sum = 0.0
            if not isinstance(dynamic_count, int) or dynamic_count < 0:
                dynamic_count = 0
            if not isinstance(completed_predictions, int) or completed_predictions < 0:
                completed_predictions = 0
            model_trust_state[m] = {
                "initial": adjusted_init,  # Use experience-adjusted initial confidence
                "dynamic_sum": float(dynamic_sum),
                "dynamic_count": int(dynamic_count),
                "completed_predictions": int(completed_predictions),
            }

        def compute_total_confidence(model):
            st = model_trust_state.get(model) or {}
            initial = st.get("initial", 1.0)
            dynamic_sum = st.get("dynamic_sum", 0.0)
            dynamic_count = st.get("dynamic_count", 0)
            if not isinstance(initial, (int, float)):
                initial = 1.0
            if not isinstance(dynamic_sum, (int, float)):
                dynamic_sum = 0.0
            if not isinstance(dynamic_count, int) or dynamic_count <= 0:
                dynamic_count = 0
            dynamic_confidence = (float(dynamic_sum) / dynamic_count) if dynamic_count > 0 else 0.0
            return float(a_weight) * float(initial) + (1.0 - float(a_weight)) * float(dynamic_confidence)

        def trust_of_pre(model):
            return compute_total_confidence(model)

        predictions = []
        for pred_model in model_names:
            print(f"Running prediction with {pred_model}...")
            prediction_file = run_prediction(pred_model, patient_data)
            if prediction_file:
                # 读取预测数据以便在前端展示推理与评分
                try:
                    with open(prediction_file, 'r', encoding='utf-8') as pf:
                        pred_data = json.load(pf)
                except Exception:
                    pred_data = {}
                predictions.append({
                    "model": pred_model,
                    "file": prediction_file,
                    "prediction_data": pred_data
                })

        if not predictions:
            print("No successful predictions")
            return None

        eval_matrix = {p["model"]: {} for p in predictions}
        for p in predictions:
            for eval_model in model_names:
                conf = run_evaluation(eval_model, p["file"]) if p["file"] else 0.0
                conf = float(conf) if isinstance(conf, (int, float)) else 0.0
                eval_matrix[p["model"]][eval_model] = conf

        dynamic_scores = {}
        for p in predictions:
            pred_model = p["model"]
            numerator = 0.0
            denom = 0.0
            for eval_model in model_names:
                if eval_model == pred_model:
                    continue
                w = float(trust_of_pre(eval_model))
                denom += w
                numerator += w * float(eval_matrix.get(pred_model, {}).get(eval_model, 0.0))
            dynamic_scores[pred_model] = (numerator / denom) if denom > 0 else 0.0

        for model, dyn in dynamic_scores.items():
            st = model_trust_state.get(model)
            if not isinstance(st, dict):
                st = {"initial": 1.0, "dynamic_sum": 0.0, "dynamic_count": 0, "completed_predictions": 0}
                model_trust_state[model] = st
            st["dynamic_sum"] = float(st.get("dynamic_sum", 0.0)) + float(dyn)
            st["dynamic_count"] = int(st.get("dynamic_count", 0)) + 1
            st["completed_predictions"] = int(st.get("completed_predictions", 0)) + 1

        a_weight = max(0.2, float(a_weight) - 0.05)
        runs = int(runs) + 1
        dynamic_state = {
            "_meta": {"schema_version": 2, "dynamic_method": "weighted_avg_v2", "a": float(a_weight), "counterfactual_runs": int(runs)},
            "models": {
                m: {
                    "dynamic_sum": float(st.get("dynamic_sum", 0.0)),
                    "dynamic_count": int(st.get("dynamic_count", 0)),
                    "completed_predictions": int(st.get("completed_predictions", 0)),
                }
                for m, st in model_trust_state.items()
            },
        }
        _save_dynamic_trust_state(dynamic_state)

        def trust_of(model):
            return compute_total_confidence(model)

        def dynamic_of(model):
            st = model_trust_state.get(model) or {}
            dynamic_sum = st.get("dynamic_sum", 0.0)
            dynamic_count = st.get("dynamic_count", 0)
            if not isinstance(dynamic_sum, (int, float)):
                dynamic_sum = 0.0
            if not isinstance(dynamic_count, int) or dynamic_count <= 0:
                return 0.0
            return float(dynamic_sum) / int(dynamic_count)

        evaluator_scores = {m: {"sum": 0.0, "count": 0} for m in model_names}
        for p in predictions:
            pred_model = p["model"]
            for eval_model in model_names:
                conf = float(eval_matrix.get(pred_model, {}).get(eval_model, 0.0))
                evaluator_scores[eval_model]["sum"] += trust_of(eval_model) * conf
                evaluator_scores[eval_model]["count"] += 1

        averaged = []
        for m, sc in evaluator_scores.items():
            avg = (sc["sum"] / sc["count"]) if sc["count"] > 0 else 0.0
            averaged.append((m, avg))
        averaged.sort(key=lambda x: x[1], reverse=True)
        selected_evaluators = [x[0] for x in averaged[:3]]

        results = []
        for p in predictions:
            pred_model = p["model"]
            prediction_file = p["file"]
            total_score = trust_of(pred_model) * 1.0
            eval_breakdown = [{
                "evaluator": pred_model,
                "score": 1.0,
                "trust_weight": trust_of(pred_model),
                "dynamic_confidence": dynamic_of(pred_model),
                "total_confidence": trust_of(pred_model),
                "weighted_score": trust_of(pred_model) * 1.0
            }]
            # 读取 patient_id 用于关联评估报告
            try:
                patient_id = p.get("prediction_data", {}).get("patient_id")
            except Exception:
                patient_id = None
            for eval_model in selected_evaluators:
                if eval_model == pred_model:
                    continue
                eval_conf = float(eval_matrix.get(pred_model, {}).get(eval_model, 0.0))
                weighted = trust_of(eval_model) * eval_conf
                total_score += weighted
                # 查找并附加评估报告与模型输出（若存在）
                eval_entry = {
                    "evaluator": eval_model,
                    "score": eval_conf,
                    "trust_weight": trust_of(eval_model),
                    "dynamic_confidence": dynamic_of(eval_model),
                    "total_confidence": trust_of(eval_model),
                    "weighted_score": weighted
                }
                try:
                    if patient_id:
                        import os, glob
                        base_dir = os.path.join("output", eval_model.split(":")[0])
                        pattern = os.path.join(base_dir, f"evaluator_{patient_id}_{eval_model.replace(':','_')}_*.json")
                        files = sorted(glob.glob(pattern), key=lambda x: os.path.getmtime(x), reverse=True)
                        if files:
                            with open(files[0], 'r', encoding='utf-8') as rf:
                                report = json.load(rf)
                            # 提取模型输出与关键理由，避免过大payload可截断
                            mo = report.get("evaluation", {}).get("model_output") if isinstance(report.get("evaluation"), dict) else None
                            if isinstance(mo, str):
                                eval_entry["model_output"] = mo[:1200]
                            eval_entry["report_file"] = files[0]
                except Exception:
                    pass
                eval_breakdown.append(eval_entry)
            results.append({
                "model": pred_model,
                "prediction_file": prediction_file,
                "dynamic_confidence": dynamic_of(pred_model),
                "total_confidence": trust_of(pred_model),
                "total_score": total_score,
                "eval_breakdown": eval_breakdown,
                "prediction_data": p.get("prediction_data", {})
            })

        best_result = max(results, key=lambda x: x["total_score"]) if results else None
        if not best_result:
            print("Selection failed")
            return None

        with open(best_result["prediction_file"], 'r', encoding='utf-8') as f:
            prediction_data = json.load(f)

        top5_models = [x[0] for x in averaged[:5]]

        risk_analyses = []
        if AdaptiveExperimentAgent:
            for risk_model in top5_models:
                try:
                    configure_dspy(risk_model)
                    agent = AdaptiveExperimentAgent()
                    input_description = prediction_data.get("input_description", "")
                    inter_text = prediction_data.get("intervention", "") or "干预措施：立即启动去甲肾上腺素输注，初始剂量为0.25μg・kg⁻¹・min⁻¹"
                    risk_out = agent.shock_risk_assessment(patient_document=input_description)
                    analysis_out = agent.analyze_intervention_and_risk(
                        current_sepsis_state_summary=getattr(risk_out, "current_sepsis_state_summary", ""),
                        intervention_and_risk=inter_text
                    )
                    lvl = getattr(analysis_out, "risk_level", "unknown")
                    score = getattr(analysis_out, "risk_score", None)
                    if not isinstance(score, (int, float)):
                        extracted = extract_model_confidence(getattr(analysis_out, "reasoning", ""))
                        score = float(extracted) if isinstance(extracted, (int, float)) else None
                    if not isinstance(score, (int, float)):
                        m = str(lvl).lower()
                        if m in ("low", "极低风险"):
                            score = 0.25
                        elif m in ("medium", "moderate", "中等风险"):
                            score = 0.5
                        elif m in ("high", "高风险"):
                            score = 0.75
                        elif m in ("critical", "极高风险"):
                            score = 0.9
                        else:
                            score = 0.5
                    risk_analyses.append({
                        "model": risk_model,
                        "risk_level": lvl,
                        "risk_score": float(score),
                        "reasoning": getattr(analysis_out, "reasoning", "")
                    })
                except Exception as _e:
                    risk_analyses.append({
                        "model": risk_model,
                        "risk_level": "unknown",
                        "risk_score": 0.5,
                        "reasoning": str(_e)
                    })

        final_weight = sum(trust_of(r["model"]) for r in risk_analyses) or 1.0
        final_score = sum(trust_of(r["model"]) * r["risk_score"] for r in risk_analyses) / final_weight if risk_analyses else 0.0
        if final_score < 0.33:
            final_level = "low"
        elif final_score < 0.66:
            final_level = "moderate"
        elif final_score < 0.85:
            final_level = "high"
        else:
            final_level = "critical"

        return {
            "prediction_model": best_result["model"],
            "evaluation_models": selected_evaluators,
            "total_confidence": best_result["total_score"],
            "trust_a": float(a_weight),
            "prediction_model_dynamic_confidence": dynamic_of(best_result["model"]),
            "prediction_model_total_confidence": trust_of(best_result["model"]),
            "evaluation_models_trust": {m: {"dynamic_confidence": dynamic_of(m), "total_confidence": trust_of(m)} for m in selected_evaluators},
            "prediction_file": best_result["prediction_file"],
            "prediction_data": prediction_data,
            "eval_breakdown": best_result.get("eval_breakdown", []),
            "all_results": results,
            "selected_evaluators": selected_evaluators,
            "evaluator_scores": {m: next((a for x, a in averaged if x == m), 0.0) for m in selected_evaluators},
            "risk_analysis_models": top5_models,
            "per_model_risk": risk_analyses,
            "final_weighted_risk_score": final_score,
            "final_weighted_risk_level": final_level
        }
    except Exception as e:
        print(f"Error in select_best_prediction: {str(e)}")
        return None


def save_best_prediction_result(best_result, output_file):
    """Save best prediction result to file, including SOFA features and reasoning"""
    try:
        # 从 best_result 中抽取最小必要信息
        prediction_data = best_result.get("prediction_data", {}) or {}
        input_description = prediction_data.get("input_description", "")
        intervention = prediction_data.get("intervention", "")
        pred = prediction_data.get("prediction", {}) or {}

        

        # 取时间序列SOFA评分（优先），以及最后时刻的静态评分（用于GUI柱状图）
        sofa_scores_series = (
            safe_get(pred, ["sofa_scores_series"])
            or {}
        )
        hourly_totals = (
            safe_get(pred, ["hourly_sofa_totals"]) or {}
        )
        sofa_scores_last = (
            safe_get(pred, ["sofa_scores"]) 
            or safe_get(pred, ["_store", "reasoning_steps", "生成临床报告", "input", "sofa_scores"]) 
            or safe_get(pred, ["stages", "生成临床报告", "input", "sofa_scores"]) 
        )

        # 提取SOFA相关生理指标（特征）与推理文本
        predicted_sofa_features = (
            safe_get(pred, ["intervention_analysis", "sofa_related_features"]) or
            safe_get(pred, ["stages", "分析干预措施和风险", "output", "sofa_related_features"]) or
            safe_get(pred, ["_store", "reasoning_steps", "分析干预措施和风险", "output", "sofa_related_features"]) or
            safe_get(pred, ["_store", "reasoning_steps", "AnalyzeInterventionAndRisk", "output", "sofa_related_features"]) or
            {}
        )
        # 提取推理文本（reasoning），而非 predicted_outcome
        predicted_outcome_reasoning = (
            safe_get(pred, ["intervention_analysis", "reasoning"]) or
            safe_get(pred, ["stages", "分析干预措施和风险", "output", "reasoning"]) or
            safe_get(pred, ["_store", "reasoning_steps", "分析干预措施和风险", "output", "reasoning"]) or
            safe_get(pred, ["stages", "生成临床报告", "output", "reasoning"]) or
            ""
        )
        risk_level = (
            safe_get(pred, ["intervention_analysis", "risk_level"]) or
            safe_get(pred, ["stages", "分析干预措施和风险", "output", "risk_level"]) or
            ""
        )
        potential_risks = (
            safe_get(pred, ["intervention_analysis", "potential_risks"]) or
            safe_get(pred, ["stages", "分析干预措施和风险", "output", "potential_risks"]) or
            []
        )

        minimal_payload = {
            "input_description": input_description,
            "intervention": intervention,
            "prediction_model": best_result.get("prediction_model"),
            "evaluation_models": best_result.get("evaluation_models", []),
            "total_confidence": best_result.get("total_confidence", 0),
            "trust_a": best_result.get("trust_a"),
            "prediction_model_dynamic_confidence": best_result.get("prediction_model_dynamic_confidence"),
            "prediction_model_total_confidence": best_result.get("prediction_model_total_confidence"),
            "evaluation_models_trust": best_result.get("evaluation_models_trust"),
        }
        # 写入时间序列以及静态截面
        if isinstance(sofa_scores_series, dict) and sofa_scores_series:
            minimal_payload["predicted_sofa_scores_series"] = sofa_scores_series
        if isinstance(hourly_totals, dict) and hourly_totals:
            minimal_payload["hourly_sofa_totals"] = hourly_totals
        if isinstance(sofa_scores_last, dict) and sofa_scores_last:
            minimal_payload["predicted_sofa_scores"] = sofa_scores_last
            if "sofa_total" in sofa_scores_last:
                try:
                    minimal_payload["predicted_sofa_total"] = float(sofa_scores_last["sofa_total"])
                except Exception:
                    pass  # 非数值则跳过

        # 追加SOFA相关生理指标与推理文本
        if isinstance(predicted_sofa_features, dict) and predicted_sofa_features:
            minimal_payload["predicted_sofa_features"] = predicted_sofa_features
        if isinstance(potential_risks, list) and potential_risks:
            minimal_payload["potential_risks"] = potential_risks
        # 若风险等级缺失或为 unknown，基于总分进行回退映射
        final_risk_level = risk_level
        if not final_risk_level or final_risk_level == "unknown":
            try:
                total_val = None
                if isinstance(sofa_scores_last, dict):
                    total_val = sofa_scores_last.get("sofa_total")
                elif isinstance(hourly_totals, dict) and hourly_totals:
                    # 取最后一个时刻的总分
                    keys = sorted(hourly_totals.keys(), key=lambda x: int(x))
                    total_val = hourly_totals.get(keys[-1]) if keys else None
                def _map_level(total):
                    try:
                        t = float(total)
                    except Exception:
                        return None
                    if t < 6:
                        return "low"
                    if t <= 9:
                        return "moderate"
                    if t <= 12:
                        return "high"
                    return "critical"
                mapped_level = _map_level(total_val)
                if mapped_level:
                    final_risk_level = mapped_level
            except Exception:
                pass
        if final_risk_level:
            minimal_payload["risk_level"] = final_risk_level
        if predicted_outcome_reasoning:
            minimal_payload["reasoning"] = predicted_outcome_reasoning

        if best_result.get("selected_evaluators"):
            minimal_payload["selected_evaluators"] = best_result.get("selected_evaluators")
        if best_result.get("evaluator_scores"):
            minimal_payload["evaluator_scores"] = best_result.get("evaluator_scores")
        if best_result.get("risk_analysis_models"):
            minimal_payload["risk_analysis_models"] = best_result.get("risk_analysis_models")
        if best_result.get("per_model_risk"):
            minimal_payload["per_model_risk"] = best_result.get("per_model_risk")
        if "final_weighted_risk_score" in best_result:
            minimal_payload["final_weighted_risk_score"] = best_result.get("final_weighted_risk_score")
        if "final_weighted_risk_level" in best_result:
            minimal_payload["final_weighted_risk_level"] = best_result.get("final_weighted_risk_level")
        if best_result.get("all_results"):
            minimal_payload["all_results"] = best_result.get("all_results")
        if best_result.get("eval_breakdown"):
            minimal_payload["eval_breakdown"] = best_result.get("eval_breakdown")

        out_dir = os.path.dirname(output_file)
        if out_dir and not os.path.exists(out_dir):
            try:
                os.makedirs(out_dir)
            except Exception:
                pass
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(minimal_payload, f, ensure_ascii=False, indent=2)
        print(f"Best prediction result saved to: {output_file}")

        # Update experience knowledge base with this result
        if update_experience_from_result is not None:
            try:
                patient_data = {
                    "input_description": prediction_data.get("input_description", ""),
                    "intervention": prediction_data.get("intervention", "")
                }
                case_id = update_experience_from_result(output_file, patient_data)
                if case_id:
                    print(f"Experience knowledge base updated with case: {case_id}")
                else:
                    print("Experience knowledge base not updated (may be disabled or error)")
            except Exception as e:
                print(f"Warning: Failed to update experience knowledge base: {e}")

        return True
    except Exception as e:
        print(f"Error saving best prediction result: {str(e)}")
        return False
