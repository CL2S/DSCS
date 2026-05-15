import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.counterfactual_plan_renderer import render_counterfactual_case_report, render_physician_case_report
from src.manifold_forecasting_trainer import (
    EndToEndForecastingManifoldTrainer,
    ForecastingTrainerConfig,
)
from src.manifold_memory import ManifoldMemoryConfig
from src.persistent_memory_store import PersistentExperienceStore
from src.tsf_data import build_eicu_sepsis3_inference_sample, derive_eicu_sepsis3_feature_schema


def parse_args():
    parser = argparse.ArgumentParser(description="Run single-patient counterfactual donor retrieval and plan rendering for eICU inputs.")
    parser.add_argument("--bundle-path", required=True)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--disable-persistent-semantic-store", action="store_true")
    parser.add_argument("--counterfactual-reranker-mode-override", choices=["rule_only", "learned_linear"], default="")
    parser.add_argument("--min-donor-similarity", type=float, default=0.45)
    parser.add_argument("--min-guideline-compatibility", type=float, default=0.25)
    parser.add_argument("--min-donor-total-score", type=float, default=0.0)
    parser.add_argument("--max-missing-care-penalty", type=float, default=0.55)
    parser.add_argument("--max-contraindication-penalty", type=float, default=0.25)
    parser.add_argument("--require-positive-delta", action="store_true")
    parser.add_argument("--uncertainty-samples", type=int, default=16)
    parser.add_argument("--max-recommended-forecast-std", type=float, default=0.75)
    parser.add_argument("--min-delta-lower-bound", type=float, default=0.0)
    parser.add_argument("--disable-uncertainty-guardrail", action="store_true")
    parser.add_argument("--rollout-steps", type=int, default=1)
    parser.add_argument("--rollout-discount", type=float, default=0.70)
    return parser.parse_args()


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _resolved_path_str(path_str: str) -> str:
    return str(Path(path_str).expanduser().resolve())


def _ensure_parent_dir(path_str: str) -> None:
    if not path_str:
        return
    Path(path_str).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _selected_counterfactual_sample(sample, case_detail: Dict[str, object]):
    selected_candidate = dict(case_detail.get("selected_candidate", {}))
    selected_static = [float(value) for value in selected_candidate.get("candidate_intervention_static", [])]
    selected_sequence = [
        [float(value) for value in step]
        for step in selected_candidate.get("candidate_intervention_sequence", [])
    ]
    if not selected_static and not selected_sequence:
        return sample
    return replace(
        sample,
        intervention_static=selected_static or list(sample.intervention_static or []),
        intervention_sequence=selected_sequence or [list(step) for step in (sample.intervention_sequence or [])],
    )


def _uncertainty_guardrail_context(
    factual_uncertainty: Dict[str, object],
    counterfactual_uncertainty: Dict[str, object],
    predicted_delta: float,
) -> Dict[str, float]:
    factual_summary = dict(dict(factual_uncertainty.get("forecast", {})).get("summary", {}))
    counterfactual_summary = dict(dict(counterfactual_uncertainty.get("forecast", {})).get("summary", {}))
    factual_std = _safe_float(factual_summary.get("mean_std"))
    counterfactual_std = _safe_float(counterfactual_summary.get("mean_std"))
    delta_std = math.sqrt(max(0.0, factual_std ** 2 + counterfactual_std ** 2))
    interval_width = 1.96 * delta_std
    return {
        "factual_mean_std": float(factual_std),
        "recommended_mean_std": float(counterfactual_std),
        "delta_std": float(delta_std),
        "delta_lower_bound": float(predicted_delta - interval_width),
        "delta_upper_bound": float(predicted_delta + interval_width),
    }


def _load_bundle(bundle_path: str) -> Dict[str, object]:
    resolved = _resolved_path_str(bundle_path)
    payload = torch.load(resolved, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid bundle payload in {resolved}.")
    if str(payload.get("bundle_type", "")) != "eicu_counterfactual_inference":
        raise ValueError(f"Unsupported bundle_type={payload.get('bundle_type')}.")
    return payload


def _build_trainer(bundle_payload: Dict[str, object], device: str, enable_persistent_semantic_store: bool):
    trainer_payload = dict(bundle_payload.get("trainer_bundle", {}))
    input_sources = dict(bundle_payload.get("input_sources", {}))
    memory_config = ManifoldMemoryConfig(**dict(trainer_payload.get("memory_config", {})))
    memory_config.device = device
    trainer_config = ForecastingTrainerConfig(**dict(trainer_payload.get("trainer_config", {})))
    trainer_config.device = device
    trainer = EndToEndForecastingManifoldTrainer(
        memory_config=memory_config,
        trainer_config=trainer_config,
        static_feature_dim=int(trainer_payload.get("static_feature_dim", memory_config.static_feature_dim)),
        kg_feature_dim=int(trainer_payload.get("kg_feature_dim", 0)),
        intervention_feature_dim=int(trainer_payload.get("intervention_feature_dim", 0)),
        intervention_sequence_dim=int(trainer_payload.get("intervention_sequence_dim", 0)),
        formation_feature_dim=int(trainer_payload.get("formation_feature_dim", 0)),
    )
    trainer.load_inference_bundle(trainer_payload)
    persistent_memory = dict(bundle_payload.get("persistent_memory", {}))
    store_path = str(persistent_memory.get("store_path", "")).strip()
    if enable_persistent_semantic_store and store_path and Path(store_path).exists():
        trainer.configure_semantic_store(
            store=PersistentExperienceStore(store_path),
            top_k=int(persistent_memory.get("semantic_top_k", 3)),
        )
    labels_csv = str(input_sources.get("eicu_sepsis3_labels_csv", "")).strip()
    if labels_csv and Path(labels_csv).exists():
        trainer.enrich_intervention_store_metadata_from_labels(labels_csv)
    return trainer


def _validate_sample_dimensions(sample, bundle_payload: Dict[str, object]) -> None:
    semantics = dict(bundle_payload.get("dataset_semantics", {}))
    expected_sequence_dim = int(len(semantics.get("sequence_feature_names", [])))
    expected_patient_dim = int(len(semantics.get("patient_feature_names", [])))
    expected_intervention_dim = int(len(semantics.get("intervention_feature_names", [])))
    expected_intervention_sequence_dim = int(len(semantics.get("intervention_sequence_feature_names", [])))
    actual_sequence_dim = len(sample.sequence[0]) if sample.sequence else 0
    actual_patient_dim = len(sample.patient_static or [])
    actual_intervention_dim = len(sample.intervention_static or [])
    actual_intervention_sequence_dim = len((sample.intervention_sequence or [[0.0]])[0]) if sample.intervention_sequence else 0

    mismatches = []
    if expected_sequence_dim and actual_sequence_dim != expected_sequence_dim:
        mismatches.append(f"sequence_dim expected {expected_sequence_dim}, got {actual_sequence_dim}")
    if expected_patient_dim and actual_patient_dim != expected_patient_dim:
        mismatches.append(f"patient_static_dim expected {expected_patient_dim}, got {actual_patient_dim}")
    if expected_intervention_dim and actual_intervention_dim != expected_intervention_dim:
        mismatches.append(f"intervention_static_dim expected {expected_intervention_dim}, got {actual_intervention_dim}")
    if expected_intervention_sequence_dim and actual_intervention_sequence_dim != expected_intervention_sequence_dim:
        mismatches.append(
            f"intervention_sequence_dim expected {expected_intervention_sequence_dim}, got {actual_intervention_sequence_dim}"
        )
    if mismatches:
        raise ValueError("Inference sample dimensions do not match the trained bundle: " + "; ".join(mismatches))


def _build_sample(bundle_payload: Dict[str, object], input_payload: Dict[str, object]):
    dataset_summary = dict(bundle_payload.get("dataset_summary", {}))
    input_sources = dict(bundle_payload.get("input_sources", {}))
    semantics = dict(bundle_payload.get("dataset_semantics", {}))
    feature_schema = None
    sample_type = str(input_payload.get("sample_type", "")).strip().lower()
    if sample_type != "forecast_sample" and "sequence" not in input_payload:
        feature_schema = derive_eicu_sepsis3_feature_schema(
            labels_csv=str(input_sources.get("eicu_sepsis3_labels_csv", "")),
            trajectory_csv=str(input_sources.get("eicu_sepsis3_trajectory_csv", "")),
            target_field=str(input_sources.get("eicu_target_field", "total_sofa")),
        )
        feature_schema["physiology_sequence_columns"] = list(semantics.get("sequence_feature_names", []))
        feature_schema["intervention_context_columns"] = list(semantics.get("intervention_sequence_feature_names", []))
        feature_schema["patient_feature_names"] = list(semantics.get("patient_feature_names", []))
    return build_eicu_sepsis3_inference_sample(
        payload=input_payload,
        labels_csv=str(input_sources.get("eicu_sepsis3_labels_csv", "")),
        trajectory_csv=str(input_sources.get("eicu_sepsis3_trajectory_csv", "")),
        dataset_name=str(dataset_summary.get("dataset_name", "eicu_sepsis3_inference")),
        history_length=int(dataset_summary.get("history_length", 4)),
        forecast_horizon=int(dataset_summary.get("forecast_horizon", 2)),
        target_field=str(input_sources.get("eicu_target_field", "total_sofa")),
        enable_kg=bool(input_sources.get("enable_kg", False)),
        kg_directory=str(input_sources.get("kg_directory", "")),
        append_kg_to_patient_static=bool(input_sources.get("append_kg_to_patient_static", True)),
        seasonality=int(dataset_summary.get("seasonality", 4)),
        feature_schema=feature_schema,
    )


def _guardrail_decision(
    case_detail: Dict[str, object],
    uncertainty_context: Dict[str, float],
    args,
) -> Dict[str, object]:
    selected_candidate = dict(case_detail.get("selected_candidate", {}))
    donor = dict(selected_candidate.get("donor", {}))
    reasons = []

    donor_stay_id = _safe_float(donor.get("stay_id"), -1.0)
    donor_similarity = _safe_float(donor.get("donor_similarity"))
    donor_guideline = _safe_float(donor.get("donor_guideline_compatibility"))
    donor_total_score = _safe_float(donor.get("donor_total_score"))
    missing_care_penalty = _safe_float(donor.get("donor_missing_care_penalty"))
    contraindication_penalty = _safe_float(donor.get("donor_contraindication_penalty"))
    predicted_delta = _safe_float(case_detail.get("selected_predicted_delta"))

    if donor_stay_id < 0.0:
        reasons.append("未找到可用 donor。")
    if donor_similarity < float(args.min_donor_similarity):
        reasons.append(f"donor 相似度 {donor_similarity:.3f} 低于阈值 {float(args.min_donor_similarity):.3f}。")
    if donor_guideline < float(args.min_guideline_compatibility):
        reasons.append(f"指南一致性 {donor_guideline:.3f} 低于阈值 {float(args.min_guideline_compatibility):.3f}。")
    if donor_total_score < float(args.min_donor_total_score):
        reasons.append(f"donor 总分 {donor_total_score:.3f} 低于阈值 {float(args.min_donor_total_score):.3f}。")
    if missing_care_penalty > float(args.max_missing_care_penalty):
        reasons.append(f"缺失照护处罚 {missing_care_penalty:.3f} 高于阈值 {float(args.max_missing_care_penalty):.3f}。")
    if contraindication_penalty > float(args.max_contraindication_penalty):
        reasons.append(f"禁忌处罚 {contraindication_penalty:.3f} 高于阈值 {float(args.max_contraindication_penalty):.3f}。")
    if args.require_positive_delta and predicted_delta <= 0.0:
        reasons.append(f"模型 proxy 改善值 {predicted_delta:.3f} 未超过 0。")
    if not bool(args.disable_uncertainty_guardrail):
        recommended_std = _safe_float(uncertainty_context.get("recommended_mean_std"))
        delta_lower_bound = _safe_float(uncertainty_context.get("delta_lower_bound"))
        if recommended_std > float(args.max_recommended_forecast_std):
            reasons.append(
                f"推荐方案预测标准差 {recommended_std:.3f} 高于阈值 {float(args.max_recommended_forecast_std):.3f}。"
            )
        if predicted_delta > 0.0 and delta_lower_bound < float(args.min_delta_lower_bound):
            reasons.append(
                f"改善下界 {delta_lower_bound:.3f} 低于阈值 {float(args.min_delta_lower_bound):.3f}，改善未明显超过不确定性。"
            )

    return {
        "passed": len(reasons) == 0,
        "status": "recommendation_ready" if len(reasons) == 0 else "review_only",
        "reasons": reasons,
        "thresholds": {
            "min_donor_similarity": float(args.min_donor_similarity),
            "min_guideline_compatibility": float(args.min_guideline_compatibility),
            "min_donor_total_score": float(args.min_donor_total_score),
            "max_missing_care_penalty": float(args.max_missing_care_penalty),
            "max_contraindication_penalty": float(args.max_contraindication_penalty),
            "require_positive_delta": bool(args.require_positive_delta),
            "max_recommended_forecast_std": float(args.max_recommended_forecast_std),
            "min_delta_lower_bound": float(args.min_delta_lower_bound),
            "uncertainty_guardrail_enabled": not bool(args.disable_uncertainty_guardrail),
        },
        "uncertainty": dict(uncertainty_context),
    }


def _finalize_report(case_detail: Dict[str, object], guardrail: Dict[str, object]) -> Dict[str, object]:
    report = render_counterfactual_case_report(case_detail)
    warnings = list(report.get("warnings", []))
    if not guardrail.get("passed", False):
        warnings = list(guardrail.get("reasons", [])) + warnings
        recommended_plan = dict(report.get("recommended_plan", {}))
        recommended_plan["actionability"] = "review_only"
        recommended_plan["actionability_note"] = "候选方案已展示，但仅供医生复核，不建议直接采纳。"
        report["recommended_plan"] = recommended_plan
        report["report_text"] = "\n".join(
            [
                "No high-confidence similar case met the configured safety thresholds.",
                "Candidate plan is still displayed for physician review.",
                *guardrail.get("reasons", []),
            ]
        )
    report["warnings"] = warnings
    report["recommendation_status"] = str(guardrail.get("status", "review_only"))
    return report


def _rolling_horizon_payload(
    rollout_summary: Dict[str, object],
    case_index: int = 0,
) -> Dict[str, object]:
    case_rollouts = list(rollout_summary.get("case_rollouts", []))
    if case_index >= len(case_rollouts):
        return {}
    case_rollout = dict(case_rollouts[case_index])
    steps = list(case_rollout.get("steps", []))
    summary = {
        "rollout_steps": int(rollout_summary.get("rollout_steps", len(steps))),
        "rollout_discount": float(rollout_summary.get("rollout_discount", 0.0)),
        "discounted_cumulative_delta": _safe_float(case_rollout.get("discounted_cumulative_delta")),
        "raw_cumulative_delta": _safe_float(case_rollout.get("raw_cumulative_delta")),
        "step_count": int(len(steps)),
        "stable_candidate_source": (
            len({str(dict(step.get("selected_candidate", {})).get("candidate_source", "")) for step in steps}) == 1
            if steps
            else False
        ),
    }
    step_rows: List[Dict[str, object]] = []
    for step in steps:
        factual_prediction = [float(value) for value in step.get("factual_prediction", [])]
        counterfactual_prediction = [
            float(value) for value in step.get("selected_counterfactual_prediction", [])
        ]
        selected_candidate = dict(step.get("selected_candidate", {}))
        step_rows.append(
            {
                "step_index": int(step.get("step_index", 0)),
                "candidate_source": str(selected_candidate.get("candidate_source", "")),
                "candidate_layer": str(selected_candidate.get("candidate_layer", "")),
                "predicted_delta": _safe_float(step.get("selected_predicted_delta")),
                "discounted_delta": _safe_float(step.get("discounted_delta")),
                "current_plan_mean": (
                    float(sum(factual_prediction) / max(1, len(factual_prediction)))
                    if factual_prediction
                    else 0.0
                ),
                "recommended_plan_mean": (
                    float(sum(counterfactual_prediction) / max(1, len(counterfactual_prediction)))
                    if counterfactual_prediction
                    else 0.0
                ),
                "repair_actions": list(selected_candidate.get("repair_actions", [])),
                "search_actions": list(selected_candidate.get("search_actions", [])),
            }
        )
    return {
        "summary": summary,
        "steps": step_rows,
    }


def _rolling_horizon_markdown(rolling_horizon: Dict[str, object]) -> str:
    summary = dict(rolling_horizon.get("summary", {}))
    steps = list(rolling_horizon.get("steps", []))
    if not steps:
        return ""
    lines = [
        "## 9. 短期滚动模拟",
        f"- 模拟步数: {summary.get('rollout_steps', 0)}",
        f"- 折扣系数: {summary.get('rollout_discount', 0.0):.2f}",
        f"- 折扣累计改善: {summary.get('discounted_cumulative_delta', 0.0):.3f}",
        f"- 原始累计改善: {summary.get('raw_cumulative_delta', 0.0):.3f}",
        f"- 方案来源是否稳定: {'是' if bool(summary.get('stable_candidate_source', False)) else '否'}",
        "",
        "| 步骤 | 方案来源 | 候选层级 | 当前方案均值 | 推荐方案均值 | 单步改善 | 折扣后改善 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in steps:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("step_index", "")),
                    str(row.get("candidate_source", "")),
                    str(row.get("candidate_layer", "")),
                    f"{_safe_float(row.get('current_plan_mean')):.3f}",
                    f"{_safe_float(row.get('recommended_plan_mean')):.3f}",
                    f"{_safe_float(row.get('predicted_delta')):.3f}",
                    f"{_safe_float(row.get('discounted_delta')):.3f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _selection_conflict_reason(conflict: str) -> str:
    mapping = {
        "future_sofa_worse": "多目标评估显示推荐方案的未来 SOFA 方向并未优于当前方案。",
        "future_lactate_worse": "多目标评估显示推荐方案的未来乳酸方向并未优于当前方案。",
        "vasopressor_risk_higher": "多目标评估显示推荐方案对应的后续升压药需求风险更高。",
        "resp_support_risk_higher": "多目标评估显示推荐方案对应的后续呼吸支持升级风险更高。",
    }
    normalized = str(conflict or "").strip()
    return mapping.get(normalized, f"多目标评估发现额外冲突：{normalized}。")


def _selection_guardrail_reasons(selected_candidate: Dict[str, object]) -> List[str]:
    components = dict(selected_candidate.get("selection_components", {}))
    neighborhood_summary = dict(selected_candidate.get("neighborhood_summary", {}))
    if not components:
        return []

    reasons: List[str] = []
    conflicts = [str(item) for item in components.get("multiobjective_conflicts", []) if str(item).strip()]
    for conflict in conflicts:
        reasons.append(_selection_conflict_reason(conflict))

    lower_bound = _safe_float(components.get("delta_lower_bound"))
    predicted_delta = _safe_float(selected_candidate.get("predicted_delta"))
    uncertainty_penalty = _safe_float(components.get("uncertainty_penalty"))
    positive_unstable_penalty = _safe_float(components.get("positive_unstable_penalty"))
    support = _safe_float(components.get("multiobjective_support"))

    if predicted_delta > 0.0 and lower_bound < 0.0:
        reasons.append(
            f"虽然均值层面的 proxy 改善为 {predicted_delta:.3f}，但多目标下界仅为 {lower_bound:.3f}，提示改善尚未稳定超过不确定性。"
        )
    if positive_unstable_penalty > 0.0:
        reasons.append("该候选属于“表面正收益但稳定性不足”的方案，因此在主排序中已被额外下调。")
    if uncertainty_penalty >= 0.25:
        reasons.append(f"多目标排序中的不确定性惩罚为 {uncertainty_penalty:.3f}，说明该方案的预测稳定性仍偏弱。")
    if support < 0.0 and not conflicts:
        reasons.append(f"多目标综合支持分为 {support:.3f}，说明除 SOFA 外的辅助临床方向未形成一致支持。")

    neighbor_consistency = _safe_float(neighborhood_summary.get("consistency"))
    neighbor_hard_rate = _safe_float(neighborhood_summary.get("hard_pass_rate"))
    neighbor_overlap_rate = _safe_float(neighborhood_summary.get("overlap_valid_rate"))
    if neighborhood_summary:
        if neighbor_consistency < 0.45:
            reasons.append(
                f"相似患者邻域一致性仅为 {neighbor_consistency:.3f}，提示当前高分 donor 缺少足够的邻域支持。"
            )
        if neighbor_hard_rate < 0.5:
            reasons.append(
                f"近邻 donor 的硬过滤通过率仅为 {neighbor_hard_rate:.3f}，说明 donor 可交换性基础仍偏弱。"
            )
        if neighbor_overlap_rate < 0.5:
            reasons.append(
                f"近邻 donor 的 overlap 通过率仅为 {neighbor_overlap_rate:.3f}，说明病程与干预可迁移性支持不足。"
            )

    deduped: List[str] = []
    for item in reasons:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _guardrail_decision(
    case_detail: Dict[str, object],
    uncertainty_context: Dict[str, float],
    rolling_horizon: Dict[str, object],
    args,
) -> Dict[str, object]:
    selected_candidate = dict(case_detail.get("selected_candidate", {}))
    donor = dict(selected_candidate.get("donor", {}))
    reasons: List[str] = []

    donor_stay_id = _safe_float(donor.get("stay_id"), -1.0)
    donor_similarity = _safe_float(donor.get("donor_similarity"))
    donor_guideline = _safe_float(donor.get("donor_guideline_compatibility"))
    donor_total_score = _safe_float(donor.get("donor_total_score"))
    missing_care_penalty = _safe_float(donor.get("donor_missing_care_penalty"))
    contraindication_penalty = _safe_float(donor.get("donor_contraindication_penalty"))
    predicted_delta = _safe_float(case_detail.get("selected_predicted_delta"))
    rollout_summary = dict(dict(rolling_horizon or {}).get("summary", {}))
    rollout_discounted_delta = _safe_float(rollout_summary.get("discounted_cumulative_delta"))
    rollout_enabled = bool(rollout_summary)
    stable_candidate_source = bool(rollout_summary.get("stable_candidate_source", False))
    neighborhood_summary = dict(selected_candidate.get("neighborhood_summary", {}))
    selection_reasons = _selection_guardrail_reasons(selected_candidate)

    if donor_stay_id < 0.0:
        reasons.append("未检索到可用的相似患者 donor。")
    if donor_similarity < float(args.min_donor_similarity):
        reasons.append(f"donor 相似度 {donor_similarity:.3f} 低于阈值 {float(args.min_donor_similarity):.3f}。")
    if donor_guideline < float(args.min_guideline_compatibility):
        reasons.append(f"指南一致性 {donor_guideline:.3f} 低于阈值 {float(args.min_guideline_compatibility):.3f}。")
    if donor_total_score < float(args.min_donor_total_score):
        reasons.append(f"donor 总分 {donor_total_score:.3f} 低于阈值 {float(args.min_donor_total_score):.3f}。")
    if missing_care_penalty > float(args.max_missing_care_penalty):
        reasons.append(f"缺失照护处罚 {missing_care_penalty:.3f} 高于阈值 {float(args.max_missing_care_penalty):.3f}。")
    if contraindication_penalty > float(args.max_contraindication_penalty):
        reasons.append(f"禁忌处罚 {contraindication_penalty:.3f} 高于阈值 {float(args.max_contraindication_penalty):.3f}。")
    if predicted_delta <= 0.0:
        reasons.append(f"单步模型 proxy 改善 {predicted_delta:.3f} 未大于 0。")
    elif bool(args.require_positive_delta) and predicted_delta <= 0.0:
        reasons.append(f"单步模型 proxy 改善 {predicted_delta:.3f} 未大于 0。")
    if rollout_enabled and rollout_discounted_delta <= 0.0:
        reasons.append(f"两步滚动折扣累计改善 {rollout_discounted_delta:.3f} 未大于 0。")
    if rollout_enabled and not stable_candidate_source:
        reasons.append("滚动模拟中候选方案来源发生切换，短期决策稳定性不足。")
    if not bool(args.disable_uncertainty_guardrail):
        recommended_std = _safe_float(uncertainty_context.get("recommended_mean_std"))
        delta_lower_bound = _safe_float(uncertainty_context.get("delta_lower_bound"))
        if recommended_std > float(args.max_recommended_forecast_std):
            reasons.append(
                f"推荐方案预测标准差 {recommended_std:.3f} 高于阈值 {float(args.max_recommended_forecast_std):.3f}。"
            )
        if delta_lower_bound < float(args.min_delta_lower_bound):
            reasons.append(
                f"改善下界 {delta_lower_bound:.3f} 低于阈值 {float(args.min_delta_lower_bound):.3f}，改善未明显超过不确定性。"
            )
    reasons.extend(selection_reasons)

    selection_components = dict(selected_candidate.get("selection_components", {}))

    return {
        "passed": len(reasons) == 0,
        "status": "recommendation_ready" if len(reasons) == 0 else "review_only",
        "reasons": reasons,
        "thresholds": {
            "min_donor_similarity": float(args.min_donor_similarity),
            "min_guideline_compatibility": float(args.min_guideline_compatibility),
            "min_donor_total_score": float(args.min_donor_total_score),
            "max_missing_care_penalty": float(args.max_missing_care_penalty),
            "max_contraindication_penalty": float(args.max_contraindication_penalty),
            "require_positive_delta": bool(args.require_positive_delta),
            "max_recommended_forecast_std": float(args.max_recommended_forecast_std),
            "min_delta_lower_bound": float(args.min_delta_lower_bound),
            "uncertainty_guardrail_enabled": not bool(args.disable_uncertainty_guardrail),
            "rollout_guardrail_enabled": rollout_enabled,
        },
        "uncertainty": dict(uncertainty_context),
        "rolling_horizon": dict(rolling_horizon or {}),
        "selection_summary": {
            "candidate_selection_score": _safe_float(selected_candidate.get("candidate_selection_score")),
            "base_rule_score": _safe_float(selected_candidate.get("stage2_base_selection_score")),
            "pre_neighborhood_score": _safe_float(selected_candidate.get("stage3_pre_neighborhood_selection_score")),
            "multiobjective_support": _safe_float(selection_components.get("multiobjective_support")),
            "uncertainty_penalty": _safe_float(selection_components.get("uncertainty_penalty")),
            "delta_lower_bound": _safe_float(selection_components.get("delta_lower_bound")),
            "positive_unstable_penalty": _safe_float(selection_components.get("positive_unstable_penalty")),
            "multiobjective_conflicts": list(selection_components.get("multiobjective_conflicts", [])),
            "neighbor_consistency": _safe_float(neighborhood_summary.get("consistency")),
            "neighbor_exchangeability_mean": _safe_float(neighborhood_summary.get("exchangeability_mean")),
            "neighbor_action_alignment_mean": _safe_float(neighborhood_summary.get("action_alignment_mean")),
            "neighbor_hard_pass_rate": _safe_float(neighborhood_summary.get("hard_pass_rate")),
            "neighbor_overlap_valid_rate": _safe_float(neighborhood_summary.get("overlap_valid_rate")),
            "auxiliary_reasons": selection_reasons,
        },
    }


def _finalize_report(case_detail: Dict[str, object], guardrail: Dict[str, object]) -> Dict[str, object]:
    report = render_counterfactual_case_report(case_detail)
    warnings = list(report.get("warnings", []))
    if not guardrail.get("passed", False):
        warnings = list(guardrail.get("reasons", [])) + warnings
        recommended_plan = dict(report.get("recommended_plan", {}))
        recommended_plan["actionability"] = "review_only"
        recommended_plan["actionability_note"] = "候选方案仍可展示，但仅供医生复核，不建议直接采纳。"
        report["recommended_plan"] = recommended_plan
        report["report_text"] = "\n".join(
            [
                "No high-confidence similar case met the configured safety thresholds.",
                "Candidate plan is still displayed for physician review.",
                *guardrail.get("reasons", []),
            ]
        )
    report["warnings"] = warnings
    report["recommendation_status"] = str(guardrail.get("status", "review_only"))
    return report


def main():
    args = parse_args()
    bundle_payload = _load_bundle(args.bundle_path)
    with open(_resolved_path_str(args.input_json), "r", encoding="utf-8") as file:
        input_payload = json.load(file)

    trainer = _build_trainer(
        bundle_payload=bundle_payload,
        device=_resolve_device(args.device),
        enable_persistent_semantic_store=not bool(args.disable_persistent_semantic_store),
    )
    if args.counterfactual_reranker_mode_override:
        trainer.counterfactual_reranker_mode = str(args.counterfactual_reranker_mode_override)
    trainer.counterfactual_rollout_steps = max(1, int(args.rollout_steps))
    trainer.counterfactual_rollout_discount = float(args.rollout_discount)
    sample = _build_sample(bundle_payload, input_payload)
    _validate_sample_dimensions(sample, bundle_payload)

    base_prediction = trainer.predict([sample], use_memory=False)[0]
    factual_prediction = trainer.predict([sample], use_memory=True)[0]
    counterfactual_summary = trainer.predict_counterfactual([sample], include_predictions=True)
    case_detail = counterfactual_summary.get("case_details", [{}])[0]
    counterfactual_prediction = counterfactual_summary.get("predictions", [])[0] if counterfactual_summary.get("predictions") else []
    counterfactual_sample = _selected_counterfactual_sample(sample, case_detail)
    uncertainty_sample_count = max(1, int(args.uncertainty_samples))
    base_uncertainty = trainer.predict_with_uncertainty(
        [sample],
        use_memory=False,
        num_samples=uncertainty_sample_count,
        include_auxiliary=False,
    ).get("samples", [{}])[0]
    factual_uncertainty = trainer.predict_with_uncertainty(
        [sample],
        use_memory=True,
        num_samples=uncertainty_sample_count,
        include_auxiliary=True,
    ).get("samples", [{}])[0]
    counterfactual_uncertainty = trainer.predict_with_uncertainty(
        [counterfactual_sample],
        use_memory=True,
        num_samples=uncertainty_sample_count,
        include_auxiliary=True,
    ).get("samples", [{}])[0]
    uncertainty_context = _uncertainty_guardrail_context(
        factual_uncertainty=factual_uncertainty,
        counterfactual_uncertainty=counterfactual_uncertainty,
        predicted_delta=_safe_float(case_detail.get("selected_predicted_delta")),
    )
    rollout_summary = (
        trainer.predict_counterfactual_rollout(
            [sample],
            rollout_steps=max(1, int(args.rollout_steps)),
            include_predictions=True,
        )
        if int(args.rollout_steps) > 1
        else {}
    )
    rolling_horizon = _rolling_horizon_payload(rollout_summary, case_index=0) if rollout_summary else {}
    guardrail = _guardrail_decision(case_detail, uncertainty_context, rolling_horizon, args)
    final_report = _finalize_report(case_detail, guardrail)
    clinical_risk_forecasts = {
        "current_plan": dict(factual_uncertainty.get("auxiliary_predictions", {})),
        "recommended_plan": dict(counterfactual_uncertainty.get("auxiliary_predictions", {})),
    }
    final_report["prediction_uncertainty"] = {
        "base_plan": dict(base_uncertainty),
        "current_plan": dict(factual_uncertainty),
        "recommended_plan": dict(counterfactual_uncertainty),
        "guardrail_summary": dict(uncertainty_context),
    }
    final_report["clinical_risk_forecasts"] = clinical_risk_forecasts
    doctor_view = render_physician_case_report(
        case_detail=case_detail,
        base_prediction=base_prediction,
        factual_prediction=factual_prediction,
        counterfactual_prediction=counterfactual_prediction,
        guardrail=guardrail,
        prediction_uncertainty=final_report["prediction_uncertainty"],
        clinical_risk_forecasts=clinical_risk_forecasts,
        rolling_horizon=rolling_horizon,
        step_hours=6,
    )
    if rolling_horizon:
        final_report["rolling_horizon"] = rolling_horizon

    result = {
        "bundle_summary": {
            "bundle_path": _resolved_path_str(args.bundle_path),
            "dataset_name": bundle_payload.get("dataset_summary", {}).get("dataset_name", ""),
            "forecast_horizon": bundle_payload.get("dataset_summary", {}).get("forecast_horizon", 0),
            "history_length": bundle_payload.get("dataset_summary", {}).get("history_length", 0),
            "enable_kg": bundle_payload.get("input_sources", {}).get("enable_kg", False),
        },
        "input_case": case_detail.get("query", {}),
        "base_prediction": base_prediction,
        "factual_prediction": factual_prediction,
        "counterfactual_prediction": counterfactual_prediction,
        "base_uncertainty": base_uncertainty,
        "factual_uncertainty": factual_uncertainty,
        "counterfactual_uncertainty": counterfactual_uncertainty,
        "clinical_risk_forecasts": clinical_risk_forecasts,
        "selected_candidate": case_detail.get("selected_candidate", {}),
        "candidate_options": case_detail.get("candidate_options", []),
        "top_donor_candidates": case_detail.get("top_donor_candidates", []),
        "guardrail": guardrail,
        "final_report": final_report,
        "rolling_horizon": rolling_horizon,
        "doctor_view": doctor_view,
    }

    if args.output_json:
        output_path = _resolved_path_str(args.output_json)
        _ensure_parent_dir(output_path)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)

    if args.output_md:
        output_md = _resolved_path_str(args.output_md)
        _ensure_parent_dir(output_md)
        with open(output_md, "w", encoding="utf-8") as file:
            file.write(str(doctor_view.get("markdown", "")))

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
