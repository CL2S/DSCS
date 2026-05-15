import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from infer_eicu_counterfactual_plan import (  # noqa: E402
    _build_sample,
    _build_trainer,
    _load_bundle,
    _resolve_device,
    _resolved_path_str,
    _safe_float,
    _validate_sample_dimensions,
)
from src.counterfactual_plan_renderer import render_counterfactual_case_report, render_physician_case_report  # noqa: E402


COMPONENT_TYPES = ["method", "dose", "timing", "context"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a first-pass compositional counterfactual intervention from "
            "the trained intervention component store."
        )
    )
    parser.add_argument("--bundle-path", required=True)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-donors", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--min-predicted-delta", type=float, default=0.0)
    parser.add_argument("--disable-persistent-semantic-store", action="store_true")
    return parser.parse_args()


def _ensure_parent_dir(path_str: str) -> None:
    if not path_str:
        return
    Path(path_str).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _read_json(path_str: str) -> Dict[str, object]:
    with open(_resolved_path_str(path_str), "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must contain an object payload.")
    return payload


def _augment_raw_patient_payload(bundle_payload: Dict[str, object], input_payload: Dict[str, object]) -> Dict[str, object]:
    """Fill missing schema columns with zeros so the hand-written template is usable."""
    if str(input_payload.get("sample_type", "")).strip().lower() == "forecast_sample" or "sequence" in input_payload:
        return input_payload

    payload = json.loads(json.dumps(input_payload, ensure_ascii=False))
    semantics = dict(bundle_payload.get("dataset_semantics", {}))
    input_sources = dict(bundle_payload.get("input_sources", {}))
    dataset_summary = dict(bundle_payload.get("dataset_summary", {}))
    target_field = str(input_sources.get("eicu_target_field", "total_sofa"))
    history_length = max(1, int(dataset_summary.get("history_length", 4)))

    context_rows = list(payload.get("context_rows", []))
    if not context_rows:
        last_sofa = _safe_float(dict(payload.get("label_row", {})).get("pre_baseline_total_sofa"), 8.0)
        context_rows = [
            {
                "bin_index": index,
                "rel_end_hours": float((index + 1) * 6),
                target_field: float(last_sofa),
            }
            for index in range(history_length)
        ]

    sequence_names = [str(name) for name in semantics.get("sequence_feature_names", [])]
    intervention_sequence_names = [str(name) for name in semantics.get("intervention_sequence_feature_names", [])]
    required_context_names = list(dict.fromkeys(sequence_names + intervention_sequence_names + [target_field]))
    for index, row in enumerate(context_rows):
        row = dict(row)
        row.setdefault("bin_index", index)
        row.setdefault("rel_end_hours", float((index + 1) * 6))
        row.setdefault(target_field, _safe_float(row.get(target_field), _safe_float(row.get("total_sofa"), 8.0)))
        for name in required_context_names:
            row.setdefault(name, 0.0)
        context_rows[index] = row

    while len(context_rows) < history_length:
        next_row = dict(context_rows[-1])
        next_row["bin_index"] = _safe_float(next_row.get("bin_index")) + 1.0
        next_row["rel_end_hours"] = _safe_float(next_row.get("rel_end_hours")) + 6.0
        context_rows.append(next_row)
    payload["context_rows"] = context_rows[-history_length:]

    label_row = dict(payload.get("label_row", {}))
    stay_id = payload.get("stay_id", label_row.get("patientunitstayid", 9000001))
    label_row.setdefault("patientunitstayid", stay_id)
    label_row.setdefault("pre_baseline_total_sofa", _safe_float(context_rows[-1].get(target_field), 8.0))
    payload["label_row"] = label_row
    payload.setdefault("future_target", [_safe_float(context_rows[-1].get(target_field), 8.0)] * int(dataset_summary.get("forecast_horizon", 2)))
    return payload


def _mean(values: Sequence[float]) -> float:
    return float(sum(float(value) for value in values) / max(1, len(values)))


def _restore_prediction(sample, normalized_prediction: Sequence[float]) -> List[float]:
    return [float(value) * sample.scale_value + sample.scale_center for value in normalized_prediction]


def _copy_sequence(sequence: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[float(value) for value in step] for step in sequence]


def _component_record(trainer, entry, component_type: str):
    component_code = str(dict(entry.intervention_plan_code or {}).get(component_type, ""))
    if not component_code:
        return None
    return trainer._intervention_component_store(component_type).get(component_code)


def _apply_component_record(
    intervention_static: Sequence[float],
    intervention_sequence: Sequence[Sequence[float]],
    record,
    sequence_width: int,
    sequence_length: int,
) -> Tuple[List[float], List[List[float]]]:
    static_values = [float(value) for value in intervention_static]
    sequence_values = _copy_sequence(intervention_sequence)
    if not sequence_values and sequence_width > 0:
        sequence_values = [[0.0] * sequence_width for _ in range(max(1, sequence_length))]
    for index_text, value in dict(record.static_values_by_index).items():
        index = int(index_text)
        if 0 <= index < len(static_values):
            static_values[index] = float(value)
    for index_text, values in dict(record.sequence_values_by_index).items():
        index = int(index_text)
        if index < 0:
            continue
        for step_index, value in enumerate(values):
            while step_index >= len(sequence_values):
                sequence_values.append([0.0] * max(1, sequence_width))
            if index >= len(sequence_values[step_index]):
                sequence_values[step_index].extend([0.0] * (index + 1 - len(sequence_values[step_index])))
            sequence_values[step_index][index] = float(value)
    return static_values, sequence_values


def _split_actions(text_value: object) -> List[str]:
    return [part for part in str(text_value or "").split("|") if part]


def _signature(static_values: Sequence[float], sequence_values: Sequence[Sequence[float]]) -> Tuple[Tuple[float, ...], Tuple[Tuple[float, ...], ...]]:
    return (
        tuple(round(float(value), 4) for value in static_values),
        tuple(tuple(round(float(value), 4) for value in row) for row in sequence_values),
    )


def _candidate_metadata(
    trainer,
    sample,
    donor_metadata: Dict[str, object],
    static_values: Sequence[float],
    sequence_values: Sequence[Sequence[float]],
    source: str,
    component_sources: Dict[str, str],
    repair_actions: Optional[Sequence[str]] = None,
):
    metadata = trainer._build_generated_candidate_metadata(
        sample=sample,
        donor_metadata=donor_metadata,
        intervention_static=static_values,
        intervention_sequence=sequence_values,
        candidate_source=source,
        repair_actions=list(repair_actions or []),
        search_actions=["component_composition"],
        candidate_layer="composition",
        strategy_family="component_composition",
        parameter_profile={"component_sources": dict(component_sources)},
        anchor_relation="multi_donor_component_mix",
        safety_rationale=[
            "从历史干预库中抽取组件后重组",
            "候选仍需通过知识约束、邻域一致性和模型预测筛选",
        ],
    )
    metadata["component_sources"] = dict(component_sources)
    return metadata


def _register_candidate(
    candidates: List[Tuple[List[float], List[List[float]], Dict[str, object]]],
    seen: set,
    static_values: Sequence[float],
    sequence_values: Sequence[Sequence[float]],
    metadata: Dict[str, object],
) -> None:
    static_list = [float(value) for value in static_values]
    sequence_list = _copy_sequence(sequence_values)
    sig = _signature(static_list, sequence_list)
    if sig in seen:
        return
    seen.add(sig)
    candidates.append((static_list, sequence_list, metadata))


def _generate_component_candidates(
    trainer,
    sample,
    ranked_donors: Sequence[Tuple[object, Dict[str, object]]],
    max_donors: int,
    max_candidates: int,
) -> List[Tuple[List[float], List[List[float]], Dict[str, object]]]:
    base_static = [float(value) for value in (sample.intervention_static or [])]
    base_sequence = _copy_sequence(sample.intervention_sequence or [])
    sequence_width = len(base_sequence[0]) if base_sequence else len(trainer.intervention_sequence_feature_names)
    sequence_length = len(base_sequence) if base_sequence else int(trainer.trainer_config.history_length)
    candidates: List[Tuple[List[float], List[List[float]], Dict[str, object]]] = []
    seen = {_signature(base_static, base_sequence)}
    top_donors = list(ranked_donors[: max(1, int(max_donors))])
    fallback_metadata = dict(top_donors[0][1]) if top_donors else {}

    repaired_static, repaired_sequence, repair_actions = trainer._repair_intervention_plan(
        sample_kg_flags=dict(sample.metadata.get("kg_flags", {})),
        intervention_static=base_static,
        intervention_sequence=base_sequence,
    )
    if repair_actions:
        _register_candidate(
            candidates,
            seen,
            repaired_static,
            repaired_sequence,
            _candidate_metadata(
                trainer,
                sample,
                fallback_metadata,
                repaired_static,
                repaired_sequence,
                source="composite_current_kg_repair",
                component_sources={"base": "current_plan"},
                repair_actions=repair_actions,
            ),
        )

    for donor_rank, (entry, donor_metadata) in enumerate(top_donors, start=1):
        donor_static = [float(value) for value in entry.intervention_static]
        donor_sequence = _copy_sequence(entry.intervention_sequence)
        _register_candidate(
            candidates,
            seen,
            donor_static,
            donor_sequence,
            _candidate_metadata(
                trainer,
                sample,
                donor_metadata,
                donor_static,
                donor_sequence,
                source="composite_full_donor_plan",
                component_sources={"full_plan": f"donor_rank_{donor_rank}"},
            ),
        )

        component_records = {
            component_type: _component_record(trainer, entry, component_type)
            for component_type in COMPONENT_TYPES
        }
        for component_type in ["method", "dose", "timing"]:
            record = component_records.get(component_type)
            if record is None:
                continue
            static_values, sequence_values = _apply_component_record(
                base_static,
                base_sequence,
                record,
                sequence_width=sequence_width,
                sequence_length=sequence_length,
            )
            _register_candidate(
                candidates,
                seen,
                static_values,
                sequence_values,
                _candidate_metadata(
                    trainer,
                    sample,
                    donor_metadata,
                    static_values,
                    sequence_values,
                    source=f"composite_current_plus_{component_type}",
                    component_sources={component_type: f"donor_rank_{donor_rank}"},
                ),
            )

        for component_group in [["method", "dose"], ["method", "timing"], ["method", "dose", "timing"]]:
            static_values = list(base_static)
            sequence_values = _copy_sequence(base_sequence)
            sources: Dict[str, str] = {}
            for component_type in component_group:
                record = component_records.get(component_type)
                if record is None:
                    continue
                static_values, sequence_values = _apply_component_record(
                    static_values,
                    sequence_values,
                    record,
                    sequence_width=sequence_width,
                    sequence_length=sequence_length,
                )
                sources[component_type] = f"donor_rank_{donor_rank}"
            if sources:
                _register_candidate(
                    candidates,
                    seen,
                    static_values,
                    sequence_values,
                    _candidate_metadata(
                        trainer,
                        sample,
                        donor_metadata,
                        static_values,
                        sequence_values,
                        source="composite_current_plus_" + "_".join(sources.keys()),
                        component_sources=sources,
                    ),
                )

    for left_index, (left_entry, left_metadata) in enumerate(top_donors[:3]):
        for right_index, (right_entry, _right_metadata) in enumerate(top_donors[:3]):
            if left_index == right_index:
                continue
            static_values = list(base_static)
            sequence_values = _copy_sequence(base_sequence)
            sources: Dict[str, str] = {}
            for component_type, entry, donor_rank in [
                ("method", left_entry, left_index + 1),
                ("dose", right_entry, right_index + 1),
                ("timing", left_entry, left_index + 1),
            ]:
                record = _component_record(trainer, entry, component_type)
                if record is None:
                    continue
                static_values, sequence_values = _apply_component_record(
                    static_values,
                    sequence_values,
                    record,
                    sequence_width=sequence_width,
                    sequence_length=sequence_length,
                )
                sources[component_type] = f"donor_rank_{donor_rank}"
            if sources:
                _register_candidate(
                    candidates,
                    seen,
                    static_values,
                    sequence_values,
                    _candidate_metadata(
                        trainer,
                        sample,
                        left_metadata,
                        static_values,
                        sequence_values,
                        source="composite_cross_donor_method_dose_timing",
                        component_sources=sources,
                    ),
                )

    return candidates[: max(1, int(max_candidates))]


def _donor_payload(metadata: Dict[str, object]) -> Dict[str, object]:
    return {
        "stay_id": _safe_float(metadata.get("stay_id"), -1.0),
        "donor_experience_id": str(metadata.get("donor_experience_id", "")),
        "donor_intervention_plan_code": dict(metadata.get("donor_intervention_plan_code", {})),
        "donor_experience_label": int(_safe_float(metadata.get("donor_experience_label"), -1.0)),
        "donor_pattern_label": int(_safe_float(metadata.get("donor_pattern_label"), -1.0)),
        "donor_trajectory_label": int(_safe_float(metadata.get("donor_trajectory_label"), -1.0)),
        "donor_similarity": _safe_float(metadata.get("donor_similarity")),
        "donor_kg_similarity": _safe_float(metadata.get("donor_kg_similarity")),
        "donor_guideline_compatibility": _safe_float(metadata.get("donor_guideline_compatibility")),
        "donor_state_match": _safe_float(metadata.get("donor_state_match")),
        "donor_missing_care_penalty": _safe_float(metadata.get("donor_missing_care_penalty")),
        "donor_contraindication_penalty": _safe_float(metadata.get("donor_contraindication_penalty")),
        "donor_total_score": _safe_float(metadata.get("donor_total_score")),
        "donor_hard_filter_valid": _safe_float(metadata.get("donor_hard_filter_valid")),
        "donor_hard_filter_reason": str(metadata.get("donor_hard_filter_reason", "")),
        "donor_overlap_score": _safe_float(metadata.get("donor_overlap_score")),
        "donor_overlap_valid": _safe_float(metadata.get("donor_overlap_valid")),
        "donor_overlap_reason": str(metadata.get("donor_overlap_reason", "")),
        "donor_neighbor_consistency": _safe_float(metadata.get("donor_neighbor_consistency")),
        "donor_neighbor_exchangeability_mean": _safe_float(metadata.get("donor_neighbor_exchangeability_mean")),
        "donor_neighbor_hard_pass_rate": _safe_float(metadata.get("donor_neighbor_hard_pass_rate")),
        "donor_neighbor_overlap_valid_rate": _safe_float(metadata.get("donor_neighbor_overlap_valid_rate")),
        "donor_learned_reranker_score": _safe_float(metadata.get("donor_learned_reranker_score")),
        "donor_reranker_adjustment": _safe_float(metadata.get("donor_reranker_adjustment")),
        "donor_reranker_mode": str(metadata.get("donor_reranker_mode", "rule_only")),
        "donor_pool_match_score": _safe_float(metadata.get("donor_pool_match_score")),
        "donor_pool_match_reward": _safe_float(metadata.get("donor_pool_match_reward")),
        "donor_pool_tags": [str(value) for value in metadata.get("donor_pool_tags", [])],
    }


def _candidate_option_payload(
    trainer,
    sample,
    static_values,
    sequence_values,
    metadata,
    predicted_delta,
    selection_score,
    selection_components,
    prediction,
    aux_predictions,
    neighborhood,
):
    return {
        "candidate_source": str(metadata.get("generated_candidate_source", "")),
        "candidate_layer": str(metadata.get("generated_candidate_layer", "composition")),
        "candidate_selection_score": float(selection_score),
        "stage2_base_selection_score": float(selection_components.get("base_rule_score", selection_score)),
        "stage3_pre_neighborhood_selection_score": float(selection_components.get("pre_neighborhood_score", selection_score)),
        "predicted_delta": float(predicted_delta),
        "predicted_counterfactual": [float(value) for value in prediction],
        "selection_components": dict(selection_components),
        "candidate_aux_predictions": dict(aux_predictions),
        "neighborhood_summary": dict(neighborhood),
        "candidate_intervention_static": [float(value) for value in static_values],
        "candidate_intervention_sequence": _copy_sequence(sequence_values),
        "repair_actions": _split_actions(metadata.get("generated_candidate_repair_actions")),
        "search_actions": _split_actions(metadata.get("generated_candidate_search_actions")),
        "candidate_strategy_family": str(metadata.get("generated_candidate_strategy_family", "component_composition")),
        "candidate_parameter_profile": dict(metadata.get("generated_candidate_parameter_profile", {})),
        "candidate_anchor_relation": str(metadata.get("generated_candidate_anchor_relation", "multi_donor_component_mix")),
        "candidate_safety_rationale": [str(item) for item in metadata.get("generated_candidate_safety_rationale", [])],
        "component_sources": dict(metadata.get("component_sources", {})),
        "donor": _donor_payload(metadata),
        "plan_summary": trainer._serialize_intervention_plan(
            intervention_static=static_values,
            intervention_sequence=sequence_values,
            base_flags=dict(metadata.get("kg_flags", sample.metadata.get("kg_flags", {}))),
        ),
    }


def _evaluate_component_candidates(trainer, sample, candidates, ranked_donors):
    trainer.eval()
    with torch.no_grad():
        encodings, manager_results, base_preds, factual_preds, factual_gates, _ = trainer._forward_batch([sample])
        factual_prediction = _restore_prediction(sample, factual_preds[0].detach().cpu().tolist())
        base_prediction = _restore_prediction(sample, base_preds[0].detach().cpu().tolist())
        factual_aux = trainer._decode_multitask_predictions(factual_gates)[0] if factual_gates else {}
        candidate_samples = [
            replace(
                sample,
                intervention_static=static_values,
                intervention_sequence=sequence_values,
                metadata={
                    **sample.metadata,
                    "counterfactual_candidate_source": metadata.get("generated_candidate_source", ""),
                },
            )
            for static_values, sequence_values, metadata in candidates
        ]
        _, _, _, candidate_preds, candidate_gates, _ = trainer._forward_batch(candidate_samples)
        candidate_aux_rows = trainer._decode_multitask_predictions(candidate_gates)

    option_rows: List[Dict[str, object]] = []
    for index, (static_values, sequence_values, metadata) in enumerate(candidates):
        prediction = _restore_prediction(sample, candidate_preds[index].detach().cpu().tolist())
        predicted_delta = _mean(factual_prediction) - _mean(prediction)
        neighborhood = trainer._counterfactual_neighbor_summary(
            candidate_metadata=metadata,
            ranked_donors=ranked_donors,
        )
        selection_score, selection_components = trainer._candidate_selection_score(
            predicted_delta=predicted_delta,
            candidate_metadata=metadata,
            current_aux_predictions=factual_aux,
            candidate_aux_predictions=candidate_aux_rows[index],
            current_gate_summary=factual_gates[0] if factual_gates else {},
            candidate_gate_summary=candidate_gates[index],
            neighborhood_summary=neighborhood,
        )
        option_rows.append(
            _candidate_option_payload(
                trainer=trainer,
                sample=sample,
                static_values=static_values,
                sequence_values=sequence_values,
                metadata=metadata,
                predicted_delta=predicted_delta,
                selection_score=selection_score,
                selection_components=selection_components,
                prediction=prediction,
                aux_predictions=candidate_aux_rows[index],
                neighborhood=neighborhood,
            )
        )
    option_rows.sort(key=lambda row: float(row.get("candidate_selection_score", 0.0)), reverse=True)
    return base_prediction, factual_prediction, factual_aux, option_rows


def _risk_forecast_payload(aux_predictions: Dict[str, object]) -> Dict[str, Dict[str, float | str]]:
    risk_payload: Dict[str, Dict[str, float | str]] = {}
    for task_name, raw_value in dict(aux_predictions or {}).items():
        value = _safe_float(raw_value, 0.0)
        risk_payload[str(task_name)] = {
            "kind": "point_estimate",
            "mean": float(value),
            "lower": float(value),
            "upper": float(value),
        }
    return risk_payload


def _guardrail(selected: Dict[str, object], min_predicted_delta: float) -> Dict[str, object]:
    donor = dict(selected.get("donor", {}))
    reasons = []
    predicted_delta = _safe_float(selected.get("predicted_delta"))
    if predicted_delta <= float(min_predicted_delta):
        reasons.append(f"组合候选的预测改善 {predicted_delta:.3f} 未超过阈值 {float(min_predicted_delta):.3f}。")
    if _safe_float(donor.get("donor_hard_filter_valid")) <= 0.0:
        reasons.append("组合候选未通过当前硬过滤约束。")
    if _safe_float(donor.get("donor_contraindication_penalty")) > 0.25:
        reasons.append("组合候选的禁忌惩罚偏高，需要医生复核。")
    if _safe_float(donor.get("donor_overlap_valid")) <= 0.0:
        reasons.append("组合候选与当前患者的状态/行动 overlap 不足。")
    return {
        "passed": len(reasons) == 0,
        "status": "composition_candidate_ready" if len(reasons) == 0 else "review_only",
        "reasons": reasons,
        "thresholds": {"min_predicted_delta": float(min_predicted_delta)},
        "selection_summary": {
            "candidate_selection_score": _safe_float(selected.get("candidate_selection_score")),
            "multiobjective_support": _safe_float(dict(selected.get("selection_components", {})).get("multiobjective_support")),
            "uncertainty_penalty": _safe_float(dict(selected.get("selection_components", {})).get("uncertainty_penalty")),
            "delta_lower_bound": _safe_float(dict(selected.get("selection_components", {})).get("delta_lower_bound")),
            "neighbor_consistency": _safe_float(dict(selected.get("neighborhood_summary", {})).get("consistency")),
        },
    }


def _case_detail(trainer, sample, selected, options, ranked_donors, factual_prediction, counterfactual_prediction):
    return {
        "query": trainer._serialize_query_snapshot(sample),
        "factual_prediction": [float(value) for value in factual_prediction],
        "selected_counterfactual_prediction": [float(value) for value in counterfactual_prediction],
        "selected_predicted_delta": _safe_float(selected.get("predicted_delta")),
        "selected_candidate_source": str(selected.get("candidate_source", "")),
        "selected_candidate": dict(selected),
        "candidate_options": list(options),
        "top_donor_candidates": [
            trainer._serialize_ranked_donor_candidate(entry, metadata)
            for entry, metadata in list(ranked_donors[:3])
        ],
    }


def main():
    args = parse_args()
    bundle_payload = _load_bundle(args.bundle_path)
    input_payload = _augment_raw_patient_payload(bundle_payload, _read_json(args.input_json))
    trainer = _build_trainer(
        bundle_payload=bundle_payload,
        device=_resolve_device(args.device),
        enable_persistent_semantic_store=not bool(args.disable_persistent_semantic_store),
    )
    sample = _build_sample(bundle_payload, input_payload)
    _validate_sample_dimensions(sample, bundle_payload)

    trainer.eval()
    with torch.no_grad():
        encodings, manager_results, _, _, _, _ = trainer._forward_batch([sample])
        ranked_donors = trainer._counterfactual_ranked_donors_from_manager_result(
            sample,
            encodings[0],
            manager_results[0],
        )

    candidates = _generate_component_candidates(
        trainer=trainer,
        sample=sample,
        ranked_donors=ranked_donors,
        max_donors=args.max_donors,
        max_candidates=args.max_candidates,
    )
    if not candidates:
        raise RuntimeError("No compositional intervention candidates could be generated from the intervention store.")

    base_prediction, factual_prediction, factual_aux, option_rows = _evaluate_component_candidates(
        trainer=trainer,
        sample=sample,
        candidates=candidates,
        ranked_donors=ranked_donors,
    )
    positive_rows = [
        row for row in option_rows if _safe_float(row.get("predicted_delta")) > float(args.min_predicted_delta)
    ]
    selected = positive_rows[0] if positive_rows else option_rows[0]
    counterfactual_prediction = [float(value) for value in selected.get("predicted_counterfactual", [])]
    guardrail = _guardrail(selected, min_predicted_delta=args.min_predicted_delta)
    case_detail = _case_detail(
        trainer=trainer,
        sample=sample,
        selected=selected,
        options=option_rows,
        ranked_donors=ranked_donors,
        factual_prediction=factual_prediction,
        counterfactual_prediction=counterfactual_prediction,
    )
    final_report = render_counterfactual_case_report(case_detail)
    doctor_view = render_physician_case_report(
        case_detail=case_detail,
        base_prediction=base_prediction,
        factual_prediction=factual_prediction,
        counterfactual_prediction=counterfactual_prediction,
        guardrail=guardrail,
        prediction_uncertainty={},
        clinical_risk_forecasts={
            "current_plan": _risk_forecast_payload(factual_aux),
            "recommended_plan": _risk_forecast_payload(selected.get("candidate_aux_predictions", {})),
        },
        rolling_horizon={},
        step_hours=6,
    )

    result = {
        "framework": {
            "name": "component_compositional_counterfactual_intervention",
            "status": "initial_feasibility_framework",
            "candidate_count": len(option_rows),
            "positive_candidate_count": len(positive_rows),
            "uses_intervention_component_store": True,
            "optimization_scope": "small_candidate_set_first_pass_not_global_optimum",
        },
        "bundle_summary": {
            "bundle_path": _resolved_path_str(args.bundle_path),
            "dataset_name": bundle_payload.get("dataset_summary", {}).get("dataset_name", ""),
            "forecast_horizon": bundle_payload.get("dataset_summary", {}).get("forecast_horizon", 0),
            "history_length": bundle_payload.get("dataset_summary", {}).get("history_length", 0),
        },
        "input_case": case_detail.get("query", {}),
        "base_prediction": base_prediction,
        "factual_prediction": factual_prediction,
        "compositional_counterfactual_prediction": counterfactual_prediction,
        "selected_compositional_candidate": selected,
        "candidate_options": option_rows,
        "top_donor_candidates": case_detail.get("top_donor_candidates", []),
        "guardrail": guardrail,
        "medical_explanation": {
            "brief": [
                "该结果来自历史干预组件重组后的模型反事实预测，不等同于已证实治疗方案。",
                "若预测改善为正，表示模型估计该组合方案下未来目标均值低于当前方案。",
                "需要结合禁忌惩罚、指南一致性、邻域一致性和医生判断后再解释。",
            ],
            "physician_view": doctor_view,
            "counterfactual_report": final_report,
        },
    }

    if args.output_json:
        output_path = _resolved_path_str(args.output_json)
        _ensure_parent_dir(output_path)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    if args.output_md:
        output_md = _resolved_path_str(args.output_md)
        _ensure_parent_dir(output_md)
        with open(output_md, "w", encoding="utf-8") as handle:
            handle.write(str(doctor_view.get("markdown", "")))

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
