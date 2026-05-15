import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infer_eicu_counterfactual_plan import _build_trainer, _load_bundle
from src.tsf_data import (
    build_eicu_sepsis3_forecasting_dataset,
    build_eicu_sepsis3_inference_sample,
    derive_eicu_sepsis3_feature_schema,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate phase-2 multitask reranking stability for eICU counterfactual candidates.")
    parser.add_argument("--bundle-path", default="memory_mvp_project/output/analysis/phase8_rollout_bundle.pt")
    parser.add_argument("--max-series-count", type=int, default=64)
    parser.add_argument("--output-path", default="memory_mvp_project/output/analysis/phase2_multitask_reranking_evaluation.json")
    return parser.parse_args()


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _option_identity(option: Dict[str, object]) -> str:
    return "|".join(
        [
            str(option.get("candidate_source", "")),
            str(option.get("candidate_layer", "")),
            str(option.get("donor_stay_id", "")),
            f"{_safe_float(option.get('predicted_delta')):.6f}",
        ]
    )


def _selection_snapshot(option: Dict[str, object]) -> Dict[str, object]:
    components = dict(option.get("selection_components", {}))
    conflicts = list(components.get("multiobjective_conflicts", []))
    return {
        "candidate_source": str(option.get("candidate_source", "")),
        "candidate_layer": str(option.get("candidate_layer", "")),
        "donor_stay_id": _safe_float(option.get("donor_stay_id"), -1.0),
        "predicted_delta": _safe_float(option.get("predicted_delta")),
        "selection_score": _safe_float(option.get("candidate_selection_score")),
        "base_rule_score": _safe_float(option.get("stage2_base_selection_score")),
        "delta_lower_bound": _safe_float(components.get("delta_lower_bound")),
        "uncertainty_penalty": _safe_float(components.get("uncertainty_penalty")),
        "multiobjective_support": _safe_float(components.get("multiobjective_support")),
        "conflict_count": float(len(conflicts)),
        "conflicts": conflicts,
    }


def main() -> None:
    args = parse_args()
    bundle_path = str(args.bundle_path)
    bundle_payload = _load_bundle(bundle_path)
    trainer = _build_trainer(bundle_payload, device="cpu", enable_persistent_semantic_store=False)

    dataset_summary = dict(bundle_payload.get("dataset_summary", {}))
    input_sources = dict(bundle_payload.get("input_sources", {}))
    dataset = build_eicu_sepsis3_forecasting_dataset(
        labels_csv=str(input_sources.get("eicu_sepsis3_labels_csv", "")),
        trajectory_csv=str(input_sources.get("eicu_sepsis3_trajectory_csv", "")),
        dataset_name="phase2_multitask_eval",
        history_length=int(dataset_summary.get("history_length", 4)),
        forecast_horizon=int(dataset_summary.get("forecast_horizon", 2)),
        max_series_count=max(8, int(args.max_series_count)),
        enable_kg=bool(input_sources.get("enable_kg", False)),
        kg_directory=str(input_sources.get("kg_directory", "")),
        append_kg_to_patient_static=bool(input_sources.get("append_kg_to_patient_static", True)),
    )
    feature_schema = derive_eicu_sepsis3_feature_schema(
        labels_csv=str(input_sources.get("eicu_sepsis3_labels_csv", "")),
        trajectory_csv=str(input_sources.get("eicu_sepsis3_trajectory_csv", "")),
        target_field=str(input_sources.get("eicu_target_field", "total_sofa")),
    )
    semantics = dict(bundle_payload.get("dataset_semantics", {}))
    feature_schema["physiology_sequence_columns"] = list(semantics.get("sequence_feature_names", []))
    feature_schema["intervention_context_columns"] = list(semantics.get("intervention_sequence_feature_names", []))
    feature_schema["patient_feature_names"] = list(semantics.get("patient_feature_names", []))

    labels_df = pd.read_csv(str(input_sources.get("eicu_sepsis3_labels_csv", "")))
    trajectory_df = pd.read_csv(str(input_sources.get("eicu_sepsis3_trajectory_csv", "")))
    labels_map = {
        int(row["patientunitstayid"]): row.to_dict()
        for _, row in labels_df.iterrows()
        if pd.notna(row.get("patientunitstayid"))
    }

    rows: List[Dict[str, object]] = []
    changed_cases: List[Dict[str, object]] = []
    history_length = int(dataset_summary.get("history_length", 4))
    target_field = str(input_sources.get("eicu_target_field", "total_sofa"))
    for sample in dataset.val_samples:
        stay_id = int(_safe_float(sample.metadata.get("stay_id"), -1.0))
        label_row = labels_map.get(stay_id)
        if label_row is None:
            continue
        stay_frame = trajectory_df.loc[trajectory_df["patientunitstayid"] == stay_id].copy()
        if stay_frame.empty:
            continue
        if "bin_index" in stay_frame.columns:
            stay_frame["bin_index"] = pd.to_numeric(stay_frame["bin_index"], errors="coerce")
            stay_frame = stay_frame.sort_values("bin_index").reset_index(drop=True)
        window_end_index = int(_safe_float(sample.metadata.get("window_end_index"), -1.0))
        context_rows = (
            stay_frame.loc[pd.to_numeric(stay_frame["bin_index"], errors="coerce") <= float(window_end_index)]
            .tail(history_length)
            .to_dict(orient="records")
        )
        inference_payload = {
            "stay_id": stay_id,
            "series_name": str(sample.metadata.get("series_name", f"stay_{stay_id}")),
            "label_row": dict(label_row),
            "context_rows": list(context_rows),
            "future_target": [float(value) for value in (sample.raw_target or [])],
        }
        inference_sample = build_eicu_sepsis3_inference_sample(
            payload=inference_payload,
            labels_csv=str(input_sources.get("eicu_sepsis3_labels_csv", "")),
            trajectory_csv=str(input_sources.get("eicu_sepsis3_trajectory_csv", "")),
            dataset_name="phase2_multitask_eval_inference",
            history_length=history_length,
            forecast_horizon=int(dataset_summary.get("forecast_horizon", 2)),
            target_field=target_field,
            enable_kg=bool(input_sources.get("enable_kg", False)),
            kg_directory=str(input_sources.get("kg_directory", "")),
            append_kg_to_patient_static=bool(input_sources.get("append_kg_to_patient_static", True)),
            seasonality=int(dataset_summary.get("seasonality", 4)),
            feature_schema=feature_schema,
        )
        result = trainer.predict_counterfactual([inference_sample], include_predictions=True)
        case = dict(result.get("case_details", [{}])[0])
        options = [dict(option) for option in case.get("candidate_options", [])]
        if not options:
            continue
        old_best = max(
            options,
            key=lambda item: (
                _safe_float(item.get("stage2_base_selection_score")),
                _safe_float(item.get("predicted_delta")),
                _safe_float(item.get("donor_similarity")),
            ),
        )
        new_best = max(
            options,
            key=lambda item: (
                _safe_float(item.get("candidate_selection_score")),
                _safe_float(item.get("predicted_delta")),
                _safe_float(item.get("donor_similarity")),
            ),
        )
        changed = _option_identity(old_best) != _option_identity(new_best)
        row = {
            "stay_id": _safe_float(dict(case.get("query", {})).get("stay_id"), -1.0),
            "series_name": str(dict(case.get("query", {})).get("series_name", "")),
            "changed": changed,
            "old": _selection_snapshot(old_best),
            "new": _selection_snapshot(new_best),
        }
        rows.append(row)
        if changed:
            changed_cases.append(row)

    old_rows = [row["old"] for row in rows]
    new_rows = [row["new"] for row in rows]

    def _mean_metric(items: List[Dict[str, object]], key: str) -> float:
        return float(mean(_safe_float(item.get(key)) for item in items)) if items else 0.0

    summary = {
        "sample_count": float(len(rows)),
        "changed_case_count": float(sum(1.0 for row in rows if bool(row.get("changed")))),
        "changed_case_rate": float(sum(1.0 for row in rows if bool(row.get("changed"))) / max(1, len(rows))),
        "old_mean_delta_lower_bound": _mean_metric(old_rows, "delta_lower_bound"),
        "new_mean_delta_lower_bound": _mean_metric(new_rows, "delta_lower_bound"),
        "old_mean_uncertainty_penalty": _mean_metric(old_rows, "uncertainty_penalty"),
        "new_mean_uncertainty_penalty": _mean_metric(new_rows, "uncertainty_penalty"),
        "old_mean_conflict_count": _mean_metric(old_rows, "conflict_count"),
        "new_mean_conflict_count": _mean_metric(new_rows, "conflict_count"),
        "old_positive_unstable_rate": float(
            sum(1.0 for row in old_rows if _safe_float(row.get("predicted_delta")) > 0.0 and _safe_float(row.get("delta_lower_bound")) < 0.0)
            / max(1, len(old_rows))
        ),
        "new_positive_unstable_rate": float(
            sum(1.0 for row in new_rows if _safe_float(row.get("predicted_delta")) > 0.0 and _safe_float(row.get("delta_lower_bound")) < 0.0)
            / max(1, len(new_rows))
        ),
    }

    output = {
        "bundle_path": str(Path(bundle_path).resolve()),
        "summary": summary,
        "changed_cases": changed_cases[:8],
        "all_cases": rows,
    }
    output_path = Path(str(args.output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
