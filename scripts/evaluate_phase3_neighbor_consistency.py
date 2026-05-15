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
    derive_eicu_sepsis3_feature_schema,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate phase-3 donor neighborhood consistency reranking.")
    parser.add_argument("--bundle-path", default="memory_mvp_project/output/analysis/phase8_rollout_bundle.pt")
    parser.add_argument("--max-series-count", type=int, default=64)
    parser.add_argument("--output-path", default="memory_mvp_project/output/analysis/phase3_neighbor_consistency_evaluation.json")
    return parser.parse_args()


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _option_identity(option: Dict[str, object]) -> str:
    return "|".join(
        [
            str(option.get("stay_id", "")),
            f"{_safe_float(option.get('pre_neighbor_total_score')):.6f}",
            f"{_safe_float(option.get('total_score')):.6f}",
        ]
    )


def _selection_snapshot(option: Dict[str, object]) -> Dict[str, object]:
    metadata = dict(option)
    return {
        "stay_id": _safe_float(metadata.get("stay_id"), -1.0),
        "similarity": _safe_float(metadata.get("donor_similarity")),
        "guideline_compatibility": _safe_float(metadata.get("donor_guideline_compatibility")),
        "state_match": _safe_float(metadata.get("donor_state_match")),
        "pre_neighbor_total_score": _safe_float(metadata.get("donor_pre_neighbor_total_score")),
        "total_score": _safe_float(metadata.get("donor_total_score")),
        "neighbor_consistency": _safe_float(metadata.get("donor_neighbor_consistency")),
        "neighbor_exchangeability_mean": _safe_float(metadata.get("donor_neighbor_exchangeability_mean")),
        "neighbor_action_alignment_mean": _safe_float(metadata.get("donor_neighbor_action_alignment_mean")),
        "neighbor_hard_pass_rate": _safe_float(metadata.get("donor_neighbor_hard_pass_rate")),
        "neighbor_overlap_valid_rate": _safe_float(metadata.get("donor_neighbor_overlap_valid_rate")),
        "hard_filter_valid": _safe_float(metadata.get("donor_hard_filter_valid")),
        "overlap_valid": _safe_float(metadata.get("donor_overlap_valid")),
    }


def main() -> None:
    args = parse_args()
    bundle_path = str(args.bundle_path)
    bundle_payload = _load_bundle(bundle_path)
    trainer = _build_trainer(bundle_payload, device="cpu", enable_persistent_semantic_store=False)

    dataset_summary = dict(bundle_payload.get("dataset_summary", {}))
    input_sources = dict(bundle_payload.get("input_sources", {}))
    semantics = dict(bundle_payload.get("dataset_semantics", {}))
    feature_schema = derive_eicu_sepsis3_feature_schema(
        labels_csv=str(input_sources.get("eicu_sepsis3_labels_csv", "")),
        trajectory_csv=str(input_sources.get("eicu_sepsis3_trajectory_csv", "")),
        target_field=str(input_sources.get("eicu_target_field", "total_sofa")),
    )
    feature_schema["physiology_sequence_columns"] = list(semantics.get("sequence_feature_names", []))
    feature_schema["intervention_context_columns"] = list(semantics.get("intervention_sequence_feature_names", []))
    feature_schema["patient_feature_names"] = list(semantics.get("patient_feature_names", []))
    dataset = build_eicu_sepsis3_forecasting_dataset(
        labels_csv=str(input_sources.get("eicu_sepsis3_labels_csv", "")),
        trajectory_csv=str(input_sources.get("eicu_sepsis3_trajectory_csv", "")),
        dataset_name="phase3_neighbor_eval",
        history_length=int(dataset_summary.get("history_length", 4)),
        forecast_horizon=int(dataset_summary.get("forecast_horizon", 2)),
        max_series_count=max(8, int(args.max_series_count)),
        enable_kg=bool(input_sources.get("enable_kg", False)),
        kg_directory=str(input_sources.get("kg_directory", "")),
        append_kg_to_patient_static=bool(input_sources.get("append_kg_to_patient_static", True)),
        feature_schema=feature_schema,
    )

    rows: List[Dict[str, object]] = []
    changed_cases: List[Dict[str, object]] = []
    weak_threshold = float(trainer.counterfactual_neighbor_min_consistency)
    trainer.eval()
    with pd.option_context("mode.copy_on_write", True):
        import torch

        with torch.no_grad():
            batch_size = max(1, int(trainer.trainer_config.batch_size))
            for start in range(0, len(dataset.val_samples), batch_size):
                batch_samples = list(dataset.val_samples[start : start + batch_size])
                encodings, manager_results, _, _, _, _ = trainer._forward_batch(batch_samples)
                for sample, encoding, manager_result in zip(batch_samples, encodings, manager_results):
                    ranked_donors = trainer._counterfactual_ranked_donors_from_manager_result(sample, encoding, manager_result)
                    if not ranked_donors:
                        continue
                    donor_rows = [dict(metadata, stay_id=float(entry.stay_id)) for entry, metadata in ranked_donors]
                    old_best = max(
                        donor_rows,
                        key=lambda item: (
                            _safe_float(item.get("donor_pre_neighbor_total_score")),
                            _safe_float(item.get("donor_similarity")),
                        ),
                    )
                    new_best = max(
                        donor_rows,
                        key=lambda item: (
                            _safe_float(item.get("donor_total_score")),
                            _safe_float(item.get("donor_similarity")),
                        ),
                    )
                    changed = _option_identity(old_best) != _option_identity(new_best)
                    row = {
                        "stay_id": _safe_float(sample.metadata.get("stay_id"), -1.0),
                        "series_name": str(sample.metadata.get("series_name", "")),
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
        "weak_neighbor_threshold": float(weak_threshold),
        "old_mean_neighbor_consistency": _mean_metric(old_rows, "neighbor_consistency"),
        "new_mean_neighbor_consistency": _mean_metric(new_rows, "neighbor_consistency"),
        "old_mean_exchangeability": _mean_metric(old_rows, "neighbor_exchangeability_mean"),
        "new_mean_exchangeability": _mean_metric(new_rows, "neighbor_exchangeability_mean"),
        "old_mean_action_alignment": _mean_metric(old_rows, "neighbor_action_alignment_mean"),
        "new_mean_action_alignment": _mean_metric(new_rows, "neighbor_action_alignment_mean"),
        "old_mean_hard_pass_rate": _mean_metric(old_rows, "neighbor_hard_pass_rate"),
        "new_mean_hard_pass_rate": _mean_metric(new_rows, "neighbor_hard_pass_rate"),
        "old_mean_overlap_valid_rate": _mean_metric(old_rows, "neighbor_overlap_valid_rate"),
        "new_mean_overlap_valid_rate": _mean_metric(new_rows, "neighbor_overlap_valid_rate"),
        "old_mean_selected_total_score": _mean_metric(old_rows, "pre_neighbor_total_score"),
        "new_mean_selected_total_score": _mean_metric(new_rows, "total_score"),
        "old_weak_neighbor_rate": float(
            sum(1.0 for row in old_rows if _safe_float(row.get("neighbor_consistency")) < weak_threshold) / max(1, len(old_rows))
        ),
        "new_weak_neighbor_rate": float(
            sum(1.0 for row in new_rows if _safe_float(row.get("neighbor_consistency")) < weak_threshold) / max(1, len(new_rows))
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
