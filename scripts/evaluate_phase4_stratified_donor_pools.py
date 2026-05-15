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
from src.tsf_data import build_eicu_sepsis3_forecasting_dataset, derive_eicu_sepsis3_feature_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate stratified donor pool modes against global donor retrieval.")
    parser.add_argument("--bundle-path", default="memory_mvp_project/output/analysis/phase8_rollout_bundle.pt")
    parser.add_argument("--max-series-count", type=int, default=64)
    parser.add_argument(
        "--modes",
        default="global,same_hospital,same_unit,adaptive",
        help="Comma separated donor pool modes to compare.",
    )
    parser.add_argument("--output-path", default="memory_mvp_project/output/analysis/phase4_stratified_donor_pool_evaluation.json")
    return parser.parse_args()


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _selection_snapshot(option: Dict[str, object]) -> Dict[str, object]:
    tags = [str(value) for value in option.get("donor_pool_tags", [])]
    return {
        "stay_id": _safe_float(option.get("stay_id"), -1.0),
        "similarity": _safe_float(option.get("donor_similarity")),
        "guideline_compatibility": _safe_float(option.get("donor_guideline_compatibility")),
        "state_match": _safe_float(option.get("donor_state_match")),
        "total_score": _safe_float(option.get("donor_total_score")),
        "pool_match_score": _safe_float(option.get("donor_pool_match_score")),
        "pool_match_reward": _safe_float(option.get("donor_pool_match_reward")),
        "neighbor_consistency": _safe_float(option.get("donor_neighbor_consistency")),
        "hard_filter_valid": _safe_float(option.get("donor_hard_filter_valid")),
        "overlap_valid": _safe_float(option.get("donor_overlap_valid")),
        "pool_tags": tags,
        "same_hospital": 1.0 if "same_hospital" in tags else 0.0,
        "same_unit_type": 1.0 if "same_unit_type" in tags else 0.0,
        "same_infection_anchor": 1.0 if "same_infection_anchor" in tags else 0.0,
    }


def _identity(snapshot: Dict[str, object]) -> str:
    return f"{int(_safe_float(snapshot.get('stay_id'), -1.0))}|{_safe_float(snapshot.get('total_score')):.6f}"


def _mean_metric(rows: List[Dict[str, object]], key: str) -> float:
    return float(mean(_safe_float(row.get(key)) for row in rows)) if rows else 0.0


def main() -> None:
    args = parse_args()
    modes = [value.strip() for value in str(args.modes).split(",") if value.strip()]
    if "global" not in modes:
        modes = ["global"] + modes

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
        dataset_name="phase4_pool_eval",
        history_length=int(dataset_summary.get("history_length", 4)),
        forecast_horizon=int(dataset_summary.get("forecast_horizon", 2)),
        max_series_count=max(8, int(args.max_series_count)),
        enable_kg=bool(input_sources.get("enable_kg", False)),
        kg_directory=str(input_sources.get("kg_directory", "")),
        append_kg_to_patient_static=bool(input_sources.get("append_kg_to_patient_static", True)),
        feature_schema=feature_schema,
    )

    original_mode = str(trainer.counterfactual_pool_mode)
    per_mode_rows: Dict[str, List[Dict[str, object]]] = {mode: [] for mode in modes}
    trainer.eval()
    with pd.option_context("mode.copy_on_write", True):
        import torch

        with torch.no_grad():
            batch_size = max(1, int(trainer.trainer_config.batch_size))
            for mode in modes:
                trainer.counterfactual_pool_mode = mode
                for start in range(0, len(dataset.val_samples), batch_size):
                    batch_samples = list(dataset.val_samples[start : start + batch_size])
                    encodings, manager_results, _, _, _, _ = trainer._forward_batch(batch_samples)
                    for sample, encoding, manager_result in zip(batch_samples, encodings, manager_results):
                        ranked_donors = trainer._counterfactual_ranked_donors_from_manager_result(sample, encoding, manager_result)
                        if not ranked_donors:
                            continue
                        entry, metadata = ranked_donors[0]
                        snapshot = _selection_snapshot(dict(metadata, stay_id=float(entry.stay_id)))
                        per_mode_rows[mode].append(
                            {
                                "query_stay_id": _safe_float(sample.metadata.get("stay_id"), -1.0),
                                "series_name": str(sample.metadata.get("series_name", "")),
                                "selection": snapshot,
                            }
                        )
    trainer.counterfactual_pool_mode = original_mode

    baseline_rows = per_mode_rows.get("global", [])
    baseline_lookup = {
        int(_safe_float(row.get("query_stay_id"), -1.0)): row["selection"]
        for row in baseline_rows
    }

    summaries: Dict[str, Dict[str, object]] = {}
    comparisons: Dict[str, Dict[str, object]] = {}
    for mode, rows in per_mode_rows.items():
        selections = [row["selection"] for row in rows]
        summaries[mode] = {
            "sample_count": float(len(selections)),
            "mean_similarity": _mean_metric(selections, "similarity"),
            "mean_guideline_compatibility": _mean_metric(selections, "guideline_compatibility"),
            "mean_neighbor_consistency": _mean_metric(selections, "neighbor_consistency"),
            "mean_total_score": _mean_metric(selections, "total_score"),
            "mean_pool_match_score": _mean_metric(selections, "pool_match_score"),
            "mean_pool_match_reward": _mean_metric(selections, "pool_match_reward"),
            "hard_filter_pass_rate": _mean_metric(selections, "hard_filter_valid"),
            "overlap_pass_rate": _mean_metric(selections, "overlap_valid"),
            "same_hospital_rate": _mean_metric(selections, "same_hospital"),
            "same_unit_type_rate": _mean_metric(selections, "same_unit_type"),
            "same_infection_anchor_rate": _mean_metric(selections, "same_infection_anchor"),
        }
        if mode == "global":
            continue
        changed_cases = []
        for row in rows:
            query_stay_id = int(_safe_float(row.get("query_stay_id"), -1.0))
            baseline = baseline_lookup.get(query_stay_id)
            current = row["selection"]
            if baseline is None or _identity(baseline) == _identity(current):
                continue
            changed_cases.append(
                {
                    "query_stay_id": float(query_stay_id),
                    "series_name": str(row.get("series_name", "")),
                    "global": baseline,
                    "current": current,
                }
            )
        comparisons[mode] = {
            "changed_case_count": float(len(changed_cases)),
            "changed_case_rate": float(len(changed_cases) / max(1, len(rows))),
            "mean_similarity_delta_vs_global": summaries[mode]["mean_similarity"] - summaries["global"]["mean_similarity"],
            "mean_guideline_delta_vs_global": summaries[mode]["mean_guideline_compatibility"] - summaries["global"]["mean_guideline_compatibility"],
            "mean_neighbor_consistency_delta_vs_global": summaries[mode]["mean_neighbor_consistency"] - summaries["global"]["mean_neighbor_consistency"],
            "mean_total_score_delta_vs_global": summaries[mode]["mean_total_score"] - summaries["global"]["mean_total_score"],
            "mean_pool_match_score_delta_vs_global": summaries[mode]["mean_pool_match_score"] - summaries["global"]["mean_pool_match_score"],
            "mean_pool_match_reward_delta_vs_global": summaries[mode]["mean_pool_match_reward"] - summaries["global"]["mean_pool_match_reward"],
            "same_hospital_rate_delta_vs_global": summaries[mode]["same_hospital_rate"] - summaries["global"]["same_hospital_rate"],
            "same_unit_type_rate_delta_vs_global": summaries[mode]["same_unit_type_rate"] - summaries["global"]["same_unit_type_rate"],
            "same_infection_anchor_rate_delta_vs_global": summaries[mode]["same_infection_anchor_rate"] - summaries["global"]["same_infection_anchor_rate"],
            "changed_cases": changed_cases[:8],
        }

    output = {
        "bundle_path": str(Path(bundle_path).resolve()),
        "modes": modes,
        "mode_summaries": summaries,
        "comparisons_vs_global": comparisons,
    }
    output_path = Path(str(args.output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
