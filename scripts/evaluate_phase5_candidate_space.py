import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infer_eicu_counterfactual_plan import _build_trainer, _load_bundle
from src.tsf_data import build_eicu_sepsis3_forecasting_dataset, derive_eicu_sepsis3_feature_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate legacy vs expanded candidate action space.")
    parser.add_argument("--bundle-path", default="memory_mvp_project/output/analysis/phase8_rollout_bundle.pt")
    parser.add_argument("--max-series-count", type=int, default=64)
    parser.add_argument("--modes", default="legacy,expanded")
    parser.add_argument("--output-path", default="memory_mvp_project/output/analysis/phase5_candidate_space_evaluation.json")
    return parser.parse_args()


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mean_metric(values: List[float]) -> float:
    return float(mean(values)) if values else 0.0


def _case_snapshot(case_detail: Dict[str, object]) -> Dict[str, object]:
    candidate_options = [dict(item) for item in case_detail.get("candidate_options", [])]
    selected_candidate = dict(case_detail.get("selected_candidate", {}))
    generated = [item for item in candidate_options if str(item.get("candidate_source", "")) != "raw_intervention_store"]
    strategy = [
        item
        for item in candidate_options
        if str(item.get("candidate_layer", "")) == "strategy"
        or str(item.get("candidate_anchor_relation", "")) == "safe_deviation"
    ]
    return {
        "stay_id": _safe_float(dict(case_detail.get("query", {})).get("stay_id"), -1.0),
        "candidate_count": float(len(candidate_options)),
        "generated_candidate_count": float(len(generated)),
        "search_candidate_count": float(
            sum(1 for item in candidate_options if str(item.get("candidate_layer", "")) in {"search", "strategy"})
        ),
        "strategy_candidate_count": float(len(strategy)),
        "unique_candidate_source_count": float(len({str(item.get("candidate_source", "")) for item in candidate_options})),
        "parameterized_candidate_count": float(
            sum(1 for item in candidate_options if dict(item.get("candidate_parameter_profile", {})))
        ),
        "generated_hard_invalid_rate": float(
            sum(1 for item in generated if _safe_float(item.get("donor_hard_filter_valid")) < 0.5) / max(1, len(generated))
        ),
        "generated_overlap_invalid_rate": float(
            sum(1 for item in generated if _safe_float(item.get("donor_overlap_valid")) < 0.5) / max(1, len(generated))
        ),
        "selected_source": str(selected_candidate.get("candidate_source", "")),
        "selected_anchor_relation": str(selected_candidate.get("candidate_anchor_relation", "")),
        "selected_layer": str(selected_candidate.get("candidate_layer", "")),
        "selected_score": _safe_float(selected_candidate.get("candidate_selection_score")),
        "selected_is_search": 1.0 if str(selected_candidate.get("candidate_layer", "")) in {"search", "strategy"} else 0.0,
        "selected_is_strategy": 1.0
        if str(selected_candidate.get("candidate_layer", "")) == "strategy"
        or str(selected_candidate.get("candidate_anchor_relation", "")) == "safe_deviation"
        else 0.0,
    }


def main() -> None:
    args = parse_args()
    modes = [value.strip() for value in str(args.modes).split(",") if value.strip()]
    if "legacy" not in modes:
        modes = ["legacy"] + modes

    bundle_path = str(args.bundle_path)
    bundle_payload = _load_bundle(bundle_path)
    trainer = _build_trainer(bundle_payload, device="cpu", enable_persistent_semantic_store=False)
    trainer.counterfactual_candidate_policy = "safe_search"

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
        dataset_name="phase5_candidate_eval",
        history_length=int(dataset_summary.get("history_length", 4)),
        forecast_horizon=int(dataset_summary.get("forecast_horizon", 2)),
        max_series_count=max(8, int(args.max_series_count)),
        enable_kg=bool(input_sources.get("enable_kg", False)),
        kg_directory=str(input_sources.get("kg_directory", "")),
        append_kg_to_patient_static=bool(input_sources.get("append_kg_to_patient_static", True)),
        feature_schema=feature_schema,
    )

    mode_rows: Dict[str, List[Dict[str, object]]] = {}
    for mode in modes:
        trainer.counterfactual_candidate_search_mode = mode
        result = trainer.predict_counterfactual(list(dataset.val_samples), include_predictions=True)
        mode_rows[mode] = [_case_snapshot(case_detail) for case_detail in result.get("case_details", [])]

    summaries: Dict[str, Dict[str, object]] = {}
    for mode, rows in mode_rows.items():
        summaries[mode] = {
            "sample_count": float(len(rows)),
            "mean_candidate_count": _mean_metric([_safe_float(row.get("candidate_count")) for row in rows]),
            "mean_generated_candidate_count": _mean_metric([_safe_float(row.get("generated_candidate_count")) for row in rows]),
            "mean_search_candidate_count": _mean_metric([_safe_float(row.get("search_candidate_count")) for row in rows]),
            "mean_strategy_candidate_count": _mean_metric([_safe_float(row.get("strategy_candidate_count")) for row in rows]),
            "mean_unique_candidate_source_count": _mean_metric([_safe_float(row.get("unique_candidate_source_count")) for row in rows]),
            "mean_parameterized_candidate_count": _mean_metric([_safe_float(row.get("parameterized_candidate_count")) for row in rows]),
            "generated_hard_invalid_rate": _mean_metric([_safe_float(row.get("generated_hard_invalid_rate")) for row in rows]),
            "generated_overlap_invalid_rate": _mean_metric([_safe_float(row.get("generated_overlap_invalid_rate")) for row in rows]),
            "selected_search_rate": _mean_metric([_safe_float(row.get("selected_is_search")) for row in rows]),
            "selected_strategy_rate": _mean_metric([_safe_float(row.get("selected_is_strategy")) for row in rows]),
            "mean_selected_score": _mean_metric([_safe_float(row.get("selected_score")) for row in rows]),
        }

    legacy_lookup = {int(_safe_float(row.get("stay_id"), -1.0)): row for row in mode_rows.get("legacy", [])}
    comparisons: Dict[str, Dict[str, object]] = {}
    for mode, rows in mode_rows.items():
        if mode == "legacy":
            continue
        changed_cases = []
        for row in rows:
            stay_id = int(_safe_float(row.get("stay_id"), -1.0))
            legacy = legacy_lookup.get(stay_id)
            if legacy is None:
                continue
            if str(legacy.get("selected_source", "")) == str(row.get("selected_source", "")) and str(
                legacy.get("selected_anchor_relation", "")
            ) == str(row.get("selected_anchor_relation", "")):
                continue
            changed_cases.append(
                {
                    "stay_id": float(stay_id),
                    "legacy_selected_source": str(legacy.get("selected_source", "")),
                    "legacy_selected_anchor_relation": str(legacy.get("selected_anchor_relation", "")),
                    "current_selected_source": str(row.get("selected_source", "")),
                    "current_selected_anchor_relation": str(row.get("selected_anchor_relation", "")),
                    "legacy_selected_score": _safe_float(legacy.get("selected_score")),
                    "current_selected_score": _safe_float(row.get("selected_score")),
                }
            )
        comparisons[mode] = {
            "changed_case_count": float(len(changed_cases)),
            "changed_case_rate": float(len(changed_cases) / max(1, len(rows))),
            "mean_candidate_count_delta_vs_legacy": summaries[mode]["mean_candidate_count"] - summaries["legacy"]["mean_candidate_count"],
            "mean_strategy_candidate_count_delta_vs_legacy": summaries[mode]["mean_strategy_candidate_count"] - summaries["legacy"]["mean_strategy_candidate_count"],
            "mean_parameterized_candidate_count_delta_vs_legacy": summaries[mode]["mean_parameterized_candidate_count"] - summaries["legacy"]["mean_parameterized_candidate_count"],
            "generated_hard_invalid_rate_delta_vs_legacy": summaries[mode]["generated_hard_invalid_rate"] - summaries["legacy"]["generated_hard_invalid_rate"],
            "generated_overlap_invalid_rate_delta_vs_legacy": summaries[mode]["generated_overlap_invalid_rate"] - summaries["legacy"]["generated_overlap_invalid_rate"],
            "selected_strategy_rate_delta_vs_legacy": summaries[mode]["selected_strategy_rate"] - summaries["legacy"]["selected_strategy_rate"],
            "mean_selected_score_delta_vs_legacy": summaries[mode]["mean_selected_score"] - summaries["legacy"]["mean_selected_score"],
            "changed_cases": changed_cases[:10],
        }

    output = {
        "bundle_path": str(Path(bundle_path).resolve()),
        "modes": modes,
        "mode_summaries": summaries,
        "comparisons_vs_legacy": comparisons,
    }
    output_path = Path(str(args.output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
