import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infer_eicu_counterfactual_plan import (
    _build_trainer,
    _guardrail_decision,
    _load_bundle,
    _rolling_horizon_payload,
    _selected_counterfactual_sample,
    _uncertainty_guardrail_context,
)
from src.tsf_data import build_eicu_sepsis3_forecasting_dataset, derive_eicu_sepsis3_feature_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate paper-ready stability and sensitivity for the current eICU counterfactual stack.")
    parser.add_argument("--bundle-path", default="memory_mvp_project/output/analysis/phase8_rollout_bundle.pt")
    parser.add_argument("--max-series-count", type=int, default=64)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--uncertainty-samples", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=2)
    parser.add_argument("--rollout-discount", type=float, default=0.70)
    parser.add_argument(
        "--configs",
        default="baseline_adaptive:expanded:adaptive:0.00,strict_lb005:expanded:adaptive:0.05,global_pool:expanded:global:0.00,legacy_candidates:legacy:adaptive:0.00",
    )
    parser.add_argument("--output-path", default="memory_mvp_project/output/analysis/phase7_paper_stability_evaluation.json")
    return parser.parse_args()


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mean_metric(values: List[float]) -> float:
    return float(mean(values)) if values else 0.0


def _std_metric(values: List[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def _parse_configs(raw_value: str) -> List[Dict[str, object]]:
    configs: List[Dict[str, object]] = []
    for chunk in str(raw_value).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, candidate_mode, pool_mode, lower_bound = [part.strip() for part in chunk.split(":")]
        configs.append(
            {
                "name": name,
                "candidate_mode": candidate_mode,
                "pool_mode": pool_mode,
                "min_delta_lower_bound": float(lower_bound),
            }
        )
    return configs


def _guardrail_args(config: Dict[str, object], uncertainty_samples: int) -> Namespace:
    return Namespace(
        min_donor_similarity=0.45,
        min_guideline_compatibility=0.25,
        min_donor_total_score=0.0,
        max_missing_care_penalty=0.55,
        max_contraindication_penalty=0.25,
        require_positive_delta=False,
        uncertainty_samples=int(uncertainty_samples),
        max_recommended_forecast_std=0.75,
        min_delta_lower_bound=float(config.get("min_delta_lower_bound", 0.0)),
        disable_uncertainty_guardrail=False,
        rollout_steps=2,
        rollout_discount=0.70,
    )


def _case_selection_snapshot(case_detail: Dict[str, object]) -> Dict[str, object]:
    selected_candidate = dict(case_detail.get("selected_candidate", {}))
    neighborhood_summary = dict(selected_candidate.get("neighborhood_summary", {}))
    return {
        "stay_id": int(_safe_float(dict(case_detail.get("query", {})).get("stay_id"), -1.0)),
        "selected_source": str(selected_candidate.get("candidate_source", "")),
        "selected_anchor_relation": str(selected_candidate.get("candidate_anchor_relation", "")),
        "selected_layer": str(selected_candidate.get("candidate_layer", "")),
        "selected_score": _safe_float(selected_candidate.get("candidate_selection_score")),
        "predicted_delta": _safe_float(selected_candidate.get("predicted_delta")),
        "neighbor_consistency": _safe_float(neighborhood_summary.get("consistency")),
        "donor_similarity": _safe_float(dict(selected_candidate.get("donor", {})).get("donor_similarity")),
        "is_safe_deviation": 1.0 if str(selected_candidate.get("candidate_anchor_relation", "")) == "safe_deviation" else 0.0,
        "is_strategy": 1.0 if str(selected_candidate.get("candidate_layer", "")) == "strategy" else 0.0,
    }


def main() -> None:
    args = parse_args()
    seeds = [int(seed.strip()) for seed in str(args.seeds).split(",") if seed.strip()]
    configs = _parse_configs(str(args.configs))

    bundle_path = str(args.bundle_path)
    bundle_payload = _load_bundle(bundle_path)
    trainer = _build_trainer(bundle_payload, device="cpu", enable_persistent_semantic_store=False)
    trainer.counterfactual_candidate_policy = "safe_search"
    trainer.counterfactual_rollout_steps = max(1, int(args.rollout_steps))
    trainer.counterfactual_rollout_discount = float(args.rollout_discount)

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
        dataset_name="phase7_paper_stability",
        history_length=int(dataset_summary.get("history_length", 4)),
        forecast_horizon=int(dataset_summary.get("forecast_horizon", 2)),
        max_series_count=max(8, int(args.max_series_count)),
        enable_kg=bool(input_sources.get("enable_kg", False)),
        kg_directory=str(input_sources.get("kg_directory", "")),
        append_kg_to_patient_static=bool(input_sources.get("append_kg_to_patient_static", True)),
        feature_schema=feature_schema,
    )

    config_outputs: Dict[str, Dict[str, object]] = {}
    baseline_rows: Dict[int, Dict[str, object]] = {}

    for config in configs:
        trainer.counterfactual_candidate_search_mode = str(config.get("candidate_mode", "expanded"))
        trainer.counterfactual_pool_mode = str(config.get("pool_mode", "adaptive"))
        guardrail_args = _guardrail_args(config, uncertainty_samples=int(args.uncertainty_samples))

        counterfactual_summary = trainer.predict_counterfactual(list(dataset.val_samples), include_predictions=True)
        case_details = [dict(item) for item in counterfactual_summary.get("case_details", [])]
        selected_samples = [
            _selected_counterfactual_sample(sample, case_detail)
            for sample, case_detail in zip(dataset.val_samples, case_details)
        ]
        rollout_summary = (
            trainer.predict_counterfactual_rollout(
                list(dataset.val_samples),
                rollout_steps=max(1, int(args.rollout_steps)),
                include_predictions=True,
            )
            if int(args.rollout_steps) > 1
            else {}
        )

        selection_rows = [_case_selection_snapshot(case_detail) for case_detail in case_details]
        seed_summaries: List[Dict[str, object]] = []
        case_statuses: Dict[int, List[str]] = {int(row["stay_id"]): [] for row in selection_rows}
        case_lower_bounds: Dict[int, List[float]] = {int(row["stay_id"]): [] for row in selection_rows}

        for seed in seeds:
            torch.manual_seed(seed)
            factual_uncertainty = trainer.predict_with_uncertainty(
                list(dataset.val_samples),
                use_memory=True,
                num_samples=max(1, int(args.uncertainty_samples)),
                include_auxiliary=True,
            ).get("samples", [])
            torch.manual_seed(seed + 1000)
            counterfactual_uncertainty = trainer.predict_with_uncertainty(
                selected_samples,
                use_memory=True,
                num_samples=max(1, int(args.uncertainty_samples)),
                include_auxiliary=True,
            ).get("samples", [])

            guardrails = []
            for idx, case_detail in enumerate(case_details):
                uncertainty_context = _uncertainty_guardrail_context(
                    factual_uncertainty=dict(factual_uncertainty[idx]),
                    counterfactual_uncertainty=dict(counterfactual_uncertainty[idx]),
                    predicted_delta=_safe_float(case_detail.get("selected_predicted_delta")),
                )
                rolling_horizon = _rolling_horizon_payload(rollout_summary, case_index=idx) if rollout_summary else {}
                guardrail = _guardrail_decision(case_detail, uncertainty_context, rolling_horizon, guardrail_args)
                guardrails.append(guardrail)
                stay_id = int(selection_rows[idx]["stay_id"])
                case_statuses[stay_id].append(str(guardrail.get("status", "review_only")))
                case_lower_bounds[stay_id].append(_safe_float(dict(guardrail.get("uncertainty", {})).get("delta_lower_bound")))

            ready_rate = _mean_metric([1.0 if str(item.get("status")) == "recommendation_ready" else 0.0 for item in guardrails])
            review_only_rate = _mean_metric([1.0 if str(item.get("status")) == "review_only" else 0.0 for item in guardrails])
            mean_delta_lower_bound = _mean_metric(
                [_safe_float(dict(item.get("uncertainty", {})).get("delta_lower_bound")) for item in guardrails]
            )
            mean_reason_count = _mean_metric([float(len(list(item.get("reasons", [])))) for item in guardrails])
            seed_summaries.append(
                {
                    "seed": seed,
                    "recommendation_ready_rate": ready_rate,
                    "review_only_rate": review_only_rate,
                    "mean_delta_lower_bound": mean_delta_lower_bound,
                    "mean_reason_count": mean_reason_count,
                }
            )

        stable_status_rate = _mean_metric(
            [1.0 if len(set(statuses)) == 1 else 0.0 for statuses in case_statuses.values()]
        )
        config_summary = {
            "sample_count": float(len(selection_rows)),
            "candidate_mode": str(config.get("candidate_mode")),
            "pool_mode": str(config.get("pool_mode")),
            "min_delta_lower_bound": float(config.get("min_delta_lower_bound", 0.0)),
            "mean_selected_score": _mean_metric([_safe_float(row.get("selected_score")) for row in selection_rows]),
            "mean_predicted_delta": _mean_metric([_safe_float(row.get("predicted_delta")) for row in selection_rows]),
            "mean_neighbor_consistency": _mean_metric([_safe_float(row.get("neighbor_consistency")) for row in selection_rows]),
            "safe_deviation_selection_rate": _mean_metric([_safe_float(row.get("is_safe_deviation")) for row in selection_rows]),
            "strategy_selection_rate": _mean_metric([_safe_float(row.get("is_strategy")) for row in selection_rows]),
            "recommendation_ready_rate_mean": _mean_metric([_safe_float(item.get("recommendation_ready_rate")) for item in seed_summaries]),
            "recommendation_ready_rate_std": _std_metric([_safe_float(item.get("recommendation_ready_rate")) for item in seed_summaries]),
            "review_only_rate_mean": _mean_metric([_safe_float(item.get("review_only_rate")) for item in seed_summaries]),
            "mean_delta_lower_bound": _mean_metric([_safe_float(item.get("mean_delta_lower_bound")) for item in seed_summaries]),
            "mean_reason_count": _mean_metric([_safe_float(item.get("mean_reason_count")) for item in seed_summaries]),
            "stable_guardrail_status_rate": stable_status_rate,
        }

        case_examples = []
        for row in selection_rows[:10]:
            stay_id = int(row["stay_id"])
            case_examples.append(
                {
                    **row,
                    "seed_statuses": list(case_statuses.get(stay_id, [])),
                    "mean_delta_lower_bound": _mean_metric(case_lower_bounds.get(stay_id, [])),
                }
            )

        config_outputs[str(config["name"])] = {
            "summary": config_summary,
            "seed_summaries": seed_summaries,
            "example_cases": case_examples,
            "selection_rows": selection_rows,
            "case_statuses": case_statuses,
        }
        if str(config["name"]) == "baseline_adaptive":
            baseline_rows = {int(row["stay_id"]): row for row in selection_rows}

    comparisons_vs_baseline: Dict[str, Dict[str, object]] = {}
    baseline_statuses = config_outputs.get("baseline_adaptive", {}).get("case_statuses", {})
    baseline_summary = dict(config_outputs.get("baseline_adaptive", {}).get("summary", {}))
    for name, payload in config_outputs.items():
        if name == "baseline_adaptive":
            continue
        selection_rows = {int(row["stay_id"]): row for row in payload.get("selection_rows", [])}
        changed_selection_cases = []
        changed_status_cases = []
        for stay_id, row in selection_rows.items():
            baseline_row = baseline_rows.get(stay_id)
            if baseline_row and (
                str(baseline_row.get("selected_source", "")) != str(row.get("selected_source", ""))
                or str(baseline_row.get("selected_anchor_relation", "")) != str(row.get("selected_anchor_relation", ""))
            ):
                changed_selection_cases.append(
                    {
                        "stay_id": float(stay_id),
                        "baseline_source": str(baseline_row.get("selected_source", "")),
                        "baseline_anchor_relation": str(baseline_row.get("selected_anchor_relation", "")),
                        "current_source": str(row.get("selected_source", "")),
                        "current_anchor_relation": str(row.get("selected_anchor_relation", "")),
                    }
                )
            current_statuses = list(dict(payload.get("case_statuses", {})).get(stay_id, []))
            base_statuses = list(dict(baseline_statuses).get(stay_id, []))
            if current_statuses != base_statuses:
                changed_status_cases.append(
                    {
                        "stay_id": float(stay_id),
                        "baseline_statuses": base_statuses,
                        "current_statuses": current_statuses,
                    }
                )
        current_summary = dict(payload.get("summary", {}))
        comparisons_vs_baseline[name] = {
            "selection_changed_case_count": float(len(changed_selection_cases)),
            "selection_changed_case_rate": float(len(changed_selection_cases) / max(1, len(selection_rows))),
            "guardrail_status_changed_case_count": float(len(changed_status_cases)),
            "guardrail_status_changed_case_rate": float(len(changed_status_cases) / max(1, len(selection_rows))),
            "recommendation_ready_rate_mean_delta": _safe_float(current_summary.get("recommendation_ready_rate_mean"))
            - _safe_float(baseline_summary.get("recommendation_ready_rate_mean")),
            "mean_selected_score_delta": _safe_float(current_summary.get("mean_selected_score"))
            - _safe_float(baseline_summary.get("mean_selected_score")),
            "mean_neighbor_consistency_delta": _safe_float(current_summary.get("mean_neighbor_consistency"))
            - _safe_float(baseline_summary.get("mean_neighbor_consistency")),
            "changed_selection_cases": changed_selection_cases[:10],
            "changed_status_cases": changed_status_cases[:10],
        }

    output = {
        "bundle_path": str(Path(bundle_path).resolve()),
        "seeds": seeds,
        "configs": configs,
        "config_results": {
            name: {
                "summary": payload.get("summary", {}),
                "seed_summaries": payload.get("seed_summaries", []),
                "example_cases": payload.get("example_cases", []),
            }
            for name, payload in config_outputs.items()
        },
        "comparisons_vs_baseline": comparisons_vs_baseline,
    }

    output_path = Path(str(args.output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
