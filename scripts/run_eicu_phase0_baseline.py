import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = PROJECT_ROOT / "run_forecasting_experiment.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed Phase 0 layered evaluation baseline for eICU experiments.")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "output" / "analysis" / "phase0_baseline"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--forecast-horizon", type=int, default=2)
    parser.add_argument("--target-field", default="total_sofa")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eicu-max-series", type=int, default=64)
    parser.add_argument("--max-train-windows-per-series", type=int, default=4)
    parser.add_argument("--counterfactual-donor-score-mode", choices=["legacy", "structured"], default="legacy")
    parser.add_argument("--counterfactual-candidate-policy", choices=["donor_only", "generated_best", "safe_search"], default="generated_best")
    parser.add_argument("--multitask-loss-weight", type=float, default=0.15)
    parser.add_argument("--test-uncertainty-samples", type=int, default=8)
    parser.add_argument("--counterfactual-label-top-k", type=int, default=8)
    parser.add_argument("--disable-counterfactual-pool-include-pattern", action="store_true")
    parser.add_argument("--disable-counterfactual-pool-include-trajectory", action="store_true")
    parser.add_argument("--disable-counterfactual-pool-global-backfill", action="store_true")
    parser.add_argument("--counterfactual-pool-min-candidates", type=int, default=64)
    parser.add_argument("--counterfactual-pool-global-limit", type=int, default=256)
    parser.add_argument("--counterfactual-pool-prefilter-top-k", type=int, default=160)
    parser.add_argument("--disable-counterfactual-overlap-filter", action="store_true")
    parser.add_argument("--disable-counterfactual-overlap-fallback", action="store_true")
    parser.add_argument("--counterfactual-overlap-weight", type=float, default=0.12)
    parser.add_argument("--counterfactual-overlap-severity-gap-max", type=float, default=1.60)
    parser.add_argument("--counterfactual-overlap-trend-gap-max", type=float, default=1.25)
    parser.add_argument("--counterfactual-overlap-state-min", type=float, default=0.35)
    parser.add_argument("--counterfactual-overlap-action-min", type=float, default=0.40)
    parser.add_argument("--counterfactual-rollout-steps", type=int, default=1)
    parser.add_argument("--counterfactual-rollout-discount", type=float, default=0.70)
    parser.add_argument("--counterfactual-reranker-mode", choices=["rule_only", "learned_linear"], default="rule_only")
    parser.add_argument("--counterfactual-reranker-blend-weight", type=float, default=0.35)
    parser.add_argument("--counterfactual-reranker-train-top-k", type=int, default=4)
    parser.add_argument("--counterfactual-reranker-max-samples", type=int, default=256)
    parser.add_argument("--counterfactual-reranker-min-examples", type=int, default=32)
    parser.add_argument("--counterfactual-reranker-ridge-l2", type=float, default=1e-3)
    parser.add_argument("--phase0-min-donor-similarity", type=float, default=0.45)
    parser.add_argument("--phase0-min-guideline-compatibility", type=float, default=0.25)
    parser.add_argument("--phase0-min-donor-total-score", type=float, default=0.0)
    parser.add_argument("--phase0-max-missing-care-penalty", type=float, default=0.55)
    parser.add_argument("--phase0-max-contraindication-penalty", type=float, default=0.25)
    parser.add_argument("--phase0-require-positive-delta", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _nested_get(payload: dict, path: list[str], default=0.0):
    cursor = payload
    for key in path:
        if not isinstance(cursor, dict):
            return default
        cursor = cursor.get(key)
    return default if cursor is None else cursor


def main() -> None:
    args = parse_args()
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows = []
    for seed in seeds:
        output_json = output_dir / f"phase0_seed{seed}_full.json"
        layered_json = output_dir / f"phase0_seed{seed}_layered.json"
        command = [
            sys.executable,
            str(RUN_SCRIPT),
            "--dataset-format",
            "eicu_sepsis3",
            "--dataset-name",
            f"eicu_phase0_baseline_seed{seed}",
            "--history-length",
            str(args.history_length),
            "--forecast-horizon",
            str(args.forecast_horizon),
            "--eicu-target-field",
            args.target_field,
            "--eicu-max-series",
            str(args.eicu_max_series),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--max-train-windows-per-series",
            str(args.max_train_windows_per_series),
            "--device",
            args.device,
            "--encoder-type",
            "transformer",
            "--transformer-d-model",
            "96",
            "--transformer-layers",
            "2",
            "--transformer-heads",
            "4",
            "--transformer-ff-dim",
            "192",
            "--transformer-dropout",
            "0.1",
            "--memory-direct-residual-mode",
            "adaptive",
            "--memory-path-coordination-mode",
            "sum",
            "--enable-kg",
            "--enable-explicit-kg-path",
            "--counterfactual-donor-score-mode",
            args.counterfactual_donor_score_mode,
            "--counterfactual-candidate-policy",
            args.counterfactual_candidate_policy,
            "--counterfactual-label-top-k",
            str(args.counterfactual_label_top_k),
            "--counterfactual-pool-min-candidates",
            str(args.counterfactual_pool_min_candidates),
            "--counterfactual-pool-global-limit",
            str(args.counterfactual_pool_global_limit),
            "--counterfactual-pool-prefilter-top-k",
            str(args.counterfactual_pool_prefilter_top_k),
            "--counterfactual-overlap-weight",
            str(args.counterfactual_overlap_weight),
            "--counterfactual-overlap-severity-gap-max",
            str(args.counterfactual_overlap_severity_gap_max),
            "--counterfactual-overlap-trend-gap-max",
            str(args.counterfactual_overlap_trend_gap_max),
            "--counterfactual-overlap-state-min",
            str(args.counterfactual_overlap_state_min),
            "--counterfactual-overlap-action-min",
            str(args.counterfactual_overlap_action_min),
            "--counterfactual-rollout-steps",
            str(args.counterfactual_rollout_steps),
            "--counterfactual-rollout-discount",
            str(args.counterfactual_rollout_discount),
            "--counterfactual-reranker-mode",
            args.counterfactual_reranker_mode,
            "--counterfactual-reranker-blend-weight",
            str(args.counterfactual_reranker_blend_weight),
            "--counterfactual-reranker-train-top-k",
            str(args.counterfactual_reranker_train_top_k),
            "--counterfactual-reranker-max-samples",
            str(args.counterfactual_reranker_max_samples),
            "--counterfactual-reranker-min-examples",
            str(args.counterfactual_reranker_min_examples),
            "--counterfactual-reranker-ridge-l2",
            str(args.counterfactual_reranker_ridge_l2),
            "--multitask-loss-weight",
            str(args.multitask_loss_weight),
            "--test-uncertainty-samples",
            str(args.test_uncertainty_samples),
            "--phase0-min-donor-similarity",
            str(args.phase0_min_donor_similarity),
            "--phase0-min-guideline-compatibility",
            str(args.phase0_min_guideline_compatibility),
            "--phase0-min-donor-total-score",
            str(args.phase0_min_donor_total_score),
            "--phase0-max-missing-care-penalty",
            str(args.phase0_max_missing_care_penalty),
            "--phase0-max-contraindication-penalty",
            str(args.phase0_max_contraindication_penalty),
            "--seed",
            str(seed),
            "--output-json",
            str(output_json),
        ]
        if args.disable_counterfactual_pool_include_pattern:
            command.append("--disable-counterfactual-pool-include-pattern")
        if args.disable_counterfactual_pool_include_trajectory:
            command.append("--disable-counterfactual-pool-include-trajectory")
        if args.disable_counterfactual_pool_global_backfill:
            command.append("--disable-counterfactual-pool-global-backfill")
        if args.disable_counterfactual_overlap_filter:
            command.append("--disable-counterfactual-overlap-filter")
        if args.disable_counterfactual_overlap_fallback:
            command.append("--disable-counterfactual-overlap-fallback")
        if args.phase0_require_positive_delta:
            command.append("--phase0-require-positive-delta")

        subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        payload = _load_json(output_json)
        layered_payload = dict(payload.get("layered_evaluation_baseline", {}))
        layered_json.write_text(json.dumps(layered_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        run_rows.append(
            {
                "seed": seed,
                "full_output_json": str(output_json),
                "layered_output_json": str(layered_json),
                "hybrid_mae": float(_nested_get(payload, ["test_metrics", "mae"], 0.0)),
                "base_mae": float(_nested_get(payload, ["base_only", "mae"], 0.0)),
                "improvement_mae": float(_nested_get(payload, ["memory_effectiveness", "improvement_mae"], 0.0)),
                "donor_similarity_mean": float(
                    _nested_get(layered_payload, ["retrieval_layer", "selected_donor_metrics", "donor_similarity", "mean"], 0.0)
                ),
                "donor_guideline_mean": float(
                    _nested_get(layered_payload, ["retrieval_layer", "selected_donor_metrics", "donor_guideline_compatibility", "mean"], 0.0)
                ),
                "donor_overlap_score_mean": float(
                    _nested_get(layered_payload, ["retrieval_layer", "selected_donor_metrics", "donor_overlap_score", "mean"], 0.0)
                ),
                "donor_overlap_valid_rate": float(
                    _nested_get(layered_payload, ["retrieval_layer", "match_quality", "overlap_valid_rate"], 0.0)
                ),
                "donor_overlap_fallback_rate": float(
                    _nested_get(layered_payload, ["retrieval_layer", "match_quality", "overlap_fallback_rate"], 0.0)
                ),
                "donor_learned_reranker_score_mean": float(
                    _nested_get(layered_payload, ["retrieval_layer", "selected_donor_metrics", "donor_learned_reranker_score", "mean"], 0.0)
                ),
                "candidate_count_mean": float(
                    _nested_get(layered_payload, ["candidate_layer", "candidate_count_distribution", "mean"], 0.0)
                ),
                "search_candidate_case_rate": float(
                    _nested_get(layered_payload, ["candidate_layer", "search_candidate_case_rate"], 0.0)
                ),
                "search_candidate_selected_rate": float(
                    _nested_get(layered_payload, ["candidate_layer", "search_candidate_selected_rate"], 0.0)
                ),
                "predicted_improvement_rate": float(
                    _nested_get(layered_payload, ["counterfactual_layer", "selection_quality", "predicted_improvement_rate"], 0.0)
                ),
                "rollout_mean_discounted_cumulative_delta": float(
                    _nested_get(layered_payload, ["rolling_horizon_layer", "mean_discounted_cumulative_delta"], 0.0)
                ),
                "rollout_positive_discounted_cumulative_rate": float(
                    _nested_get(layered_payload, ["rolling_horizon_layer", "positive_discounted_cumulative_rate"], 0.0)
                ),
                "rollout_second_step_available_rate": float(
                    _nested_get(layered_payload, ["rolling_horizon_layer", "second_step_available_rate"], 0.0)
                ),
                "rollout_stable_candidate_source_rate": float(
                    _nested_get(layered_payload, ["rolling_horizon_layer", "stable_candidate_source_rate"], 0.0)
                ),
                "recommendation_ready_rate": float(
                    _nested_get(layered_payload, ["report_layer", "status_rates", "recommendation_ready_rate"], 0.0)
                ),
                "review_only_rate": float(
                    _nested_get(layered_payload, ["report_layer", "status_rates", "review_only_rate"], 0.0)
                ),
            }
        )

    csv_path = output_dir / "phase0_baseline_runs.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(run_rows[0].keys()) if run_rows else [])
        if run_rows:
            writer.writeheader()
            writer.writerows(run_rows)

    summary = {
        "seeds": seeds,
        "run_count": len(run_rows),
        "runs": run_rows,
        "aggregate": {
            "hybrid_mae_mean": mean(row["hybrid_mae"] for row in run_rows) if run_rows else 0.0,
            "base_mae_mean": mean(row["base_mae"] for row in run_rows) if run_rows else 0.0,
            "improvement_mae_mean": mean(row["improvement_mae"] for row in run_rows) if run_rows else 0.0,
            "donor_similarity_mean": mean(row["donor_similarity_mean"] for row in run_rows) if run_rows else 0.0,
            "donor_guideline_mean": mean(row["donor_guideline_mean"] for row in run_rows) if run_rows else 0.0,
            "donor_overlap_score_mean": mean(row["donor_overlap_score_mean"] for row in run_rows) if run_rows else 0.0,
            "donor_overlap_valid_rate_mean": mean(row["donor_overlap_valid_rate"] for row in run_rows) if run_rows else 0.0,
            "donor_overlap_fallback_rate_mean": mean(row["donor_overlap_fallback_rate"] for row in run_rows) if run_rows else 0.0,
            "donor_learned_reranker_score_mean": mean(row["donor_learned_reranker_score_mean"] for row in run_rows) if run_rows else 0.0,
            "candidate_count_mean": mean(row["candidate_count_mean"] for row in run_rows) if run_rows else 0.0,
            "search_candidate_case_rate_mean": mean(row["search_candidate_case_rate"] for row in run_rows) if run_rows else 0.0,
            "search_candidate_selected_rate_mean": mean(row["search_candidate_selected_rate"] for row in run_rows) if run_rows else 0.0,
            "predicted_improvement_rate_mean": mean(row["predicted_improvement_rate"] for row in run_rows) if run_rows else 0.0,
            "rollout_mean_discounted_cumulative_delta_mean": mean(
                row["rollout_mean_discounted_cumulative_delta"] for row in run_rows
            ) if run_rows else 0.0,
            "rollout_positive_discounted_cumulative_rate_mean": mean(
                row["rollout_positive_discounted_cumulative_rate"] for row in run_rows
            ) if run_rows else 0.0,
            "rollout_second_step_available_rate_mean": mean(
                row["rollout_second_step_available_rate"] for row in run_rows
            ) if run_rows else 0.0,
            "rollout_stable_candidate_source_rate_mean": mean(
                row["rollout_stable_candidate_source_rate"] for row in run_rows
            ) if run_rows else 0.0,
            "recommendation_ready_rate_mean": mean(row["recommendation_ready_rate"] for row in run_rows) if run_rows else 0.0,
            "review_only_rate_mean": mean(row["review_only_rate"] for row in run_rows) if run_rows else 0.0,
        },
        "artifacts": {
            "csv_path": str(csv_path),
        },
    }
    summary_path = output_dir / "phase0_baseline_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
