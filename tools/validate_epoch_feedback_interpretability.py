import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = PROJECT_ROOT / "run_forecasting_experiment.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small baseline-vs-epoch-feedback comparison to verify interpretable training feedback activates without catastrophically harming factual forecasting."
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "output" / "analysis" / "epoch_feedback_interpretability"))
    parser.add_argument("--eicu-max-series", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-train-windows-per-series", type=int, default=4)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--forecast-horizon", type=int, default=2)
    parser.add_argument("--encoder-type", default="transformer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--error-attribution-max-samples", type=int, default=256)
    parser.add_argument("--include-final-epoch-check", action="store_true")
    return parser.parse_args()


def _error_slice(
    factual: Dict[str, object],
    split: str,
    slice_name: str,
) -> Dict[str, float]:
    attribution = factual.get("memory_path_audit", {}).get("error_attribution", {}).get(split, {})
    slices = attribution.get("slices", {})
    selected = dict(slices.get(slice_name, {}))
    return {
        "count": float(selected.get("count", 0.0)),
        "kg_underused_rate": float(selected.get("kg_underused_rate", 0.0)),
        "path_conflict_rate": float(selected.get("path_conflict_rate", 0.0)),
        "semantic_hit_rate": float(selected.get("semantic_hit_rate", 0.0)),
        "archive_use_rate": float(selected.get("archive_use_rate", 0.0)),
        "mean_semantic_top_score": float(selected.get("mean_semantic_top_score", 0.0)),
        "mean_kg_gate": float(selected.get("mean_kg_gate", 0.0)),
        "mean_path_alignment": float(selected.get("mean_path_alignment", 0.0)),
        "mean_experience_gate": float(selected.get("mean_experience_gate", 0.0)),
        "mean_memory_delta_strength": float(selected.get("mean_memory_delta_strength", 0.0)),
    }


def _extract_metrics(payload: Dict[str, object]) -> Dict[str, float | bool]:
    factual = payload["evaluation_lines"]["factual_forecasting"]
    enabled = factual["metrics"]["memory_enabled"]
    audit = factual["memory_path_audit"]["factual_path_audit"]
    semantic = audit.get("semantic_retrieval", {})
    persistent = payload.get("persistent_memory", {})
    epoch_feedback = payload.get("memory_diagnostics", {}).get("epoch_feedback", {})
    history = list(epoch_feedback.get("history", []))
    last_feedback = history[-1] if history else {}
    last_attr = dict(last_feedback.get("error_attribution", {}))
    last_weights = dict(last_feedback.get("updated_weights", {}))
    val_hard = _error_slice(factual, "validation", "hard_cases")
    val_kg = _error_slice(factual, "validation", "kg_active_cases")
    test_hard = _error_slice(factual, "test", "hard_cases")
    test_kg = _error_slice(factual, "test", "kg_active_cases")
    val_attr = factual.get("memory_path_audit", {}).get("error_attribution", {}).get("validation", {})
    test_attr = factual.get("memory_path_audit", {}).get("error_attribution", {}).get("test", {})
    return {
        "hybrid_mae": float(enabled["mae"]),
        "hybrid_rmse": float(enabled["rmse"]),
        "improvement_mae": float(factual["improvement"]["improvement_mae"]),
        "path_alignment": float(audit.get("direct_memory_means", {}).get("path_alignment", 0.0)),
        "experience_archive_use_rate": float(audit.get("experience_archive_use_rate", 0.0)),
        "semantic_hit_rate": float(semantic.get("hit_rate", 0.0)),
        "semantic_mean_top_score": float(semantic.get("mean_top_score", 0.0)),
        "loaded_persistent_prototypes": float(persistent.get("loaded_persistent_prototypes", 0.0)),
        "epoch_feedback_enabled": bool(epoch_feedback.get("enabled", False)),
        "epoch_feedback_history_len": float(len(history)),
        "validation_error_attribution_available": bool(val_attr),
        "test_error_attribution_available": bool(test_attr),
        "hard_case_count": float(last_attr.get("hard_case_count", 0.0)),
        "kg_underused_rate": float(last_attr.get("kg_underused_rate", 0.0)),
        "path_conflict_rate": float(last_attr.get("path_conflict_rate", 0.0)),
        "semantic_hit_rate_on_hard_cases": float(last_attr.get("semantic_hit_rate_on_hard_cases", 0.0)),
        "archive_use_rate_on_hard_cases": float(last_attr.get("archive_use_rate_on_hard_cases", 0.0)),
        "final_hard_example_weight": float(last_weights.get("hard_example_weight", 0.0)),
        "final_kg_consistency_weight": float(last_weights.get("kg_consistency_weight", 0.0)),
        "final_path_alignment_weight": float(last_weights.get("path_alignment_weight", 0.0)),
        "validation_hard_path_conflict_rate": float(val_hard["path_conflict_rate"]),
        "validation_hard_mean_path_alignment": float(val_hard["mean_path_alignment"]),
        "validation_hard_mean_memory_delta_strength": float(val_hard["mean_memory_delta_strength"]),
        "validation_kg_active_count": float(val_kg["count"]),
        "validation_kg_active_kg_underused_rate": float(val_kg["kg_underused_rate"]),
        "validation_kg_active_mean_kg_gate": float(val_kg["mean_kg_gate"]),
        "validation_kg_active_mean_memory_delta_strength": float(val_kg["mean_memory_delta_strength"]),
        "test_hard_path_conflict_rate": float(test_hard["path_conflict_rate"]),
        "test_hard_mean_path_alignment": float(test_hard["mean_path_alignment"]),
        "test_hard_mean_memory_delta_strength": float(test_hard["mean_memory_delta_strength"]),
        "test_kg_active_count": float(test_kg["count"]),
        "test_kg_active_kg_underused_rate": float(test_kg["kg_underused_rate"]),
        "test_kg_active_mean_kg_gate": float(test_kg["mean_kg_gate"]),
        "test_kg_active_mean_memory_delta_strength": float(test_kg["mean_memory_delta_strength"]),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stores_dir = output_dir / "stores"
    stores_dir.mkdir(parents=True, exist_ok=True)

    common_args: List[str] = [
        sys.executable,
        str(RUN_SCRIPT),
        "--dataset-format",
        "eicu_sepsis3",
        "--eicu-target-field",
        "total_sofa",
        "--eicu-max-series",
        str(args.eicu_max_series),
        "--history-length",
        str(args.history_length),
        "--forecast-horizon",
        str(args.forecast_horizon),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--max-train-windows-per-series",
        str(args.max_train_windows_per_series),
        "--encoder-type",
        str(args.encoder_type),
        "--seed",
        str(args.seed),
        "--enable-kg",
        "--disable-kg-static-concat",
        "--enable-explicit-kg-path",
        "--kg-residual-weight",
        "0.06",
        "--memory-path-coordination-mode",
        "adaptive",
        "--persistent-semantic-top-k",
        "3",
        "--error-attribution-max-samples",
        str(args.error_attribution_max_samples),
    ]

    variants = {
        "baseline_primed": [
            "--persistent-memory-store",
            str(stores_dir / "baseline_primed_store"),
            "--prime-persistent-memory-before-fit",
        ],
        "epoch_feedback_primed": [
            "--persistent-memory-store",
            str(stores_dir / "epoch_feedback_primed_store"),
            "--prime-persistent-memory-before-fit",
            "--enable-epoch-feedback",
            "--hard-example-weight",
            "0.35",
            "--kg-consistency-weight",
            "0.06",
            "--path-alignment-weight",
            "0.05",
            "--epoch-feedback-momentum",
            "0.35",
            "--feedback-top-error-rate",
            "0.35",
        ],
    }
    if args.include_final_epoch_check:
        variants["baseline_primed_final_epoch"] = [
            "--persistent-memory-store",
            str(stores_dir / "baseline_primed_final_epoch_store"),
            "--prime-persistent-memory-before-fit",
            "--checkpoint-selection-mode",
            "final_epoch",
        ]
        variants["epoch_feedback_primed_final_epoch"] = [
            "--persistent-memory-store",
            str(stores_dir / "epoch_feedback_primed_final_epoch_store"),
            "--prime-persistent-memory-before-fit",
            "--enable-epoch-feedback",
            "--hard-example-weight",
            "0.35",
            "--kg-consistency-weight",
            "0.06",
            "--path-alignment-weight",
            "0.05",
            "--epoch-feedback-momentum",
            "0.35",
            "--feedback-top-error-rate",
            "0.35",
            "--checkpoint-selection-mode",
            "final_epoch",
        ]

    payloads: Dict[str, Dict[str, object]] = {}
    metric_rows: Dict[str, Dict[str, float | bool]] = {}
    for name, variant_args in variants.items():
        output_json = output_dir / f"{name}.json"
        command = common_args + variant_args + ["--output-json", str(output_json)]
        subprocess.run(command, cwd=str(PROJECT_ROOT.parent), check=True)
        with output_json.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        payloads[name] = payload
        metric_rows[name] = _extract_metrics(payload)

    baseline = metric_rows["baseline_primed"]
    feedback = metric_rows["epoch_feedback_primed"]
    summary = {
        "question": "Does epoch feedback improve interpretability signals without catastrophically hurting factual forecasting?",
        "variant_metrics": metric_rows,
        "deltas_feedback_minus_baseline": {
            key: float(feedback[key]) - float(baseline[key])
            for key in [
                "hybrid_mae",
                "hybrid_rmse",
                "improvement_mae",
                "path_alignment",
                "experience_archive_use_rate",
                "semantic_hit_rate",
                "kg_underused_rate",
                "path_conflict_rate",
                "semantic_hit_rate_on_hard_cases",
                "archive_use_rate_on_hard_cases",
                "validation_hard_path_conflict_rate",
                "validation_hard_mean_path_alignment",
                "validation_kg_active_mean_memory_delta_strength",
                "test_hard_path_conflict_rate",
                "test_hard_mean_path_alignment",
                "test_kg_active_mean_memory_delta_strength",
            ]
        },
        "acceptance_checks": {
            "epoch_feedback_history_present": bool(feedback["epoch_feedback_history_len"] >= 1.0),
            "baseline_and_feedback_have_symmetric_error_attribution": bool(
                baseline["validation_error_attribution_available"]
                and baseline["test_error_attribution_available"]
                and feedback["validation_error_attribution_available"]
                and feedback["test_error_attribution_available"]
            ),
            "feedback_preserves_semantic_signal": bool(feedback["semantic_hit_rate"] > 0.0),
            "feedback_reduces_or_holds_validation_hard_path_conflict": bool(
                float(feedback["validation_hard_path_conflict_rate"]) <= float(baseline["validation_hard_path_conflict_rate"]) + 0.05
            ),
            "feedback_reduces_or_holds_test_hard_path_conflict": bool(
                float(feedback["test_hard_path_conflict_rate"]) <= float(baseline["test_hard_path_conflict_rate"]) + 0.05
            ),
            "feedback_reduces_or_holds_validation_kg_underuse": bool(
                float(feedback["validation_kg_active_kg_underused_rate"]) <= float(baseline["validation_kg_active_kg_underused_rate"]) + 0.05
            ),
            "feedback_not_catastrophic_on_mae": bool(
                float(feedback["hybrid_mae"]) <= float(baseline["hybrid_mae"]) + 0.03
            ),
        },
        "memory_diagnostics_snapshot": {
            name: payloads[name].get("memory_diagnostics", {}).get("epoch_feedback", {})
            for name in payloads
        },
    }
    if args.include_final_epoch_check:
        final_baseline = metric_rows["baseline_primed_final_epoch"]
        final_feedback = metric_rows["epoch_feedback_primed_final_epoch"]
        summary["final_epoch_comparison"] = {
            "question": "When selecting the final epoch instead of the best validation checkpoint, does epoch feedback still help?",
            "variant_metrics": {
                "baseline_primed_final_epoch": final_baseline,
                "epoch_feedback_primed_final_epoch": final_feedback,
            },
            "deltas_feedback_minus_baseline": {
                key: float(final_feedback[key]) - float(final_baseline[key])
                for key in [
                    "hybrid_mae",
                    "hybrid_rmse",
                    "improvement_mae",
                    "path_alignment",
                    "experience_archive_use_rate",
                    "semantic_hit_rate",
                    "test_hard_path_conflict_rate",
                    "test_hard_mean_path_alignment",
                    "test_hard_mean_memory_delta_strength",
                    "test_kg_active_mean_memory_delta_strength",
                ]
            },
            "acceptance_checks": {
                "feedback_not_better_than_baseline_on_final_epoch_mae": bool(
                    float(final_feedback["hybrid_mae"]) >= float(final_baseline["hybrid_mae"])
                ),
                "feedback_collapses_or_reduces_archive_use": bool(
                    float(final_feedback["experience_archive_use_rate"])
                    <= float(final_baseline["experience_archive_use_rate"]) - 0.5
                ),
                "feedback_increases_or_holds_test_hard_path_conflict": bool(
                    float(final_feedback["test_hard_path_conflict_rate"])
                    >= float(final_baseline["test_hard_path_conflict_rate"]) - 0.01
                ),
                "best_checkpoint_masks_later_drift": bool(
                    abs(float(feedback["hybrid_mae"]) - float(baseline["hybrid_mae"])) < 1e-12
                    and abs(float(final_feedback["test_hard_path_conflict_rate"]) - float(final_baseline["test_hard_path_conflict_rate"])) > 0.1
                ),
                "final_epoch_archive_collapse_is_general": bool(
                    float(final_feedback["experience_archive_use_rate"]) <= 0.05
                    and float(final_baseline["experience_archive_use_rate"]) <= 0.05
                ),
            },
        }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
