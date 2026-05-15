import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from src.counterfactual_plan_renderer import render_counterfactual_case_report
from src.forecasting_metrics import forecasting_metrics
from src.manifold_forecasting_trainer import (
    EndToEndForecastingManifoldTrainer,
    ForecastingTrainerConfig,
)
from src.manifold_memory import ManifoldMemoryConfig
from src.persistent_memory_store import PersistentExperienceStore, forecast_sample_identity
from src.ts_formation import PATTERN_LABELS, TRAJECTORY_LABELS
from src.tsf_data import build_eicu_sepsis3_forecasting_dataset, build_tsf_forecasting_dataset


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EICU_PROCESSED_DIR = PROJECT_ROOT.parent / "eicu数据库" / "processed"
DEFAULT_EICU_LABELS_CSV = DEFAULT_EICU_PROCESSED_DIR / "eicu_sepsis3_labels.csv"
DEFAULT_EICU_TRAJECTORY_CSV = DEFAULT_EICU_PROCESSED_DIR / "eicu_sepsis3_sofa_6h_trajectory.csv"
DEFAULT_EICU_KG_DIR = PROJECT_ROOT / "input" / "knowledge" / "13_processed_ready" / "sepsis_kg_guideline_enhanced"


def parse_args():
    parser = argparse.ArgumentParser(description="Run forecasting experiments on TSForecasting benchmarks or eICU Sepsis-3 trajectories.")
    parser.add_argument("--dataset-format", choices=["tsf", "eicu_sepsis3"], default="tsf")
    parser.add_argument("--tsf", default="")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--history-length", type=int, default=0)
    parser.add_argument("--forecast-horizon", type=int, default=0)
    parser.add_argument("--max-train-windows-per-series", type=int, default=24)
    parser.add_argument("--eicu-sepsis3-labels-csv", default=str(DEFAULT_EICU_LABELS_CSV))
    parser.add_argument("--eicu-sepsis3-trajectory-csv", default=str(DEFAULT_EICU_TRAJECTORY_CSV))
    parser.add_argument("--eicu-target-field", default="total_sofa")
    parser.add_argument("--eicu-max-series", type=int, default=0)
    parser.add_argument("--enable-kg", action="store_true")
    parser.add_argument("--kg-directory", default=str(DEFAULT_EICU_KG_DIR))
    parser.add_argument("--disable-kg-static-concat", action="store_true")
    parser.add_argument("--enable-explicit-kg-path", action="store_true")
    parser.add_argument("--kg-residual-weight", type=float, default=0.12)
    parser.add_argument("--kg-alignment-floor", type=float, default=0.05)

    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aux-base-loss-weight", type=float, default=0.3)
    parser.add_argument("--multitask-loss-weight", type=float, default=0.15)
    parser.add_argument("--align-loss-weight", type=float, default=0.02)
    parser.add_argument("--temporal-smoothness-weight", type=float, default=0.02)
    parser.add_argument("--memory-direct-residual-weight", type=float, default=0.1)
    parser.add_argument("--memory-direct-residual-mode", choices=["fixed", "adaptive"], default="fixed")
    parser.add_argument("--memory-path-coordination-mode", choices=["sum", "adaptive"], default="sum")
    parser.add_argument("--branch-decoupling-mode", choices=["pre_projection", "post_projection"], default="post_projection")
    parser.add_argument("--retrieval-projection-scale", type=float, default=0.30)
    parser.add_argument("--disable-branch-decoupling", action="store_true")
    parser.add_argument("--disable-memory-harm-control", action="store_true")
    parser.add_argument("--memory-quality-floor", type=float, default=0.18)
    parser.add_argument("--harm-stable-quality-boost", type=float, default=0.0)
    parser.add_argument("--harm-flat-quality-boost", type=float, default=0.0)
    parser.add_argument("--memory-min-path-alignment", type=float, default=-0.20)
    parser.add_argument("--memory-residual-cap-ratio", type=float, default=0.35)
    parser.add_argument("--memory-gain-audit-top-k", type=int, default=12)
    parser.add_argument(
        "--memory-gain-audit-max-cases",
        type=int,
        default=512,
        help="Maximum per-case memory gain rows written to output JSON. 0 writes all cases.",
    )
    parser.add_argument("--enable-epoch-feedback", action="store_true")
    parser.add_argument("--hard-example-weight", type=float, default=0.35)
    parser.add_argument("--kg-consistency-weight", type=float, default=0.06)
    parser.add_argument("--path-alignment-weight", type=float, default=0.05)
    parser.add_argument("--archive-retention-weight", type=float, default=0.04)
    parser.add_argument("--memory-delta-floor-weight", type=float, default=0.03)
    parser.add_argument("--archive-retention-target", type=float, default=0.10)
    parser.add_argument("--memory-delta-floor", type=float, default=0.05)
    parser.add_argument("--epoch-feedback-momentum", type=float, default=0.35)
    parser.add_argument("--feedback-top-error-rate", type=float, default=0.35)
    parser.add_argument("--error-attribution-max-samples", type=int, default=256)
    parser.add_argument("--checkpoint-selection-mode", choices=["best_val_mae", "final_epoch"], default="best_val_mae")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--sim-threshold", type=float, default=0.92)
    parser.add_argument("--merge-alpha", type=float, default=0.2)
    parser.add_argument("--decay", type=float, default=0.997)
    parser.add_argument("--forget-threshold", type=float, default=0.08)
    parser.add_argument("--max-memory", type=int, default=256)
    parser.add_argument("--memory-min-label", type=int, default=1)
    parser.add_argument("--memory-max-label", type=int, default=256)
    parser.add_argument("--memory-max-patient-label", type=int, default=3)
    parser.add_argument("--memory-support-penalty", type=float, default=0.02)
    parser.add_argument("--memory-collapse-penalty", type=float, default=0.03)
    parser.add_argument("--memory-temperature", type=float, default=0.15)
    parser.add_argument(
        "--memory-refresh-interval",
        type=int,
        default=0,
        help="Online memory bank refresh interval during training. 0 builds once before training and once after checkpoint selection; positive N rebuilds every N epochs.",
    )
    parser.add_argument("--encoder-type", choices=["gru", "transformer"], default="gru")
    parser.add_argument("--manifold-dim", type=int, default=32)
    parser.add_argument("--manifold-value-dim", type=int, default=48)
    parser.add_argument("--manifold-fusion-hidden-dim", type=int, default=64)
    parser.add_argument("--gru-hidden-dim", type=int, default=64)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument("--gru-dropout", type=float, default=0.1)
    parser.add_argument("--gru-bidirectional", action="store_true", default=True)
    parser.add_argument("--transformer-d-model", type=int, default=96)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-ff-dim", type=int, default=192)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument("--transformer-max-length", type=int, default=256)
    parser.add_argument("--static-hidden-dim", type=int, default=16)
    parser.add_argument("--persistent-memory-store", default="")
    parser.add_argument("--persistent-memory-scope", choices=["dataset", "all"], default="dataset")
    parser.add_argument("--persistent-memory-allowed-splits", default="train,external")
    parser.add_argument("--disable-persistent-memory-strict-no-test", action="store_true")
    parser.add_argument("--disable-persistent-memory-exclude-current-eval", action="store_true")
    parser.add_argument("--persistent-prototype-update-mode", choices=["incremental", "rebuild"], default="incremental")
    parser.add_argument("--build-persistent-memory-only", action="store_true")
    parser.add_argument("--persistent-memory-build-splits", default="train")
    parser.add_argument("--disable-persistent-memory-progress", action="store_true")
    parser.add_argument("--persistent-memory-progress-interval", type=int, default=1000)
    parser.add_argument("--persistent-semantic-top-k", type=int, default=3)
    parser.add_argument("--prime-persistent-memory-before-fit", action="store_true")
    parser.add_argument("--disable-persistent-memory-reuse", action="store_true")
    parser.add_argument("--disable-persistent-memory-export", action="store_true")
    parser.add_argument("--skip-posthoc-diagnostics", action="store_true")
    parser.add_argument("--counterfactual-store-predictions", action="store_true")
    parser.add_argument(
        "--counterfactual-donor-score-mode",
        choices=["legacy", "structured"],
        default="legacy",
        help="Counterfactual donor reranking mode. 'legacy' is the current default because it gave the best 512-stay balance between donor realism and improvement proxy.",
    )
    parser.add_argument("--counterfactual-base-similarity-weight", type=float, default=1.0)
    parser.add_argument("--counterfactual-kg-similarity-weight", type=float, default=0.25)
    parser.add_argument("--counterfactual-guideline-weight", type=float, default=0.20)
    parser.add_argument("--counterfactual-guideline-score-weight", type=float, default=0.05)
    parser.add_argument("--counterfactual-penalty-weight", type=float, default=0.20)
    parser.add_argument("--disable-counterfactual-hard-filter", action="store_true")
    parser.add_argument(
        "--counterfactual-candidate-policy",
        choices=["donor_only", "generated_best", "safe_search"],
        default="donor_only",
        help="Counterfactual intervention candidate mode. 'generated_best' evaluates donor-original and KG-repaired donor plans, while 'safe_search' adds a constrained template-search layer on top of the safety-repair layer.",
    )
    parser.add_argument("--counterfactual-candidate-feasibility-weight", type=float, default=0.15)
    parser.add_argument("--counterfactual-candidate-penalty-weight", type=float, default=0.10)
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
    parser.add_argument("--enable-transition-memory", action="store_true")
    parser.add_argument("--transition-top-k", type=int, default=6)
    parser.add_argument("--transition-state-weight", type=float, default=0.65)
    parser.add_argument("--transition-action-weight", type=float, default=0.35)
    parser.add_argument("--transition-score-weight", type=float, default=0.12)
    parser.add_argument("--transition-template-blend-weight", type=float, default=0.10)
    parser.add_argument("--transition-selection-weight", type=float, default=0.04)
    parser.add_argument("--transition-temperature", type=float, default=0.20)
    parser.add_argument("--transition-utility-scale", type=float, default=2.0)
    parser.add_argument("--transition-utility-alignment-weight", type=float, default=0.0)
    parser.add_argument("--transition-anchor-blend-weight", type=float, default=0.0)
    parser.add_argument("--transition-signature-match-weight", type=float, default=0.08)
    parser.add_argument("--transition-min-confidence", type=float, default=0.50)
    parser.add_argument("--transition-min-support", type=float, default=0.20)
    parser.add_argument("--transition-min-expected-utility", type=float, default=0.05)
    parser.add_argument("--transition-min-signature-weight", type=float, default=0.0)
    parser.add_argument("--transition-stable-regime-penalty", type=float, default=1.0)
    parser.add_argument("--transition-flat-pattern-penalty", type=float, default=1.0)
    parser.add_argument("--transition-utility-bias", type=float, default=-0.05)
    parser.add_argument("--transition-utility-temperature", type=float, default=0.08)
    parser.add_argument("--disable-transition-partial-signature", action="store_true")
    parser.add_argument("--transition-residual-cap-ratio", type=float, default=0.30)
    parser.add_argument("--transition-trunk-weight", type=float, default=0.0)
    parser.add_argument(
        "--transition-factual-residual-mode",
        choices=["additive", "delta_to_base", "delta_to_fusion_base"],
        default="additive",
    )
    parser.add_argument("--transition-positive-only", action="store_true")
    parser.add_argument("--transition-action-change-weight", type=float, default=0.05)
    parser.add_argument("--transition-candidate-action-change-weight", type=float, default=0.04)
    parser.add_argument("--disable-transition-trunk-path", action="store_true")
    parser.add_argument("--disable-transition-factual-path", action="store_true")
    parser.add_argument("--disable-transition-donor-path", action="store_true")

    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-inference-bundle", default="")
    parser.add_argument("--test-uncertainty-samples", type=int, default=8)
    parser.add_argument("--phase0-min-donor-similarity", type=float, default=0.45)
    parser.add_argument("--phase0-min-guideline-compatibility", type=float, default=0.25)
    parser.add_argument("--phase0-min-donor-total-score", type=float, default=0.0)
    parser.add_argument("--phase0-max-missing-care-penalty", type=float, default=0.55)
    parser.add_argument("--phase0-max-contraindication-penalty", type=float, default=0.25)
    parser.add_argument("--phase0-require-positive-delta", action="store_true")
    parser.add_argument("--uncertainty-delta-threshold", type=float, default=2.0,
                        help="Flag counterfactual deltas as uncertain if |delta| < prediction_std * threshold (0=disable)")
    args = parser.parse_args()
    if args.dataset_format == "tsf" and not args.tsf:
        parser.error("--tsf is required when --dataset-format tsf")
    return args


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device_arg


def _metric_improvements(hybrid: Dict[str, float], base_only: Dict[str, float]) -> Dict[str, float]:
    return {
        "improvement_mae": base_only["mae"] - hybrid["mae"],
        "improvement_rmse": base_only["rmse"] - hybrid["rmse"],
        "improvement_smape": base_only["smape"] - hybrid["smape"],
        "improvement_mase": base_only["mase"] - hybrid["mase"],
    }


def _resolved_path_str(path_str: str) -> str:
    return str(Path(path_str).expanduser().resolve())


def _ensure_parent_dir(path_str: str) -> None:
    if not path_str:
        return
    Path(path_str).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _dedupe_memory_seed_samples(samples):
    deduped = []
    seen_ids = set()
    for sample in samples:
        sample_id = forecast_sample_identity(sample)
        if sample_id in seen_ids:
            continue
        seen_ids.add(sample_id)
        deduped.append(sample)
    return deduped


def _sample_vector_dim(sample: object, attr_name: str, fallback_attr: str = "") -> int:
    values = getattr(sample, attr_name, None)
    if not values and fallback_attr:
        values = getattr(sample, fallback_attr, None)
    return len(values or [])


def _sample_sequence_step_dim(sample: object, attr_name: str) -> int:
    sequence = getattr(sample, attr_name, None) or []
    if not sequence:
        return 0
    first_step = sequence[0] if sequence else []
    return len(first_step or [])


def _filter_memory_seed_samples_for_schema(
    samples: Sequence[object],
    expected_patient_dim: int,
    expected_intervention_dim: int,
    expected_intervention_sequence_dim: int,
) -> Tuple[List[object], Dict[str, object]]:
    kept: List[object] = []
    audit: Dict[str, object] = {
        "enabled": True,
        "input_total": len(samples),
        "kept_total": 0,
        "excluded_total": 0,
        "expected_patient_dim": int(expected_patient_dim),
        "expected_intervention_dim": int(expected_intervention_dim),
        "expected_intervention_sequence_dim": int(expected_intervention_sequence_dim),
        "excluded_patient_static_dim": 0,
        "excluded_intervention_static_dim": 0,
        "excluded_intervention_sequence_dim": 0,
        "excluded_persistent_source": 0,
        "excluded_runtime_source": 0,
        "examples": [],
    }
    for sample in samples:
        metadata = dict(getattr(sample, "metadata", {}) or {})
        patient_dim = _sample_vector_dim(sample, "patient_static", "static")
        intervention_dim = _sample_vector_dim(sample, "intervention_static")
        intervention_sequence_dim = _sample_sequence_step_dim(sample, "intervention_sequence")
        reasons: List[str] = []
        if int(patient_dim) != int(expected_patient_dim):
            reasons.append("patient_static_dim")
            audit["excluded_patient_static_dim"] = int(audit["excluded_patient_static_dim"]) + 1
        if int(intervention_dim) != int(expected_intervention_dim):
            reasons.append("intervention_static_dim")
            audit["excluded_intervention_static_dim"] = int(audit["excluded_intervention_static_dim"]) + 1
        if int(expected_intervention_sequence_dim) > 0 and int(intervention_sequence_dim) not in {0, int(expected_intervention_sequence_dim)}:
            reasons.append("intervention_sequence_dim")
            audit["excluded_intervention_sequence_dim"] = int(audit["excluded_intervention_sequence_dim"]) + 1
        if reasons:
            if bool(metadata.get("persistent_source", False)):
                audit["excluded_persistent_source"] = int(audit["excluded_persistent_source"]) + 1
            else:
                audit["excluded_runtime_source"] = int(audit["excluded_runtime_source"]) + 1
            examples = list(audit["examples"])
            if len(examples) < 8:
                examples.append(
                    {
                        "series_name": str(metadata.get("series_name", "")),
                        "experience_id": str(metadata.get("experience_id", "")),
                        "persistent_source": bool(metadata.get("persistent_source", False)),
                        "reasons": reasons,
                        "patient_dim": int(patient_dim),
                        "intervention_dim": int(intervention_dim),
                        "intervention_sequence_dim": int(intervention_sequence_dim),
                    }
                )
                audit["examples"] = examples
            continue
        kept.append(sample)
    audit["kept_total"] = len(kept)
    audit["excluded_total"] = int(audit["input_total"]) - len(kept)
    return kept, audit


def _csv_values(value: str) -> List[str]:
    return [item.strip().lower() for item in str(value or "").split(",") if item.strip()]


def _metadata_identifier(metadata: Dict[str, object], keys: Sequence[str]) -> object:
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "null"}:
            return value
    return None


def _persistent_exclusion_ids(samples: Sequence[object]) -> Tuple[List[object], List[object]]:
    stay_ids: List[object] = []
    patient_ids: List[object] = []
    for sample in samples:
        metadata = dict(getattr(sample, "metadata", {}) or {})
        stay_id = _metadata_identifier(
            metadata,
            ["stay_id", "patientunitstayid", "source_stay_id", "source_stay_id_hash"],
        )
        patient_id = _metadata_identifier(
            metadata,
            ["patient_id", "uniquepid", "patienthealthsystemstayid", "source_patient_id_hash"],
        )
        if stay_id is not None:
            stay_ids.append(stay_id)
        if patient_id is not None:
            patient_ids.append(patient_id)
    return stay_ids, patient_ids


def _samples_for_persistent_build(dataset, split_names: Sequence[str]) -> Dict[str, Sequence[object]]:
    available = {
        "train": dataset.train_samples,
        "val": dataset.val_samples,
        "validation": dataset.val_samples,
        "test": dataset.test_samples,
    }
    selected: Dict[str, Sequence[object]] = {}
    for split_name in split_names:
        key = str(split_name).strip().lower()
        if not key:
            continue
        if key == "all":
            selected["train"] = dataset.train_samples
            selected["val"] = dataset.val_samples
            selected["test"] = dataset.test_samples
            continue
        if key not in available:
            raise ValueError(f"Unsupported persistent memory build split: {split_name}")
        normalized_key = "val" if key == "validation" else key
        selected[normalized_key] = available[key]
    return selected


def _log_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _format_progress_bar(processed: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    ratio = min(1.0, max(0.0, float(processed) / float(total)))
    filled = int(round(width * ratio))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _make_persistent_progress_callback(enabled: bool, split_name: str) -> Optional[Callable[[Dict[str, object]], None]]:
    if not enabled:
        return None
    start_time = time.time()

    def callback(event: Dict[str, object]) -> None:
        stage = str(event.get("stage", ""))
        source = str(event.get("source", split_name))
        elapsed = max(0.0, time.time() - start_time)
        prefix = f"[persistent-memory][{source}]"
        if stage == "scan_start":
            _log_progress(
                f"{prefix} scanning entries: total={int(event.get('total', 0))}, "
                f"existing={int(event.get('existing_entry_count', 0))}"
            )
            return
        if stage == "scan":
            processed = int(event.get("processed", 0))
            total = int(event.get("total", 0))
            pct = (100.0 * processed / total) if total else 100.0
            _log_progress(
                f"{prefix} {_format_progress_bar(processed, total)} "
                f"{processed}/{total} ({pct:5.1f}%) "
                f"inserted={int(event.get('inserted', 0))} skipped={int(event.get('skipped', 0))} "
                f"elapsed={elapsed:0.1f}s"
            )
            return
        if stage == "append_start":
            _log_progress(f"{prefix} appending {int(event.get('inserted', 0))} new entries to jsonl")
            return
        if stage == "append_done":
            _log_progress(f"{prefix} jsonl append complete")
            return
        if stage == "prototype_start":
            _log_progress(
                f"{prefix} updating prototypes: mode={event.get('mode', '')}, "
                f"new_rows={int(event.get('new_rows', 0))}"
            )
            return
        if stage == "prototype_done":
            _log_progress(
                f"{prefix} prototype update complete: count={int(event.get('prototype_count', 0))}, "
                f"rebuilt={bool(event.get('prototype_rebuilt', False))}"
            )
            return
        if stage == "done":
            _log_progress(
                f"{prefix} done: inserted={int(event.get('inserted_count', 0))}, "
                f"skipped={int(event.get('skipped_count', 0))}, "
                f"entries={int(event.get('entry_count', 0))}, "
                f"prototypes={int(event.get('prototype_count', 0))}, elapsed={elapsed:0.1f}s"
            )

    return callback


def _training_budget_summary(dataset, args) -> Dict[str, float]:
    train_windows = len(dataset.train_samples)
    batch_size = max(1, int(args.batch_size))
    steps_per_epoch = (train_windows + batch_size - 1) // batch_size
    return {
        "train_windows": float(train_windows),
        "val_windows": float(len(dataset.val_samples)),
        "test_windows": float(len(dataset.test_samples)),
        "batch_size": float(batch_size),
        "epochs": float(args.epochs),
        "steps_per_epoch": float(steps_per_epoch),
        "total_optimizer_steps": float(steps_per_epoch * max(1, int(args.epochs))),
    }


def _factual_metric_verdict(improvement: Dict[str, float]) -> Dict[str, object]:
    metric_signs = {
        "mae": improvement["improvement_mae"] > 0.0,
        "rmse": improvement["improvement_rmse"] > 0.0,
        "smape": improvement["improvement_smape"] > 0.0,
    }
    positive_count = sum(1 for is_positive in metric_signs.values() if is_positive)
    if positive_count == len(metric_signs):
        overall = "positive"
    elif positive_count == 0:
        overall = "negative"
    else:
        overall = "mixed"
    return {
        "overall": overall,
        "positive_metric_count": float(positive_count),
        "metric_signs": metric_signs,
    }


def _build_factual_forecasting_line(
    hybrid_metrics: Dict[str, float],
    base_metrics: Dict[str, float],
    trainer,
    dataset,
    args,
    example_trace: Dict[str, object],
    factual_path_audit: Dict[str, object],
    memory_gain_audit: Dict[str, object],
    transition_gate_audit: Dict[str, object],
    error_attribution: Dict[str, object],
) -> Dict[str, object]:
    improvement = _metric_improvements(hybrid_metrics, base_metrics)
    direct_memory = dict(trainer.memory_diagnostics.get("direct_memory_means", {}))
    retrieval_mix = dict(trainer.memory_diagnostics.get("experience_retrieval_source_rate", {}))
    semantic_retrieval = dict(trainer.memory_diagnostics.get("semantic_retrieval", {}))
    return {
        "enabled": True,
        "question": "Does memory improve held-out factual forecasting?",
        "scope": "Compares trainer.predict(..., use_memory=True/False) on the same test set. Counterfactual donor reranking does not enter this line.",
        "purity_contract": {
            "uses_memory_enabled_predictions": True,
            "uses_memory_disabled_baseline": True,
            "uses_counterfactual_donor_selection": False,
            "uses_clinical_plausibility_reranking": False,
            "counterfactual_controls_change_this_line": False,
        },
        "training_budget": _training_budget_summary(dataset, args),
        "metrics": {
            "memory_enabled": hybrid_metrics,
            "memory_disabled": base_metrics,
        },
        "improvement": improvement,
        "verdict": _factual_metric_verdict(improvement),
        "observability": {
            "posthoc_diagnostics_collected": not bool(args.skip_posthoc_diagnostics),
            "memory_diagnostics_available": bool(trainer.memory_diagnostics),
            "example_trace_available": bool(example_trace),
            "factual_path_audit_available": bool(factual_path_audit),
            "error_attribution_available": bool(error_attribution),
        },
        "memory_path_audit": {
            "direct_memory_means": direct_memory,
            "experience_retrieval_source_rate": retrieval_mix,
            "semantic_retrieval": semantic_retrieval,
            "factual_path_audit": factual_path_audit,
            "memory_gain_audit": memory_gain_audit,
            "transition_gate_audit": transition_gate_audit,
            "error_attribution": error_attribution,
        },
    }


def _build_counterfactual_donor_ranking_line(counterfactual_summary: Dict[str, object]) -> Dict[str, object]:
    if not counterfactual_summary:
        return {
            "enabled": False,
            "question": "How does donor retrieval and donor candidate selection behave?",
            "scope": "Not available because this run did not execute eICU counterfactual evaluation.",
        }
    return {
        "enabled": True,
        "question": "How does donor retrieval and donor candidate selection behave?",
        "scope": "Summarizes donor retrieval, donor scoring, and candidate selection. These fields do not prove factual forecasting gains.",
        "purity_contract": {
            "affects_factual_forecasting_metrics": False,
            "uses_model_self_prediction_proxy": True,
            "uses_ground_truth_counterfactual_outcomes": False,
        },
        "selection_quality": {
            "donor_found_rate": _safe_float(counterfactual_summary.get("donor_found_rate")),
            "donor_exact_experience_match_rate": _safe_float(counterfactual_summary.get("donor_exact_experience_match_rate")),
            "donor_total_score_mean": _safe_float(counterfactual_summary.get("donor_total_score_mean")),
            "donor_overlap_valid_rate": _safe_float(counterfactual_summary.get("donor_overlap_valid_rate")),
            "donor_overlap_fallback_rate": _safe_float(counterfactual_summary.get("donor_overlap_fallback_rate")),
            "predicted_improvement_rate": _safe_float(counterfactual_summary.get("predicted_improvement_rate")),
            "mean_predicted_delta": _safe_float(counterfactual_summary.get("mean_predicted_delta")),
            "generated_candidate_available_rate": _safe_float(counterfactual_summary.get("generated_candidate_available_rate")),
            "generated_candidate_selected_rate": _safe_float(counterfactual_summary.get("generated_candidate_selected_rate")),
            "selected_candidate_source_counts": dict(counterfactual_summary.get("selected_candidate_source_counts", {})),
            "donor_overlap_reason_counts": dict(counterfactual_summary.get("donor_overlap_reason_counts", {})),
        },
        "retrieval_profile": {
            "donor_similarity_mean": _safe_float(counterfactual_summary.get("donor_similarity_mean")),
            "donor_kg_similarity_mean": _safe_float(counterfactual_summary.get("donor_kg_similarity_mean")),
            "donor_overlap_score_mean": _safe_float(counterfactual_summary.get("donor_overlap_score_mean")),
            "donor_learned_reranker_score_mean": _safe_float(counterfactual_summary.get("donor_learned_reranker_score_mean")),
            "donor_reranker_adjustment_mean": _safe_float(counterfactual_summary.get("donor_reranker_adjustment_mean")),
            "donor_score_mode": str(counterfactual_summary.get("donor_score_mode", "")),
            "donor_reranker_mode": str(counterfactual_summary.get("donor_reranker_mode", "")),
            "counterfactual_candidate_policy": str(counterfactual_summary.get("counterfactual_candidate_policy", "")),
        },
        "transition_signal": {
            "donor_transition_score_mean": _safe_float(counterfactual_summary.get("donor_transition_score_mean")),
            "donor_transition_confidence_mean": _safe_float(counterfactual_summary.get("donor_transition_confidence_mean")),
            "donor_transition_improvement_rate_mean": _safe_float(counterfactual_summary.get("donor_transition_improvement_rate_mean")),
            "donor_action_change_score_mean": _safe_float(counterfactual_summary.get("donor_action_change_score_mean")),
        },
    }


def _build_clinical_plausibility_line(counterfactual_summary: Dict[str, object]) -> Dict[str, object]:
    if not counterfactual_summary:
        return {
            "enabled": False,
            "question": "Are the selected donor plans clinically plausible?",
            "scope": "Not available because this run did not execute eICU counterfactual evaluation.",
        }
    return {
        "enabled": True,
        "question": "Are the selected donor plans clinically plausible?",
        "scope": "Uses KG/state compatibility and penalty summaries only. This line evaluates plausibility, not causal benefit.",
        "purity_contract": {
            "affects_factual_forecasting_metrics": False,
            "uses_ground_truth_counterfactual_outcomes": False,
            "should_not_be_read_as_factual_gain": True,
        },
        "plausibility_metrics": {
            "donor_guideline_compatibility_mean": _safe_float(counterfactual_summary.get("donor_guideline_compatibility_mean")),
            "donor_state_match_mean": _safe_float(counterfactual_summary.get("donor_state_match_mean")),
            "donor_overlap_score_mean": _safe_float(counterfactual_summary.get("donor_overlap_score_mean")),
            "donor_overlap_valid_rate": _safe_float(counterfactual_summary.get("donor_overlap_valid_rate")),
            "donor_overlap_fallback_rate": _safe_float(counterfactual_summary.get("donor_overlap_fallback_rate")),
            "donor_learned_reranker_score_mean": _safe_float(counterfactual_summary.get("donor_learned_reranker_score_mean")),
            "donor_missing_care_penalty_mean": _safe_float(counterfactual_summary.get("donor_missing_care_penalty_mean")),
            "donor_contraindication_penalty_mean": _safe_float(counterfactual_summary.get("donor_contraindication_penalty_mean")),
            "donor_hard_filter_fallback_rate": _safe_float(counterfactual_summary.get("donor_hard_filter_fallback_rate")),
            "donor_overlap_reason_counts": dict(counterfactual_summary.get("donor_overlap_reason_counts", {})),
        },
        "directionality": {
            "higher_is_better": [
                "donor_guideline_compatibility_mean",
                "donor_state_match_mean",
                "donor_overlap_score_mean",
                "donor_overlap_valid_rate",
            ],
            "lower_is_better": [
                "donor_overlap_fallback_rate",
                "donor_missing_care_penalty_mean",
                "donor_contraindication_penalty_mean",
                "donor_hard_filter_fallback_rate",
            ],
        },
    }


def _build_diagnostic_summary(
    memory_gain_audit: Dict[str, object],
    val_memory_gain_audit: Dict[str, object],
    factual_path_audit: Dict[str, object],
    counterfactual_summary: Dict[str, object],
    uncertainty_analysis: Dict[str, object],
    memory_diagnostics: Dict[str, object],
    transition_gate_audit: Dict[str, object],
) -> Dict[str, object]:
    gain_audit = memory_gain_audit if memory_gain_audit else {}
    val_audit = val_memory_gain_audit if val_memory_gain_audit else {}

    def _find_worst_subgroup(slices: list, metric: str = "harmed_rate") -> tuple[str, float]:
        worst_name, worst_val = "", 0.0
        for row in slices:
            val = float(row.get(metric, 0))
            if val > worst_val:
                worst_val = val
                worst_name = str(row.get("group", ""))
        return worst_name, worst_val

    # Factual layer verdict
    test_helped = float(gain_audit.get("helped_rate", 0)) if gain_audit else 0.0
    test_harmed = float(gain_audit.get("harmed_rate", 0)) if gain_audit else 0.0
    test_gain = float(gain_audit.get("mean_memory_gain_mae", 0)) if gain_audit else 0.0
    val_helped = float(val_audit.get("helped_rate", 0)) if val_audit else 0.0
    val_harmed = float(val_audit.get("harmed_rate", 0)) if val_audit else 0.0
    overfit_gap = abs(test_gain - float(val_audit.get("mean_memory_gain_mae", 0))) if val_audit else 0.0

    traj_slices = list(gain_audit.get("subgroup_slices", {}).get("by_trajectory", [])) if gain_audit else []
    pat_slices = list(gain_audit.get("subgroup_slices", {}).get("by_pattern", [])) if gain_audit else []
    level_slices = list(gain_audit.get("subgroup_slices", {}).get("by_baseline_level", [])) if gain_audit else []

    worst_traj, worst_traj_h = _find_worst_subgroup(traj_slices)
    worst_pat, worst_pat_h = _find_worst_subgroup(pat_slices)
    worst_level, worst_level_h = _find_worst_subgroup(level_slices)

    if test_gain > 0.02 and test_helped > 0.55 and test_harmed < 0.35:
        factual_verdict = "strong_positive"
    elif test_gain > 0.005:
        factual_verdict = "stable_positive"
    elif test_gain > -0.005:
        factual_verdict = "neutral"
    else:
        factual_verdict = "negative"

    # Retrieval layer verdict
    fpa = factual_path_audit if factual_path_audit else {}
    sem = fpa.get("semantic_retrieval", {})
    sem_hit_rate = float(sem.get("hit_rate", 0))
    sem_mean_blend = float(sem.get("mean_template_blend_weight", 0))
    trans_blocked_rate = float(fpa.get("transition_means", {}).get("transition_gate_blocked", 0))
    tga = transition_gate_audit if transition_gate_audit else {}
    blocked_reasons = dict(tga.get("blocked_reason_distribution", {}))

    retrieval_verdict = "healthy" if sem_hit_rate > 0.5 and sem_mean_blend > 0.15 else "degraded"

    # Candidate layer verdict
    cf = counterfactual_summary if counterfactual_summary else {}
    donor_found = float(cf.get("donor_found_rate", 0))
    candidate_avail = float(cf.get("generated_candidate_available_rate", 0))
    improvement_rate = float(cf.get("predicted_improvement_rate", 0))

    if donor_found > 0.5 and candidate_avail > 0.5:
        candidate_verdict = "healthy"
    elif donor_found > 0.2:
        candidate_verdict = "partial"
    else:
        candidate_verdict = "limited"

    # Uncertainty layer
    uncertainty_enabled = bool(uncertainty_analysis.get("enabled", False)) if uncertainty_analysis else False
    forecast_mean_std = float(uncertainty_analysis.get("forecast_mean_std", 0)) if uncertainty_analysis else 0.0

    # Bottleneck ranking
    bottlenecks: List[Dict[str, object]] = []
    if test_harmed > 0.35:
        bottlenecks.append({"layer": "factual_memory", "severity": "high", "evidence": f"harmed_rate={test_harmed:.2f}, worst_subgroup={worst_traj}({worst_traj_h:.2f})"})
    if sem_hit_rate < 0.5 or sem_mean_blend < 0.15:
        bottlenecks.append({"layer": "retrieval", "severity": "medium", "evidence": f"sem_hit={sem_hit_rate:.2f}, mean_blend={sem_mean_blend:.3f}"})
    if trans_blocked_rate > 0.5:
        top_reason = max(blocked_reasons, key=blocked_reasons.get) if blocked_reasons else "unknown"
        bottlenecks.append({"layer": "transition_gate", "severity": "medium", "evidence": f"blocked_rate={trans_blocked_rate:.2f}, top_reason={top_reason}"})
    if donor_found < 0.5:
        bottlenecks.append({"layer": "donor_retrieval", "severity": "medium", "evidence": f"donor_found={donor_found:.2f}"})
    if overfit_gap > 0.02:
        bottlenecks.append({"layer": "generalization", "severity": "medium", "evidence": f"test_gain={test_gain:.4f}, val_gain={val_audit.get('mean_memory_gain_mae', 0):.4f}"})
    bottlenecks.sort(key=lambda b: {"high": 0, "medium": 1, "low": 2}.get(str(b.get("severity", "")), 3))

    return {
        "generated": True,
        "factual_layer": {
            "verdict": factual_verdict,
            "test_helped_rate": test_helped,
            "test_harmed_rate": test_harmed,
            "test_mean_gain": test_gain,
            "val_helped_rate": val_helped,
            "val_harmed_rate": val_harmed,
            "overfit_gap": overfit_gap,
            "worst_subgroups": {
                "by_trajectory": {"name": worst_traj, "harmed_rate": worst_traj_h},
                "by_pattern": {"name": worst_pat, "harmed_rate": worst_pat_h},
                "by_baseline_level": {"name": worst_level, "harmed_rate": worst_level_h},
            },
        },
        "retrieval_layer": {
            "verdict": retrieval_verdict,
            "semantic_hit_rate": sem_hit_rate,
            "mean_template_blend_weight": sem_mean_blend,
            "transition_gate_blocked_rate": trans_blocked_rate,
            "top_blocked_reason": max(blocked_reasons, key=blocked_reasons.get) if blocked_reasons else "none",
        },
        "candidate_layer": {
            "verdict": candidate_verdict,
            "donor_found_rate": donor_found,
            "candidate_available_rate": candidate_avail,
            "predicted_improvement_rate": improvement_rate,
        },
        "uncertainty_layer": {
            "enabled": uncertainty_enabled,
            "forecast_mean_std": forecast_mean_std,
        },
        "bottleneck_ranking": bottlenecks,
    }


def _build_memory_usefulness_assessment(
    factual_line: Dict[str, object],
    donor_line: Dict[str, object],
    clinical_line: Dict[str, object],
    args,
) -> Dict[str, object]:
    training_budget = factual_line["training_budget"]
    warnings = []
    if bool(args.skip_posthoc_diagnostics):
        warnings.append("skip_posthoc_diagnostics=true；本次 run 无法审计记忆路径使用强度与检索细节。")
    if training_budget["total_optimizer_steps"] < 20.0:
        warnings.append("总优化步数过少；当前 run 更适合做 smoke/mid-scale 对比，不足以稳定回答记忆是否有效。")
    if donor_line.get("enabled"):
        warnings.append("counterfactual donor ranking 与 clinical plausibility 只提供辅助证据，不能替代 factual forecasting 结论。")
    factual_audit = factual_line.get("memory_path_audit", {}).get("factual_path_audit", {})
    mean_memory_delta = _safe_float(
        factual_audit.get("transition_means", {}).get("base_to_fusion_delta_strength"),
        0.0,
    )
    if factual_audit and mean_memory_delta <= 1e-4:
        warnings.append("factual path audit 显示 base 与 fusion 的平均差异极小；即使结果有波动，也不能说明记忆路径被有效使用。")

    factual_verdict = str(factual_line["verdict"]["overall"])
    if factual_verdict == "positive":
        overall_answer = "supports_memory_for_factual_forecasting"
    elif factual_verdict == "negative":
        overall_answer = "does_not_support_memory_for_factual_forecasting"
    else:
        overall_answer = "mixed_or_inconclusive_for_factual_forecasting"

    answerability = "limited" if warnings else "good"
    return {
        "primary_question": "Does memory help?",
        "primary_evidence_line": "factual_forecasting",
        "overall_answer": overall_answer,
        "factual_forecasting_verdict": factual_verdict,
        "mechanistic_answerability": answerability,
        "counterfactual_is_secondary_evidence": bool(donor_line.get("enabled")),
        "clinical_plausibility_is_secondary_evidence": bool(clinical_line.get("enabled")),
        "warnings": warnings,
    }


def _mean(values: Sequence[float]) -> float:
    numeric = [float(value) for value in values]
    if not numeric:
        return 0.0
    return float(sum(numeric) / max(1, len(numeric)))


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = max(0.0, min(1.0, float(quantile))) * float(len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - float(lower)
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _distribution_summary(values: Sequence[float]) -> Dict[str, float]:
    numeric = [float(value) for value in values]
    if not numeric:
        return {
            "count": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "max": 0.0,
            "positive_rate": 0.0,
            "nonzero_rate": 0.0,
        }
    sorted_values = sorted(numeric)
    mean_value = _mean(numeric)
    variance = sum((value - mean_value) ** 2 for value in numeric) / max(1, len(numeric))
    return {
        "count": float(len(numeric)),
        "mean": float(mean_value),
        "std": float(variance ** 0.5),
        "min": float(sorted_values[0]),
        "p25": float(_percentile(sorted_values, 0.25)),
        "median": float(_percentile(sorted_values, 0.50)),
        "p75": float(_percentile(sorted_values, 0.75)),
        "max": float(sorted_values[-1]),
        "positive_rate": float(sum(1 for value in numeric if value > 0.0) / max(1, len(numeric))),
        "nonzero_rate": float(sum(1 for value in numeric if abs(value) > 1e-8) / max(1, len(numeric))),
    }


def _point_regression_metrics(truth_values: Sequence[float], pred_values: Sequence[float]) -> Dict[str, float]:
    paired = list(zip(truth_values, pred_values))
    if not paired:
        return {
            "point_count": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "mean_bias": 0.0,
        }
    errors = [float(pred) - float(truth) for truth, pred in paired]
    abs_errors = [abs(error) for error in errors]
    mse = sum(error ** 2 for error in errors) / max(1, len(errors))
    return {
        "point_count": float(len(errors)),
        "mae": float(sum(abs_errors) / max(1, len(abs_errors))),
        "rmse": float(mse ** 0.5),
        "mean_bias": float(sum(errors) / max(1, len(errors))),
    }


def _grouped_factual_slice(
    samples,
    truth: Sequence[Sequence[float]],
    hybrid_predictions: Sequence[Sequence[float]],
    base_predictions: Sequence[Sequence[float]],
    group_mode: str,
) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = {}
    for sample, truth_row, hybrid_row, base_row in zip(samples, truth, hybrid_predictions, base_predictions):
        if group_mode == "pattern":
            group_key = str(PATTERN_LABELS[int(sample.pattern_label)])
        elif group_mode == "trajectory":
            group_key = str(TRAJECTORY_LABELS[int(sample.trajectory_label)])
        else:
            group_key = f"{PATTERN_LABELS[int(sample.pattern_label)]}|{TRAJECTORY_LABELS[int(sample.trajectory_label)]}"
        bucket = grouped.setdefault(
            group_key,
            {
                "truth": [],
                "hybrid": [],
                "base": [],
                "series_count": 0,
            },
        )
        bucket["truth"].extend(float(value) for value in truth_row)
        bucket["hybrid"].extend(float(value) for value in hybrid_row)
        bucket["base"].extend(float(value) for value in base_row)
        bucket["series_count"] = int(bucket["series_count"]) + 1

    rows: List[Dict[str, object]] = []
    for group_key in sorted(grouped):
        bucket = grouped[group_key]
        hybrid_metrics = _point_regression_metrics(bucket["truth"], bucket["hybrid"])
        base_metrics = _point_regression_metrics(bucket["truth"], bucket["base"])
        rows.append(
            {
                "group": group_key,
                "series_count": float(bucket["series_count"]),
                "point_count": float(hybrid_metrics["point_count"]),
                "memory_enabled_mae": float(hybrid_metrics["mae"]),
                "memory_disabled_mae": float(base_metrics["mae"]),
                "memory_enabled_rmse": float(hybrid_metrics["rmse"]),
                "memory_disabled_rmse": float(base_metrics["rmse"]),
                "memory_enabled_mean_bias": float(hybrid_metrics["mean_bias"]),
                "memory_disabled_mean_bias": float(base_metrics["mean_bias"]),
                "improvement_mae": float(base_metrics["mae"] - hybrid_metrics["mae"]),
                "improvement_rmse": float(base_metrics["rmse"] - hybrid_metrics["rmse"]),
            }
        )
    return rows


def _safe_label(labels: Sequence[str], label_index: object, default: str = "unknown") -> str:
    try:
        index = int(float(label_index))
    except (TypeError, ValueError):
        return default
    if 0 <= index < len(labels):
        return str(labels[index])
    return default


def _baseline_level_bin(value: float) -> str:
    if value < 4.0:
        return "level_lt_4"
    if value < 8.0:
        return "level_4_to_8"
    if value < 12.0:
        return "level_8_to_12"
    return "level_ge_12"


def _case_error_summary(truth_row: Sequence[float], pred_row: Sequence[float]) -> Dict[str, float]:
    metrics = _point_regression_metrics(truth_row, pred_row)
    return {
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "mean_bias": float(metrics["mean_bias"]),
    }


def _memory_gain_group_summary(records: Sequence[Dict[str, object]], group_key: str) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record.get(group_key, "unknown")), []).append(record)

    rows: List[Dict[str, object]] = []
    for key in sorted(grouped):
        bucket = grouped[key]
        gains = [float(row.get("memory_gain_mae", 0.0)) for row in bucket]
        base_mae = [float(row.get("base_mae", 0.0)) for row in bucket]
        memory_mae = [float(row.get("memory_enabled_mae", 0.0)) for row in bucket]
        delta_strength = [float(row.get("memory_delta_strength", 0.0)) for row in bucket]
        rows.append(
            {
                "group": key,
                "case_count": float(len(bucket)),
                "helped_rate": float(sum(1 for value in gains if value > 0.0) / max(1, len(gains))),
                "harmed_rate": float(sum(1 for value in gains if value < 0.0) / max(1, len(gains))),
                "mean_memory_gain_mae": float(sum(gains) / max(1, len(gains))),
                "memory_gain_mae_distribution": _distribution_summary(gains),
                "mean_base_mae": float(sum(base_mae) / max(1, len(base_mae))),
                "mean_memory_enabled_mae": float(sum(memory_mae) / max(1, len(memory_mae))),
                "mean_memory_delta_strength": float(sum(delta_strength) / max(1, len(delta_strength))),
            }
        )
    return rows


def _build_memory_gain_audit(
    samples,
    truth: Sequence[Sequence[float]],
    hybrid_predictions: Sequence[Sequence[float]],
    base_predictions: Sequence[Sequence[float]],
    max_cases: int = 512,
    top_k: int = 12,
) -> Dict[str, object]:
    records: List[Dict[str, object]] = []
    for index, (sample, truth_row, hybrid_row, base_row) in enumerate(
        zip(samples, truth, hybrid_predictions, base_predictions)
    ):
        base_summary = _case_error_summary(truth_row, base_row)
        memory_summary = _case_error_summary(truth_row, hybrid_row)
        memory_delta_values = [
            abs(float(memory_value) - float(base_value))
            for memory_value, base_value in zip(hybrid_row, base_row)
        ]
        metadata = dict(getattr(sample, "metadata", {}) or {})
        baseline_level = float(sample.raw_context[-1]) if getattr(sample, "raw_context", None) else float(truth_row[0])
        pattern_label = _safe_label(PATTERN_LABELS, getattr(sample, "pattern_label", -1))
        trajectory_label = _safe_label(TRAJECTORY_LABELS, getattr(sample, "trajectory_label", -1))
        record = {
            "case_index": float(index),
            "series_name": str(metadata.get("series_name", f"case_{index}")),
            "stay_id": _safe_float(metadata.get("stay_id"), -1.0),
            "patient_id": _safe_float(metadata.get("patient_id", metadata.get("uniquepid", -1.0)), -1.0),
            "pattern": pattern_label,
            "trajectory": trajectory_label,
            "pattern_trajectory": f"{pattern_label}|{trajectory_label}",
            "experience_label": float(getattr(sample, "experience_label", -1)),
            "baseline_level": float(baseline_level),
            "baseline_level_bin": _baseline_level_bin(float(baseline_level)),
            "base_mae": float(base_summary["mae"]),
            "memory_enabled_mae": float(memory_summary["mae"]),
            "memory_gain_mae": float(base_summary["mae"] - memory_summary["mae"]),
            "base_rmse": float(base_summary["rmse"]),
            "memory_enabled_rmse": float(memory_summary["rmse"]),
            "memory_gain_rmse": float(base_summary["rmse"] - memory_summary["rmse"]),
            "base_mean_bias": float(base_summary["mean_bias"]),
            "memory_enabled_mean_bias": float(memory_summary["mean_bias"]),
            "memory_delta_strength": float(sum(memory_delta_values) / max(1, len(memory_delta_values))),
        }
        records.append(record)

    gains = [float(record["memory_gain_mae"]) for record in records]
    deltas = [float(record["memory_delta_strength"]) for record in records]
    helped_count = sum(1 for value in gains if value > 0.0)
    harmed_count = sum(1 for value in gains if value < 0.0)
    unchanged_count = len(gains) - helped_count - harmed_count
    top_n = max(0, int(top_k))
    case_limit = int(max_cases)
    if case_limit > 0:
        case_details = records[:case_limit]
    else:
        case_details = records

    return {
        "enabled": True,
        "sample_count": float(len(records)),
        "case_detail_count": float(len(case_details)),
        "case_detail_limit": float(case_limit),
        "outcome_counts": {
            "helped": float(helped_count),
            "harmed": float(harmed_count),
            "unchanged": float(unchanged_count),
        },
        "helped_rate": float(helped_count / max(1, len(records))),
        "harmed_rate": float(harmed_count / max(1, len(records))),
        "mean_memory_gain_mae": float(sum(gains) / max(1, len(gains))),
        "memory_gain_mae_distribution": _distribution_summary(gains),
        "memory_delta_strength_distribution": _distribution_summary(deltas),
        "top_helped_cases": sorted(records, key=lambda row: float(row["memory_gain_mae"]), reverse=True)[:top_n],
        "top_harmed_cases": sorted(records, key=lambda row: float(row["memory_gain_mae"]))[:top_n],
        "subgroup_slices": {
            "by_pattern": _memory_gain_group_summary(records, "pattern"),
            "by_trajectory": _memory_gain_group_summary(records, "trajectory"),
            "by_pattern_trajectory": _memory_gain_group_summary(records, "pattern_trajectory"),
            "by_baseline_level": _memory_gain_group_summary(records, "baseline_level_bin"),
        },
        "case_details": case_details,
    }


def _build_transition_gate_audit(
    factual_path_audit: Dict[str, object],
    memory_gain_audit: Dict[str, object],
) -> Dict[str, object]:
    transition_cases = list(factual_path_audit.get("transition_case_details", []))
    gain_cases = list(memory_gain_audit.get("case_details", []))
    transition_enabled = bool(factual_path_audit.get("transition_means", {}).get("transition_gate_blocked", -1.0) >= 0.0)
    if not transition_cases or not transition_enabled:
        return {
            "enabled": False,
            "sample_count": float(len(transition_cases)),
            "reason": "transition_disabled_or_no_case_details",
        }

    gain_by_stay: Dict[str, float] = {}
    gain_by_index: Dict[int, float] = {}
    for record in gain_cases:
        stay_id = str(record.get("stay_id", ""))
        gain = float(record.get("memory_gain_mae", 0.0))
        if stay_id:
            gain_by_stay[stay_id] = gain
        gain_by_index[int(record.get("case_index", -1))] = gain

    matched_count = 0
    blocked_helped = 0.0
    blocked_harmed = 0.0
    active_helped = 0.0
    active_harmed = 0.0
    blocked_cases: List[Dict[str, object]] = []
    active_cases: List[Dict[str, object]] = []

    for case in transition_cases:
        case_idx = int(case.get("case_index", -1))
        stay_id = str(case.get("series_name", ""))
        memory_gain = gain_by_index.get(case_idx, gain_by_stay.get(stay_id, 0.0))
        is_blocked = float(case.get("transition_gate_blocked", 0.0)) > 0.5
        annotated = dict(case)
        annotated["memory_gain_mae"] = float(memory_gain)
        if memory_gain != 0.0:
            matched_count += 1
        if is_blocked:
            blocked_cases.append(annotated)
            if memory_gain > 0.0:
                blocked_helped += 1.0
            elif memory_gain < 0.0:
                blocked_harmed += 1.0
        else:
            active_cases.append(annotated)
            if memory_gain > 0.0:
                active_helped += 1.0
            elif memory_gain < 0.0:
                active_harmed += 1.0

    total_blocked = float(len(blocked_cases))
    total_active = float(len(active_cases))
    total = max(1.0, total_blocked + total_active)

    def _transition_group_summary(cases: List[Dict[str, object]], group_key: str) -> List[Dict[str, object]]:
        grouped: Dict[str, List[Dict[str, object]]] = {}
        for case in cases:
            grouped.setdefault(str(case.get(group_key, "unknown")), []).append(case)
        rows: List[Dict[str, object]] = []
        for key in sorted(grouped):
            bucket = grouped[key]
            gains = [float(c.get("memory_gain_mae", 0.0)) for c in bucket]
            rows.append({
                "group": key,
                "case_count": float(len(bucket)),
                "blocked_rate": float(sum(1 for c in bucket if float(c.get("transition_gate_blocked", 0.0)) > 0.5) / max(1, len(bucket))),
                "mean_memory_gain_mae": float(sum(gains) / max(1, len(gains))),
                "helped_rate": float(sum(1 for g in gains if g > 0.0) / max(1, len(gains))),
                "harmed_rate": float(sum(1 for g in gains if g < 0.0) / max(1, len(gains))),
            })
        return rows

    # blocked gate reason distribution
    reason_counts: Dict[str, float] = dict(factual_path_audit.get("transition_gate_blocked_reasons", {}))

    return {
        "enabled": True,
        "transition_case_count": float(len(transition_cases)),
        "transition_blocked_rate": float(total_blocked / total),
        "transition_active_rate": float(total_active / total),
        "gain_matched_count": float(matched_count),
        "blocked_helped_count": float(blocked_helped),
        "blocked_harmed_count": float(blocked_harmed),
        "active_helped_count": float(active_helped),
        "active_harmed_count": float(active_harmed),
        "blocked_helped_rate": float(blocked_helped / max(1.0, total_blocked)),
        "blocked_harmed_rate": float(blocked_harmed / max(1.0, total_blocked)),
        "active_helped_rate": float(active_helped / max(1.0, total_active)),
        "active_harmed_rate": float(active_harmed / max(1.0, total_active)),
        "blocked_reason_distribution": reason_counts,
        "subgroup_slices": {
            "by_pattern": _transition_group_summary(transition_cases, "pattern"),
            "by_trajectory": _transition_group_summary(transition_cases, "trajectory"),
        },
        "blocked_case_examples": sorted(blocked_cases, key=lambda c: float(c.get("memory_gain_mae", 0.0)))[:20],
        "active_case_examples": sorted(active_cases, key=lambda c: float(c.get("memory_gain_mae", 0.0)), reverse=True)[:20],
    }


def _phase0_report_thresholds(args) -> Dict[str, float]:
    return {
        "min_donor_similarity": float(args.phase0_min_donor_similarity),
        "min_guideline_compatibility": float(args.phase0_min_guideline_compatibility),
        "min_donor_total_score": float(args.phase0_min_donor_total_score),
        "max_missing_care_penalty": float(args.phase0_max_missing_care_penalty),
        "max_contraindication_penalty": float(args.phase0_max_contraindication_penalty),
        "require_positive_delta": float(1.0 if args.phase0_require_positive_delta else 0.0),
        "uncertainty_delta_threshold": float(args.uncertainty_delta_threshold),
    }


def _phase0_report_decision(case_detail: Dict[str, object], thresholds: Dict[str, float]) -> Dict[str, object]:
    selected_candidate = dict(case_detail.get("selected_candidate", {}))
    donor = dict(selected_candidate.get("donor", {}))
    reason_codes: List[str] = []

    donor_stay_id = _safe_float(donor.get("stay_id"), -1.0)
    donor_similarity = _safe_float(donor.get("donor_similarity"))
    donor_guideline = _safe_float(donor.get("donor_guideline_compatibility"))
    donor_total_score = _safe_float(donor.get("donor_total_score"))
    missing_care_penalty = _safe_float(donor.get("donor_missing_care_penalty"))
    contraindication_penalty = _safe_float(donor.get("donor_contraindication_penalty"))
    predicted_delta = _safe_float(case_detail.get("selected_predicted_delta"))
    prediction_uncertainty = _safe_float(case_detail.get("prediction_uncertainty_mean_std", 0.0))
    uncertainty_threshold = max(0.0, float(thresholds.get("uncertainty_delta_threshold", 2.0)))

    if donor_stay_id < 0.0:
        reason_codes.append("no_donor")
    if donor_similarity < thresholds["min_donor_similarity"]:
        reason_codes.append("low_similarity")
    if donor_guideline < thresholds["min_guideline_compatibility"]:
        reason_codes.append("low_guideline")
    if donor_total_score < thresholds["min_donor_total_score"]:
        reason_codes.append("low_total_score")
    if missing_care_penalty > thresholds["max_missing_care_penalty"]:
        reason_codes.append("high_missing_care_penalty")
    if contraindication_penalty > thresholds["max_contraindication_penalty"]:
        reason_codes.append("high_contraindication_penalty")
    if thresholds["require_positive_delta"] > 0.5 and predicted_delta <= 0.0:
        reason_codes.append("non_positive_delta")
    if uncertainty_threshold > 0.0 and prediction_uncertainty > 0.0 and abs(predicted_delta) < prediction_uncertainty * uncertainty_threshold:
        reason_codes.append("uncertain_delta")

    return {
        "status": "recommendation_ready" if len(reason_codes) == 0 else "review_only",
        "reason_codes": reason_codes,
    }


def _build_layered_evaluation_baseline(
    dataset,
    truth: Sequence[Sequence[float]],
    hybrid_predictions: Sequence[Sequence[float]],
    base_predictions: Sequence[Sequence[float]],
    hybrid_metrics: Dict[str, float],
    base_metrics: Dict[str, float],
    counterfactual_summary: Dict[str, object],
    rollout_summary: Dict[str, object],
    memory_gain_audit: Dict[str, object],
    uncertainty_analysis: Dict[str, object],
    args,
) -> Dict[str, object]:
    case_details = list(counterfactual_summary.get("case_details", []))
    thresholds = _phase0_report_thresholds(args)

    hybrid_flat = [float(value) for row in hybrid_predictions for value in row]
    base_flat = [float(value) for row in base_predictions for value in row]
    truth_flat = [float(value) for row in truth for value in row]
    hybrid_signed_errors = [pred - truth_value for pred, truth_value in zip(hybrid_flat, truth_flat)]
    base_signed_errors = [pred - truth_value for pred, truth_value in zip(base_flat, truth_flat)]
    hybrid_abs_errors = [abs(value) for value in hybrid_signed_errors]
    base_abs_errors = [abs(value) for value in base_signed_errors]

    factual_layer = {
        "aggregate_metrics": {
            "memory_enabled": hybrid_metrics,
            "memory_disabled": base_metrics,
            "improvement": _metric_improvements(hybrid_metrics, base_metrics),
        },
        "error_distribution": {
            "memory_enabled_abs_error": _distribution_summary(hybrid_abs_errors),
            "memory_disabled_abs_error": _distribution_summary(base_abs_errors),
            "memory_enabled_signed_error": _distribution_summary(hybrid_signed_errors),
            "memory_disabled_signed_error": _distribution_summary(base_signed_errors),
        },
        "subgroup_slices": {
            "by_pattern": _grouped_factual_slice(dataset.test_samples, truth, hybrid_predictions, base_predictions, "pattern"),
            "by_trajectory": _grouped_factual_slice(dataset.test_samples, truth, hybrid_predictions, base_predictions, "trajectory"),
            "by_pattern_trajectory": _grouped_factual_slice(
                dataset.test_samples,
                truth,
                hybrid_predictions,
                base_predictions,
                "pattern_trajectory",
            ),
        },
        "memory_gain_audit_summary": {
            "sample_count": _safe_float(memory_gain_audit.get("sample_count")),
            "helped_rate": _safe_float(memory_gain_audit.get("helped_rate")),
            "harmed_rate": _safe_float(memory_gain_audit.get("harmed_rate")),
            "mean_memory_gain_mae": _safe_float(memory_gain_audit.get("mean_memory_gain_mae")),
            "outcome_counts": dict(memory_gain_audit.get("outcome_counts", {})),
            "subgroup_slices": dict(memory_gain_audit.get("subgroup_slices", {})),
        },
    }

    selected_candidates = [dict(case.get("selected_candidate", {})) for case in case_details]
    selected_donors = [dict(candidate.get("donor", {})) for candidate in selected_candidates]
    top3_experience_match: List[float] = []
    top3_pattern_match: List[float] = []
    top3_trajectory_match: List[float] = []
    for case in case_details:
        query = dict(case.get("query", {}))
        top_donors = list(case.get("top_donor_candidates", []))[:3]
        if not top_donors:
            continue
        query_experience = int(float(query.get("experience_label", -1)))
        query_pattern = int(float(query.get("pattern_label", -1)))
        query_trajectory = int(float(query.get("trajectory_label", -1)))
        top3_experience_match.append(
            float(sum(1 for donor in top_donors if int(float(donor.get("donor_experience_label", -999))) == query_experience) / len(top_donors))
        )
        top3_pattern_match.append(
            float(sum(1 for donor in top_donors if int(float(donor.get("donor_pattern_label", -999))) == query_pattern) / len(top_donors))
        )
        top3_trajectory_match.append(
            float(sum(1 for donor in top_donors if int(float(donor.get("donor_trajectory_label", -999))) == query_trajectory) / len(top_donors))
        )

    retrieval_layer = {
        "case_count": float(len(case_details)),
        "selected_donor_metrics": {
            "donor_similarity": _distribution_summary([_safe_float(donor.get("donor_similarity")) for donor in selected_donors]),
            "donor_kg_similarity": _distribution_summary([_safe_float(donor.get("donor_kg_similarity")) for donor in selected_donors]),
            "donor_guideline_compatibility": _distribution_summary(
                [_safe_float(donor.get("donor_guideline_compatibility")) for donor in selected_donors]
            ),
            "donor_state_match": _distribution_summary([_safe_float(donor.get("donor_state_match")) for donor in selected_donors]),
            "donor_missing_care_penalty": _distribution_summary(
                [_safe_float(donor.get("donor_missing_care_penalty")) for donor in selected_donors]
            ),
            "donor_contraindication_penalty": _distribution_summary(
                [_safe_float(donor.get("donor_contraindication_penalty")) for donor in selected_donors]
            ),
            "donor_total_score": _distribution_summary([_safe_float(donor.get("donor_total_score")) for donor in selected_donors]),
            "donor_overlap_score": _distribution_summary([_safe_float(donor.get("donor_overlap_score")) for donor in selected_donors]),
            "donor_learned_reranker_score": _distribution_summary(
                [_safe_float(donor.get("donor_learned_reranker_score")) for donor in selected_donors]
            ),
        },
        "match_quality": {
            "donor_found_rate": _safe_float(counterfactual_summary.get("donor_found_rate")),
            "exact_experience_match_rate": _safe_float(counterfactual_summary.get("donor_exact_experience_match_rate")),
            "overlap_valid_rate": _safe_float(counterfactual_summary.get("donor_overlap_valid_rate")),
            "overlap_fallback_rate": _safe_float(counterfactual_summary.get("donor_overlap_fallback_rate")),
            "reranker_mode": str(counterfactual_summary.get("donor_reranker_mode", "")),
            "overlap_reason_counts": dict(counterfactual_summary.get("donor_overlap_reason_counts", {})),
            "top3_same_experience_label_rate_mean": _mean(top3_experience_match),
            "top3_same_pattern_label_rate_mean": _mean(top3_pattern_match),
            "top3_same_trajectory_label_rate_mean": _mean(top3_trajectory_match),
        },
    }

    candidate_options = [dict(option) for case in case_details for option in case.get("candidate_options", [])]
    candidate_source_counts = Counter(str(option.get("candidate_source", "")) for option in candidate_options)
    selected_source_counts = Counter(str(case.get("selected_candidate_source", "")) for case in case_details)
    search_option_count = sum(1 for option in candidate_options if str(option.get("candidate_layer", "")) == "search")
    selected_search_count = sum(
        1
        for candidate in selected_candidates
        if str(candidate.get("candidate_layer", "")) == "search"
    )
    unsafe_option_count = sum(
        1
        for option in candidate_options
        if (
            _safe_float(option.get("donor_missing_care_penalty")) > thresholds["max_missing_care_penalty"]
            or _safe_float(option.get("donor_contraindication_penalty")) > thresholds["max_contraindication_penalty"]
        )
    )
    candidate_layer = {
        "candidate_count_distribution": _distribution_summary([float(len(case.get("candidate_options", []))) for case in case_details]),
        "generated_candidate_case_rate": _safe_float(counterfactual_summary.get("generated_candidate_available_rate")),
        "generated_candidate_selected_rate": _safe_float(counterfactual_summary.get("generated_candidate_selected_rate")),
        "search_candidate_case_rate": _safe_float(counterfactual_summary.get("search_candidate_available_rate")),
        "search_candidate_selected_rate": _safe_float(counterfactual_summary.get("search_candidate_selected_rate")),
        "selected_repair_action_rate": float(
            sum(1 for candidate in selected_candidates if list(candidate.get("repair_actions", []))) / max(1, len(selected_candidates))
        ),
        "candidate_option_repair_rate": float(
            sum(1 for option in candidate_options if list(option.get("repair_actions", []))) / max(1, len(candidate_options))
        ),
        "candidate_option_search_rate": float(search_option_count / max(1, len(candidate_options))),
        "selected_candidate_search_rate": float(selected_search_count / max(1, len(selected_candidates))),
        "unsafe_candidate_option_rate": float(unsafe_option_count / max(1, len(candidate_options))),
        "candidate_source_counts_all_options": dict(candidate_source_counts),
        "selected_candidate_source_counts": dict(selected_source_counts),
    }

    selected_deltas = [_safe_float(case.get("selected_predicted_delta")) for case in case_details]
    donor_original_deltas: List[float] = []
    repaired_candidate_deltas: List[float] = []
    repaired_beats_original = 0.0
    repaired_comparable_cases = 0.0
    for case in case_details:
        donor_original = None
        repaired = None
        for option in case.get("candidate_options", []):
            source = str(option.get("candidate_source", ""))
            if source == "donor_original":
                donor_original = option
            elif source == "generated_kg_repaired":
                repaired = option
        if donor_original is not None:
            donor_original_deltas.append(_safe_float(donor_original.get("predicted_delta")))
        if repaired is not None:
            repaired_candidate_deltas.append(_safe_float(repaired.get("predicted_delta")))
        if donor_original is not None and repaired is not None:
            repaired_comparable_cases += 1.0
            if _safe_float(repaired.get("predicted_delta")) > _safe_float(donor_original.get("predicted_delta")):
                repaired_beats_original += 1.0

    counterfactual_layer = {
        "selection_quality": {
            "mean_predicted_delta": _safe_float(counterfactual_summary.get("mean_predicted_delta")),
            "predicted_improvement_rate": _safe_float(counterfactual_summary.get("predicted_improvement_rate")),
            "selected_delta_distribution": _distribution_summary(selected_deltas),
            "donor_original_delta_distribution": _distribution_summary(donor_original_deltas),
            "generated_repaired_delta_distribution": _distribution_summary(repaired_candidate_deltas),
            "generated_repaired_beats_original_rate": float(repaired_beats_original / max(1.0, repaired_comparable_cases)),
        },
        "selection_context": {
            "donor_score_mode": str(counterfactual_summary.get("donor_score_mode", "")),
            "candidate_policy": str(counterfactual_summary.get("counterfactual_candidate_policy", "")),
            "repaired_comparable_case_count": float(repaired_comparable_cases),
        },
    }

    per_sample_uncertainty: Dict[int, float] = {}
    for row in uncertainty_analysis.get("per_sample_rows", []):
        idx = int(row.get("sample_index", -1))
        std_val = float(row.get("forecast_mean_std", 0.0))
        if idx >= 0:
            per_sample_uncertainty[idx] = std_val
    global_forecast_mean_std = float(uncertainty_analysis.get("forecast_mean_std", 0.0))
    for case in case_details:
        case_idx = int(_safe_float(case.get("case_index", -1.0)))
        case["prediction_uncertainty_mean_std"] = float(
            per_sample_uncertainty.get(case_idx, global_forecast_mean_std)
        )

    report_status_counter: Counter = Counter()
    report_reason_counter: Counter = Counter()
    confidence_counter: Counter = Counter()
    for case in case_details:
        decision = _phase0_report_decision(case, thresholds)
        report_status_counter.update([str(decision["status"])])
        report_reason_counter.update(decision["reason_codes"])
        report_payload = render_counterfactual_case_report(case)
        confidence = str(report_payload.get("recommended_plan", {}).get("confidence", ""))
        if confidence:
            confidence_counter.update([confidence])
    total_cases = max(1, len(case_details))
    report_layer = {
        "thresholds": thresholds,
        "status_counts": dict(report_status_counter),
        "status_rates": {
            "recommendation_ready_rate": float(report_status_counter.get("recommendation_ready", 0) / total_cases),
            "review_only_rate": float(report_status_counter.get("review_only", 0) / total_cases),
        },
        "review_reason_counts": dict(report_reason_counter),
        "confidence_counts": dict(confidence_counter),
    }
    rolling_horizon_layer = {
        "enabled": bool(int(args.counterfactual_rollout_steps) > 1),
        "rollout_steps": int(rollout_summary.get("rollout_steps", max(1, int(args.counterfactual_rollout_steps)))),
        "rollout_discount": float(
            rollout_summary.get("rollout_discount", float(args.counterfactual_rollout_discount))
        ),
        "mean_discounted_cumulative_delta": _safe_float(
            rollout_summary.get("mean_discounted_cumulative_delta")
        ),
        "positive_discounted_cumulative_rate": _safe_float(
            rollout_summary.get("positive_discounted_cumulative_rate")
        ),
        "second_step_available_rate": _safe_float(rollout_summary.get("second_step_available_rate")),
        "stable_candidate_source_rate": _safe_float(rollout_summary.get("stable_candidate_source_rate")),
        "step_positive_rates": [
            float(value) for value in rollout_summary.get("step_positive_rates", [])
        ],
    }

    return {
        "enabled": bool(args.dataset_format == "eicu_sepsis3"),
        "scope": "Phase 0 layered baseline for factual, retrieval, candidate, counterfactual, and report layers.",
        "notices": [
            "factual 层结论与 donor reranking 解耦，优先用于判断记忆路径是否真的提升预测。",
            "counterfactual 层仍然基于模型 self-prediction proxy，不应解读为真实因果疗效证明。",
            "report 层 recommendation_ready/review_only 统计基于 Phase 0 基线阈值，不等同于临床上线阈值。",
        ],
        "run_context": {
            "dataset_name": str(args.dataset_name or "unnamed_dataset"),
            "seed": int(args.seed),
            "test_window_count": float(len(dataset.test_samples)),
            "training_budget": _training_budget_summary(dataset, args),
        },
        "factual_layer": factual_layer,
        "retrieval_layer": retrieval_layer,
        "candidate_layer": candidate_layer,
        "counterfactual_layer": counterfactual_layer,
        "report_layer": report_layer,
        "rolling_horizon_layer": rolling_horizon_layer,
    }


def main():
    args = parse_args()
    persistent_progress_enabled = bool(args.build_persistent_memory_only and not args.disable_persistent_memory_progress)
    if persistent_progress_enabled:
        _log_progress(
            "[persistent-memory] loading dataset for build-only mode "
            f"(format={args.dataset_format}, max_series={args.eicu_max_series}, "
            f"max_train_windows_per_series={args.max_train_windows_per_series})"
        )
    if args.dataset_format == "eicu_sepsis3":
        args.eicu_sepsis3_labels_csv = _resolved_path_str(args.eicu_sepsis3_labels_csv)
        args.eicu_sepsis3_trajectory_csv = _resolved_path_str(args.eicu_sepsis3_trajectory_csv)
        args.kg_directory = _resolved_path_str(args.kg_directory)
        dataset = build_eicu_sepsis3_forecasting_dataset(
            labels_csv=args.eicu_sepsis3_labels_csv,
            trajectory_csv=args.eicu_sepsis3_trajectory_csv,
            dataset_name=args.dataset_name or "eicu_sepsis3_forecasting",
            history_length=args.history_length or 4,
            forecast_horizon=args.forecast_horizon or 2,
            max_train_windows_per_series=args.max_train_windows_per_series,
            target_field=args.eicu_target_field,
            max_series_count=args.eicu_max_series or None,
            enable_kg=args.enable_kg,
            kg_directory=args.kg_directory,
            append_kg_to_patient_static=not bool(args.disable_kg_static_concat),
        )
    else:
        args.tsf = _resolved_path_str(args.tsf)
        dataset = build_tsf_forecasting_dataset(
            tsf_path=args.tsf,
            dataset_name=args.dataset_name,
            history_length=args.history_length or None,
            forecast_horizon=args.forecast_horizon or None,
            max_train_windows_per_series=args.max_train_windows_per_series,
        )

    if args.build_persistent_memory_only:
        if persistent_progress_enabled:
            _log_progress(
                "[persistent-memory] dataset ready: "
                f"train={len(dataset.train_samples)}, val={len(dataset.val_samples)}, test={len(dataset.test_samples)}, "
                f"series={dataset.series_count}"
            )
        if not args.persistent_memory_store:
            raise ValueError("--persistent-memory-store is required with --build-persistent-memory-only")
        persistent_store = PersistentExperienceStore(args.persistent_memory_store)
        selected_splits = _samples_for_persistent_build(dataset, _csv_values(args.persistent_memory_build_splits) or ["train"])
        build_summaries: Dict[str, object] = {}
        for split_name, split_samples in selected_splits.items():
            if persistent_progress_enabled:
                _log_progress(
                    f"[persistent-memory][{split_name}] build split start: samples={len(split_samples)}, "
                    f"store={args.persistent_memory_store}"
                )
            build_summaries[split_name] = persistent_store.upsert_samples(
                split_samples,
                source=split_name,
                prototype_update_mode=args.persistent_prototype_update_mode,
                progress_callback=_make_persistent_progress_callback(persistent_progress_enabled, split_name),
                progress_interval=args.persistent_memory_progress_interval,
            )
        result = {
            "mode": "build_persistent_memory_only",
            "dataset_summary": dataset.summary(),
            "store_path": args.persistent_memory_store,
            "build_splits": list(selected_splits.keys()),
            "prototype_update_mode": args.persistent_prototype_update_mode,
            "progress_enabled": persistent_progress_enabled,
            "progress_interval": int(args.persistent_memory_progress_interval),
            "split_sample_counts": {
                split_name: len(split_samples)
                for split_name, split_samples in selected_splits.items()
            },
            "build_summaries": build_summaries,
            "store_summary": persistent_store.summary(),
        }
        if persistent_progress_enabled:
            _log_progress(
                "[persistent-memory] build-only complete: "
                f"entries={int(result['store_summary'].get('entry_count', 0))}, "
                f"prototypes={int(result['store_summary'].get('prototype_count', 0))}"
            )
        if args.output_json:
            output_json = _resolved_path_str(args.output_json)
            _ensure_parent_dir(output_json)
            with open(output_json, "w", encoding="utf-8") as file:
                json.dump(result, file, ensure_ascii=False, indent=2)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    memory_config = ManifoldMemoryConfig(
        sequence_feature_dim=len(dataset.sequence_feature_names),
        static_feature_dim=len(dataset.patient_feature_names or dataset.static_feature_names),
        manifold_dim=args.manifold_dim,
        value_dim=args.manifold_value_dim,
        fusion_hidden_dim=args.manifold_fusion_hidden_dim,
        top_k=args.top_k,
        temperature=args.memory_temperature,
        similarity_threshold=args.sim_threshold,
        merge_alpha=args.merge_alpha,
        decay=args.decay,
        forget_threshold=args.forget_threshold,
        max_memory=args.max_memory,
        same_label_merge_only=False,
        min_label_memory=args.memory_min_label,
        max_label_memory=args.memory_max_label,
        max_patient_label_memory=args.memory_max_patient_label,
        support_penalty=args.memory_support_penalty,
        collapse_penalty=args.memory_collapse_penalty,
        encoder_type=args.encoder_type,
        gru_hidden_dim=args.gru_hidden_dim,
        gru_layers=args.gru_layers,
        gru_bidirectional=args.gru_bidirectional,
        gru_dropout=args.gru_dropout,
        transformer_d_model=args.transformer_d_model,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_ff_dim=args.transformer_ff_dim,
        transformer_dropout=args.transformer_dropout,
        transformer_max_length=args.transformer_max_length,
        static_hidden_dim=args.static_hidden_dim,
        device=_resolve_device(args.device),
    )
    trainer_config = ForecastingTrainerConfig(
        forecast_horizon=dataset.forecast_horizon,
        seasonality=dataset.seasonality,
        history_length=dataset.history_length,
        series_count=dataset.series_count,
        dataset_name=dataset.dataset_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        aux_base_loss_weight=args.aux_base_loss_weight,
        multitask_loss_weight=args.multitask_loss_weight,
        align_loss_weight=args.align_loss_weight,
        temporal_smoothness_weight=args.temporal_smoothness_weight,
        grad_clip=args.grad_clip,
        device=_resolve_device(args.device),
        seed=args.seed,
        enable_epoch_feedback=bool(args.enable_epoch_feedback),
        hard_example_weight=args.hard_example_weight,
        kg_consistency_weight=args.kg_consistency_weight,
        path_alignment_weight=args.path_alignment_weight,
        archive_retention_weight=args.archive_retention_weight,
        memory_delta_floor_weight=args.memory_delta_floor_weight,
        archive_retention_target=args.archive_retention_target,
        memory_delta_floor=args.memory_delta_floor,
        epoch_feedback_momentum=args.epoch_feedback_momentum,
        feedback_top_error_rate=args.feedback_top_error_rate,
        checkpoint_selection_mode=args.checkpoint_selection_mode,
        memory_refresh_interval=max(0, int(args.memory_refresh_interval)),
    )

    trainer = EndToEndForecastingManifoldTrainer(
        memory_config=memory_config,
        trainer_config=trainer_config,
        static_feature_dim=len(dataset.patient_feature_names or dataset.static_feature_names),
        kg_feature_dim=len(dataset.kg_feature_names or []),
        intervention_feature_dim=len(dataset.intervention_feature_names or []),
        intervention_sequence_dim=len(dataset.intervention_sequence_feature_names or []),
        formation_feature_dim=len(dataset.formation_feature_names),
    )
    trainer.enable_explicit_kg_path = bool(args.enable_explicit_kg_path and len(dataset.kg_feature_names or []) > 0)
    trainer.kg_residual_weight = float(args.kg_residual_weight)
    trainer.kg_alignment_floor = float(args.kg_alignment_floor)
    trainer.memory_direct_residual_weight = float(args.memory_direct_residual_weight)
    trainer.memory_direct_residual_mode = args.memory_direct_residual_mode
    trainer.memory_path_coordination_mode = args.memory_path_coordination_mode
    trainer.branch_decoupling_mode = str(args.branch_decoupling_mode)
    trainer.branch_decoupling_enabled = not bool(args.disable_branch_decoupling)
    trainer.retrieval_projection_scale = float(args.retrieval_projection_scale)
    trainer.memory_harm_control_enabled = not bool(args.disable_memory_harm_control)
    trainer.memory_quality_floor = float(args.memory_quality_floor)
    trainer.harm_stable_quality_boost = float(args.harm_stable_quality_boost)
    trainer.harm_flat_quality_boost = float(args.harm_flat_quality_boost)
    trainer.memory_min_path_alignment = float(args.memory_min_path_alignment)
    trainer.memory_residual_cap_ratio = float(args.memory_residual_cap_ratio)
    if bool(args.enable_epoch_feedback) and trainer.memory_path_coordination_mode == "sum":
        trainer.memory_path_coordination_mode = "adaptive"
    trainer.counterfactual_donor_score_mode = args.counterfactual_donor_score_mode
    trainer.counterfactual_base_similarity_weight = float(args.counterfactual_base_similarity_weight)
    trainer.counterfactual_kg_similarity_weight = float(args.counterfactual_kg_similarity_weight)
    trainer.counterfactual_guideline_weight = float(args.counterfactual_guideline_weight)
    trainer.counterfactual_guideline_score_weight = float(args.counterfactual_guideline_score_weight)
    trainer.counterfactual_penalty_weight = float(args.counterfactual_penalty_weight)
    trainer.counterfactual_hard_filter_enabled = not bool(args.disable_counterfactual_hard_filter)
    trainer.counterfactual_candidate_policy = args.counterfactual_candidate_policy
    trainer.counterfactual_candidate_feasibility_weight = float(args.counterfactual_candidate_feasibility_weight)
    trainer.counterfactual_candidate_penalty_weight = float(args.counterfactual_candidate_penalty_weight)
    trainer.counterfactual_label_top_k = int(args.counterfactual_label_top_k)
    trainer.counterfactual_pool_include_pattern = not bool(args.disable_counterfactual_pool_include_pattern)
    trainer.counterfactual_pool_include_trajectory = not bool(args.disable_counterfactual_pool_include_trajectory)
    trainer.counterfactual_pool_enable_global_backfill = not bool(args.disable_counterfactual_pool_global_backfill)
    trainer.counterfactual_pool_min_candidates = int(args.counterfactual_pool_min_candidates)
    trainer.counterfactual_pool_global_limit = int(args.counterfactual_pool_global_limit)
    trainer.counterfactual_pool_prefilter_top_k = int(args.counterfactual_pool_prefilter_top_k)
    trainer.counterfactual_overlap_filter_enabled = not bool(args.disable_counterfactual_overlap_filter)
    trainer.counterfactual_overlap_fallback_enabled = not bool(args.disable_counterfactual_overlap_fallback)
    trainer.counterfactual_overlap_weight = float(args.counterfactual_overlap_weight)
    trainer.counterfactual_overlap_severity_gap_max = float(args.counterfactual_overlap_severity_gap_max)
    trainer.counterfactual_overlap_trend_gap_max = float(args.counterfactual_overlap_trend_gap_max)
    trainer.counterfactual_overlap_state_min = float(args.counterfactual_overlap_state_min)
    trainer.counterfactual_overlap_action_min = float(args.counterfactual_overlap_action_min)
    trainer.counterfactual_rollout_steps = max(1, int(args.counterfactual_rollout_steps))
    trainer.counterfactual_rollout_discount = float(args.counterfactual_rollout_discount)
    trainer.counterfactual_reranker_mode = str(args.counterfactual_reranker_mode)
    trainer.counterfactual_reranker_blend_weight = float(args.counterfactual_reranker_blend_weight)
    trainer.counterfactual_reranker_train_top_k = int(args.counterfactual_reranker_train_top_k)
    trainer.counterfactual_reranker_max_samples = int(args.counterfactual_reranker_max_samples)
    trainer.counterfactual_reranker_min_examples = int(args.counterfactual_reranker_min_examples)
    trainer.counterfactual_reranker_ridge_l2 = float(args.counterfactual_reranker_ridge_l2)
    trainer.enable_transition_memory = bool(args.enable_transition_memory)
    trainer.enable_transition_factual_path = not bool(args.disable_transition_factual_path)
    trainer.enable_transition_donor_path = not bool(args.disable_transition_donor_path)
    trainer.transition_top_k = int(args.transition_top_k)
    trainer.transition_state_weight = float(args.transition_state_weight)
    trainer.transition_action_weight = float(args.transition_action_weight)
    trainer.transition_score_weight = float(args.transition_score_weight)
    trainer.transition_template_blend_weight = float(args.transition_template_blend_weight)
    trainer.transition_selection_weight = float(args.transition_selection_weight)
    trainer.transition_temperature = float(args.transition_temperature)
    trainer.transition_utility_scale = float(args.transition_utility_scale)
    trainer.transition_utility_alignment_weight = float(args.transition_utility_alignment_weight)
    trainer.transition_anchor_blend_weight = float(args.transition_anchor_blend_weight)
    trainer.transition_signature_match_weight = float(args.transition_signature_match_weight)
    trainer.transition_min_confidence = float(args.transition_min_confidence)
    trainer.transition_min_support = float(args.transition_min_support)
    trainer.transition_min_expected_utility = float(args.transition_min_expected_utility)
    trainer.transition_min_signature_weight = float(args.transition_min_signature_weight)
    trainer.transition_stable_regime_penalty = float(args.transition_stable_regime_penalty)
    trainer.transition_flat_pattern_penalty = float(args.transition_flat_pattern_penalty)
    trainer.transition_utility_bias = float(args.transition_utility_bias)
    trainer.transition_utility_temperature = float(args.transition_utility_temperature)
    trainer.transition_signature_partial_match = not bool(args.disable_transition_partial_signature)
    trainer.transition_residual_cap_ratio = float(args.transition_residual_cap_ratio)
    trainer.transition_trunk_weight = float(args.transition_trunk_weight)
    trainer.enable_transition_trunk_path = bool(
        (float(args.transition_trunk_weight) > 0.0) and (not bool(args.disable_transition_trunk_path))
    )
    trainer.transition_factual_residual_mode = str(args.transition_factual_residual_mode)
    trainer.transition_positive_only = bool(args.transition_positive_only)
    trainer.transition_action_change_weight = float(args.transition_action_change_weight)
    trainer.transition_candidate_action_change_weight = float(args.transition_candidate_action_change_weight)
    trainer.sequence_feature_names = list(dataset.sequence_feature_names or [])
    trainer.patient_feature_names = list(dataset.patient_feature_names or dataset.static_feature_names or [])
    trainer.intervention_feature_names = list(dataset.intervention_feature_names or [])
    trainer.intervention_sequence_feature_names = list(dataset.intervention_sequence_feature_names or [])
    persistent_store = None
    persistent_samples = []
    persistent_summary = {}
    persistent_prototypes = []
    persistent_prefit_prime = {}
    persistent_reuse_audit = {}
    persistent_prototype_audit = {}
    memory_seed_schema_audit = {}
    persistent_allowed_splits = _csv_values(args.persistent_memory_allowed_splits) or ["train", "external"]
    exclude_eval_stay_ids: List[object] = []
    exclude_eval_patient_ids: List[object] = []
    if not args.disable_persistent_memory_exclude_current_eval:
        exclude_eval_stay_ids, exclude_eval_patient_ids = _persistent_exclusion_ids(
            list(dataset.val_samples) + list(dataset.test_samples)
        )
    if args.persistent_memory_store:
        persistent_store = PersistentExperienceStore(args.persistent_memory_store)
        if args.prime_persistent_memory_before_fit and not args.disable_persistent_memory_reuse and not args.disable_persistent_memory_export:
            persistent_prefit_prime = persistent_store.upsert_samples(
                dataset.train_samples,
                source="train_prefit_prime",
                prototype_update_mode=args.persistent_prototype_update_mode,
            )
        if not args.disable_persistent_memory_reuse:
            persistent_samples, persistent_reuse_audit = persistent_store.load_samples(
                dataset_name=dataset.dataset_name if args.persistent_memory_scope == "dataset" else None,
                seasonality=dataset.seasonality if args.persistent_memory_scope == "dataset" else None,
                forecast_horizon=dataset.forecast_horizon if args.persistent_memory_scope == "dataset" else None,
                allowed_splits=persistent_allowed_splits,
                exclude_stay_ids=exclude_eval_stay_ids,
                exclude_patient_ids=exclude_eval_patient_ids,
                strict_no_test=not bool(args.disable_persistent_memory_strict_no_test),
                return_audit=True,
            )
            persistent_prototypes, persistent_prototype_audit = persistent_store.load_prototypes(
                dataset_name=dataset.dataset_name if args.persistent_memory_scope == "dataset" else None,
                seasonality=dataset.seasonality if args.persistent_memory_scope == "dataset" else None,
                forecast_horizon=dataset.forecast_horizon if args.persistent_memory_scope == "dataset" else None,
                allowed_splits=persistent_allowed_splits,
                strict_no_test=not bool(args.disable_persistent_memory_strict_no_test),
                return_audit=True,
            )
        persistent_summary = persistent_store.summary()
        trainer.configure_semantic_store(
            store=persistent_store if not args.disable_persistent_memory_reuse else None,
            top_k=args.persistent_semantic_top_k,
        )
        trainer.configure_neural_cache(
            store=persistent_store,
            reuse_enabled=not args.disable_persistent_memory_reuse,
            export_enabled=not args.disable_persistent_memory_export,
        )

    memory_seed_samples, memory_seed_schema_audit = _filter_memory_seed_samples_for_schema(
        _dedupe_memory_seed_samples(persistent_samples + list(dataset.train_samples)),
        expected_patient_dim=len(dataset.patient_feature_names or dataset.static_feature_names or []),
        expected_intervention_dim=len(dataset.intervention_feature_names or []),
        expected_intervention_sequence_dim=len(dataset.intervention_sequence_feature_names or []),
    )
    if not memory_seed_samples:
        raise RuntimeError(
            "No memory seed samples remain after schema filtering. "
            f"Audit: {memory_seed_schema_audit}"
        )

    trainer.fit(
        dataset.train_samples,
        dataset.val_samples,
        memory_seed_samples=memory_seed_samples,
        collect_diagnostics=not args.skip_posthoc_diagnostics,
    )
    reranker_training_summary = trainer.fit_counterfactual_reranker(dataset.train_samples)

    hybrid_predictions = trainer.predict(dataset.test_samples, use_memory=True)
    base_predictions = trainer.predict(dataset.test_samples, use_memory=False)
    multitask_metrics = (
        trainer.evaluate_multitask(dataset.test_samples, use_memory=True)
        if args.dataset_format == "eicu_sepsis3"
        else {"enabled": False, "task_metrics": {}}
    )
    uncertainty_analysis = trainer.analyze_uncertainty(
        dataset.test_samples,
        use_memory=True,
        num_samples=max(1, int(args.test_uncertainty_samples)),
    )
    example_uncertainty = trainer.predict_with_uncertainty(
        [dataset.test_samples[0]],
        use_memory=True,
        num_samples=max(1, int(args.test_uncertainty_samples)),
        include_auxiliary=True,
    )
    example_auxiliary_prediction = trainer.predict_with_auxiliary(
        [dataset.test_samples[0]],
        use_memory=True,
    )
    counterfactual_summary = (
        trainer.predict_counterfactual(
            dataset.test_samples,
            include_predictions=args.counterfactual_store_predictions,
        )
        if args.dataset_format == "eicu_sepsis3"
        else {}
    )
    rollout_summary = (
        trainer.predict_counterfactual_rollout(
            dataset.test_samples,
            rollout_steps=int(args.counterfactual_rollout_steps),
            include_predictions=False,
        )
        if args.dataset_format == "eicu_sepsis3" and int(args.counterfactual_rollout_steps) > 1
        else {}
    )
    example_counterfactual_report = {}
    if counterfactual_summary.get("case_details"):
        example_counterfactual_report = render_counterfactual_case_report(counterfactual_summary["case_details"][0])
    example_trace = {} if args.skip_posthoc_diagnostics else trainer.inspect_sample(dataset.test_samples[0])
    truth = [sample.raw_target for sample in dataset.test_samples]
    insample = [
        [step[0] * sample.scale_value + sample.scale_center for step in sample.sequence]
        for sample in dataset.test_samples
    ]
    hybrid_metrics = forecasting_metrics(truth, hybrid_predictions, insample, dataset.seasonality)
    base_metrics = forecasting_metrics(truth, base_predictions, insample, dataset.seasonality)
    memory_effectiveness = _metric_improvements(hybrid_metrics, base_metrics)
    memory_gain_audit = _build_memory_gain_audit(
        samples=dataset.test_samples,
        truth=truth,
        hybrid_predictions=hybrid_predictions,
        base_predictions=base_predictions,
        max_cases=int(args.memory_gain_audit_max_cases),
        top_k=int(args.memory_gain_audit_top_k),
    )
    factual_path_audit = trainer.audit_factual_memory_path(
        dataset.test_samples,
        max_samples=min(len(dataset.test_samples), 256),
    )
    transition_gate_audit = _build_transition_gate_audit(
        factual_path_audit=factual_path_audit,
        memory_gain_audit=memory_gain_audit,
    )
    val_memory_gain_audit: Dict[str, object] = {}
    if not args.skip_posthoc_diagnostics and dataset.val_samples:
        val_hybrid_preds = trainer.predict(
            dataset.val_samples,
            use_memory=True,
        )
        val_base_only = trainer.predict(
            dataset.val_samples,
            use_memory=False,
        )
        val_truth = [sample.raw_target for sample in dataset.val_samples]
        val_hybrid = [list(row) for row in val_hybrid_preds]
        val_base = [list(row) for row in val_base_only]
        val_memory_gain_audit = _build_memory_gain_audit(
            samples=dataset.val_samples,
            truth=val_truth,
            hybrid_predictions=val_hybrid,
            base_predictions=val_base,
            max_cases=int(args.memory_gain_audit_max_cases),
            top_k=int(args.memory_gain_audit_top_k),
        )
    error_attribution = {}
    if not args.skip_posthoc_diagnostics:
        max_attr_samples = max(0, int(args.error_attribution_max_samples))
        error_attribution = {
            "validation": trainer.analyze_error_attribution(
                dataset.val_samples,
                max_samples=min(len(dataset.val_samples), max_attr_samples) if max_attr_samples > 0 else 0,
            ),
            "test": trainer.analyze_error_attribution(
                dataset.test_samples,
                max_samples=min(len(dataset.test_samples), max_attr_samples) if max_attr_samples > 0 else 0,
            ),
        }
    factual_line = _build_factual_forecasting_line(
        hybrid_metrics=hybrid_metrics,
        base_metrics=base_metrics,
        trainer=trainer,
        dataset=dataset,
        args=args,
        example_trace=example_trace,
        factual_path_audit=factual_path_audit,
        memory_gain_audit=memory_gain_audit,
        transition_gate_audit=transition_gate_audit,
        error_attribution=error_attribution,
    )
    donor_ranking_line = _build_counterfactual_donor_ranking_line(counterfactual_summary)
    clinical_plausibility_line = _build_clinical_plausibility_line(counterfactual_summary)
    memory_usefulness_assessment = _build_memory_usefulness_assessment(
        factual_line=factual_line,
        donor_line=donor_ranking_line,
        clinical_line=clinical_plausibility_line,
        args=args,
    )
    diagnostic_summary = _build_diagnostic_summary(
        memory_gain_audit=memory_gain_audit,
        val_memory_gain_audit=val_memory_gain_audit,
        factual_path_audit=factual_path_audit,
        counterfactual_summary=counterfactual_summary,
        uncertainty_analysis=uncertainty_analysis,
        memory_diagnostics=trainer.memory_diagnostics,
        transition_gate_audit=transition_gate_audit,
    )
    layered_evaluation_baseline = _build_layered_evaluation_baseline(
        dataset=dataset,
        truth=truth,
        hybrid_predictions=hybrid_predictions,
        base_predictions=base_predictions,
        hybrid_metrics=hybrid_metrics,
        base_metrics=base_metrics,
        counterfactual_summary=counterfactual_summary,
        rollout_summary=rollout_summary,
        memory_gain_audit=memory_gain_audit,
        uncertainty_analysis=uncertainty_analysis,
        args=args,
    )

    persistent_update = {}
    if persistent_store is not None and not args.disable_persistent_memory_export:
        persistent_update = persistent_store.upsert_samples(
            dataset.train_samples,
            source="train",
            prototype_update_mode=args.persistent_prototype_update_mode,
        )
        persistent_summary = persistent_store.summary()

    input_sources = {
        "dataset_format": args.dataset_format,
        "tsf": args.tsf,
        "eicu_sepsis3_labels_csv": args.eicu_sepsis3_labels_csv if args.dataset_format == "eicu_sepsis3" else "",
        "eicu_sepsis3_trajectory_csv": args.eicu_sepsis3_trajectory_csv if args.dataset_format == "eicu_sepsis3" else "",
        "eicu_target_field": args.eicu_target_field if args.dataset_format == "eicu_sepsis3" else "",
        "eicu_max_series": args.eicu_max_series if args.dataset_format == "eicu_sepsis3" else 0,
        "enable_kg": bool(args.enable_kg) if args.dataset_format == "eicu_sepsis3" else False,
        "kg_directory": args.kg_directory if args.dataset_format == "eicu_sepsis3" and args.enable_kg else "",
        "append_kg_to_patient_static": (not bool(args.disable_kg_static_concat)) if args.dataset_format == "eicu_sepsis3" else False,
    }
    dataset_semantics = {
        "pattern_labels": PATTERN_LABELS,
        "trajectory_labels": TRAJECTORY_LABELS,
        "experience_label_count": len(PATTERN_LABELS) * len(TRAJECTORY_LABELS),
        "formation_feature_names": dataset.formation_feature_names,
        "static_feature_names": dataset.static_feature_names,
        "patient_feature_names": dataset.patient_feature_names or [],
        "intervention_feature_names": dataset.intervention_feature_names or [],
        "intervention_sequence_feature_names": dataset.intervention_sequence_feature_names or [],
        "kg_feature_names": dataset.kg_feature_names or [],
        "sequence_feature_names": dataset.sequence_feature_names,
        "aux_target_names": dataset.aux_target_names or [],
    }
    inference_bundle_path = ""
    if args.output_inference_bundle:
        inference_bundle_path = _resolved_path_str(args.output_inference_bundle)
        _ensure_parent_dir(inference_bundle_path)
        torch.save(
            {
                "schema_version": 1,
                "bundle_type": "eicu_counterfactual_inference",
                "dataset_summary": dataset.summary(),
                "input_sources": input_sources,
                "dataset_semantics": dataset_semantics,
                "persistent_memory": {
                    "store_path": args.persistent_memory_store,
                    "semantic_top_k": args.persistent_semantic_top_k,
                    "reuse_disabled": bool(args.disable_persistent_memory_reuse),
                    "enabled": bool(args.persistent_memory_store),
                    "allowed_splits": persistent_allowed_splits,
                    "strict_no_test": not bool(args.disable_persistent_memory_strict_no_test),
                    "exclude_current_eval": not bool(args.disable_persistent_memory_exclude_current_eval),
                    "prototype_update_mode": args.persistent_prototype_update_mode,
                },
                "trainer_bundle": trainer.export_inference_bundle(),
            },
            inference_bundle_path,
        )

    result = {
        "dataset_summary": dataset.summary(),
        "input_sources": input_sources,
        "dataset_semantics": dataset_semantics,
        "trainer_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "enable_epoch_feedback": bool(args.enable_epoch_feedback),
            "multitask_loss_weight": args.multitask_loss_weight,
            "test_uncertainty_samples": args.test_uncertainty_samples,
            "hard_example_weight": args.hard_example_weight,
            "kg_consistency_weight": args.kg_consistency_weight,
            "path_alignment_weight": args.path_alignment_weight,
            "archive_retention_weight": args.archive_retention_weight,
            "memory_delta_floor_weight": args.memory_delta_floor_weight,
            "archive_retention_target": args.archive_retention_target,
            "memory_delta_floor": args.memory_delta_floor,
            "epoch_feedback_momentum": args.epoch_feedback_momentum,
            "feedback_top_error_rate": args.feedback_top_error_rate,
            "error_attribution_max_samples": args.error_attribution_max_samples,
            "checkpoint_selection_mode": args.checkpoint_selection_mode,
            "device": _resolve_device(args.device),
            "forecast_horizon": dataset.forecast_horizon,
            "history_length": dataset.history_length,
            "series_count": dataset.series_count,
        },
        "memory_config": {
            "top_k": args.top_k,
            "similarity_threshold": args.sim_threshold,
            "merge_alpha": args.merge_alpha,
            "max_memory": args.max_memory,
            "encoder_type": args.encoder_type,
            "manifold_dim": args.manifold_dim,
            "value_dim": args.manifold_value_dim,
            "gru_hidden_dim": args.gru_hidden_dim,
            "transformer_d_model": args.transformer_d_model,
            "transformer_layers": args.transformer_layers,
            "transformer_heads": args.transformer_heads,
            "memory_direct_residual_weight": args.memory_direct_residual_weight,
            "memory_direct_residual_mode": args.memory_direct_residual_mode,
            "memory_path_coordination_mode": trainer.memory_path_coordination_mode,
            "memory_harm_control_enabled": not bool(args.disable_memory_harm_control),
            "memory_quality_floor": args.memory_quality_floor,
            "memory_min_path_alignment": args.memory_min_path_alignment,
            "memory_residual_cap_ratio": args.memory_residual_cap_ratio,
            "memory_gain_audit_top_k": args.memory_gain_audit_top_k,
            "memory_gain_audit_max_cases": args.memory_gain_audit_max_cases,
            "memory_refresh_interval": int(args.memory_refresh_interval),
            "enable_epoch_feedback": bool(args.enable_epoch_feedback),
            "hard_example_weight": args.hard_example_weight,
            "kg_consistency_weight": args.kg_consistency_weight,
            "path_alignment_weight": args.path_alignment_weight,
            "archive_retention_weight": args.archive_retention_weight,
            "memory_delta_floor_weight": args.memory_delta_floor_weight,
            "archive_retention_target": args.archive_retention_target,
            "memory_delta_floor": args.memory_delta_floor,
            "epoch_feedback_momentum": args.epoch_feedback_momentum,
            "feedback_top_error_rate": args.feedback_top_error_rate,
            "error_attribution_max_samples": args.error_attribution_max_samples,
            "checkpoint_selection_mode": args.checkpoint_selection_mode,
            "enable_explicit_kg_path": bool(args.enable_explicit_kg_path),
            "disable_kg_static_concat": bool(args.disable_kg_static_concat),
            "kg_residual_weight": args.kg_residual_weight,
            "kg_alignment_floor": args.kg_alignment_floor,
            "memory_min_label": args.memory_min_label,
            "memory_max_label": args.memory_max_label,
            "memory_max_patient_label": args.memory_max_patient_label,
            "skip_posthoc_diagnostics": bool(args.skip_posthoc_diagnostics),
            "counterfactual_store_predictions": bool(args.counterfactual_store_predictions),
            "counterfactual_donor_score_mode": args.counterfactual_donor_score_mode,
            "counterfactual_base_similarity_weight": args.counterfactual_base_similarity_weight,
            "counterfactual_kg_similarity_weight": args.counterfactual_kg_similarity_weight,
            "counterfactual_guideline_weight": args.counterfactual_guideline_weight,
            "counterfactual_guideline_score_weight": args.counterfactual_guideline_score_weight,
            "counterfactual_penalty_weight": args.counterfactual_penalty_weight,
            "counterfactual_hard_filter_enabled": not bool(args.disable_counterfactual_hard_filter),
            "counterfactual_candidate_policy": args.counterfactual_candidate_policy,
            "counterfactual_candidate_feasibility_weight": args.counterfactual_candidate_feasibility_weight,
            "counterfactual_candidate_penalty_weight": args.counterfactual_candidate_penalty_weight,
            "counterfactual_label_top_k": args.counterfactual_label_top_k,
            "counterfactual_pool_include_pattern": not bool(args.disable_counterfactual_pool_include_pattern),
            "counterfactual_pool_include_trajectory": not bool(args.disable_counterfactual_pool_include_trajectory),
            "counterfactual_pool_enable_global_backfill": not bool(args.disable_counterfactual_pool_global_backfill),
            "counterfactual_pool_min_candidates": args.counterfactual_pool_min_candidates,
            "counterfactual_pool_global_limit": args.counterfactual_pool_global_limit,
            "counterfactual_pool_prefilter_top_k": args.counterfactual_pool_prefilter_top_k,
            "counterfactual_overlap_filter_enabled": not bool(args.disable_counterfactual_overlap_filter),
            "counterfactual_overlap_fallback_enabled": not bool(args.disable_counterfactual_overlap_fallback),
            "counterfactual_overlap_weight": args.counterfactual_overlap_weight,
            "counterfactual_overlap_severity_gap_max": args.counterfactual_overlap_severity_gap_max,
            "counterfactual_overlap_trend_gap_max": args.counterfactual_overlap_trend_gap_max,
            "counterfactual_overlap_state_min": args.counterfactual_overlap_state_min,
            "counterfactual_overlap_action_min": args.counterfactual_overlap_action_min,
            "counterfactual_reranker_mode": args.counterfactual_reranker_mode,
            "counterfactual_reranker_blend_weight": args.counterfactual_reranker_blend_weight,
            "counterfactual_reranker_train_top_k": args.counterfactual_reranker_train_top_k,
            "counterfactual_reranker_max_samples": args.counterfactual_reranker_max_samples,
            "counterfactual_reranker_min_examples": args.counterfactual_reranker_min_examples,
            "counterfactual_reranker_ridge_l2": args.counterfactual_reranker_ridge_l2,
            "enable_transition_memory": bool(args.enable_transition_memory),
            "enable_transition_factual_path": not bool(args.disable_transition_factual_path),
            "enable_transition_donor_path": not bool(args.disable_transition_donor_path),
            "transition_top_k": args.transition_top_k,
            "transition_state_weight": args.transition_state_weight,
            "transition_action_weight": args.transition_action_weight,
            "transition_score_weight": args.transition_score_weight,
            "transition_template_blend_weight": args.transition_template_blend_weight,
            "transition_selection_weight": args.transition_selection_weight,
            "transition_temperature": args.transition_temperature,
            "transition_utility_scale": args.transition_utility_scale,
            "transition_utility_alignment_weight": args.transition_utility_alignment_weight,
            "transition_anchor_blend_weight": args.transition_anchor_blend_weight,
            "transition_signature_match_weight": args.transition_signature_match_weight,
            "transition_trunk_weight": args.transition_trunk_weight,
            "enable_transition_trunk_path": bool(
                (float(args.transition_trunk_weight) > 0.0) and (not bool(args.disable_transition_trunk_path))
            ),
            "transition_factual_residual_mode": args.transition_factual_residual_mode,
            "transition_positive_only": bool(args.transition_positive_only),
            "transition_action_change_weight": args.transition_action_change_weight,
            "transition_candidate_action_change_weight": args.transition_candidate_action_change_weight,
            "enable_kg": bool(args.enable_kg),
        },
        "parameter_count": trainer.parameter_count(),
        "training_summary": trainer.training_summary,
        "memory_diagnostics": trainer.memory_diagnostics,
        "counterfactual_reranker_training": reranker_training_summary,
        "test_metrics": hybrid_metrics,
        "test_multitask_metrics": multitask_metrics,
        "test_uncertainty_analysis": uncertainty_analysis,
        "base_only": base_metrics,
        "memory_effectiveness": memory_effectiveness,
        "memory_gain_audit": memory_gain_audit,
        "transition_gate_audit": transition_gate_audit,
        "val_memory_gain_audit": val_memory_gain_audit,
        "counterfactual_evaluation": counterfactual_summary,
        "counterfactual_rollout_evaluation": rollout_summary,
        "layered_evaluation_baseline": layered_evaluation_baseline,
        "evaluation_lines": {
            "factual_forecasting": factual_line,
            "counterfactual_donor_ranking": donor_ranking_line,
            "clinical_plausibility": clinical_plausibility_line,
        },
        "memory_usefulness_assessment": memory_usefulness_assessment,
        "diagnostic_summary": diagnostic_summary,
        "persistent_memory": {
            "enabled": bool(args.persistent_memory_store),
            "store_path": args.persistent_memory_store,
            "scope": args.persistent_memory_scope,
            "allowed_splits": persistent_allowed_splits,
            "strict_no_test": not bool(args.disable_persistent_memory_strict_no_test),
            "exclude_current_eval": not bool(args.disable_persistent_memory_exclude_current_eval),
            "prototype_update_mode": args.persistent_prototype_update_mode,
            "excluded_eval_stay_count": len(set(str(value) for value in exclude_eval_stay_ids)),
            "excluded_eval_patient_count": len(set(str(value) for value in exclude_eval_patient_ids)),
            "loaded_persistent_samples": len(persistent_samples),
            "loaded_persistent_prototypes": len(persistent_prototypes),
            "memory_seed_schema_audit": memory_seed_schema_audit,
            "semantic_top_k": args.persistent_semantic_top_k,
            "primed_before_fit": bool(args.prime_persistent_memory_before_fit),
            "prefit_prime_summary": persistent_prefit_prime,
            "model_fingerprint": trainer.current_model_fingerprint,
            "neural_cache": trainer.neural_cache_status,
            "export_disabled": bool(args.disable_persistent_memory_export),
            "reuse_disabled": bool(args.disable_persistent_memory_reuse),
            "store_summary": persistent_summary,
            "reuse_audit": persistent_reuse_audit,
            "prototype_audit": persistent_prototype_audit,
            "update_summary": persistent_update,
        },
        "inference_bundle": {
            "saved": bool(inference_bundle_path),
            "path": inference_bundle_path,
        },
        "example_prediction": {
            "series_name": dataset.test_samples[0].metadata.get("series_name", "series_0"),
            "pattern_label": PATTERN_LABELS[dataset.test_samples[0].pattern_label],
            "trajectory_label": TRAJECTORY_LABELS[dataset.test_samples[0].trajectory_label],
            "experience_label": int(dataset.test_samples[0].experience_label),
            "formation_features": {
                name: value for name, value in zip(dataset.formation_feature_names, dataset.test_samples[0].formation_features)
            },
            "truth": truth[0],
            "hybrid_prediction": hybrid_predictions[0],
            "base_prediction": base_predictions[0],
            "uncertainty": example_uncertainty.get("samples", [{}])[0] if example_uncertainty.get("samples") else {},
            "auxiliary_prediction": example_auxiliary_prediction.get("auxiliary_predictions", [{}])[0]
            if example_auxiliary_prediction.get("auxiliary_predictions")
            else {},
            "counterfactual_prediction": counterfactual_summary.get("predictions", [])[0] if counterfactual_summary.get("predictions") else [],
            "counterfactual_donor_stay_id": counterfactual_summary.get("donor_stay_ids", [])[0] if counterfactual_summary else -1.0,
        },
        "example_counterfactual_report": example_counterfactual_report,
        "example_retrieval_trace": example_trace,
    }

    output_json = args.output_json
    if output_json:
        output_json = _resolved_path_str(output_json)
        _ensure_parent_dir(output_json)
        with open(output_json, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
