import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from src.evaluate import basic_metrics
from src.manifold_memory import ManifoldMemoryBlueprint, ManifoldMemoryConfig
from src.manifold_trainer import EndToEndManifoldTrainer, ManifoldTrainerConfig
from src.memory_model import DynamicMemoryClassifier
from src.temporal_data import (
    build_patient_forecast_dataset,
    build_window_forecast_dataset,
    gather_standardized_split,
    grouped_temporal_cv_splits,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EICU_PROCESSED_DIR = (PROJECT_ROOT.parent / "eicu\u6570\u636e\u5e93" / "processed").resolve()
DEFAULT_EICU_LABELS_CSV = (DEFAULT_EICU_PROCESSED_DIR / "eicu_sepsis3_labels.csv").resolve()
DEFAULT_EICU_TRAJECTORY_CSV = (DEFAULT_EICU_PROCESSED_DIR / "eicu_sepsis3_sofa_6h_trajectory.csv").resolve()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run temporal memory experiments on ICU JSON data or eICU Sepsis-3 CSV trajectories."
    )
    parser.add_argument("--json", default="")
    parser.add_argument("--dataset-format", choices=["legacy_json", "eicu_sepsis3"], default="legacy_json")
    parser.add_argument(
        "--eicu-sepsis3-labels-csv",
        default="../eicu数据库/processed/eicu_sepsis3_labels.csv",
    )
    parser.add_argument(
        "--eicu-sepsis3-trajectory-csv",
        default="../eicu数据库/processed/eicu_sepsis3_sofa_6h_trajectory.csv",
    )
    parser.add_argument("--task-type", choices=["patient", "window"], default="patient")
    parser.add_argument("--feature-mode", choices=["full", "stats", "prototype"], default="prototype")
    parser.add_argument(
        "--target-mode",
        choices=[
            "binary_worsening",
            "trajectory_shape",
            "trajectory_shape_balanced",
            "sepsis3_label",
            "sepsis3_label_baseline0",
            "septic_shock_label_operational",
            "septic_shock_label_relaxed",
        ],
        default="binary_worsening",
    )
    parser.add_argument("--model-family", choices=["legacy", "manifold"], default="legacy")
    parser.add_argument("--history-window", type=int, default=4)
    parser.add_argument("--forecast-horizon", type=int, default=3)
    parser.add_argument("--worsening-delta", type=float, default=2.0)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-csv", default="")
    parser.add_argument("--export-splits-csv", default="")

    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--sim-threshold", type=float, default=0.92)
    parser.add_argument("--merge-alpha", type=float, default=0.2)
    parser.add_argument("--decay", type=float, default=0.997)
    parser.add_argument("--forget-threshold", type=float, default=0.1)
    parser.add_argument("--max-memory", type=int, default=256)
    parser.add_argument("--base-epochs", type=int, default=4)
    parser.add_argument("--base-learning-rate", type=float, default=0.08)
    parser.add_argument("--base-l2", type=float, default=1e-4)
    parser.add_argument("--positive-weight-scale", type=float, default=1.0)
    parser.add_argument("--prototype-confidence", type=float, default=0.84)
    parser.add_argument("--memory-temperature", type=float, default=0.12)
    parser.add_argument("--correction-confidence", type=float, default=0.78)
    parser.add_argument("--uncertainty-low", type=float, default=0.2)
    parser.add_argument("--uncertainty-high", type=float, default=0.8)
    parser.add_argument("--correction-scale", type=float, default=0.45)
    parser.add_argument(
        "--optimize-for",
        choices=["f1_macro", "balanced_accuracy", "recall_priority", "clinical_warning"],
        default="f1_macro",
    )
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument("--min-precision", type=float, default=0.0)
    parser.add_argument("--min-specificity", type=float, default=0.0)
    parser.add_argument("--threshold-min", type=float, default=0.15)
    parser.add_argument("--threshold-max", type=float, default=0.85)

    parser.add_argument("--manifold-epochs", type=int, default=12)
    parser.add_argument("--manifold-batch-size", type=int, default=16)
    parser.add_argument("--manifold-learning-rate", type=float, default=1e-3)
    parser.add_argument("--manifold-weight-decay", type=float, default=1e-4)
    parser.add_argument("--manifold-aux-base-loss-weight", type=float, default=0.35)
    parser.add_argument("--manifold-align-loss-weight", type=float, default=0.05)
    parser.add_argument("--manifold-compact-loss-weight", type=float, default=0.03)
    parser.add_argument("--manifold-separation-loss-weight", type=float, default=0.03)
    parser.add_argument("--manifold-temporal-smoothness-weight", type=float, default=0.04)
    parser.add_argument("--manifold-separation-margin", type=float, default=1.0)
    parser.add_argument("--manifold-grad-clip", type=float, default=1.0)
    parser.add_argument("--manifold-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--manifold-dim", type=int, default=32)
    parser.add_argument("--manifold-value-dim", type=int, default=48)
    parser.add_argument("--manifold-fusion-hidden-dim", type=int, default=64)
    parser.add_argument("--gru-hidden-dim", type=int, default=64)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument("--gru-dropout", type=float, default=0.1)
    parser.add_argument("--gru-bidirectional", action="store_true", default=True)
    parser.add_argument("--static-hidden-dim", type=int, default=16)
    parser.add_argument("--memory-min-label", type=int, default=8)
    parser.add_argument("--memory-max-label", type=int, default=96)
    parser.add_argument("--memory-max-patient-label", type=int, default=3)
    parser.add_argument("--memory-support-penalty", type=float, default=0.04)
    parser.add_argument("--memory-collapse-penalty", type=float, default=0.06)
    parser.add_argument("--memory-rerank-strength", type=float, default=0.18)
    parser.add_argument("--memory-rerank-top-classes", type=int, default=2)
    parser.add_argument("--memory-rerank-candidates-per-class", type=int, default=4)
    parser.add_argument("--memory-confidence-floor", type=float, default=0.15)
    parser.add_argument("--memory-confidence-sharpness", type=float, default=12.0)
    parser.add_argument("--memory-confidence-margin-sharpness", type=float, default=20.0)
    parser.add_argument("--memory-uncertainty-floor", type=float, default=0.35)

    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    if args.eicu_sepsis3_labels_csv == parser.get_default("eicu_sepsis3_labels_csv"):
        args.eicu_sepsis3_labels_csv = str(DEFAULT_EICU_LABELS_CSV)
    if args.eicu_sepsis3_trajectory_csv == parser.get_default("eicu_sepsis3_trajectory_csv"):
        args.eicu_sepsis3_trajectory_csv = str(DEFAULT_EICU_TRAJECTORY_CSV)
    return args


def _select_base_threshold(model: DynamicMemoryClassifier, y_true: List[int], positive_probs: List[float]) -> Dict[str, float]:
    best_score = None
    best_stats = None
    grid_min = int(model.threshold_min * 100)
    grid_max = int(model.threshold_max * 100)
    for threshold_idx in range(grid_min, grid_max + 1):
        threshold = threshold_idx / 100.0
        stats = model._binary_stats(y_true, positive_probs, threshold)
        score = model._threshold_score(stats)
        if best_score is None or score > best_score:
            best_score = score
            best_stats = stats
    return best_stats


def _mean(values: List[float]) -> float:
    return sum(values) / max(1, len(values))


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_value = _mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _aggregate_metric_dict(dicts: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not dicts:
        return {}
    keys = dicts[0].keys()
    result: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = [float(item[key]) for item in dicts]
        result[key] = {"mean": _mean(values), "std": _std(values)}
    return result


def _export_dataset_csv(
    path: str,
    feature_names: List[str],
    dataset_rows: List[List[float]],
    labels: List[int],
    metadata: List[Dict[str, float]],
):
    metadata_keys = sorted({key for item in metadata for key in item.keys()})
    fieldnames = metadata_keys + list(feature_names) + ["target_label", "target_label_name", "target_worsened"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, label, meta in zip(dataset_rows, labels, metadata):
            record = {key: meta.get(key, "") for key in metadata_keys}
            record["target_label"] = label
            record["target_label_name"] = meta.get("trajectory_label", "")
            record["target_worsened"] = label
            record.update({name: value for name, value in zip(feature_names, row)})
            writer.writerow(record)


def _export_temporal_splits_csv(path: str, dataset, splits: List[Tuple[List[int], List[int], List[int]]]) -> None:
    metadata_keys = sorted({key for item in dataset.metadata for key in item.keys()})
    fieldnames = [
        "fold",
        "split",
        "row_index",
        "patient_id",
        "stay_id",
        "subject_id",
        "target_label",
        "target_label_name",
    ] + metadata_keys
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fold_id, (train_idx, val_idx, test_idx) in enumerate(splits, start=1):
            for split_name, indices in (
                ("train", train_idx),
                ("val", val_idx),
                ("test", test_idx),
            ):
                for row_index in indices:
                    meta = dataset.metadata[row_index]
                    label = dataset.y_data[row_index]
                    record = {
                        "fold": fold_id,
                        "split": split_name,
                        "row_index": row_index,
                        "patient_id": dataset.patient_ids[row_index],
                        "stay_id": meta.get("stay_id", dataset.patient_ids[row_index]),
                        "subject_id": meta.get("subject_id", ""),
                        "target_label": label,
                        "target_label_name": dataset.label_names[label] if label < len(dataset.label_names) else "",
                    }
                    record.update({key: meta.get(key, "") for key in metadata_keys})
                    writer.writerow(record)


def _build_split_strategy_summary(
    dataset,
    splits: List[Tuple[List[int], List[int], List[int]]],
    export_path: str,
    seed: int,
) -> Dict[str, object]:
    assignment_rows = sum(len(train_idx) + len(val_idx) + len(test_idx) for train_idx, val_idx, test_idx in splits)
    fold_summaries = []
    for fold_id, (train_idx, val_idx, test_idx) in enumerate(splits, start=1):
        fold_summaries.append(
            {
                "fold": fold_id,
                "train_rows": len(train_idx),
                "val_rows": len(val_idx),
                "test_rows": len(test_idx),
                "train_patients": len(set(dataset.patient_ids[idx] for idx in train_idx)),
                "val_patients": len(set(dataset.patient_ids[idx] for idx in val_idx)),
                "test_patients": len(set(dataset.patient_ids[idx] for idx in test_idx)),
            }
        )
    return {
        "method": "grouped_temporal_cv",
        "group_key": "patient_id",
        "folds": len(splits),
        "seed": seed,
        "assignment_rows": assignment_rows,
        "unique_patient_count": len(set(dataset.patient_ids)),
        "export_splits_csv": export_path,
        "fold_summaries": fold_summaries,
    }


def _metric_deltas(hybrid_metrics: Dict[str, float], base_metrics: Dict[str, float]) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    for key in hybrid_metrics:
        if key in base_metrics:
            deltas[f"delta_{key}"] = hybrid_metrics[key] - base_metrics[key]
    return deltas


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _ensure_parent_dir(path_str: str) -> None:
    if not path_str:
        return
    Path(path_str).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _resolved_path_str(path_str: str) -> str:
    if not path_str:
        return ""
    return str(Path(path_str).expanduser().resolve())


def _build_manifold_blueprint_summary(dataset, args) -> Dict[str, object]:
    if not dataset.sequence_data:
        return {"enabled": False}

    config = ManifoldMemoryConfig(
        sequence_feature_dim=len(dataset.sequence_feature_names),
        static_feature_dim=len(dataset.static_data[0]) if dataset.static_data else 0,
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
        min_label_memory=args.memory_min_label,
        max_label_memory=args.memory_max_label,
        max_patient_label_memory=args.memory_max_patient_label,
        support_penalty=args.memory_support_penalty,
        collapse_penalty=args.memory_collapse_penalty,
        rerank_strength=args.memory_rerank_strength,
        rerank_top_classes=args.memory_rerank_top_classes,
        rerank_candidates_per_class=args.memory_rerank_candidates_per_class,
        confidence_floor=args.memory_confidence_floor,
        confidence_sharpness=args.memory_confidence_sharpness,
        confidence_margin_sharpness=args.memory_confidence_margin_sharpness,
        uncertainty_floor=args.memory_uncertainty_floor,
        seed=args.seed,
        gru_hidden_dim=args.gru_hidden_dim,
        gru_layers=args.gru_layers,
        gru_bidirectional=args.gru_bidirectional,
        gru_dropout=args.gru_dropout,
        static_hidden_dim=args.static_hidden_dim,
        device=_resolve_device(args.manifold_device),
    )
    blueprint = ManifoldMemoryBlueprint(config)
    encoding, readout, fused = blueprint.forward(
        dataset.sequence_data[0],
        dataset.static_data[0] if dataset.static_data else None,
        metadata=dataset.metadata[0] if dataset.metadata else None,
    )
    return {
        "enabled": True,
        "config": {
            "encoder_type": config.encoder_type,
            "sequence_feature_dim": config.sequence_feature_dim,
            "static_feature_dim": config.static_feature_dim,
            "manifold_dim": config.manifold_dim,
            "value_dim": config.value_dim,
            "fusion_hidden_dim": config.fusion_hidden_dim,
            "gru_hidden_dim": config.gru_hidden_dim,
            "gru_layers": config.gru_layers,
            "gru_bidirectional": config.gru_bidirectional,
            "gru_dropout": config.gru_dropout,
            "static_hidden_dim": config.static_hidden_dim,
            "top_k": config.top_k,
            "temperature": config.temperature,
            "similarity_threshold": config.similarity_threshold,
            "merge_alpha": config.merge_alpha,
            "decay": config.decay,
            "forget_threshold": config.forget_threshold,
            "max_memory": config.max_memory,
            "min_label_memory": config.min_label_memory,
            "max_label_memory": config.max_label_memory,
            "max_patient_label_memory": config.max_patient_label_memory,
            "support_penalty": config.support_penalty,
            "collapse_penalty": config.collapse_penalty,
            "rerank_strength": config.rerank_strength,
            "rerank_top_classes": config.rerank_top_classes,
            "rerank_candidates_per_class": config.rerank_candidates_per_class,
            "confidence_floor": config.confidence_floor,
            "confidence_sharpness": config.confidence_sharpness,
            "confidence_margin_sharpness": config.confidence_margin_sharpness,
            "uncertainty_floor": config.uncertainty_floor,
            "device": config.device,
        },
        "sample_forward_shapes": {
            "sequence_length": len(dataset.sequence_data[0]),
            "sequence_feature_dim": len(dataset.sequence_data[0][0]) if dataset.sequence_data[0] else 0,
            "query_dim": len(encoding.query),
            "key_dim": len(encoding.key),
            "value_dim": len(encoding.value),
            "readout_dim": len(readout.readout),
            "fused_dim": len(fused),
        },
        "trainable_parameter_count": blueprint.parameter_count(),
    }


def _build_dataset(args):
    common_kwargs = {
        "json_path": args.json,
        "dataset_format": args.dataset_format,
        "eicu_sepsis3_labels_csv": args.eicu_sepsis3_labels_csv,
        "eicu_sepsis3_trajectory_csv": args.eicu_sepsis3_trajectory_csv,
        "worsening_delta": args.worsening_delta,
        "feature_mode": args.feature_mode,
        "target_mode": args.target_mode,
    }
    if args.task_type == "window":
        return build_window_forecast_dataset(
            history_window=args.history_window,
            forecast_horizon=args.forecast_horizon,
            **common_kwargs,
        )
    return build_patient_forecast_dataset(**common_kwargs)


def _target_definition_text(args) -> str:
    if args.target_mode == "binary_worsening":
        return f"max_future_sofa_total >= history_last_sofa_total + {args.worsening_delta}"
    if args.target_mode == "trajectory_shape":
        return "trajectory classes derived from future mean delta, last delta, and volatility"
    if args.target_mode == "trajectory_shape_balanced":
        return "balanced trajectory classes: improve, worsen, mixed"
    if args.target_mode == "sepsis3_label":
        return "explicit Sepsis-3 label from eICU suspected-infection anchor and SOFA delta"
    if args.target_mode == "sepsis3_label_baseline0":
        return "explicit Sepsis-3 label using baseline SOFA = 0 sensitivity analysis"
    if args.target_mode == "septic_shock_label_operational":
        return "operational septic shock label from eICU labels CSV"
    return "relaxed septic shock label from eICU labels CSV"


def _subset_list(data: Sequence, indices: Sequence[int]) -> List:
    return [data[idx] for idx in indices]


def _run_legacy_fold(args, dataset, fold_id: int, train_idx: List[int], val_idx: List[int], test_idx: List[int]):
    x_train, y_train, x_val, y_val, x_test, y_test = gather_standardized_split(
        dataset.x_data, dataset.y_data, train_idx, val_idx, test_idx
    )
    train_groups = [dataset.patient_ids[idx] for idx in train_idx]

    model = DynamicMemoryClassifier(
        top_k=args.top_k,
        sim_threshold=args.sim_threshold,
        merge_alpha=args.merge_alpha,
        decay=args.decay,
        forget_threshold=args.forget_threshold,
        max_memory=args.max_memory,
        base_epochs=args.base_epochs,
        base_learning_rate=args.base_learning_rate,
        base_l2=args.base_l2,
        positive_weight_scale=args.positive_weight_scale,
        prototype_confidence=args.prototype_confidence,
        memory_temperature=args.memory_temperature,
        correction_confidence=args.correction_confidence,
        uncertainty_low=args.uncertainty_low,
        uncertainty_high=args.uncertainty_high,
        correction_scale=args.correction_scale,
        optimize_for=args.optimize_for,
        min_recall=args.min_recall,
        min_precision=args.min_precision,
        min_specificity=args.min_specificity,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        seed=args.seed + fold_id,
    )
    model.fit(x_train, y_train, x_val, y_val, sample_groups=train_groups, label_names=dataset.label_names)

    val_prob = model.predict_proba(x_val)
    test_prob = model.predict_proba(x_test)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    val_metrics = basic_metrics(y_val, val_pred, val_prob)
    test_metrics = basic_metrics(y_test, test_pred, test_prob)

    base_val_prob = model.predict_base_proba(x_val)
    base_test_prob = model.predict_base_proba(x_test)
    if len(base_val_prob[0]) == 2:
        base_threshold_stats = _select_base_threshold(model, y_val, [prob[1] for prob in base_val_prob])
        base_threshold = base_threshold_stats["threshold"]
        base_val_pred = [1 if prob[1] >= base_threshold else 0 for prob in base_val_prob]
        base_test_pred = [1 if prob[1] >= base_threshold else 0 for prob in base_test_prob]
    else:
        base_threshold = None
        base_val_pred = [max(range(len(prob)), key=lambda idx: prob[idx]) for prob in base_val_prob]
        base_test_pred = [max(range(len(prob)), key=lambda idx: prob[idx]) for prob in base_test_prob]
    base_val_metrics = basic_metrics(y_val, base_val_pred, base_val_prob)
    base_test_metrics = basic_metrics(y_test, base_test_pred, base_test_prob)

    return {
        "decision_threshold": model.decision_threshold,
        "anchor_weight": model.anchor_weight,
        "correction_weight": model.correction_weight,
        "blend_weight": model.anchor_weight + model.correction_weight,
        "memory_size": len(model.memory),
        "threshold_selection_summary": model.threshold_selection_summary,
        "memory_diagnostics": model.memory_diagnostics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "base_only": {
            "decision_threshold": base_threshold,
            "val_metrics": base_val_metrics,
            "test_metrics": base_test_metrics,
        },
    }


def _run_manifold_fold(args, dataset, fold_id: int, train_idx: List[int], val_idx: List[int], test_idx: List[int]):
    train_sequences = _subset_list(dataset.sequence_data, train_idx)
    train_static = _subset_list(dataset.static_data, train_idx)
    train_labels = _subset_list(dataset.y_data, train_idx)
    train_metadata = _subset_list(dataset.metadata, train_idx)

    val_sequences = _subset_list(dataset.sequence_data, val_idx)
    val_static = _subset_list(dataset.static_data, val_idx)
    val_labels = _subset_list(dataset.y_data, val_idx)
    val_metadata = _subset_list(dataset.metadata, val_idx)

    test_sequences = _subset_list(dataset.sequence_data, test_idx)
    test_static = _subset_list(dataset.static_data, test_idx)
    test_labels = _subset_list(dataset.y_data, test_idx)
    test_metadata = _subset_list(dataset.metadata, test_idx)

    memory_config = ManifoldMemoryConfig(
        sequence_feature_dim=len(dataset.sequence_feature_names),
        static_feature_dim=len(train_static[0]) if train_static else 0,
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
        min_label_memory=args.memory_min_label,
        max_label_memory=args.memory_max_label,
        max_patient_label_memory=args.memory_max_patient_label,
        support_penalty=args.memory_support_penalty,
        collapse_penalty=args.memory_collapse_penalty,
        rerank_strength=args.memory_rerank_strength,
        rerank_top_classes=args.memory_rerank_top_classes,
        rerank_candidates_per_class=args.memory_rerank_candidates_per_class,
        confidence_floor=args.memory_confidence_floor,
        confidence_sharpness=args.memory_confidence_sharpness,
        confidence_margin_sharpness=args.memory_confidence_margin_sharpness,
        uncertainty_floor=args.memory_uncertainty_floor,
        seed=args.seed + fold_id,
        gru_hidden_dim=args.gru_hidden_dim,
        gru_layers=args.gru_layers,
        gru_bidirectional=args.gru_bidirectional,
        gru_dropout=args.gru_dropout,
        static_hidden_dim=args.static_hidden_dim,
        device=_resolve_device(args.manifold_device),
    )
    trainer_config = ManifoldTrainerConfig(
        epochs=args.manifold_epochs,
        batch_size=args.manifold_batch_size,
        learning_rate=args.manifold_learning_rate,
        weight_decay=args.manifold_weight_decay,
        aux_base_loss_weight=args.manifold_aux_base_loss_weight,
        align_loss_weight=args.manifold_align_loss_weight,
        compact_loss_weight=args.manifold_compact_loss_weight,
        separation_loss_weight=args.manifold_separation_loss_weight,
        temporal_smoothness_weight=args.manifold_temporal_smoothness_weight,
        separation_margin=args.manifold_separation_margin,
        grad_clip=args.manifold_grad_clip,
        device=_resolve_device(args.manifold_device),
        seed=args.seed + fold_id,
    )
    trainer = EndToEndManifoldTrainer(
        memory_config=memory_config,
        trainer_config=trainer_config,
        num_classes=len(dataset.label_names),
        label_names=dataset.label_names,
    )
    trainer.fit(
        train_sequences=train_sequences,
        train_static=train_static,
        y_train=train_labels,
        train_metadata=train_metadata,
        val_sequences=val_sequences,
        val_static=val_static,
        y_val=val_labels,
        val_metadata=val_metadata,
    )

    val_prob = trainer.predict_proba(val_sequences, val_static, val_metadata)
    test_prob = trainer.predict_proba(test_sequences, test_static, test_metadata)
    val_pred = [max(range(len(prob)), key=lambda idx: prob[idx]) for prob in val_prob]
    test_pred = [max(range(len(prob)), key=lambda idx: prob[idx]) for prob in test_prob]
    val_metrics = basic_metrics(val_labels, val_pred, val_prob)
    test_metrics = basic_metrics(test_labels, test_pred, test_prob)

    base_val_prob = trainer.predict_base_proba(val_sequences, val_static, val_metadata)
    base_test_prob = trainer.predict_base_proba(test_sequences, test_static, test_metadata)
    base_val_pred = [max(range(len(prob)), key=lambda idx: prob[idx]) for prob in base_val_prob]
    base_test_pred = [max(range(len(prob)), key=lambda idx: prob[idx]) for prob in base_test_prob]
    base_val_metrics = basic_metrics(val_labels, base_val_pred, base_val_prob)
    base_test_metrics = basic_metrics(test_labels, base_test_pred, base_test_prob)

    return {
        "decision_threshold": None,
        "anchor_weight": 0.0,
        "correction_weight": 1.0,
        "blend_weight": 1.0,
        "memory_size": len(trainer.manifold.memory_bank),
        "threshold_selection_summary": trainer.training_summary,
        "memory_diagnostics": trainer.memory_diagnostics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "base_only": {
            "decision_threshold": None,
            "val_metrics": base_val_metrics,
            "test_metrics": base_test_metrics,
        },
    }


def main():
    args = parse_args()
    if args.dataset_format == "legacy_json" and not args.json:
        raise ValueError("legacy_json mode requires --json.")
    if args.dataset_format == "eicu_sepsis3":
        args.eicu_sepsis3_labels_csv = _resolved_path_str(args.eicu_sepsis3_labels_csv)
        args.eicu_sepsis3_trajectory_csv = _resolved_path_str(args.eicu_sepsis3_trajectory_csv)
    args.json = _resolved_path_str(args.json) if args.json else ""
    _ensure_parent_dir(args.output_json)
    _ensure_parent_dir(args.export_csv)
    _ensure_parent_dir(args.export_splits_csv)

    dataset = _build_dataset(args)
    if args.export_csv:
        _export_dataset_csv(args.export_csv, dataset.feature_names, dataset.x_data, dataset.y_data, dataset.metadata)

    splits = grouped_temporal_cv_splits(dataset.patient_ids, args.folds, args.seed)
    if args.export_splits_csv:
        _export_temporal_splits_csv(args.export_splits_csv, dataset, splits)

    fold_results = []
    hybrid_metrics_list = []
    base_metrics_list = []
    memory_effects = []
    diagnostics_list = []

    for fold_id, (train_idx, val_idx, test_idx) in enumerate(splits, start=1):
        if args.model_family == "manifold":
            fold_payload = _run_manifold_fold(args, dataset, fold_id, train_idx, val_idx, test_idx)
        else:
            fold_payload = _run_legacy_fold(args, dataset, fold_id, train_idx, val_idx, test_idx)

        memory_effect = _metric_deltas(fold_payload["test_metrics"], fold_payload["base_only"]["test_metrics"])

        fold_result = {
            "fold": fold_id,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "train_patients": len(set(dataset.patient_ids[idx] for idx in train_idx)),
            "val_patients": len(set(dataset.patient_ids[idx] for idx in val_idx)),
            "test_patients": len(set(dataset.patient_ids[idx] for idx in test_idx)),
            **fold_payload,
            "memory_effectiveness": memory_effect,
        }
        fold_results.append(fold_result)
        hybrid_metrics_list.append(fold_payload["test_metrics"])
        base_metrics_list.append(fold_payload["base_only"]["test_metrics"])
        memory_effects.append(memory_effect)
        diagnostics_list.append(
            {
                "blend_weight": float(fold_payload["blend_weight"]),
                "anchor_weight": float(fold_payload["anchor_weight"]),
                "correction_weight": float(fold_payload["correction_weight"]),
                **{key: float(value) for key, value in fold_payload["memory_diagnostics"].items()},
            }
        )

    unique_patients = len(set(dataset.patient_ids))
    label_counter = Counter(dataset.y_data)
    dataset_summary = {
        "model_family": args.model_family,
        "dataset_format": args.dataset_format,
        "task_type": args.task_type,
        "feature_mode": args.feature_mode,
        "target_mode": args.target_mode,
        "label_names": dataset.label_names,
        "sample_count": len(dataset.x_data),
        "patient_count": unique_patients,
        "feature_count": len(dataset.feature_names),
        "class_count": len(dataset.label_names),
        "class_distribution": {
            dataset.label_names[class_id]: label_counter.get(class_id, 0) for class_id in range(len(dataset.label_names))
        },
        "history_window": args.history_window if args.task_type == "window" else 7,
        "forecast_horizon": args.forecast_horizon if args.task_type == "window" else 13,
        "target_definition": _target_definition_text(args),
        "input_sources": {
            "json": args.json,
            "eicu_sepsis3_labels_csv": args.eicu_sepsis3_labels_csv if args.dataset_format == "eicu_sepsis3" else "",
            "eicu_sepsis3_trajectory_csv": args.eicu_sepsis3_trajectory_csv
            if args.dataset_format == "eicu_sepsis3"
            else "",
        },
        "split_strategy": _build_split_strategy_summary(
            dataset=dataset,
            splits=splits,
            export_path=args.export_splits_csv,
            seed=args.seed,
        ),
        "manifold_blueprint": _build_manifold_blueprint_summary(dataset, args),
    }
    if args.model_family == "manifold":
        dataset_summary["manifold_training"] = {
            "epochs": args.manifold_epochs,
            "batch_size": args.manifold_batch_size,
            "learning_rate": args.manifold_learning_rate,
            "weight_decay": args.manifold_weight_decay,
            "aux_base_loss_weight": args.manifold_aux_base_loss_weight,
            "align_loss_weight": args.manifold_align_loss_weight,
            "compact_loss_weight": args.manifold_compact_loss_weight,
            "separation_loss_weight": args.manifold_separation_loss_weight,
            "temporal_smoothness_weight": args.manifold_temporal_smoothness_weight,
            "separation_margin": args.manifold_separation_margin,
            "grad_clip": args.manifold_grad_clip,
            "device": _resolve_device(args.manifold_device),
            "memory_min_label": args.memory_min_label,
            "memory_max_label": args.memory_max_label,
            "memory_max_patient_label": args.memory_max_patient_label,
            "memory_support_penalty": args.memory_support_penalty,
            "memory_collapse_penalty": args.memory_collapse_penalty,
            "memory_rerank_strength": args.memory_rerank_strength,
            "memory_rerank_top_classes": args.memory_rerank_top_classes,
            "memory_rerank_candidates_per_class": args.memory_rerank_candidates_per_class,
            "memory_confidence_floor": args.memory_confidence_floor,
            "memory_confidence_sharpness": args.memory_confidence_sharpness,
            "memory_confidence_margin_sharpness": args.memory_confidence_margin_sharpness,
            "memory_uncertainty_floor": args.memory_uncertainty_floor,
        }
    if args.target_mode == "binary_worsening" and len(dataset.label_names) == 2:
        dataset_summary["positive_count"] = label_counter.get(1, 0)
        dataset_summary["positive_rate"] = label_counter.get(1, 0) / max(1, len(dataset.y_data))

    result = {
        "dataset_summary": dataset_summary,
        "cv_folds": args.folds,
        "fold_results": fold_results,
        "aggregate_test_metrics": _aggregate_metric_dict(hybrid_metrics_list),
        "aggregate_base_metrics": _aggregate_metric_dict(base_metrics_list),
        "aggregate_memory_effectiveness": _aggregate_metric_dict(memory_effects),
        "aggregate_memory_diagnostics": _aggregate_metric_dict(diagnostics_list),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
