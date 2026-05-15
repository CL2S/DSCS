import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.manifold_forecasting_trainer import EndToEndForecastingManifoldTrainer, ForecastingTrainerConfig
from src.manifold_memory import ManifoldMemoryConfig
from src.ts_formation import PATTERN_LABELS, TRAJECTORY_LABELS
from src.tsf_data import ForecastSample, build_eicu_sepsis3_forecasting_dataset


DEFAULT_LABELS_CSV = PROJECT_ROOT.parent / "eicu数据库" / "processed" / "eicu_sepsis3_labels.csv"
DEFAULT_TRAJECTORY_CSV = PROJECT_ROOT.parent / "eicu数据库" / "processed" / "eicu_sepsis3_sofa_6h_trajectory.csv"
DEFAULT_KG_DIR = PROJECT_ROOT / "input" / "knowledge" / "13_processed_ready" / "sepsis_kg_guideline_enhanced"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "analysis" / "mem_mod_round20" / "donor_case_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export case-wise donor comparison between legacy and structured tuned counterfactual scoring.")
    parser.add_argument("--labels-csv", default=str(DEFAULT_LABELS_CSV))
    parser.add_argument("--trajectory-csv", default=str(DEFAULT_TRAJECTORY_CSV))
    parser.add_argument("--kg-directory", default=str(DEFAULT_KG_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--forecast-horizon", type=int, default=2)
    parser.add_argument("--max-series", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _mean(values: List[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _kg_flag(flags: Dict[str, object], name: str) -> float:
    if not flags:
        return 0.0
    candidates = [name, f"kg_{name}" if not name.startswith("kg_") else name[3:]]
    for candidate in candidates:
        if candidate in flags:
            try:
                return float(flags.get(candidate, 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _label_name(index: int, labels: List[str]) -> str:
    if 0 <= int(index) < len(labels):
        return labels[int(index)]
    return "unknown"


def build_trainer(dataset, args: argparse.Namespace) -> EndToEndForecastingManifoldTrainer:
    memory_config = ManifoldMemoryConfig(
        sequence_feature_dim=len(dataset.sequence_feature_names),
        static_feature_dim=len(dataset.patient_feature_names or dataset.static_feature_names),
        manifold_dim=32,
        value_dim=48,
        fusion_hidden_dim=64,
        top_k=6,
        temperature=0.15,
        similarity_threshold=0.92,
        merge_alpha=0.2,
        decay=0.997,
        forget_threshold=0.08,
        max_memory=256,
        same_label_merge_only=False,
        min_label_memory=1,
        max_label_memory=256,
        max_patient_label_memory=3,
        support_penalty=0.02,
        collapse_penalty=0.03,
        encoder_type="transformer",
        transformer_d_model=96,
        transformer_layers=2,
        transformer_heads=4,
        transformer_ff_dim=192,
        transformer_dropout=0.1,
        transformer_max_length=256,
        static_hidden_dim=16,
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
        learning_rate=1e-3,
        weight_decay=1e-4,
        aux_base_loss_weight=0.3,
        align_loss_weight=0.02,
        temporal_smoothness_weight=0.02,
        grad_clip=1.0,
        device=_resolve_device(args.device),
        seed=args.seed,
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
    trainer.memory_direct_residual_weight = 0.1
    trainer.memory_direct_residual_mode = "adaptive"
    trainer.memory_path_coordination_mode = "sum"
    trainer.intervention_feature_names = list(dataset.intervention_feature_names or [])
    trainer.intervention_sequence_feature_names = list(dataset.intervention_sequence_feature_names or [])
    return trainer


def configure_mode(trainer: EndToEndForecastingManifoldTrainer, mode_name: str) -> None:
    if mode_name == "legacy":
        trainer.counterfactual_donor_score_mode = "legacy"
        trainer.counterfactual_base_similarity_weight = 1.0
        trainer.counterfactual_kg_similarity_weight = 0.25
        trainer.counterfactual_guideline_weight = 0.20
        trainer.counterfactual_guideline_score_weight = 0.05
        trainer.counterfactual_penalty_weight = 0.20
        return
    if mode_name == "structured_tuned":
        trainer.counterfactual_donor_score_mode = "structured"
        trainer.counterfactual_base_similarity_weight = 1.0
        trainer.counterfactual_kg_similarity_weight = 0.30
        trainer.counterfactual_guideline_weight = 0.20
        trainer.counterfactual_guideline_score_weight = 0.05
        trainer.counterfactual_penalty_weight = 0.18
        return
    raise ValueError(f"Unsupported mode: {mode_name}")


def collect_mode_rows(
    trainer: EndToEndForecastingManifoldTrainer,
    samples: List[ForecastSample],
    mode_name: str,
) -> List[Dict[str, object]]:
    configure_mode(trainer, mode_name)
    factual_predictions = trainer.predict(samples, use_memory=True)
    rows: List[Dict[str, object]] = []

    trainer.eval()
    with torch.no_grad():
        batch_size = max(1, trainer.trainer_config.batch_size)
        for start in range(0, len(samples), batch_size):
            batch_samples = list(samples[start : start + batch_size])
            batch_factual = factual_predictions[start : start + len(batch_samples)]
            encodings, manager_results, _, _, _, _ = trainer._forward_batch(batch_samples)
            counterfactual_batch: List[ForecastSample] = []
            donor_metadata_batch: List[Dict[str, object]] = []
            for sample, encoding, manager_result in zip(batch_samples, encodings, manager_results):
                donor_intervention, donor_intervention_sequence, donor_metadata = trainer._counterfactual_donor_from_manager_result(
                    sample,
                    encoding,
                    manager_result,
                )
                counterfactual_batch.append(
                    replace(
                        sample,
                        intervention_static=donor_intervention,
                        intervention_sequence=donor_intervention_sequence,
                    )
                )
                donor_metadata_batch.append(donor_metadata)
            _, _, _, counterfactual_predictions, _, _ = trainer._forward_batch(counterfactual_batch)

            for sample, factual_prediction, counterfactual_prediction, donor_metadata in zip(
                batch_samples,
                batch_factual,
                counterfactual_predictions.detach().cpu().tolist(),
                donor_metadata_batch,
            ):
                restored_counterfactual = [
                    value * sample.scale_value + sample.scale_center
                    for value in counterfactual_prediction
                ]
                kg_flags = dict(sample.metadata.get("kg_flags", {}))
                rows.append(
                    {
                        "sample_id": str(sample.metadata.get("experience_id", f"{sample.metadata.get('series_name', 'unknown')}::{sample.metadata.get('window_end_index', -1)}")),
                        "stay_id": float(sample.metadata.get("stay_id", -1.0)),
                        "series_name": str(sample.metadata.get("series_name", "")),
                        "window_end_index": float(sample.metadata.get("window_end_index", -1.0)),
                        "pattern_label": int(sample.pattern_label),
                        "pattern_name": _label_name(int(sample.pattern_label), PATTERN_LABELS),
                        "trajectory_label": int(sample.trajectory_label),
                        "trajectory_name": _label_name(int(sample.trajectory_label), TRAJECTORY_LABELS),
                        "experience_label": int(sample.experience_label),
                        "query_state_sepsis": _kg_flag(kg_flags, "state_sepsis"),
                        "query_state_septic_shock": _kg_flag(kg_flags, "state_septic_shock"),
                        "query_state_organ_dysfunction": _kg_flag(kg_flags, "state_organ_dysfunction"),
                        "query_state_hypotension": _kg_flag(kg_flags, "state_hypotension"),
                        "query_state_high_lactate": _kg_flag(kg_flags, "state_high_lactate"),
                        "factual_prediction_mean": _mean([float(value) for value in factual_prediction]),
                        "counterfactual_prediction_mean": _mean(restored_counterfactual),
                        "predicted_delta": _mean([float(value) for value in factual_prediction]) - _mean(restored_counterfactual),
                        "donor_stay_id": float(donor_metadata.get("stay_id", -1.0)),
                        "donor_experience_id": str(donor_metadata.get("donor_experience_id", "")),
                        "donor_experience_label": int(float(donor_metadata.get("donor_experience_label", -1.0))),
                        "donor_pattern_label": int(float(donor_metadata.get("donor_pattern_label", -1.0))),
                        "donor_trajectory_label": int(float(donor_metadata.get("donor_trajectory_label", -1.0))),
                        "donor_similarity": float(donor_metadata.get("donor_similarity", 0.0)),
                        "donor_kg_similarity": float(donor_metadata.get("donor_kg_similarity", 0.0)),
                        "donor_guideline_compatibility": float(donor_metadata.get("donor_guideline_compatibility", 0.0)),
                        "donor_state_match": float(donor_metadata.get("donor_state_match", 0.0)),
                        "donor_missing_care_penalty": float(donor_metadata.get("donor_missing_care_penalty", 0.0)),
                        "donor_contraindication_penalty": float(donor_metadata.get("donor_contraindication_penalty", 0.0)),
                        "donor_total_score": float(donor_metadata.get("donor_total_score", 0.0)),
                    }
                )
    return rows


def compare_rows(
    legacy_rows: List[Dict[str, object]],
    structured_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    legacy_map = {row["sample_id"]: row for row in legacy_rows}
    structured_map = {row["sample_id"]: row for row in structured_rows}
    merged: List[Dict[str, object]] = []
    sample_ids = [row["sample_id"] for row in legacy_rows if row["sample_id"] in structured_map]

    for sample_id in sample_ids:
        legacy = legacy_map[sample_id]
        structured = structured_map[sample_id]
        donor_changed = (
            legacy["donor_stay_id"] != structured["donor_stay_id"]
            or legacy["donor_experience_id"] != structured["donor_experience_id"]
        )
        structured_clinical_gain = (
            (structured["donor_guideline_compatibility"] - legacy["donor_guideline_compatibility"])
            + (structured["donor_state_match"] - legacy["donor_state_match"])
            + (legacy["donor_missing_care_penalty"] - structured["donor_missing_care_penalty"])
            + (legacy["donor_contraindication_penalty"] - structured["donor_contraindication_penalty"])
        )
        legacy_effect_gain = legacy["predicted_delta"] - structured["predicted_delta"]
        merged.append(
            {
                "sample_id": sample_id,
                "stay_id": legacy["stay_id"],
                "series_name": legacy["series_name"],
                "window_end_index": legacy["window_end_index"],
                "pattern_label": legacy["pattern_label"],
                "pattern_name": legacy["pattern_name"],
                "trajectory_label": legacy["trajectory_label"],
                "trajectory_name": legacy["trajectory_name"],
                "experience_label": legacy["experience_label"],
                "query_state_sepsis": legacy["query_state_sepsis"],
                "query_state_septic_shock": legacy["query_state_septic_shock"],
                "query_state_organ_dysfunction": legacy["query_state_organ_dysfunction"],
                "query_state_hypotension": legacy["query_state_hypotension"],
                "query_state_high_lactate": legacy["query_state_high_lactate"],
                "factual_prediction_mean": legacy["factual_prediction_mean"],
                "legacy_donor_stay_id": legacy["donor_stay_id"],
                "legacy_donor_experience_id": legacy["donor_experience_id"],
                "legacy_donor_experience_label": legacy["donor_experience_label"],
                "legacy_donor_pattern_label": legacy["donor_pattern_label"],
                "legacy_donor_trajectory_label": legacy["donor_trajectory_label"],
                "legacy_donor_similarity": legacy["donor_similarity"],
                "legacy_donor_kg_similarity": legacy["donor_kg_similarity"],
                "legacy_donor_guideline_compatibility": legacy["donor_guideline_compatibility"],
                "legacy_donor_state_match": legacy["donor_state_match"],
                "legacy_donor_missing_care_penalty": legacy["donor_missing_care_penalty"],
                "legacy_donor_contraindication_penalty": legacy["donor_contraindication_penalty"],
                "legacy_donor_total_score": legacy["donor_total_score"],
                "legacy_counterfactual_prediction_mean": legacy["counterfactual_prediction_mean"],
                "legacy_predicted_delta": legacy["predicted_delta"],
                "structured_tuned_donor_stay_id": structured["donor_stay_id"],
                "structured_tuned_donor_experience_id": structured["donor_experience_id"],
                "structured_tuned_donor_experience_label": structured["donor_experience_label"],
                "structured_tuned_donor_pattern_label": structured["donor_pattern_label"],
                "structured_tuned_donor_trajectory_label": structured["donor_trajectory_label"],
                "structured_tuned_donor_similarity": structured["donor_similarity"],
                "structured_tuned_donor_kg_similarity": structured["donor_kg_similarity"],
                "structured_tuned_donor_guideline_compatibility": structured["donor_guideline_compatibility"],
                "structured_tuned_donor_state_match": structured["donor_state_match"],
                "structured_tuned_donor_missing_care_penalty": structured["donor_missing_care_penalty"],
                "structured_tuned_donor_contraindication_penalty": structured["donor_contraindication_penalty"],
                "structured_tuned_donor_total_score": structured["donor_total_score"],
                "structured_tuned_counterfactual_prediction_mean": structured["counterfactual_prediction_mean"],
                "structured_tuned_predicted_delta": structured["predicted_delta"],
                "donor_changed": int(donor_changed),
                "structured_clinical_gain_score": float(structured_clinical_gain),
                "legacy_effect_gain_score": float(legacy_effect_gain),
            }
        )
    return merged


def _write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_spotlight(rows: List[Dict[str, object]], path: Path) -> None:
    changed_rows = [row for row in rows if int(row["donor_changed"]) == 1]
    changed_rows.sort(
        key=lambda row: (
            row["structured_clinical_gain_score"],
            abs(row["legacy_effect_gain_score"]),
        ),
        reverse=True,
    )
    spotlight = changed_rows[:20]
    lines = [
        "# Legacy vs Structured Tuned Donor 逐例聚焦表",
        "",
        "## 怎么看这张表",
        "",
        "- 先看 `query_state_*`，确认当前病例是否处在 sepsis / shock / hypotension / high_lactate 等状态。",
        "- 再看 `legacy_donor_*` 和 `structured_tuned_donor_*` 两组 donor 是否发生变化。",
        "- 若 `structured_tuned_donor_guideline_compatibility` 更高、`structured_tuned_donor_state_match` 更高、两个 penalty 更低，说明 structured donor 在规则层面更临床一致。",
        "- 若 `legacy_predicted_delta` 更大，说明 legacy donor 在当前模型下给出的反事实改善 proxy 更强。",
        "- `structured_clinical_gain_score` 是临床一致性增益汇总分，只用于排序，不是正式评价指标。",
        "- `legacy_effect_gain_score > 0` 表示 legacy 的效果 proxy 更强；`< 0` 表示 structured tuned 更强。",
        "",
        "## 聚焦病例",
        "",
        "| stay_id | exp_label | legacy donor | structured donor | legacy guideline | structured guideline | legacy state | structured state | legacy penalty | structured penalty | legacy delta | structured delta | clinical gain | effect gain |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in spotlight:
        legacy_penalty = float(row["legacy_donor_missing_care_penalty"]) + float(row["legacy_donor_contraindication_penalty"])
        structured_penalty = float(row["structured_tuned_donor_missing_care_penalty"]) + float(row["structured_tuned_donor_contraindication_penalty"])
        lines.append(
            f"| {int(row['stay_id'])} | {int(row['experience_label'])} | {int(row['legacy_donor_stay_id'])} | {int(row['structured_tuned_donor_stay_id'])} | "
            f"{row['legacy_donor_guideline_compatibility']:.3f} | {row['structured_tuned_donor_guideline_compatibility']:.3f} | "
            f"{row['legacy_donor_state_match']:.3f} | {row['structured_tuned_donor_state_match']:.3f} | "
            f"{legacy_penalty:.3f} | {structured_penalty:.3f} | "
            f"{row['legacy_predicted_delta']:.3f} | {row['structured_tuned_predicted_delta']:.3f} | "
            f"{row['structured_clinical_gain_score']:.3f} | {row['legacy_effect_gain_score']:.3f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_summary(rows: List[Dict[str, object]], path: Path) -> None:
    donor_changed = sum(int(row["donor_changed"]) for row in rows)
    structured_clinically_better = sum(1 for row in rows if float(row["structured_clinical_gain_score"]) > 0.0)
    legacy_effect_better = sum(1 for row in rows if float(row["legacy_effect_gain_score"]) > 0.0)
    summary = {
        "case_count": len(rows),
        "donor_changed_count": donor_changed,
        "donor_changed_rate": donor_changed / max(1, len(rows)),
        "structured_clinically_better_count": structured_clinically_better,
        "structured_clinically_better_rate": structured_clinically_better / max(1, len(rows)),
        "legacy_effect_proxy_better_count": legacy_effect_better,
        "legacy_effect_proxy_better_rate": legacy_effect_better / max(1, len(rows)),
        "mean_structured_clinical_gain_score": _mean([float(row["structured_clinical_gain_score"]) for row in rows]),
        "mean_legacy_effect_gain_score": _mean([float(row["legacy_effect_gain_score"]) for row in rows]),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_eicu_sepsis3_forecasting_dataset(
        labels_csv=str(Path(args.labels_csv).resolve()),
        trajectory_csv=str(Path(args.trajectory_csv).resolve()),
        dataset_name="eicu_sepsis3_donor_case_comparison",
        history_length=args.history_length,
        forecast_horizon=args.forecast_horizon,
        target_field="total_sofa",
        max_series_count=args.max_series,
        enable_kg=True,
        kg_directory=str(Path(args.kg_directory).resolve()),
    )
    trainer = build_trainer(dataset, args)
    trainer.fit(
        dataset.train_samples,
        dataset.val_samples,
        memory_seed_samples=list(dataset.train_samples),
        collect_diagnostics=False,
    )

    legacy_rows = collect_mode_rows(trainer, dataset.test_samples, mode_name="legacy")
    structured_rows = collect_mode_rows(trainer, dataset.test_samples, mode_name="structured_tuned")
    comparison_rows = compare_rows(legacy_rows, structured_rows)

    csv_path = output_dir / "legacy_vs_structured_tuned_case_comparison.csv"
    spotlight_path = output_dir / "legacy_vs_structured_tuned_case_spotlight.md"
    summary_path = output_dir / "legacy_vs_structured_tuned_case_summary.json"

    _write_csv(comparison_rows, csv_path)
    _write_markdown_spotlight(comparison_rows, spotlight_path)
    _write_summary(comparison_rows, summary_path)

    print(json.dumps(
        {
            "output_dir": str(output_dir),
            "csv_path": str(csv_path),
            "spotlight_path": str(spotlight_path),
            "summary_path": str(summary_path),
            "case_count": len(comparison_rows),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
