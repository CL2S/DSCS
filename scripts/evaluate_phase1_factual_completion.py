import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from run_forecasting_experiment import (
    DEFAULT_EICU_KG_DIR,
    DEFAULT_EICU_LABELS_CSV,
    DEFAULT_EICU_TRAJECTORY_CSV,
)
from src.manifold_forecasting_trainer import EndToEndForecastingManifoldTrainer, ForecastingTrainerConfig
from src.manifold_memory import ManifoldMemoryConfig
from src.tsf_data import ForecastSample, build_eicu_sepsis3_forecasting_dataset
from src.ts_formation import PATTERN_LABELS, TRAJECTORY_LABELS


def _build_trainer(dataset, final_mode: bool) -> EndToEndForecastingManifoldTrainer:
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
        encoder_type="gru",
        gru_hidden_dim=64,
        gru_layers=1,
        gru_bidirectional=True,
        gru_dropout=0.1,
        static_hidden_dim=16,
        device="cpu",
    )
    trainer_config = ForecastingTrainerConfig(
        forecast_horizon=dataset.forecast_horizon,
        seasonality=dataset.seasonality,
        history_length=dataset.history_length,
        series_count=dataset.series_count,
        dataset_name=dataset.dataset_name,
        epochs=3,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=1e-4,
        aux_base_loss_weight=0.3,
        multitask_loss_weight=0.15,
        align_loss_weight=0.02,
        temporal_smoothness_weight=0.02,
        grad_clip=1.0,
        device="cpu",
        seed=42,
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
    if final_mode:
        trainer.branch_decoupling_mode = "pre_projection"
        trainer.factual_prediction_trunk_scale = 0.08
        trainer.factual_calibration_weight = 0.05
    else:
        trainer.branch_decoupling_mode = "post_projection"
        trainer.factual_prediction_trunk_scale = 0.0
        trainer.factual_calibration_weight = 0.0
    return trainer


def _subset(samples: Sequence[ForecastSample], predicate: Callable[[ForecastSample], bool]) -> List[ForecastSample]:
    return [sample for sample in samples if predicate(sample)]


def _metrics_for_subset(trainer: EndToEndForecastingManifoldTrainer, samples: Sequence[ForecastSample]) -> Dict[str, object]:
    if not samples:
        return {"count": 0.0}
    return {
        "count": float(len(samples)),
        "factual_only": trainer.evaluate(samples, use_memory=False),
        "with_memory": trainer.evaluate(samples, use_memory=True),
        "calibration": trainer.evaluate_factual_calibration(samples),
    }


def _improvement(final_row: Dict[str, object], baseline_row: Dict[str, object]) -> Dict[str, float]:
    if not baseline_row or float(baseline_row.get("count", 0.0)) <= 0.0:
        return {}
    return {
        "factual_mae_delta": float(final_row["factual_only"]["mae"] - baseline_row["factual_only"]["mae"]),
        "factual_rmse_delta": float(final_row["factual_only"]["rmse"] - baseline_row["factual_only"]["rmse"]),
        "memory_mae_delta": float(final_row["with_memory"]["mae"] - baseline_row["with_memory"]["mae"]),
        "calibration_gap_delta": float(
            final_row["calibration"]["calibration_gap"] - baseline_row["calibration"]["calibration_gap"]
        ),
        "coverage_95_delta": float(
            final_row["calibration"]["interval_95_coverage"] - baseline_row["calibration"]["interval_95_coverage"]
        ),
    }


def main() -> None:
    dataset = build_eicu_sepsis3_forecasting_dataset(
        labels_csv=str(DEFAULT_EICU_LABELS_CSV),
        trajectory_csv=str(DEFAULT_EICU_TRAJECTORY_CSV),
        dataset_name="phase1_completion_eval",
        history_length=4,
        forecast_horizon=2,
        max_series_count=64,
        enable_kg=True,
        kg_directory=str(DEFAULT_EICU_KG_DIR),
        append_kg_to_patient_static=True,
    )

    formation_index = {name: idx for idx, name in enumerate(dataset.formation_feature_names)}
    regime_mix_idx = formation_index["regime_mix_score"]
    regime_values = sorted(float(sample.formation_features[regime_mix_idx]) for sample in dataset.val_samples)
    regime_threshold = regime_values[max(0, int(0.65 * max(0, len(regime_values) - 1)))] if regime_values else 0.0
    spike_label = PATTERN_LABELS.index("spike")
    shifted_label = TRAJECTORY_LABELS.index("shifted_regime")

    subsets = {
        "all_val": list(dataset.val_samples),
        "high_volatility": _subset(dataset.val_samples, lambda sample: float(sample.formation_features[regime_mix_idx]) >= regime_threshold),
        "phase_switch": _subset(
            dataset.val_samples,
            lambda sample: int(sample.pattern_label) == spike_label or int(sample.trajectory_label) == shifted_label,
        ),
        "vasopressor_related": _subset(
            dataset.val_samples,
            lambda sample: (
                float(dict(sample.aux_targets or {}).get("future_vasopressor_need", 0.0)) > 0.0
                or float(dict(sample.metadata.get("kg_flags", {})).get("kg_state_hypotension", 0.0)) > 0.0
                or float(dict(sample.metadata.get("kg_flags", {})).get("kg_treat_vasopressor", 0.0)) > 0.0
            ),
        ),
    }

    baseline_trainer = _build_trainer(dataset, final_mode=False)
    baseline_trainer.fit(dataset.train_samples, dataset.val_samples, memory_seed_samples=dataset.train_samples, collect_diagnostics=False)

    final_trainer = _build_trainer(dataset, final_mode=True)
    final_trainer.fit(dataset.train_samples, dataset.val_samples, memory_seed_samples=dataset.train_samples, collect_diagnostics=False)

    baseline_results = {name: _metrics_for_subset(baseline_trainer, subset_samples) for name, subset_samples in subsets.items()}
    final_results = {name: _metrics_for_subset(final_trainer, subset_samples) for name, subset_samples in subsets.items()}

    comparison = {
        name: _improvement(final_results[name], baseline_results[name])
        for name in subsets
    }
    output = {
        "dataset_summary": {
            "dataset_name": dataset.dataset_name,
            "train_count": float(len(dataset.train_samples)),
            "val_count": float(len(dataset.val_samples)),
            "test_count": float(len(dataset.test_samples)),
            "regime_mix_threshold": float(regime_threshold),
        },
        "baseline": baseline_results,
        "final": final_results,
        "comparison": comparison,
        "final_gate_probe": final_trainer.inspect_sample(dataset.test_samples[0]).get("gates", {}),
    }

    output_path = Path("memory_mvp_project/output/analysis/phase1_completion_evaluation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
if __name__ == "__main__":
    main()
