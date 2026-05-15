import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from src.manifold_memory import ManifoldMemoryConfig, TorchManifoldEncodingOutput
from src.memory_components import ComponentReadResult, ExperienceMemory, PatternMemory, TrajectoryMemory
from src.persistent_memory_store import PersistentExperienceStore, SemanticPrototypeHit
from src.ts_formation import PATTERN_LABELS, TRAJECTORY_LABELS


@dataclass
class ManagerReadResult:
    component_results: Dict[str, ComponentReadResult]
    planner_weights: Dict[str, float]
    scenario_features: Dict[str, float]
    semantic_result: Dict[str, object]


class MetaMemoryManager:
    def __init__(
        self,
        base_config: ManifoldMemoryConfig,
        series_count: int,
        seasonality: int,
        dataset_name: str,
    ):
        self.series_count = max(1, int(series_count))
        self.seasonality = max(1, int(seasonality))
        self.dataset_name = dataset_name

        pattern_config = copy.deepcopy(base_config)
        pattern_config.same_label_merge_only = True
        pattern_config.min_label_memory = max(2, min(8, base_config.min_label_memory))
        pattern_config.max_label_memory = min(base_config.max_label_memory, 56)
        pattern_config.max_memory = min(base_config.max_memory, 128)

        trajectory_config = copy.deepcopy(base_config)
        trajectory_config.same_label_merge_only = True
        trajectory_config.similarity_threshold = min(0.985, base_config.similarity_threshold + 0.02)
        trajectory_config.max_label_memory = min(base_config.max_label_memory, 72)
        trajectory_config.max_memory = min(base_config.max_memory, 160)

        experience_config = copy.deepcopy(base_config)
        experience_config.same_label_merge_only = False
        experience_config.max_memory = base_config.max_memory
        experience_config.top_k = max(base_config.top_k, 8)
        experience_config.support_penalty = max(0.01, base_config.support_penalty * 0.75)

        if base_config.encoder_type == "transformer":
            # Transformer embeddings are more homogeneous in the current
            # eICU setting, so we tighten merge thresholds to avoid
            # collapsing the memory bank into a few over-merged prototypes.
            pattern_config.similarity_threshold = max(pattern_config.similarity_threshold, 0.975)
            pattern_config.support_penalty = max(pattern_config.support_penalty, base_config.support_penalty * 1.6)
            pattern_config.collapse_penalty = max(pattern_config.collapse_penalty, base_config.collapse_penalty * 1.8)

            trajectory_config.similarity_threshold = max(trajectory_config.similarity_threshold, 0.992)
            trajectory_config.support_penalty = max(trajectory_config.support_penalty, base_config.support_penalty * 1.8)
            trajectory_config.collapse_penalty = max(trajectory_config.collapse_penalty, base_config.collapse_penalty * 2.0)

            experience_config.same_label_merge_only = True
            experience_config.min_label_memory = max(4, base_config.min_label_memory)
            experience_config.similarity_threshold = max(experience_config.similarity_threshold, 0.997)
            experience_config.support_penalty = max(experience_config.support_penalty, base_config.support_penalty * 2.2)
            experience_config.collapse_penalty = max(experience_config.collapse_penalty, base_config.collapse_penalty * 2.4)
            experience_config.max_patient_label_memory = max(base_config.max_patient_label_memory, 6)

        self.pattern_memory = PatternMemory("pattern", pattern_config)
        self.trajectory_memory = TrajectoryMemory("trajectory", trajectory_config)
        self.experience_memory = ExperienceMemory("experience", experience_config)
        self.semantic_store: Optional[PersistentExperienceStore] = None
        self.semantic_top_k = 3

    def configure_semantic_store(
        self,
        store: Optional[PersistentExperienceStore],
        top_k: int = 3,
    ) -> None:
        self.semantic_store = store
        self.semantic_top_k = max(1, int(top_k))

    def _severity_bucket(self, metadata: Optional[Dict[str, object]], formation_features: List[float]) -> str:
        sofa_value = None if metadata is None else metadata.get("pre_baseline_total_sofa")
        try:
            severity_value = float(sofa_value)
        except (TypeError, ValueError):
            severity_value = max(0.0, float(formation_features[2]) * 8.0 + abs(float(formation_features[11])) * 4.0)
        if severity_value < 4.0:
            return "low"
        if severity_value < 8.0:
            return "moderate"
        if severity_value < 12.0:
            return "high"
        return "critical"

    def _expected_direction_type(self, formation_features: List[float], trajectory_label: int) -> str:
        medium_slope = float(formation_features[1])
        local_slope = float(formation_features[0])
        trend_signal = 0.65 * medium_slope + 0.35 * local_slope
        if trend_signal > 0.08:
            return "mostly_up"
        if trend_signal < -0.08:
            return "mostly_down"
        if trajectory_label > len(TRAJECTORY_LABELS) // 2:
            return "mostly_up"
        if trajectory_label < len(TRAJECTORY_LABELS) // 3:
            return "mostly_down"
        return "flat"

    def reset(self):
        self.pattern_memory.memory_bank = []
        self.trajectory_memory.memory_bank = []
        self.experience_memory.memory_bank = []
        self.experience_memory.archive_bank = []

    def _softmax_dict(self, values: Dict[str, float]) -> Dict[str, float]:
        max_value = max(values.values())
        exps = {key: math.exp(value - max_value) for key, value in values.items()}
        denom = sum(exps.values()) or 1.0
        return {key: value / denom for key, value in exps.items()}

    def _smoothed_label_prior(self, label_index: int, label_count: int, sharpness: float = 0.78) -> torch.Tensor:
        prior = torch.full((label_count,), (1.0 - sharpness) / max(1, label_count - 1), dtype=torch.float32)
        prior[int(label_index)] = sharpness
        return prior

    def _experience_prior(self, pattern_label: int, trajectory_label: int) -> torch.Tensor:
        label_count = len(PATTERN_LABELS) * len(TRAJECTORY_LABELS)
        target_label = int(pattern_label) * len(TRAJECTORY_LABELS) + int(trajectory_label)
        prior = torch.full((label_count,), 0.15 / max(1, label_count - 1), dtype=torch.float32)
        prior[target_label] = 0.7
        for pattern_offset in range(len(TRAJECTORY_LABELS)):
            same_pattern = int(pattern_label) * len(TRAJECTORY_LABELS) + pattern_offset
            prior[same_pattern] += 0.05 / len(TRAJECTORY_LABELS)
        for trajectory_offset in range(len(PATTERN_LABELS)):
            same_trajectory = trajectory_offset * len(TRAJECTORY_LABELS) + int(trajectory_label)
            prior[same_trajectory] += 0.05 / len(PATTERN_LABELS)
        return prior / prior.sum().clamp_min(1e-12)

    def _semantic_hits(
        self,
        formation_features: List[float],
        pattern_label: int,
        trajectory_label: int,
        experience_label: int,
        horizon: int,
        metadata: Optional[Dict[str, object]] = None,
        kg_features: Optional[List[float]] = None,
        intervention_static: Optional[List[float]] = None,
        intervention_sequence: Optional[List[List[float]]] = None,
    ) -> List[SemanticPrototypeHit]:
        if self.semantic_store is None:
            return []
        return self.semantic_store.semantic_retrieve(
            formation_features=formation_features,
            pattern_label=pattern_label,
            trajectory_label=trajectory_label,
            experience_label=experience_label,
            kg_features=kg_features,
            kg_flags=dict((metadata or {}).get("kg_flags", {})),
            intervention_static=intervention_static,
            intervention_sequence=intervention_sequence,
            future_direction_type=self._expected_direction_type(formation_features, trajectory_label),
            severity_bucket=self._severity_bucket(metadata, formation_features),
            dataset_name=self.dataset_name,
            seasonality=self.seasonality,
            forecast_horizon=horizon,
            top_k=self.semantic_top_k,
        )

    def _semantic_result(
        self, hits: List[SemanticPrototypeHit], expected_direction_type: str = "flat"
    ) -> Dict[str, object]:
        if not hits:
            return {
                "enabled": self.semantic_store is not None,
                "hit_count": 0,
                "top_score": 0.0,
                "top_experience_label": -1,
                "prototype_ids": [],
                "prototype_labels": [],
                "support_sum": 0.0,
                "template_curve": [],
                "template_confidence": 0.0,
                "template_blend_weight": 0.0,
                "direction_alignment": 1.0,
                "entropy_penalty": 1.0,
            }
        template_weights = [math.exp(float(hit.score) * 3.0) for hit in hits]
        weight_sum = sum(template_weights) or 1.0
        normalized_weights = [weight / weight_sum for weight in template_weights]
        curve_dim = len(hits[0].future_curve)
        template_curve = [
            float(
                sum(
                    normalized_weights[index] * float(hits[index].future_curve[curve_index])
                    for index in range(len(hits))
                )
            )
            for curve_index in range(curve_dim)
        ]
        template_confidence = min(
            0.95,
            0.22 + 0.58 * float(hits[0].score) + 0.08 * min(1.0, math.log1p(float(sum(hit.support for hit in hits))) / math.log(65.0)),
        )
        direction_hits = sum(
            1 for hit in hits if str(hit.future_direction_type) == str(expected_direction_type)
        )
        direction_alignment = 0.35 + 0.65 * (float(direction_hits) / max(1, len(hits)))
        mean_entropy = float(sum(hit.outcome_entropy for hit in hits)) / max(1, len(hits))
        entropy_penalty = max(0.30, 1.0 - mean_entropy)
        template_blend_weight = min(0.42, 0.08 + 0.26 * float(hits[0].score))
        return {
            "enabled": True,
            "hit_count": len(hits),
            "top_score": float(hits[0].score),
            "top_experience_label": int(hits[0].experience_label),
            "prototype_ids": [hit.prototype_id for hit in hits],
            "prototype_labels": [int(hit.experience_label) for hit in hits],
            "support_sum": float(sum(hit.support for hit in hits)),
            "template_curve": template_curve,
            "template_confidence": float(template_confidence),
            "template_blend_weight": float(template_blend_weight),
            "direction_alignment": float(direction_alignment),
            "entropy_penalty": float(entropy_penalty),
            "direction_hits": int(direction_hits),
            "mean_outcome_entropy": float(mean_entropy),
        }

    def _blend_experience_prior(
        self,
        base_prior: torch.Tensor,
        hits: List[SemanticPrototypeHit],
    ) -> torch.Tensor:
        if not hits:
            return base_prior
        label_count = len(PATTERN_LABELS) * len(TRAJECTORY_LABELS)
        semantic_prior = torch.zeros((label_count,), dtype=torch.float32)
        for hit in hits:
            if 0 <= int(hit.experience_label) < label_count:
                semantic_prior[int(hit.experience_label)] += float(hit.score)
        if semantic_prior.sum().item() <= 1e-12:
            return base_prior
        semantic_prior = semantic_prior / semantic_prior.sum().clamp_min(1e-12)
        blend_ratio = min(0.35, 0.12 + 0.18 * float(hits[0].score))
        return ((1.0 - blend_ratio) * base_prior + blend_ratio * semantic_prior) / (
            ((1.0 - blend_ratio) * base_prior + blend_ratio * semantic_prior).sum().clamp_min(1e-12)
        )

    def _adjust_planner_weights(
        self,
        planner_weights: Dict[str, float],
        hits: List[SemanticPrototypeHit],
    ) -> Dict[str, float]:
        if not hits:
            return planner_weights
        adjusted = dict(planner_weights)
        semantic_boost = min(0.18, 0.05 + 0.12 * float(hits[0].score))
        adjusted["experience"] += semantic_boost
        total = sum(adjusted.values()) or 1.0
        return {key: value / total for key, value in adjusted.items()}

    def _scenario_features(self, horizon: int, history_length: int, formation_features: List[float]) -> Dict[str, float]:
        seasonality_ratio = float(horizon) / max(1.0, float(self.seasonality))
        horizon_ratio = float(horizon) / max(1.0, float(history_length))
        series_density = min(1.0, self.series_count / 48.0)
        low_series_penalty = 1.0 - series_density
        volatility = formation_features[2]
        stability = formation_features[14]
        regime_mix = formation_features[15]
        return {
            "horizon_ratio": float(horizon_ratio),
            "seasonality_ratio": float(seasonality_ratio),
            "series_density": float(series_density),
            "low_series_penalty": float(low_series_penalty),
            "volatility": float(volatility),
            "stability": float(stability),
            "regime_mix": float(regime_mix),
        }

    def plan_weights(self, horizon: int, history_length: int, formation_features: List[float]) -> tuple[Dict[str, float], Dict[str, float]]:
        local_slope = formation_features[0]
        medium_slope = formation_features[1]
        volatility = formation_features[2]
        volatility_ratio = formation_features[3]
        seasonal_strength = formation_features[5]
        level_shift = formation_features[6]
        change_proxy = formation_features[11]
        stability = formation_features[14]
        regime_mix = formation_features[15]
        scenario = self._scenario_features(horizon=horizon, history_length=history_length, formation_features=formation_features)

        long_horizon = min(2.0, scenario["horizon_ratio"])
        seasonal_bonus = min(1.0, seasonal_strength + 0.25 * scenario["seasonality_ratio"])
        sparse_series_penalty = scenario["low_series_penalty"]

        raw = {
            "pattern": (
                1.1
                + 1.2 * abs(local_slope)
                + 1.1 * volatility
                + 0.8 * volatility_ratio
                + 0.6 * change_proxy
                - 0.9 * long_horizon
            ),
            "trajectory": (
                1.1
                + 0.8 * abs(medium_slope)
                + 0.8 * seasonal_bonus
                + 0.9 * long_horizon
                + 0.6 * abs(level_shift)
                + 0.5 * regime_mix
            ),
            "experience": (
                0.9
                + 0.9 * long_horizon
                + 0.6 * regime_mix
                + 0.5 * seasonal_bonus
                + 0.4 * change_proxy
                + 0.4 * stability
                - 1.1 * sparse_series_penalty * max(0.0, long_horizon - 0.8)
            ),
        }
        return self._softmax_dict(raw), scenario

    def write(
        self,
        encoding: TorchManifoldEncodingOutput,
        pattern_label: int,
        trajectory_label: int,
        experience_label: int,
        metadata: Dict[str, object],
        activity: float = 1.0,
    ):
        self.pattern_memory.write(encoding, label=pattern_label, metadata=metadata, activity=activity)
        self.trajectory_memory.write(encoding, label=trajectory_label, metadata=metadata, activity=activity)
        self.experience_memory.write(encoding, label=experience_label, metadata=metadata, activity=activity)

    def read(
        self,
        encoding: TorchManifoldEncodingOutput,
        horizon: int,
        history_length: int,
        formation_features: List[float],
        pattern_label: int,
        trajectory_label: int,
        experience_label: int,
        metadata: Optional[Dict[str, object]] = None,
        kg_features: Optional[List[float]] = None,
        intervention_static: Optional[List[float]] = None,
        intervention_sequence: Optional[List[List[float]]] = None,
    ) -> ManagerReadResult:
        planner_weights, scenario_features = self.plan_weights(
            horizon=horizon,
            history_length=history_length,
            formation_features=formation_features,
        )
        semantic_hits = self._semantic_hits(
            formation_features=formation_features,
            pattern_label=pattern_label,
            trajectory_label=trajectory_label,
            experience_label=experience_label,
            horizon=horizon,
            metadata=metadata,
            kg_features=kg_features,
            intervention_static=intervention_static,
            intervention_sequence=intervention_sequence,
        )
        planner_weights = self._adjust_planner_weights(planner_weights, semantic_hits)
        pattern_prior = self._smoothed_label_prior(pattern_label, len(PATTERN_LABELS))
        trajectory_prior = self._smoothed_label_prior(trajectory_label, len(TRAJECTORY_LABELS))
        experience_prior = self._experience_prior(pattern_label=pattern_label, trajectory_label=trajectory_label)
        experience_prior = self._blend_experience_prior(experience_prior, semantic_hits)
        component_results = {
            "pattern": self.pattern_memory.read(encoding, label_prior=pattern_prior),
            "trajectory": self.trajectory_memory.read(encoding, label_prior=trajectory_prior),
            "experience": self.experience_memory.read(encoding, label_prior=experience_prior),
        }
        return ManagerReadResult(
            component_results=component_results,
            planner_weights=planner_weights,
            scenario_features=scenario_features,
            semantic_result=self._semantic_result(
                semantic_hits,
                expected_direction_type=self._expected_direction_type(formation_features, trajectory_label),
            ),
        )

    def summarize(self) -> Dict[str, Dict[str, float]]:
        return {
            "pattern_memory": self.pattern_memory.summarize(),
            "trajectory_memory": self.trajectory_memory.summarize(),
            "experience_memory": self.experience_memory.summarize(),
            "manager_context": {
                "series_count": float(self.series_count),
                "seasonality": float(self.seasonality),
            },
        }
