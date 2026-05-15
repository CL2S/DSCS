import copy
import hashlib
import json
import math
import random
from datetime import datetime
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src.kg_integration import cosine_similarity
from src.manifold_memory import (
    ManifoldMemoryBlueprint,
    ManifoldMemoryConfig,
    ManifoldMemoryItem,
    TorchManifoldEncodingOutput,
)
from src.memory_manager import ManagerReadResult, MetaMemoryManager
from src.persistent_memory_store import PersistentExperienceStore, forecast_sample_identity
from src.ts_formation import PATTERN_LABELS, TRAJECTORY_LABELS, build_window_formation
from src.tsf_data import (
    EICU_AUX_BINARY_TARGET_NAMES,
    EICU_AUX_REGRESSION_TARGET_NAMES,
    EICU_AUX_TARGET_NAMES,
    ForecastSample,
)


@dataclass
class ForecastingTrainerConfig:
    forecast_horizon: int
    seasonality: int = 1
    history_length: int = 1
    series_count: int = 1
    dataset_name: str = ""
    epochs: int = 6
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    aux_base_loss_weight: float = 0.3
    multitask_loss_weight: float = 0.15
    align_loss_weight: float = 0.02
    temporal_smoothness_weight: float = 0.02
    grad_clip: float = 1.0
    device: str = "cpu"
    seed: int = 42
    enable_epoch_feedback: bool = False
    hard_example_weight: float = 0.0
    kg_consistency_weight: float = 0.0
    path_alignment_weight: float = 0.0
    archive_retention_weight: float = 0.0
    memory_delta_floor_weight: float = 0.0
    archive_retention_target: float = 0.10
    memory_delta_floor: float = 0.05
    epoch_feedback_momentum: float = 0.35
    feedback_top_error_rate: float = 0.35
    checkpoint_selection_mode: str = "best_val_mae"
    memory_refresh_interval: int = 0


@dataclass
class InterventionStoreEntry:
    stay_id: float
    experience_id: str
    experience_label: int
    pattern_label: int
    trajectory_label: int
    patient_embedding: List[float]
    intervention_plan_code: Dict[str, str]
    intervention_static: List[float]
    intervention_sequence: List[List[float]]
    kg_features: List[float]
    kg_guideline_score: float
    metadata: Dict[str, object]


@dataclass
class InterventionComponentRecord:
    component_code: str
    component_type: str
    static_values_by_index: Dict[str, float]
    sequence_values_by_index: Dict[str, List[float]]
    static_feature_names_by_index: Dict[str, str]
    sequence_feature_names_by_index: Dict[str, str]
    summary: Dict[str, object]


@dataclass
class TransitionStoreEntry:
    experience_label: int
    state_vector: List[float]
    action_vector: List[float]
    future_curve: List[float]
    transition_utility: float
    utility_vector: List[float]
    support: int
    metadata: Dict[str, object]


class EndToEndForecastingManifoldTrainer(nn.Module):
    TEMPORAL_PROFILE_FEATURE_NAMES = [
        "observed_ratio_mean",
        "recent_observed_ratio_mean",
        "mean_last_observed_gap_steps",
        "max_last_observed_gap_steps",
        "mean_longest_missing_streak",
        "mean_trailing_missing_streak",
        "recent_change_mean_abs",
        "recent_trend_mean_abs",
        "target_recent_trend",
        "target_recent_acceleration",
        "time_gap_last_hours",
        "time_gap_mean_hours",
        "time_gap_std_hours",
        "time_irregularity",
    ]

    def __init__(
        self,
        memory_config: ManifoldMemoryConfig,
        trainer_config: ForecastingTrainerConfig,
        static_feature_dim: int,
        kg_feature_dim: int,
        intervention_feature_dim: int,
        intervention_sequence_dim: int,
        formation_feature_dim: int,
    ):
        super().__init__()
        self.memory_config = memory_config
        self.trainer_config = trainer_config
        self.device = torch.device(trainer_config.device)
        random.seed(trainer_config.seed)
        torch.manual_seed(trainer_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(trainer_config.seed)

        self.manifold = ManifoldMemoryBlueprint(memory_config)
        self.memory_manager = MetaMemoryManager(
            memory_config,
            series_count=trainer_config.series_count,
            seasonality=trainer_config.seasonality,
            dataset_name=trainer_config.dataset_name,
        )
        self.intervention_feature_dim = intervention_feature_dim
        self.intervention_sequence_dim = intervention_sequence_dim
        has_intervention_input = intervention_feature_dim > 0 or intervention_sequence_dim > 0
        self.intervention_context_dim = max(16, memory_config.static_hidden_dim * 2) if has_intervention_input else 0
        self.intervention_static_projector = None
        self.intervention_sequence_encoder = None
        self.intervention_sequence_projector = None
        self.intervention_fuser = None
        if intervention_feature_dim > 0:
            self.intervention_static_projector = nn.Sequential(
                nn.Linear(intervention_feature_dim, self.intervention_context_dim),
                nn.GELU(),
                nn.LayerNorm(self.intervention_context_dim) if memory_config.use_layer_norm else nn.Identity(),
            )
        if intervention_sequence_dim > 0:
            sequence_hidden_dim = max(16, memory_config.static_hidden_dim * 2)
            self.intervention_sequence_encoder = nn.GRU(
                input_size=intervention_sequence_dim,
                hidden_size=sequence_hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            self.intervention_sequence_projector = nn.Sequential(
                nn.Linear(sequence_hidden_dim, self.intervention_context_dim),
                nn.GELU(),
                nn.LayerNorm(self.intervention_context_dim) if memory_config.use_layer_norm else nn.Identity(),
            )
        if has_intervention_input:
            intervention_input_dim = 0
            if intervention_feature_dim > 0:
                intervention_input_dim += self.intervention_context_dim
            if intervention_sequence_dim > 0:
                intervention_input_dim += self.intervention_context_dim
            self.intervention_fuser = nn.Sequential(
                nn.Linear(intervention_input_dim, self.intervention_context_dim),
                nn.GELU(),
                nn.LayerNorm(self.intervention_context_dim) if memory_config.use_layer_norm else nn.Identity(),
            )
        embedding_dim = self.manifold.encoder.embedding_dim
        self.factual_projection = self._make_space_adapter(
            embedding_dim,
            use_layer_norm=memory_config.use_layer_norm,
        )
        self.retrieval_projection = self._make_space_adapter(
            embedding_dim,
            use_layer_norm=memory_config.use_layer_norm,
        )
        self.factual_projection_scale = 0.30
        self.retrieval_projection_scale = 0.30
        self.retrieval_head_delta_scale = 0.30
        self._zero_last_linear(self.factual_projection)
        self._zero_last_linear(self.retrieval_projection)
        self.branch_decoupling_mode = "post_projection"
        self.branch_decoupling_enabled = True
        self.temporal_profile_feature_names = list(self.TEMPORAL_PROFILE_FEATURE_NAMES)
        self.temporal_profile_dim = len(self.temporal_profile_feature_names)
        branch_context_dim = self.temporal_profile_dim + formation_feature_dim
        branch_hidden_dim = max(16, memory_config.static_hidden_dim * 2)
        self.factual_branch_projector = nn.Sequential(
            nn.Linear(embedding_dim + branch_context_dim, branch_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(branch_hidden_dim) if memory_config.use_layer_norm else nn.Identity(),
            nn.Linear(branch_hidden_dim, embedding_dim),
        )
        self.retrieval_branch_projector = nn.Sequential(
            nn.Linear(embedding_dim + branch_context_dim, branch_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(branch_hidden_dim) if memory_config.use_layer_norm else nn.Identity(),
            nn.Linear(branch_hidden_dim, embedding_dim),
        )
        self.factual_branch_gate = nn.Sequential(
            nn.Linear(branch_context_dim, branch_hidden_dim),
            nn.GELU(),
            nn.Linear(branch_hidden_dim, 1),
        )
        self.retrieval_branch_gate = nn.Sequential(
            nn.Linear(branch_context_dim, branch_hidden_dim),
            nn.GELU(),
            nn.Linear(branch_hidden_dim, 1),
        )
        self.factual_branch_scale = 0.08
        self.retrieval_branch_scale = 0.08
        self._zero_last_linear(self.factual_branch_projector)
        self._zero_last_linear(self.retrieval_branch_projector)
        temporal_hidden_dim = max(16, memory_config.static_hidden_dim * 2)
        self.factual_temporal_projector = nn.Sequential(
            nn.Linear(self.temporal_profile_dim, temporal_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(temporal_hidden_dim) if memory_config.use_layer_norm else nn.Identity(),
            nn.Linear(temporal_hidden_dim, embedding_dim),
        )
        self.factual_temporal_gate = nn.Sequential(
            nn.Linear(self.temporal_profile_dim + formation_feature_dim, temporal_hidden_dim),
            nn.GELU(),
            nn.Linear(temporal_hidden_dim, 1),
        )
        self.factual_temporal_path_scale = 0.10
        self._zero_last_linear(self.factual_temporal_projector)
        base_regressor_input_dim = embedding_dim + self.intervention_context_dim
        factual_prediction_hidden_dim = max(16, memory_config.fusion_hidden_dim)
        self.factual_prediction_trunk = nn.Sequential(
            nn.Linear(base_regressor_input_dim + branch_context_dim, factual_prediction_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(factual_prediction_hidden_dim) if memory_config.use_layer_norm else nn.Identity(),
            nn.Linear(factual_prediction_hidden_dim, base_regressor_input_dim),
        )
        self.factual_prediction_gate = nn.Sequential(
            nn.Linear(branch_context_dim, factual_prediction_hidden_dim),
            nn.GELU(),
            nn.Linear(factual_prediction_hidden_dim, 1),
        )
        self.factual_prediction_trunk_scale = 0.08
        self._zero_last_linear(self.factual_prediction_trunk)
        self.factual_scale_head = nn.Linear(base_regressor_input_dim, trainer_config.forecast_horizon)
        nn.init.zeros_(self.factual_scale_head.weight)
        nn.init.constant_(self.factual_scale_head.bias, -2.0)
        self.factual_calibration_weight = 0.05
        self.retrieval_query_head = nn.Linear(embedding_dim, memory_config.manifold_dim)
        self.retrieval_key_head = nn.Linear(embedding_dim, memory_config.manifold_dim)
        self.retrieval_value_head = nn.Linear(embedding_dim, memory_config.value_dim)
        nn.init.zeros_(self.retrieval_query_head.weight)
        nn.init.zeros_(self.retrieval_query_head.bias)
        nn.init.zeros_(self.retrieval_key_head.weight)
        nn.init.zeros_(self.retrieval_key_head.bias)
        nn.init.zeros_(self.retrieval_value_head.weight)
        nn.init.zeros_(self.retrieval_value_head.bias)
        self.base_regressor = nn.Linear(base_regressor_input_dim, trainer_config.forecast_horizon)
        self.multitask_task_names = list(EICU_AUX_TARGET_NAMES)
        self.multitask_regression_tasks = set(EICU_AUX_REGRESSION_TARGET_NAMES)
        self.multitask_binary_tasks = set(EICU_AUX_BINARY_TARGET_NAMES)
        multitask_input_dim = memory_config.fusion_hidden_dim + trainer_config.forecast_horizon + self.intervention_context_dim
        multitask_hidden_dim = max(16, memory_config.fusion_hidden_dim)
        self.multitask_trunk = nn.Sequential(
            nn.Linear(multitask_input_dim, multitask_hidden_dim),
            nn.GELU(),
            nn.Dropout(memory_config.gru_dropout),
            nn.Linear(multitask_hidden_dim, multitask_hidden_dim),
            nn.GELU(),
        )
        self.multitask_heads = nn.ModuleDict(
            {
                task_name: nn.Linear(multitask_hidden_dim, 1)
                for task_name in self.multitask_task_names
            }
        )
        self.aux_target_stats: Dict[str, Dict[str, float]] = {
            task_name: {"mean": 0.0, "std": 1.0}
            for task_name in self.multitask_regression_tasks
        }

        self.memory_projectors = nn.ModuleDict(
            {
                "pattern": nn.Linear(memory_config.value_dim, embedding_dim),
                "trajectory": nn.Linear(memory_config.value_dim, embedding_dim),
                "experience": nn.Linear(memory_config.value_dim, embedding_dim),
            }
        )
        self.memory_residual_heads = nn.ModuleDict(
            {
                "experience": nn.Linear(memory_config.value_dim, trainer_config.forecast_horizon),
            }
        )
        self.direct_residual_gate = nn.Sequential(
            nn.Linear(7, max(8, memory_config.fusion_hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(8, memory_config.fusion_hidden_dim // 2), 1),
        )
        self.memory_direct_residual_weight = 0.1
        self.memory_direct_residual_mode = "fixed"
        self.memory_harm_control_enabled = True
        self.memory_quality_floor = 0.18
        self.memory_min_path_alignment = -0.20
        self.memory_residual_cap_ratio = 0.35
        self.harm_stable_quality_boost = 0.0
        self.harm_flat_quality_boost = 0.0
        self.counterfactual_donor_score_mode = "legacy"
        self.counterfactual_base_similarity_weight = 1.0
        self.counterfactual_kg_similarity_weight = 0.20
        self.counterfactual_guideline_weight = 0.15
        self.counterfactual_guideline_score_weight = 0.10
        self.counterfactual_penalty_weight = 0.15
        self.counterfactual_hard_filter_enabled = True
        self.counterfactual_candidate_policy = "donor_only"
        self.counterfactual_candidate_search_mode = "expanded"
        self.counterfactual_candidate_feasibility_weight = 0.15
        self.counterfactual_candidate_penalty_weight = 0.10
        self.counterfactual_multitask_sofa_weight = 0.22
        self.counterfactual_multitask_lactate_weight = 0.10
        self.counterfactual_multitask_vasopressor_weight = 0.12
        self.counterfactual_multitask_respiratory_weight = 0.08
        self.counterfactual_multitask_uncertainty_weight = 0.15
        self.counterfactual_multitask_lower_bound_weight = 0.65
        self.counterfactual_multitask_conflict_weight = 0.04
        self.counterfactual_multitask_positive_unstable_weight = 0.25
        self.counterfactual_search_template_limit = 4
        self.counterfactual_search_guideline_drop_max = 0.12
        self.counterfactual_search_missing_care_tolerance = 0.05
        self.counterfactual_search_overlap_drop_max = 0.20
        self.counterfactual_parameterized_template_limit = 8
        self.counterfactual_rollout_steps = 1
        self.counterfactual_rollout_discount = 0.70
        self.counterfactual_label_top_k = 8
        self.counterfactual_pool_mode = "adaptive"
        self.counterfactual_pool_include_pattern = True
        self.counterfactual_pool_include_trajectory = True
        self.counterfactual_pool_include_hospital = True
        self.counterfactual_pool_include_unit_type = True
        self.counterfactual_pool_include_infection_anchor = True
        self.counterfactual_pool_enable_global_backfill = True
        self.counterfactual_pool_local_min_candidates = 24
        self.counterfactual_pool_min_candidates = 64
        self.counterfactual_pool_global_limit = 256
        self.counterfactual_pool_prefilter_top_k = 160
        self.counterfactual_pool_match_weight = 0.10
        self.counterfactual_pool_same_hospital_weight = 0.55
        self.counterfactual_pool_same_unit_weight = 0.30
        self.counterfactual_pool_same_infection_anchor_weight = 0.15
        self.counterfactual_overlap_filter_enabled = True
        self.counterfactual_overlap_fallback_enabled = True
        self.counterfactual_overlap_weight = 0.12
        self.counterfactual_overlap_severity_gap_max = 1.60
        self.counterfactual_overlap_trend_gap_max = 1.25
        self.counterfactual_overlap_state_min = 0.35
        self.counterfactual_overlap_action_min = 0.40
        self.counterfactual_neighbor_top_k = 4
        self.counterfactual_neighbor_weight = 0.24
        self.counterfactual_neighbor_penalty_weight = 0.52
        self.counterfactual_neighbor_min_consistency = 0.45
        self.counterfactual_neighbor_similarity_band = 0.12
        self.counterfactual_neighbor_self_weight = 0.42
        self.counterfactual_reranker_mode = "rule_only"
        self.counterfactual_reranker_blend_weight = 0.35
        self.counterfactual_reranker_train_top_k = 4
        self.counterfactual_reranker_max_samples = 256
        self.counterfactual_reranker_min_examples = 32
        self.counterfactual_reranker_ridge_l2 = 1e-3
        self.enable_transition_memory = False
        self.enable_transition_factual_path = True
        self.enable_transition_donor_path = True
        self.transition_top_k = 6
        self.transition_state_weight = 0.65
        self.transition_action_weight = 0.35
        self.transition_score_weight = 0.12
        self.transition_template_blend_weight = 0.10
        self.transition_selection_weight = 0.04
        self.transition_temperature = 0.20
        self.transition_utility_scale = 2.0
        self.transition_utility_alignment_weight = 0.0
        self.transition_anchor_blend_weight = 0.0
        self.transition_signature_match_weight = 0.08
        self.transition_min_confidence = 0.50
        self.transition_min_support = 0.20
        self.transition_min_expected_utility = 0.05
        self.transition_min_signature_weight = 0.0
        self.transition_stable_regime_penalty = 1.0
        self.transition_flat_pattern_penalty = 1.0
        self.transition_utility_bias = -0.05
        self.transition_utility_temperature = 0.08
        self.transition_residual_cap_ratio = 0.30
        self.transition_signature_partial_match = True
        self.enable_transition_trunk_path = False
        self.transition_trunk_weight = 0.0
        self.transition_factual_residual_mode = "additive"
        self.transition_positive_only = False
        self.transition_action_change_weight = 0.05
        self.transition_candidate_action_change_weight = 0.04
        self.memory_path_coordinator = nn.Sequential(
            nn.Linear(16, max(8, memory_config.fusion_hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(8, memory_config.fusion_hidden_dim // 2), 2),
        )
        self.memory_path_coordination_mode = "sum"
        self.enable_epoch_feedback = bool(trainer_config.enable_epoch_feedback)
        self.dynamic_hard_example_weight = max(0.0, float(trainer_config.hard_example_weight))
        self.dynamic_kg_consistency_weight = max(0.0, float(trainer_config.kg_consistency_weight))
        self.dynamic_path_alignment_weight = max(0.0, float(trainer_config.path_alignment_weight))
        self.dynamic_archive_retention_weight = max(0.0, float(trainer_config.archive_retention_weight))
        self.dynamic_memory_delta_floor_weight = max(0.0, float(trainer_config.memory_delta_floor_weight))
        self.epoch_feedback_history: List[Dict[str, object]] = []
        self.kg_feature_dim = max(0, int(kg_feature_dim))
        self.enable_explicit_kg_path = self.kg_feature_dim > 0
        self.kg_residual_weight = 0.12
        self.kg_alignment_floor = 0.05
        self.kg_context_dim = max(8, memory_config.static_hidden_dim * 2) if self.kg_feature_dim > 0 else 0
        self.kg_projector = None
        self.kg_residual_head = None
        self.kg_gate = None
        if self.kg_feature_dim > 0:
            kg_input_dim = self.kg_feature_dim + 4
            self.kg_projector = nn.Sequential(
                nn.Linear(kg_input_dim, self.kg_context_dim),
                nn.GELU(),
                nn.LayerNorm(self.kg_context_dim) if memory_config.use_layer_norm else nn.Identity(),
                nn.Linear(self.kg_context_dim, self.kg_context_dim),
                nn.GELU(),
            )
            self.kg_residual_head = nn.Linear(self.kg_context_dim, trainer_config.forecast_horizon)
            self.kg_gate = nn.Sequential(
                nn.Linear(kg_input_dim, self.kg_context_dim),
                nn.GELU(),
                nn.Linear(self.kg_context_dim, 1),
            )
        self.prototype_curve_encoder = nn.Sequential(
            nn.Linear(trainer_config.forecast_horizon, memory_config.value_dim),
            nn.GELU(),
            nn.Linear(memory_config.value_dim, memory_config.value_dim),
        )
        transition_context_dim = trainer_config.forecast_horizon * 2 + 6
        self.transition_context_projector = nn.Sequential(
            nn.Linear(transition_context_dim, memory_config.fusion_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(memory_config.fusion_hidden_dim) if memory_config.use_layer_norm else nn.Identity(),
            nn.Linear(memory_config.fusion_hidden_dim, memory_config.fusion_hidden_dim),
            nn.GELU(),
        )
        self.transition_trunk_gate = nn.Sequential(
            nn.Linear(6, max(8, memory_config.fusion_hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(8, memory_config.fusion_hidden_dim // 2), 1),
        )
        gate_extra_dim = 3 + 3 + 4
        gate_input_dim = embedding_dim + self.intervention_context_dim + formation_feature_dim + gate_extra_dim
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, memory_config.fusion_hidden_dim),
            nn.GELU(),
            nn.Linear(memory_config.fusion_hidden_dim, 3),
        )

        fusion_extra_dim = 3 + 3 + 3 + 4
        fusion_input_dim = embedding_dim * 4 + self.intervention_context_dim + fusion_extra_dim
        self.multi_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, memory_config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(memory_config.gru_dropout),
            nn.Linear(memory_config.fusion_hidden_dim, memory_config.fusion_hidden_dim),
            nn.GELU(),
        )
        self.bucket_heads = nn.ModuleDict(
            {
                "short": nn.Linear(memory_config.fusion_hidden_dim, trainer_config.forecast_horizon),
                "mid": nn.Linear(memory_config.fusion_hidden_dim, trainer_config.forecast_horizon),
                "long": nn.Linear(memory_config.fusion_hidden_dim, trainer_config.forecast_horizon),
            }
        )
        self.register_buffer("bucket_masks", self._build_bucket_masks(trainer_config.forecast_horizon))
        self.to(self.device)

        self.sequence_mean: Optional[torch.Tensor] = None
        self.sequence_std: Optional[torch.Tensor] = None
        self.static_mean: Optional[torch.Tensor] = None
        self.static_std: Optional[torch.Tensor] = None
        self.intervention_mean: Optional[torch.Tensor] = None
        self.intervention_std: Optional[torch.Tensor] = None
        self.intervention_sequence_mean: Optional[torch.Tensor] = None
        self.intervention_sequence_std: Optional[torch.Tensor] = None
        self.formation_mean: Optional[torch.Tensor] = None
        self.formation_std: Optional[torch.Tensor] = None

        self.best_epoch = 0
        self.training_summary: Dict[str, float] = {}
        self.memory_diagnostics: Dict[str, object] = {}
        self.counterfactual_reranker_state: Dict[str, object] = {}
        self.cache_store: Optional[PersistentExperienceStore] = None
        self.neural_cache_reuse_enabled = True
        self.neural_cache_export_enabled = True
        self.current_model_fingerprint = ""
        self.neural_cache_status: Dict[str, object] = {}
        self.intervention_store_entries: List[InterventionStoreEntry] = []
        self.intervention_plan_store: Dict[str, Dict[str, object]] = {}
        self.intervention_method_store: Dict[str, InterventionComponentRecord] = {}
        self.intervention_dose_store: Dict[str, InterventionComponentRecord] = {}
        self.intervention_timing_store: Dict[str, InterventionComponentRecord] = {}
        self.intervention_context_store: Dict[str, InterventionComponentRecord] = {}
        self.intervention_store_by_label: Dict[int, List[int]] = {}
        self.intervention_store_by_pattern: Dict[int, List[int]] = {}
        self.intervention_store_by_trajectory: Dict[int, List[int]] = {}
        self.intervention_store_by_hospital: Dict[str, List[int]] = {}
        self.intervention_store_by_unit_type: Dict[str, List[int]] = {}
        self.intervention_store_by_infection_anchor: Dict[str, List[int]] = {}
        self.intervention_store_embedding_cache: Dict[int, torch.Tensor] = {}
        self.transition_store_entries: List[TransitionStoreEntry] = []
        self.transition_store_by_label: Dict[int, List[int]] = {}
        self.transition_state_cache: Dict[int, torch.Tensor] = {}
        self.transition_action_cache: Dict[int, torch.Tensor] = {}
        self.transition_future_cache: Dict[int, torch.Tensor] = {}
        self.transition_utility_cache: Dict[int, torch.Tensor] = {}
        self.transition_utility_vector_cache: Dict[int, torch.Tensor] = {}
        self.transition_future_mean_cache: Dict[int, torch.Tensor] = {}
        self.transition_label_support: Dict[int, int] = {}
        self.transition_label_utility_mean: Dict[int, float] = {}
        self.transition_label_improvement_rate: Dict[int, float] = {}
        self.transition_label_utility_vector_mean: Dict[int, torch.Tensor] = {}
        self.transition_signature_cache: Dict[int, List[str]] = {}
        self.transition_signature_payload_cache: Dict[int, List[Dict[str, object]]] = {}
        self.transition_signature_counts: Dict[str, int] = {}
        self.sequence_feature_names: List[str] = []
        self.patient_feature_names: List[str] = []
        self.intervention_feature_names: List[str] = []
        self.intervention_sequence_feature_names: List[str] = []

    @staticmethod
    def _make_space_adapter(input_dim: int, use_layer_norm: bool) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.LayerNorm(input_dim) if use_layer_norm else nn.Identity(),
            nn.Linear(input_dim, input_dim),
        )

    @staticmethod
    def _zero_last_linear(module: nn.Module) -> None:
        for layer in reversed(list(module.modules())):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                return

    def _residual_space_projection(self, base_embedding: torch.Tensor, adapter: nn.Module) -> torch.Tensor:
        scale = self.factual_projection_scale
        if adapter is self.retrieval_projection:
            scale = self.retrieval_projection_scale
        return base_embedding + scale * adapter(base_embedding)

    @staticmethod
    def _align_tensor_rank(tensor: torch.Tensor, target_rank: int) -> torch.Tensor:
        while tensor.dim() > target_rank and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        while tensor.dim() < target_rank:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _branch_context(
        self,
        temporal_profile: torch.Tensor,
        normalized_formation: torch.Tensor,
    ) -> torch.Tensor:
        target_rank = max(temporal_profile.dim(), normalized_formation.dim(), 2)
        temporal_profile = self._align_tensor_rank(temporal_profile, target_rank)
        normalized_formation = self._align_tensor_rank(normalized_formation, target_rank)
        return torch.cat([temporal_profile, normalized_formation], dim=-1)

    def _apply_branch_decoupling(
        self,
        base_embedding: torch.Tensor,
        branch_context: torch.Tensor,
        projector: nn.Module,
        gate_module: nn.Module,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_rank = base_embedding.dim()
        target_rank = max(base_embedding.dim(), branch_context.dim(), 2)
        base_embedding = self._align_tensor_rank(base_embedding, target_rank)
        branch_context = self._align_tensor_rank(branch_context, target_rank)
        branch_delta = projector(torch.cat([base_embedding, branch_context], dim=-1))
        branch_gate = torch.sigmoid(gate_module(branch_context))
        branch_embedding = base_embedding + scale * branch_gate * branch_delta
        branch_strength = branch_delta.abs().mean(dim=-1)
        if original_rank == 1:
            return branch_embedding.squeeze(0), branch_gate.squeeze(0), branch_strength.squeeze(0)
        return branch_embedding, branch_gate.squeeze(-1), branch_strength

    def _augment_factual_prediction_input(
        self,
        base_input: torch.Tensor,
        temporal_profile: torch.Tensor,
        normalized_formation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        branch_context = self._branch_context(
            temporal_profile=temporal_profile,
            normalized_formation=normalized_formation,
        )
        return self._apply_branch_decoupling(
            base_embedding=base_input,
            branch_context=branch_context,
            projector=self.factual_prediction_trunk,
            gate_module=self.factual_prediction_gate,
            scale=self.factual_prediction_trunk_scale,
        )

    def _predict_factual_scale(self, prediction_input: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.factual_scale_head(prediction_input)) + 1e-3
        return scale

    def _split_encoding_spaces(
        self,
        encoding: TorchManifoldEncodingOutput,
        normalized_formation: torch.Tensor,
        temporal_profile: torch.Tensor,
    ) -> tuple[torch.Tensor, TorchManifoldEncodingOutput, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        branch_context = self._branch_context(
            temporal_profile=temporal_profile,
            normalized_formation=normalized_formation,
        )
        if not self.branch_decoupling_enabled:
            retrieval_encoding = TorchManifoldEncodingOutput(
                query=encoding.query,
                key=encoding.key,
                value=encoding.value,
                input_embedding=encoding.input_embedding,
                metadata=dict(encoding.metadata or {}),
            )
            if encoding.input_embedding.dim() > 1:
                batch_dim = int(encoding.input_embedding.shape[0])
                zero_gates = torch.zeros(batch_dim, dtype=torch.float32, device=self.device)
            else:
                zero_gates = torch.zeros(1, dtype=torch.float32, device=self.device)
            return encoding.input_embedding, retrieval_encoding, zero_gates, zero_gates, zero_gates, zero_gates
        if self.branch_decoupling_mode == "post_projection":
            factual_embedding = self._residual_space_projection(encoding.input_embedding, self.factual_projection)
            retrieval_embedding = self._residual_space_projection(encoding.input_embedding, self.retrieval_projection)
            factual_embedding, factual_branch_gate, factual_branch_strength = self._apply_branch_decoupling(
                base_embedding=factual_embedding,
                branch_context=branch_context,
                projector=self.factual_branch_projector,
                gate_module=self.factual_branch_gate,
                scale=self.factual_branch_scale,
            )
            retrieval_embedding, retrieval_branch_gate, retrieval_branch_strength = self._apply_branch_decoupling(
                base_embedding=retrieval_embedding,
                branch_context=branch_context,
                projector=self.retrieval_branch_projector,
                gate_module=self.retrieval_branch_gate,
                scale=self.retrieval_branch_scale,
            )
        else:
            factual_seed, factual_branch_gate, factual_branch_strength = self._apply_branch_decoupling(
                base_embedding=encoding.input_embedding,
                branch_context=branch_context,
                projector=self.factual_branch_projector,
                gate_module=self.factual_branch_gate,
                scale=self.factual_branch_scale,
            )
            retrieval_seed, retrieval_branch_gate, retrieval_branch_strength = self._apply_branch_decoupling(
                base_embedding=encoding.input_embedding,
                branch_context=branch_context,
                projector=self.retrieval_branch_projector,
                gate_module=self.retrieval_branch_gate,
                scale=self.retrieval_branch_scale,
            )
            factual_embedding = self._residual_space_projection(factual_seed, self.factual_projection)
            retrieval_embedding = self._residual_space_projection(retrieval_seed, self.retrieval_projection)
        retrieval_encoding = TorchManifoldEncodingOutput(
            query=F.normalize(
                encoding.query + self.retrieval_head_delta_scale * self.retrieval_query_head(retrieval_embedding),
                dim=-1,
            ),
            key=F.normalize(
                encoding.key + self.retrieval_head_delta_scale * self.retrieval_key_head(retrieval_embedding),
                dim=-1,
            ),
            value=encoding.value + self.retrieval_head_delta_scale * self.retrieval_value_head(retrieval_embedding),
            input_embedding=retrieval_embedding,
            metadata=dict(encoding.metadata or {}),
        )
        return (
            factual_embedding,
            retrieval_encoding,
            factual_branch_gate,
            factual_branch_strength,
            retrieval_branch_gate,
            retrieval_branch_strength,
        )

    def configure_semantic_store(
        self,
        store: Optional[PersistentExperienceStore],
        top_k: int = 3,
    ) -> None:
        self.memory_manager.configure_semantic_store(store=store, top_k=top_k)

    def configure_neural_cache(
        self,
        store: Optional[PersistentExperienceStore],
        reuse_enabled: bool = True,
        export_enabled: bool = True,
    ) -> None:
        self.cache_store = store
        self.neural_cache_reuse_enabled = bool(reuse_enabled)
        self.neural_cache_export_enabled = bool(export_enabled)

    def _memory_read_kwargs(self, sample: ForecastSample) -> Dict[str, object]:
        return {
            "metadata": sample.metadata,
            "kg_features": list(sample.kg_features or sample.metadata.get("kg_features", []) or []),
            "intervention_static": list(sample.intervention_static or []),
            "intervention_sequence": [list(step) for step in (sample.intervention_sequence or [])],
        }

    def _memory_item_to_dict(self, item: ManifoldMemoryItem) -> Dict[str, object]:
        return {
            "key": [float(value) for value in item.key],
            "value": [float(value) for value in item.value],
            "label": int(item.label),
            "activity": float(item.activity),
            "support": int(item.support),
            "metadata": dict(item.metadata),
        }

    def _memory_item_from_dict(self, payload: Dict[str, object]) -> ManifoldMemoryItem:
        return ManifoldMemoryItem(
            key=[float(value) for value in payload.get("key", [])],
            value=[float(value) for value in payload.get("value", [])],
            label=int(payload.get("label", 0)),
            activity=float(payload.get("activity", 1.0)),
            support=int(payload.get("support", 1)),
            metadata=dict(payload.get("metadata", {})),
        )

    def compute_model_fingerprint(self) -> str:
        digest = hashlib.sha1()
        signature = {
            "forecast_horizon": self.trainer_config.forecast_horizon,
            "seasonality": self.trainer_config.seasonality,
            "history_length": self.trainer_config.history_length,
            "series_count": self.trainer_config.series_count,
            "dataset_name": self.trainer_config.dataset_name,
            "seed": self.trainer_config.seed,
            "intervention_feature_dim": self.intervention_feature_dim,
            "intervention_sequence_dim": self.intervention_sequence_dim,
            "memory_direct_residual_weight": self.memory_direct_residual_weight,
            "memory_direct_residual_mode": self.memory_direct_residual_mode,
            "memory_path_coordination_mode": self.memory_path_coordination_mode,
            "memory_harm_control_enabled": self.memory_harm_control_enabled,
            "memory_quality_floor": self.memory_quality_floor,
            "memory_min_path_alignment": self.memory_min_path_alignment,
            "memory_residual_cap_ratio": self.memory_residual_cap_ratio,
            "harm_stable_quality_boost": self.harm_stable_quality_boost,
            "harm_flat_quality_boost": self.harm_flat_quality_boost,
            "enable_epoch_feedback": self.enable_epoch_feedback,
            "hard_example_weight": self.dynamic_hard_example_weight,
            "kg_consistency_weight": self.dynamic_kg_consistency_weight,
            "path_alignment_weight": self.dynamic_path_alignment_weight,
            "memory_config": vars(self.memory_config),
        }
        digest.update(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        state_dict = self.state_dict()
        for name in sorted(state_dict.keys()):
            tensor = state_dict[name].detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        return f"mf_{digest.hexdigest()[:20]}"

    def export_neural_cache(self) -> Dict[str, object]:
        return {
            "schema_version": 1,
            "created_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "model_fingerprint": self.current_model_fingerprint,
            "dataset_name": self.trainer_config.dataset_name,
            "seasonality": self.trainer_config.seasonality,
            "forecast_horizon": self.trainer_config.forecast_horizon,
            "banks": {
                "pattern": [self._memory_item_to_dict(item) for item in self.memory_manager.pattern_memory.memory_bank],
                "trajectory": [self._memory_item_to_dict(item) for item in self.memory_manager.trajectory_memory.memory_bank],
                "experience_hot": [self._memory_item_to_dict(item) for item in self.memory_manager.experience_memory.memory_bank],
                "experience_archive": [self._memory_item_to_dict(item) for item in self.memory_manager.experience_memory.archive_bank],
            },
        }

    def load_neural_cache(self, payload: Dict[str, object]) -> None:
        banks = payload.get("banks", {})
        self.memory_manager.pattern_memory.memory_bank = [
            self._memory_item_from_dict(item) for item in banks.get("pattern", [])
        ]
        self.memory_manager.trajectory_memory.memory_bank = [
            self._memory_item_from_dict(item) for item in banks.get("trajectory", [])
        ]
        self.memory_manager.experience_memory.memory_bank = [
            self._memory_item_from_dict(item) for item in banks.get("experience_hot", [])
        ]
        self.memory_manager.experience_memory.archive_bank = [
            self._memory_item_from_dict(item) for item in banks.get("experience_archive", [])
        ]

    @staticmethod
    def _optional_tensor_to_payload(tensor: Optional[torch.Tensor]) -> Optional[List[float]]:
        if tensor is None:
            return None
        return tensor.detach().cpu().tolist()

    def _optional_tensor_from_payload(self, payload: Optional[Sequence[float]]) -> Optional[torch.Tensor]:
        if payload is None:
            return None
        return torch.tensor(payload, dtype=torch.float32, device=self.device)

    def _normalizer_payload(self) -> Dict[str, object]:
        return {
            "sequence_mean": self._optional_tensor_to_payload(self.sequence_mean),
            "sequence_std": self._optional_tensor_to_payload(self.sequence_std),
            "static_mean": self._optional_tensor_to_payload(self.static_mean),
            "static_std": self._optional_tensor_to_payload(self.static_std),
            "intervention_mean": self._optional_tensor_to_payload(self.intervention_mean),
            "intervention_std": self._optional_tensor_to_payload(self.intervention_std),
            "intervention_sequence_mean": self._optional_tensor_to_payload(self.intervention_sequence_mean),
            "intervention_sequence_std": self._optional_tensor_to_payload(self.intervention_sequence_std),
            "formation_mean": self._optional_tensor_to_payload(self.formation_mean),
            "formation_std": self._optional_tensor_to_payload(self.formation_std),
            "aux_target_stats": {key: dict(value) for key, value in self.aux_target_stats.items()},
        }

    def _load_normalizer_payload(self, payload: Dict[str, object]) -> None:
        self.sequence_mean = self._optional_tensor_from_payload(payload.get("sequence_mean"))
        self.sequence_std = self._optional_tensor_from_payload(payload.get("sequence_std"))
        self.static_mean = self._optional_tensor_from_payload(payload.get("static_mean"))
        self.static_std = self._optional_tensor_from_payload(payload.get("static_std"))
        self.intervention_mean = self._optional_tensor_from_payload(payload.get("intervention_mean"))
        self.intervention_std = self._optional_tensor_from_payload(payload.get("intervention_std"))
        self.intervention_sequence_mean = self._optional_tensor_from_payload(payload.get("intervention_sequence_mean"))
        self.intervention_sequence_std = self._optional_tensor_from_payload(payload.get("intervention_sequence_std"))
        self.formation_mean = self._optional_tensor_from_payload(payload.get("formation_mean"))
        self.formation_std = self._optional_tensor_from_payload(payload.get("formation_std"))
        aux_target_stats = payload.get("aux_target_stats", {})
        if isinstance(aux_target_stats, dict):
            self.aux_target_stats = {
                str(key): {
                    "mean": float(dict(value).get("mean", 0.0)),
                    "std": max(1e-6, float(dict(value).get("std", 1.0))),
                }
                for key, value in aux_target_stats.items()
                if str(key) in self.multitask_regression_tasks
            }

    @staticmethod
    def _intervention_store_entry_to_dict(entry: InterventionStoreEntry) -> Dict[str, object]:
        return {
            "stay_id": float(entry.stay_id),
            "experience_id": str(entry.experience_id),
            "experience_label": int(entry.experience_label),
            "pattern_label": int(entry.pattern_label),
            "trajectory_label": int(entry.trajectory_label),
            "patient_embedding": [float(value) for value in entry.patient_embedding],
            "intervention_plan_code": dict(entry.intervention_plan_code),
            "intervention_static": [float(value) for value in entry.intervention_static],
            "intervention_sequence": [list(step) for step in entry.intervention_sequence],
            "kg_features": [float(value) for value in entry.kg_features],
            "kg_guideline_score": float(entry.kg_guideline_score),
            "metadata": dict(entry.metadata),
        }

    @staticmethod
    def _intervention_store_entry_from_dict(payload: Dict[str, object]) -> InterventionStoreEntry:
        return InterventionStoreEntry(
            stay_id=float(payload.get("stay_id", -1.0)),
            experience_id=str(payload.get("experience_id", "")),
            experience_label=int(payload.get("experience_label", 0)),
            pattern_label=int(payload.get("pattern_label", 0)),
            trajectory_label=int(payload.get("trajectory_label", 0)),
            patient_embedding=[float(value) for value in payload.get("patient_embedding", [])],
            intervention_plan_code={
                str(key): str(value)
                for key, value in dict(payload.get("intervention_plan_code", {})).items()
            },
            intervention_static=[float(value) for value in payload.get("intervention_static", [])],
            intervention_sequence=[list(step) for step in payload.get("intervention_sequence", [])],
            kg_features=[float(value) for value in payload.get("kg_features", [])],
            kg_guideline_score=float(payload.get("kg_guideline_score", 0.0)),
            metadata=dict(payload.get("metadata", {})),
        )

    @staticmethod
    def _intervention_component_record_to_dict(record: InterventionComponentRecord) -> Dict[str, object]:
        return {
            "component_code": str(record.component_code),
            "component_type": str(record.component_type),
            "static_values_by_index": {
                str(key): float(value)
                for key, value in dict(record.static_values_by_index).items()
            },
            "sequence_values_by_index": {
                str(key): [float(value) for value in values]
                for key, values in dict(record.sequence_values_by_index).items()
            },
            "static_feature_names_by_index": {
                str(key): str(value)
                for key, value in dict(record.static_feature_names_by_index).items()
            },
            "sequence_feature_names_by_index": {
                str(key): str(value)
                for key, value in dict(record.sequence_feature_names_by_index).items()
            },
            "summary": dict(record.summary),
        }

    @staticmethod
    def _intervention_component_record_from_dict(payload: Dict[str, object]) -> InterventionComponentRecord:
        return InterventionComponentRecord(
            component_code=str(payload.get("component_code", "")),
            component_type=str(payload.get("component_type", "")),
            static_values_by_index={
                str(key): float(value)
                for key, value in dict(payload.get("static_values_by_index", {})).items()
            },
            sequence_values_by_index={
                str(key): [float(value) for value in values]
                for key, values in dict(payload.get("sequence_values_by_index", {})).items()
            },
            static_feature_names_by_index={
                str(key): str(value)
                for key, value in dict(payload.get("static_feature_names_by_index", {})).items()
            },
            sequence_feature_names_by_index={
                str(key): str(value)
                for key, value in dict(payload.get("sequence_feature_names_by_index", {})).items()
            },
            summary=dict(payload.get("summary", {})),
        )

    def _intervention_component_stores_payload(self) -> Dict[str, object]:
        return {
            "schema_version": 2,
            "plan_store": dict(self.intervention_plan_store),
            "method_store": {
                code: self._intervention_component_record_to_dict(record)
                for code, record in self.intervention_method_store.items()
            },
            "dose_store": {
                code: self._intervention_component_record_to_dict(record)
                for code, record in self.intervention_dose_store.items()
            },
            "timing_store": {
                code: self._intervention_component_record_to_dict(record)
                for code, record in self.intervention_timing_store.items()
            },
            "context_store": {
                code: self._intervention_component_record_to_dict(record)
                for code, record in self.intervention_context_store.items()
            },
        }

    def _load_intervention_component_stores(self, payload: Dict[str, object]) -> None:
        self.intervention_plan_store = {
            str(key): dict(value)
            for key, value in dict(payload.get("plan_store", {})).items()
        }
        self.intervention_method_store = {
            str(key): self._intervention_component_record_from_dict(dict(value))
            for key, value in dict(payload.get("method_store", {})).items()
        }
        self.intervention_dose_store = {
            str(key): self._intervention_component_record_from_dict(dict(value))
            for key, value in dict(payload.get("dose_store", {})).items()
        }
        self.intervention_timing_store = {
            str(key): self._intervention_component_record_from_dict(dict(value))
            for key, value in dict(payload.get("timing_store", {})).items()
        }
        self.intervention_context_store = {
            str(key): self._intervention_component_record_from_dict(dict(value))
            for key, value in dict(payload.get("context_store", {})).items()
        }

    def _intervention_component_store(self, component_type: str) -> Dict[str, InterventionComponentRecord]:
        if component_type == "method":
            return self.intervention_method_store
        if component_type == "dose":
            return self.intervention_dose_store
        if component_type == "timing":
            return self.intervention_timing_store
        return self.intervention_context_store

    def _intervention_feature_component_type(self, feature_name: str, is_sequence: bool) -> str:
        name = str(feature_name).strip().lower()
        if any(token in name for token in ["offset", "minute", "hour", "time", "timing", "duration", "course", "window"]):
            return "timing"
        if any(token in name for token in ["dose", "dosage", "amount", "rate", "score", "intensity", "level", "mean", "max", "last"]):
            if "any" not in name and "flag" not in name:
                return "dose"
        if any(token in name for token in ["antibiotic", "antimicrobial", "vasopressor", "resp", "vent", "blood_culture", "lactate", "map", "sofa", "monitor", "exam", "treat", "any", "flag"]):
            return "method"
        return "context" if not is_sequence else "dose"

    def _intervention_component_index_groups(self) -> Dict[str, Dict[str, List[int]]]:
        groups: Dict[str, Dict[str, List[int]]] = {
            "method": {"static": [], "sequence": []},
            "dose": {"static": [], "sequence": []},
            "timing": {"static": [], "sequence": []},
            "context": {"static": [], "sequence": []},
        }
        for index, name in enumerate(self.intervention_feature_names):
            component_type = self._intervention_feature_component_type(name, is_sequence=False)
            groups[component_type]["static"].append(index)
        for index, name in enumerate(self.intervention_sequence_feature_names):
            component_type = self._intervention_feature_component_type(name, is_sequence=True)
            groups[component_type]["sequence"].append(index)
        return groups

    @staticmethod
    def _stable_component_code(prefix: str, payload: Dict[str, object]) -> str:
        digest = hashlib.sha1(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return f"{prefix}_{digest[:16]}"

    def _build_intervention_component_record(
        self,
        component_type: str,
        static_indices: Sequence[int],
        sequence_indices: Sequence[int],
        intervention_static: Sequence[float],
        intervention_sequence: Sequence[Sequence[float]],
        base_flags: Optional[Dict[str, object]] = None,
    ) -> InterventionComponentRecord:
        static_values = {
            str(index): float(intervention_static[index])
            for index in static_indices
            if 0 <= int(index) < len(intervention_static)
        }
        sequence_values = {
            str(index): [
                float(step[index]) if 0 <= int(index) < len(step) else 0.0
                for step in intervention_sequence
            ]
            for index in sequence_indices
        }
        static_names = {
            str(index): str(self.intervention_feature_names[index])
            for index in static_indices
            if 0 <= int(index) < len(self.intervention_feature_names)
        }
        sequence_names = {
            str(index): str(self.intervention_sequence_feature_names[index])
            for index in sequence_indices
            if 0 <= int(index) < len(self.intervention_sequence_feature_names)
        }
        action_profile = self._intervention_action_profile(
            intervention_static=intervention_static,
            intervention_sequence=intervention_sequence,
            base_flags=base_flags,
        )
        summary = {
            "component_type": component_type,
            "static_feature_count": float(len(static_values)),
            "sequence_feature_count": float(len(sequence_values)),
            "sequence_length": float(len(intervention_sequence)),
            "static_strength": float(self._mean_abs(list(static_values.values()))),
            "sequence_strength": float(
                self._mean_abs([value for values in sequence_values.values() for value in values])
            ),
            "active_action_profile": {
                key: float(value)
                for key, value in action_profile.items()
                if float(value) > 0.0
            },
        }
        code_payload = {
            "component_type": component_type,
            "static_values_by_index": {
                key: round(float(value), 8) for key, value in static_values.items()
            },
            "sequence_values_by_index": {
                key: [round(float(value), 8) for value in values]
                for key, values in sequence_values.items()
            },
            "static_feature_names_by_index": static_names,
            "sequence_feature_names_by_index": sequence_names,
        }
        component_code = self._stable_component_code(f"int_{component_type}", code_payload)
        return InterventionComponentRecord(
            component_code=component_code,
            component_type=component_type,
            static_values_by_index=static_values,
            sequence_values_by_index=sequence_values,
            static_feature_names_by_index=static_names,
            sequence_feature_names_by_index=sequence_names,
            summary=summary,
        )

    def _register_intervention_component(self, record: InterventionComponentRecord) -> str:
        store = self._intervention_component_store(record.component_type)
        if record.component_code not in store:
            store[record.component_code] = record
        return record.component_code

    def _store_intervention_plan_components(
        self,
        intervention_static: Sequence[float],
        intervention_sequence: Sequence[Sequence[float]],
        base_flags: Optional[Dict[str, object]] = None,
    ) -> Dict[str, str]:
        static_values = [float(value) for value in intervention_static]
        sequence_values = [[float(value) for value in step] for step in intervention_sequence]
        groups = self._intervention_component_index_groups()
        component_codes: Dict[str, str] = {}
        for component_type in ["method", "dose", "timing", "context"]:
            record = self._build_intervention_component_record(
                component_type=component_type,
                static_indices=groups[component_type]["static"],
                sequence_indices=groups[component_type]["sequence"],
                intervention_static=static_values,
                intervention_sequence=sequence_values,
                base_flags=base_flags,
            )
            component_codes[component_type] = self._register_intervention_component(record)

        plan_payload = {
            "schema_version": 2,
            "static_dim": int(len(static_values)),
            "sequence_dim": int(len(self.intervention_sequence_feature_names)),
            "sequence_length": int(len(sequence_values)),
            "components": dict(component_codes),
        }
        plan_code = self._stable_component_code("int_plan", plan_payload)
        self.intervention_plan_store[plan_code] = dict(plan_payload)
        return {
            "plan": plan_code,
            **component_codes,
        }

    def _reconstruct_intervention_plan_from_code(
        self,
        intervention_plan_code: Dict[str, str],
    ) -> tuple[List[float], List[List[float]]]:
        plan_code = str(dict(intervention_plan_code or {}).get("plan", ""))
        plan_record = dict(self.intervention_plan_store.get(plan_code, {}))
        static_dim = int(plan_record.get("static_dim", len(self.intervention_feature_names)))
        sequence_dim = int(plan_record.get("sequence_dim", len(self.intervention_sequence_feature_names)))
        sequence_length = int(plan_record.get("sequence_length", 0))
        if sequence_length <= 0:
            sequence_length = max(0, int(self.trainer_config.history_length)) if sequence_dim > 0 else 0
        reconstructed_static = [0.0] * max(0, static_dim)
        reconstructed_sequence = [
            [0.0] * max(0, sequence_dim)
            for _ in range(max(0, sequence_length))
        ]

        component_codes = dict(plan_record.get("components", {}))
        for component_type in ["method", "dose", "timing", "context"]:
            component_code = str(
                component_codes.get(component_type)
                or dict(intervention_plan_code or {}).get(component_type, "")
            )
            if not component_code:
                continue
            record = self._intervention_component_store(component_type).get(component_code)
            if record is None:
                continue
            for index_text, value in record.static_values_by_index.items():
                index = int(index_text)
                if 0 <= index < len(reconstructed_static):
                    reconstructed_static[index] = float(value)
            for index_text, values in record.sequence_values_by_index.items():
                index = int(index_text)
                if index < 0:
                    continue
                if not reconstructed_sequence and sequence_dim > 0:
                    reconstructed_sequence = [[0.0] * sequence_dim for _ in range(len(values))]
                for step_index, value in enumerate(values):
                    while step_index >= len(reconstructed_sequence):
                        reconstructed_sequence.append([0.0] * max(0, sequence_dim))
                    if 0 <= index < len(reconstructed_sequence[step_index]):
                        reconstructed_sequence[step_index][index] = float(value)
        return reconstructed_static, reconstructed_sequence

    def _materialize_intervention_entry(self, entry: InterventionStoreEntry) -> InterventionStoreEntry:
        if entry.intervention_plan_code:
            plan_code = str(entry.intervention_plan_code.get("plan", ""))
            if plan_code and plan_code not in self.intervention_plan_store:
                return entry
            reconstructed_static, reconstructed_sequence = self._reconstruct_intervention_plan_from_code(
                entry.intervention_plan_code
            )
            if reconstructed_static or reconstructed_sequence:
                entry.intervention_static = reconstructed_static
                entry.intervention_sequence = reconstructed_sequence
        elif entry.intervention_static or entry.intervention_sequence:
            entry.intervention_plan_code = self._store_intervention_plan_components(
                intervention_static=entry.intervention_static,
                intervention_sequence=entry.intervention_sequence,
                base_flags=dict(entry.metadata.get("kg_flags", {})),
            )
        return entry

    @staticmethod
    def _transition_store_entry_to_dict(entry: TransitionStoreEntry) -> Dict[str, object]:
        return {
            "experience_label": int(entry.experience_label),
            "state_vector": [float(value) for value in entry.state_vector],
            "action_vector": [float(value) for value in entry.action_vector],
            "future_curve": [float(value) for value in entry.future_curve],
            "transition_utility": float(entry.transition_utility),
            "utility_vector": [float(value) for value in entry.utility_vector],
            "support": int(entry.support),
            "metadata": dict(entry.metadata),
        }

    @staticmethod
    def _transition_store_entry_from_dict(payload: Dict[str, object]) -> TransitionStoreEntry:
        return TransitionStoreEntry(
            experience_label=int(payload.get("experience_label", 0)),
            state_vector=[float(value) for value in payload.get("state_vector", [])],
            action_vector=[float(value) for value in payload.get("action_vector", [])],
            future_curve=[float(value) for value in payload.get("future_curve", [])],
            transition_utility=float(payload.get("transition_utility", 0.0)),
            utility_vector=[float(value) for value in payload.get("utility_vector", [])],
            support=int(payload.get("support", 1)),
            metadata=dict(payload.get("metadata", {})),
        )

    def _rebuild_intervention_store_cache(self) -> None:
        self.intervention_store_by_label = {}
        self.intervention_store_by_pattern = {}
        self.intervention_store_by_trajectory = {}
        self.intervention_store_by_hospital = {}
        self.intervention_store_by_unit_type = {}
        self.intervention_store_by_infection_anchor = {}
        self.intervention_store_embedding_cache = {}
        for index, entry in enumerate(self.intervention_store_entries):
            entry = self._materialize_intervention_entry(entry)
            self.intervention_store_entries[index] = entry
            self.intervention_store_by_label.setdefault(int(entry.experience_label), []).append(index)
            self.intervention_store_by_pattern.setdefault(int(entry.pattern_label), []).append(index)
            self.intervention_store_by_trajectory.setdefault(int(entry.trajectory_label), []).append(index)
            hospital_key = self._counterfactual_pool_key(entry.metadata.get("hospitalid"))
            if hospital_key:
                self.intervention_store_by_hospital.setdefault(hospital_key, []).append(index)
            unit_key = self._counterfactual_pool_key(entry.metadata.get("unittype"))
            if unit_key:
                self.intervention_store_by_unit_type.setdefault(unit_key, []).append(index)
            infection_key = self._counterfactual_pool_key(entry.metadata.get("infection_anchor_type"))
            if infection_key:
                self.intervention_store_by_infection_anchor.setdefault(infection_key, []).append(index)
        for label, indices in self.intervention_store_by_label.items():
            self.intervention_store_embedding_cache[label] = torch.tensor(
                [self.intervention_store_entries[index].patient_embedding for index in indices],
                dtype=torch.float32,
                device=self.device,
            )

    @staticmethod
    def _counterfactual_pool_key(value: object) -> str:
        if value is None:
            return ""
        try:
            numeric = float(value)
            if math.isnan(numeric):
                return ""
            if abs(numeric - round(numeric)) < 1e-9:
                return str(int(round(numeric)))
            return f"{numeric:.6f}"
        except (TypeError, ValueError):
            text = str(value).strip().lower()
            if text in {"", "nan", "none", "unknown"}:
                return ""
            return text

    def enrich_intervention_store_metadata_from_labels(self, labels_csv: str) -> None:
        labels_path = str(labels_csv).strip()
        if not labels_path:
            return
        frame = pd.read_csv(labels_path)
        if "patientunitstayid" not in frame.columns:
            return
        contextual_columns = [
            "patientunitstayid",
            "hospitalid",
            "wardid",
            "unittype",
            "infection_anchor_type",
            "infection_anchor_value",
            "suspected_infection_from",
        ]
        available_columns = [column for column in contextual_columns if column in frame.columns]
        if "patientunitstayid" not in available_columns:
            return
        frame = frame[available_columns].drop_duplicates(subset=["patientunitstayid"]).copy()
        frame["patientunitstayid"] = pd.to_numeric(frame["patientunitstayid"], errors="coerce")
        frame = frame.dropna(subset=["patientunitstayid"])
        lookup = {
            int(row["patientunitstayid"]): row.to_dict()
            for _, row in frame.iterrows()
        }
        for entry in self.intervention_store_entries:
            stay_id = int(round(float(entry.stay_id)))
            row = lookup.get(stay_id)
            if not row:
                continue
            for key in contextual_columns:
                if key == "patientunitstayid" or key not in row:
                    continue
                value = row.get(key)
                if value is None:
                    continue
                try:
                    if pd.isna(value):
                        continue
                except Exception:
                    pass
                if isinstance(value, str):
                    value = value.strip()
                    if not value:
                        continue
                entry.metadata[key] = value
        self._rebuild_intervention_store_cache()

    def _rebuild_transition_store_cache(self) -> None:
        self.transition_store_by_label = {}
        self.transition_state_cache = {}
        self.transition_action_cache = {}
        self.transition_future_cache = {}
        self.transition_utility_cache = {}
        self.transition_utility_vector_cache = {}
        self.transition_future_mean_cache = {}
        self.transition_label_support = {}
        self.transition_label_utility_mean = {}
        self.transition_label_improvement_rate = {}
        self.transition_label_utility_vector_mean = {}
        self.transition_signature_cache = {}
        self.transition_signature_payload_cache = {}
        self.transition_signature_counts = {}
        for index, entry in enumerate(self.transition_store_entries):
            self.transition_store_by_label.setdefault(int(entry.experience_label), []).append(index)
            signature = str(entry.metadata.get("clinical_state_signature", ""))
            if signature:
                self.transition_signature_counts[signature] = self.transition_signature_counts.get(signature, 0) + 1
        for label, indices in self.transition_store_by_label.items():
            self.transition_state_cache[label] = torch.tensor(
                [self.transition_store_entries[index].state_vector for index in indices],
                dtype=torch.float32,
                device=self.device,
            )
            self.transition_action_cache[label] = torch.tensor(
                [self.transition_store_entries[index].action_vector for index in indices],
                dtype=torch.float32,
                device=self.device,
            )
            self.transition_future_cache[label] = torch.tensor(
                [self.transition_store_entries[index].future_curve for index in indices],
                dtype=torch.float32,
                device=self.device,
            )
            self.transition_utility_cache[label] = torch.tensor(
                [self.transition_store_entries[index].transition_utility for index in indices],
                dtype=torch.float32,
                device=self.device,
            )
            self.transition_utility_vector_cache[label] = torch.tensor(
                [self.transition_store_entries[index].utility_vector for index in indices],
                dtype=torch.float32,
                device=self.device,
            )
            self.transition_future_mean_cache[label] = self.transition_future_cache[label].mean(dim=0)
            utilities = [self.transition_store_entries[index].transition_utility for index in indices]
            self.transition_label_support[label] = int(len(indices))
            self.transition_label_utility_mean[label] = float(sum(utilities) / max(1, len(utilities)))
            self.transition_label_improvement_rate[label] = float(
                sum(1.0 for value in utilities if value > 0.0) / max(1, len(utilities))
            )
            self.transition_label_utility_vector_mean[label] = self.transition_utility_vector_cache[label].mean(dim=0)
            self.transition_signature_cache[label] = [
                str(self.transition_store_entries[index].metadata.get("clinical_state_signature", ""))
                for index in indices
            ]
            self.transition_signature_payload_cache[label] = [
                dict(self.transition_store_entries[index].metadata.get("clinical_state_signature_features", {}))
                for index in indices
            ]

    def _runtime_settings_payload(self) -> Dict[str, object]:
        return {
            "temporal_profile_feature_names": list(self.temporal_profile_feature_names),
            "branch_decoupling_mode": str(self.branch_decoupling_mode),
            "branch_decoupling_enabled": bool(self.branch_decoupling_enabled),
            "factual_projection_scale": float(self.factual_projection_scale),
            "retrieval_projection_scale": float(self.retrieval_projection_scale),
            "retrieval_head_delta_scale": float(self.retrieval_head_delta_scale),
            "factual_branch_scale": float(self.factual_branch_scale),
            "retrieval_branch_scale": float(self.retrieval_branch_scale),
            "factual_temporal_path_scale": float(self.factual_temporal_path_scale),
            "factual_prediction_trunk_scale": float(self.factual_prediction_trunk_scale),
            "factual_calibration_weight": float(self.factual_calibration_weight),
            "enable_explicit_kg_path": bool(self.enable_explicit_kg_path),
            "kg_residual_weight": float(self.kg_residual_weight),
            "kg_alignment_floor": float(self.kg_alignment_floor),
            "memory_direct_residual_weight": float(self.memory_direct_residual_weight),
            "memory_direct_residual_mode": str(self.memory_direct_residual_mode),
            "memory_path_coordination_mode": str(self.memory_path_coordination_mode),
            "memory_harm_control_enabled": bool(self.memory_harm_control_enabled),
            "memory_quality_floor": float(self.memory_quality_floor),
            "memory_min_path_alignment": float(self.memory_min_path_alignment),
            "memory_residual_cap_ratio": float(self.memory_residual_cap_ratio),
            "harm_stable_quality_boost": float(self.harm_stable_quality_boost),
            "harm_flat_quality_boost": float(self.harm_flat_quality_boost),
            "counterfactual_donor_score_mode": str(self.counterfactual_donor_score_mode),
            "counterfactual_base_similarity_weight": float(self.counterfactual_base_similarity_weight),
            "counterfactual_kg_similarity_weight": float(self.counterfactual_kg_similarity_weight),
            "counterfactual_guideline_weight": float(self.counterfactual_guideline_weight),
            "counterfactual_guideline_score_weight": float(self.counterfactual_guideline_score_weight),
            "counterfactual_penalty_weight": float(self.counterfactual_penalty_weight),
            "counterfactual_hard_filter_enabled": bool(self.counterfactual_hard_filter_enabled),
            "counterfactual_candidate_policy": str(self.counterfactual_candidate_policy),
            "counterfactual_candidate_search_mode": str(self.counterfactual_candidate_search_mode),
            "counterfactual_candidate_feasibility_weight": float(self.counterfactual_candidate_feasibility_weight),
            "counterfactual_candidate_penalty_weight": float(self.counterfactual_candidate_penalty_weight),
            "counterfactual_search_template_limit": int(self.counterfactual_search_template_limit),
            "counterfactual_search_guideline_drop_max": float(self.counterfactual_search_guideline_drop_max),
            "counterfactual_search_missing_care_tolerance": float(self.counterfactual_search_missing_care_tolerance),
            "counterfactual_search_overlap_drop_max": float(self.counterfactual_search_overlap_drop_max),
            "counterfactual_parameterized_template_limit": int(self.counterfactual_parameterized_template_limit),
            "counterfactual_rollout_steps": int(self.counterfactual_rollout_steps),
            "counterfactual_rollout_discount": float(self.counterfactual_rollout_discount),
            "counterfactual_label_top_k": int(self.counterfactual_label_top_k),
            "counterfactual_pool_mode": str(self.counterfactual_pool_mode),
            "counterfactual_pool_include_pattern": bool(self.counterfactual_pool_include_pattern),
            "counterfactual_pool_include_trajectory": bool(self.counterfactual_pool_include_trajectory),
            "counterfactual_pool_include_hospital": bool(self.counterfactual_pool_include_hospital),
            "counterfactual_pool_include_unit_type": bool(self.counterfactual_pool_include_unit_type),
            "counterfactual_pool_include_infection_anchor": bool(self.counterfactual_pool_include_infection_anchor),
            "counterfactual_pool_enable_global_backfill": bool(self.counterfactual_pool_enable_global_backfill),
            "counterfactual_pool_local_min_candidates": int(self.counterfactual_pool_local_min_candidates),
            "counterfactual_pool_min_candidates": int(self.counterfactual_pool_min_candidates),
            "counterfactual_pool_global_limit": int(self.counterfactual_pool_global_limit),
            "counterfactual_pool_prefilter_top_k": int(self.counterfactual_pool_prefilter_top_k),
            "counterfactual_pool_match_weight": float(self.counterfactual_pool_match_weight),
            "counterfactual_pool_same_hospital_weight": float(self.counterfactual_pool_same_hospital_weight),
            "counterfactual_pool_same_unit_weight": float(self.counterfactual_pool_same_unit_weight),
            "counterfactual_pool_same_infection_anchor_weight": float(
                self.counterfactual_pool_same_infection_anchor_weight
            ),
            "counterfactual_overlap_filter_enabled": bool(self.counterfactual_overlap_filter_enabled),
            "counterfactual_overlap_fallback_enabled": bool(self.counterfactual_overlap_fallback_enabled),
            "counterfactual_overlap_weight": float(self.counterfactual_overlap_weight),
            "counterfactual_overlap_severity_gap_max": float(self.counterfactual_overlap_severity_gap_max),
            "counterfactual_overlap_trend_gap_max": float(self.counterfactual_overlap_trend_gap_max),
            "counterfactual_overlap_state_min": float(self.counterfactual_overlap_state_min),
            "counterfactual_overlap_action_min": float(self.counterfactual_overlap_action_min),
            "counterfactual_neighbor_top_k": int(self.counterfactual_neighbor_top_k),
            "counterfactual_neighbor_weight": float(self.counterfactual_neighbor_weight),
            "counterfactual_neighbor_penalty_weight": float(self.counterfactual_neighbor_penalty_weight),
            "counterfactual_neighbor_min_consistency": float(self.counterfactual_neighbor_min_consistency),
            "counterfactual_neighbor_similarity_band": float(self.counterfactual_neighbor_similarity_band),
            "counterfactual_neighbor_self_weight": float(self.counterfactual_neighbor_self_weight),
            "counterfactual_reranker_mode": str(self.counterfactual_reranker_mode),
            "counterfactual_reranker_blend_weight": float(self.counterfactual_reranker_blend_weight),
            "counterfactual_reranker_train_top_k": int(self.counterfactual_reranker_train_top_k),
            "counterfactual_reranker_max_samples": int(self.counterfactual_reranker_max_samples),
            "counterfactual_reranker_min_examples": int(self.counterfactual_reranker_min_examples),
            "counterfactual_reranker_ridge_l2": float(self.counterfactual_reranker_ridge_l2),
            "counterfactual_reranker_state": dict(self.counterfactual_reranker_state),
            "enable_transition_memory": bool(self.enable_transition_memory),
            "enable_transition_factual_path": bool(self.enable_transition_factual_path),
            "enable_transition_donor_path": bool(self.enable_transition_donor_path),
            "enable_transition_trunk_path": bool(self.enable_transition_trunk_path),
            "transition_top_k": int(self.transition_top_k),
            "transition_state_weight": float(self.transition_state_weight),
            "transition_action_weight": float(self.transition_action_weight),
            "transition_score_weight": float(self.transition_score_weight),
            "transition_template_blend_weight": float(self.transition_template_blend_weight),
            "transition_selection_weight": float(self.transition_selection_weight),
            "transition_temperature": float(self.transition_temperature),
            "transition_utility_scale": float(self.transition_utility_scale),
            "transition_utility_alignment_weight": float(self.transition_utility_alignment_weight),
            "transition_anchor_blend_weight": float(self.transition_anchor_blend_weight),
            "transition_signature_match_weight": float(self.transition_signature_match_weight),
            "transition_min_confidence": float(self.transition_min_confidence),
            "transition_min_support": float(self.transition_min_support),
            "transition_min_expected_utility": float(self.transition_min_expected_utility),
            "transition_min_signature_weight": float(self.transition_min_signature_weight),
            "transition_stable_regime_penalty": float(self.transition_stable_regime_penalty),
            "transition_flat_pattern_penalty": float(self.transition_flat_pattern_penalty),
            "transition_utility_bias": float(self.transition_utility_bias),
            "transition_utility_temperature": float(self.transition_utility_temperature),
            "transition_signature_partial_match": bool(self.transition_signature_partial_match),
            "transition_residual_cap_ratio": float(self.transition_residual_cap_ratio),
            "transition_trunk_weight": float(self.transition_trunk_weight),
            "transition_factual_residual_mode": str(self.transition_factual_residual_mode),
            "transition_positive_only": bool(self.transition_positive_only),
            "transition_action_change_weight": float(self.transition_action_change_weight),
            "transition_candidate_action_change_weight": float(self.transition_candidate_action_change_weight),
            "sequence_feature_names": list(self.sequence_feature_names),
            "patient_feature_names": list(self.patient_feature_names),
            "intervention_feature_names": list(self.intervention_feature_names),
            "intervention_sequence_feature_names": list(self.intervention_sequence_feature_names),
        }

    def _load_runtime_settings(self, payload: Dict[str, object]) -> None:
        self.temporal_profile_feature_names = list(
            payload.get("temporal_profile_feature_names", self.temporal_profile_feature_names)
        )
        self.temporal_profile_dim = len(self.temporal_profile_feature_names)
        self.branch_decoupling_mode = str(payload.get("branch_decoupling_mode", self.branch_decoupling_mode))
        self.branch_decoupling_enabled = bool(payload.get("branch_decoupling_enabled", self.branch_decoupling_enabled))
        self.factual_projection_scale = float(payload.get("factual_projection_scale", self.factual_projection_scale))
        self.retrieval_projection_scale = float(payload.get("retrieval_projection_scale", self.retrieval_projection_scale))
        self.retrieval_head_delta_scale = float(payload.get("retrieval_head_delta_scale", self.retrieval_head_delta_scale))
        self.factual_branch_scale = float(payload.get("factual_branch_scale", self.factual_branch_scale))
        self.retrieval_branch_scale = float(payload.get("retrieval_branch_scale", self.retrieval_branch_scale))
        self.factual_temporal_path_scale = float(
            payload.get("factual_temporal_path_scale", self.factual_temporal_path_scale)
        )
        self.factual_prediction_trunk_scale = float(
            payload.get("factual_prediction_trunk_scale", self.factual_prediction_trunk_scale)
        )
        self.factual_calibration_weight = float(
            payload.get("factual_calibration_weight", self.factual_calibration_weight)
        )
        self.enable_explicit_kg_path = bool(payload.get("enable_explicit_kg_path", self.enable_explicit_kg_path))
        self.kg_residual_weight = float(payload.get("kg_residual_weight", self.kg_residual_weight))
        self.kg_alignment_floor = float(payload.get("kg_alignment_floor", self.kg_alignment_floor))
        self.memory_direct_residual_weight = float(
            payload.get("memory_direct_residual_weight", self.memory_direct_residual_weight)
        )
        self.memory_direct_residual_mode = str(payload.get("memory_direct_residual_mode", self.memory_direct_residual_mode))
        self.memory_path_coordination_mode = str(
            payload.get("memory_path_coordination_mode", self.memory_path_coordination_mode)
        )
        self.memory_harm_control_enabled = bool(
            payload.get("memory_harm_control_enabled", self.memory_harm_control_enabled)
        )
        self.memory_quality_floor = float(payload.get("memory_quality_floor", self.memory_quality_floor))
        self.memory_min_path_alignment = float(
            payload.get("memory_min_path_alignment", self.memory_min_path_alignment)
        )
        self.memory_residual_cap_ratio = float(
            payload.get("memory_residual_cap_ratio", self.memory_residual_cap_ratio)
        )
        self.harm_stable_quality_boost = float(
            payload.get("harm_stable_quality_boost", self.harm_stable_quality_boost)
        )
        self.harm_flat_quality_boost = float(
            payload.get("harm_flat_quality_boost", self.harm_flat_quality_boost)
        )
        self.counterfactual_donor_score_mode = str(
            payload.get("counterfactual_donor_score_mode", self.counterfactual_donor_score_mode)
        )
        self.counterfactual_base_similarity_weight = float(
            payload.get("counterfactual_base_similarity_weight", self.counterfactual_base_similarity_weight)
        )
        self.counterfactual_kg_similarity_weight = float(
            payload.get("counterfactual_kg_similarity_weight", self.counterfactual_kg_similarity_weight)
        )
        self.counterfactual_guideline_weight = float(
            payload.get("counterfactual_guideline_weight", self.counterfactual_guideline_weight)
        )
        self.counterfactual_guideline_score_weight = float(
            payload.get("counterfactual_guideline_score_weight", self.counterfactual_guideline_score_weight)
        )
        self.counterfactual_penalty_weight = float(
            payload.get("counterfactual_penalty_weight", self.counterfactual_penalty_weight)
        )
        self.counterfactual_hard_filter_enabled = bool(
            payload.get("counterfactual_hard_filter_enabled", self.counterfactual_hard_filter_enabled)
        )
        self.counterfactual_candidate_policy = str(
            payload.get("counterfactual_candidate_policy", self.counterfactual_candidate_policy)
        )
        self.counterfactual_candidate_search_mode = str(
            payload.get("counterfactual_candidate_search_mode", self.counterfactual_candidate_search_mode)
        )
        self.counterfactual_candidate_feasibility_weight = float(
            payload.get(
                "counterfactual_candidate_feasibility_weight",
                self.counterfactual_candidate_feasibility_weight,
            )
        )
        self.counterfactual_candidate_penalty_weight = float(
            payload.get("counterfactual_candidate_penalty_weight", self.counterfactual_candidate_penalty_weight)
        )
        self.counterfactual_search_template_limit = int(
            payload.get("counterfactual_search_template_limit", self.counterfactual_search_template_limit)
        )
        self.counterfactual_search_guideline_drop_max = float(
            payload.get("counterfactual_search_guideline_drop_max", self.counterfactual_search_guideline_drop_max)
        )
        self.counterfactual_search_missing_care_tolerance = float(
            payload.get(
                "counterfactual_search_missing_care_tolerance",
                self.counterfactual_search_missing_care_tolerance,
            )
        )
        self.counterfactual_search_overlap_drop_max = float(
            payload.get("counterfactual_search_overlap_drop_max", self.counterfactual_search_overlap_drop_max)
        )
        self.counterfactual_parameterized_template_limit = int(
            payload.get(
                "counterfactual_parameterized_template_limit",
                self.counterfactual_parameterized_template_limit,
            )
        )
        self.counterfactual_rollout_steps = int(
            payload.get("counterfactual_rollout_steps", self.counterfactual_rollout_steps)
        )
        self.counterfactual_rollout_discount = float(
            payload.get("counterfactual_rollout_discount", self.counterfactual_rollout_discount)
        )
        self.counterfactual_label_top_k = int(payload.get("counterfactual_label_top_k", self.counterfactual_label_top_k))
        self.counterfactual_pool_mode = str(payload.get("counterfactual_pool_mode", self.counterfactual_pool_mode))
        self.counterfactual_pool_include_pattern = bool(
            payload.get("counterfactual_pool_include_pattern", self.counterfactual_pool_include_pattern)
        )
        self.counterfactual_pool_include_trajectory = bool(
            payload.get("counterfactual_pool_include_trajectory", self.counterfactual_pool_include_trajectory)
        )
        self.counterfactual_pool_include_hospital = bool(
            payload.get("counterfactual_pool_include_hospital", self.counterfactual_pool_include_hospital)
        )
        self.counterfactual_pool_include_unit_type = bool(
            payload.get("counterfactual_pool_include_unit_type", self.counterfactual_pool_include_unit_type)
        )
        self.counterfactual_pool_include_infection_anchor = bool(
            payload.get(
                "counterfactual_pool_include_infection_anchor",
                self.counterfactual_pool_include_infection_anchor,
            )
        )
        self.counterfactual_pool_enable_global_backfill = bool(
            payload.get("counterfactual_pool_enable_global_backfill", self.counterfactual_pool_enable_global_backfill)
        )
        self.counterfactual_pool_local_min_candidates = int(
            payload.get("counterfactual_pool_local_min_candidates", self.counterfactual_pool_local_min_candidates)
        )
        self.counterfactual_pool_min_candidates = int(
            payload.get("counterfactual_pool_min_candidates", self.counterfactual_pool_min_candidates)
        )
        self.counterfactual_pool_global_limit = int(
            payload.get("counterfactual_pool_global_limit", self.counterfactual_pool_global_limit)
        )
        self.counterfactual_pool_prefilter_top_k = int(
            payload.get("counterfactual_pool_prefilter_top_k", self.counterfactual_pool_prefilter_top_k)
        )
        self.counterfactual_pool_match_weight = float(
            payload.get("counterfactual_pool_match_weight", self.counterfactual_pool_match_weight)
        )
        self.counterfactual_pool_same_hospital_weight = float(
            payload.get(
                "counterfactual_pool_same_hospital_weight",
                self.counterfactual_pool_same_hospital_weight,
            )
        )
        self.counterfactual_pool_same_unit_weight = float(
            payload.get("counterfactual_pool_same_unit_weight", self.counterfactual_pool_same_unit_weight)
        )
        self.counterfactual_pool_same_infection_anchor_weight = float(
            payload.get(
                "counterfactual_pool_same_infection_anchor_weight",
                self.counterfactual_pool_same_infection_anchor_weight,
            )
        )
        self.counterfactual_overlap_filter_enabled = bool(
            payload.get("counterfactual_overlap_filter_enabled", self.counterfactual_overlap_filter_enabled)
        )
        self.counterfactual_overlap_fallback_enabled = bool(
            payload.get("counterfactual_overlap_fallback_enabled", self.counterfactual_overlap_fallback_enabled)
        )
        self.counterfactual_overlap_weight = float(
            payload.get("counterfactual_overlap_weight", self.counterfactual_overlap_weight)
        )
        self.counterfactual_overlap_severity_gap_max = float(
            payload.get("counterfactual_overlap_severity_gap_max", self.counterfactual_overlap_severity_gap_max)
        )
        self.counterfactual_overlap_trend_gap_max = float(
            payload.get("counterfactual_overlap_trend_gap_max", self.counterfactual_overlap_trend_gap_max)
        )
        self.counterfactual_overlap_state_min = float(
            payload.get("counterfactual_overlap_state_min", self.counterfactual_overlap_state_min)
        )
        self.counterfactual_overlap_action_min = float(
            payload.get("counterfactual_overlap_action_min", self.counterfactual_overlap_action_min)
        )
        self.counterfactual_neighbor_top_k = int(
            payload.get("counterfactual_neighbor_top_k", self.counterfactual_neighbor_top_k)
        )
        self.counterfactual_neighbor_weight = float(
            payload.get("counterfactual_neighbor_weight", self.counterfactual_neighbor_weight)
        )
        self.counterfactual_neighbor_penalty_weight = float(
            payload.get(
                "counterfactual_neighbor_penalty_weight",
                self.counterfactual_neighbor_penalty_weight,
            )
        )
        self.counterfactual_neighbor_min_consistency = float(
            payload.get(
                "counterfactual_neighbor_min_consistency",
                self.counterfactual_neighbor_min_consistency,
            )
        )
        self.counterfactual_neighbor_similarity_band = float(
            payload.get(
                "counterfactual_neighbor_similarity_band",
                self.counterfactual_neighbor_similarity_band,
            )
        )
        self.counterfactual_neighbor_self_weight = float(
            payload.get(
                "counterfactual_neighbor_self_weight",
                self.counterfactual_neighbor_self_weight,
            )
        )
        self.counterfactual_reranker_mode = str(
            payload.get("counterfactual_reranker_mode", self.counterfactual_reranker_mode)
        )
        self.counterfactual_reranker_blend_weight = float(
            payload.get("counterfactual_reranker_blend_weight", self.counterfactual_reranker_blend_weight)
        )
        self.counterfactual_reranker_train_top_k = int(
            payload.get("counterfactual_reranker_train_top_k", self.counterfactual_reranker_train_top_k)
        )
        self.counterfactual_reranker_max_samples = int(
            payload.get("counterfactual_reranker_max_samples", self.counterfactual_reranker_max_samples)
        )
        self.counterfactual_reranker_min_examples = int(
            payload.get("counterfactual_reranker_min_examples", self.counterfactual_reranker_min_examples)
        )
        self.counterfactual_reranker_ridge_l2 = float(
            payload.get("counterfactual_reranker_ridge_l2", self.counterfactual_reranker_ridge_l2)
        )
        reranker_state = payload.get("counterfactual_reranker_state", self.counterfactual_reranker_state)
        self.counterfactual_reranker_state = dict(reranker_state) if isinstance(reranker_state, dict) else {}
        self.enable_transition_memory = bool(payload.get("enable_transition_memory", self.enable_transition_memory))
        self.enable_transition_factual_path = bool(
            payload.get("enable_transition_factual_path", self.enable_transition_factual_path)
        )
        self.enable_transition_donor_path = bool(
            payload.get("enable_transition_donor_path", self.enable_transition_donor_path)
        )
        self.enable_transition_trunk_path = bool(
            payload.get("enable_transition_trunk_path", self.enable_transition_trunk_path)
        )
        self.transition_top_k = int(payload.get("transition_top_k", self.transition_top_k))
        self.transition_state_weight = float(payload.get("transition_state_weight", self.transition_state_weight))
        self.transition_action_weight = float(payload.get("transition_action_weight", self.transition_action_weight))
        self.transition_score_weight = float(payload.get("transition_score_weight", self.transition_score_weight))
        self.transition_template_blend_weight = float(
            payload.get("transition_template_blend_weight", self.transition_template_blend_weight)
        )
        self.transition_selection_weight = float(
            payload.get("transition_selection_weight", self.transition_selection_weight)
        )
        self.transition_temperature = float(payload.get("transition_temperature", self.transition_temperature))
        self.transition_utility_scale = float(payload.get("transition_utility_scale", self.transition_utility_scale))
        self.transition_utility_alignment_weight = float(
            payload.get("transition_utility_alignment_weight", self.transition_utility_alignment_weight)
        )
        self.transition_anchor_blend_weight = float(
            payload.get("transition_anchor_blend_weight", self.transition_anchor_blend_weight)
        )
        self.transition_signature_match_weight = float(
            payload.get("transition_signature_match_weight", self.transition_signature_match_weight)
        )
        self.transition_min_confidence = float(payload.get("transition_min_confidence", self.transition_min_confidence))
        self.transition_min_support = float(payload.get("transition_min_support", self.transition_min_support))
        self.transition_min_expected_utility = float(
            payload.get("transition_min_expected_utility", self.transition_min_expected_utility)
        )
        self.transition_min_signature_weight = float(
            payload.get("transition_min_signature_weight", self.transition_min_signature_weight)
        )
        self.transition_stable_regime_penalty = float(
            payload.get("transition_stable_regime_penalty", self.transition_stable_regime_penalty)
        )
        self.transition_flat_pattern_penalty = float(
            payload.get("transition_flat_pattern_penalty", self.transition_flat_pattern_penalty)
        )
        self.transition_utility_bias = float(
            payload.get("transition_utility_bias", self.transition_utility_bias)
        )
        self.transition_utility_temperature = float(
            payload.get("transition_utility_temperature", self.transition_utility_temperature)
        )
        self.transition_signature_partial_match = bool(
            payload.get("transition_signature_partial_match", self.transition_signature_partial_match)
        )
        self.transition_residual_cap_ratio = float(
            payload.get("transition_residual_cap_ratio", self.transition_residual_cap_ratio)
        )
        self.transition_trunk_weight = float(payload.get("transition_trunk_weight", self.transition_trunk_weight))
        self.transition_factual_residual_mode = str(
            payload.get("transition_factual_residual_mode", self.transition_factual_residual_mode)
        )
        self.transition_positive_only = bool(payload.get("transition_positive_only", self.transition_positive_only))
        self.transition_action_change_weight = float(
            payload.get("transition_action_change_weight", self.transition_action_change_weight)
        )
        self.transition_candidate_action_change_weight = float(
            payload.get(
                "transition_candidate_action_change_weight",
                self.transition_candidate_action_change_weight,
            )
        )
        self.sequence_feature_names = list(payload.get("sequence_feature_names", self.sequence_feature_names))
        self.patient_feature_names = list(payload.get("patient_feature_names", self.patient_feature_names))
        self.intervention_feature_names = list(payload.get("intervention_feature_names", self.intervention_feature_names))
        self.intervention_sequence_feature_names = list(
            payload.get("intervention_sequence_feature_names", self.intervention_sequence_feature_names)
        )

    def export_inference_bundle(self) -> Dict[str, object]:
        return {
            "schema_version": 1,
            "created_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "model_fingerprint": self.current_model_fingerprint,
            "memory_config": vars(self.memory_config),
            "trainer_config": vars(self.trainer_config),
            "static_feature_dim": int(len(self.static_mean) if self.static_mean is not None else self.memory_config.static_feature_dim),
            "kg_feature_dim": int(self.kg_feature_dim),
            "intervention_feature_dim": int(self.intervention_feature_dim),
            "intervention_sequence_dim": int(self.intervention_sequence_dim),
            "formation_feature_dim": int(len(self.formation_mean) if self.formation_mean is not None else 0),
            "model_state_dict": self.state_dict(),
            "normalizers": self._normalizer_payload(),
            "runtime_settings": self._runtime_settings_payload(),
            "neural_cache": self.export_neural_cache(),
            "intervention_component_stores": self._intervention_component_stores_payload(),
            "intervention_store_entries": [
                self._intervention_store_entry_to_dict(entry) for entry in self.intervention_store_entries
            ],
            "transition_store_entries": [
                self._transition_store_entry_to_dict(entry) for entry in self.transition_store_entries
            ],
            "training_summary": dict(self.training_summary),
            "memory_diagnostics": dict(self.memory_diagnostics),
        }

    def load_inference_bundle(self, payload: Dict[str, object]) -> None:
        state_dict = payload.get("model_state_dict", {})
        load_result = self.load_state_dict(state_dict, strict=False)
        self._load_normalizer_payload(dict(payload.get("normalizers", {})))
        self._load_runtime_settings(dict(payload.get("runtime_settings", {})))
        neural_cache = payload.get("neural_cache")
        if isinstance(neural_cache, dict):
            self.load_neural_cache(neural_cache)
        component_stores = payload.get("intervention_component_stores")
        if isinstance(component_stores, dict):
            self._load_intervention_component_stores(component_stores)
        self.intervention_store_entries = [
            self._intervention_store_entry_from_dict(entry)
            for entry in payload.get("intervention_store_entries", [])
        ]
        self._rebuild_intervention_store_cache()
        self.transition_store_entries = [
            self._transition_store_entry_from_dict(entry)
            for entry in payload.get("transition_store_entries", [])
        ]
        self._rebuild_transition_store_cache()
        self.current_model_fingerprint = str(payload.get("model_fingerprint", ""))
        self.training_summary = dict(payload.get("training_summary", {}))
        self.memory_diagnostics = dict(payload.get("memory_diagnostics", {}))
        if isinstance(state_dict, dict) and state_dict:
            self.memory_diagnostics["bundle_load"] = {
                "strict": False,
                "missing_keys": list(load_result.missing_keys),
                "unexpected_keys": list(load_result.unexpected_keys),
            }
        self.neural_cache_status = {
            "enabled": False,
            "model_fingerprint": self.current_model_fingerprint,
            "loaded": isinstance(neural_cache, dict),
            "exported": False,
        }
        self.eval()

    def _restore_or_rebuild_final_memory_bank(self, memory_samples: Sequence[ForecastSample]) -> None:
        self.current_model_fingerprint = self.compute_model_fingerprint()
        self.neural_cache_status = {
            "enabled": bool(self.cache_store is not None),
            "model_fingerprint": self.current_model_fingerprint,
            "loaded": False,
            "exported": False,
        }
        cache_payload = None
        if self.cache_store is not None and self.neural_cache_reuse_enabled:
            cache_payload = self.cache_store.load_model_cache(self.current_model_fingerprint)
        if cache_payload is not None:
            self.load_neural_cache(cache_payload)
            banks = cache_payload.get("banks", {})
            self.neural_cache_status.update(
                {
                    "loaded": True,
                    "pattern_memory_size": len(banks.get("pattern", [])),
                    "trajectory_memory_size": len(banks.get("trajectory", [])),
                    "experience_memory_size": len(banks.get("experience_hot", [])),
                    "experience_archive_size": len(banks.get("experience_archive", [])),
                }
            )
            return

        self._build_memory_bank(memory_samples)
        if self.cache_store is not None and self.neural_cache_export_enabled:
            meta = self.cache_store.save_model_cache(self.current_model_fingerprint, self.export_neural_cache())
            self.neural_cache_status.update(
                {
                    "exported": True,
                    "cache_path": meta.get("store_path", ""),
                    "pattern_memory_size": int(meta.get("pattern_memory_size", 0)),
                    "trajectory_memory_size": int(meta.get("trajectory_memory_size", 0)),
                    "experience_memory_size": int(meta.get("experience_memory_size", 0)),
                    "experience_archive_size": int(meta.get("experience_archive_size", 0)),
                }
            )

    def _build_bucket_masks(self, horizon: int) -> torch.Tensor:
        short_end = max(1, (horizon + 2) // 3)
        mid_end = max(short_end + 1, (2 * horizon + 2) // 3)
        masks = torch.zeros(3, horizon, dtype=torch.float32)
        masks[0, :short_end] = 1.0
        masks[1, short_end:mid_end] = 1.0
        masks[2, mid_end:] = 1.0
        return masks

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return float(max(lower, min(upper, value)))

    def _find_feature_index(self, feature_names: Sequence[str], candidates: Sequence[str]) -> int:
        normalized_names = [str(name).strip().lower() for name in feature_names]
        lowered_candidates = [str(candidate).strip().lower() for candidate in candidates]
        for candidate in lowered_candidates:
            if candidate in normalized_names:
                return normalized_names.index(candidate)
        for candidate in lowered_candidates:
            for index, name in enumerate(normalized_names):
                if candidate in name:
                    return index
        return -1

    def _temporal_profile_vector(self, metadata: Dict[str, object]) -> List[float]:
        dynamic_profile = dict(metadata.get("dynamic_profile", {})) if isinstance(metadata, dict) else {}
        return [
            float(dynamic_profile.get(feature_name, 0.0))
            for feature_name in self.temporal_profile_feature_names
        ]

    def _temporal_profile_tensor(self, metadata: Dict[str, object]) -> torch.Tensor:
        return torch.tensor(
            self._temporal_profile_vector(metadata),
            dtype=torch.float32,
            device=self.device,
        )

    def _temporal_profile_batch(self, samples: Sequence[ForecastSample]) -> torch.Tensor:
        if not samples:
            return torch.zeros((0, self.temporal_profile_dim), dtype=torch.float32, device=self.device)
        return torch.stack(
            [self._temporal_profile_tensor(sample.metadata if isinstance(sample.metadata, dict) else {}) for sample in samples],
            dim=0,
        )

    def _augment_factual_embedding(
        self,
        factual_embedding: torch.Tensor,
        temporal_profile: torch.Tensor,
        normalized_formation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        squeeze_output = False
        if factual_embedding.dim() == 1:
            factual_embedding = factual_embedding.unsqueeze(0)
            temporal_profile = temporal_profile.unsqueeze(0)
            normalized_formation = normalized_formation.unsqueeze(0)
            squeeze_output = True
        temporal_context = self.factual_temporal_projector(temporal_profile)
        temporal_gate_input = torch.cat([temporal_profile, normalized_formation], dim=-1)
        temporal_gate = torch.sigmoid(self.factual_temporal_gate(temporal_gate_input))
        augmented = factual_embedding + self.factual_temporal_path_scale * temporal_gate * temporal_context
        temporal_strength = temporal_context.abs().mean(dim=-1)
        if squeeze_output:
            return augmented.squeeze(0), temporal_gate.squeeze(0), temporal_strength.squeeze(0)
        return augmented, temporal_gate.squeeze(-1), temporal_strength

    def _get_static_feature(self, feature_values: Sequence[float], candidates: Sequence[str], default: float = 0.0) -> float:
        index = self._find_feature_index(self.intervention_feature_names, candidates)
        if index < 0 or index >= len(feature_values):
            return float(default)
        try:
            return float(feature_values[index])
        except (TypeError, ValueError):
            return float(default)

    def _set_static_feature(self, feature_values: List[float], candidates: Sequence[str], value: float) -> bool:
        index = self._find_feature_index(self.intervention_feature_names, candidates)
        if index < 0 or index >= len(feature_values):
            return False
        feature_values[index] = float(value)
        return True

    def _ensure_intervention_sequence_shape(self, intervention_sequence: Sequence[Sequence[float]]) -> List[List[float]]:
        if intervention_sequence:
            return [list(step) for step in intervention_sequence]
        feature_dim = len(self.intervention_sequence_feature_names)
        if feature_dim <= 0:
            return []
        step_count = max(1, int(self.trainer_config.history_length))
        return [[0.0] * feature_dim for _ in range(step_count)]

    def _set_sequence_feature_constant(
        self,
        intervention_sequence: Sequence[Sequence[float]],
        candidates: Sequence[str],
        value: float,
    ) -> List[List[float]]:
        updated_sequence = self._ensure_intervention_sequence_shape(intervention_sequence)
        index = self._find_feature_index(self.intervention_sequence_feature_names, candidates)
        if index < 0:
            return updated_sequence
        for step in updated_sequence:
            if index < len(step):
                step[index] = float(value)
        return updated_sequence

    def _infer_intervention_action_flags(
        self,
        intervention_static: Sequence[float],
        intervention_sequence: Sequence[Sequence[float]],
    ) -> Dict[str, float]:
        antibiotic_offset = self._get_static_feature(intervention_static, ["antibiotic_offset_minutes"], default=-1.0)
        antibiotic_duration = self._get_static_feature(intervention_static, ["antibiotic_course_ge_72h"], default=0.0)
        vasopressor_static = max(
            self._get_static_feature(intervention_static, ["post_24h_vasopressor_any"], default=0.0),
            self._get_static_feature(intervention_static, ["intervention_vasopressor_any_last"], default=0.0),
            self._get_static_feature(intervention_static, ["intervention_vasopressor_any_mean"], default=0.0),
            self._get_static_feature(intervention_static, ["intervention_vasopressor_any_max"], default=0.0),
            self._get_static_feature(intervention_static, ["intervention_vasopressor_score_last"], default=0.0),
            self._get_static_feature(intervention_static, ["intervention_vasopressor_score_mean"], default=0.0),
            self._get_static_feature(intervention_static, ["intervention_vasopressor_score_max"], default=0.0),
        )
        respiratory_static = max(
            self._get_static_feature(intervention_static, ["intervention_resp_support_last"], default=0.0),
            self._get_static_feature(intervention_static, ["intervention_resp_support_mean"], default=0.0),
            self._get_static_feature(intervention_static, ["intervention_resp_support_max"], default=0.0),
        )

        vasopressor_sequence = 0.0
        respiratory_sequence = 0.0
        if intervention_sequence:
            vaso_any_index = self._find_feature_index(self.intervention_sequence_feature_names, ["vasopressor_any"])
            vaso_score_index = self._find_feature_index(self.intervention_sequence_feature_names, ["vasopressor_score"])
            resp_index = self._find_feature_index(self.intervention_sequence_feature_names, ["resp_support"])
            for step in intervention_sequence:
                if vaso_any_index >= 0 and vaso_any_index < len(step):
                    vasopressor_sequence = max(vasopressor_sequence, float(step[vaso_any_index]))
                if vaso_score_index >= 0 and vaso_score_index < len(step):
                    vasopressor_sequence = max(vasopressor_sequence, float(step[vaso_score_index]))
                if resp_index >= 0 and resp_index < len(step):
                    respiratory_sequence = max(respiratory_sequence, float(step[resp_index]))

        return {
            "treat_early_antimicrobial": 1.0 if (0.0 <= antibiotic_offset <= 180.0 or antibiotic_duration > 0.0) else 0.0,
            "treat_vasopressor": 1.0 if max(vasopressor_static, vasopressor_sequence) > 0.0 else 0.0,
            "treat_respiratory_support": 1.0 if max(respiratory_static, respiratory_sequence) > 0.0 else 0.0,
        }

    def _merge_candidate_flags(
        self,
        base_flags: Dict[str, object],
        intervention_static: Sequence[float],
        intervention_sequence: Sequence[Sequence[float]],
    ) -> Dict[str, object]:
        merged = dict(base_flags or {})
        merged.update(self._infer_intervention_action_flags(intervention_static, intervention_sequence))
        return merged

    @staticmethod
    def _is_state_flag_name(name: str) -> bool:
        normalized = str(name).strip().lower()
        return normalized.startswith("state_") or normalized.startswith("kg_state_")

    def _build_plan_evaluation_flags(
        self,
        sample_kg_flags: Dict[str, object],
        reference_flags: Dict[str, object],
        intervention_static: Sequence[float],
        intervention_sequence: Sequence[Sequence[float]],
    ) -> Dict[str, object]:
        merged: Dict[str, object] = {}
        for key, value in dict(reference_flags or {}).items():
            if not self._is_state_flag_name(key):
                merged[key] = value
        for key, value in dict(sample_kg_flags or {}).items():
            if self._is_state_flag_name(key):
                merged[key] = value
        return self._merge_candidate_flags(merged, intervention_static, intervention_sequence)

    @staticmethod
    def _mean_abs(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        return float(sum(abs(float(value)) for value in values) / max(1, len(values)))

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        return float(sum(float(value) for value in values) / max(1, len(values)))

    def _bounded_transition_component(self, raw_value: float) -> float:
        scale = max(1e-6, float(self.transition_utility_scale))
        return float(torch.tanh(torch.tensor(float(raw_value) / scale)).item())

    def _intervention_action_profile(
        self,
        intervention_static: Sequence[float],
        intervention_sequence: Sequence[Sequence[float]],
        base_flags: Optional[Dict[str, object]] = None,
    ) -> Dict[str, float]:
        merged_flags = self._merge_candidate_flags(dict(base_flags or {}), intervention_static, intervention_sequence)
        antibiotic_offset = self._get_static_feature(intervention_static, ["antibiotic_offset_minutes"], default=-1.0)
        antibiotic_duration = self._get_static_feature(intervention_static, ["antibiotic_course_ge_72h"], default=0.0)
        vasopressor_last = max(
            self._get_static_feature(intervention_static, ["intervention_vasopressor_any_last"], default=0.0),
            self._get_static_feature(intervention_static, ["intervention_vasopressor_score_last"], default=0.0),
        )
        vasopressor_mean = max(
            self._get_static_feature(intervention_static, ["intervention_vasopressor_any_mean"], default=0.0),
            self._get_static_feature(intervention_static, ["intervention_vasopressor_score_mean"], default=0.0),
        )
        vasopressor_max = max(
            self._get_static_feature(intervention_static, ["post_24h_vasopressor_any"], default=0.0),
            self._get_static_feature(intervention_static, ["intervention_vasopressor_any_max"], default=0.0),
            self._get_static_feature(intervention_static, ["intervention_vasopressor_score_max"], default=0.0),
        )
        respiratory_last = self._get_static_feature(intervention_static, ["intervention_resp_support_last"], default=0.0)
        respiratory_mean = self._get_static_feature(intervention_static, ["intervention_resp_support_mean"], default=0.0)
        respiratory_max = self._get_static_feature(intervention_static, ["intervention_resp_support_max"], default=0.0)

        vasopressor_seq_values: List[float] = []
        respiratory_seq_values: List[float] = []
        if intervention_sequence:
            vaso_any_index = self._find_feature_index(self.intervention_sequence_feature_names, ["vasopressor_any"])
            vaso_score_index = self._find_feature_index(self.intervention_sequence_feature_names, ["vasopressor_score"])
            resp_index = self._find_feature_index(self.intervention_sequence_feature_names, ["resp_support"])
            for step in intervention_sequence:
                if vaso_any_index >= 0 and vaso_any_index < len(step):
                    vasopressor_seq_values.append(float(step[vaso_any_index]))
                if vaso_score_index >= 0 and vaso_score_index < len(step):
                    vasopressor_seq_values.append(float(step[vaso_score_index]))
                if resp_index >= 0 and resp_index < len(step):
                    respiratory_seq_values.append(float(step[resp_index]))

        vasopressor_seq_mean = self._mean_abs(vasopressor_seq_values)
        respiratory_seq_mean = self._mean_abs(respiratory_seq_values)
        vasopressor_seq_last = float(vasopressor_seq_values[-1]) if vasopressor_seq_values else 0.0
        respiratory_seq_last = float(respiratory_seq_values[-1]) if respiratory_seq_values else 0.0
        vasopressor_seq_delta = float(vasopressor_seq_values[-1] - vasopressor_seq_values[0]) if len(vasopressor_seq_values) > 1 else 0.0
        respiratory_seq_delta = float(respiratory_seq_values[-1] - respiratory_seq_values[0]) if len(respiratory_seq_values) > 1 else 0.0

        if 0.0 <= antibiotic_offset <= 360.0:
            antibiotic_timeliness = self._clamp(1.0 - antibiotic_offset / 360.0, 0.0, 1.0)
        else:
            antibiotic_timeliness = 0.0

        return {
            "treat_early_antimicrobial": self._kg_flag(merged_flags, "treat_early_antimicrobial"),
            "treat_vasopressor": self._kg_flag(merged_flags, "treat_vasopressor"),
            "treat_respiratory_support": self._kg_flag(merged_flags, "treat_respiratory_support"),
            "exam_blood_culture": self._kg_flag(merged_flags, "exam_blood_culture"),
            "exam_lactate": self._kg_flag(merged_flags, "exam_lactate"),
            "monitor_map65": self._kg_flag(merged_flags, "monitor_map65"),
            "monitor_lactate_repeat": self._kg_flag(merged_flags, "monitor_lactate_repeat"),
            "antibiotic_timeliness": float(antibiotic_timeliness),
            "antibiotic_duration_flag": float(1.0 if antibiotic_duration > 0.0 else 0.0),
            "vasopressor_intensity_last": float(max(vasopressor_last, vasopressor_seq_last)),
            "vasopressor_intensity_mean": float(max(vasopressor_mean, vasopressor_seq_mean)),
            "vasopressor_intensity_max": float(max(vasopressor_max, max(vasopressor_seq_values) if vasopressor_seq_values else 0.0)),
            "vasopressor_trend": float(vasopressor_seq_delta),
            "respiratory_intensity_last": float(max(respiratory_last, respiratory_seq_last)),
            "respiratory_intensity_mean": float(max(respiratory_mean, respiratory_seq_mean)),
            "respiratory_intensity_max": float(max(respiratory_max, max(respiratory_seq_values) if respiratory_seq_values else 0.0)),
            "respiratory_trend": float(respiratory_seq_delta),
            "sequence_strength": float(self._mean_abs([float(value) for step in intervention_sequence for value in step]) if intervention_sequence else 0.0),
            "sequence_peak": float(max([abs(float(value)) for step in intervention_sequence for value in step], default=0.0)),
            "static_strength": float(self._mean_abs(intervention_static)),
        }

    def _transition_query_preference_vector(self, sample: ForecastSample) -> torch.Tensor:
        kg_flags = dict(sample.metadata.get("kg_flags", {}))
        weights = torch.tensor([0.24, 0.24, 0.18, 0.18, 0.10, 0.06], dtype=torch.float32, device=self.device)
        if self._kg_flag(kg_flags, "state_septic_shock") > 0.0 or self._kg_flag(kg_flags, "state_hypotension") > 0.0:
            weights += torch.tensor([0.10, 0.02, 0.10, 0.00, -0.04, 0.00], dtype=torch.float32, device=self.device)
        if self._kg_flag(kg_flags, "state_high_lactate") > 0.0:
            weights += torch.tensor([0.04, 0.00, 0.06, 0.00, 0.00, 0.00], dtype=torch.float32, device=self.device)
        if self._kg_flag(kg_flags, "state_organ_dysfunction") > 0.0:
            weights += torch.tensor([0.00, 0.08, 0.00, 0.08, 0.02, 0.00], dtype=torch.float32, device=self.device)
        recent_delta = 0.0
        if sample.raw_context:
            recent_delta = float(sample.raw_context[-1] - sample.raw_context[0]) / max(1e-6, float(sample.scale_value))
        if recent_delta > 0.2:
            weights += torch.tensor([0.05, 0.05, 0.04, 0.04, 0.02, 0.00], dtype=torch.float32, device=self.device)
        weights = torch.clamp(weights, min=0.01)
        return weights / weights.sum()

    def _clinical_state_vector(
        self,
        sample: ForecastSample,
        sample_kg_flags: Optional[Dict[str, object]] = None,
    ) -> List[float]:
        kg_flags = dict(sample_kg_flags or sample.metadata.get("kg_flags", {}))
        scale_value = max(1e-6, float(sample.scale_value))
        last_value = float(sample.raw_context[-1]) if sample.raw_context else 0.0
        first_value = float(sample.raw_context[0]) if sample.raw_context else 0.0
        recent_level = (last_value - float(sample.scale_center)) / scale_value
        recent_delta = (last_value - first_value) / scale_value
        formation = sample.formation_features
        return [
            float(formation[0]),
            float(formation[1]),
            float(formation[2]),
            float(formation[6]),
            float(formation[11]),
            float(formation[14]),
            float(recent_level),
            float(recent_delta),
            self._kg_flag(kg_flags, "state_sepsis"),
            self._kg_flag(kg_flags, "state_septic_shock"),
            self._kg_flag(kg_flags, "state_organ_dysfunction"),
            self._kg_flag(kg_flags, "state_hypotension"),
            self._kg_flag(kg_flags, "state_high_lactate"),
        ]

    @staticmethod
    def _numeric_bin(value: float, thresholds: Sequence[float], labels: Sequence[str]) -> str:
        for threshold, label in zip(thresholds, labels):
            if float(value) <= float(threshold):
                return str(label)
        return str(labels[-1])

    def _clinical_state_signature_payload(
        self,
        sample: ForecastSample,
        sample_kg_flags: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        kg_flags = dict(sample_kg_flags or sample.metadata.get("kg_flags", {}))
        scale_value = max(1e-6, float(sample.scale_value))
        last_value = float(sample.raw_context[-1]) if sample.raw_context else 0.0
        first_value = float(sample.raw_context[0]) if sample.raw_context else last_value
        normalized_level = (last_value - float(sample.scale_center)) / scale_value
        normalized_trend = (last_value - first_value) / scale_value
        volatility = float(sample.formation_features[2]) if len(sample.formation_features) > 2 else 0.0
        change_proxy = float(sample.formation_features[11]) if len(sample.formation_features) > 11 else abs(normalized_trend)
        severity_bin = self._numeric_bin(
            last_value,
            thresholds=[3.0, 7.0, 11.0],
            labels=["low", "moderate", "high", "critical"],
        )
        level_bin = self._numeric_bin(
            normalized_level,
            thresholds=[-0.50, 0.50, 1.50],
            labels=["below_baseline", "near_baseline", "above_baseline", "far_above_baseline"],
        )
        trend_bin = self._numeric_bin(
            normalized_trend,
            thresholds=[-0.25, 0.25],
            labels=["improving", "stable", "worsening"],
        )
        volatility_bin = self._numeric_bin(
            max(abs(volatility), abs(change_proxy)),
            thresholds=[0.15, 0.50],
            labels=["quiet", "variable", "unstable"],
        )
        active_kg_flags = [
            flag
            for flag in [
                "state_sepsis",
                "state_septic_shock",
                "state_organ_dysfunction",
                "state_hypotension",
                "state_high_lactate",
            ]
            if self._kg_flag(kg_flags, flag) > 0.0
        ]
        kg_state = "+".join(active_kg_flags) if active_kg_flags else "kg_none"
        signature = "|".join(
            [
                f"severity={severity_bin}",
                f"level={level_bin}",
                f"trend={trend_bin}",
                f"volatility={volatility_bin}",
                f"trajectory={int(sample.trajectory_label)}",
                f"kg={kg_state}",
            ]
        )
        return {
            "signature": signature,
            "schema_version": 1,
            "severity_bin": severity_bin,
            "level_bin": level_bin,
            "trend_bin": trend_bin,
            "volatility_bin": volatility_bin,
            "trajectory_label": int(sample.trajectory_label),
            "pattern_label": int(sample.pattern_label),
            "active_kg_flags": active_kg_flags,
            "last_value": float(last_value),
            "normalized_level": float(normalized_level),
            "normalized_trend": float(normalized_trend),
        }

    def _clinical_state_signature(
        self,
        sample: ForecastSample,
        sample_kg_flags: Optional[Dict[str, object]] = None,
    ) -> str:
        return str(self._clinical_state_signature_payload(sample, sample_kg_flags).get("signature", ""))

    def _clinical_state_signature_similarity(
        self,
        query_payload: Dict[str, object],
        entry_payload: Dict[str, object],
    ) -> float:
        if not self.transition_signature_partial_match:
            return 1.0 if str(query_payload.get("signature", "")) == str(entry_payload.get("signature", "")) else 0.0
        score = 0.0
        total = 0.0
        for key in ["severity_bin", "level_bin", "trend_bin", "volatility_bin"]:
            if str(query_payload.get(key, "")) == str(entry_payload.get(key, "")):
                score += 1.0
            total += 1.0
        if int(query_payload.get("trajectory_label", -1)) == int(entry_payload.get("trajectory_label", -2)):
            score += 1.0
        total += 1.0
        if int(query_payload.get("pattern_label", -1)) == int(entry_payload.get("pattern_label", -2)):
            score += 0.5
        total += 0.5
        query_kg = set(str(f) for f in query_payload.get("active_kg_flags", []))
        entry_kg = set(str(f) for f in entry_payload.get("active_kg_flags", []))
        if query_kg or entry_kg:
            kg_union = query_kg | entry_kg
            kg_intersection = query_kg & entry_kg
            score += float(len(kg_intersection) / max(1, len(kg_union)))
        else:
            score += 1.0
        total += 1.0
        return score / max(1.0, total)

    @staticmethod
    def _transition_source_audit_metadata(sample: ForecastSample) -> Dict[str, object]:
        metadata = dict(sample.metadata or {})
        source_split = str(
            metadata.get("persistent_split")
            or metadata.get("split")
            or metadata.get("source_split")
            or "train_runtime"
        ).lower()
        return {
            "source_split": source_split,
            "source_dataset": str(metadata.get("dataset_name", "")),
            "source_stay_id": metadata.get("source_stay_id", metadata.get("stay_id", "")),
            "source_stay_id_hash": metadata.get("source_stay_id_hash", ""),
            "source_patient_id_hash": metadata.get("source_patient_id_hash", ""),
            "is_allowed_for_reuse": bool(metadata.get("is_allowed_for_reuse", source_split in {"train", "external", "train_runtime"})),
        }

    def _intervention_action_vector(
        self,
        intervention_static: Sequence[float],
        intervention_sequence: Sequence[Sequence[float]],
        base_flags: Optional[Dict[str, object]] = None,
    ) -> tuple[List[float], Dict[str, object]]:
        action_profile = self._intervention_action_profile(
            intervention_static=intervention_static,
            intervention_sequence=intervention_sequence,
            base_flags=base_flags,
        )
        merged_flags = self._merge_candidate_flags(dict(base_flags or {}), intervention_static, intervention_sequence)
        return [
            float(action_profile["treat_early_antimicrobial"]),
            float(action_profile["antibiotic_timeliness"]),
            float(action_profile["antibiotic_duration_flag"]),
            float(action_profile["treat_vasopressor"]),
            float(action_profile["vasopressor_intensity_last"]),
            float(action_profile["vasopressor_intensity_mean"]),
            float(action_profile["vasopressor_intensity_max"]),
            float(action_profile["vasopressor_trend"]),
            float(action_profile["treat_respiratory_support"]),
            float(action_profile["respiratory_intensity_last"]),
            float(action_profile["respiratory_intensity_mean"]),
            float(action_profile["respiratory_intensity_max"]),
            float(action_profile["respiratory_trend"]),
            float(action_profile["exam_blood_culture"]),
            float(action_profile["exam_lactate"]),
            float(action_profile["monitor_map65"]),
            float(action_profile["monitor_lactate_repeat"]),
            float(action_profile["static_strength"]),
            float(action_profile["sequence_strength"]),
            float(action_profile["sequence_peak"]),
        ], merged_flags

    def _static_feature_map(self, values: Sequence[float], feature_names: Sequence[str]) -> Dict[str, float]:
        return {
            str(name): float(values[index])
            for index, name in enumerate(feature_names)
            if index < len(values)
        }

    def _sequence_feature_map(self, sequence: Sequence[Sequence[float]]) -> List[Dict[str, float]]:
        sequence_rows: List[Dict[str, float]] = []
        for step_index, step in enumerate(sequence):
            row = self._static_feature_map(step, self.intervention_sequence_feature_names)
            row["step_index"] = float(step_index)
            sequence_rows.append(row)
        return sequence_rows

    @staticmethod
    def _active_flag_names(flags: Dict[str, object], prefix: str = "") -> List[str]:
        active: List[str] = []
        for key, value in sorted(dict(flags or {}).items()):
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if numeric_value > 0.0:
                active.append(f"{prefix}{key}")
        return active

    def _serialize_intervention_plan(
        self,
        intervention_static: Sequence[float],
        intervention_sequence: Sequence[Sequence[float]],
        base_flags: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        action_profile = self._intervention_action_profile(
            intervention_static=intervention_static,
            intervention_sequence=intervention_sequence,
            base_flags=base_flags,
        )
        inferred_flags = self._merge_candidate_flags(dict(base_flags or {}), intervention_static, intervention_sequence)
        return {
            "intervention_static_map": self._static_feature_map(intervention_static, self.intervention_feature_names),
            "intervention_sequence_map": self._sequence_feature_map(intervention_sequence),
            "action_profile": {key: float(value) for key, value in action_profile.items()},
            "active_flags": self._active_flag_names(inferred_flags),
            "component_partition": {
                component_type: {
                    "static_feature_names": [
                        str(self.intervention_feature_names[index])
                        for index in group["static"]
                        if 0 <= index < len(self.intervention_feature_names)
                    ],
                    "sequence_feature_names": [
                        str(self.intervention_sequence_feature_names[index])
                        for index in group["sequence"]
                        if 0 <= index < len(self.intervention_sequence_feature_names)
                    ],
                }
                for component_type, group in self._intervention_component_index_groups().items()
            },
        }

    def _serialize_query_snapshot(self, sample: ForecastSample) -> Dict[str, object]:
        kg_flags = dict(sample.metadata.get("kg_flags", {}))
        dynamic_profile = {
            str(key): float(value)
            for key, value in dict(sample.metadata.get("dynamic_profile", {})).items()
        }
        formation_map = {
            "local_slope": float(sample.formation_features[0]),
            "medium_slope": float(sample.formation_features[1]),
            "volatility": float(sample.formation_features[2]),
            "change_proxy": float(sample.formation_features[11]),
            "stability_score": float(sample.formation_features[14]),
            "regime_mix_score": float(sample.formation_features[15]),
        }
        return {
            "stay_id": float(sample.metadata.get("stay_id", -1.0)),
            "series_name": str(sample.metadata.get("series_name", "")),
            "pattern_label": int(sample.pattern_label),
            "pattern_name": PATTERN_LABELS[int(sample.pattern_label)],
            "trajectory_label": int(sample.trajectory_label),
            "trajectory_name": TRAJECTORY_LABELS[int(sample.trajectory_label)],
            "experience_label": int(sample.experience_label),
            "raw_context": [float(value) for value in sample.raw_context],
            "formation_features": formation_map,
            "kg_flags": {str(key): float(value) for key, value in kg_flags.items()},
            "active_state_flags": [name for name in self._active_flag_names(kg_flags) if self._is_state_flag_name(name)],
            "dynamic_profile": dynamic_profile,
            "temporal_feature_version": str(sample.metadata.get("temporal_feature_version", "")),
            "current_plan": self._serialize_intervention_plan(
                intervention_static=sample.intervention_static or [],
                intervention_sequence=sample.intervention_sequence or [],
                base_flags=kg_flags,
            ),
        }

    def _counterfactual_pool_membership_tags(
        self,
        sample: ForecastSample,
        donor_metadata: Dict[str, object],
    ) -> List[str]:
        tags: List[str] = []
        if self._counterfactual_pool_key(sample.metadata.get("hospitalid")) and self._counterfactual_pool_key(sample.metadata.get("hospitalid")) == self._counterfactual_pool_key(donor_metadata.get("hospitalid")):
            tags.append("same_hospital")
        if self._counterfactual_pool_key(sample.metadata.get("unittype")) and self._counterfactual_pool_key(sample.metadata.get("unittype")) == self._counterfactual_pool_key(donor_metadata.get("unittype")):
            tags.append("same_unit_type")
        if self._counterfactual_pool_key(sample.metadata.get("infection_anchor_type")) and self._counterfactual_pool_key(sample.metadata.get("infection_anchor_type")) == self._counterfactual_pool_key(donor_metadata.get("infection_anchor_type")):
            tags.append("same_infection_anchor")
        if int(sample.pattern_label) == int(float(donor_metadata.get("donor_pattern_label", -999))):
            tags.append("same_pattern")
        if int(sample.trajectory_label) == int(float(donor_metadata.get("donor_trajectory_label", -999))):
            tags.append("same_trajectory")
        if int(sample.experience_label) == int(float(donor_metadata.get("donor_experience_label", -999))):
            tags.append("same_experience")
        return tags

    def _counterfactual_pool_match_components(
        self,
        sample: ForecastSample,
        donor_metadata: Dict[str, object],
    ) -> Dict[str, object]:
        tags = self._counterfactual_pool_membership_tags(sample, donor_metadata)
        score = 0.0
        if "same_hospital" in tags:
            score += float(self.counterfactual_pool_same_hospital_weight)
        if "same_unit_type" in tags:
            score += float(self.counterfactual_pool_same_unit_weight)
        if "same_infection_anchor" in tags:
            score += float(self.counterfactual_pool_same_infection_anchor_weight)
        score = min(1.0, max(0.0, score))
        reward = float(self.counterfactual_pool_match_weight) * score
        return {
            "tags": tags,
            "score": float(score),
            "reward": float(reward),
        }

    def _serialize_ranked_donor_candidate(
        self,
        entry: InterventionStoreEntry,
        metadata: Dict[str, object],
    ) -> Dict[str, object]:
        return {
            "stay_id": float(entry.stay_id),
            "donor_experience_id": str(metadata.get("donor_experience_id", entry.experience_id)),
            "donor_intervention_plan_code": dict(
                metadata.get("donor_intervention_plan_code", entry.intervention_plan_code)
            ),
            "donor_experience_label": int(float(metadata.get("donor_experience_label", entry.experience_label))),
            "donor_pattern_label": int(float(metadata.get("donor_pattern_label", entry.pattern_label))),
            "donor_trajectory_label": int(float(metadata.get("donor_trajectory_label", entry.trajectory_label))),
            "donor_similarity": float(metadata.get("donor_similarity", 0.0)),
            "donor_kg_similarity": float(metadata.get("donor_kg_similarity", 0.0)),
            "donor_guideline_compatibility": float(metadata.get("donor_guideline_compatibility", 0.0)),
            "donor_state_match": float(metadata.get("donor_state_match", 0.0)),
            "donor_missing_care_penalty": float(metadata.get("donor_missing_care_penalty", 0.0)),
            "donor_contraindication_penalty": float(metadata.get("donor_contraindication_penalty", 0.0)),
            "donor_total_score": float(metadata.get("donor_total_score", 0.0)),
            "donor_transition_score": float(metadata.get("donor_transition_score", 0.0)),
            "donor_action_change_score": float(metadata.get("donor_action_change_score", 0.0)),
            "donor_hard_filter_valid": float(metadata.get("donor_hard_filter_valid", 0.0)),
            "donor_hard_filter_reason": str(metadata.get("donor_hard_filter_reason", "")),
            "donor_overlap_score": float(metadata.get("donor_overlap_score", 0.0)),
            "donor_overlap_valid": float(metadata.get("donor_overlap_valid", 0.0)),
            "donor_overlap_reason": str(metadata.get("donor_overlap_reason", "")),
            "donor_pre_neighbor_total_score": float(metadata.get("donor_pre_neighbor_total_score", 0.0)),
            "donor_neighbor_consistency": float(metadata.get("donor_neighbor_consistency", 0.0)),
            "donor_neighbor_exchangeability_mean": float(metadata.get("donor_neighbor_exchangeability_mean", 0.0)),
            "donor_neighbor_action_alignment_mean": float(metadata.get("donor_neighbor_action_alignment_mean", 0.0)),
            "donor_neighbor_hard_pass_rate": float(metadata.get("donor_neighbor_hard_pass_rate", 0.0)),
            "donor_neighbor_overlap_valid_rate": float(metadata.get("donor_neighbor_overlap_valid_rate", 0.0)),
            "donor_neighbor_bonus": float(metadata.get("donor_neighbor_bonus", 0.0)),
            "donor_neighbor_penalty": float(metadata.get("donor_neighbor_penalty", 0.0)),
            "donor_learned_reranker_score": float(metadata.get("donor_learned_reranker_score", 0.0)),
            "donor_reranker_adjustment": float(metadata.get("donor_reranker_adjustment", 0.0)),
            "donor_reranker_mode": str(metadata.get("donor_reranker_mode", "rule_only")),
            "donor_pool_match_score": float(metadata.get("donor_pool_match_score", 0.0)),
            "donor_pool_match_reward": float(metadata.get("donor_pool_match_reward", 0.0)),
            "donor_pool_tags": [str(value) for value in metadata.get("donor_pool_tags", [])],
            "plan_summary": self._serialize_intervention_plan(
                intervention_static=entry.intervention_static,
                intervention_sequence=entry.intervention_sequence,
                base_flags=dict(metadata.get("kg_flags", {})),
            ),
        }

    def _transition_candidate_labels(
        self,
        sample: ForecastSample,
        manager_result: Optional[ManagerReadResult] = None,
        preferred_label: Optional[int] = None,
    ) -> List[int]:
        labels: List[int] = []
        if preferred_label is not None and int(preferred_label) >= 0:
            labels.append(int(preferred_label))
        if manager_result is not None:
            top_label = int(manager_result.component_results["experience"].top_label)
            if top_label >= 0 and top_label not in labels:
                labels.append(top_label)
            for matched_label in manager_result.component_results["experience"].matched_labels:
                label_int = int(matched_label)
                if label_int >= 0 and label_int not in labels:
                    labels.append(label_int)
        sample_label = int(sample.experience_label)
        if sample_label not in labels:
            labels.append(sample_label)
        return labels

    def _empty_transition_readout(self) -> Dict[str, object]:
        return {
            "enabled": bool(self.enable_transition_memory),
            "hit_count": 0,
            "top_score": 0.0,
            "confidence": 0.0,
            "expected_utility": 0.0,
            "expected_advantage": 0.0,
            "expected_action_gain": 0.0,
            "improvement_rate": 0.0,
            "improvement_gain": 0.0,
            "support_strength": 0.0,
            "transition_score": 0.0,
            "residual_scale": 0.0,
            "query_clinical_state_signature": "",
            "matched_clinical_state_signature_count": 0,
            "matched_clinical_state_signature_weight": 0.0,
            "transition_signature_match_weight": float(self.transition_signature_match_weight),
            "prior_curve": [0.0] * self.trainer_config.forecast_horizon,
            "residual_curve": [0.0] * self.trainer_config.forecast_horizon,
            "template_curve": [0.0] * self.trainer_config.forecast_horizon,
        }

    def _transition_context_tensor(self, transition_readout: Dict[str, object]) -> torch.Tensor:
        context_values = list(transition_readout.get("template_curve", [])) + list(transition_readout.get("residual_curve", []))
        context_values.extend(
            [
                float(transition_readout.get("confidence", 0.0)),
                float(transition_readout.get("expected_utility", 0.0)),
                float(transition_readout.get("expected_advantage", 0.0)),
                float(transition_readout.get("improvement_rate", 0.0)),
                float(transition_readout.get("support_strength", 0.0)),
                float(transition_readout.get("transition_score", 0.0)),
            ]
        )
        return torch.tensor(context_values, dtype=torch.float32, device=self.device)

    def _transition_trunk_adjustment(self, transition_readout: Dict[str, object]) -> tuple[torch.Tensor, Dict[str, float]]:
        if (
            not self.enable_transition_memory
            or not self.enable_transition_factual_path
            or not self.enable_transition_trunk_path
            or int(transition_readout.get("hit_count", 0)) <= 0
        ):
            return torch.zeros(self.memory_config.fusion_hidden_dim, dtype=torch.float32, device=self.device), {
                "transition_trunk_gate": 0.0,
                "transition_trunk_strength": 0.0,
            }
        context_tensor = self._transition_context_tensor(transition_readout)
        projected_context = self.transition_context_projector(context_tensor)
        gate_input = torch.tensor(
            [
                float(transition_readout.get("confidence", 0.0)),
                float(transition_readout.get("expected_utility", 0.0)),
                float(transition_readout.get("expected_advantage", 0.0)),
                float(transition_readout.get("improvement_rate", 0.0)),
                float(transition_readout.get("support_strength", 0.0)),
                float(transition_readout.get("transition_score", 0.0)),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        trunk_gate = torch.sigmoid(self.transition_trunk_gate(gate_input)).squeeze(-1)
        trunk_scale = float(self.transition_trunk_weight) * trunk_gate
        adjustment = projected_context * trunk_scale
        return adjustment, {
            "transition_trunk_gate": float(trunk_gate.item()),
            "transition_trunk_strength": float(adjustment.abs().mean().item()),
        }

    def _bounded_transition_utility(self, raw_utility: float) -> float:
        return self._bounded_transition_component(raw_utility)

    def _transition_utility_vector_from_sample(self, sample: ForecastSample) -> List[float]:
        if not sample.raw_target:
            return [0.0] * 6
        current_level = float(sample.raw_context[-1]) if sample.raw_context else float(sample.scale_center)
        scale_value = max(1e-6, float(sample.scale_value))
        future_values = [float(value) for value in sample.raw_target]
        first_future = future_values[0]
        last_future = future_values[-1]
        mean_future = self._mean(future_values)
        worst_future = max(future_values)
        horizon_slope = first_future - last_future
        volatility = self._mean_abs(
            [future_values[index + 1] - future_values[index] for index in range(len(future_values) - 1)]
        )
        component_values = [
            (current_level - first_future) / scale_value,
            (current_level - mean_future) / scale_value,
            (current_level - worst_future) / scale_value,
            (current_level - last_future) / scale_value,
            horizon_slope / scale_value,
            -volatility / scale_value,
        ]
        return [self._bounded_transition_component(value) for value in component_values]

    def _transition_utility_from_sample(self, sample: ForecastSample) -> float:
        utility_vector = self._transition_utility_vector_from_sample(sample)
        preference_vector = self._transition_query_preference_vector(sample)
        utility_tensor = torch.tensor(utility_vector, dtype=torch.float32, device=self.device)
        return float(torch.dot(utility_tensor, preference_vector).item())

    def _transition_store_signature_audit(self) -> Dict[str, object]:
        total = len(self.transition_store_entries)
        signature_count = sum(
            1 for entry in self.transition_store_entries if str(entry.metadata.get("clinical_state_signature", ""))
        )
        source_split_counts: Dict[str, int] = {}
        reusable_count = 0
        vector_dims: Dict[str, int] = {}
        for entry in self.transition_store_entries:
            metadata = dict(entry.metadata or {})
            split = str(metadata.get("source_split", "unknown")).lower()
            source_split_counts[split] = source_split_counts.get(split, 0) + 1
            if bool(metadata.get("is_allowed_for_reuse", False)):
                reusable_count += 1
            vector_dim = str(metadata.get("clinical_state_vector_dim", len(entry.state_vector)))
            vector_dims[vector_dim] = vector_dims.get(vector_dim, 0) + 1
        top_signatures = [
            {"signature": signature, "count": int(count)}
            for signature, count in sorted(
                self.transition_signature_counts.items(),
                key=lambda item: (-int(item[1]), str(item[0])),
            )[:12]
        ]
        return {
            "schema_version": 1,
            "entry_count": int(total),
            "signature_count": int(signature_count),
            "signature_coverage_rate": float(signature_count / max(1, total)),
            "unique_signature_count": int(len(self.transition_signature_counts)),
            "top_signatures": top_signatures,
            "source_split_counts": source_split_counts,
            "reuse_allowed_rate": float(reusable_count / max(1, total)),
            "clinical_state_vector_dim_counts": vector_dims,
            "signature_match_weight": float(self.transition_signature_match_weight),
        }

    def _retrieve_transition_readout(
        self,
        sample: ForecastSample,
        intervention_static: Sequence[float],
        intervention_sequence: Sequence[Sequence[float]],
        candidate_labels: Optional[Sequence[int]] = None,
        base_flags: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        if not self.enable_transition_memory or not self.transition_store_entries:
            return self._empty_transition_readout()
        labels = [int(label) for label in (candidate_labels or [int(sample.experience_label)]) if int(label) in self.transition_store_by_label]
        if not labels:
            return self._empty_transition_readout()

        state_vector = self._clinical_state_vector(sample)
        query_signature = self._clinical_state_signature(sample)
        action_vector, _ = self._intervention_action_vector(
            intervention_static=intervention_static,
            intervention_sequence=intervention_sequence,
            base_flags=base_flags,
        )
        state_query = F.normalize(torch.tensor(state_vector, dtype=torch.float32, device=self.device), dim=0)
        action_query = F.normalize(torch.tensor(action_vector, dtype=torch.float32, device=self.device), dim=0)

        score_parts: List[torch.Tensor] = []
        future_parts: List[torch.Tensor] = []
        utility_parts: List[torch.Tensor] = []
        utility_vector_parts: List[torch.Tensor] = []
        signature_parts: List[str] = []
        query_preference = self._transition_query_preference_vector(sample)
        for label in labels:
            state_cache = self.transition_state_cache.get(label)
            action_cache = self.transition_action_cache.get(label)
            future_cache = self.transition_future_cache.get(label)
            utility_cache = self.transition_utility_cache.get(label)
            utility_vector_cache = self.transition_utility_vector_cache.get(label)
            signature_cache = self.transition_signature_cache.get(label, [])
            signature_payload_cache = self.transition_signature_payload_cache.get(label, [])
            if state_cache is None or action_cache is None or future_cache is None or utility_cache is None or utility_vector_cache is None:
                continue
            state_scores = torch.matmul(F.normalize(state_cache, dim=-1), state_query)
            action_scores = torch.matmul(F.normalize(action_cache, dim=-1), action_query)
            utility_alignment = torch.matmul(utility_vector_cache, query_preference)
            query_signature_payload = self._clinical_state_signature_payload(sample)
            if self.transition_signature_partial_match and signature_payload_cache:
                signature_scores = torch.tensor(
                    [self._clinical_state_signature_similarity(query_signature_payload, payload) for payload in signature_payload_cache],
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                signature_scores = torch.tensor(
                    [1.0 if signature and signature == query_signature else 0.0 for signature in signature_cache],
                    dtype=torch.float32,
                    device=self.device,
                )
            if int(signature_scores.numel()) != int(state_scores.numel()):
                signature_scores = torch.zeros_like(state_scores)
            combined_scores = (
                float(self.transition_state_weight) * state_scores
                + float(self.transition_action_weight) * action_scores
                + float(self.transition_utility_alignment_weight) * utility_alignment
                + float(self.transition_signature_match_weight) * signature_scores
            )
            score_parts.append(combined_scores)
            future_parts.append(future_cache)
            utility_parts.append(utility_cache)
            utility_vector_parts.append(utility_vector_cache)
            signature_parts.extend(signature_cache if len(signature_cache) == int(combined_scores.numel()) else [""] * int(combined_scores.numel()))
        if not score_parts:
            return self._empty_transition_readout()

        combined_scores = torch.cat(score_parts, dim=0)
        future_curves = torch.cat(future_parts, dim=0)
        utilities = torch.cat(utility_parts, dim=0)
        utility_vectors = torch.cat(utility_vector_parts, dim=0)
        top_k = min(max(1, int(self.transition_top_k)), int(combined_scores.numel()))
        top_scores, top_indices = torch.topk(combined_scores, k=top_k)
        weights = torch.softmax(top_scores / max(1e-6, float(self.transition_temperature)), dim=0)
        top_index_values = [int(index) for index in top_indices.detach().cpu().tolist()]
        matched_signature_count = sum(
            1 for index in top_index_values if 0 <= index < len(signature_parts) and signature_parts[index] == query_signature
        )
        matched_signature_weight = 0.0
        for weight, index in zip(weights.detach().cpu().tolist(), top_index_values):
            if 0 <= index < len(signature_parts) and signature_parts[index] == query_signature:
                matched_signature_weight += float(weight)
        weighted_template_curve = (future_curves[top_indices] * weights.unsqueeze(-1)).sum(dim=0)
        anchor_curve = future_curves[top_indices[0]]
        anchor_blend_weight = self._clamp(float(self.transition_anchor_blend_weight), 0.0, 1.0)
        template_curve = weighted_template_curve * (1.0 - anchor_blend_weight) + anchor_curve * anchor_blend_weight
        top_utility_vectors = utility_vectors[top_indices]
        query_utilities = torch.matmul(top_utility_vectors, query_preference)
        expected_utility = float((query_utilities * weights).sum().item())
        improvement_rate = float((((query_utilities > 0.0).float()) * weights).sum().item())
        top_score = float(top_scores[0].item()) if top_scores.numel() > 0 else 0.0
        label_support_total = float(sum(self.transition_label_support.get(label, 0) for label in labels))
        support_strength = self._clamp(
            math.log1p(label_support_total) / math.log1p(max(2.0, float(self.transition_top_k * 4))),
            0.0,
            1.0,
        )
        utility_prior_vectors = [
            self.transition_label_utility_vector_mean[label]
            for label in labels
            if label in self.transition_label_utility_vector_mean
        ]
        if utility_prior_vectors:
            utility_prior = float(
                torch.stack(utility_prior_vectors, dim=0).mean(dim=0).mul(query_preference).sum().item()
            )
        else:
            utility_prior_values = [self.transition_label_utility_mean.get(label, 0.0) for label in labels]
            utility_prior = float(sum(utility_prior_values) / max(1, len(utility_prior_values)))
        expected_advantage = float(expected_utility - utility_prior)
        prior_curve_parts = [
            self.transition_future_mean_cache[label]
            for label in labels
            if label in self.transition_future_mean_cache
        ]
        if prior_curve_parts:
            prior_curve = torch.stack(prior_curve_parts, dim=0).mean(dim=0)
        else:
            prior_curve = torch.zeros_like(template_curve)
        residual_curve = template_curve - prior_curve
        confidence = self._clamp(
            0.45 * (0.5 * (top_score + 1.0))
            + 0.30 * support_strength
            + 0.25 * improvement_rate,
            0.0,
            1.0,
        )
        residual_scale = self._clamp(
            confidence
            * (0.30 + 0.40 * support_strength + 0.30 * min(1.0, abs(expected_advantage) + 0.25 * abs(expected_utility))),
            0.0,
            1.0,
        )
        transition_score = float(
            confidence
            * (0.5 + 0.5 * support_strength)
            * (
                0.50 * expected_utility
                + 0.25 * expected_advantage
                + 0.25 * (2.0 * improvement_rate - 1.0)
            )
        )
        return {
            "enabled": True,
            "hit_count": int(top_k),
            "top_score": float(top_score),
            "confidence": float(confidence),
            "expected_utility": float(expected_utility),
            "expected_advantage": float(expected_advantage),
            "improvement_rate": float(improvement_rate),
            "support_strength": float(support_strength),
            "transition_score": float(transition_score),
            "residual_scale": float(residual_scale),
            "query_clinical_state_signature": query_signature,
            "matched_clinical_state_signature_count": int(matched_signature_count),
            "matched_clinical_state_signature_weight": float(matched_signature_weight),
            "transition_signature_match_weight": float(self.transition_signature_match_weight),
            "prior_curve": [float(value) for value in prior_curve.detach().cpu().tolist()],
            "residual_curve": [float(value) for value in residual_curve.detach().cpu().tolist()],
            "template_curve": [float(value) for value in template_curve.detach().cpu().tolist()],
        }

    def _retrieve_transition_readout_with_baseline(
        self,
        sample: ForecastSample,
        intervention_static: Sequence[float],
        intervention_sequence: Sequence[Sequence[float]],
        candidate_labels: Optional[Sequence[int]] = None,
        base_flags: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        candidate_readout = self._retrieve_transition_readout(
            sample=sample,
            intervention_static=intervention_static,
            intervention_sequence=intervention_sequence,
            candidate_labels=candidate_labels,
            base_flags=base_flags,
        )
        baseline_readout = self._retrieve_transition_readout(
            sample=sample,
            intervention_static=sample.intervention_static or [],
            intervention_sequence=sample.intervention_sequence or [],
            candidate_labels=candidate_labels,
            base_flags=dict(sample.metadata.get("kg_flags", {})),
        )
        enriched_readout = dict(candidate_readout)
        enriched_readout["baseline_expected_utility"] = float(baseline_readout["expected_utility"])
        enriched_readout["baseline_improvement_rate"] = float(baseline_readout["improvement_rate"])
        enriched_readout["expected_action_gain"] = float(
            float(candidate_readout["expected_utility"]) - float(baseline_readout["expected_utility"])
        )
        enriched_readout["improvement_gain"] = float(
            float(candidate_readout["improvement_rate"]) - float(baseline_readout["improvement_rate"])
        )
        return enriched_readout

    def _transition_factual_residual(
        self,
        sample: ForecastSample,
        base_prediction: torch.Tensor,
        coordinated_memory_residual: torch.Tensor,
        transition_readout: Dict[str, object],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        template_tensor = torch.tensor(
            transition_readout["template_curve"],
            dtype=torch.float32,
            device=self.device,
        )
        mode = str(self.transition_factual_residual_mode or "additive").strip().lower()
        if mode == "delta_to_base":
            residual_source = template_tensor - base_prediction
        elif mode == "delta_to_fusion_base":
            residual_source = template_tensor - (base_prediction + coordinated_memory_residual)
        else:
            residual_source = template_tensor

        template_delta = template_tensor - base_prediction
        if float(template_delta.abs().sum().item()) <= 1e-8 or float(coordinated_memory_residual.abs().sum().item()) <= 1e-8:
            transition_alignment = 0.0
        else:
            transition_alignment = float(
                F.cosine_similarity(
                    template_delta.unsqueeze(0),
                    coordinated_memory_residual.unsqueeze(0),
                    dim=-1,
                ).item()
            )
        agreement_gate = 0.5 + 0.5 * max(0.0, transition_alignment)
        confidence = float(transition_readout.get("confidence", 0.0))
        support_strength = float(transition_readout.get("support_strength", 0.0))
        expected_utility = float(transition_readout.get("expected_utility", 0.0))
        signature_weight = float(transition_readout.get("matched_clinical_state_signature_weight", 0.0))
        structural_reasons: List[str] = []
        gate_reasons: List[str] = []
        if int(transition_readout.get("hit_count", 0)) <= 0:
            structural_reasons.append("no_transition_hit")
            gate_reasons.append("no_transition_hit")
        if confidence < float(self.transition_min_confidence):
            structural_reasons.append("low_confidence")
            gate_reasons.append("low_confidence")
        if support_strength < float(self.transition_min_support):
            structural_reasons.append("low_support")
            gate_reasons.append("low_support")
        if signature_weight < float(self.transition_min_signature_weight):
            structural_reasons.append("low_signature_match")
            gate_reasons.append("low_signature_match")
        structural_blocked = 1.0 if structural_reasons else 0.0
        utility_bias = float(self.transition_utility_bias)
        utility_temperature = max(0.01, float(self.transition_utility_temperature))
        utility_sigmoid = float(torch.sigmoid(torch.tensor((expected_utility - utility_bias) / utility_temperature)).item())
        utility_factor = 0.1 + 0.9 * utility_sigmoid
        if expected_utility < float(self.transition_min_expected_utility):
            gate_reasons.append("low_utility")
        pattern_name = PATTERN_LABELS[int(sample.pattern_label)] if 0 <= int(sample.pattern_label) < len(PATTERN_LABELS) else ""
        trajectory_name = (
            TRAJECTORY_LABELS[int(sample.trajectory_label)]
            if 0 <= int(sample.trajectory_label) < len(TRAJECTORY_LABELS)
            else ""
        )
        pattern_factor = 1.0
        trajectory_factor = 1.0
        if pattern_name == "flat":
            pattern_factor = self._clamp(float(self.transition_flat_pattern_penalty), 0.0, 1.0)
        if trajectory_name == "stable_regime":
            trajectory_factor = self._clamp(float(self.transition_stable_regime_penalty), 0.0, 1.0)
        base_scale = float(transition_readout.get("residual_scale", 0.0)) * agreement_gate
        if structural_blocked > 0.5:
            final_scale = 0.0
        else:
            final_scale = base_scale * min(utility_factor, pattern_factor, trajectory_factor)
        if self.transition_positive_only:
            final_scale *= max(0.0, float(transition_readout["expected_utility"]))
        transition_residual = residual_source * (float(self.transition_template_blend_weight) * final_scale)
        raw_transition_residual = transition_residual
        cap_ratio = max(0.0, float(self.transition_residual_cap_ratio))
        residual_cap = cap_ratio * max(0.15, float(base_prediction.detach().abs().mean().item()))
        if residual_cap > 0.0:
            transition_residual = torch.clamp(transition_residual, min=-residual_cap, max=residual_cap)
        transition_summary = {
            "transition_alignment": float(transition_alignment),
            "transition_gate_scale": float(final_scale),
            "transition_residual_strength": float(transition_residual.abs().mean().item()),
            "transition_raw_residual_strength": float(raw_transition_residual.abs().mean().item()),
            "transition_safe_gate": float(1.0 - structural_blocked),
            "transition_structural_blocked": float(structural_blocked),
            "transition_utility_factor": float(utility_factor),
            "transition_pattern_factor": float(pattern_factor),
            "transition_trajectory_factor": float(trajectory_factor),
            "transition_residual_cap": float(residual_cap),
            "transition_residual_cap_applied": float(
                1.0 if float(raw_transition_residual.abs().max().item()) > residual_cap > 0.0 else 0.0
            ),
            "transition_gate_blocked": float(1.0 if structural_reasons else 0.0),
            "transition_gate_reasons": "|".join(gate_reasons),
        }
        return transition_residual, transition_summary

    def _action_change_score(
        self,
        sample: ForecastSample,
        candidate_flags: Dict[str, object],
        intervention_static: Sequence[float],
        intervention_sequence: Sequence[Sequence[float]],
    ) -> Dict[str, float]:
        sample_kg_flags = dict(sample.metadata.get("kg_flags", {}))
        current_flags = self._build_plan_evaluation_flags(
            sample_kg_flags=sample_kg_flags,
            reference_flags=sample_kg_flags,
            intervention_static=sample.intervention_static or [],
            intervention_sequence=sample.intervention_sequence or [],
        )
        current_components = self._counterfactual_guideline_components(sample_kg_flags, current_flags)
        candidate_components = self._counterfactual_guideline_components(sample_kg_flags, candidate_flags)
        current_profile = self._intervention_action_profile(
            intervention_static=sample.intervention_static or [],
            intervention_sequence=sample.intervention_sequence or [],
            base_flags=sample_kg_flags,
        )
        candidate_profile = self._intervention_action_profile(
            intervention_static=intervention_static,
            intervention_sequence=intervention_sequence,
            base_flags=sample_kg_flags,
        )

        relevant_features: List[str] = []
        if self._kg_flag(sample_kg_flags, "state_sepsis") > 0.0:
            relevant_features.extend(["treat_early_antimicrobial", "exam_blood_culture", "exam_lactate"])
        if self._kg_flag(sample_kg_flags, "state_septic_shock") > 0.0 or self._kg_flag(sample_kg_flags, "state_hypotension") > 0.0:
            relevant_features.extend(["treat_vasopressor", "monitor_map65", "monitor_lactate_repeat"])
        if self._kg_flag(sample_kg_flags, "state_high_lactate") > 0.0 and "exam_lactate" not in relevant_features:
            relevant_features.append("exam_lactate")

        feature_delta = 0.0
        for feature_name in relevant_features:
            current_value = self._kg_flag(current_flags, feature_name)
            candidate_value = self._kg_flag(candidate_flags, feature_name)
            if current_value <= 0.0 and candidate_value > 0.0:
                feature_delta += 0.18
            elif current_value > 0.0 and candidate_value <= 0.0:
                feature_delta -= 0.22

        antimicrobial_gain = (
            0.65 * (float(candidate_profile["antibiotic_timeliness"]) - float(current_profile["antibiotic_timeliness"]))
            + 0.35 * (float(candidate_profile["antibiotic_duration_flag"]) - float(current_profile["antibiotic_duration_flag"]))
        )
        vasopressor_gain = (
            0.50 * (float(candidate_profile["vasopressor_intensity_mean"]) - float(current_profile["vasopressor_intensity_mean"]))
            + 0.35 * (float(candidate_profile["vasopressor_intensity_last"]) - float(current_profile["vasopressor_intensity_last"]))
            + 0.15 * (float(candidate_profile["vasopressor_intensity_max"]) - float(current_profile["vasopressor_intensity_max"]))
        )
        respiratory_gain = (
            0.55 * (float(candidate_profile["respiratory_intensity_mean"]) - float(current_profile["respiratory_intensity_mean"]))
            + 0.45 * (float(candidate_profile["respiratory_intensity_last"]) - float(current_profile["respiratory_intensity_last"]))
        )
        monitoring_gain = (
            0.30 * (float(candidate_profile["exam_blood_culture"]) - float(current_profile["exam_blood_culture"]))
            + 0.30 * (float(candidate_profile["exam_lactate"]) - float(current_profile["exam_lactate"]))
            + 0.20 * (float(candidate_profile["monitor_map65"]) - float(current_profile["monitor_map65"]))
            + 0.20 * (float(candidate_profile["monitor_lactate_repeat"]) - float(current_profile["monitor_lactate_repeat"]))
        )
        state_weighted_gain = 0.0
        if self._kg_flag(sample_kg_flags, "state_sepsis") > 0.0:
            state_weighted_gain += 0.35 * antimicrobial_gain + 0.15 * monitoring_gain
        if self._kg_flag(sample_kg_flags, "state_septic_shock") > 0.0 or self._kg_flag(sample_kg_flags, "state_hypotension") > 0.0:
            state_weighted_gain += 0.35 * vasopressor_gain + 0.10 * monitoring_gain
        if self._kg_flag(sample_kg_flags, "state_high_lactate") > 0.0:
            state_weighted_gain += 0.10 * monitoring_gain
        if self._kg_flag(sample_kg_flags, "treat_respiratory_support") > 0.0:
            state_weighted_gain += 0.10 * respiratory_gain

        raw_score = (
            0.45 * (float(candidate_components["guideline_compatibility"]) - float(current_components["guideline_compatibility"]))
            + 0.20 * (float(current_components["missing_care_penalty"]) - float(candidate_components["missing_care_penalty"]))
            + 0.20 * (float(current_components["contraindication_penalty"]) - float(candidate_components["contraindication_penalty"]))
            + 0.10 * feature_delta
            + 0.20 * state_weighted_gain
        )
        return {
            "action_change_score": self._clamp(raw_score, -1.0, 1.0),
            "current_guideline_compatibility": float(current_components["guideline_compatibility"]),
            "candidate_guideline_compatibility": float(candidate_components["guideline_compatibility"]),
        }

    def _counterfactual_hard_filter_decision(
        self,
        sample_kg_flags: Dict[str, object],
        donor_flags: Dict[str, object],
    ) -> Dict[str, object]:
        reasons: List[str] = []
        if self._kg_flag(sample_kg_flags, "state_sepsis") > 0.0 and self._kg_flag(donor_flags, "treat_early_antimicrobial") <= 0.0:
            reasons.append("missing_early_antimicrobial")
        if (
            self._kg_flag(sample_kg_flags, "state_septic_shock") > 0.0
            or self._kg_flag(sample_kg_flags, "state_hypotension") > 0.0
        ) and self._kg_flag(donor_flags, "treat_vasopressor") <= 0.0:
            reasons.append("missing_vasopressor")
        if (
            self._kg_flag(sample_kg_flags, "state_high_lactate") > 0.0
            and self._kg_flag(donor_flags, "exam_lactate") <= 0.0
            and self._kg_flag(donor_flags, "monitor_lactate_repeat") <= 0.0
        ):
            reasons.append("missing_lactate_followup")
        return {
            "is_valid": len(reasons) == 0,
            "reasons": reasons,
        }

    def _knowledge_quality_from_flags(
        self,
        sample_kg_flags: Dict[str, object],
        candidate_flags: Dict[str, object],
        guideline_alignment: float,
    ) -> Dict[str, object]:
        components = self._counterfactual_guideline_components(sample_kg_flags, candidate_flags)
        hard_filter = self._counterfactual_hard_filter_decision(sample_kg_flags, candidate_flags)
        feasibility = 1.0 - min(
            1.0,
            0.70 * float(components["contraindication_penalty"]) + 0.30 * float(components["missing_care_penalty"]),
        )
        quality = (
            0.40
            + 0.25 * float(guideline_alignment)
            + 0.20 * float(components["guideline_compatibility"])
            + 0.10 * float(components["state_match"])
            - 0.15 * float(components["contraindication_penalty"])
        )
        write_confidence = self._clamp(quality * (0.80 + 0.20 * float(feasibility)), 0.05, 1.20)
        return {
            "knowledge_quality_score": self._clamp(quality, 0.05, 1.20),
            "knowledge_feasibility_score": self._clamp(feasibility, 0.0, 1.0),
            "guideline_compatibility": float(components["guideline_compatibility"]),
            "missing_care_penalty": float(components["missing_care_penalty"]),
            "contraindication_penalty": float(components["contraindication_penalty"]),
            "state_match": float(components["state_match"]),
            "write_confidence": float(write_confidence),
            "hard_valid": bool(hard_filter["is_valid"]),
            "hard_filter_reasons": list(hard_filter["reasons"]),
        }

    def _knowledge_quality_from_sample(self, sample: ForecastSample) -> Dict[str, object]:
        sample_flags = dict(sample.metadata.get("kg_flags", {}))
        candidate_flags = self._merge_candidate_flags(
            sample_flags,
            sample.intervention_static or [],
            sample.intervention_sequence or [],
        )
        return self._knowledge_quality_from_flags(
            sample_kg_flags=sample_flags,
            candidate_flags=candidate_flags,
            guideline_alignment=float(sample.metadata.get("kg_guideline_alignment", 0.0)),
        )

    def _repair_intervention_plan(
        self,
        sample_kg_flags: Dict[str, object],
        intervention_static: Sequence[float],
        intervention_sequence: Sequence[Sequence[float]],
    ) -> Tuple[List[float], List[List[float]], List[str]]:
        repaired_static = [float(value) for value in intervention_static]
        repaired_sequence = [list(step) for step in intervention_sequence]
        repair_actions: List[str] = []

        if self._kg_flag(sample_kg_flags, "state_sepsis") > 0.0:
            antibiotic_offset = self._get_static_feature(repaired_static, ["antibiotic_offset_minutes"], default=-1.0)
            if antibiotic_offset < 0.0 or antibiotic_offset > 180.0:
                if self._set_static_feature(repaired_static, ["antibiotic_offset_minutes"], 180.0):
                    repair_actions.append("early_antimicrobial")
            if self._set_static_feature(
                repaired_static,
                ["antibiotic_course_ge_72h"],
                max(1.0, self._get_static_feature(repaired_static, ["antibiotic_course_ge_72h"], default=0.0)),
            ):
                if "early_antimicrobial" not in repair_actions:
                    repair_actions.append("antimicrobial_course")

        if self._kg_flag(sample_kg_flags, "state_septic_shock") > 0.0 or self._kg_flag(sample_kg_flags, "state_hypotension") > 0.0:
            vaso_actions = 0
            vaso_actions += int(self._set_static_feature(repaired_static, ["post_24h_vasopressor_any"], 1.0))
            vaso_actions += int(self._set_static_feature(repaired_static, ["intervention_vasopressor_any_last"], 1.0))
            vaso_actions += int(self._set_static_feature(repaired_static, ["intervention_vasopressor_any_mean"], 1.0))
            vaso_actions += int(self._set_static_feature(repaired_static, ["intervention_vasopressor_any_max"], 1.0))
            vaso_actions += int(self._set_static_feature(repaired_static, ["intervention_vasopressor_score_last"], 1.0))
            vaso_actions += int(self._set_static_feature(repaired_static, ["intervention_vasopressor_score_mean"], 1.0))
            vaso_actions += int(self._set_static_feature(repaired_static, ["intervention_vasopressor_score_max"], 1.0))
            repaired_sequence = self._set_sequence_feature_constant(repaired_sequence, ["vasopressor_any"], 1.0)
            repaired_sequence = self._set_sequence_feature_constant(repaired_sequence, ["vasopressor_score"], 1.0)
            if vaso_actions > 0 or repaired_sequence:
                repair_actions.append("vasopressor_support")

        if self._kg_flag(sample_kg_flags, "treat_respiratory_support") > 0.0:
            resp_actions = 0
            resp_actions += int(self._set_static_feature(repaired_static, ["intervention_resp_support_last"], 1.0))
            resp_actions += int(self._set_static_feature(repaired_static, ["intervention_resp_support_mean"], 1.0))
            resp_actions += int(self._set_static_feature(repaired_static, ["intervention_resp_support_max"], 1.0))
            repaired_sequence = self._set_sequence_feature_constant(repaired_sequence, ["resp_support"], 1.0)
            if resp_actions > 0 or repaired_sequence:
                repair_actions.append("respiratory_support")

        return repaired_static, repaired_sequence, repair_actions

    @staticmethod
    def _candidate_plan_signature(
        intervention_static: Sequence[float],
        intervention_sequence: Sequence[Sequence[float]],
    ) -> Tuple[Tuple[float, ...], Tuple[Tuple[float, ...], ...]]:
        return (
            tuple(round(float(value), 4) for value in intervention_static),
            tuple(tuple(round(float(value), 4) for value in step) for step in intervention_sequence),
        )

    def _set_vasopressor_template(
        self,
        intervention_static: Sequence[float],
        intervention_sequence: Sequence[Sequence[float]],
        intensity: float,
    ) -> Tuple[List[float], List[List[float]]]:
        updated_static = [float(value) for value in intervention_static]
        updated_sequence = [list(step) for step in intervention_sequence]
        intensity = self._clamp(float(intensity), 0.0, 1.0)
        any_value = 1.0 if intensity > 0.0 else 0.0
        self._set_static_feature(updated_static, ["post_24h_vasopressor_any"], any_value)
        self._set_static_feature(updated_static, ["intervention_vasopressor_any_last"], any_value)
        self._set_static_feature(updated_static, ["intervention_vasopressor_any_mean"], any_value)
        self._set_static_feature(updated_static, ["intervention_vasopressor_any_max"], any_value)
        self._set_static_feature(updated_static, ["intervention_vasopressor_score_last"], intensity)
        self._set_static_feature(updated_static, ["intervention_vasopressor_score_mean"], intensity)
        self._set_static_feature(updated_static, ["intervention_vasopressor_score_max"], intensity)
        updated_sequence = self._set_sequence_feature_constant(updated_sequence, ["vasopressor_any"], any_value)
        updated_sequence = self._set_sequence_feature_constant(updated_sequence, ["vasopressor_score"], intensity)
        return updated_static, updated_sequence

    def _set_antimicrobial_template(
        self,
        intervention_static: Sequence[float],
        intervention_sequence: Sequence[Sequence[float]],
        offset_minutes: float,
        ensure_duration: bool = True,
    ) -> Tuple[List[float], List[List[float]]]:
        updated_static = [float(value) for value in intervention_static]
        updated_sequence = [list(step) for step in intervention_sequence]
        self._set_static_feature(updated_static, ["antibiotic_offset_minutes"], max(0.0, float(offset_minutes)))
        if ensure_duration:
            self._set_static_feature(
                updated_static,
                ["antibiotic_course_ge_72h"],
                max(1.0, self._get_static_feature(updated_static, ["antibiotic_course_ge_72h"], default=0.0)),
            )
        return updated_static, updated_sequence

    def _set_respiratory_template(
        self,
        intervention_static: Sequence[float],
        intervention_sequence: Sequence[Sequence[float]],
        intensity: float,
    ) -> Tuple[List[float], List[List[float]]]:
        updated_static = [float(value) for value in intervention_static]
        updated_sequence = [list(step) for step in intervention_sequence]
        intensity = self._clamp(float(intensity), 0.0, 1.0)
        self._set_static_feature(updated_static, ["intervention_resp_support_last"], intensity)
        self._set_static_feature(updated_static, ["intervention_resp_support_mean"], intensity)
        self._set_static_feature(updated_static, ["intervention_resp_support_max"], intensity)
        updated_sequence = self._set_sequence_feature_constant(updated_sequence, ["resp_support"], intensity)
        return updated_static, updated_sequence

    def _template_search_intervention_candidates(
        self,
        sample: ForecastSample,
        anchor_static: Sequence[float],
        anchor_sequence: Sequence[Sequence[float]],
    ) -> List[Dict[str, object]]:
        sample_kg_flags = dict(sample.metadata.get("kg_flags", {}))
        template_candidates: List[Dict[str, object]] = []
        seen_signatures = {self._candidate_plan_signature(anchor_static, anchor_sequence)}
        search_mode = str(self.counterfactual_candidate_search_mode).strip().lower()

        def register_candidate(
            candidate_static: Sequence[float],
            candidate_sequence: Sequence[Sequence[float]],
            candidate_source: str,
            search_actions: Sequence[str],
            candidate_layer: str = "search",
            strategy_family: str = "template",
            anchor_relation: str = "donor_centered",
            parameter_profile: Optional[Dict[str, object]] = None,
            safety_rationale: Optional[Sequence[str]] = None,
        ) -> None:
            signature = self._candidate_plan_signature(candidate_static, candidate_sequence)
            if signature in seen_signatures:
                return
            seen_signatures.add(signature)
            template_candidates.append(
                {
                    "candidate_static": [float(value) for value in candidate_static],
                    "candidate_sequence": [[float(value) for value in step] for step in candidate_sequence],
                    "candidate_source": str(candidate_source),
                    "search_actions": [str(action) for action in search_actions if str(action)],
                    "candidate_layer": str(candidate_layer),
                    "strategy_family": str(strategy_family),
                    "anchor_relation": str(anchor_relation),
                    "parameter_profile": dict(parameter_profile or {}),
                    "safety_rationale": [str(item) for item in (safety_rationale or []) if str(item).strip()],
                }
            )

        if self._kg_flag(sample_kg_flags, "state_sepsis") > 0.0:
            antibiotic_static = [float(value) for value in anchor_static]
            antibiotic_sequence = [list(step) for step in anchor_sequence]
            changed = False
            antibiotic_offset = self._get_static_feature(antibiotic_static, ["antibiotic_offset_minutes"], default=-1.0)
            target_offset = 60.0 if antibiotic_offset < 0.0 or antibiotic_offset > 60.0 else float(antibiotic_offset)
            changed |= self._set_static_feature(antibiotic_static, ["antibiotic_offset_minutes"], target_offset)
            changed |= self._set_static_feature(
                antibiotic_static,
                ["antibiotic_course_ge_72h"],
                max(1.0, self._get_static_feature(antibiotic_static, ["antibiotic_course_ge_72h"], default=0.0)),
            )
            if changed:
                register_candidate(
                    antibiotic_static,
                    antibiotic_sequence,
                    "generated_template_antimicrobial_fast",
                    ["template_earlier_antimicrobial"],
                    parameter_profile={
                        "antibiotic_offset_minutes": float(target_offset),
                        "antibiotic_course_ge_72h": 1.0,
                    },
                    safety_rationale=[
                        "抗菌时机被限制在早期窗口内",
                        "仅修正明确的早期抗菌缺口",
                    ],
                )

        if self._kg_flag(sample_kg_flags, "state_septic_shock") > 0.0 or self._kg_flag(sample_kg_flags, "state_hypotension") > 0.0:
            low_vaso_static, low_vaso_sequence = self._set_vasopressor_template(anchor_static, anchor_sequence, intensity=0.5)
            register_candidate(
                low_vaso_static,
                low_vaso_sequence,
                "generated_template_vasopressor_low",
                ["template_vasopressor_low"],
                parameter_profile={"vasopressor_intensity": 0.5},
                safety_rationale=[
                    "升压强度限制在低强度模板",
                    "不直接跳到最大强度",
                ],
            )
            standard_vaso_static, standard_vaso_sequence = self._set_vasopressor_template(anchor_static, anchor_sequence, intensity=1.0)
            register_candidate(
                standard_vaso_static,
                standard_vaso_sequence,
                "generated_template_vasopressor_standard",
                ["template_vasopressor_standard"],
                parameter_profile={"vasopressor_intensity": 1.0},
                safety_rationale=["保持标准强度升压支持", "不突破既有强度上限"],
            )

        if (
            self._kg_flag(sample_kg_flags, "state_sepsis") > 0.0
            and (
                self._kg_flag(sample_kg_flags, "state_septic_shock") > 0.0
                or self._kg_flag(sample_kg_flags, "state_hypotension") > 0.0
                or self._kg_flag(sample_kg_flags, "state_high_lactate") > 0.0
            )
        ):
            bundle_static = [float(value) for value in anchor_static]
            bundle_sequence = [list(step) for step in anchor_sequence]
            self._set_static_feature(bundle_static, ["antibiotic_offset_minutes"], 60.0)
            self._set_static_feature(
                bundle_static,
                ["antibiotic_course_ge_72h"],
                max(1.0, self._get_static_feature(bundle_static, ["antibiotic_course_ge_72h"], default=0.0)),
            )
            bundle_static, bundle_sequence = self._set_vasopressor_template(bundle_static, bundle_sequence, intensity=0.75)
            register_candidate(
                bundle_static,
                bundle_sequence,
                "generated_template_sepsis_bundle",
                ["template_earlier_antimicrobial", "template_vasopressor_standard"],
                parameter_profile={
                    "antibiotic_offset_minutes": 60.0,
                    "vasopressor_intensity": 0.75,
                },
                safety_rationale=[
                    "仅在脓毒症合并循环风险时启用组合模板",
                    "组合仍受 hard filter 和 overlap 约束",
                ],
            )

        if self._kg_flag(sample_kg_flags, "treat_respiratory_support") > 0.0:
            respiratory_static, respiratory_sequence = self._set_respiratory_template(anchor_static, anchor_sequence, intensity=1.0)
            register_candidate(
                respiratory_static,
                respiratory_sequence,
                "generated_template_respiratory_support",
                ["template_respiratory_support"],
                parameter_profile={"respiratory_support_intensity": 1.0},
                safety_rationale=["呼吸支持强度限制在既有标度范围内"],
            )

        if search_mode != "legacy":
            if self._kg_flag(sample_kg_flags, "state_sepsis") > 0.0:
                for offset_minutes, candidate_source, search_action in [
                    (0.0, "generated_template_antimicrobial_immediate", "template_antimicrobial_immediate"),
                    (120.0, "generated_template_antimicrobial_2h", "template_antimicrobial_2h"),
                ]:
                    antibiotic_static, antibiotic_sequence = self._set_antimicrobial_template(
                        anchor_static,
                        anchor_sequence,
                        offset_minutes=offset_minutes,
                        ensure_duration=True,
                    )
                    register_candidate(
                        antibiotic_static,
                        antibiotic_sequence,
                        candidate_source,
                        [search_action],
                        candidate_layer="search",
                        strategy_family="antimicrobial_timing",
                        anchor_relation="safe_deviation",
                        parameter_profile={
                            "antibiotic_offset_minutes": float(offset_minutes),
                            "antibiotic_course_ge_72h": 1.0,
                        },
                        safety_rationale=[
                            "只在预定义时间窗内调整抗菌时机",
                            "不改变治疗域数量",
                        ],
                    )

            if self._kg_flag(sample_kg_flags, "state_septic_shock") > 0.0 or self._kg_flag(sample_kg_flags, "state_hypotension") > 0.0:
                for intensity, candidate_source, search_action in [
                    (0.35, "generated_template_vasopressor_bridge", "template_vasopressor_bridge"),
                    (0.65, "generated_template_vasopressor_moderate", "template_vasopressor_moderate"),
                ]:
                    vaso_static, vaso_sequence = self._set_vasopressor_template(anchor_static, anchor_sequence, intensity=intensity)
                    register_candidate(
                        vaso_static,
                        vaso_sequence,
                        candidate_source,
                        [search_action],
                        candidate_layer="search",
                        strategy_family="vasopressor_intensity",
                        anchor_relation="safe_deviation",
                        parameter_profile={"vasopressor_intensity": float(intensity)},
                        safety_rationale=[
                            "升压药仅在有限强度格点内搜索",
                            "保持在 0 到 1 的既有强度标度内",
                        ],
                    )

            if self._kg_flag(sample_kg_flags, "treat_respiratory_support") > 0.0:
                respiratory_static, respiratory_sequence = self._set_respiratory_template(
                    anchor_static,
                    anchor_sequence,
                    intensity=0.5,
                )
                register_candidate(
                    respiratory_static,
                    respiratory_sequence,
                    "generated_template_respiratory_support_low",
                    ["template_respiratory_support_low"],
                    candidate_layer="search",
                    strategy_family="respiratory_intensity",
                    anchor_relation="safe_deviation",
                    parameter_profile={"respiratory_support_intensity": 0.5},
                    safety_rationale=["呼吸支持只在低强度格点内偏移"],
                )

            if (
                self._kg_flag(sample_kg_flags, "state_sepsis") > 0.0
                and (
                    self._kg_flag(sample_kg_flags, "state_hypotension") > 0.0
                    or self._kg_flag(sample_kg_flags, "state_high_lactate") > 0.0
                )
            ):
                combo_static, combo_sequence = self._set_antimicrobial_template(
                    anchor_static,
                    anchor_sequence,
                    offset_minutes=60.0,
                    ensure_duration=True,
                )
                combo_static, combo_sequence = self._set_vasopressor_template(
                    combo_static,
                    combo_sequence,
                    intensity=0.65,
                )
                register_candidate(
                    combo_static,
                    combo_sequence,
                    "generated_strategy_sepsis_hemodynamic_combo",
                    ["template_antimicrobial_2h", "template_vasopressor_moderate", "template_combo_sepsis_hemodynamic"],
                    candidate_layer="strategy",
                    strategy_family="sepsis_hemodynamic_combo",
                    anchor_relation="safe_deviation",
                    parameter_profile={
                        "antibiotic_offset_minutes": 60.0,
                        "vasopressor_intensity": 0.65,
                    },
                    safety_rationale=[
                        "只组合抗感染与血流动力学两个高价值动作域",
                        "每个参数都限制在预定义安全格点，不做开放式任意生成",
                    ],
                )

        candidate_limit = (
            max(0, int(self.counterfactual_search_template_limit))
            if search_mode == "legacy"
            else max(0, int(self.counterfactual_parameterized_template_limit))
        )
        return template_candidates[:candidate_limit]

    def _search_candidate_is_admissible(
        self,
        anchor_metadata: Dict[str, object],
        candidate_metadata: Dict[str, object],
    ) -> bool:
        anchor_hard_valid = bool(anchor_metadata.get("donor_hard_filter_valid", 0.0))
        candidate_hard_valid = bool(candidate_metadata.get("donor_hard_filter_valid", 0.0))
        if self.counterfactual_hard_filter_enabled and anchor_hard_valid and not candidate_hard_valid:
            return False
        if float(candidate_metadata.get("donor_contraindication_penalty", 0.0)) > float(anchor_metadata.get("donor_contraindication_penalty", 0.0)) + 1e-6:
            return False
        max_missing = float(anchor_metadata.get("donor_missing_care_penalty", 0.0)) + float(self.counterfactual_search_missing_care_tolerance)
        if float(candidate_metadata.get("donor_missing_care_penalty", 0.0)) > max_missing + 1e-6:
            return False
        min_guideline = float(anchor_metadata.get("donor_guideline_compatibility", 0.0)) - float(self.counterfactual_search_guideline_drop_max)
        if float(candidate_metadata.get("donor_guideline_compatibility", 0.0)) + 1e-6 < min_guideline:
            return False
        if self.counterfactual_overlap_filter_enabled:
            anchor_overlap_score = float(anchor_metadata.get("donor_overlap_score", 0.0))
            candidate_overlap_score = float(candidate_metadata.get("donor_overlap_score", 0.0))
            if candidate_overlap_score + float(self.counterfactual_search_overlap_drop_max) + 1e-6 < anchor_overlap_score:
                return False
        return True

    def _build_generated_candidate_metadata(
        self,
        sample: ForecastSample,
        donor_metadata: Dict[str, object],
        intervention_static: Sequence[float],
        intervention_sequence: Sequence[Sequence[float]],
        candidate_source: str,
        repair_actions: Sequence[str],
        search_actions: Optional[Sequence[str]] = None,
        candidate_layer: str = "safety",
        strategy_family: str = "donor_local",
        parameter_profile: Optional[Dict[str, object]] = None,
        anchor_relation: str = "donor_centered",
        safety_rationale: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        sample_kg_flags = dict(sample.metadata.get("kg_flags", {}))
        donor_flags = dict(donor_metadata.get("kg_flags", {}))
        candidate_flags = self._build_plan_evaluation_flags(
            sample_kg_flags=sample_kg_flags,
            reference_flags=donor_flags,
            intervention_static=intervention_static,
            intervention_sequence=intervention_sequence,
        )
        action_change = self._action_change_score(
            sample=sample,
            candidate_flags=candidate_flags,
            intervention_static=intervention_static,
            intervention_sequence=intervention_sequence,
        )
        sample_overlap_profile = self._counterfactual_overlap_profile_from_sample(sample, sample_kg_flags=sample_kg_flags)
        donor_overlap_profile = self._counterfactual_overlap_profile_from_metadata(donor_metadata)
        donor_overlap_profile["action_set"] = self._counterfactual_action_set_from_flags(candidate_flags)
        overlap_decision = self._counterfactual_overlap_decision(sample_overlap_profile, donor_overlap_profile)
        transition_readout = self._empty_transition_readout()
        if self.enable_transition_memory and self.enable_transition_donor_path:
            transition_readout = self._retrieve_transition_readout_with_baseline(
                sample=sample,
                intervention_static=intervention_static,
                intervention_sequence=intervention_sequence,
                candidate_labels=self._transition_candidate_labels(
                    sample,
                    preferred_label=int(donor_metadata.get("donor_experience_label", sample.experience_label)),
                ),
                base_flags=candidate_flags,
            )
        score_components = self._compute_counterfactual_donor_score(
            sample_kg_flags=sample_kg_flags,
            donor_flags=candidate_flags,
            embedding_similarity=float(donor_metadata.get("donor_similarity", 0.0)),
            kg_similarity=float(donor_metadata.get("donor_kg_similarity", 0.0)),
            donor_guideline_score=float(donor_metadata.get("kg_guideline_alignment", 0.0)),
            transition_utility=float(transition_readout["expected_utility"]),
            transition_confidence=float(transition_readout["confidence"]),
            transition_advantage=float(transition_readout["expected_advantage"]),
            transition_action_gain=float(transition_readout["expected_action_gain"]),
            transition_improvement_rate=float(transition_readout["improvement_rate"]),
            transition_improvement_gain=float(transition_readout["improvement_gain"]),
            transition_support_strength=float(transition_readout["support_strength"]),
                action_change_score=float(action_change["action_change_score"]),
            )
        hard_filter = self._counterfactual_hard_filter_decision(sample_kg_flags, candidate_flags)
        overlap_adjustment = float(self.counterfactual_overlap_weight) * (float(overlap_decision["overlap_score"]) - 0.5)
        provisional_metadata = {
            "donor_similarity": float(donor_metadata.get("donor_similarity", 0.0)),
            "donor_kg_similarity": float(score_components["kg_similarity"]),
            "donor_guideline_compatibility": float(score_components["guideline_compatibility"]),
            "donor_state_match": float(score_components["state_match"]),
            "donor_missing_care_penalty": float(score_components["missing_care_penalty"]),
            "donor_contraindication_penalty": float(score_components["contraindication_penalty"]),
            "donor_hard_filter_valid": 1.0 if hard_filter["is_valid"] else 0.0,
            "donor_overlap_score": float(overlap_decision["overlap_score"]),
            "donor_overlap_valid": 1.0 if overlap_decision["is_valid"] else 0.0,
            "donor_transition_score": float(score_components["transition_score"]),
            "donor_action_change_score": float(score_components["action_change_score"]),
            "donor_total_score": float(score_components["final_score"] + overlap_adjustment),
            "donor_experience_label": float(donor_metadata.get("donor_experience_label", sample.experience_label)),
            "donor_pattern_label": float(donor_metadata.get("donor_pattern_label", sample.pattern_label)),
            "donor_trajectory_label": float(donor_metadata.get("donor_trajectory_label", sample.trajectory_label)),
        }
        learned_reranker_score = self._counterfactual_reranker_predict(sample, provisional_metadata)
        reranker_adjustment = float(self.counterfactual_reranker_blend_weight) * float(learned_reranker_score)
        metadata = dict(donor_metadata)
        metadata.update(
            {
                "kg_flags": candidate_flags,
                "donor_kg_similarity": float(score_components["kg_similarity"]),
                "donor_guideline_compatibility": float(score_components["guideline_compatibility"]),
                "donor_state_match": float(score_components["state_match"]),
                "donor_missing_care_penalty": float(score_components["missing_care_penalty"]),
                "donor_contraindication_penalty": float(score_components["contraindication_penalty"]),
                "donor_total_score": float(score_components["final_score"] + overlap_adjustment + reranker_adjustment),
                "donor_overlap_score": float(overlap_decision["overlap_score"]),
                "donor_overlap_valid": 1.0 if overlap_decision["is_valid"] else 0.0,
                "donor_overlap_reason": "|".join(str(reason) for reason in overlap_decision["reasons"]),
                "donor_overlap_severity_gap": float(overlap_decision["severity_gap"]),
                "donor_overlap_trend_gap": float(overlap_decision["trend_gap"]),
                "donor_overlap_state_score": float(overlap_decision["state_overlap"]),
                "donor_overlap_action_score": float(overlap_decision["action_overlap"]),
                "donor_overlap_adjustment": float(overlap_adjustment),
                "donor_pre_neighbor_total_score": float(donor_metadata.get("donor_pre_neighbor_total_score", 0.0)),
                "donor_neighbor_consistency": float(donor_metadata.get("donor_neighbor_consistency", 0.0)),
                "donor_neighbor_exchangeability_mean": float(donor_metadata.get("donor_neighbor_exchangeability_mean", 0.0)),
                "donor_neighbor_action_alignment_mean": float(donor_metadata.get("donor_neighbor_action_alignment_mean", 0.0)),
                "donor_neighbor_state_alignment_mean": float(donor_metadata.get("donor_neighbor_state_alignment_mean", 0.0)),
                "donor_neighbor_hard_pass_rate": float(donor_metadata.get("donor_neighbor_hard_pass_rate", 0.0)),
                "donor_neighbor_overlap_valid_rate": float(donor_metadata.get("donor_neighbor_overlap_valid_rate", 0.0)),
                "donor_neighbor_anchor_rank": float(donor_metadata.get("donor_neighbor_anchor_rank", -1.0)),
                "donor_neighbor_bonus": float(donor_metadata.get("donor_neighbor_bonus", 0.0)),
                "donor_neighbor_penalty": float(donor_metadata.get("donor_neighbor_penalty", 0.0)),
                "donor_learned_reranker_score": float(learned_reranker_score),
                "donor_reranker_adjustment": float(reranker_adjustment),
                "donor_reranker_mode": str(self.counterfactual_reranker_mode),
                "donor_transition_utility": float(score_components["transition_utility"]),
                "donor_transition_confidence": float(score_components["transition_confidence"]),
                "donor_transition_advantage": float(score_components["transition_advantage"]),
                "donor_transition_action_gain": float(score_components["transition_action_gain"]),
                "donor_transition_improvement_rate": float(score_components["transition_improvement_rate"]),
                "donor_transition_improvement_gain": float(score_components["transition_improvement_gain"]),
                "donor_transition_support_strength": float(score_components["transition_support_strength"]),
                "donor_transition_score": float(score_components["transition_score"]),
                "donor_action_change_score": float(score_components["action_change_score"]),
                "donor_source": str(candidate_source),
                "generated_candidate_source": str(candidate_source),
                "generated_candidate_layer": str(candidate_layer),
                "generated_candidate_repair_actions": "|".join(str(action) for action in repair_actions),
                "generated_candidate_search_actions": "|".join(str(action) for action in (search_actions or [])),
                "generated_candidate_strategy_family": str(strategy_family),
                "generated_candidate_parameter_profile": dict(parameter_profile or {}),
                "generated_candidate_anchor_relation": str(anchor_relation),
                "generated_candidate_safety_rationale": [str(item) for item in (safety_rationale or []) if str(item).strip()],
                "donor_hard_filter_valid": 1.0 if hard_filter["is_valid"] else 0.0,
                "donor_hard_filter_reason": "|".join(str(reason) for reason in hard_filter["reasons"]),
            }
        )
        return metadata

    def _generate_counterfactual_intervention_candidates(
        self,
        sample: ForecastSample,
        donor_intervention: Sequence[float],
        donor_intervention_sequence: Sequence[Sequence[float]],
        donor_metadata: Dict[str, object],
    ) -> List[Tuple[List[float], List[List[float]], Dict[str, object]]]:
        base_metadata = self._build_generated_candidate_metadata(
            sample=sample,
            donor_metadata=donor_metadata,
            intervention_static=donor_intervention,
            intervention_sequence=donor_intervention_sequence,
            candidate_source=str(donor_metadata.get("donor_source", "raw_intervention_store")),
            repair_actions=[],
            search_actions=[],
            candidate_layer="safety",
            strategy_family="donor_original",
            parameter_profile={},
            anchor_relation="donor_centered",
            safety_rationale=["直接保留 donor 原方案，作为后续偏离与重排的参照基线"],
        )
        candidates = [
            (
                list(donor_intervention),
                [list(step) for step in donor_intervention_sequence],
                base_metadata,
            )
        ]
        if self.counterfactual_candidate_policy not in {"generated_best", "safe_search"}:
            return candidates

        search_anchor_static = list(donor_intervention)
        search_anchor_sequence = [list(step) for step in donor_intervention_sequence]
        search_anchor_metadata = base_metadata
        repaired_static, repaired_sequence, repair_actions = self._repair_intervention_plan(
            sample_kg_flags=dict(sample.metadata.get("kg_flags", {})),
            intervention_static=donor_intervention,
            intervention_sequence=donor_intervention_sequence,
        )
        if repair_actions and (
            repaired_static != list(donor_intervention)
            or repaired_sequence != [list(step) for step in donor_intervention_sequence]
        ):
            candidates.append(
                (
                    repaired_static,
                    repaired_sequence,
                    self._build_generated_candidate_metadata(
                        sample=sample,
                        donor_metadata=donor_metadata,
                        intervention_static=repaired_static,
                        intervention_sequence=repaired_sequence,
                        candidate_source="generated_kg_repaired",
                        repair_actions=repair_actions,
                        search_actions=[],
                        candidate_layer="safety",
                        strategy_family="kg_repair",
                        parameter_profile={},
                        anchor_relation="donor_centered",
                        safety_rationale=["仅针对明确 care gap 做知识约束修补", "不扩展到开放式动作搜索"],
                    ),
                )
            )
            search_anchor_static = repaired_static
            search_anchor_sequence = repaired_sequence
            search_anchor_metadata = candidates[-1][2]

        if self.counterfactual_candidate_policy != "safe_search":
            return candidates

        seen_signatures = {
            self._candidate_plan_signature(candidate_intervention, candidate_sequence)
            for candidate_intervention, candidate_sequence, _ in candidates
        }
        for template_candidate in self._template_search_intervention_candidates(
            sample=sample,
            anchor_static=search_anchor_static,
            anchor_sequence=search_anchor_sequence,
        ):
            template_static = list(template_candidate.get("candidate_static", []))
            template_sequence = [list(step) for step in template_candidate.get("candidate_sequence", [])]
            template_source = str(template_candidate.get("candidate_source", ""))
            search_actions = [str(action) for action in template_candidate.get("search_actions", [])]
            signature = self._candidate_plan_signature(template_static, template_sequence)
            if signature in seen_signatures:
                continue
            template_metadata = self._build_generated_candidate_metadata(
                sample=sample,
                donor_metadata=donor_metadata,
                intervention_static=template_static,
                intervention_sequence=template_sequence,
                candidate_source=template_source,
                repair_actions=repair_actions,
                search_actions=search_actions,
                candidate_layer=str(template_candidate.get("candidate_layer", "search")),
                strategy_family=str(template_candidate.get("strategy_family", "template")),
                parameter_profile=dict(template_candidate.get("parameter_profile", {})),
                anchor_relation=str(template_candidate.get("anchor_relation", "donor_centered")),
                safety_rationale=[str(item) for item in template_candidate.get("safety_rationale", [])],
            )
            if not self._search_candidate_is_admissible(search_anchor_metadata, template_metadata):
                continue
            seen_signatures.add(signature)
            candidates.append((template_static, template_sequence, template_metadata))
        return candidates

    @staticmethod
    def _counterfactual_mean_scale(gate_summary: Dict[str, object]) -> float:
        scale_tensor = gate_summary.get("_predicted_factual_scale_tensor")
        if scale_tensor is None:
            return 0.0
        scale_values = torch.as_tensor(scale_tensor, dtype=torch.float32).flatten()
        if scale_values.numel() == 0:
            return 0.0
        return float(scale_values.abs().mean().item())

    @staticmethod
    def _counterfactual_regression_gain(
        current_aux_predictions: Dict[str, float],
        candidate_aux_predictions: Dict[str, float],
        task_name: str,
    ) -> float:
        return float(
            float(current_aux_predictions.get(task_name, 0.0))
            - float(candidate_aux_predictions.get(task_name, 0.0))
        )

    def _candidate_selection_score(
        self,
        predicted_delta: float,
        candidate_metadata: Dict[str, object],
        current_aux_predictions: Optional[Dict[str, float]] = None,
        candidate_aux_predictions: Optional[Dict[str, float]] = None,
        current_gate_summary: Optional[Dict[str, object]] = None,
        candidate_gate_summary: Optional[Dict[str, object]] = None,
        neighborhood_summary: Optional[Dict[str, object]] = None,
    ) -> Tuple[float, Dict[str, object]]:
        feasibility_bonus = (
            float(candidate_metadata.get("donor_guideline_compatibility", 0.0))
            + 0.5 * float(candidate_metadata.get("donor_state_match", 0.0))
        )
        penalty_value = (
            float(candidate_metadata.get("donor_contraindication_penalty", 0.0))
            + 0.5 * float(candidate_metadata.get("donor_missing_care_penalty", 0.0))
        )
        similarity_bonus = 0.05 * float(candidate_metadata.get("donor_similarity", 0.0))
        action_change_bonus = float(self.transition_candidate_action_change_weight) * float(candidate_metadata.get("donor_action_change_score", 0.0))
        transition_bonus = 0.0
        if self.enable_transition_memory and self.enable_transition_donor_path:
            transition_bonus = float(self.transition_selection_weight) * float(candidate_metadata.get("donor_transition_score", 0.0))
        base_score = float(
            predicted_delta
            + self.counterfactual_candidate_feasibility_weight * feasibility_bonus
            + similarity_bonus
            + action_change_bonus
            + transition_bonus
            - self.counterfactual_candidate_penalty_weight * penalty_value
        )
        current_aux_predictions = dict(current_aux_predictions or {})
        candidate_aux_predictions = dict(candidate_aux_predictions or {})
        current_gate_summary = dict(current_gate_summary or {})
        candidate_gate_summary = dict(candidate_gate_summary or {})
        sofa_gain = self._counterfactual_regression_gain(
            current_aux_predictions,
            candidate_aux_predictions,
            "future_sofa_delta_mean",
        )
        lactate_gain = self._counterfactual_regression_gain(
            current_aux_predictions,
            candidate_aux_predictions,
            "future_lactate_delta",
        )
        vasopressor_gain = self._counterfactual_regression_gain(
            current_aux_predictions,
            candidate_aux_predictions,
            "future_vasopressor_need",
        )
        respiratory_gain = self._counterfactual_regression_gain(
            current_aux_predictions,
            candidate_aux_predictions,
            "future_resp_support_escalation",
        )
        current_scale = self._counterfactual_mean_scale(current_gate_summary)
        candidate_scale = self._counterfactual_mean_scale(candidate_gate_summary)
        delta_std = math.sqrt(max(0.0, current_scale ** 2 + candidate_scale ** 2))
        delta_lower_bound = float(predicted_delta - 1.96 * delta_std)
        multiobjective_support = float(
            self.counterfactual_multitask_sofa_weight * sofa_gain
            + self.counterfactual_multitask_lactate_weight * lactate_gain
            + self.counterfactual_multitask_vasopressor_weight * vasopressor_gain
            + self.counterfactual_multitask_respiratory_weight * respiratory_gain
        )
        conflicts: List[str] = []
        if sofa_gain < -0.01:
            conflicts.append("future_sofa_worse")
        if lactate_gain < -0.01:
            conflicts.append("future_lactate_worse")
        if vasopressor_gain < -0.02:
            conflicts.append("vasopressor_risk_higher")
        if respiratory_gain < -0.02:
            conflicts.append("resp_support_risk_higher")
        positive_unstable_penalty = 0.0
        if predicted_delta > 0.0 and delta_lower_bound < 0.0:
            positive_unstable_penalty = float(
                self.counterfactual_multitask_positive_unstable_weight
                * max(0.0, predicted_delta - delta_lower_bound)
            )
        uncertainty_penalty = float(
            self.counterfactual_multitask_uncertainty_weight * candidate_scale
            + self.counterfactual_multitask_lower_bound_weight * max(0.0, -delta_lower_bound)
            + self.counterfactual_multitask_conflict_weight * float(len(conflicts))
            + positive_unstable_penalty
        )
        pre_neighborhood_score = float(base_score + multiobjective_support - uncertainty_penalty)
        neighborhood_summary = dict(neighborhood_summary or {})
        neighbor_consistency = float(neighborhood_summary.get("consistency", 0.0))
        neighbor_exchangeability = float(neighborhood_summary.get("exchangeability_mean", 0.0))
        neighbor_action_alignment = float(neighborhood_summary.get("action_alignment_mean", 0.0))
        neighbor_hard_pass_rate = float(neighborhood_summary.get("hard_pass_rate", 0.0))
        neighbor_overlap_valid_rate = float(neighborhood_summary.get("overlap_valid_rate", 0.0))
        neighbor_adjustment = self._counterfactual_neighbor_adjustment(neighborhood_summary)
        neighborhood_bonus = float(neighbor_adjustment["bonus"])
        neighborhood_penalty = float(neighbor_adjustment["penalty"])
        final_score = float(pre_neighborhood_score + neighborhood_bonus - neighborhood_penalty)
        selection_components = {
            "base_rule_score": float(base_score),
            "future_sofa_delta_gain": float(sofa_gain),
            "future_lactate_delta_gain": float(lactate_gain),
            "future_vasopressor_need_reduction": float(vasopressor_gain),
            "future_resp_support_reduction": float(respiratory_gain),
            "current_forecast_scale": float(current_scale),
            "candidate_forecast_scale": float(candidate_scale),
            "delta_std_proxy": float(delta_std),
            "delta_lower_bound": float(delta_lower_bound),
            "multiobjective_support": float(multiobjective_support),
            "uncertainty_penalty": float(uncertainty_penalty),
            "positive_unstable_penalty": float(positive_unstable_penalty),
            "multiobjective_conflicts": list(conflicts),
            "pre_neighborhood_score": float(pre_neighborhood_score),
            "neighbor_consistency": float(neighbor_consistency),
            "neighbor_exchangeability_mean": float(neighbor_exchangeability),
            "neighbor_action_alignment_mean": float(neighbor_action_alignment),
            "neighbor_hard_pass_rate": float(neighbor_hard_pass_rate),
            "neighbor_overlap_valid_rate": float(neighbor_overlap_valid_rate),
            "neighborhood_bonus": float(neighborhood_bonus),
            "neighborhood_penalty": float(neighborhood_penalty),
            "final_score": float(final_score),
        }
        return final_score, selection_components

    @staticmethod
    def _rollout_safe_scale(values: Sequence[float]) -> Tuple[float, float]:
        if not values:
            return 0.0, 1.0
        center = float(sum(float(value) for value in values) / max(1, len(values)))
        variance = float(
            sum((float(value) - center) ** 2 for value in values) / max(1, len(values))
        )
        return center, max(1e-6, math.sqrt(max(variance, 0.0)))

    def _target_feature_index(self, sample: ForecastSample) -> int:
        target_field = str(sample.metadata.get("target_field", "")).strip()
        if target_field and self.sequence_feature_names:
            for index, feature_name in enumerate(self.sequence_feature_names):
                if str(feature_name) == target_field:
                    return index
        return 0

    def _rollout_next_context(
        self,
        sample: ForecastSample,
        projected_prediction: Sequence[float],
    ) -> List[float]:
        if not sample.raw_context:
            seed_value = float(projected_prediction[0]) if projected_prediction else 0.0
            return [seed_value] * max(1, self.trainer_config.history_length)
        next_value = float(projected_prediction[0]) if projected_prediction else float(sample.raw_context[-1])
        if len(sample.raw_context) == 1:
            return [next_value]
        return [float(value) for value in sample.raw_context[1:]] + [next_value]

    def _project_rollout_target(
        self,
        sample: ForecastSample,
        projected_prediction: Sequence[float],
        next_context: Sequence[float],
    ) -> List[float]:
        forecast_horizon = max(1, int(self.trainer_config.forecast_horizon))
        projected_values = [float(value) for value in projected_prediction]
        if len(projected_values) > 1:
            target = projected_values[1:]
            last_value = projected_values[-1]
        else:
            last_value = float(next_context[-1]) if next_context else 0.0
            target = [last_value]
        while len(target) < forecast_horizon:
            target.append(last_value)
        return target[:forecast_horizon]

    def _project_rollout_sequence(
        self,
        sample: ForecastSample,
        next_context: Sequence[float],
        scale_center: float,
        scale_value: float,
    ) -> List[List[float]]:
        target_index = self._target_feature_index(sample)
        sequence_dim = len(sample.sequence[0]) if sample.sequence else max(1, len(self.sequence_feature_names))
        if sequence_dim <= 0:
            sequence_dim = 1
        if sample.sequence:
            projected_sequence = [list(row) for row in sample.sequence[1:]]
            template_row = list(projected_sequence[-1] if projected_sequence else sample.sequence[-1])
        else:
            projected_sequence = []
            template_row = [0.0] * sequence_dim
        projected_sequence.append(list(template_row))
        projected_sequence = projected_sequence[-max(1, len(next_context)) :]
        for row_index, raw_value in enumerate(next_context[-len(projected_sequence) :]):
            if target_index >= len(projected_sequence[row_index]):
                projected_sequence[row_index].extend([0.0] * (target_index + 1 - len(projected_sequence[row_index])))
            projected_sequence[row_index][target_index] = (float(raw_value) - scale_center) / scale_value
        return projected_sequence

    def _project_rollout_sample(
        self,
        sample: ForecastSample,
        case_detail: Dict[str, object],
        step_index: int,
    ) -> ForecastSample:
        selected_candidate = dict(case_detail.get("selected_candidate", {}))
        projected_prediction = [
            float(value)
            for value in case_detail.get("selected_counterfactual_prediction", [])
        ]
        next_context = self._rollout_next_context(sample, projected_prediction)
        next_target = self._project_rollout_target(sample, projected_prediction, next_context)
        scale_center, scale_value = self._rollout_safe_scale(next_context)
        projected_sequence = self._project_rollout_sequence(sample, next_context, scale_center, scale_value)
        normalized_target = [
            (float(value) - scale_center) / scale_value
            for value in next_target
        ]
        end_index = float(sample.metadata.get("window_end_index", len(next_context)))
        formation = build_window_formation(
            next_context,
            seasonality=max(1, int(self.trainer_config.seasonality)),
            end_index=int(round(end_index)) + 1,
        )
        selected_static = [
            float(value)
            for value in selected_candidate.get("candidate_intervention_static", sample.intervention_static or [])
        ]
        selected_sequence = [
            [float(value) for value in step]
            for step in selected_candidate.get("candidate_intervention_sequence", sample.intervention_sequence or [])
        ]
        projected_metadata = dict(sample.metadata)
        projected_metadata["window_end_index"] = float(end_index + 1.0)
        projected_metadata["counterfactual_rollout_parent_step"] = float(step_index)
        projected_metadata["counterfactual_rollout_projected"] = 1.0
        projected_metadata["counterfactual_rollout_parent_source"] = str(
            selected_candidate.get("candidate_source", "")
        )
        return replace(
            sample,
            sequence=projected_sequence,
            target=normalized_target,
            metadata=projected_metadata,
            scale_center=float(scale_center),
            scale_value=float(scale_value),
            raw_context=[float(value) for value in next_context],
            raw_target=[float(value) for value in next_target],
            formation_features=[float(value) for value in formation.features],
            pattern_label=int(formation.pattern_label),
            trajectory_label=int(formation.trajectory_label),
            experience_label=int(formation.experience_label),
            intervention_static=selected_static or list(sample.intervention_static or []),
            intervention_sequence=selected_sequence or [list(step) for step in (sample.intervention_sequence or [])],
        )

    def _stack_sequences(self, sequences: Sequence[Sequence[Sequence[float]]]) -> torch.Tensor:
        return torch.tensor(sequences, dtype=torch.float32, device=self.device)

    def _stack_vectors(self, vectors: Sequence[Sequence[float]]) -> torch.Tensor:
        return torch.tensor(vectors, dtype=torch.float32, device=self.device)

    def _fit_normalizers(self, samples: Sequence[ForecastSample]):
        sequence_tensor = self._stack_sequences([sample.sequence for sample in samples])
        self.sequence_mean = sequence_tensor.mean(dim=(0, 1))
        self.sequence_std = sequence_tensor.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)

        static_tensor = self._stack_vectors([sample.patient_static or sample.static for sample in samples])
        self.static_mean = static_tensor.mean(dim=0)
        self.static_std = static_tensor.std(dim=0, unbiased=False).clamp_min(1e-6)

        if self.intervention_feature_dim > 0:
            intervention_tensor = self._stack_vectors([sample.intervention_static or [] for sample in samples])
            self.intervention_mean = intervention_tensor.mean(dim=0)
            self.intervention_std = intervention_tensor.std(dim=0, unbiased=False).clamp_min(1e-6)
        else:
            self.intervention_mean = None
            self.intervention_std = None

        if self.intervention_sequence_dim > 0:
            intervention_sequence_tensor = self._stack_sequences(
                [sample.intervention_sequence or [[0.0] * self.intervention_sequence_dim] for sample in samples]
            )
            self.intervention_sequence_mean = intervention_sequence_tensor.mean(dim=(0, 1))
            self.intervention_sequence_std = intervention_sequence_tensor.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)
        else:
            self.intervention_sequence_mean = None
            self.intervention_sequence_std = None

        formation_tensor = self._stack_vectors([sample.formation_features for sample in samples])
        self.formation_mean = formation_tensor.mean(dim=0)
        self.formation_std = formation_tensor.std(dim=0, unbiased=False).clamp_min(1e-6)
        self._fit_aux_target_stats(samples)

    def _sample_aux_targets(self, sample: ForecastSample) -> Dict[str, float]:
        payload = sample.aux_targets
        if not isinstance(payload, dict):
            payload = sample.metadata.get("aux_targets", {}) if isinstance(sample.metadata, dict) else {}
        if not isinstance(payload, dict):
            return {}
        return {
            str(key): float(value)
            for key, value in payload.items()
            if str(key) in self.multitask_task_names
        }

    def _fit_aux_target_stats(self, samples: Sequence[ForecastSample]) -> None:
        stats: Dict[str, Dict[str, float]] = {
            task_name: {"mean": 0.0, "std": 1.0}
            for task_name in self.multitask_regression_tasks
        }
        for task_name in self.multitask_regression_tasks:
            values = [
                float(targets[task_name])
                for targets in (self._sample_aux_targets(sample) for sample in samples)
                if task_name in targets
            ]
            if not values:
                continue
            mean_value = sum(values) / max(1, len(values))
            variance = sum((value - mean_value) ** 2 for value in values) / max(1, len(values))
            stats[task_name] = {
                "mean": float(mean_value),
                "std": float(max(math.sqrt(max(variance, 0.0)), 1e-6)),
            }
        self.aux_target_stats = stats

    def _normalize_aux_target(self, task_name: str, value: float) -> float:
        if task_name in self.multitask_binary_tasks:
            return float(value)
        stats = self.aux_target_stats.get(task_name, {"mean": 0.0, "std": 1.0})
        return (float(value) - float(stats.get("mean", 0.0))) / max(1e-6, float(stats.get("std", 1.0)))

    def _restore_aux_prediction(self, task_name: str, value: float) -> float:
        if task_name in self.multitask_binary_tasks:
            return float(value)
        stats = self.aux_target_stats.get(task_name, {"mean": 0.0, "std": 1.0})
        return float(value) * max(1e-6, float(stats.get("std", 1.0))) + float(stats.get("mean", 0.0))

    def _normalize_sequence(self, sequence_steps: Sequence[Sequence[float]]) -> torch.Tensor:
        sequence_tensor = torch.tensor(sequence_steps, dtype=torch.float32, device=self.device)
        return (sequence_tensor - self.sequence_mean) / self.sequence_std

    def _normalize_sequence_batch(self, sequences: Sequence[Sequence[Sequence[float]]]) -> torch.Tensor:
        sequence_tensor = self._stack_sequences(sequences)
        return (sequence_tensor - self.sequence_mean) / self.sequence_std

    def _normalize_static(self, static_vector: Sequence[float]) -> torch.Tensor:
        static_tensor = torch.tensor(static_vector, dtype=torch.float32, device=self.device)
        return (static_tensor - self.static_mean) / self.static_std

    def _normalize_static_batch(self, static_vectors: Sequence[Sequence[float]]) -> torch.Tensor:
        static_tensor = self._stack_vectors(static_vectors)
        return (static_tensor - self.static_mean) / self.static_std

    def _normalize_intervention(self, intervention_vector: Sequence[float]) -> torch.Tensor:
        if self.intervention_feature_dim <= 0:
            return torch.zeros(0, dtype=torch.float32, device=self.device)
        intervention_tensor = torch.tensor(intervention_vector, dtype=torch.float32, device=self.device)
        return (intervention_tensor - self.intervention_mean) / self.intervention_std

    def _normalize_intervention_batch(self, intervention_vectors: Sequence[Sequence[float]]) -> torch.Tensor:
        if self.intervention_feature_dim <= 0:
            return torch.zeros((len(intervention_vectors), 0), dtype=torch.float32, device=self.device)
        intervention_tensor = self._stack_vectors(intervention_vectors)
        return (intervention_tensor - self.intervention_mean) / self.intervention_std

    def _normalize_intervention_sequence(self, intervention_sequence: Sequence[Sequence[float]]) -> torch.Tensor:
        if self.intervention_sequence_dim <= 0:
            return torch.zeros((0, 0), dtype=torch.float32, device=self.device)
        if not intervention_sequence:
            intervention_sequence = [[0.0] * self.intervention_sequence_dim]
        sequence_tensor = torch.tensor(intervention_sequence, dtype=torch.float32, device=self.device)
        return (sequence_tensor - self.intervention_sequence_mean) / self.intervention_sequence_std

    def _normalize_intervention_sequence_batch(
        self,
        intervention_sequences: Sequence[Sequence[Sequence[float]]],
    ) -> torch.Tensor:
        if self.intervention_sequence_dim <= 0:
            return torch.zeros((len(intervention_sequences), 0, 0), dtype=torch.float32, device=self.device)
        padded_sequences: List[List[List[float]]] = []
        max_length = max(1, max(len(sequence) for sequence in intervention_sequences))
        zero_step = [0.0] * self.intervention_sequence_dim
        for sequence in intervention_sequences:
            sequence_list = [list(step) for step in sequence] if sequence else []
            if not sequence_list:
                sequence_list = [list(zero_step)]
            if len(sequence_list) < max_length:
                sequence_list = sequence_list + [list(zero_step) for _ in range(max_length - len(sequence_list))]
            padded_sequences.append(sequence_list)
        sequence_tensor = self._stack_sequences(padded_sequences)
        return (sequence_tensor - self.intervention_sequence_mean) / self.intervention_sequence_std

    def _project_intervention(
        self,
        normalized_intervention: torch.Tensor,
        normalized_intervention_sequence: torch.Tensor,
    ) -> torch.Tensor:
        is_batched = normalized_intervention.dim() > 1 or normalized_intervention_sequence.dim() > 2
        batch_size = 1
        if normalized_intervention.dim() > 1:
            batch_size = normalized_intervention.size(0)
        elif normalized_intervention_sequence.dim() > 2:
            batch_size = normalized_intervention_sequence.size(0)
        if self.intervention_context_dim <= 0:
            if is_batched:
                return torch.zeros((batch_size, 0), dtype=torch.float32, device=self.device)
            return torch.zeros(0, dtype=torch.float32, device=self.device)
        pieces: List[torch.Tensor] = []
        if self.intervention_static_projector is not None:
            intervention_input = normalized_intervention if normalized_intervention.dim() > 1 else normalized_intervention.unsqueeze(0)
            pieces.append(self.intervention_static_projector(intervention_input))
        if self.intervention_sequence_encoder is not None and normalized_intervention_sequence.numel() > 0:
            sequence_input = (
                normalized_intervention_sequence
                if normalized_intervention_sequence.dim() > 2
                else normalized_intervention_sequence.unsqueeze(0)
            )
            _, hidden = self.intervention_sequence_encoder(sequence_input)
            pieces.append(self.intervention_sequence_projector(hidden[-1]))
        if not pieces:
            if is_batched:
                return torch.zeros((batch_size, self.intervention_context_dim), dtype=torch.float32, device=self.device)
            return torch.zeros(self.intervention_context_dim, dtype=torch.float32, device=self.device)
        if len(pieces) == 1:
            result = pieces[0]
        else:
            result = self.intervention_fuser(torch.cat(pieces, dim=-1))
        return result if is_batched else result.squeeze(0)

    def _multitask_head_input(
        self,
        fused_representation: torch.Tensor,
        base_prediction: torch.Tensor,
        intervention_embedding: torch.Tensor,
    ) -> torch.Tensor:
        parts = [fused_representation, base_prediction]
        if self.intervention_context_dim > 0:
            parts.append(intervention_embedding)
        return torch.cat(parts, dim=-1)

    def _predict_multitask_outputs(
        self,
        fused_representation: torch.Tensor,
        base_prediction: torch.Tensor,
        intervention_embedding: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        trunk = self.multitask_trunk(
            self._multitask_head_input(
                fused_representation=fused_representation,
                base_prediction=base_prediction,
                intervention_embedding=intervention_embedding,
            )
        )
        return {
            task_name: self.multitask_heads[task_name](trunk).squeeze(-1)
            for task_name in self.multitask_task_names
        }

    def _normalize_formation(self, formation_vector: Sequence[float]) -> torch.Tensor:
        formation_tensor = torch.tensor(formation_vector, dtype=torch.float32, device=self.device)
        return (formation_tensor - self.formation_mean) / self.formation_std

    def _normalize_formation_batch(self, formation_vectors: Sequence[Sequence[float]]) -> torch.Tensor:
        formation_tensor = self._stack_vectors(formation_vectors)
        return (formation_tensor - self.formation_mean) / self.formation_std

    def _batch_encoding_output(
        self,
        batch_encoding: TorchManifoldEncodingOutput,
        index: int,
        metadata: Dict[str, object],
    ) -> TorchManifoldEncodingOutput:
        return TorchManifoldEncodingOutput(
            query=batch_encoding.query[index],
            key=batch_encoding.key[index],
            value=batch_encoding.value[index],
            input_embedding=batch_encoding.input_embedding[index],
            metadata=metadata,
        )

    def _prepare_batch_tensors(
        self,
        samples: Sequence[ForecastSample],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_sequence = self._normalize_sequence_batch([sample.sequence for sample in samples])
        normalized_static = self._normalize_static_batch([sample.patient_static or sample.static for sample in samples])
        normalized_intervention = self._normalize_intervention_batch([sample.intervention_static or [] for sample in samples])
        normalized_intervention_sequence = self._normalize_intervention_sequence_batch(
            [sample.intervention_sequence or [] for sample in samples]
        )
        normalized_formation = self._normalize_formation_batch([sample.formation_features for sample in samples])
        return (
            normalized_sequence,
            normalized_static,
            normalized_intervention,
            normalized_intervention_sequence,
            normalized_formation,
        )

    def _encode_batch(
        self,
        samples: Sequence[ForecastSample],
    ) -> tuple[
        TorchManifoldEncodingOutput,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            normalized_sequence,
            normalized_static,
            normalized_intervention,
            normalized_intervention_sequence,
            normalized_formation,
        ) = self._prepare_batch_tensors(samples)
        batch_encoding = self.manifold.encode_input_torch(
            normalized_sequence,
            normalized_static,
            metadata={"batch_size": float(len(samples))},
        )
        if batch_encoding.input_embedding.dim() == 1:
            batch_encoding = TorchManifoldEncodingOutput(
                query=batch_encoding.query.unsqueeze(0),
                key=batch_encoding.key.unsqueeze(0),
                value=batch_encoding.value.unsqueeze(0),
                input_embedding=batch_encoding.input_embedding.unsqueeze(0),
                metadata=batch_encoding.metadata,
            )
        temporal_profile = self._temporal_profile_batch(samples)
        (
            factual_embedding,
            retrieval_encoding,
            factual_branch_gate,
            factual_branch_strength,
            retrieval_branch_gate,
            retrieval_branch_strength,
        ) = self._split_encoding_spaces(
            batch_encoding,
            normalized_formation=normalized_formation,
            temporal_profile=temporal_profile,
        )
        factual_embedding, temporal_gate, temporal_strength = self._augment_factual_embedding(
            factual_embedding=factual_embedding,
            temporal_profile=temporal_profile,
            normalized_formation=normalized_formation,
        )
        intervention_embedding = self._project_intervention(
            normalized_intervention,
            normalized_intervention_sequence,
        )
        if intervention_embedding.dim() == 1:
            intervention_embedding = intervention_embedding.unsqueeze(0)
        return (
            retrieval_encoding,
            factual_embedding,
            intervention_embedding,
            normalized_formation,
            temporal_gate,
            temporal_strength,
            factual_branch_gate,
            factual_branch_strength,
            retrieval_branch_gate,
            retrieval_branch_strength,
        )

    def _iter_batches(self, total_size: int) -> List[List[int]]:
        order = list(range(total_size))
        rng = random.Random(self.trainer_config.seed + self.best_epoch + total_size)
        rng.shuffle(order)
        return [order[start : start + self.trainer_config.batch_size] for start in range(0, total_size, self.trainer_config.batch_size)]

    def _manifold_regularization(
        self,
        encodings: Sequence[TorchManifoldEncodingOutput],
        metadata: Sequence[Dict[str, object]],
    ) -> torch.Tensor:
        if not encodings:
            return torch.zeros((), dtype=torch.float32, device=self.device)

        queries = torch.stack([encoding.query for encoding in encodings], dim=0)
        keys = torch.stack([encoding.key for encoding in encodings], dim=0)
        embeddings = torch.stack([encoding.input_embedding for encoding in encodings], dim=0)
        align_loss = (1.0 - F.cosine_similarity(queries, keys, dim=-1)).mean()

        temporal_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        temporal_pairs = 0
        for left_idx in range(len(encodings)):
            for right_idx in range(left_idx + 1, len(encodings)):
                if metadata[left_idx].get("stay_id") != metadata[right_idx].get("stay_id"):
                    continue
                gap = abs(
                    float(metadata[left_idx].get("window_end_index", 0.0))
                    - float(metadata[right_idx].get("window_end_index", 0.0))
                )
                if gap <= 1.0:
                    distance = torch.norm(embeddings[left_idx] - embeddings[right_idx], p=2)
                    temporal_sum = temporal_sum + distance.pow(2)
                    temporal_pairs += 1

        temporal_loss = temporal_sum / max(1, temporal_pairs)
        return (
            self.trainer_config.align_loss_weight * align_loss
            + self.trainer_config.temporal_smoothness_weight * temporal_loss
        )

    def _activity_from_sample(self, sample: ForecastSample) -> float:
        slope_strength = abs(sample.formation_features[0]) + 0.6 * abs(sample.formation_features[1])
        volatility = sample.formation_features[2]
        regime_mix = sample.formation_features[15]
        knowledge_quality = self._knowledge_quality_from_sample(sample)
        base_activity = 1.0 + 0.12 * min(4.0, slope_strength) + 0.08 * min(4.0, volatility) + 0.15 * regime_mix
        return base_activity * (0.85 + 0.30 * float(knowledge_quality["knowledge_quality_score"]))

    def _write_metadata(self, sample: ForecastSample) -> Dict[str, object]:
        knowledge_quality = self._knowledge_quality_from_sample(sample)
        overlap_profile = self._counterfactual_overlap_profile_from_sample(sample)
        metadata = dict(sample.metadata)
        metadata.update(
            {
                "write_confidence": float(knowledge_quality["write_confidence"]),
                "freshness": 1.0,
                "pattern_label": sample.pattern_label,
                "trajectory_label": sample.trajectory_label,
                "experience_label": sample.experience_label,
                "regime_mix_score": sample.formation_features[15],
                "patient_static": list(sample.patient_static or sample.static),
                "intervention_static": list(sample.intervention_static or []),
                "intervention_sequence": [list(step) for step in (sample.intervention_sequence or [])],
                "kg_features": list(sample.kg_features or []),
                "kg_guideline_alignment": float(sample.metadata.get("kg_guideline_alignment", 0.0)),
                "knowledge_quality_score": float(knowledge_quality["knowledge_quality_score"]),
                "knowledge_feasibility_score": float(knowledge_quality["knowledge_feasibility_score"]),
                "knowledge_guideline_compatibility": float(knowledge_quality["guideline_compatibility"]),
                "knowledge_missing_care_penalty": float(knowledge_quality["missing_care_penalty"]),
                "knowledge_contraindication_penalty": float(knowledge_quality["contraindication_penalty"]),
                "knowledge_hard_valid": float(knowledge_quality["hard_valid"]),
                "knowledge_hard_filter_reason": "|".join(knowledge_quality["hard_filter_reasons"]),
                "recent_delta": float(overlap_profile.get("trend", 0.0)),
                "overlap_profile": {
                    "severity": float(overlap_profile.get("severity", 0.0)),
                    "trend": float(overlap_profile.get("trend", 0.0)),
                    "state_set": list(overlap_profile.get("state_set", [])),
                    "action_set": list(
                        overlap_profile.get("required_action_set", overlap_profile.get("action_set", []))
                    ),
                },
                "experience_id": forecast_sample_identity(sample),
            }
        )
        return metadata

    def _build_memory_bank(self, samples: Sequence[ForecastSample]):
        self.memory_manager.reset()
        self.eval()
        with torch.no_grad():
            batch_size = max(1, self.trainer_config.batch_size)
            for start in range(0, len(samples), batch_size):
                batch_samples = list(samples[start : start + batch_size])
                batch_encoding, _, _, _, _, _, _, _, _, _ = self._encode_batch(batch_samples)
                for index, sample in enumerate(batch_samples):
                    encoding = self._batch_encoding_output(batch_encoding, index, sample.metadata)
                    self.memory_manager.write(
                        encoding=encoding,
                        pattern_label=sample.pattern_label,
                        trajectory_label=sample.trajectory_label,
                        experience_label=sample.experience_label,
                        metadata=self._write_metadata(sample),
                        activity=self._activity_from_sample(sample),
                    )

    def _build_intervention_store(self, samples: Sequence[ForecastSample]) -> None:
        self.intervention_store_entries = []
        self.intervention_plan_store = {}
        self.intervention_method_store = {}
        self.intervention_dose_store = {}
        self.intervention_timing_store = {}
        self.intervention_context_store = {}
        self.eval()
        with torch.no_grad():
            batch_size = max(1, self.trainer_config.batch_size)
            for start in range(0, len(samples), batch_size):
                batch_samples = list(samples[start : start + batch_size])
                batch_encoding, _, _, _, _, _, _, _, _, _ = self._encode_batch(batch_samples)
                for offset, sample in enumerate(batch_samples):
                    intervention_static = [float(value) for value in (sample.intervention_static or [])]
                    intervention_sequence = [
                        [float(value) for value in step]
                        for step in (sample.intervention_sequence or [])
                    ]
                    intervention_plan_code = self._store_intervention_plan_components(
                        intervention_static=intervention_static,
                        intervention_sequence=intervention_sequence,
                        base_flags=dict(sample.metadata.get("kg_flags", {})),
                    )
                    metadata = self._write_metadata(sample)
                    metadata["intervention_plan_code"] = dict(intervention_plan_code)
                    entry = InterventionStoreEntry(
                        stay_id=float(sample.metadata.get("stay_id", -1.0)),
                        experience_id=str(sample.metadata.get("experience_id", forecast_sample_identity(sample))),
                        experience_label=int(sample.experience_label),
                        pattern_label=int(sample.pattern_label),
                        trajectory_label=int(sample.trajectory_label),
                        patient_embedding=[
                            float(value) for value in batch_encoding.input_embedding[offset].detach().cpu().tolist()
                        ],
                        intervention_plan_code=intervention_plan_code,
                        intervention_static=intervention_static,
                        intervention_sequence=intervention_sequence,
                        kg_features=[float(value) for value in (sample.kg_features or [])],
                        kg_guideline_score=float(sample.metadata.get("kg_guideline_alignment", 0.0)),
                        metadata=metadata,
                    )
                    self.intervention_store_entries.append(entry)
        self._rebuild_intervention_store_cache()

    def _build_transition_store(self, samples: Sequence[ForecastSample]) -> None:
        self.transition_store_entries = []
        self.transition_store_by_label = {}
        self.transition_state_cache = {}
        self.transition_action_cache = {}
        self.transition_future_cache = {}
        self.transition_utility_cache = {}
        self.transition_utility_vector_cache = {}
        self.transition_future_mean_cache = {}
        self.transition_label_support = {}
        self.transition_label_utility_mean = {}
        self.transition_label_improvement_rate = {}
        self.transition_label_utility_vector_mean = {}
        self.transition_signature_cache = {}
        self.transition_signature_payload_cache = {}
        self.transition_signature_counts = {}
        for sample in samples:
            state_vector = self._clinical_state_vector(sample)
            state_signature_payload = self._clinical_state_signature_payload(sample)
            action_vector, _ = self._intervention_action_vector(
                intervention_static=sample.intervention_static or [],
                intervention_sequence=sample.intervention_sequence or [],
                base_flags=dict(sample.metadata.get("kg_flags", {})),
            )
            utility_vector = self._transition_utility_vector_from_sample(sample)
            source_audit = self._transition_source_audit_metadata(sample)
            entry = TransitionStoreEntry(
                experience_label=int(sample.experience_label),
                state_vector=state_vector,
                action_vector=action_vector,
                future_curve=[float(value) for value in sample.target],
                transition_utility=float(self._transition_utility_from_sample(sample)),
                utility_vector=utility_vector,
                support=1,
                metadata={
                    "experience_id": str(sample.metadata.get("experience_id", forecast_sample_identity(sample))),
                    "stay_id": float(sample.metadata.get("stay_id", -1.0)),
                    "window_end_index": float(sample.metadata.get("window_end_index", -1.0)),
                    "clinical_state_signature": str(state_signature_payload["signature"]),
                    "clinical_state_signature_schema_version": int(state_signature_payload["schema_version"]),
                    "clinical_state_signature_features": dict(state_signature_payload),
                    "clinical_state_vector_dim": int(len(state_vector)),
                    **source_audit,
                },
            )
            index = len(self.transition_store_entries)
            self.transition_store_entries.append(entry)
            self.transition_store_by_label.setdefault(entry.experience_label, []).append(index)
            signature = str(entry.metadata.get("clinical_state_signature", ""))
            if signature:
                self.transition_signature_counts[signature] = self.transition_signature_counts.get(signature, 0) + 1
        for label, indices in self.transition_store_by_label.items():
            self.transition_state_cache[label] = torch.tensor(
                [self.transition_store_entries[index].state_vector for index in indices],
                dtype=torch.float32,
                device=self.device,
            )
            self.transition_action_cache[label] = torch.tensor(
                [self.transition_store_entries[index].action_vector for index in indices],
                dtype=torch.float32,
                device=self.device,
            )
            self.transition_future_cache[label] = torch.tensor(
                [self.transition_store_entries[index].future_curve for index in indices],
                dtype=torch.float32,
                device=self.device,
            )
            self.transition_utility_cache[label] = torch.tensor(
                [self.transition_store_entries[index].transition_utility for index in indices],
                dtype=torch.float32,
                device=self.device,
            )
            self.transition_utility_vector_cache[label] = torch.tensor(
                [self.transition_store_entries[index].utility_vector for index in indices],
                dtype=torch.float32,
                device=self.device,
            )
            self.transition_future_mean_cache[label] = self.transition_future_cache[label].mean(dim=0)
            utilities = [self.transition_store_entries[index].transition_utility for index in indices]
            self.transition_label_support[label] = int(len(indices))
            self.transition_label_utility_mean[label] = float(sum(utilities) / max(1, len(utilities)))
            self.transition_label_improvement_rate[label] = float(sum(1.0 for value in utilities if value > 0.0) / max(1, len(utilities)))
            self.transition_label_utility_vector_mean[label] = self.transition_utility_vector_cache[label].mean(dim=0)
            self.transition_signature_cache[label] = [
                str(self.transition_store_entries[index].metadata.get("clinical_state_signature", ""))
                for index in indices
            ]
            self.transition_signature_payload_cache[label] = [
                dict(self.transition_store_entries[index].metadata.get("clinical_state_signature_features", {}))
                for index in indices
            ]

    def _deduplicate_memory_samples(self, samples: Sequence[ForecastSample]) -> List[ForecastSample]:
        unique_samples: List[ForecastSample] = []
        seen = set()
        for sample in samples:
            experience_id = sample.metadata.get("experience_id")
            if experience_id is None:
                experience_id = forecast_sample_identity(sample)
            if experience_id in seen:
                continue
            seen.add(experience_id)
            unique_samples.append(sample)
        return unique_samples

    def _planner_tensor(self, planner_weights: Dict[str, float]) -> torch.Tensor:
        return torch.tensor(
            [planner_weights["pattern"], planner_weights["trajectory"], planner_weights["experience"]],
            dtype=torch.float32,
            device=self.device,
        )

    def _scenario_tensor(self, scenario_features: Dict[str, float]) -> torch.Tensor:
        return torch.tensor(
            [
                scenario_features["horizon_ratio"],
                scenario_features["seasonality_ratio"],
                scenario_features["series_density"],
                scenario_features["regime_mix"],
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def _kg_path_inputs(self, sample: ForecastSample) -> tuple[Optional[torch.Tensor], Dict[str, float]]:
        if self.kg_feature_dim <= 0:
            return None, {
                "kg_gate": 0.0,
                "kg_residual_strength": 0.0,
                "kg_feature_density": 0.0,
                "kg_guideline_alignment": 0.0,
                "kg_state_load": 0.0,
                "kg_care_load": 0.0,
            }
        kg_features = [float(value) for value in (sample.kg_features or [])]
        if len(kg_features) < self.kg_feature_dim:
            kg_features = kg_features + [0.0] * (self.kg_feature_dim - len(kg_features))
        elif len(kg_features) > self.kg_feature_dim:
            kg_features = kg_features[: self.kg_feature_dim]
        kg_flags = dict(sample.metadata.get("kg_flags", {}))
        active_count = sum(1.0 for value in kg_features if abs(float(value)) > 1e-8)
        feature_density = active_count / max(1.0, float(self.kg_feature_dim))
        state_load = (
            self._kg_flag(kg_flags, "state_sepsis")
            + self._kg_flag(kg_flags, "state_septic_shock")
            + self._kg_flag(kg_flags, "state_organ_dysfunction")
            + self._kg_flag(kg_flags, "state_hypotension")
            + self._kg_flag(kg_flags, "state_high_lactate")
        ) / 5.0
        care_load = (
            self._kg_flag(kg_flags, "treat_early_antimicrobial")
            + self._kg_flag(kg_flags, "treat_vasopressor")
            + self._kg_flag(kg_flags, "treat_respiratory_support")
            + self._kg_flag(kg_flags, "exam_blood_culture")
            + self._kg_flag(kg_flags, "exam_lactate")
            + self._kg_flag(kg_flags, "monitor_lactate_repeat")
        ) / 6.0
        guideline_alignment = float(sample.metadata.get("kg_guideline_alignment", 0.0))
        kg_input = torch.tensor(
            kg_features + [guideline_alignment, feature_density, state_load, care_load],
            dtype=torch.float32,
            device=self.device,
        )
        return kg_input, {
            "kg_gate": 0.0,
            "kg_residual_strength": 0.0,
            "kg_feature_density": float(feature_density),
            "kg_guideline_alignment": float(guideline_alignment),
            "kg_state_load": float(state_load),
            "kg_care_load": float(care_load),
        }

    def _kg_prediction_residual(self, sample: ForecastSample) -> tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
        zero_residual = torch.zeros((self.trainer_config.forecast_horizon,), dtype=torch.float32, device=self.device)
        zero_aux = {
            "kg_gate_tensor": torch.zeros((), dtype=torch.float32, device=self.device),
            "kg_target_tensor": torch.zeros((), dtype=torch.float32, device=self.device),
        }
        if (
            not self.enable_explicit_kg_path
            or self.kg_feature_dim <= 0
            or self.kg_projector is None
            or self.kg_residual_head is None
            or self.kg_gate is None
        ):
            return zero_residual, {
                "kg_gate": 0.0,
                "kg_residual_strength": 0.0,
                "kg_feature_density": 0.0,
                "kg_guideline_alignment": 0.0,
                "kg_state_load": 0.0,
                "kg_care_load": 0.0,
            }, zero_aux
        kg_input, kg_summary = self._kg_path_inputs(sample)
        if kg_input is None:
            return zero_residual, kg_summary, zero_aux
        gate_value = torch.sigmoid(self.kg_gate(kg_input)).squeeze(-1)
        alignment = max(self.kg_alignment_floor, 0.5 + 0.5 * float(kg_summary["kg_guideline_alignment"]))
        residual = self.kg_residual_head(self.kg_projector(kg_input)) * (self.kg_residual_weight * gate_value * alignment)
        kg_target = min(
            1.0,
            0.25
            + 0.35 * float(kg_summary["kg_state_load"])
            + 0.20 * float(kg_summary["kg_care_load"])
            + 0.20 * alignment,
        )
        kg_summary.update(
            {
                "kg_gate": float(gate_value.item()),
                "kg_residual_strength": float(residual.abs().mean().item()),
            }
        )
        return residual, kg_summary, {
            "kg_gate_tensor": gate_value,
            "kg_target_tensor": torch.tensor(kg_target, dtype=torch.float32, device=self.device),
        }

    def _bucket_residual(self, fused_representation: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
        short_residual = self.bucket_heads["short"](fused_representation) * self.bucket_masks[0]
        mid_residual = self.bucket_heads["mid"](fused_representation) * self.bucket_masks[1]
        long_residual = self.bucket_heads["long"](fused_representation) * self.bucket_masks[2]
        residual = short_residual + mid_residual + long_residual
        bucket_summary = {
            "short_bucket_strength": float(short_residual.abs().mean().item()),
            "mid_bucket_strength": float(mid_residual.abs().mean().item()),
            "long_bucket_strength": float(long_residual.abs().mean().item()),
        }
        return residual, bucket_summary

    def _build_fused_representation(
        self,
        factual_embedding: torch.Tensor,
        intervention_embedding: torch.Tensor,
        normalized_formation: torch.Tensor,
        manager_result: ManagerReadResult,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        planner_tensor = self._planner_tensor(manager_result.planner_weights)
        semantic_template_confidence = float(manager_result.semantic_result.get("template_confidence", 0.0))
        confidence_tensor = torch.tensor(
            [
                manager_result.component_results["pattern"].confidence,
                manager_result.component_results["trajectory"].confidence,
                max(manager_result.component_results["experience"].confidence, semantic_template_confidence),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        scenario_tensor = self._scenario_tensor(manager_result.scenario_features)
        gate_input = torch.cat(
            [
                factual_embedding,
                intervention_embedding,
                normalized_formation,
                planner_tensor,
                confidence_tensor,
                scenario_tensor,
            ],
            dim=-1,
        )
        learned_gate_logits = self.gate_network(gate_input)
        gate_weights = torch.softmax(
            learned_gate_logits
            + torch.log(planner_tensor.clamp_min(1e-6))
            + torch.log(confidence_tensor.clamp_min(0.05)),
            dim=-1,
        )

        projected_pattern = self.memory_projectors["pattern"](manager_result.component_results["pattern"].readout)
        projected_trajectory = self.memory_projectors["trajectory"](manager_result.component_results["trajectory"].readout)
        experience_readout = manager_result.component_results["experience"].readout
        semantic_blend_weight = float(manager_result.semantic_result.get("template_blend_weight", 0.0))
        semantic_curve = manager_result.semantic_result.get("template_curve", [])
        archive_weight = float(manager_result.component_results["experience"].archive_weight)
        archive_weight_tensor = manager_result.component_results["experience"].archive_weight_tensor
        if semantic_curve:
            semantic_tensor = torch.tensor(semantic_curve, dtype=torch.float32, device=self.device)
            semantic_value = self.prototype_curve_encoder(semantic_tensor)
            experience_readout = experience_readout * (1.0 - semantic_blend_weight) + semantic_value * semantic_blend_weight
        projected_experience = self.memory_projectors["experience"](experience_readout)

        gated_pattern = projected_pattern * gate_weights[0]
        gated_trajectory = projected_trajectory * gate_weights[1]
        gated_experience = projected_experience * gate_weights[2]

        direct_experience = self.memory_residual_heads["experience"](experience_readout)
        direct_experience = direct_experience * gate_weights[2] * confidence_tensor[2]
        direct_gate_input = torch.cat(
            [
                gate_weights[2:3],
                planner_tensor[2:3],
                confidence_tensor[2:3],
                scenario_tensor,
            ],
            dim=-1,
        )
        adaptive_direct_gate = torch.sigmoid(self.direct_residual_gate(direct_gate_input)).squeeze(-1)
        if self.memory_direct_residual_mode == "adaptive":
            direct_residual_scale = self.memory_direct_residual_weight * adaptive_direct_gate
        else:
            direct_residual_scale = torch.tensor(
                self.memory_direct_residual_weight,
                dtype=torch.float32,
                device=self.device,
            )
        direct_memory_residual = direct_residual_scale * direct_experience

        fusion_input = torch.cat(
            [
                factual_embedding,
                gated_pattern,
                gated_trajectory,
                gated_experience,
                intervention_embedding,
                gate_weights,
                planner_tensor,
                confidence_tensor,
                scenario_tensor,
            ],
            dim=-1,
        )
        fused_representation = self.multi_fusion(fusion_input)
        gate_summary = {
            "pattern_gate": float(gate_weights[0].item()),
            "trajectory_gate": float(gate_weights[1].item()),
            "experience_gate": float(gate_weights[2].item()),
            "memory_weighted_confidence": float((gate_weights * confidence_tensor).sum().item()),
            "semantic_template_blend_weight": float(semantic_blend_weight),
            "archive_weight": archive_weight,
            "direct_pattern_strength": 0.0,
            "direct_trajectory_strength": 0.0,
            "direct_experience_strength": float(direct_experience.abs().mean().item()),
            "direct_memory_strength": float(direct_memory_residual.abs().mean().item()),
            "adaptive_direct_gate": float(adaptive_direct_gate.item()),
            "direct_residual_scale": float(direct_residual_scale.item()),
            "pattern_confidence": float(confidence_tensor[0].item()),
            "trajectory_confidence": float(confidence_tensor[1].item()),
            "experience_confidence": float(confidence_tensor[2].item()),
            "_archive_weight_tensor": archive_weight_tensor
            if archive_weight_tensor is not None
            else torch.zeros((), dtype=torch.float32, device=self.device),
            "_experience_gate_tensor": gate_weights[2],
        }
        return fused_representation, direct_memory_residual, planner_tensor, confidence_tensor, scenario_tensor, gate_summary

    def _apply_memory_harm_control(
        self,
        base_prediction: torch.Tensor,
        coordinated_memory_residual: torch.Tensor,
        gate_summary: Dict[str, float],
    ) -> tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
        original_strength = coordinated_memory_residual.abs().mean()
        if not self.memory_harm_control_enabled:
            return coordinated_memory_residual, {
                "memory_harm_control_enabled": False,
                "memory_harm_control_scale": 1.0,
                "memory_harm_confidence_scale": 1.0,
                "memory_harm_quality": float(
                    gate_summary.get("memory_weighted_confidence", gate_summary.get("experience_confidence", 0.0))
                ),
                "memory_harm_alignment_scale": 1.0,
                "memory_harm_cap_scale": 1.0,
                "memory_harm_pre_strength": float(original_strength.item()),
                "memory_harm_post_strength": float(original_strength.item()),
            }, {"coordinated_strength_tensor": original_strength}

        quality_floor = max(1e-6, float(self.memory_quality_floor))
        sample_pattern = str(gate_summary.get("_sample_pattern", ""))
        sample_trajectory = str(gate_summary.get("_sample_trajectory", ""))
        if sample_trajectory == "stable_regime":
            quality_floor += max(0.0, float(self.harm_stable_quality_boost))
        if sample_pattern == "flat":
            quality_floor += max(0.0, float(self.harm_flat_quality_boost))
        memory_quality = float(
            gate_summary.get("memory_weighted_confidence", gate_summary.get("experience_confidence", 0.0))
        )
        confidence_scale = min(1.0, max(0.0, memory_quality / quality_floor))

        path_alignment = float(gate_summary.get("path_alignment", 0.0))
        min_alignment = float(self.memory_min_path_alignment)
        if path_alignment >= 0.0:
            alignment_scale = 1.0
        elif path_alignment <= min_alignment:
            alignment_scale = 0.0
        else:
            alignment_scale = (path_alignment - min_alignment) / max(1e-6, -min_alignment)

        residual_cap_ratio = max(0.0, float(self.memory_residual_cap_ratio))
        base_strength = base_prediction.abs().mean().detach()
        residual_cap = residual_cap_ratio * torch.clamp(base_strength, min=1.0)
        if float(original_strength.item()) <= 1e-8 or residual_cap_ratio <= 0.0:
            cap_scale = 1.0
        else:
            cap_scale = min(1.0, float((residual_cap / original_strength.clamp_min(1e-8)).item()))

        final_scale_value = min(confidence_scale, alignment_scale, cap_scale)
        final_scale = torch.tensor(final_scale_value, dtype=torch.float32, device=self.device)
        controlled_residual = coordinated_memory_residual * final_scale
        controlled_strength = controlled_residual.abs().mean()
        summary = {
            "memory_harm_control_enabled": True,
            "memory_harm_control_scale": float(final_scale_value),
            "memory_harm_confidence_scale": float(confidence_scale),
            "memory_harm_quality": float(memory_quality),
            "memory_harm_alignment_scale": float(alignment_scale),
            "memory_harm_cap_scale": float(cap_scale),
            "memory_harm_pre_strength": float(original_strength.item()),
            "memory_harm_post_strength": float(controlled_strength.item()),
        }
        return controlled_residual, summary, {"coordinated_strength_tensor": controlled_strength}

    def _coordinate_memory_paths(
        self,
        residual_prediction: torch.Tensor,
        direct_memory_residual: torch.Tensor,
        planner_tensor: torch.Tensor,
        confidence_tensor: torch.Tensor,
        scenario_tensor: torch.Tensor,
        gate_summary: Dict[str, float],
    ) -> tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
        path_disagreement = float((residual_prediction - direct_memory_residual).abs().mean().item())
        path_alignment_tensor = F.cosine_similarity(
            residual_prediction.unsqueeze(0),
            direct_memory_residual.unsqueeze(0),
            dim=-1,
        ).squeeze(0)
        path_alignment = float(path_alignment_tensor.item())
        if self.memory_path_coordination_mode == "adaptive":
            coordination_input = torch.cat(
                [
                    planner_tensor,
                    confidence_tensor,
                    scenario_tensor,
                    torch.tensor(
                        [
                            gate_summary["pattern_gate"],
                            gate_summary["trajectory_gate"],
                            gate_summary["experience_gate"],
                            gate_summary["adaptive_direct_gate"],
                            float(residual_prediction.abs().mean().item()),
                            float(direct_memory_residual.abs().mean().item()),
                        ],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                ],
                dim=-1,
            )
            path_weights = torch.softmax(self.memory_path_coordinator(coordination_input), dim=-1) * 2.0
            fusion_path_scale = path_weights[0]
            direct_path_scale = path_weights[1]
        else:
            fusion_path_scale = torch.tensor(1.0, dtype=torch.float32, device=self.device)
            direct_path_scale = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        coordinated_memory_residual = fusion_path_scale * residual_prediction + direct_path_scale * direct_memory_residual
        coordination_summary = {
            "fusion_path_scale": float(fusion_path_scale.item()),
            "direct_path_scale": float(direct_path_scale.item()),
            "coordinated_memory_strength": float(coordinated_memory_residual.abs().mean().item()),
            "path_disagreement": path_disagreement,
            "path_alignment": path_alignment,
        }
        return coordinated_memory_residual, coordination_summary, {
            "path_alignment_tensor": path_alignment_tensor,
            "fusion_strength_tensor": residual_prediction.abs().mean(),
            "direct_strength_tensor": direct_memory_residual.abs().mean(),
            "coordinated_strength_tensor": coordinated_memory_residual.abs().mean(),
        }

    def _forward_sample(self, sample: ForecastSample):
        normalized_sequence = self._normalize_sequence(sample.sequence)
        normalized_static = self._normalize_static(sample.patient_static or sample.static)
        normalized_intervention = self._normalize_intervention(sample.intervention_static or [])
        normalized_intervention_sequence = self._normalize_intervention_sequence(sample.intervention_sequence or [])
        normalized_formation = self._normalize_formation(sample.formation_features)
        encoding = self.manifold.encode_input_torch(
            normalized_sequence,
            normalized_static,
            metadata=sample.metadata,
        )
        temporal_profile = self._temporal_profile_tensor(sample.metadata if isinstance(sample.metadata, dict) else {})
        (
            factual_embedding,
            retrieval_encoding,
            factual_branch_gate,
            factual_branch_strength,
            retrieval_branch_gate,
            retrieval_branch_strength,
        ) = self._split_encoding_spaces(
            encoding,
            normalized_formation=normalized_formation,
            temporal_profile=temporal_profile,
        )
        factual_embedding, temporal_gate, temporal_strength = self._augment_factual_embedding(
            factual_embedding=factual_embedding,
            temporal_profile=temporal_profile,
            normalized_formation=normalized_formation,
        )
        intervention_embedding = self._project_intervention(
            normalized_intervention,
            normalized_intervention_sequence,
        )
        factual_prediction_input, factual_prediction_gate, factual_prediction_strength = self._augment_factual_prediction_input(
            base_input=torch.cat([factual_embedding, intervention_embedding], dim=-1),
            temporal_profile=temporal_profile,
            normalized_formation=normalized_formation,
        )
        predicted_factual_scale = self._predict_factual_scale(factual_prediction_input)
        raw_base_prediction = self.base_regressor(factual_prediction_input)
        kg_residual, kg_summary, _ = self._kg_prediction_residual(sample)
        base_prediction = raw_base_prediction + kg_residual
        manager_result = self.memory_manager.read(
            encoding=retrieval_encoding,
            horizon=self.trainer_config.forecast_horizon,
            history_length=self.trainer_config.history_length,
            formation_features=sample.formation_features,
            pattern_label=sample.pattern_label,
            trajectory_label=sample.trajectory_label,
            experience_label=sample.experience_label,
            **self._memory_read_kwargs(sample),
        )
        fused_representation, direct_memory_residual, planner_tensor, confidence_tensor, scenario_tensor, gate_summary = self._build_fused_representation(
            factual_embedding=factual_embedding,
            intervention_embedding=intervention_embedding,
            normalized_formation=normalized_formation,
            manager_result=manager_result,
        )
        multitask_outputs = self._predict_multitask_outputs(
            fused_representation=fused_representation,
            base_prediction=base_prediction,
            intervention_embedding=intervention_embedding,
        )
        transition_readout = self._empty_transition_readout()
        transition_residual = torch.zeros_like(base_prediction)
        if self.enable_transition_memory and self.enable_transition_factual_path:
            transition_readout = self._retrieve_transition_readout(
                sample=sample,
                intervention_static=sample.intervention_static or [],
                intervention_sequence=sample.intervention_sequence or [],
                candidate_labels=self._transition_candidate_labels(sample, manager_result=manager_result),
                base_flags=dict(sample.metadata.get("kg_flags", {})),
        )
        gate_summary.update(kg_summary)
        transition_trunk_adjustment, trunk_summary = self._transition_trunk_adjustment(transition_readout)
        gate_summary.update(trunk_summary)
        residual_prediction, bucket_summary = self._bucket_residual(fused_representation + transition_trunk_adjustment)
        coordinated_memory_residual, coordination_summary, _ = self._coordinate_memory_paths(
            residual_prediction=residual_prediction,
            direct_memory_residual=direct_memory_residual,
            planner_tensor=planner_tensor,
            confidence_tensor=confidence_tensor,
            scenario_tensor=scenario_tensor,
            gate_summary=gate_summary,
        )
        gate_summary.update(coordination_summary)
        gate_summary["_sample_pattern"] = (
            PATTERN_LABELS[int(sample.pattern_label)]
            if 0 <= int(sample.pattern_label) < len(PATTERN_LABELS)
            else ""
        )
        gate_summary["_sample_trajectory"] = (
            TRAJECTORY_LABELS[int(sample.trajectory_label)]
            if 0 <= int(sample.trajectory_label) < len(TRAJECTORY_LABELS)
            else ""
        )
        coordinated_memory_residual, harm_summary, _ = self._apply_memory_harm_control(
            base_prediction=base_prediction,
            coordinated_memory_residual=coordinated_memory_residual,
            gate_summary=gate_summary,
        )
        gate_summary.update(harm_summary)
        if self.enable_transition_memory and self.enable_transition_factual_path:
            transition_residual, transition_summary = self._transition_factual_residual(
                sample=sample,
                base_prediction=base_prediction,
                coordinated_memory_residual=coordinated_memory_residual,
                transition_readout=transition_readout,
            )
            gate_summary.update(transition_summary)
        else:
            gate_summary.update(
                {
                    "transition_alignment": 0.0,
                    "transition_gate_scale": 0.0,
                    "transition_residual_strength": 0.0,
                    "transition_raw_residual_strength": 0.0,
                    "transition_safe_gate": 0.0,
                    "transition_structural_blocked": 1.0,
                    "transition_utility_factor": 0.1,
                    "transition_pattern_factor": 0.0,
                    "transition_trajectory_factor": 0.0,
                    "transition_residual_cap": 0.0,
                    "transition_residual_cap_applied": 0.0,
                    "transition_gate_blocked": 1.0,
                    "transition_gate_reasons": "transition_disabled",
                }
        )
        gate_summary.update(
            {
                "factual_branch_gate": float(factual_branch_gate.item()),
                "factual_branch_strength": float(factual_branch_strength.item()),
                "retrieval_branch_gate": float(retrieval_branch_gate.item()),
                "retrieval_branch_strength": float(retrieval_branch_strength.item()),
                "factual_prediction_gate": float(factual_prediction_gate.item()),
                "factual_prediction_strength": float(factual_prediction_strength.item()),
                "factual_predicted_scale_mean": float(predicted_factual_scale.mean().item()),
                "temporal_factual_gate": float(temporal_gate.item()),
                "temporal_factual_strength": float(temporal_strength.item()),
                "transition_confidence": float(transition_readout["confidence"]),
                "transition_expected_utility": float(transition_readout["expected_utility"]),
                "transition_top_score": float(transition_readout["top_score"]),
                "transition_residual_scale": float(transition_readout["residual_scale"]),
                "transition_support_strength": float(transition_readout.get("support_strength", 0.0)),
                "transition_signature_weight": float(
                    transition_readout.get("matched_clinical_state_signature_weight", 0.0)
                ),
            }
        )
        gate_summary["_predicted_factual_scale_tensor"] = predicted_factual_scale
        for task_name, prediction in multitask_outputs.items():
            gate_summary[f"_multitask_{task_name}_tensor"] = prediction
        fusion_prediction = base_prediction + coordinated_memory_residual + transition_residual
        return retrieval_encoding, manager_result, base_prediction, fusion_prediction, gate_summary, bucket_summary

    def _forward_batch(
        self,
        samples: Sequence[ForecastSample],
    ) -> tuple[List[TorchManifoldEncodingOutput], List[ManagerReadResult], torch.Tensor, torch.Tensor, List[Dict[str, float]], List[Dict[str, float]]]:
        if not samples:
            empty_tensor = torch.zeros((0, self.trainer_config.forecast_horizon), dtype=torch.float32, device=self.device)
            return [], [], empty_tensor, empty_tensor, [], []
        (
            batch_encoding,
            factual_embedding_batch,
            intervention_embedding_batch,
            normalized_formation_batch,
            temporal_gate_batch,
            temporal_strength_batch,
            factual_branch_gate_batch,
            factual_branch_strength_batch,
            retrieval_branch_gate_batch,
            retrieval_branch_strength_batch,
        ) = self._encode_batch(samples)
        temporal_profile_batch = self._temporal_profile_batch(samples)
        factual_prediction_input_batch, factual_prediction_gate_batch, factual_prediction_strength_batch = self._augment_factual_prediction_input(
            base_input=torch.cat([factual_embedding_batch, intervention_embedding_batch], dim=-1),
            temporal_profile=temporal_profile_batch,
            normalized_formation=normalized_formation_batch,
        )
        predicted_factual_scale_batch = self._predict_factual_scale(factual_prediction_input_batch)
        raw_base_predictions = self.base_regressor(factual_prediction_input_batch)

        encodings: List[TorchManifoldEncodingOutput] = []
        manager_results: List[ManagerReadResult] = []
        base_predictions: List[torch.Tensor] = []
        fusion_predictions: List[torch.Tensor] = []
        gate_summaries: List[Dict[str, float]] = []
        bucket_summaries: List[Dict[str, float]] = []

        for index, sample in enumerate(samples):
            encoding = self._batch_encoding_output(batch_encoding, index, sample.metadata)
            manager_result = self.memory_manager.read(
                encoding=encoding,
                horizon=self.trainer_config.forecast_horizon,
                history_length=self.trainer_config.history_length,
                formation_features=sample.formation_features,
                pattern_label=sample.pattern_label,
                trajectory_label=sample.trajectory_label,
                experience_label=sample.experience_label,
                **self._memory_read_kwargs(sample),
            )
            fused_representation, direct_memory_residual, planner_tensor, confidence_tensor, scenario_tensor, gate_summary = self._build_fused_representation(
                factual_embedding=factual_embedding_batch[index],
                intervention_embedding=intervention_embedding_batch[index],
                normalized_formation=normalized_formation_batch[index],
                manager_result=manager_result,
            )
            gate_summary.update(
                {
                    "factual_branch_gate": float(factual_branch_gate_batch[index].item()),
                    "factual_branch_strength": float(factual_branch_strength_batch[index].item()),
                    "retrieval_branch_gate": float(retrieval_branch_gate_batch[index].item()),
                    "retrieval_branch_strength": float(retrieval_branch_strength_batch[index].item()),
                    "factual_prediction_gate": float(factual_prediction_gate_batch[index].item()),
                    "factual_prediction_strength": float(factual_prediction_strength_batch[index].item()),
                    "factual_predicted_scale_mean": float(predicted_factual_scale_batch[index].mean().item()),
                    "temporal_factual_gate": float(temporal_gate_batch[index].item()),
                    "temporal_factual_strength": float(temporal_strength_batch[index].item()),
                }
            )
            gate_summary["_predicted_factual_scale_tensor"] = predicted_factual_scale_batch[index]
            transition_readout = self._empty_transition_readout()
            transition_residual = torch.zeros_like(raw_base_predictions[index])
            if self.enable_transition_memory and self.enable_transition_factual_path:
                transition_readout = self._retrieve_transition_readout(
                    sample=sample,
                    intervention_static=sample.intervention_static or [],
                    intervention_sequence=sample.intervention_sequence or [],
                    candidate_labels=self._transition_candidate_labels(sample, manager_result=manager_result),
                    base_flags=dict(sample.metadata.get("kg_flags", {})),
                )
            transition_trunk_adjustment, trunk_summary = self._transition_trunk_adjustment(transition_readout)
            gate_summary.update(trunk_summary)
            residual_prediction, bucket_summary = self._bucket_residual(fused_representation + transition_trunk_adjustment)
            coordinated_memory_residual, coordination_summary, coordination_aux = self._coordinate_memory_paths(
                residual_prediction=residual_prediction,
                direct_memory_residual=direct_memory_residual,
                planner_tensor=planner_tensor,
                confidence_tensor=confidence_tensor,
                scenario_tensor=scenario_tensor,
                gate_summary=gate_summary,
            )
            gate_summary.update(coordination_summary)
            sample_obj = samples[index]
            gate_summary["_sample_pattern"] = (
                PATTERN_LABELS[int(sample_obj.pattern_label)]
                if 0 <= int(sample_obj.pattern_label) < len(PATTERN_LABELS)
                else ""
            )
            gate_summary["_sample_trajectory"] = (
                TRAJECTORY_LABELS[int(sample_obj.trajectory_label)]
                if 0 <= int(sample_obj.trajectory_label) < len(TRAJECTORY_LABELS)
                else ""
            )
            coordinated_memory_residual, harm_summary, harm_aux = self._apply_memory_harm_control(
                base_prediction=raw_base_predictions[index],
                coordinated_memory_residual=coordinated_memory_residual,
                gate_summary=gate_summary,
            )
            gate_summary.update(harm_summary)
            gate_summary["_archive_signal_tensor"] = (
                torch.as_tensor(gate_summary.get("_archive_weight_tensor"), dtype=torch.float32, device=self.device)
                * torch.as_tensor(gate_summary.get("_experience_gate_tensor"), dtype=torch.float32, device=self.device)
            )
            kg_residual, kg_summary, kg_aux = self._kg_prediction_residual(sample)
            structural_base_prediction = raw_base_predictions[index] + kg_residual
            gate_summary.update(kg_summary)
            multitask_outputs = self._predict_multitask_outputs(
                fused_representation=fused_representation,
                base_prediction=structural_base_prediction,
                intervention_embedding=intervention_embedding_batch[index],
            )
            if self.enable_transition_memory and self.enable_transition_factual_path:
                transition_residual, transition_summary = self._transition_factual_residual(
                    sample=sample,
                    base_prediction=structural_base_prediction,
                    coordinated_memory_residual=coordinated_memory_residual,
                    transition_readout=transition_readout,
                )
                gate_summary.update(transition_summary)
            else:
                gate_summary.update(
                    {
                        "transition_alignment": 0.0,
                        "transition_gate_scale": 0.0,
                        "transition_residual_strength": 0.0,
                        "transition_raw_residual_strength": 0.0,
                        "transition_safe_gate": 0.0,
                        "transition_structural_blocked": 1.0,
                        "transition_utility_factor": 0.1,
                        "transition_pattern_factor": 0.0,
                        "transition_trajectory_factor": 0.0,
                        "transition_residual_cap": 0.0,
                        "transition_residual_cap_applied": 0.0,
                        "transition_gate_blocked": 1.0,
                        "transition_gate_reasons": "transition_disabled",
                    }
                )
            gate_summary.update(
                {
                    "transition_confidence": float(transition_readout["confidence"]),
                    "transition_expected_utility": float(transition_readout["expected_utility"]),
                    "transition_top_score": float(transition_readout["top_score"]),
                    "transition_residual_scale": float(transition_readout["residual_scale"]),
                    "transition_support_strength": float(transition_readout.get("support_strength", 0.0)),
                    "transition_signature_weight": float(
                        transition_readout.get("matched_clinical_state_signature_weight", 0.0)
                    ),
                }
            )
            encodings.append(encoding)
            manager_results.append(manager_result)
            base_predictions.append(structural_base_prediction)
            fusion_predictions.append(structural_base_prediction + coordinated_memory_residual + transition_residual)
            gate_summary["_kg_gate_tensor"] = kg_aux["kg_gate_tensor"]
            gate_summary["_kg_target_tensor"] = kg_aux["kg_target_tensor"]
            gate_summary["_path_alignment_tensor"] = coordination_aux["path_alignment_tensor"]
            gate_summary["_memory_delta_tensor"] = harm_aux["coordinated_strength_tensor"]
            for task_name, prediction in multitask_outputs.items():
                gate_summary[f"_multitask_{task_name}_tensor"] = prediction
            gate_summaries.append(gate_summary)
            bucket_summaries.append(bucket_summary)
        return (
            encodings,
            manager_results,
            torch.stack(base_predictions, dim=0),
            torch.stack(fusion_predictions, dim=0),
            gate_summaries,
            bucket_summaries,
        )

    def _hard_example_weights(self, fusion_losses: torch.Tensor) -> torch.Tensor:
        if fusion_losses.numel() == 0 or self.dynamic_hard_example_weight <= 1e-8:
            return torch.ones_like(fusion_losses)
        detached = fusion_losses.detach()
        centered = detached - detached.mean()
        scale = detached.std(unbiased=False).clamp_min(1e-6)
        weights = 1.0 + self.dynamic_hard_example_weight * torch.relu(centered / scale)
        return torch.clamp(weights, min=1.0, max=1.0 + 2.5 * self.dynamic_hard_example_weight)

    def _auxiliary_feedback_losses(
        self,
        gate_summaries: Sequence[Dict[str, object]],
        sample_weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        zero = torch.zeros((), dtype=torch.float32, device=self.device)
        if not gate_summaries:
            return {
                "kg_consistency_loss": zero,
                "path_alignment_loss": zero,
                "archive_retention_loss": zero,
                "memory_delta_floor_loss": zero,
            }
        kg_gates = torch.stack(
            [
                torch.as_tensor(summary.get("_kg_gate_tensor", zero), dtype=torch.float32, device=self.device)
                for summary in gate_summaries
            ]
        )
        kg_targets = torch.stack(
            [
                torch.as_tensor(summary.get("_kg_target_tensor", zero), dtype=torch.float32, device=self.device)
                for summary in gate_summaries
            ]
        )
        path_alignments = torch.stack(
            [
                torch.as_tensor(summary.get("_path_alignment_tensor", zero), dtype=torch.float32, device=self.device)
                for summary in gate_summaries
            ]
        )
        archive_signals = torch.stack(
            [
                torch.as_tensor(summary.get("_archive_signal_tensor", zero), dtype=torch.float32, device=self.device)
                for summary in gate_summaries
            ]
        )
        memory_deltas = torch.stack(
            [
                torch.as_tensor(summary.get("_memory_delta_tensor", zero), dtype=torch.float32, device=self.device)
                for summary in gate_summaries
            ]
        )
        focus_weights = 1.0 + torch.relu(sample_weights.detach() - 1.0)
        archive_target = torch.full_like(archive_signals, float(self.trainer_config.archive_retention_target))
        memory_delta_target = torch.full_like(memory_deltas, float(self.trainer_config.memory_delta_floor))
        return {
            "kg_consistency_loss": (torch.relu(kg_targets - kg_gates).pow(2) * focus_weights).mean(),
            "path_alignment_loss": (torch.relu(-path_alignments) * focus_weights).mean(),
            "archive_retention_loss": (torch.relu(archive_target - archive_signals) * focus_weights).mean(),
            "memory_delta_floor_loss": (torch.relu(memory_delta_target - memory_deltas) * focus_weights).mean(),
        }

    def _multitask_prediction_losses(
        self,
        batch_samples: Sequence[ForecastSample],
        gate_summaries: Sequence[Dict[str, object]],
        sample_weights: torch.Tensor,
    ) -> Dict[str, object]:
        zero = torch.zeros((), dtype=torch.float32, device=self.device)
        task_losses: Dict[str, torch.Tensor] = {}
        task_counts: Dict[str, int] = {}
        total_terms: List[torch.Tensor] = []
        for task_name in self.multitask_task_names:
            task_predictions: List[torch.Tensor] = []
            task_targets: List[float] = []
            task_weight_pieces: List[torch.Tensor] = []
            for sample, gate_summary, sample_weight in zip(batch_samples, gate_summaries, sample_weights):
                aux_targets = self._sample_aux_targets(sample)
                if task_name not in aux_targets:
                    continue
                prediction_tensor = gate_summary.get(f"_multitask_{task_name}_tensor")
                if prediction_tensor is None:
                    continue
                task_predictions.append(torch.as_tensor(prediction_tensor, dtype=torch.float32, device=self.device).reshape(()))
                task_targets.append(float(aux_targets[task_name]))
                task_weight_pieces.append(torch.as_tensor(sample_weight, dtype=torch.float32, device=self.device).reshape(()))
            task_counts[task_name] = len(task_targets)
            if not task_predictions:
                task_losses[task_name] = zero
                continue
            prediction_tensor = torch.stack(task_predictions)
            weight_tensor = torch.stack(task_weight_pieces)
            target_tensor = torch.tensor(task_targets, dtype=torch.float32, device=self.device)
            if task_name in self.multitask_binary_tasks:
                losses = F.binary_cross_entropy_with_logits(prediction_tensor, target_tensor, reduction="none")
            else:
                normalized_targets = torch.tensor(
                    [self._normalize_aux_target(task_name, value) for value in task_targets],
                    dtype=torch.float32,
                    device=self.device,
                )
                losses = F.smooth_l1_loss(prediction_tensor, normalized_targets, reduction="none")
            task_loss = (losses * weight_tensor).mean()
            task_losses[task_name] = task_loss
            total_terms.append(task_loss)
        total_loss = torch.stack(total_terms).mean() if total_terms else zero
        return {
            "total_loss": total_loss,
            "task_losses": task_losses,
            "task_counts": task_counts,
        }

    def _factual_calibration_loss(
        self,
        base_predictions: torch.Tensor,
        targets: torch.Tensor,
        gate_summaries: Sequence[Dict[str, object]],
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.factual_calibration_weight <= 0.0:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        scale_tensors: List[torch.Tensor] = []
        for gate_summary in gate_summaries:
            scale_tensor = gate_summary.get("_predicted_factual_scale_tensor")
            if scale_tensor is None:
                return torch.zeros((), dtype=torch.float32, device=self.device)
            scale_tensors.append(torch.as_tensor(scale_tensor, dtype=torch.float32, device=self.device))
        if not scale_tensors:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        predicted_scale = torch.stack(scale_tensors, dim=0).clamp_min(1e-3)
        squared_error = (base_predictions - targets) ** 2
        nll = 0.5 * (squared_error / predicted_scale.pow(2) + 2.0 * torch.log(predicted_scale))
        per_sample_loss = nll.mean(dim=-1)
        weight_tensor = torch.as_tensor(sample_weights, dtype=torch.float32, device=self.device).reshape(-1)
        return (per_sample_loss * weight_tensor).mean()

    def _analyze_epoch_errors(
        self,
        samples: Sequence[ForecastSample],
        max_samples: int = 256,
    ) -> Dict[str, object]:
        subset = list(samples[: max(0, int(max_samples))]) if max_samples > 0 else list(samples)
        if not subset:
            return {
                "sample_count": 0.0,
                "hard_case_count": 0.0,
                "hard_case_threshold": 0.0,
                "mean_error": 0.0,
                "hard_case_mean_error": 0.0,
                "kg_underused_rate": 0.0,
                "path_conflict_rate": 0.0,
                "semantic_hit_rate_on_hard_cases": 0.0,
                "archive_use_rate_on_hard_cases": 0.0,
                "mean_semantic_top_score_on_hard_cases": 0.0,
                "mean_kg_gate_on_hard_cases": 0.0,
                "mean_path_alignment_on_hard_cases": 0.0,
                "mean_experience_gate_on_hard_cases": 0.0,
                "kg_active_case_count": 0.0,
                "kg_active_hard_case_count": 0.0,
                "slices": {},
            }

        records: List[Dict[str, float | bool]] = []
        self.eval()
        with torch.no_grad():
            batch_size = max(1, self.trainer_config.batch_size)
            for start in range(0, len(subset), batch_size):
                batch_samples = subset[start : start + batch_size]
                targets = torch.tensor([sample.target for sample in batch_samples], dtype=torch.float32, device=self.device)
                _, manager_results, base_preds, fusion_preds, gate_summaries, _ = self._forward_batch(batch_samples)
                batch_errors = F.smooth_l1_loss(fusion_preds, targets, reduction="none").mean(dim=-1).detach().cpu().tolist()
                for batch_index, (manager_result, gate_summary, error_value) in enumerate(zip(manager_results, gate_summaries, batch_errors)):
                    experience_result = manager_result.component_results["experience"]
                    semantic_result = manager_result.semantic_result
                    kg_state_load = float(gate_summary.get("kg_state_load", 0.0))
                    kg_guideline_alignment = float(gate_summary.get("kg_guideline_alignment", 0.0))
                    kg_gate = float(gate_summary.get("kg_gate", 0.0))
                    path_alignment = float(gate_summary.get("path_alignment", 0.0))
                    kg_active = bool((kg_state_load > 0.20) or (kg_guideline_alignment > 0.10))
                    kg_underused = bool(kg_active and kg_gate < 0.25)
                    records.append(
                        {
                            "error": float(error_value),
                            "kg_gate": kg_gate,
                            "kg_state_load": kg_state_load,
                            "kg_guideline_alignment": kg_guideline_alignment,
                            "path_alignment": path_alignment,
                            "semantic_hit": bool(int(semantic_result.get("hit_count", 0)) > 0),
                            "semantic_top_score": float(semantic_result.get("top_score", 0.0)),
                            "archive_used": bool(experience_result.archive_used),
                            "experience_gate": float(gate_summary.get("experience_gate", 0.0)),
                            "memory_delta_strength": float((fusion_preds[batch_index] - base_preds[batch_index]).abs().mean().item()),
                            "kg_active": kg_active,
                            "kg_underused": kg_underused,
                            "path_conflict": bool(path_alignment < 0.0),
                        }
                    )

        errors = sorted(float(record["error"]) for record in records)
        quantile_index = min(
            len(errors) - 1,
            max(0, int(math.floor((1.0 - self.trainer_config.feedback_top_error_rate) * (len(errors) - 1)))),
        )
        hard_threshold = float(errors[quantile_index])
        hard_records = [record for record in records if float(record["error"]) >= hard_threshold]
        kg_active_records = [record for record in records if bool(record["kg_active"])]
        kg_active_hard_records = [record for record in hard_records if bool(record["kg_active"])]

        def _rate(predicate) -> float:
            if not hard_records:
                return 0.0
            return float(sum(1.0 for record in hard_records if predicate(record)) / len(hard_records))

        def _slice_summary(slice_records: Sequence[Dict[str, float | bool]]) -> Dict[str, float]:
            if not slice_records:
                return {
                    "count": 0.0,
                    "mean_error": 0.0,
                    "kg_underused_rate": 0.0,
                    "path_conflict_rate": 0.0,
                    "semantic_hit_rate": 0.0,
                    "archive_use_rate": 0.0,
                    "mean_semantic_top_score": 0.0,
                    "mean_kg_gate": 0.0,
                    "mean_path_alignment": 0.0,
                    "mean_experience_gate": 0.0,
                    "mean_memory_delta_strength": 0.0,
                }
            return {
                "count": float(len(slice_records)),
                "mean_error": float(sum(float(record["error"]) for record in slice_records) / len(slice_records)),
                "kg_underused_rate": float(sum(1.0 for record in slice_records if bool(record["kg_underused"])) / len(slice_records)),
                "path_conflict_rate": float(sum(1.0 for record in slice_records if bool(record["path_conflict"])) / len(slice_records)),
                "semantic_hit_rate": float(sum(1.0 for record in slice_records if bool(record["semantic_hit"])) / len(slice_records)),
                "archive_use_rate": float(sum(1.0 for record in slice_records if bool(record["archive_used"])) / len(slice_records)),
                "mean_semantic_top_score": float(sum(float(record["semantic_top_score"]) for record in slice_records) / len(slice_records)),
                "mean_kg_gate": float(sum(float(record["kg_gate"]) for record in slice_records) / len(slice_records)),
                "mean_path_alignment": float(sum(float(record["path_alignment"]) for record in slice_records) / len(slice_records)),
                "mean_experience_gate": float(sum(float(record["experience_gate"]) for record in slice_records) / len(slice_records)),
                "mean_memory_delta_strength": float(sum(float(record["memory_delta_strength"]) for record in slice_records) / len(slice_records)),
            }

        slice_summaries = {
            "overall": _slice_summary(records),
            "hard_cases": _slice_summary(hard_records),
            "kg_active_cases": _slice_summary(kg_active_records),
            "kg_active_hard_cases": _slice_summary(kg_active_hard_records),
        }

        return {
            "sample_count": float(len(records)),
            "hard_case_count": float(len(hard_records)),
            "hard_case_threshold": hard_threshold,
            "mean_error": float(sum(float(record["error"]) for record in records) / max(1, len(records))),
            "hard_case_mean_error": float(sum(float(record["error"]) for record in hard_records) / max(1, len(hard_records))),
            "kg_underused_rate": _rate(
                lambda record: (
                    (float(record["kg_state_load"]) > 0.20 or float(record["kg_guideline_alignment"]) > 0.10)
                    and float(record["kg_gate"]) < 0.25
                )
            ),
            "path_conflict_rate": _rate(lambda record: float(record["path_alignment"]) < 0.0),
            "semantic_hit_rate_on_hard_cases": _rate(lambda record: bool(record["semantic_hit"])),
            "archive_use_rate_on_hard_cases": _rate(lambda record: bool(record["archive_used"])),
            "mean_semantic_top_score_on_hard_cases": float(
                sum(float(record["semantic_top_score"]) for record in hard_records) / max(1, len(hard_records))
            ),
            "mean_kg_gate_on_hard_cases": float(
                sum(float(record["kg_gate"]) for record in hard_records) / max(1, len(hard_records))
            ),
            "mean_path_alignment_on_hard_cases": float(
                sum(float(record["path_alignment"]) for record in hard_records) / max(1, len(hard_records))
            ),
            "mean_experience_gate_on_hard_cases": float(
                sum(float(record["experience_gate"]) for record in hard_records) / max(1, len(hard_records))
            ),
            "kg_active_case_count": float(len(kg_active_records)),
            "kg_active_hard_case_count": float(len(kg_active_hard_records)),
            "slices": slice_summaries,
        }

    def analyze_error_attribution(
        self,
        samples: Sequence[ForecastSample],
        max_samples: int = 256,
    ) -> Dict[str, object]:
        return self._analyze_epoch_errors(samples, max_samples=max_samples)

    def _update_epoch_feedback(self, summary: Dict[str, object]) -> Dict[str, float]:
        momentum = min(0.95, max(0.0, float(self.trainer_config.epoch_feedback_momentum)))
        base_hard = max(0.0, float(self.trainer_config.hard_example_weight))
        base_kg = max(0.0, float(self.trainer_config.kg_consistency_weight))
        base_path = max(0.0, float(self.trainer_config.path_alignment_weight))
        hard_case_rate = float(summary.get("hard_case_count", 0.0)) / max(1.0, float(summary.get("sample_count", 0.0)))
        kg_active_underuse = float(summary.get("slices", {}).get("kg_active_cases", {}).get("kg_underused_rate", 0.0))
        target_hard = base_hard * (1.0 + 0.50 * hard_case_rate)
        target_kg = base_kg * (1.0 + 1.20 * max(float(summary.get("kg_underused_rate", 0.0)), kg_active_underuse))
        target_path = base_path * (1.0 + 1.10 * float(summary.get("path_conflict_rate", 0.0)))
        base_archive = max(0.0, float(self.trainer_config.archive_retention_weight))
        base_memory_delta = max(0.0, float(self.trainer_config.memory_delta_floor_weight))
        hard_archive_use = float(summary.get("archive_use_rate_on_hard_cases", 0.0))
        archive_gap = max(0.0, float(self.trainer_config.archive_retention_target) - hard_archive_use)
        memory_delta_gap = max(
            0.0,
            float(self.trainer_config.memory_delta_floor)
            - float(summary.get("slices", {}).get("hard_cases", {}).get("mean_memory_delta_strength", 0.0)),
        )
        target_archive = base_archive * (1.0 + 1.60 * archive_gap)
        target_memory_delta = base_memory_delta * (1.0 + 1.40 * memory_delta_gap)
        self.dynamic_hard_example_weight = (1.0 - momentum) * self.dynamic_hard_example_weight + momentum * target_hard
        self.dynamic_kg_consistency_weight = (1.0 - momentum) * self.dynamic_kg_consistency_weight + momentum * target_kg
        self.dynamic_path_alignment_weight = (1.0 - momentum) * self.dynamic_path_alignment_weight + momentum * target_path
        self.dynamic_archive_retention_weight = (1.0 - momentum) * self.dynamic_archive_retention_weight + momentum * target_archive
        self.dynamic_memory_delta_floor_weight = (1.0 - momentum) * self.dynamic_memory_delta_floor_weight + momentum * target_memory_delta
        return {
            "hard_example_weight": float(self.dynamic_hard_example_weight),
            "kg_consistency_weight": float(self.dynamic_kg_consistency_weight),
            "path_alignment_weight": float(self.dynamic_path_alignment_weight),
            "archive_retention_weight": float(self.dynamic_archive_retention_weight),
            "memory_delta_floor_weight": float(self.dynamic_memory_delta_floor_weight),
        }

    def fit(
        self,
        train_samples: Sequence[ForecastSample],
        val_samples: Sequence[ForecastSample],
        memory_seed_samples: Optional[Sequence[ForecastSample]] = None,
        collect_diagnostics: bool = True,
    ):
        self._fit_normalizers(train_samples)
        if self.enable_epoch_feedback and self.memory_path_coordination_mode == "sum":
            self.memory_path_coordination_mode = "adaptive"
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.trainer_config.learning_rate,
            weight_decay=self.trainer_config.weight_decay,
        )
        memory_samples = self._deduplicate_memory_samples(memory_seed_samples or train_samples)
        self._build_memory_bank(memory_samples)
        memory_bank_build_count = 1

        best_state = copy.deepcopy(self.state_dict())
        best_score = float("inf")
        best_epoch = 0

        for epoch in range(self.trainer_config.epochs):
            self.best_epoch = epoch
            self.train()
            epoch_loss = 0.0
            batch_count = 0
            epoch_fusion_loss = 0.0
            epoch_base_loss = 0.0
            epoch_kg_loss = 0.0
            epoch_path_loss = 0.0
            epoch_archive_loss = 0.0
            epoch_memory_delta_loss = 0.0
            epoch_factual_calibration_loss = 0.0
            epoch_multitask_loss = 0.0
            epoch_hard_weight = 0.0
            epoch_multitask_task_losses = {task_name: 0.0 for task_name in self.multitask_task_names}
            epoch_multitask_task_counts = {task_name: 0.0 for task_name in self.multitask_task_names}

            for batch_indices in self._iter_batches(len(train_samples)):
                optimizer.zero_grad()
                batch_samples = [train_samples[sample_index] for sample_index in batch_indices]
                targets = torch.tensor([sample.target for sample in batch_samples], dtype=torch.float32, device=self.device)
                encodings, _, base_preds, fusion_preds, gate_summaries, _ = self._forward_batch(batch_samples)
                fusion_losses = F.smooth_l1_loss(fusion_preds, targets, reduction="none").mean(dim=-1)
                base_losses = F.smooth_l1_loss(base_preds, targets, reduction="none").mean(dim=-1)
                sample_weights = self._hard_example_weights(fusion_losses)
                auxiliary_losses = self._auxiliary_feedback_losses(gate_summaries, sample_weights=sample_weights)
                multitask_losses = self._multitask_prediction_losses(
                    batch_samples=batch_samples,
                    gate_summaries=gate_summaries,
                    sample_weights=sample_weights,
                )
                factual_calibration_loss = self._factual_calibration_loss(
                    base_predictions=base_preds,
                    targets=targets,
                    gate_summaries=gate_summaries,
                    sample_weights=sample_weights,
                )
                regularization = self._manifold_regularization(encodings, [sample.metadata for sample in batch_samples])
                loss = (
                    (fusion_losses * sample_weights).mean()
                    + self.trainer_config.aux_base_loss_weight * (base_losses * sample_weights).mean()
                    + self.trainer_config.multitask_loss_weight * multitask_losses["total_loss"]
                    + self.factual_calibration_weight * factual_calibration_loss
                    + regularization
                    + self.dynamic_kg_consistency_weight * auxiliary_losses["kg_consistency_loss"]
                    + self.dynamic_path_alignment_weight * auxiliary_losses["path_alignment_loss"]
                    + self.dynamic_archive_retention_weight * auxiliary_losses["archive_retention_loss"]
                    + self.dynamic_memory_delta_floor_weight * auxiliary_losses["memory_delta_floor_loss"]
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.trainer_config.grad_clip)
                optimizer.step()

                epoch_loss += float(loss.item())
                epoch_fusion_loss += float(fusion_losses.mean().item())
                epoch_base_loss += float(base_losses.mean().item())
                epoch_kg_loss += float(auxiliary_losses["kg_consistency_loss"].item())
                epoch_path_loss += float(auxiliary_losses["path_alignment_loss"].item())
                epoch_archive_loss += float(auxiliary_losses["archive_retention_loss"].item())
                epoch_memory_delta_loss += float(auxiliary_losses["memory_delta_floor_loss"].item())
                epoch_factual_calibration_loss += float(factual_calibration_loss.item())
                epoch_multitask_loss += float(multitask_losses["total_loss"].item())
                epoch_hard_weight += float(sample_weights.mean().item())
                for task_name in self.multitask_task_names:
                    epoch_multitask_task_losses[task_name] += float(multitask_losses["task_losses"][task_name].item())
                    epoch_multitask_task_counts[task_name] += float(multitask_losses["task_counts"].get(task_name, 0))
                batch_count += 1

            memory_refresh_interval = max(0, int(self.trainer_config.memory_refresh_interval))
            if memory_refresh_interval > 0 and (epoch + 1) % memory_refresh_interval == 0:
                self._build_memory_bank(memory_samples)
                memory_bank_build_count += 1
            val_metrics = self.evaluate(val_samples, use_memory=True)
            epoch_feedback_summary: Dict[str, object] = {}
            updated_weights = {
                "hard_example_weight": float(self.dynamic_hard_example_weight),
                "kg_consistency_weight": float(self.dynamic_kg_consistency_weight),
                "path_alignment_weight": float(self.dynamic_path_alignment_weight),
                "archive_retention_weight": float(self.dynamic_archive_retention_weight),
                "memory_delta_floor_weight": float(self.dynamic_memory_delta_floor_weight),
            }
            if self.enable_epoch_feedback:
                epoch_feedback_summary = self._analyze_epoch_errors(val_samples)
                updated_weights = self._update_epoch_feedback(epoch_feedback_summary)
                self.epoch_feedback_history.append(
                    {
                        "epoch": float(epoch),
                        "error_attribution": epoch_feedback_summary,
                        "updated_weights": updated_weights,
                    }
                )
            if val_metrics["mae"] < best_score:
                best_score = val_metrics["mae"]
                best_state = copy.deepcopy(self.state_dict())
                best_epoch = epoch

            self.training_summary = {
                "last_epoch_loss": epoch_loss / max(1, batch_count),
                "last_epoch_fusion_loss": epoch_fusion_loss / max(1, batch_count),
                "last_epoch_base_loss": epoch_base_loss / max(1, batch_count),
                "last_epoch_kg_consistency_loss": epoch_kg_loss / max(1, batch_count),
                "last_epoch_path_alignment_loss": epoch_path_loss / max(1, batch_count),
                "last_epoch_archive_retention_loss": epoch_archive_loss / max(1, batch_count),
                "last_epoch_memory_delta_floor_loss": epoch_memory_delta_loss / max(1, batch_count),
                "last_epoch_factual_calibration_loss": epoch_factual_calibration_loss / max(1, batch_count),
                "last_epoch_multitask_loss": epoch_multitask_loss / max(1, batch_count),
                "last_epoch_multitask_task_losses": {
                    task_name: epoch_multitask_task_losses[task_name] / max(1, batch_count)
                    for task_name in self.multitask_task_names
                },
                "last_epoch_multitask_task_counts": dict(epoch_multitask_task_counts),
                "last_epoch_mean_hard_weight": epoch_hard_weight / max(1, batch_count),
                "best_val_mae": best_score,
                "best_epoch": float(best_epoch),
                "epoch_feedback_enabled": bool(self.enable_epoch_feedback),
                "dynamic_loss_weights": updated_weights,
                "epoch_feedback_history": list(self.epoch_feedback_history),
            }

        if self.trainer_config.checkpoint_selection_mode == "best_val_mae":
            self.load_state_dict(best_state)
        self._restore_or_rebuild_final_memory_bank(memory_samples)
        memory_bank_build_count += 0 if self.neural_cache_status.get("loaded") else 1
        self._build_intervention_store(memory_samples)
        self._build_transition_store(memory_samples)
        if collect_diagnostics:
            self.memory_diagnostics = self._collect_diagnostics(memory_samples)
        else:
            self.memory_diagnostics = {}
        self.memory_diagnostics["memory_seed_sample_count"] = float(len(memory_samples))
        self.memory_diagnostics["intervention_store_size"] = float(len(self.intervention_store_entries))
        self.memory_diagnostics["intervention_component_store"] = {
            "schema_version": 2,
            "plan_count": float(len(self.intervention_plan_store)),
            "method_component_count": float(len(self.intervention_method_store)),
            "dose_component_count": float(len(self.intervention_dose_store)),
            "timing_component_count": float(len(self.intervention_timing_store)),
            "context_component_count": float(len(self.intervention_context_store)),
            "entry_uses_component_codes_rate": float(
                sum(1.0 for entry in self.intervention_store_entries if bool(entry.intervention_plan_code))
                / max(1, len(self.intervention_store_entries))
            ),
        }
        self.memory_diagnostics["transition_store_size"] = float(len(self.transition_store_entries))
        self.memory_diagnostics["transition_store_signature_audit"] = self._transition_store_signature_audit()
        self.memory_diagnostics["transition_memory_enabled"] = bool(self.enable_transition_memory)
        self.memory_diagnostics["transition_factual_path_enabled"] = bool(self.enable_transition_factual_path)
        self.memory_diagnostics["transition_donor_path_enabled"] = bool(self.enable_transition_donor_path)
        self.memory_diagnostics["neural_cache"] = dict(self.neural_cache_status)
        self.memory_diagnostics["epoch_feedback"] = {
            "enabled": bool(self.enable_epoch_feedback),
            "dynamic_loss_weights": {
                "hard_example_weight": float(self.dynamic_hard_example_weight),
                "kg_consistency_weight": float(self.dynamic_kg_consistency_weight),
                "path_alignment_weight": float(self.dynamic_path_alignment_weight),
                "archive_retention_weight": float(self.dynamic_archive_retention_weight),
                "memory_delta_floor_weight": float(self.dynamic_memory_delta_floor_weight),
            },
            "history": list(self.epoch_feedback_history),
        }
        self.memory_diagnostics["checkpoint_selection_mode"] = str(self.trainer_config.checkpoint_selection_mode)
        self.memory_diagnostics["memory_bank_build_policy"] = {
            "refresh_interval": int(self.trainer_config.memory_refresh_interval),
            "build_count": float(memory_bank_build_count),
            "per_epoch_rebuild_enabled": int(self.trainer_config.memory_refresh_interval) > 0,
        }

    def _predict_normalized(self, samples: Sequence[ForecastSample], use_memory: bool) -> List[List[float]]:
        predictions: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            batch_size = max(1, self.trainer_config.batch_size)
            for start in range(0, len(samples), batch_size):
                batch_samples = list(samples[start : start + batch_size])
                _, _, base_preds, fusion_preds, _, _ = self._forward_batch(batch_samples)
                batch_predictions = fusion_preds if use_memory else base_preds
                predictions.extend(batch_predictions.detach().cpu().tolist())
        return predictions

    def predict(self, samples: Sequence[ForecastSample], use_memory: bool = True) -> List[List[float]]:
        normalized_predictions = self._predict_normalized(samples, use_memory=use_memory)
        restored: List[List[float]] = []
        for sample, prediction in zip(samples, normalized_predictions):
            restored.append([value * sample.scale_value + sample.scale_center for value in prediction])
        return restored

    def _decode_multitask_predictions(
        self,
        gate_summaries: Sequence[Dict[str, object]],
    ) -> List[Dict[str, float]]:
        decoded: List[Dict[str, float]] = []
        for gate_summary in gate_summaries:
            row: Dict[str, float] = {}
            for task_name in self.multitask_task_names:
                tensor_value = gate_summary.get(f"_multitask_{task_name}_tensor")
                if tensor_value is None:
                    continue
                prediction_value = float(torch.as_tensor(tensor_value, dtype=torch.float32, device=self.device).item())
                if task_name in self.multitask_binary_tasks:
                    row[task_name] = float(torch.sigmoid(torch.tensor(prediction_value)).item())
                else:
                    row[task_name] = self._restore_aux_prediction(task_name, prediction_value)
            decoded.append(row)
        return decoded

    def predict_with_auxiliary(
        self,
        samples: Sequence[ForecastSample],
        use_memory: bool = True,
        preserve_mode: bool = False,
    ) -> Dict[str, object]:
        normalized_predictions: List[List[float]] = []
        auxiliary_predictions: List[Dict[str, float]] = []
        previous_mode = self.training
        if not preserve_mode:
            self.eval()
        with torch.no_grad():
            batch_size = max(1, self.trainer_config.batch_size)
            for start in range(0, len(samples), batch_size):
                batch_samples = list(samples[start : start + batch_size])
                _, _, base_preds, fusion_preds, gate_summaries, _ = self._forward_batch(batch_samples)
                batch_predictions = fusion_preds if use_memory else base_preds
                normalized_predictions.extend(batch_predictions.detach().cpu().tolist())
                auxiliary_predictions.extend(
                    self._decode_multitask_predictions(gate_summaries) if use_memory else [{} for _ in batch_samples]
                )

        restored_predictions: List[List[float]] = []
        for sample, prediction in zip(samples, normalized_predictions):
            restored_predictions.append([value * sample.scale_value + sample.scale_center for value in prediction])
        if preserve_mode:
            self.train(mode=previous_mode)
        return {
            "predictions": restored_predictions,
            "auxiliary_predictions": auxiliary_predictions,
        }

    def predict_with_uncertainty(
        self,
        samples: Sequence[ForecastSample],
        use_memory: bool = True,
        num_samples: int = 16,
        include_auxiliary: bool = True,
    ) -> Dict[str, object]:
        if not samples:
            return {"num_samples": 0, "samples": []}

        mc_samples = max(1, int(num_samples))
        was_training = self.training
        prediction_draws: List[torch.Tensor] = []
        auxiliary_draws: Dict[str, List[torch.Tensor]] = {task_name: [] for task_name in self.multitask_task_names}
        self.train()
        with torch.no_grad():
            for _ in range(mc_samples):
                batch_result = self.predict_with_auxiliary(
                    samples,
                    use_memory=use_memory,
                    preserve_mode=True,
                )
                prediction_draws.append(torch.tensor(batch_result["predictions"], dtype=torch.float32))
                if include_auxiliary and use_memory:
                    aux_rows = list(batch_result["auxiliary_predictions"])
                    for task_name in self.multitask_task_names:
                        auxiliary_draws[task_name].append(
                            torch.tensor(
                                [float(row.get(task_name, 0.0)) for row in aux_rows],
                                dtype=torch.float32,
                            )
                        )
        self.train(mode=was_training)

        prediction_tensor = torch.stack(prediction_draws, dim=0)
        prediction_mean = prediction_tensor.mean(dim=0)
        prediction_std = prediction_tensor.std(dim=0, unbiased=False)
        prediction_lower = torch.quantile(prediction_tensor, 0.025, dim=0)
        prediction_upper = torch.quantile(prediction_tensor, 0.975, dim=0)

        sample_rows: List[Dict[str, object]] = []
        for sample_index, sample in enumerate(samples):
            forecast_mean = [float(value) for value in prediction_mean[sample_index].tolist()]
            forecast_std = [float(value) for value in prediction_std[sample_index].tolist()]
            forecast_lower = [float(value) for value in prediction_lower[sample_index].tolist()]
            forecast_upper = [float(value) for value in prediction_upper[sample_index].tolist()]
            auxiliary_summary: Dict[str, Dict[str, float | str]] = {}
            if include_auxiliary and use_memory:
                for task_name in self.multitask_task_names:
                    if not auxiliary_draws[task_name]:
                        continue
                    task_tensor = torch.stack(auxiliary_draws[task_name], dim=0)
                    task_series = task_tensor[:, sample_index]
                    auxiliary_summary[task_name] = {
                        "kind": "binary_probability" if task_name in self.multitask_binary_tasks else "regression",
                        "mean": float(task_series.mean().item()),
                        "std": float(task_series.std(unbiased=False).item()),
                        "lower": float(torch.quantile(task_series, 0.025).item()),
                        "upper": float(torch.quantile(task_series, 0.975).item()),
                    }
            sample_rows.append(
                {
                    "forecast": {
                        "mean": forecast_mean,
                        "std": forecast_std,
                        "lower": forecast_lower,
                        "upper": forecast_upper,
                        "summary": {
                            "mean_std": float(sum(forecast_std) / max(1, len(forecast_std))),
                            "max_std": float(max(forecast_std) if forecast_std else 0.0),
                        },
                    },
                    "auxiliary_predictions": auxiliary_summary,
                }
            )
        return {
            "num_samples": mc_samples,
            "samples": sample_rows,
        }

    def evaluate_multitask(
        self,
        samples: Sequence[ForecastSample],
        use_memory: bool = True,
    ) -> Dict[str, object]:
        if not samples or not use_memory:
            return {
                "enabled": bool(use_memory),
                "task_metrics": {},
            }
        prediction_bundle = self.predict_with_auxiliary(samples, use_memory=use_memory)
        auxiliary_predictions = list(prediction_bundle.get("auxiliary_predictions", []))
        task_metrics: Dict[str, Dict[str, float]] = {}
        for task_name in self.multitask_task_names:
            paired: List[Tuple[float, float]] = []
            for sample, prediction_row in zip(samples, auxiliary_predictions):
                aux_targets = self._sample_aux_targets(sample)
                if task_name not in aux_targets or task_name not in prediction_row:
                    continue
                paired.append((float(aux_targets[task_name]), float(prediction_row[task_name])))
            if not paired:
                continue
            truth_values = [item[0] for item in paired]
            predicted_values = [item[1] for item in paired]
            if task_name in self.multitask_binary_tasks:
                binary_predictions = [1.0 if value >= 0.5 else 0.0 for value in predicted_values]
                task_metrics[task_name] = {
                    "count": float(len(paired)),
                    "brier": float(sum((pred - truth) ** 2 for truth, pred in paired) / len(paired)),
                    "accuracy": float(
                        sum(1.0 for truth, pred in zip(truth_values, binary_predictions) if float(truth) == float(pred))
                        / len(paired)
                    ),
                    "positive_rate": float(sum(truth_values) / len(paired)),
                    "mean_prediction": float(sum(predicted_values) / len(paired)),
                }
            else:
                absolute_errors = [abs(pred - truth) for truth, pred in paired]
                squared_errors = [(pred - truth) ** 2 for truth, pred in paired]
                task_metrics[task_name] = {
                    "count": float(len(paired)),
                    "mae": float(sum(absolute_errors) / len(absolute_errors)),
                    "rmse": float(math.sqrt(sum(squared_errors) / len(squared_errors))),
                    "truth_mean": float(sum(truth_values) / len(truth_values)),
                    "prediction_mean": float(sum(predicted_values) / len(predicted_values)),
                }
        return {
            "enabled": True,
            "task_metrics": task_metrics,
        }

    def analyze_uncertainty(
        self,
        samples: Sequence[ForecastSample],
        use_memory: bool = True,
        num_samples: int = 16,
    ) -> Dict[str, object]:
        bundle = self.predict_with_uncertainty(
            samples,
            use_memory=use_memory,
            num_samples=num_samples,
            include_auxiliary=use_memory,
        )
        sample_rows = list(bundle.get("samples", []))
        if not sample_rows:
            return {
                "enabled": False,
                "num_samples": int(num_samples),
                "forecast_mean_std": 0.0,
                "forecast_max_std_mean": 0.0,
                "auxiliary_mean_std": {},
            }
        forecast_mean_std = [
            float(dict(row.get("forecast", {})).get("summary", {}).get("mean_std", 0.0))
            for row in sample_rows
        ]
        forecast_max_std = [
            float(dict(row.get("forecast", {})).get("summary", {}).get("max_std", 0.0))
            for row in sample_rows
        ]
        auxiliary_mean_std: Dict[str, float] = {}
        for task_name in self.multitask_task_names:
            values = [
                float(dict(row.get("auxiliary_predictions", {})).get(task_name, {}).get("std", 0.0))
                for row in sample_rows
                if task_name in dict(row.get("auxiliary_predictions", {}))
            ]
            if values:
                auxiliary_mean_std[task_name] = float(sum(values) / len(values))
        per_sample_rows: List[Dict[str, object]] = []
        for index, row in enumerate(sample_rows):
            per_sample_rows.append({
                "sample_index": float(index),
                "forecast_mean_std": float(forecast_mean_std[index]) if index < len(forecast_mean_std) else 0.0,
                "forecast_max_std": float(forecast_max_std[index]) if index < len(forecast_max_std) else 0.0,
            })
        return {
            "enabled": True,
            "num_samples": int(bundle.get("num_samples", num_samples)),
            "forecast_mean_std": float(sum(forecast_mean_std) / len(forecast_mean_std)),
            "forecast_max_std_mean": float(sum(forecast_max_std) / len(forecast_max_std)),
            "auxiliary_mean_std": auxiliary_mean_std,
            "per_sample_rows": per_sample_rows,
        }

    @staticmethod
    def _jaccard_similarity(left: Sequence[str], right: Sequence[str]) -> float:
        left_set = {str(value) for value in left if str(value)}
        right_set = {str(value) for value in right if str(value)}
        union = left_set.union(right_set)
        if not union:
            return 1.0
        return float(len(left_set.intersection(right_set)) / len(union))

    def _counterfactual_required_actions_from_sample_kg(self, sample_kg_flags: Dict[str, object]) -> List[str]:
        required: List[str] = []
        if self._kg_flag(sample_kg_flags, "state_sepsis") > 0.0:
            required.append("treat_early_antimicrobial")
            required.append("exam_blood_culture")
        if (
            self._kg_flag(sample_kg_flags, "state_septic_shock") > 0.0
            or self._kg_flag(sample_kg_flags, "state_hypotension") > 0.0
        ):
            required.append("treat_vasopressor")
            required.append("monitor_map65")
        if self._kg_flag(sample_kg_flags, "state_high_lactate") > 0.0:
            required.append("exam_lactate")
            required.append("monitor_lactate_repeat")
        if (
            self._kg_flag(sample_kg_flags, "state_organ_dysfunction") > 0.0
            and self._kg_flag(sample_kg_flags, "state_septic_shock") > 0.0
        ):
            required.append("treat_respiratory_support")
        deduped: List[str] = []
        for name in required:
            if name not in deduped:
                deduped.append(name)
        return deduped

    def _counterfactual_action_set_from_flags(self, flags: Dict[str, object]) -> List[str]:
        action_names = [
            "treat_early_antimicrobial",
            "treat_vasopressor",
            "treat_respiratory_support",
            "exam_blood_culture",
            "exam_lactate",
            "monitor_map65",
            "monitor_lactate_repeat",
        ]
        return [name for name in action_names if self._kg_flag(flags, name) > 0.0]

    def _counterfactual_state_set_from_flags(self, flags: Dict[str, object]) -> List[str]:
        state_names = [
            "state_sepsis",
            "state_septic_shock",
            "state_organ_dysfunction",
            "state_hypotension",
            "state_high_lactate",
        ]
        return [name for name in state_names if self._kg_flag(flags, name) > 0.0]

    def _counterfactual_overlap_profile_from_sample(
        self,
        sample: ForecastSample,
        sample_kg_flags: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        kg_flags = dict(sample_kg_flags or sample.metadata.get("kg_flags", {}))
        state_vector = self._clinical_state_vector(sample, sample_kg_flags=kg_flags)
        recent_level = float(state_vector[6]) if len(state_vector) > 6 else 0.0
        recent_delta = float(state_vector[7]) if len(state_vector) > 7 else 0.0
        severity = (
            abs(recent_level)
            + 0.70 * self._kg_flag(kg_flags, "state_septic_shock")
            + 0.50 * self._kg_flag(kg_flags, "state_hypotension")
            + 0.30 * self._kg_flag(kg_flags, "state_high_lactate")
            + 0.20 * self._kg_flag(kg_flags, "state_organ_dysfunction")
        )
        return {
            "severity": float(severity),
            "trend": float(recent_delta),
            "state_set": self._counterfactual_state_set_from_flags(kg_flags),
            "required_action_set": self._counterfactual_required_actions_from_sample_kg(kg_flags),
        }

    def _counterfactual_overlap_profile_from_metadata(self, metadata: Dict[str, object]) -> Dict[str, object]:
        stored = metadata.get("overlap_profile", {})
        if isinstance(stored, dict) and stored:
            return {
                "severity": float(stored.get("severity", 0.0)),
                "trend": float(stored.get("trend", 0.0)),
                "state_set": [str(value) for value in stored.get("state_set", []) if str(value)],
                "action_set": [str(value) for value in stored.get("action_set", []) if str(value)],
            }
        donor_flags = dict(metadata.get("kg_flags", {}))
        severity = (
            0.70 * self._kg_flag(donor_flags, "state_septic_shock")
            + 0.50 * self._kg_flag(donor_flags, "state_hypotension")
            + 0.30 * self._kg_flag(donor_flags, "state_high_lactate")
            + 0.20 * self._kg_flag(donor_flags, "state_organ_dysfunction")
        )
        return {
            "severity": float(severity),
            "trend": float(metadata.get("recent_delta", 0.0)),
            "state_set": self._counterfactual_state_set_from_flags(donor_flags),
            "action_set": self._counterfactual_action_set_from_flags(donor_flags),
        }

    def _counterfactual_overlap_decision(
        self,
        sample_profile: Dict[str, object],
        donor_profile: Dict[str, object],
    ) -> Dict[str, object]:
        severity_gap = abs(float(sample_profile.get("severity", 0.0)) - float(donor_profile.get("severity", 0.0)))
        trend_gap = abs(float(sample_profile.get("trend", 0.0)) - float(donor_profile.get("trend", 0.0)))
        severity_score = self._clamp(
            1.0 - severity_gap / max(1e-6, float(self.counterfactual_overlap_severity_gap_max)),
            0.0,
            1.0,
        )
        trend_score = self._clamp(
            1.0 - trend_gap / max(1e-6, float(self.counterfactual_overlap_trend_gap_max)),
            0.0,
            1.0,
        )
        state_overlap = self._jaccard_similarity(
            sample_profile.get("state_set", []),
            donor_profile.get("state_set", []),
        )
        required_actions = [str(value) for value in sample_profile.get("required_action_set", []) if str(value)]
        donor_actions = {str(value) for value in donor_profile.get("action_set", []) if str(value)}
        if required_actions:
            action_overlap = float(sum(1 for action in required_actions if action in donor_actions) / len(required_actions))
        else:
            action_overlap = 1.0
        overlap_score = (
            0.35 * float(severity_score)
            + 0.20 * float(trend_score)
            + 0.25 * float(state_overlap)
            + 0.20 * float(action_overlap)
        )
        reasons: List[str] = []
        if float(severity_gap) > float(self.counterfactual_overlap_severity_gap_max):
            reasons.append("severity_gap")
        if float(trend_gap) > float(self.counterfactual_overlap_trend_gap_max):
            reasons.append("trend_gap")
        if float(state_overlap) < float(self.counterfactual_overlap_state_min):
            reasons.append("state_overlap")
        if float(action_overlap) < float(self.counterfactual_overlap_action_min):
            reasons.append("action_overlap")
        is_valid = True
        if self.counterfactual_overlap_filter_enabled:
            is_valid = len(reasons) == 0
        return {
            "overlap_score": float(overlap_score),
            "severity_gap": float(severity_gap),
            "trend_gap": float(trend_gap),
            "state_overlap": float(state_overlap),
            "action_overlap": float(action_overlap),
            "is_valid": bool(is_valid),
            "reasons": reasons,
        }

    def _counterfactual_exchangeability_score(self, metadata: Dict[str, object]) -> float:
        guideline = float(metadata.get("donor_guideline_compatibility", 0.0))
        state_match = float(metadata.get("donor_state_match", 0.0))
        hard_valid = float(metadata.get("donor_hard_filter_valid", 0.0))
        overlap_score = float(metadata.get("donor_overlap_score", 0.0))
        overlap_valid = float(metadata.get("donor_overlap_valid", 0.0))
        missing_penalty = float(metadata.get("donor_missing_care_penalty", 0.0))
        contraindication_penalty = float(metadata.get("donor_contraindication_penalty", 0.0))
        score = (
            0.28 * guideline
            + 0.24 * state_match
            + 0.18 * hard_valid
            + 0.18 * overlap_score
            + 0.12 * overlap_valid
            - 0.18 * missing_penalty
            - 0.22 * contraindication_penalty
        )
        return float(self._clamp(score, 0.0, 1.0))

    def _counterfactual_dedup_ranked_donors(
        self,
        ranked_donors: Sequence[Tuple[InterventionStoreEntry, Dict[str, object]]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[InterventionStoreEntry, Dict[str, object]]]:
        deduped: Dict[float, Tuple[InterventionStoreEntry, Dict[str, object]]] = {}
        for entry, metadata in ranked_donors:
            stay_id = float(entry.stay_id)
            current = deduped.get(stay_id)
            if current is None or float(metadata.get("donor_total_score", 0.0)) > float(current[1].get("donor_total_score", 0.0)):
                deduped[stay_id] = (entry, dict(metadata))
        ordered = sorted(
            deduped.values(),
            key=lambda item: float(item[1].get("donor_total_score", 0.0)),
            reverse=True,
        )
        if top_k is not None:
            ordered = ordered[: max(1, int(top_k))]
        return ordered

    def _counterfactual_neighbor_peer_affinity(
        self,
        anchor_metadata: Dict[str, object],
        peer_metadata: Dict[str, object],
    ) -> float:
        anchor_similarity = float(anchor_metadata.get("donor_similarity", 0.0))
        peer_similarity = float(peer_metadata.get("donor_similarity", 0.0))
        similarity_band = max(0.02, float(self.counterfactual_neighbor_similarity_band))
        similarity_gap = max(0.0, anchor_similarity - peer_similarity)
        closeness = 1.0 - min(1.0, similarity_gap / similarity_band)
        pattern_match = 1.0 if int(float(anchor_metadata.get("donor_pattern_label", -999))) == int(float(peer_metadata.get("donor_pattern_label", -998))) else 0.0
        trajectory_match = 1.0 if int(float(anchor_metadata.get("donor_trajectory_label", -999))) == int(float(peer_metadata.get("donor_trajectory_label", -998))) else 0.0
        experience_match = 1.0 if int(float(anchor_metadata.get("donor_experience_label", -999))) == int(float(peer_metadata.get("donor_experience_label", -998))) else 0.0
        peer_exchangeability = self._counterfactual_exchangeability_score(peer_metadata)
        return float(
            0.35 * closeness
            + 0.20 * pattern_match
            + 0.15 * trajectory_match
            + 0.10 * experience_match
            + 0.20 * peer_exchangeability
        )

    def _counterfactual_neighbor_summary(
        self,
        candidate_metadata: Dict[str, object],
        ranked_donors: Sequence[Tuple[InterventionStoreEntry, Dict[str, object]]],
    ) -> Dict[str, object]:
        deduped_donors = self._counterfactual_dedup_ranked_donors(ranked_donors)
        anchor_stay_id = float(candidate_metadata.get("stay_id", -1.0))
        anchor_rank = -1.0
        for rank, (entry, _) in enumerate(deduped_donors, start=1):
            if float(entry.stay_id) == anchor_stay_id:
                anchor_rank = float(rank)
                break
        anchor_similarity = float(candidate_metadata.get("donor_similarity", 0.0))
        scored_neighbors = []
        for entry, metadata in deduped_donors:
            if float(entry.stay_id) == anchor_stay_id:
                continue
            affinity = self._counterfactual_neighbor_peer_affinity(candidate_metadata, metadata)
            peer_similarity = float(metadata.get("donor_similarity", 0.0))
            similarity_gap = max(0.0, anchor_similarity - peer_similarity)
            same_pattern = int(float(candidate_metadata.get("donor_pattern_label", -999))) == int(float(metadata.get("donor_pattern_label", -998)))
            same_trajectory = int(float(candidate_metadata.get("donor_trajectory_label", -999))) == int(float(metadata.get("donor_trajectory_label", -998)))
            within_band = similarity_gap <= max(0.02, float(self.counterfactual_neighbor_similarity_band))
            if within_band or same_pattern or same_trajectory:
                scored_neighbors.append((float(affinity), entry, metadata))
        if not scored_neighbors:
            scored_neighbors = [
                (
                    float(self._counterfactual_neighbor_peer_affinity(candidate_metadata, metadata)),
                    entry,
                    metadata,
                )
                for entry, metadata in deduped_donors
                if float(entry.stay_id) != anchor_stay_id
            ]
        scored_neighbors.sort(key=lambda item: item[0], reverse=True)
        peer_neighbors = [
            (entry, metadata, affinity)
            for affinity, entry, metadata in scored_neighbors[: max(1, int(self.counterfactual_neighbor_top_k))]
        ]
        if not peer_neighbors:
            return {
                "neighbor_count": 0.0,
                "consistency": 0.0,
                "self_exchangeability": 0.0,
                "exchangeability_mean": 0.0,
                "action_alignment_mean": 0.0,
                "state_alignment_mean": 0.0,
                "hard_pass_rate": 0.0,
                "overlap_valid_rate": 0.0,
                "peer_similarity_mean": 0.0,
                "peer_affinity_mean": 0.0,
                "anchor_rank": float(anchor_rank),
            }

        candidate_flags = dict(candidate_metadata.get("kg_flags", {}))
        candidate_action_set = self._counterfactual_action_set_from_flags(candidate_flags)
        candidate_state_set = self._counterfactual_state_set_from_flags(candidate_flags)
        self_exchangeability = self._counterfactual_exchangeability_score(candidate_metadata)

        weighted_exchangeability = 0.0
        weighted_action_alignment = 0.0
        weighted_state_alignment = 0.0
        weighted_hard_rate = 0.0
        weighted_overlap_rate = 0.0
        weighted_similarity = 0.0
        weighted_affinity = 0.0
        total_weight = 0.0

        for entry, metadata, affinity in peer_neighbors:
            donor_weight = max(
                0.05,
                0.65 * float(metadata.get("donor_similarity", 0.0)) + 0.35 * float(affinity),
            )
            donor_flags = dict(metadata.get("kg_flags", {}))
            donor_action_set = self._counterfactual_action_set_from_flags(donor_flags)
            donor_state_set = self._counterfactual_state_set_from_flags(donor_flags)
            exchangeability = self._counterfactual_exchangeability_score(metadata)
            action_alignment = self._jaccard_similarity(candidate_action_set, donor_action_set)
            state_alignment = self._jaccard_similarity(candidate_state_set, donor_state_set)
            weighted_exchangeability += donor_weight * exchangeability
            weighted_action_alignment += donor_weight * float(action_alignment)
            weighted_state_alignment += donor_weight * float(state_alignment)
            weighted_hard_rate += donor_weight * float(metadata.get("donor_hard_filter_valid", 0.0))
            weighted_overlap_rate += donor_weight * float(metadata.get("donor_overlap_valid", 0.0))
            weighted_similarity += donor_weight * float(metadata.get("donor_similarity", 0.0))
            weighted_affinity += donor_weight * float(affinity)
            total_weight += donor_weight

        if total_weight <= 0.0:
            total_weight = float(len(peer_neighbors))
        exchangeability_mean = weighted_exchangeability / total_weight
        action_alignment_mean = weighted_action_alignment / total_weight
        state_alignment_mean = weighted_state_alignment / total_weight
        hard_pass_rate = weighted_hard_rate / total_weight
        overlap_valid_rate = weighted_overlap_rate / total_weight
        peer_similarity_mean = weighted_similarity / total_weight
        peer_affinity_mean = weighted_affinity / total_weight
        peer_consistency = (
            0.34 * exchangeability_mean
            + 0.24 * action_alignment_mean
            + 0.16 * state_alignment_mean
            + 0.13 * hard_pass_rate
            + 0.13 * overlap_valid_rate
        )
        self_weight = float(self._clamp(float(self.counterfactual_neighbor_self_weight), 0.2, 0.8))
        coverage_factor = min(1.0, float(len(peer_neighbors)) / max(2.0, float(self.counterfactual_neighbor_top_k)))
        consistency = self_weight * self_exchangeability + (1.0 - self_weight) * peer_consistency
        consistency *= 0.85 + 0.15 * coverage_factor
        return {
            "neighbor_count": float(len(peer_neighbors)),
            "consistency": float(self._clamp(consistency, 0.0, 1.0)),
            "self_exchangeability": float(self_exchangeability),
            "exchangeability_mean": float(exchangeability_mean),
            "action_alignment_mean": float(action_alignment_mean),
            "state_alignment_mean": float(state_alignment_mean),
            "hard_pass_rate": float(hard_pass_rate),
            "overlap_valid_rate": float(overlap_valid_rate),
            "peer_similarity_mean": float(peer_similarity_mean),
            "peer_affinity_mean": float(peer_affinity_mean),
            "anchor_rank": float(anchor_rank),
        }

    def _counterfactual_neighbor_adjustment(self, neighborhood_summary: Dict[str, object]) -> Dict[str, float]:
        consistency = float(neighborhood_summary.get("consistency", 0.0))
        self_exchangeability = float(neighborhood_summary.get("self_exchangeability", 0.0))
        hard_pass_rate = float(neighborhood_summary.get("hard_pass_rate", 0.0))
        overlap_valid_rate = float(neighborhood_summary.get("overlap_valid_rate", 0.0))
        neighbor_count = float(neighborhood_summary.get("neighbor_count", 0.0))
        bonus = float(self.counterfactual_neighbor_weight * max(0.0, consistency - 0.45))
        penalty_scale = (
            1.0
            + 0.50 * max(0.0, 0.50 - hard_pass_rate)
            + 0.50 * max(0.0, 0.50 - overlap_valid_rate)
            + 0.25 * max(0.0, 2.0 - neighbor_count)
            + 0.40 * max(0.0, 0.35 - self_exchangeability)
        )
        if hard_pass_rate < 0.05 and overlap_valid_rate < 0.05 and consistency < float(self.counterfactual_neighbor_min_consistency):
            penalty_scale += 0.55
        penalty = float(
            self.counterfactual_neighbor_penalty_weight
            * max(0.0, float(self.counterfactual_neighbor_min_consistency) - consistency)
            * penalty_scale
        )
        return {
            "bonus": float(bonus),
            "penalty": float(penalty),
        }

    def _counterfactual_reranker_feature_names(self) -> List[str]:
        return [
            "donor_similarity",
            "donor_kg_similarity",
            "donor_guideline_compatibility",
            "donor_state_match",
            "donor_missing_care_penalty",
            "donor_contraindication_penalty",
            "donor_hard_filter_valid",
            "donor_overlap_score",
            "donor_overlap_valid",
            "donor_transition_score",
            "donor_action_change_score",
            "donor_total_score",
            "pattern_match",
            "trajectory_match",
            "experience_match",
        ]

    def _counterfactual_reranker_feature_map(
        self,
        sample: ForecastSample,
        metadata: Dict[str, object],
    ) -> Dict[str, float]:
        query_experience = int(float(sample.metadata.get("experience_label", -1)))
        query_pattern = int(float(sample.metadata.get("pattern_label", -1)))
        query_trajectory = int(float(sample.metadata.get("trajectory_label", -1)))
        donor_experience = int(float(metadata.get("donor_experience_label", -999)))
        donor_pattern = int(float(metadata.get("donor_pattern_label", -999)))
        donor_trajectory = int(float(metadata.get("donor_trajectory_label", -999)))
        return {
            "donor_similarity": float(metadata.get("donor_similarity", 0.0)),
            "donor_kg_similarity": float(metadata.get("donor_kg_similarity", 0.0)),
            "donor_guideline_compatibility": float(metadata.get("donor_guideline_compatibility", 0.0)),
            "donor_state_match": float(metadata.get("donor_state_match", 0.0)),
            "donor_missing_care_penalty": float(metadata.get("donor_missing_care_penalty", 0.0)),
            "donor_contraindication_penalty": float(metadata.get("donor_contraindication_penalty", 0.0)),
            "donor_hard_filter_valid": float(metadata.get("donor_hard_filter_valid", 0.0)),
            "donor_overlap_score": float(metadata.get("donor_overlap_score", 0.0)),
            "donor_overlap_valid": float(metadata.get("donor_overlap_valid", 0.0)),
            "donor_transition_score": float(metadata.get("donor_transition_score", 0.0)),
            "donor_action_change_score": float(metadata.get("donor_action_change_score", 0.0)),
            "donor_total_score": float(metadata.get("donor_total_score", 0.0)),
            "pattern_match": 1.0 if donor_pattern == query_pattern else 0.0,
            "trajectory_match": 1.0 if donor_trajectory == query_trajectory else 0.0,
            "experience_match": 1.0 if donor_experience == query_experience else 0.0,
        }

    def _counterfactual_reranker_feature_vector(
        self,
        sample: ForecastSample,
        metadata: Dict[str, object],
    ) -> List[float]:
        feature_map = self._counterfactual_reranker_feature_map(sample, metadata)
        return [float(feature_map[name]) for name in self._counterfactual_reranker_feature_names()]

    def _counterfactual_reranker_predict(
        self,
        sample: ForecastSample,
        metadata: Dict[str, object],
    ) -> float:
        if str(self.counterfactual_reranker_mode) != "learned_linear":
            return 0.0
        state = dict(self.counterfactual_reranker_state or {})
        feature_names = list(state.get("feature_names", []))
        weights = list(state.get("weights", []))
        if not feature_names or not weights or len(feature_names) != len(weights):
            return 0.0
        feature_map = self._counterfactual_reranker_feature_map(sample, metadata)
        score = float(state.get("bias", 0.0))
        for name, weight in zip(feature_names, weights):
            score += float(weight) * float(feature_map.get(str(name), 0.0))
        return float(max(-1.5, min(1.5, score)))

    def fit_counterfactual_reranker(
        self,
        samples: Sequence[ForecastSample],
    ) -> Dict[str, object]:
        if str(self.counterfactual_reranker_mode) != "learned_linear":
            self.counterfactual_reranker_state = {
                "status": "disabled",
                "mode": str(self.counterfactual_reranker_mode),
                "fit_examples": 0.0,
            }
            self.training_summary["counterfactual_reranker"] = dict(self.counterfactual_reranker_state)
            self.memory_diagnostics["counterfactual_reranker"] = dict(self.counterfactual_reranker_state)
            return dict(self.counterfactual_reranker_state)

        subset = list(samples[: max(0, int(self.counterfactual_reranker_max_samples))]) if self.counterfactual_reranker_max_samples > 0 else list(samples)
        feature_rows: List[List[float]] = []
        targets: List[float] = []
        previous_mode = str(self.counterfactual_reranker_mode)
        self.counterfactual_reranker_mode = "rule_only"
        self.eval()
        try:
            with torch.no_grad():
                batch_size = max(1, self.trainer_config.batch_size)
                for start in range(0, len(subset), batch_size):
                    batch_samples = list(subset[start : start + batch_size])
                    encodings, manager_results, _, factual_predictions, _, _ = self._forward_batch(batch_samples)
                    factual_rows = factual_predictions.detach().cpu().tolist()
                    donor_eval_samples: List[ForecastSample] = []
                    donor_records: List[Tuple[ForecastSample, List[float], Dict[str, object]]] = []
                    for sample, factual_prediction, encoding, manager_result in zip(
                        batch_samples,
                        factual_rows,
                        encodings,
                        manager_results,
                    ):
                        ranked_donors = self._counterfactual_ranked_donors_from_manager_result(sample, encoding, manager_result)
                        for donor_entry, donor_metadata in ranked_donors[: max(1, int(self.counterfactual_reranker_train_top_k))]:
                            donor_eval_samples.append(
                                replace(
                                    sample,
                                    intervention_static=list(donor_entry.intervention_static),
                                    intervention_sequence=[list(step) for step in donor_entry.intervention_sequence],
                                )
                            )
                            donor_records.append((sample, list(factual_prediction), dict(donor_metadata)))
                    if not donor_eval_samples:
                        continue
                    _, _, _, donor_predictions, _, _ = self._forward_batch(donor_eval_samples)
                    for donor_sample, (sample, factual_prediction, donor_metadata), donor_prediction in zip(
                        donor_eval_samples,
                        donor_records,
                        donor_predictions.detach().cpu().tolist(),
                    ):
                        restored_factual = [
                            float(value) * float(sample.scale_value) + float(sample.scale_center)
                            for value in factual_prediction
                        ]
                        restored_donor = [
                            float(value) * float(donor_sample.scale_value) + float(donor_sample.scale_center)
                            for value in donor_prediction
                        ]
                        predicted_delta = float(
                            sum(restored_factual) / max(1, len(restored_factual))
                            - sum(restored_donor) / max(1, len(restored_donor))
                        )
                        feature_rows.append(self._counterfactual_reranker_feature_vector(sample, donor_metadata))
                        targets.append(predicted_delta)
        finally:
            self.counterfactual_reranker_mode = previous_mode

        fit_examples = len(feature_rows)
        if fit_examples < max(1, int(self.counterfactual_reranker_min_examples)):
            self.counterfactual_reranker_state = {
                "status": "insufficient_examples",
                "mode": previous_mode,
                "fit_examples": float(fit_examples),
            }
            self.training_summary["counterfactual_reranker"] = dict(self.counterfactual_reranker_state)
            self.memory_diagnostics["counterfactual_reranker"] = dict(self.counterfactual_reranker_state)
            return dict(self.counterfactual_reranker_state)

        feature_tensor = torch.tensor(feature_rows, dtype=torch.float32)
        target_tensor = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
        bias_column = torch.ones((feature_tensor.shape[0], 1), dtype=torch.float32)
        design = torch.cat([feature_tensor, bias_column], dim=1)
        reg_eye = torch.eye(design.shape[1], dtype=torch.float32)
        reg_eye[-1, -1] = 0.0
        ridge_lambda = max(0.0, float(self.counterfactual_reranker_ridge_l2))
        lhs = design.transpose(0, 1) @ design + ridge_lambda * reg_eye
        rhs = design.transpose(0, 1) @ target_tensor
        try:
            solution = torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            solution = torch.linalg.pinv(lhs) @ rhs
        weights = solution[:-1, 0]
        bias = float(solution[-1, 0].item())
        predictions = feature_tensor @ weights + bias
        residuals = predictions - target_tensor[:, 0]
        rmse = float(torch.sqrt(torch.mean(residuals ** 2)).item())
        target_mean = float(target_tensor[:, 0].mean().item())
        target_var = float(torch.mean((target_tensor[:, 0] - target_mean) ** 2).item())
        r2 = 0.0 if target_var <= 1e-8 else float(max(-1.0, min(1.0, 1.0 - float(torch.mean(residuals ** 2).item()) / target_var)))
        self.counterfactual_reranker_state = {
            "status": "fitted",
            "mode": previous_mode,
            "fit_examples": float(fit_examples),
            "feature_names": self._counterfactual_reranker_feature_names(),
            "weights": [float(value) for value in weights.detach().cpu().tolist()],
            "bias": float(bias),
            "train_rmse": float(rmse),
            "train_r2": float(r2),
            "target_mean": float(target_mean),
            "ridge_l2": float(ridge_lambda),
        }
        self.training_summary["counterfactual_reranker"] = dict(self.counterfactual_reranker_state)
        self.memory_diagnostics["counterfactual_reranker"] = dict(self.counterfactual_reranker_state)
        return dict(self.counterfactual_reranker_state)

    def _counterfactual_candidate_indices(
        self,
        sample: ForecastSample,
        manager_result: ManagerReadResult,
    ) -> List[int]:
        candidate_labels: List[int] = []
        top_label = int(manager_result.component_results["experience"].top_label)
        if top_label >= 0:
            candidate_labels.append(top_label)
        for label in manager_result.component_results["experience"].matched_labels:
            label_int = int(label)
            if label_int >= 0 and label_int not in candidate_labels:
                candidate_labels.append(label_int)
            if len(candidate_labels) >= max(1, int(self.counterfactual_label_top_k)):
                break
        if int(sample.experience_label) not in candidate_labels:
            candidate_labels.append(int(sample.experience_label))

        ordered_indices: List[int] = []
        seen_indices = set()

        def _extend(indices: Sequence[int]) -> None:
            for index in indices:
                index_int = int(index)
                if index_int in seen_indices:
                    continue
                seen_indices.add(index_int)
                ordered_indices.append(index_int)

        query_metadata = dict(sample.metadata)
        subtype_indices: List[int] = []
        subtype_seen = set()

        def _append_subtype(indices: Sequence[int]) -> None:
            for index in indices:
                index_int = int(index)
                if index_int in subtype_seen:
                    continue
                subtype_seen.add(index_int)
                subtype_indices.append(index_int)

        for label in candidate_labels:
            _append_subtype(self.intervention_store_by_label.get(int(label), []))
        if self.counterfactual_pool_include_pattern:
            _append_subtype(self.intervention_store_by_pattern.get(int(sample.pattern_label), []))
        if self.counterfactual_pool_include_trajectory:
            _append_subtype(self.intervention_store_by_trajectory.get(int(sample.trajectory_label), []))

        hospital_indices = []
        if self.counterfactual_pool_include_hospital:
            hospital_key = self._counterfactual_pool_key(query_metadata.get("hospitalid"))
            hospital_indices = list(self.intervention_store_by_hospital.get(hospital_key, [])) if hospital_key else []
        unit_indices = []
        if self.counterfactual_pool_include_unit_type:
            unit_key = self._counterfactual_pool_key(query_metadata.get("unittype"))
            unit_indices = list(self.intervention_store_by_unit_type.get(unit_key, [])) if unit_key else []
        infection_indices = []
        if self.counterfactual_pool_include_infection_anchor:
            infection_key = self._counterfactual_pool_key(query_metadata.get("infection_anchor_type"))
            infection_indices = list(self.intervention_store_by_infection_anchor.get(infection_key, [])) if infection_key else []

        subtype_set = set(subtype_indices)

        def _intersect(primary: Sequence[int], secondary: set[int]) -> List[int]:
            return [int(index) for index in primary if int(index) in secondary]

        pool_mode = str(self.counterfactual_pool_mode).strip().lower()
        local_min_candidates = max(1, int(self.counterfactual_pool_local_min_candidates))

        if pool_mode in {"same_hospital", "hospital"}:
            _extend(_intersect(subtype_indices, set(hospital_indices)))
            if not ordered_indices:
                _extend(hospital_indices)
            if not ordered_indices:
                _extend(subtype_indices)
        elif pool_mode in {"same_unit", "unit"}:
            _extend(_intersect(subtype_indices, set(unit_indices)))
            if not ordered_indices:
                _extend(unit_indices)
            if not ordered_indices:
                _extend(subtype_indices)
        elif pool_mode in {"adaptive", "stratified"}:
            _extend(_intersect(hospital_indices, subtype_set))
            _extend(_intersect(unit_indices, subtype_set))
            _extend(_intersect(infection_indices, subtype_set))
            if not ordered_indices:
                _extend(hospital_indices)
            if not ordered_indices:
                _extend(unit_indices)
            if not ordered_indices:
                _extend(infection_indices)
            if not ordered_indices:
                _extend(subtype_indices)
        else:
            _extend(subtype_indices)

        min_candidates = max(1, int(self.counterfactual_pool_min_candidates))
        allow_global_backfill = bool(self.counterfactual_pool_enable_global_backfill) and (
            pool_mode == "global" or not ordered_indices
        )
        if allow_global_backfill and len(ordered_indices) < min_candidates:
            _extend(range(len(self.intervention_store_entries)))

        max_candidates = max(1, int(self.counterfactual_pool_global_limit))
        if len(ordered_indices) > max_candidates:
            ordered_indices = ordered_indices[:max_candidates]
        return ordered_indices

    def _counterfactual_ranked_donors_from_manager_result(
        self,
        sample: ForecastSample,
        encoding: TorchManifoldEncodingOutput,
        manager_result: ManagerReadResult,
    ) -> List[Tuple[InterventionStoreEntry, Dict[str, object]]]:
        sample_stay_id = sample.metadata.get("stay_id")
        query = F.normalize(encoding.input_embedding.detach(), dim=0)
        sample_kg_features = list(sample.kg_features or sample.metadata.get("kg_features", []) or [])
        sample_kg_flags = dict(sample.metadata.get("kg_flags", {}))
        sample_overlap_profile = self._counterfactual_overlap_profile_from_sample(sample, sample_kg_flags=sample_kg_flags)
        candidate_indices = self._counterfactual_candidate_indices(sample, manager_result)
        if not candidate_indices:
            return []

        filtered_indices: List[int] = []
        for candidate_index in candidate_indices:
            entry = self.intervention_store_entries[candidate_index]
            if entry.stay_id == float(sample_stay_id):
                continue
            filtered_indices.append(int(candidate_index))
        if not filtered_indices:
            return []

        candidate_embeddings = torch.tensor(
            [self.intervention_store_entries[index].patient_embedding for index in filtered_indices],
            dtype=torch.float32,
            device=self.device,
        )
        similarities = torch.matmul(F.normalize(candidate_embeddings, dim=-1), query)
        prefilter_top_k = max(1, int(self.counterfactual_pool_prefilter_top_k))
        if similarities.numel() > prefilter_top_k:
            top_positions = torch.topk(similarities, k=prefilter_top_k, largest=True).indices.tolist()
            filtered_indices = [filtered_indices[int(position)] for position in top_positions]
            candidate_embeddings = torch.tensor(
                [self.intervention_store_entries[index].patient_embedding for index in filtered_indices],
                dtype=torch.float32,
                device=self.device,
            )
            similarities = torch.matmul(F.normalize(candidate_embeddings, dim=-1), query)

        ranked_candidates: List[Tuple[InterventionStoreEntry, Dict[str, object]]] = []
        scored_positions = []
        for position, candidate_index in enumerate(filtered_indices):
            entry = self.intervention_store_entries[candidate_index]
            contextual_metadata = dict(entry.metadata)
            donor_flags = self._merge_candidate_flags(
                dict(contextual_metadata.get("kg_flags", {})),
                entry.intervention_static,
                entry.intervention_sequence,
            )
            candidate_plan_flags = self._build_plan_evaluation_flags(
                sample_kg_flags=sample_kg_flags,
                reference_flags=donor_flags,
                intervention_static=entry.intervention_static,
                intervention_sequence=entry.intervention_sequence,
            )
            action_change = self._action_change_score(
                sample=sample,
                candidate_flags=candidate_plan_flags,
                intervention_static=entry.intervention_static,
                intervention_sequence=entry.intervention_sequence,
            )
            kg_similarity = cosine_similarity(sample_kg_features, entry.kg_features)
            embedding_similarity = float(similarities[position].item())
            transition_readout = self._empty_transition_readout()
            if self.enable_transition_memory and self.enable_transition_donor_path:
                transition_readout = self._retrieve_transition_readout_with_baseline(
                    sample=sample,
                    intervention_static=entry.intervention_static,
                    intervention_sequence=entry.intervention_sequence,
                    candidate_labels=self._transition_candidate_labels(
                        sample,
                        manager_result=manager_result,
                        preferred_label=entry.experience_label,
                    ),
                    base_flags=donor_flags,
                )
            score_components = self._compute_counterfactual_donor_score(
                sample_kg_flags=sample_kg_flags,
                donor_flags=candidate_plan_flags,
                embedding_similarity=embedding_similarity,
                kg_similarity=kg_similarity,
                donor_guideline_score=float(entry.kg_guideline_score),
                transition_utility=float(transition_readout["expected_utility"]),
                transition_confidence=float(transition_readout["confidence"]),
                transition_advantage=float(transition_readout["expected_advantage"]),
                transition_action_gain=float(transition_readout["expected_action_gain"]),
                transition_improvement_rate=float(transition_readout["improvement_rate"]),
                transition_improvement_gain=float(transition_readout["improvement_gain"]),
                transition_support_strength=float(transition_readout["support_strength"]),
                action_change_score=float(action_change["action_change_score"]),
            )
            hard_filter = self._counterfactual_hard_filter_decision(sample_kg_flags, candidate_plan_flags)
            donor_overlap_profile = self._counterfactual_overlap_profile_from_metadata(entry.metadata)
            donor_overlap_profile["action_set"] = self._counterfactual_action_set_from_flags(candidate_plan_flags)
            overlap_decision = self._counterfactual_overlap_decision(sample_overlap_profile, donor_overlap_profile)
            overlap_adjustment = float(self.counterfactual_overlap_weight) * (float(overlap_decision["overlap_score"]) - 0.5)
            pool_match = self._counterfactual_pool_match_components(sample, contextual_metadata)
            provisional_metadata = {
                "donor_similarity": float(embedding_similarity),
                "donor_kg_similarity": float(score_components["kg_similarity"]),
                "donor_guideline_compatibility": float(score_components["guideline_compatibility"]),
                "donor_state_match": float(score_components["state_match"]),
                "donor_missing_care_penalty": float(score_components["missing_care_penalty"]),
                "donor_contraindication_penalty": float(score_components["contraindication_penalty"]),
                "donor_total_score": float(score_components["final_score"] + overlap_adjustment),
                "donor_experience_label": float(entry.experience_label),
                "donor_pattern_label": float(entry.pattern_label),
                "donor_trajectory_label": float(entry.trajectory_label),
                "donor_hard_filter_valid": 1.0 if hard_filter["is_valid"] else 0.0,
                "donor_overlap_score": float(overlap_decision["overlap_score"]),
                "donor_overlap_valid": 1.0 if overlap_decision["is_valid"] else 0.0,
                "donor_pool_match_score": float(pool_match["score"]),
                "donor_pool_match_reward": float(pool_match["reward"]),
                "donor_pool_tags": list(pool_match["tags"]),
                "donor_transition_score": float(score_components["transition_score"]),
                "donor_action_change_score": float(score_components["action_change_score"]),
                "donor_pre_neighbor_total_score": 0.0,
            }
            learned_reranker_score = self._counterfactual_reranker_predict(sample, provisional_metadata)
            reranker_adjustment = float(self.counterfactual_reranker_blend_weight) * float(learned_reranker_score)
            final_score = (
                float(score_components["final_score"])
                + float(overlap_adjustment)
                + float(reranker_adjustment)
                + float(pool_match["reward"])
            )
            provisional_metadata["donor_pre_neighbor_total_score"] = float(final_score)
            provisional_metadata["donor_total_score"] = float(final_score)
            scored_positions.append(
                (
                    float(final_score),
                    position,
                    entry,
                    embedding_similarity,
                    score_components,
                    donor_flags,
                    hard_filter,
                    overlap_decision,
                    overlap_adjustment,
                    learned_reranker_score,
                    reranker_adjustment,
                    dict(
                        provisional_metadata,
                        kg_flags=dict(donor_flags),
                    ),
                )
            )
        provisional_ranked_donors = [
            (entry, dict(provisional_metadata))
            for (
                _final_score,
                _position,
                entry,
                _embedding_similarity,
                _score_components,
                _donor_flags,
                _hard_filter,
                _overlap_decision,
                _overlap_adjustment,
                _learned_reranker_score,
                _reranker_adjustment,
                provisional_metadata,
            ) in scored_positions
        ]
        adjusted_positions = []
        for (
            raw_final_score,
            position,
            entry,
            embedding_similarity,
            score_components,
            donor_flags,
            hard_filter,
            overlap_decision,
            overlap_adjustment,
            learned_reranker_score,
            reranker_adjustment,
            provisional_metadata,
        ) in scored_positions:
            neighborhood_summary = self._counterfactual_neighbor_summary(
                candidate_metadata=provisional_metadata,
                ranked_donors=provisional_ranked_donors,
            )
            neighbor_adjustment = self._counterfactual_neighbor_adjustment(neighborhood_summary)
            donor_neighbor_bonus = float(neighbor_adjustment["bonus"])
            donor_neighbor_penalty = float(neighbor_adjustment["penalty"])
            adjusted_final_score = float(raw_final_score + donor_neighbor_bonus - donor_neighbor_penalty)
            adjusted_positions.append(
                (
                    float(adjusted_final_score),
                    position,
                    entry,
                    embedding_similarity,
                    score_components,
                    donor_flags,
                    hard_filter,
                    overlap_decision,
                    overlap_adjustment,
                    learned_reranker_score,
                    reranker_adjustment,
                    provisional_metadata,
                    neighborhood_summary,
                    donor_neighbor_bonus,
                    donor_neighbor_penalty,
                    raw_final_score,
                )
            )
        adjusted_positions.sort(key=lambda item: item[0], reverse=True)
        hard_feasible = [
            item for item in adjusted_positions if bool(item[6]["is_valid"]) or not self.counterfactual_hard_filter_enabled
        ]
        overlap_feasible = [
            item for item in hard_feasible if bool(item[7]["is_valid"]) or not self.counterfactual_overlap_filter_enabled
        ]
        neighbor_feasible = [
            item
            for item in overlap_feasible
            if (
                float(item[12].get("consistency", 0.0)) >= max(0.40, 0.85 * float(self.counterfactual_neighbor_min_consistency))
                or (
                    float(item[12].get("self_exchangeability", 0.0)) >= 0.55
                    and (bool(item[6]["is_valid"]) or bool(item[7]["is_valid"]))
                )
            )
        ]
        if neighbor_feasible:
            ranked_positions = neighbor_feasible
            overlap_filter_fallback = False
        elif overlap_feasible:
            ranked_positions = overlap_feasible
            overlap_filter_fallback = False
        elif hard_feasible and self.counterfactual_overlap_fallback_enabled:
            ranked_positions = hard_feasible
            overlap_filter_fallback = bool(self.counterfactual_overlap_filter_enabled)
        elif hard_feasible:
            ranked_positions = hard_feasible
            overlap_filter_fallback = bool(self.counterfactual_overlap_filter_enabled)
        else:
            ranked_positions = adjusted_positions
            overlap_filter_fallback = bool(self.counterfactual_overlap_filter_enabled and bool(adjusted_positions))
        hard_filter_fallback = bool(self.counterfactual_hard_filter_enabled and not hard_feasible and adjusted_positions)
        for (
            final_score,
            position,
            entry,
            embedding_similarity,
            score_components,
            donor_flags,
            hard_filter,
            overlap_decision,
            overlap_adjustment,
            learned_reranker_score,
            reranker_adjustment,
            provisional_metadata,
            neighborhood_summary,
            donor_neighbor_bonus,
            donor_neighbor_penalty,
            raw_final_score,
        ) in ranked_positions:
            metadata = dict(entry.metadata)
            metadata.update(
                {
                    "kg_flags": donor_flags,
                    "donor_similarity": float(embedding_similarity),
                    "donor_kg_similarity": float(score_components["kg_similarity"]),
                    "donor_guideline_compatibility": float(score_components["guideline_compatibility"]),
                    "donor_state_match": float(score_components["state_match"]),
                    "donor_missing_care_penalty": float(score_components["missing_care_penalty"]),
                    "donor_contraindication_penalty": float(score_components["contraindication_penalty"]),
                    "donor_total_score": float(final_score),
                    "donor_score_mode": self.counterfactual_donor_score_mode,
                    "donor_source": "raw_intervention_store",
                    "donor_experience_id": str(entry.experience_id),
                    "donor_intervention_plan_code": dict(entry.intervention_plan_code),
                    "donor_experience_label": float(entry.experience_label),
                    "donor_pattern_label": float(entry.pattern_label),
                    "donor_trajectory_label": float(entry.trajectory_label),
                    "donor_hard_filter_valid": 1.0 if hard_filter["is_valid"] else 0.0,
                    "donor_hard_filter_reason": "|".join(str(reason) for reason in hard_filter["reasons"]),
                    "donor_hard_filter_fallback": 1.0 if hard_filter_fallback else 0.0,
                    "donor_overlap_score": float(overlap_decision["overlap_score"]),
                    "donor_overlap_valid": 1.0 if overlap_decision["is_valid"] else 0.0,
                    "donor_overlap_reason": "|".join(str(reason) for reason in overlap_decision["reasons"]),
                    "donor_overlap_fallback": 1.0 if overlap_filter_fallback else 0.0,
                    "donor_overlap_severity_gap": float(overlap_decision["severity_gap"]),
                    "donor_overlap_trend_gap": float(overlap_decision["trend_gap"]),
                    "donor_overlap_state_score": float(overlap_decision["state_overlap"]),
                    "donor_overlap_action_score": float(overlap_decision["action_overlap"]),
                    "donor_overlap_adjustment": float(overlap_adjustment),
                    "donor_pre_neighbor_total_score": float(raw_final_score),
                    "donor_neighbor_consistency": float(neighborhood_summary.get("consistency", 0.0)),
                    "donor_neighbor_self_exchangeability": float(neighborhood_summary.get("self_exchangeability", 0.0)),
                    "donor_neighbor_exchangeability_mean": float(neighborhood_summary.get("exchangeability_mean", 0.0)),
                    "donor_neighbor_action_alignment_mean": float(neighborhood_summary.get("action_alignment_mean", 0.0)),
                    "donor_neighbor_state_alignment_mean": float(neighborhood_summary.get("state_alignment_mean", 0.0)),
                    "donor_neighbor_hard_pass_rate": float(neighborhood_summary.get("hard_pass_rate", 0.0)),
                    "donor_neighbor_overlap_valid_rate": float(neighborhood_summary.get("overlap_valid_rate", 0.0)),
                    "donor_neighbor_peer_similarity_mean": float(neighborhood_summary.get("peer_similarity_mean", 0.0)),
                    "donor_neighbor_peer_affinity_mean": float(neighborhood_summary.get("peer_affinity_mean", 0.0)),
                    "donor_neighbor_anchor_rank": float(neighborhood_summary.get("anchor_rank", -1.0)),
                    "donor_neighbor_bonus": float(donor_neighbor_bonus),
                    "donor_neighbor_penalty": float(donor_neighbor_penalty),
                    "donor_learned_reranker_score": float(learned_reranker_score),
                    "donor_reranker_adjustment": float(reranker_adjustment),
                    "donor_reranker_mode": str(self.counterfactual_reranker_mode),
                    "donor_pool_match_score": float(provisional_metadata.get("donor_pool_match_score", 0.0)),
                    "donor_pool_match_reward": float(provisional_metadata.get("donor_pool_match_reward", 0.0)),
                    "donor_pool_tags": [str(value) for value in provisional_metadata.get("donor_pool_tags", [])],
                    "donor_transition_utility": float(score_components["transition_utility"]),
                    "donor_transition_confidence": float(score_components["transition_confidence"]),
                    "donor_transition_advantage": float(score_components["transition_advantage"]),
                    "donor_transition_action_gain": float(score_components["transition_action_gain"]),
                    "donor_transition_improvement_rate": float(score_components["transition_improvement_rate"]),
                    "donor_transition_improvement_gain": float(score_components["transition_improvement_gain"]),
                    "donor_transition_support_strength": float(score_components["transition_support_strength"]),
                    "donor_transition_score": float(score_components["transition_score"]),
                    "donor_action_change_score": float(score_components["action_change_score"]),
                }
            )
            ranked_candidates.append((entry, metadata))
        ranked_candidates.sort(key=lambda item: float(item[1].get("donor_total_score", 0.0)), reverse=True)
        return ranked_candidates

    def _counterfactual_donor_from_manager_result(
        self,
        sample: ForecastSample,
        encoding: TorchManifoldEncodingOutput,
        manager_result: ManagerReadResult,
    ) -> tuple[List[float], List[List[float]], Dict[str, object]]:
        ranked_candidates = self._counterfactual_ranked_donors_from_manager_result(sample, encoding, manager_result)
        if ranked_candidates:
            top_entry, top_metadata = ranked_candidates[0]
            return top_entry.intervention_static, top_entry.intervention_sequence, top_metadata
        return list(sample.intervention_static or []), [list(step) for step in (sample.intervention_sequence or [])], {}

    @staticmethod
    def _kg_flag(flags: Dict[str, object], name: str) -> float:
        if not flags:
            return 0.0
        candidates = [name]
        if name.startswith("kg_"):
            candidates.append(name[3:])
        else:
            candidates.append(f"kg_{name}")
        for candidate in candidates:
            if candidate in flags:
                try:
                    return float(flags.get(candidate, 0.0) or 0.0)
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    def _counterfactual_guideline_components(
        self,
        sample_kg_flags: Dict[str, object],
        donor_flags: Dict[str, object],
    ) -> Dict[str, float]:
        relevant_care_weights: List[tuple[str, float]] = []
        if self._kg_flag(sample_kg_flags, "state_sepsis") > 0.0:
            relevant_care_weights.extend(
                [
                    ("treat_early_antimicrobial", 0.35),
                    ("exam_blood_culture", 0.20),
                    ("exam_lactate", 0.20),
                    ("exam_sofa", 0.10),
                ]
            )
        if self._kg_flag(sample_kg_flags, "state_septic_shock") > 0.0 or self._kg_flag(sample_kg_flags, "state_hypotension") > 0.0:
            relevant_care_weights.extend(
                [
                    ("treat_vasopressor", 0.35),
                    ("monitor_map65", 0.15),
                    ("monitor_lactate_repeat", 0.10),
                ]
            )
        if self._kg_flag(sample_kg_flags, "state_high_lactate") > 0.0:
            relevant_care_weights.append(("exam_lactate", 0.10))

        total_care_weight = float(sum(weight for _, weight in relevant_care_weights))
        guideline_compatibility = 0.0
        missing_care_penalty = 0.0
        if total_care_weight > 0.0:
            matched_weight = sum(weight * self._kg_flag(donor_flags, feature_name) for feature_name, weight in relevant_care_weights)
            guideline_compatibility = float(matched_weight / total_care_weight)
            missing_care_penalty = float(max(0.0, 1.0 - guideline_compatibility))

        state_weight_map = {
            "state_sepsis": 0.35,
            "state_septic_shock": 0.25,
            "state_organ_dysfunction": 0.15,
            "state_hypotension": 0.15,
            "state_high_lactate": 0.10,
        }
        active_state_weight = 0.0
        matched_state_weight = 0.0
        for state_name, weight in state_weight_map.items():
            if self._kg_flag(sample_kg_flags, state_name) > 0.0:
                active_state_weight += weight
                matched_state_weight += weight * self._kg_flag(donor_flags, state_name)
        state_match = float(matched_state_weight / active_state_weight) if active_state_weight > 0.0 else 0.0

        contraindication_penalty = 0.0
        if self._kg_flag(sample_kg_flags, "state_sepsis") <= 0.0 and self._kg_flag(donor_flags, "treat_early_antimicrobial") > 0.0:
            contraindication_penalty += 0.30
        if (
            self._kg_flag(sample_kg_flags, "state_septic_shock") <= 0.0
            and self._kg_flag(sample_kg_flags, "state_hypotension") <= 0.0
            and self._kg_flag(donor_flags, "treat_vasopressor") > 0.0
        ):
            contraindication_penalty += 0.40
        if (
            self._kg_flag(sample_kg_flags, "state_septic_shock") > 0.0
            or self._kg_flag(sample_kg_flags, "state_hypotension") > 0.0
        ) and self._kg_flag(donor_flags, "treat_vasopressor") <= 0.0:
            contraindication_penalty += 0.25
        if self._kg_flag(sample_kg_flags, "state_organ_dysfunction") > 0.0 and self._kg_flag(donor_flags, "state_organ_dysfunction") <= 0.0:
            contraindication_penalty += 0.15

        return {
            "guideline_compatibility": float(guideline_compatibility),
            "state_match": float(state_match),
            "missing_care_penalty": float(missing_care_penalty),
            "contraindication_penalty": float(contraindication_penalty),
        }

    def _compute_counterfactual_donor_score(
        self,
        sample_kg_flags: Dict[str, object],
        donor_flags: Dict[str, object],
        embedding_similarity: float,
        kg_similarity: float,
        donor_guideline_score: float,
        transition_utility: float = 0.0,
        transition_confidence: float = 0.0,
        transition_advantage: float = 0.0,
        transition_action_gain: float = 0.0,
        transition_improvement_rate: float = 0.0,
        transition_improvement_gain: float = 0.0,
        transition_support_strength: float = 0.0,
        action_change_score: float = 0.0,
    ) -> Dict[str, float]:
        components = self._counterfactual_guideline_components(sample_kg_flags, donor_flags)
        guideline_compatibility = float(components["guideline_compatibility"])
        state_match = float(components["state_match"])
        missing_care_penalty = float(components["missing_care_penalty"])
        contraindication_penalty = float(components["contraindication_penalty"])
        transition_score = float(
            float(transition_confidence)
            * (0.5 + 0.5 * float(transition_support_strength))
            * (
                0.55 * float(transition_action_gain)
                + 0.20 * float(transition_advantage)
                + 0.20 * float(transition_improvement_gain)
                + 0.05 * float(transition_utility)
            )
        )

        if self.counterfactual_donor_score_mode == "structured":
            kg_component = 0.70 * float(kg_similarity) + 0.30 * float(state_match)
            total_penalty = float(0.65 * contraindication_penalty + 0.35 * missing_care_penalty)
        else:
            kg_component = float(kg_similarity)
            total_penalty = float(contraindication_penalty)

        final_score = (
            self.counterfactual_base_similarity_weight * float(embedding_similarity)
            + self.counterfactual_kg_similarity_weight * kg_component
            + self.counterfactual_guideline_weight * guideline_compatibility
            + self.counterfactual_guideline_score_weight * float(donor_guideline_score)
            + float(self.transition_action_change_weight) * float(action_change_score)
            + float(self.transition_score_weight) * transition_score
            - self.counterfactual_penalty_weight * total_penalty
        )
        return {
            "final_score": float(final_score),
            "kg_similarity": float(kg_similarity),
            "guideline_compatibility": float(guideline_compatibility),
            "state_match": float(state_match),
            "missing_care_penalty": float(missing_care_penalty),
            "contraindication_penalty": float(contraindication_penalty),
            "transition_utility": float(transition_utility),
            "transition_confidence": float(transition_confidence),
            "transition_advantage": float(transition_advantage),
            "transition_action_gain": float(transition_action_gain),
            "transition_improvement_rate": float(transition_improvement_rate),
            "transition_improvement_gain": float(transition_improvement_gain),
            "transition_support_strength": float(transition_support_strength),
            "transition_score": float(transition_score),
            "action_change_score": float(action_change_score),
        }

    def predict_counterfactual(
        self,
        samples: Sequence[ForecastSample],
        include_predictions: bool = True,
    ) -> Dict[str, object]:
        case_details: List[Dict[str, object]] = []
        predictions: List[List[float]] = []
        donor_stay_ids: List[float] = []
        deltas: List[float] = []
        donor_similarities: List[float] = []
        donor_kg_similarities: List[float] = []
        donor_guideline_compatibilities: List[float] = []
        donor_state_matches: List[float] = []
        donor_missing_care_penalties: List[float] = []
        donor_contraindication_penalties: List[float] = []
        donor_total_scores: List[float] = []
        donor_transition_utilities: List[float] = []
        donor_transition_confidences: List[float] = []
        donor_transition_advantages: List[float] = []
        donor_transition_action_gains: List[float] = []
        donor_transition_improvement_rates: List[float] = []
        donor_transition_improvement_gains: List[float] = []
        donor_transition_scores: List[float] = []
        donor_action_change_scores: List[float] = []
        donor_hard_filter_fallbacks: List[float] = []
        donor_overlap_scores: List[float] = []
        donor_overlap_valids: List[float] = []
        donor_overlap_fallbacks: List[float] = []
        donor_learned_reranker_scores: List[float] = []
        donor_reranker_adjustments: List[float] = []
        donor_overlap_reason_counts: Dict[str, int] = {}
        selected_candidate_sources: List[str] = []
        generated_candidate_selected_count = 0.0
        generated_candidate_available_count = 0.0
        search_candidate_selected_count = 0.0
        search_candidate_available_count = 0.0
        exact_label_matches = 0.0
        self.eval()
        with torch.no_grad():
            batch_size = max(1, self.trainer_config.batch_size)
            for start in range(0, len(samples), batch_size):
                batch_samples = list(samples[start : start + batch_size])
                encodings, manager_results, _, factual_batch_predictions, factual_gate_summaries, _ = self._forward_batch(batch_samples)
                batch_factual = [
                    [float(value) * sample.scale_value + sample.scale_center for value in prediction]
                    for sample, prediction in zip(batch_samples, factual_batch_predictions.detach().cpu().tolist())
                ]
                batch_factual_aux_predictions = self._decode_multitask_predictions(factual_gate_summaries)
                expanded_counterfactual_samples: List[ForecastSample] = []
                candidate_groups: List[List[Tuple[ForecastSample, Dict[str, object]]]] = []
                ranked_donor_groups: List[List[Tuple[InterventionStoreEntry, Dict[str, object]]]] = []
                for sample, factual_prediction, encoding, manager_result in zip(batch_samples, batch_factual, encodings, manager_results):
                    ranked_donors = self._counterfactual_ranked_donors_from_manager_result(sample, encoding, manager_result)
                    if ranked_donors:
                        donor_entry, donor_metadata = ranked_donors[0]
                        donor_intervention = donor_entry.intervention_static
                        donor_intervention_sequence = donor_entry.intervention_sequence
                    else:
                        donor_metadata = {}
                        donor_intervention = list(sample.intervention_static or [])
                        donor_intervention_sequence = [list(step) for step in (sample.intervention_sequence or [])]
                    candidates = self._generate_counterfactual_intervention_candidates(
                        sample=sample,
                        donor_intervention=donor_intervention,
                        donor_intervention_sequence=donor_intervention_sequence,
                        donor_metadata=donor_metadata,
                    )
                    if len(candidates) > 1:
                        generated_candidate_available_count += 1.0
                    if any(str(candidate_metadata.get("generated_candidate_layer", "")) == "search" for _, _, candidate_metadata in candidates):
                        search_candidate_available_count += 1.0
                    candidate_group: List[Tuple[ForecastSample, Dict[str, object]]] = []
                    for candidate_intervention, candidate_sequence, candidate_metadata in candidates:
                        candidate_sample = replace(
                            sample,
                            intervention_static=candidate_intervention,
                            intervention_sequence=candidate_sequence,
                            metadata={
                                **sample.metadata,
                                "counterfactual_donor_stay_id": candidate_metadata.get("stay_id", -1.0),
                                "counterfactual_candidate_source": candidate_metadata.get("generated_candidate_source", ""),
                            },
                        )
                        candidate_group.append((candidate_sample, candidate_metadata))
                        expanded_counterfactual_samples.append(candidate_sample)
                    candidate_groups.append(candidate_group)
                    ranked_donor_groups.append(ranked_donors)

                _, _, _, counterfactual_predictions, candidate_gate_summaries, _ = self._forward_batch(expanded_counterfactual_samples)
                prediction_rows = counterfactual_predictions.detach().cpu().tolist()
                candidate_aux_prediction_rows = self._decode_multitask_predictions(candidate_gate_summaries)
                prediction_offset = 0
                for sample, factual_prediction, factual_aux_predictions, factual_gate_summary, candidate_group, ranked_donors in zip(
                    batch_samples,
                    batch_factual,
                    batch_factual_aux_predictions,
                    factual_gate_summaries,
                    candidate_groups,
                    ranked_donor_groups,
                ):
                    selected_prediction: List[float] = []
                    selected_metadata: Dict[str, object] = {}
                    selected_delta = float("-inf")
                    selected_score = float("-inf")
                    candidate_option_details: List[Dict[str, object]] = []
                    for candidate_sample, candidate_metadata in candidate_group:
                        neighborhood_summary = self._counterfactual_neighbor_summary(
                            candidate_metadata=candidate_metadata,
                            ranked_donors=ranked_donors,
                        )
                        candidate_prediction = prediction_rows[prediction_offset]
                        candidate_aux_predictions = candidate_aux_prediction_rows[prediction_offset]
                        candidate_gate_summary = candidate_gate_summaries[prediction_offset]
                        prediction_offset += 1
                        restored_candidate_prediction = [
                            value * sample.scale_value + sample.scale_center
                            for value in candidate_prediction
                        ]
                        predicted_delta = sum(factual_prediction) / max(1, len(factual_prediction)) - sum(restored_candidate_prediction) / max(1, len(restored_candidate_prediction))
                        selection_score, selection_components = self._candidate_selection_score(
                            predicted_delta,
                            candidate_metadata,
                            current_aux_predictions=factual_aux_predictions,
                            candidate_aux_predictions=candidate_aux_predictions,
                            current_gate_summary=factual_gate_summary,
                            candidate_gate_summary=candidate_gate_summary,
                            neighborhood_summary=neighborhood_summary,
                        )
                        if selection_score > selected_score:
                            selected_score = selection_score
                            selected_delta = predicted_delta
                            selected_prediction = restored_candidate_prediction
                            selected_metadata = {
                                **candidate_metadata,
                                "candidate_selection_score": float(selection_score),
                                "stage2_base_selection_score": float(selection_components.get("base_rule_score", selection_score)),
                                "stage3_pre_neighborhood_selection_score": float(
                                    selection_components.get("pre_neighborhood_score", selection_score)
                                ),
                                "selected_predicted_delta": float(predicted_delta),
                                "selection_components": dict(selection_components),
                                "candidate_aux_predictions": dict(candidate_aux_predictions),
                                "neighborhood_summary": dict(neighborhood_summary),
                            }
                        candidate_option_details.append(
                            {
                                "candidate_source": str(candidate_metadata.get("generated_candidate_source", "")),
                                "candidate_layer": str(candidate_metadata.get("generated_candidate_layer", "safety")),
                                "candidate_selection_score": float(selection_score),
                                "stage2_base_selection_score": float(selection_components.get("base_rule_score", selection_score)),
                                "stage3_pre_neighborhood_selection_score": float(
                                    selection_components.get("pre_neighborhood_score", selection_score)
                                ),
                                "predicted_delta": float(predicted_delta),
                                "predicted_counterfactual": restored_candidate_prediction,
                                "selection_components": dict(selection_components),
                                "candidate_aux_predictions": dict(candidate_aux_predictions),
                                "neighborhood_summary": dict(neighborhood_summary),
                                "candidate_intervention_static": [float(value) for value in (candidate_sample.intervention_static or [])],
                                "candidate_intervention_sequence": [
                                    [float(value) for value in step]
                                    for step in (candidate_sample.intervention_sequence or [])
                                ],
                                "repair_actions": [
                                    action
                                    for action in str(candidate_metadata.get("generated_candidate_repair_actions", "")).split("|")
                                    if action
                                ],
                                "search_actions": [
                                    action
                                    for action in str(candidate_metadata.get("generated_candidate_search_actions", "")).split("|")
                                    if action
                                ],
                                "candidate_strategy_family": str(candidate_metadata.get("generated_candidate_strategy_family", "")),
                                "candidate_parameter_profile": dict(candidate_metadata.get("generated_candidate_parameter_profile", {})),
                                "candidate_anchor_relation": str(candidate_metadata.get("generated_candidate_anchor_relation", "donor_centered")),
                                "candidate_safety_rationale": [
                                    str(item) for item in candidate_metadata.get("generated_candidate_safety_rationale", [])
                                ],
                                "donor_stay_id": float(candidate_metadata.get("stay_id", -1.0)),
                                "donor_similarity": float(candidate_metadata.get("donor_similarity", 0.0)),
                                "donor_kg_similarity": float(candidate_metadata.get("donor_kg_similarity", 0.0)),
                                "donor_guideline_compatibility": float(candidate_metadata.get("donor_guideline_compatibility", 0.0)),
                                "donor_state_match": float(candidate_metadata.get("donor_state_match", 0.0)),
                                "donor_missing_care_penalty": float(candidate_metadata.get("donor_missing_care_penalty", 0.0)),
                                "donor_contraindication_penalty": float(candidate_metadata.get("donor_contraindication_penalty", 0.0)),
                                "donor_overlap_score": float(candidate_metadata.get("donor_overlap_score", 0.0)),
                                "donor_overlap_valid": float(candidate_metadata.get("donor_overlap_valid", 0.0)),
                                "donor_overlap_reason": str(candidate_metadata.get("donor_overlap_reason", "")),
                                "donor_neighbor_consistency": float(candidate_metadata.get("donor_neighbor_consistency", 0.0)),
                                "donor_neighbor_exchangeability_mean": float(candidate_metadata.get("donor_neighbor_exchangeability_mean", 0.0)),
                                "donor_neighbor_hard_pass_rate": float(candidate_metadata.get("donor_neighbor_hard_pass_rate", 0.0)),
                                "donor_neighbor_overlap_valid_rate": float(candidate_metadata.get("donor_neighbor_overlap_valid_rate", 0.0)),
                                "donor_learned_reranker_score": float(candidate_metadata.get("donor_learned_reranker_score", 0.0)),
                                "donor_reranker_adjustment": float(candidate_metadata.get("donor_reranker_adjustment", 0.0)),
                                "donor_pool_match_score": float(candidate_metadata.get("donor_pool_match_score", 0.0)),
                                "donor_pool_match_reward": float(candidate_metadata.get("donor_pool_match_reward", 0.0)),
                                "donor_pool_tags": [str(value) for value in candidate_metadata.get("donor_pool_tags", [])],
                                "donor_total_score": float(candidate_metadata.get("donor_total_score", 0.0)),
                                "plan_summary": self._serialize_intervention_plan(
                                    intervention_static=candidate_sample.intervention_static or [],
                                    intervention_sequence=candidate_sample.intervention_sequence or [],
                                    base_flags=dict(candidate_metadata.get("kg_flags", {})),
                                ),
                            }
                        )
                    if selected_metadata.get("generated_candidate_source") == "generated_kg_repaired":
                        generated_candidate_selected_count += 1.0
                    if str(selected_metadata.get("generated_candidate_layer", "")) == "search":
                        search_candidate_selected_count += 1.0
                    restored_prediction = selected_prediction
                    if include_predictions:
                        predictions.append(restored_prediction)
                    donor_stay_ids.append(float(selected_metadata.get("stay_id", -1.0)))
                    donor_similarities.append(float(selected_metadata.get("donor_similarity", 0.0)))
                    donor_kg_similarities.append(float(selected_metadata.get("donor_kg_similarity", 0.0)))
                    donor_guideline_compatibilities.append(float(selected_metadata.get("donor_guideline_compatibility", 0.0)))
                    donor_state_matches.append(float(selected_metadata.get("donor_state_match", 0.0)))
                    donor_missing_care_penalties.append(float(selected_metadata.get("donor_missing_care_penalty", 0.0)))
                    donor_contraindication_penalties.append(float(selected_metadata.get("donor_contraindication_penalty", 0.0)))
                    donor_total_scores.append(float(selected_metadata.get("donor_total_score", 0.0)))
                    donor_transition_utilities.append(float(selected_metadata.get("donor_transition_utility", 0.0)))
                    donor_transition_confidences.append(float(selected_metadata.get("donor_transition_confidence", 0.0)))
                    donor_transition_advantages.append(float(selected_metadata.get("donor_transition_advantage", 0.0)))
                    donor_transition_action_gains.append(float(selected_metadata.get("donor_transition_action_gain", 0.0)))
                    donor_transition_improvement_rates.append(float(selected_metadata.get("donor_transition_improvement_rate", 0.0)))
                    donor_transition_improvement_gains.append(float(selected_metadata.get("donor_transition_improvement_gain", 0.0)))
                    donor_transition_scores.append(float(selected_metadata.get("donor_transition_score", 0.0)))
                    donor_action_change_scores.append(float(selected_metadata.get("donor_action_change_score", 0.0)))
                    donor_hard_filter_fallbacks.append(float(selected_metadata.get("donor_hard_filter_fallback", 0.0)))
                    donor_overlap_scores.append(float(selected_metadata.get("donor_overlap_score", 0.0)))
                    donor_overlap_valids.append(float(selected_metadata.get("donor_overlap_valid", 0.0)))
                    donor_overlap_fallbacks.append(float(selected_metadata.get("donor_overlap_fallback", 0.0)))
                    donor_learned_reranker_scores.append(float(selected_metadata.get("donor_learned_reranker_score", 0.0)))
                    donor_reranker_adjustments.append(float(selected_metadata.get("donor_reranker_adjustment", 0.0)))
                    for reason in str(selected_metadata.get("donor_overlap_reason", "")).split("|"):
                        if reason:
                            donor_overlap_reason_counts[str(reason)] = donor_overlap_reason_counts.get(str(reason), 0) + 1
                    selected_candidate_sources.append(str(selected_metadata.get("generated_candidate_source", "")))
                    if int(selected_metadata.get("donor_experience_label", -1.0)) == int(sample.experience_label):
                        exact_label_matches += 1.0
                    deltas.append(float(selected_delta))
                    case_details.append(
                        {
                            "query": self._serialize_query_snapshot(sample),
                            "factual_prediction": [float(value) for value in factual_prediction],
                            "selected_counterfactual_prediction": [float(value) for value in restored_prediction],
                            "selected_predicted_delta": float(selected_delta),
                            "selected_candidate_source": str(selected_metadata.get("generated_candidate_source", "")),
                            "selected_candidate": {
                                "candidate_source": str(selected_metadata.get("generated_candidate_source", "")),
                                "candidate_layer": str(selected_metadata.get("generated_candidate_layer", "safety")),
                                "candidate_selection_score": float(selected_metadata.get("candidate_selection_score", selected_score)),
                                "stage2_base_selection_score": float(selected_metadata.get("stage2_base_selection_score", selected_score)),
                                "stage3_pre_neighborhood_selection_score": float(
                                    selected_metadata.get("stage3_pre_neighborhood_selection_score", selected_score)
                                ),
                                "predicted_delta": float(selected_metadata.get("selected_predicted_delta", selected_delta)),
                                "selection_components": dict(selected_metadata.get("selection_components", {})),
                                "candidate_aux_predictions": dict(selected_metadata.get("candidate_aux_predictions", {})),
                                "neighborhood_summary": dict(selected_metadata.get("neighborhood_summary", {})),
                                "repair_actions": [
                                    action
                                    for action in str(selected_metadata.get("generated_candidate_repair_actions", "")).split("|")
                                    if action
                                ],
                                "search_actions": [
                                    action
                                    for action in str(selected_metadata.get("generated_candidate_search_actions", "")).split("|")
                                    if action
                                ],
                                "candidate_strategy_family": str(selected_metadata.get("generated_candidate_strategy_family", "")),
                                "candidate_parameter_profile": dict(selected_metadata.get("generated_candidate_parameter_profile", {})),
                                "candidate_anchor_relation": str(selected_metadata.get("generated_candidate_anchor_relation", "donor_centered")),
                                "candidate_safety_rationale": [
                                    str(item) for item in selected_metadata.get("generated_candidate_safety_rationale", [])
                                ],
                                "donor": {
                                    "stay_id": float(selected_metadata.get("stay_id", -1.0)),
                                    "donor_experience_id": str(selected_metadata.get("donor_experience_id", "")),
                                    "donor_intervention_plan_code": dict(
                                        selected_metadata.get("donor_intervention_plan_code", {})
                                    ),
                                    "donor_experience_label": int(float(selected_metadata.get("donor_experience_label", -1.0))),
                                    "donor_pattern_label": int(float(selected_metadata.get("donor_pattern_label", -1.0))),
                                    "donor_trajectory_label": int(float(selected_metadata.get("donor_trajectory_label", -1.0))),
                                    "donor_similarity": float(selected_metadata.get("donor_similarity", 0.0)),
                                    "donor_kg_similarity": float(selected_metadata.get("donor_kg_similarity", 0.0)),
                                    "donor_guideline_compatibility": float(selected_metadata.get("donor_guideline_compatibility", 0.0)),
                                    "donor_state_match": float(selected_metadata.get("donor_state_match", 0.0)),
                                    "donor_missing_care_penalty": float(selected_metadata.get("donor_missing_care_penalty", 0.0)),
                                    "donor_contraindication_penalty": float(selected_metadata.get("donor_contraindication_penalty", 0.0)),
                                    "donor_total_score": float(selected_metadata.get("donor_total_score", 0.0)),
                                    "donor_hard_filter_valid": float(selected_metadata.get("donor_hard_filter_valid", 0.0)),
                                    "donor_hard_filter_reason": str(selected_metadata.get("donor_hard_filter_reason", "")),
                                    "donor_overlap_score": float(selected_metadata.get("donor_overlap_score", 0.0)),
                                    "donor_overlap_valid": float(selected_metadata.get("donor_overlap_valid", 0.0)),
                                    "donor_overlap_reason": str(selected_metadata.get("donor_overlap_reason", "")),
                                    "donor_neighbor_consistency": float(selected_metadata.get("donor_neighbor_consistency", 0.0)),
                                    "donor_neighbor_exchangeability_mean": float(selected_metadata.get("donor_neighbor_exchangeability_mean", 0.0)),
                                    "donor_neighbor_hard_pass_rate": float(selected_metadata.get("donor_neighbor_hard_pass_rate", 0.0)),
                                    "donor_neighbor_overlap_valid_rate": float(selected_metadata.get("donor_neighbor_overlap_valid_rate", 0.0)),
                                    "donor_learned_reranker_score": float(selected_metadata.get("donor_learned_reranker_score", 0.0)),
                                    "donor_reranker_adjustment": float(selected_metadata.get("donor_reranker_adjustment", 0.0)),
                                    "donor_reranker_mode": str(selected_metadata.get("donor_reranker_mode", "rule_only")),
                                    "donor_pool_match_score": float(selected_metadata.get("donor_pool_match_score", 0.0)),
                                    "donor_pool_match_reward": float(selected_metadata.get("donor_pool_match_reward", 0.0)),
                                    "donor_pool_tags": [str(value) for value in selected_metadata.get("donor_pool_tags", [])],
                                },
                                "plan_summary": next(
                                    (
                                        dict(option.get("plan_summary", {}))
                                        for option in candidate_option_details
                                        if str(option.get("candidate_source", "")) == str(selected_metadata.get("generated_candidate_source", ""))
                                    ),
                                    {},
                                ),
                                "candidate_intervention_static": next(
                                    (
                                        list(option.get("candidate_intervention_static", []))
                                        for option in candidate_option_details
                                        if str(option.get("candidate_source", "")) == str(selected_metadata.get("generated_candidate_source", ""))
                                    ),
                                    [],
                                ),
                                "candidate_intervention_sequence": next(
                                    (
                                        [list(step) for step in option.get("candidate_intervention_sequence", [])]
                                        for option in candidate_option_details
                                        if str(option.get("candidate_source", "")) == str(selected_metadata.get("generated_candidate_source", ""))
                                    ),
                                    [],
                                ),
                            },
                            "candidate_options": candidate_option_details,
                            "top_donor_candidates": [
                                self._serialize_ranked_donor_candidate(entry, metadata)
                                for entry, metadata in ranked_donors[:3]
                            ],
                        }
                    )
        return {
            "predictions": predictions if include_predictions else [],
            "case_details": case_details,
            "donor_stay_ids": donor_stay_ids,
            "donor_found_rate": float(sum(1 for stay_id in donor_stay_ids if stay_id >= 0.0) / max(1, len(donor_stay_ids))),
            "donor_similarity_mean": float(sum(donor_similarities) / max(1, len(donor_similarities))),
            "donor_kg_similarity_mean": float(sum(donor_kg_similarities) / max(1, len(donor_kg_similarities))),
            "donor_guideline_compatibility_mean": float(sum(donor_guideline_compatibilities) / max(1, len(donor_guideline_compatibilities))),
            "donor_state_match_mean": float(sum(donor_state_matches) / max(1, len(donor_state_matches))),
            "donor_missing_care_penalty_mean": float(sum(donor_missing_care_penalties) / max(1, len(donor_missing_care_penalties))),
            "donor_contraindication_penalty_mean": float(sum(donor_contraindication_penalties) / max(1, len(donor_contraindication_penalties))),
            "donor_total_score_mean": float(sum(donor_total_scores) / max(1, len(donor_total_scores))),
            "donor_transition_utility_mean": float(sum(donor_transition_utilities) / max(1, len(donor_transition_utilities))),
            "donor_transition_confidence_mean": float(sum(donor_transition_confidences) / max(1, len(donor_transition_confidences))),
            "donor_transition_advantage_mean": float(sum(donor_transition_advantages) / max(1, len(donor_transition_advantages))),
            "donor_transition_action_gain_mean": float(sum(donor_transition_action_gains) / max(1, len(donor_transition_action_gains))),
            "donor_transition_improvement_rate_mean": float(sum(donor_transition_improvement_rates) / max(1, len(donor_transition_improvement_rates))),
            "donor_transition_improvement_gain_mean": float(sum(donor_transition_improvement_gains) / max(1, len(donor_transition_improvement_gains))),
            "donor_transition_score_mean": float(sum(donor_transition_scores) / max(1, len(donor_transition_scores))),
            "donor_action_change_score_mean": float(sum(donor_action_change_scores) / max(1, len(donor_action_change_scores))),
            "donor_hard_filter_fallback_rate": float(sum(donor_hard_filter_fallbacks) / max(1, len(donor_hard_filter_fallbacks))),
            "donor_overlap_score_mean": float(sum(donor_overlap_scores) / max(1, len(donor_overlap_scores))),
            "donor_overlap_valid_rate": float(sum(donor_overlap_valids) / max(1, len(donor_overlap_valids))),
            "donor_overlap_fallback_rate": float(sum(donor_overlap_fallbacks) / max(1, len(donor_overlap_fallbacks))),
            "donor_learned_reranker_score_mean": float(sum(donor_learned_reranker_scores) / max(1, len(donor_learned_reranker_scores))),
            "donor_reranker_adjustment_mean": float(sum(donor_reranker_adjustments) / max(1, len(donor_reranker_adjustments))),
            "donor_overlap_reason_counts": dict(sorted(donor_overlap_reason_counts.items())),
            "donor_exact_experience_match_rate": float(exact_label_matches / max(1, len(samples))),
            "donor_score_mode": self.counterfactual_donor_score_mode,
            "donor_reranker_mode": str(self.counterfactual_reranker_mode),
            "donor_reranker_state": dict(self.counterfactual_reranker_state),
            "counterfactual_candidate_policy": self.counterfactual_candidate_policy,
            "counterfactual_candidate_search_mode": self.counterfactual_candidate_search_mode,
            "generated_candidate_available_rate": float(generated_candidate_available_count / max(1, len(samples))),
            "generated_candidate_selected_rate": float(generated_candidate_selected_count / max(1, len(samples))),
            "search_candidate_available_rate": float(search_candidate_available_count / max(1, len(samples))),
            "search_candidate_selected_rate": float(search_candidate_selected_count / max(1, len(samples))),
            "selected_candidate_source_counts": {
                source: selected_candidate_sources.count(source)
                for source in sorted(set(selected_candidate_sources))
            },
            "mean_predicted_delta": float(sum(deltas) / max(1, len(deltas))),
            "predicted_improvement_rate": float(sum(1 for delta in deltas if delta > 0.0) / max(1, len(deltas))),
        }

    def predict_counterfactual_rollout(
        self,
        samples: Sequence[ForecastSample],
        rollout_steps: Optional[int] = None,
        include_predictions: bool = True,
    ) -> Dict[str, object]:
        total_steps = max(1, int(rollout_steps or self.counterfactual_rollout_steps))
        discount = float(self.counterfactual_rollout_discount)
        if not samples:
            return {
                "rollout_steps": total_steps,
                "rollout_discount": discount,
                "step_summaries": [],
                "case_rollouts": [],
                "mean_discounted_cumulative_delta": 0.0,
                "positive_discounted_cumulative_rate": 0.0,
                "second_step_available_rate": 0.0,
                "stable_candidate_source_rate": 0.0,
                "step_positive_rates": [],
            }

        current_samples = list(samples)
        case_rollouts: List[Dict[str, object]] = [
            {
                "query": self._serialize_query_snapshot(sample),
                "steps": [],
                "discounted_cumulative_delta": 0.0,
                "raw_cumulative_delta": 0.0,
            }
            for sample in samples
        ]
        step_summaries: List[Dict[str, object]] = []

        for step_idx in range(total_steps):
            step_summary = self.predict_counterfactual(current_samples, include_predictions=include_predictions)
            case_details = list(step_summary.get("case_details", []))
            step_summaries.append(
                {
                    "step_index": int(step_idx + 1),
                    "mean_predicted_delta": float(step_summary.get("mean_predicted_delta", 0.0)),
                    "predicted_improvement_rate": float(step_summary.get("predicted_improvement_rate", 0.0)),
                    "search_candidate_selected_rate": float(step_summary.get("search_candidate_selected_rate", 0.0)),
                    "generated_candidate_selected_rate": float(step_summary.get("generated_candidate_selected_rate", 0.0)),
                    "case_details": case_details,
                }
            )
            next_samples: List[ForecastSample] = []
            for case_index, case_detail in enumerate(case_details):
                selected_candidate = dict(case_detail.get("selected_candidate", {}))
                predicted_delta = float(case_detail.get("selected_predicted_delta", 0.0))
                discounted_delta = float((discount ** step_idx) * predicted_delta)
                rollout_case = case_rollouts[case_index]
                rollout_case["steps"].append(
                    {
                        "step_index": int(step_idx + 1),
                        "query": dict(case_detail.get("query", {})),
                        "factual_prediction": [
                            float(value) for value in case_detail.get("factual_prediction", [])
                        ],
                        "selected_counterfactual_prediction": [
                            float(value)
                            for value in case_detail.get("selected_counterfactual_prediction", [])
                        ],
                        "selected_predicted_delta": float(predicted_delta),
                        "discounted_delta": float(discounted_delta),
                        "selected_candidate": selected_candidate,
                        "top_donor_candidates": list(case_detail.get("top_donor_candidates", [])),
                    }
                )
                rollout_case["discounted_cumulative_delta"] = float(
                    rollout_case.get("discounted_cumulative_delta", 0.0) + discounted_delta
                )
                rollout_case["raw_cumulative_delta"] = float(
                    rollout_case.get("raw_cumulative_delta", 0.0) + predicted_delta
                )
                if step_idx + 1 < total_steps:
                    next_samples.append(
                        self._project_rollout_sample(
                            sample=current_samples[case_index],
                            case_detail=case_detail,
                            step_index=step_idx + 1,
                        )
                    )
            current_samples = next_samples

        discounted_cumulative_deltas = [
            float(case.get("discounted_cumulative_delta", 0.0))
            for case in case_rollouts
        ]
        second_step_available_count = sum(
            1.0 for case in case_rollouts if len(case.get("steps", [])) >= 2
        )
        stable_candidate_source_count = 0.0
        stable_candidate_source_denominator = 0.0
        for case in case_rollouts:
            steps = list(case.get("steps", []))
            if len(steps) < 2:
                continue
            stable_candidate_source_denominator += 1.0
            first_source = str(dict(steps[0].get("selected_candidate", {})).get("candidate_source", ""))
            second_source = str(dict(steps[1].get("selected_candidate", {})).get("candidate_source", ""))
            if first_source and first_source == second_source:
                stable_candidate_source_count += 1.0

        step_positive_rates: List[float] = []
        for step_summary in step_summaries:
            case_details = list(step_summary.get("case_details", []))
            positive_rate = float(
                sum(
                    1.0
                    for case_detail in case_details
                    if float(case_detail.get("selected_predicted_delta", 0.0)) > 0.0
                )
                / max(1, len(case_details))
            )
            step_positive_rates.append(positive_rate)

        return {
            "rollout_steps": int(total_steps),
            "rollout_discount": float(discount),
            "step_summaries": step_summaries,
            "case_rollouts": case_rollouts,
            "mean_discounted_cumulative_delta": float(
                sum(discounted_cumulative_deltas) / max(1, len(discounted_cumulative_deltas))
            ),
            "positive_discounted_cumulative_rate": float(
                sum(1.0 for value in discounted_cumulative_deltas if value > 0.0)
                / max(1, len(discounted_cumulative_deltas))
            ),
            "second_step_available_rate": float(second_step_available_count / max(1, len(case_rollouts))),
            "stable_candidate_source_rate": float(
                stable_candidate_source_count / max(1.0, stable_candidate_source_denominator)
            ),
            "step_positive_rates": step_positive_rates,
        }

    def inspect_sample(self, sample: ForecastSample) -> Dict[str, object]:
        self.eval()
        with torch.no_grad():
            normalized_sequence = self._normalize_sequence(sample.sequence)
            normalized_static = self._normalize_static(sample.patient_static or sample.static)
            normalized_intervention = self._normalize_intervention(sample.intervention_static or [])
            normalized_intervention_sequence = self._normalize_intervention_sequence(sample.intervention_sequence or [])
            normalized_formation = self._normalize_formation(sample.formation_features)
            encoding = self.manifold.encode_input_torch(
                normalized_sequence,
                normalized_static,
                metadata=sample.metadata,
            )
            temporal_profile = self._temporal_profile_tensor(sample.metadata if isinstance(sample.metadata, dict) else {})
            (
                factual_embedding,
                retrieval_encoding,
                factual_branch_gate,
                factual_branch_strength,
                retrieval_branch_gate,
                retrieval_branch_strength,
            ) = self._split_encoding_spaces(
                encoding,
                normalized_formation=normalized_formation,
                temporal_profile=temporal_profile,
            )
            factual_embedding, temporal_gate, temporal_strength = self._augment_factual_embedding(
                factual_embedding=factual_embedding,
                temporal_profile=temporal_profile,
                normalized_formation=normalized_formation,
            )
            manager_result = self.memory_manager.read(
                encoding=retrieval_encoding,
                horizon=self.trainer_config.forecast_horizon,
                history_length=self.trainer_config.history_length,
                formation_features=sample.formation_features,
                pattern_label=sample.pattern_label,
                trajectory_label=sample.trajectory_label,
                experience_label=sample.experience_label,
                **self._memory_read_kwargs(sample),
            )
            intervention_embedding = self._project_intervention(
                normalized_intervention,
                normalized_intervention_sequence,
            )
            fused_representation, direct_memory_residual, planner_tensor, confidence_tensor, scenario_tensor, gate_summary = self._build_fused_representation(
                factual_embedding=factual_embedding,
                intervention_embedding=intervention_embedding,
                normalized_formation=normalized_formation,
                manager_result=manager_result,
            )
            residual_prediction, bucket_summary = self._bucket_residual(fused_representation)
            _, coordination_summary, _ = self._coordinate_memory_paths(
                residual_prediction=residual_prediction,
                direct_memory_residual=direct_memory_residual,
                planner_tensor=planner_tensor,
                confidence_tensor=confidence_tensor,
                scenario_tensor=scenario_tensor,
                gate_summary=gate_summary,
            )
            gate_summary.update(coordination_summary)
            factual_prediction_input, factual_prediction_gate, factual_prediction_strength = self._augment_factual_prediction_input(
                base_input=torch.cat([factual_embedding, intervention_embedding], dim=-1),
                temporal_profile=temporal_profile,
                normalized_formation=normalized_formation,
            )
            predicted_factual_scale = self._predict_factual_scale(factual_prediction_input)
            gate_summary.update(
                {
                    "factual_branch_gate": float(factual_branch_gate.item()),
                    "factual_branch_strength": float(factual_branch_strength.item()),
                    "retrieval_branch_gate": float(retrieval_branch_gate.item()),
                    "retrieval_branch_strength": float(retrieval_branch_strength.item()),
                    "factual_prediction_gate": float(factual_prediction_gate.item()),
                    "factual_prediction_strength": float(factual_prediction_strength.item()),
                    "factual_predicted_scale_mean": float(predicted_factual_scale.mean().item()),
                    "temporal_factual_gate": float(temporal_gate.item()),
                    "temporal_factual_strength": float(temporal_strength.item()),
                }
            )
            gate_summary["_predicted_factual_scale_tensor"] = predicted_factual_scale
            serializable_gates = {
                key: float(value.item()) if isinstance(value, torch.Tensor) else value
                for key, value in gate_summary.items()
                if not str(key).startswith("_")
            }
            return {
                "planner_weights": manager_result.planner_weights,
                "scenario_features": manager_result.scenario_features,
                "semantic_retrieval": manager_result.semantic_result,
                "gates": serializable_gates,
                "component_confidences": {
                    "pattern": float(confidence_tensor[0].item()),
                    "trajectory": float(confidence_tensor[1].item()),
                    "experience": float(confidence_tensor[2].item()),
                },
                "bucket_summary": bucket_summary,
                "components": {
                    name: {
                        "confidence": result.confidence,
                        "top_label": result.top_label,
                        "max_similarity": result.max_similarity,
                        "matched_indices": result.matched_indices[:5],
                        "matched_labels": result.matched_labels[:5],
                        "retrieval_source": result.retrieval_source,
                        "archive_used": result.archive_used,
                        "archive_confidence": result.archive_confidence,
                    }
                    for name, result in manager_result.component_results.items()
                },
                "scenario_tensor": scenario_tensor.detach().cpu().tolist(),
            }

    def _collect_diagnostics(self, samples: Sequence[ForecastSample]) -> Dict[str, object]:
        gate_sums = {"pattern_gate": 0.0, "trajectory_gate": 0.0, "experience_gate": 0.0}
        direct_gate_sums = {
            "direct_pattern_strength": 0.0,
            "direct_trajectory_strength": 0.0,
            "direct_experience_strength": 0.0,
            "direct_memory_strength": 0.0,
            "adaptive_direct_gate": 0.0,
            "direct_residual_scale": 0.0,
            "fusion_path_scale": 0.0,
            "direct_path_scale": 0.0,
            "coordinated_memory_strength": 0.0,
            "path_disagreement": 0.0,
            "path_alignment": 0.0,
            "factual_branch_gate": 0.0,
            "factual_branch_strength": 0.0,
            "retrieval_branch_gate": 0.0,
            "retrieval_branch_strength": 0.0,
            "factual_prediction_gate": 0.0,
            "factual_prediction_strength": 0.0,
            "factual_predicted_scale_mean": 0.0,
            "temporal_factual_gate": 0.0,
            "temporal_factual_strength": 0.0,
            "memory_harm_control_scale": 0.0,
            "memory_harm_confidence_scale": 0.0,
            "memory_harm_quality": 0.0,
            "memory_harm_alignment_scale": 0.0,
            "memory_harm_cap_scale": 0.0,
            "memory_harm_pre_strength": 0.0,
            "memory_harm_post_strength": 0.0,
        }
        planner_sums = {"pattern": 0.0, "trajectory": 0.0, "experience": 0.0}
        confidence_sums = {"pattern": 0.0, "trajectory": 0.0, "experience": 0.0}
        bucket_sums = {"short_bucket_strength": 0.0, "mid_bucket_strength": 0.0, "long_bucket_strength": 0.0}
        retrieval_source_counts = {"hot": 0.0, "blended": 0.0}
        archive_use_count = 0.0
        semantic_hit_count = 0.0
        semantic_top_score_sum = 0.0
        semantic_support_sum = 0.0
        semantic_template_confidence_sum = 0.0
        semantic_template_blend_sum = 0.0
        semantic_label_counts: Dict[str, float] = {}
        count = 0

        self.eval()
        with torch.no_grad():
            for sample in samples:
                normalized_sequence = self._normalize_sequence(sample.sequence)
                normalized_static = self._normalize_static(sample.patient_static or sample.static)
                normalized_intervention = self._normalize_intervention(sample.intervention_static or [])
                normalized_intervention_sequence = self._normalize_intervention_sequence(sample.intervention_sequence or [])
                normalized_formation = self._normalize_formation(sample.formation_features)
                encoding = self.manifold.encode_input_torch(
                    normalized_sequence,
                    normalized_static,
                    metadata=sample.metadata,
                )
                temporal_profile = self._temporal_profile_tensor(sample.metadata if isinstance(sample.metadata, dict) else {})
                (
                    factual_embedding,
                    retrieval_encoding,
                    factual_branch_gate,
                    factual_branch_strength,
                    retrieval_branch_gate,
                    retrieval_branch_strength,
                ) = self._split_encoding_spaces(
                    encoding,
                    normalized_formation=normalized_formation,
                    temporal_profile=temporal_profile,
                )
                factual_embedding, temporal_gate, temporal_strength = self._augment_factual_embedding(
                    factual_embedding=factual_embedding,
                    temporal_profile=temporal_profile,
                    normalized_formation=normalized_formation,
                )
                manager_result = self.memory_manager.read(
                    encoding=retrieval_encoding,
                    horizon=self.trainer_config.forecast_horizon,
                    history_length=self.trainer_config.history_length,
                    formation_features=sample.formation_features,
                    pattern_label=sample.pattern_label,
                    trajectory_label=sample.trajectory_label,
                    experience_label=sample.experience_label,
                    **self._memory_read_kwargs(sample),
                )
                intervention_embedding = self._project_intervention(
                    normalized_intervention,
                    normalized_intervention_sequence,
                )
                fused_representation, direct_memory_residual, planner_tensor, confidence_tensor, scenario_tensor, gate_summary = self._build_fused_representation(
                    factual_embedding=factual_embedding,
                    intervention_embedding=intervention_embedding,
                    normalized_formation=normalized_formation,
                    manager_result=manager_result,
                )
                residual_prediction, bucket_summary = self._bucket_residual(fused_representation)
                coordinated_memory_residual, coordination_summary, _ = self._coordinate_memory_paths(
                    residual_prediction=residual_prediction,
                    direct_memory_residual=direct_memory_residual,
                    planner_tensor=planner_tensor,
                    confidence_tensor=confidence_tensor,
                    scenario_tensor=scenario_tensor,
                    gate_summary=gate_summary,
                )
                gate_summary.update(coordination_summary)
                factual_prediction_input, factual_prediction_gate, factual_prediction_strength = self._augment_factual_prediction_input(
                    base_input=torch.cat([factual_embedding, intervention_embedding], dim=-1),
                    temporal_profile=temporal_profile,
                    normalized_formation=normalized_formation,
                )
                predicted_factual_scale = self._predict_factual_scale(factual_prediction_input)
                raw_base_prediction = self.base_regressor(factual_prediction_input)
                gate_summary["_sample_pattern"] = (
                    PATTERN_LABELS[int(sample.pattern_label)]
                    if 0 <= int(sample.pattern_label) < len(PATTERN_LABELS)
                    else ""
                )
                gate_summary["_sample_trajectory"] = (
                    TRAJECTORY_LABELS[int(sample.trajectory_label)]
                    if 0 <= int(sample.trajectory_label) < len(TRAJECTORY_LABELS)
                    else ""
                )
                _, harm_summary, _ = self._apply_memory_harm_control(
                    base_prediction=raw_base_prediction,
                    coordinated_memory_residual=coordinated_memory_residual,
                    gate_summary=gate_summary,
                )
                gate_summary.update(
                    {
                        "factual_branch_gate": float(factual_branch_gate.item()),
                        "factual_branch_strength": float(factual_branch_strength.item()),
                        "retrieval_branch_gate": float(retrieval_branch_gate.item()),
                        "retrieval_branch_strength": float(retrieval_branch_strength.item()),
                        "factual_prediction_gate": float(factual_prediction_gate.item()),
                        "factual_prediction_strength": float(factual_prediction_strength.item()),
                        "factual_predicted_scale_mean": float(predicted_factual_scale.mean().item()),
                        "temporal_factual_gate": float(temporal_gate.item()),
                        "temporal_factual_strength": float(temporal_strength.item()),
                    }
                )
                gate_summary.update(harm_summary)
                for key in gate_sums:
                    gate_sums[key] += gate_summary[key]
                for key in direct_gate_sums:
                    direct_gate_sums[key] += gate_summary[key]
                for key, value in manager_result.planner_weights.items():
                    planner_sums[key] += float(value)
                confidence_sums["pattern"] += float(confidence_tensor[0].item())
                confidence_sums["trajectory"] += float(confidence_tensor[1].item())
                confidence_sums["experience"] += float(confidence_tensor[2].item())
                for key in bucket_sums:
                    bucket_sums[key] += bucket_summary[key]
                if manager_result.component_results["experience"].archive_used:
                    archive_use_count += 1.0
                retrieval_source_counts[manager_result.component_results["experience"].retrieval_source] = (
                    retrieval_source_counts.get(manager_result.component_results["experience"].retrieval_source, 0.0) + 1.0
                )
                if int(manager_result.semantic_result.get("hit_count", 0)) > 0:
                    semantic_hit_count += 1.0
                    semantic_top_score_sum += float(manager_result.semantic_result.get("top_score", 0.0))
                    semantic_support_sum += float(manager_result.semantic_result.get("support_sum", 0.0))
                    semantic_template_confidence_sum += float(manager_result.semantic_result.get("template_confidence", 0.0))
                    semantic_template_blend_sum += float(manager_result.semantic_result.get("template_blend_weight", 0.0))
                    top_label = str(int(manager_result.semantic_result.get("top_experience_label", -1)))
                    semantic_label_counts[top_label] = semantic_label_counts.get(top_label, 0.0) + 1.0
                count += 1

        component_diagnostics = self.memory_manager.summarize()
        component_diagnostics["gate_means"] = {key: value / max(1, count) for key, value in gate_sums.items()}
        component_diagnostics["direct_memory_means"] = {
            key: value / max(1, count) for key, value in direct_gate_sums.items()
        }
        component_diagnostics["planner_means"] = {key: value / max(1, count) for key, value in planner_sums.items()}
        component_diagnostics["confidence_means"] = {
            key: value / max(1, count) for key, value in confidence_sums.items()
        }
        component_diagnostics["bucket_means"] = {key: value / max(1, count) for key, value in bucket_sums.items()}
        component_diagnostics["experience_archive_use_rate"] = archive_use_count / max(1, count)
        component_diagnostics["experience_retrieval_source_rate"] = {
            key: value / max(1, count) for key, value in retrieval_source_counts.items()
        }
        component_diagnostics["semantic_retrieval"] = {
            "enabled": bool(self.memory_manager.semantic_store is not None),
            "hit_rate": semantic_hit_count / max(1, count),
            "mean_top_score": semantic_top_score_sum / max(1, semantic_hit_count),
            "mean_support_sum": semantic_support_sum / max(1, semantic_hit_count),
            "mean_template_confidence": semantic_template_confidence_sum / max(1, semantic_hit_count),
            "mean_template_blend_weight": semantic_template_blend_sum / max(1, semantic_hit_count),
            "top_label_rate": {
                key: value / max(1, semantic_hit_count) for key, value in semantic_label_counts.items()
            },
        }
        component_diagnostics["bucket_layout"] = {
            "short_steps": float(self.bucket_masks[0].sum().item()),
            "mid_steps": float(self.bucket_masks[1].sum().item()),
            "long_steps": float(self.bucket_masks[2].sum().item()),
        }
        return component_diagnostics

    def audit_factual_memory_path(
        self,
        samples: Sequence[ForecastSample],
        max_samples: int = 0,
    ) -> Dict[str, object]:
        audited_samples = list(samples[:max_samples]) if max_samples and max_samples > 0 else list(samples)
        if not audited_samples:
            return {
                "sample_count": 0.0,
                "gate_means": {},
                "direct_memory_means": {},
                "transition_means": {},
                "retrieval_source_rate": {},
                "semantic_retrieval": {},
            }

        gate_sums = {"pattern_gate": 0.0, "trajectory_gate": 0.0, "experience_gate": 0.0}
        direct_sums = {
            "direct_memory_strength": 0.0,
            "direct_residual_scale": 0.0,
            "coordinated_memory_strength": 0.0,
            "path_alignment": 0.0,
            "path_disagreement": 0.0,
            "kg_gate": 0.0,
            "kg_residual_strength": 0.0,
            "kg_feature_density": 0.0,
            "kg_guideline_alignment": 0.0,
            "kg_state_load": 0.0,
            "kg_care_load": 0.0,
            "memory_harm_control_scale": 0.0,
            "memory_harm_confidence_scale": 0.0,
            "memory_harm_quality": 0.0,
            "memory_harm_alignment_scale": 0.0,
            "memory_harm_cap_scale": 0.0,
            "memory_harm_pre_strength": 0.0,
            "memory_harm_post_strength": 0.0,
        }
        transition_sums = {
            "transition_confidence": 0.0,
            "transition_expected_utility": 0.0,
            "transition_top_score": 0.0,
            "transition_residual_scale": 0.0,
            "transition_alignment": 0.0,
            "transition_gate_scale": 0.0,
            "transition_residual_strength": 0.0,
            "base_to_fusion_delta_strength": 0.0,
            "transition_safe_gate": 0.0,
            "transition_structural_blocked": 0.0,
            "transition_gate_blocked": 0.0,
            "transition_utility_factor": 0.0,
            "transition_pattern_factor": 0.0,
            "transition_trajectory_factor": 0.0,
            "transition_residual_cap": 0.0,
            "transition_residual_cap_applied": 0.0,
            "transition_support_strength": 0.0,
            "transition_signature_weight": 0.0,
        }
        transition_gate_blocked_reasons: Dict[str, float] = {}
        transition_case_details: List[Dict[str, object]] = []
        retrieval_source_counts: Dict[str, float] = {}
        archive_use_count = 0.0
        semantic_hit_count = 0.0
        semantic_top_score_sum = 0.0
        semantic_template_blend_sum = 0.0
        count = 0

        self.eval()
        with torch.no_grad():
            batch_size = max(1, self.trainer_config.batch_size)
            for start in range(0, len(audited_samples), batch_size):
                batch_samples = audited_samples[start : start + batch_size]
                _, manager_results, base_preds, fusion_preds, gate_summaries, _ = self._forward_batch(batch_samples)
                for sample_index, manager_result in enumerate(manager_results):
                    gate_summary = gate_summaries[sample_index]
                    for key in gate_sums:
                        gate_sums[key] += float(gate_summary.get(key, 0.0))
                    for key in direct_sums:
                        direct_sums[key] += float(gate_summary.get(key, 0.0))
                    for key in transition_sums:
                        if key == "base_to_fusion_delta_strength":
                            delta_strength = float((fusion_preds[sample_index] - base_preds[sample_index]).abs().mean().item())
                            transition_sums[key] += delta_strength
                        else:
                            transition_sums[key] += float(gate_summary.get(key, 0.0))
                    blocked_reasons = str(gate_summary.get("transition_gate_reasons", ""))
                    if blocked_reasons:
                        for reason in blocked_reasons.split("|"):
                            reason = reason.strip()
                            if reason:
                                transition_gate_blocked_reasons[reason] = transition_gate_blocked_reasons.get(reason, 0.0) + 1.0
                    sem = manager_result.semantic_result
                    retrieval_trace = {
                        "semantic_hit_count": int(sem.get("hit_count", 0)),
                        "prototype_ids": list(sem.get("prototype_ids", [])),
                        "prototype_labels": list(sem.get("prototype_labels", [])),
                        "top_score": float(sem.get("top_score", 0.0)),
                        "template_blend_weight": float(sem.get("template_blend_weight", 0.0)),
                        "template_confidence": float(sem.get("template_confidence", 0.0)),
                        "direction_alignment": float(sem.get("direction_alignment", 1.0)),
                        "entropy_penalty": float(sem.get("entropy_penalty", 1.0)),
                        "direction_hits": int(sem.get("direction_hits", 0)),
                        "mean_outcome_entropy": float(sem.get("mean_outcome_entropy", 0.0)),
                        "experience_archive_used": bool(manager_result.component_results["experience"].archive_used),
                        "retrieval_source": str(manager_result.component_results["experience"].retrieval_source or "unknown"),
                    }
                    transition_case_details.append(
                        {
                            "case_index": float(count),
                            "series_name": str(getattr(batch_samples[sample_index], "series_name", f"case_{count}")),
                            "pattern": str(PATTERN_LABELS[int(getattr(batch_samples[sample_index], "pattern_label", -1))]) if 0 <= int(getattr(batch_samples[sample_index], "pattern_label", -1)) < len(PATTERN_LABELS) else "",
                            "trajectory": str(TRAJECTORY_LABELS[int(getattr(batch_samples[sample_index], "trajectory_label", -1))]) if 0 <= int(getattr(batch_samples[sample_index], "trajectory_label", -1)) < len(TRAJECTORY_LABELS) else "",
                            "transition_safe_gate": float(gate_summary.get("transition_safe_gate", 0.0)),
                            "transition_gate_blocked": float(gate_summary.get("transition_gate_blocked", 0.0)),
                            "transition_gate_reasons": str(gate_summary.get("transition_gate_reasons", "")),
                            "transition_gate_scale": float(gate_summary.get("transition_gate_scale", 0.0)),
                            "transition_confidence": float(gate_summary.get("transition_confidence", 0.0)),
                            "transition_expected_utility": float(gate_summary.get("transition_expected_utility", 0.0)),
                            "transition_support_strength": float(gate_summary.get("transition_support_strength", 0.0)),
                            "transition_signature_weight": float(gate_summary.get("transition_signature_weight", 0.0)),
                            "transition_structural_blocked": float(gate_summary.get("transition_structural_blocked", 0.0)),
                            "transition_utility_factor": float(gate_summary.get("transition_utility_factor", 1.0)),
                            "transition_pattern_factor": float(gate_summary.get("transition_pattern_factor", 1.0)),
                            "transition_trajectory_factor": float(gate_summary.get("transition_trajectory_factor", 1.0)),
                            "transition_residual_strength": float(gate_summary.get("transition_residual_strength", 0.0)),
                            "transition_residual_cap": float(gate_summary.get("transition_residual_cap", 0.0)),
                            "base_to_fusion_delta": float((fusion_preds[sample_index] - base_preds[sample_index]).abs().mean().item()),
                            "retrieval_trace": retrieval_trace,
                        }
                    )
                    retrieval_source = str(manager_result.component_results["experience"].retrieval_source or "unknown")
                    retrieval_source_counts[retrieval_source] = retrieval_source_counts.get(retrieval_source, 0.0) + 1.0
                    if manager_result.component_results["experience"].archive_used:
                        archive_use_count += 1.0
                    if int(manager_result.semantic_result.get("hit_count", 0)) > 0:
                        semantic_hit_count += 1.0
                        semantic_top_score_sum += float(manager_result.semantic_result.get("top_score", 0.0))
                        semantic_template_blend_sum += float(manager_result.semantic_result.get("template_blend_weight", 0.0))
                    count += 1

        return {
            "sample_count": float(count),
            "gate_means": {key: value / max(1, count) for key, value in gate_sums.items()},
            "direct_memory_means": {key: value / max(1, count) for key, value in direct_sums.items()},
            "transition_means": {key: value / max(1, count) for key, value in transition_sums.items()},
            "transition_gate_blocked_reasons": transition_gate_blocked_reasons,
            "transition_gate_blocked_rate": float(transition_sums["transition_gate_blocked"] / max(1, count)),
            "transition_case_details": transition_case_details,
            "retrieval_source_rate": {key: value / max(1, count) for key, value in retrieval_source_counts.items()},
            "experience_archive_use_rate": archive_use_count / max(1, count),
            "semantic_retrieval": {
                "hit_rate": semantic_hit_count / max(1, count),
                "mean_top_score": semantic_top_score_sum / max(1, semantic_hit_count),
                "mean_template_blend_weight": semantic_template_blend_sum / max(1, semantic_hit_count),
            },
        }

    def evaluate_factual_calibration(self, samples: Sequence[ForecastSample]) -> Dict[str, float]:
        if not samples:
            return {
                "sample_count": 0.0,
                "mean_predicted_scale": 0.0,
                "mean_absolute_error": 0.0,
                "calibration_gap": 0.0,
                "interval_95_coverage": 0.0,
            }
        predicted_scales: List[float] = []
        absolute_errors: List[float] = []
        covered_steps = 0.0
        total_steps = 0.0
        self.eval()
        with torch.no_grad():
            batch_size = max(1, self.trainer_config.batch_size)
            for start in range(0, len(samples), batch_size):
                batch_samples = list(samples[start : start + batch_size])
                _, _, base_preds, _, gate_summaries, _ = self._forward_batch(batch_samples)
                for sample, prediction_row, gate_summary in zip(batch_samples, base_preds.detach().cpu().tolist(), gate_summaries):
                    scale_tensor = gate_summary.get("_predicted_factual_scale_tensor")
                    if scale_tensor is None:
                        continue
                    scale_row = torch.as_tensor(scale_tensor, dtype=torch.float32).detach().cpu().tolist()
                    restored_prediction = [float(value) * sample.scale_value + sample.scale_center for value in prediction_row]
                    restored_scale = [float(value) * sample.scale_value for value in scale_row]
                    for truth_value, pred_value, scale_value in zip(sample.raw_target, restored_prediction, restored_scale):
                        interval_radius = 1.96 * max(1e-6, float(scale_value))
                        predicted_scales.append(float(scale_value))
                        absolute_errors.append(abs(float(pred_value) - float(truth_value)))
                        covered_steps += 1.0 if abs(float(pred_value) - float(truth_value)) <= interval_radius else 0.0
                        total_steps += 1.0
        mean_scale = float(sum(predicted_scales) / max(1, len(predicted_scales)))
        mean_abs_error = float(sum(absolute_errors) / max(1, len(absolute_errors)))
        return {
            "sample_count": float(len(samples)),
            "mean_predicted_scale": mean_scale,
            "mean_absolute_error": mean_abs_error,
            "calibration_gap": float(abs(mean_scale - mean_abs_error)),
            "interval_95_coverage": float(covered_steps / max(1.0, total_steps)),
        }

    def evaluate(self, samples: Sequence[ForecastSample], use_memory: bool = True) -> Dict[str, float]:
        predictions = self.predict(samples, use_memory=use_memory)
        truth = [sample.raw_target for sample in samples]
        insample = [
            [step[0] * sample.scale_value + sample.scale_center for step in sample.sequence]
            for sample in samples
        ]
        from src.forecasting_metrics import forecasting_metrics

        return forecasting_metrics(truth, predictions, insample, seasonality=self.trainer_config.seasonality)
