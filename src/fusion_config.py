#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fused System Configuration — bridges LLM-driven SOFA prediction with
the manifold memory bank, KG integration, and persistent experience store.

Architecture:
    Patient Data → LLM Encoder → Memory Bank (Pattern/Trajectory/Experience)
                   ↘ LLM Predictor → LLM Counterfactual → LLM Supervisor → Output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LLMEncoderConfig:
    """Configuration for the LLM-based patient state encoder."""

    # Ollama connection
    ollama_base_url: str = "http://localhost:11434"
    encoder_model: str = "gemma3:12b"

    # Output dimensions — must match ManifoldMemoryConfig
    manifold_dim: int = 32
    value_dim: int = 48
    context_dim: int = 256  # total context = 3 * seq_context + static_context

    # Sequence encoding (from formation features + time steps)
    sequence_feature_dim: int = 16  # formation feature count
    sequence_context_dim: int = 64  # GRU hidden dim equivalent

    # Static encoding (KG + patient demographics)
    static_feature_dim: int = 56  # KG flags + static patient info
    static_context_dim: int = 16

    # LLM parameters
    temperature: float = 0.3
    max_tokens: int = 2048

    # Projection network
    projection_hidden_dim: int = 128
    use_projection_norm: bool = True

    device: str = "cpu"
    seed: int = 42


@dataclass
class LLMPredictorConfig:
    """Configuration for the LLM-based SOFA predictor."""

    ollama_base_url: str = "http://localhost:11434"
    predictor_model: str = "deepseek-r1:32b"  # primary reasoning model

    # Multi-model ensemble
    ensemble_models: List[str] = field(default_factory=lambda: [
        "deepseek-r1:32b",
        "gemma3:12b",
        "qwen3:30b",
    ])

    temperature: float = 0.3
    max_tokens: int = 4096
    forecast_horizon: int = 8  # hours ahead

    # Uncertainty estimation
    num_samples: int = 3  # MC samples for uncertainty
    uncertainty_method: str = "multi_model"  # "multi_model" | "temperature" | "none"


@dataclass
class LLMCounterfactualConfig:
    """Configuration for the LLM-based counterfactual intervention generator."""

    ollama_base_url: str = "http://localhost:11434"
    generator_model: str = "deepseek-r1:32b"

    temperature: float = 0.4
    max_tokens: int = 4096

    # Candidate generation
    max_candidates: int = 5
    candidate_diversity_threshold: float = 0.3

    # KG repair integration
    enable_kg_repair: bool = True
    enable_template_search: bool = True


@dataclass
class LLMSupervisorConfig:
    """Configuration for the LLM Clinical Supervisor."""

    ollama_base_url: str = "http://localhost:11434"
    supervisor_model: str = "gemma3:12b"  # lighter model for gating

    temperature: float = 0.0  # deterministic for clinical decisions
    max_tokens: int = 1024

    # Gate thresholds
    memory_gate_confidence_threshold: float = 0.6
    retrieval_rerank_top_k: int = 5
    utility_assessment_threshold: float = 0.3

    # Decision flags
    enable_memory_gate: bool = True
    enable_retrieval_filter: bool = True
    enable_utility_assessment: bool = True


@dataclass
class FusedSystemConfig:
    """Master configuration for the fused system."""

    # Sub-system configs
    encoder: LLMEncoderConfig = field(default_factory=LLMEncoderConfig)
    predictor: LLMPredictorConfig = field(default_factory=LLMPredictorConfig)
    counterfactual: LLMCounterfactualConfig = field(default_factory=LLMCounterfactualConfig)
    supervisor: LLMSupervisorConfig = field(default_factory=LLMSupervisorConfig)

    # KG integration
    kg_directory: str = ""
    kg_mapping_path: str = ""

    # Persistent experience store
    persist_directory: str = ""
    enable_persistence: bool = True

    # Memory bank
    max_memory: int = 256
    dataset_name: str = "fused_eicu"
    series_count: int = 16
    seasonality: int = 1

    # Runtime
    device: str = "cpu"
    seed: int = 42
    verbose: bool = True


@dataclass
class ClinicalStateSignature:
    """Structured clinical state extracted by the LLM encoder."""

    severity_bin: str = "moderate"       # low / moderate / high / critical
    level_bin: str = "near_baseline"     # below / near / above / far_above baseline
    trend_bin: str = "stable"           # improving / stable / worsening
    volatility_bin: str = "quiet"       # quiet / variable / unstable
    trajectory_label: str = "stable_regime"
    pattern_label: str = "flat"

    sofa_current: float = 0.0
    sofa_trend_6h: float = 0.0
    baseline_mae_history: float = 0.0

    kg_flags: Dict[str, float] = field(default_factory=dict)
    kg_guideline_alignment: float = 0.0

    # Numerical feature vectors
    formation_features: List[float] = field(default_factory=list)
    kg_features: List[float] = field(default_factory=list)
    patient_static: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "severity": self.severity_bin,
            "level": self.level_bin,
            "trend": self.trend_bin,
            "volatility": self.volatility_bin,
            "trajectory": self.trajectory_label,
            "pattern": self.pattern_label,
            "sofa_current": self.sofa_current,
            "sofa_trend_6h": self.sofa_trend_6h,
            "kg_flags": {k: v for k, v in self.kg_flags.items() if v > 0},
            "guideline_alignment": self.kg_guideline_alignment,
        }


@dataclass
class FusedPredictionOutput:
    """Complete output from the fused prediction pipeline."""

    patient_id: str = ""
    model_name: str = ""

    # SOFA predictions
    hourly_sofa_totals: Dict[str, float] = field(default_factory=dict)
    sofa_scores: Dict[str, float] = field(default_factory=dict)
    sofa_scores_series: Dict[str, List[float]] = field(default_factory=dict)

    # Risk assessment
    risk_level: str = "unknown"
    risk_confidence: float = 0.0
    reasoning: str = ""

    # Memory context
    memory_confidence: float = 0.0
    memory_matched_count: int = 0
    memory_matched_labels: List[int] = field(default_factory=list)
    template_blend_weight: float = 0.0
    semantic_hit_count: int = 0

    # Clinical state
    clinical_state: Optional[ClinicalStateSignature] = None

    # Supervisor decisions
    supervisor_gate_decision: bool = True
    supervisor_gate_reason: str = ""
    supervisor_filtered_count: int = 0
    supervisor_utility_score: float = 0.0

    # Counterfactual (if generated)
    counterfactual_candidates: List[Dict[str, object]] = field(default_factory=list)
    best_alternative: Optional[Dict[str, object]] = None

    # Uncertainty
    prediction_uncertainty: float = 0.0
    prediction_std: float = 0.0

    # KG integration
    kg_features: Dict[str, float] = field(default_factory=dict)
    guideline_alignment: float = 0.0

    # Metadata
    metadata: Dict[str, object] = field(default_factory=dict)
    timestamp: str = ""
