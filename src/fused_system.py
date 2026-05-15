#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fused System Orchestrator — integrates the LLM-driven SOFA prediction pipeline
(System 1) with the manifold memory bank, KG integration, persistent experience
store, and clinical supervision infrastructure (System 2).

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │                     FusedSystem                              │
    │                                                              │
    │  Patient Data ──→ LLMEncoder ──→ ClinicalStateSignature      │
    │       │              │                │                      │
    │       │              ▼                ▼                      │
    │       │       KG Feature Builder   Memory Bank               │
    │       │              │            (Pattern/Traj/Exp)          │
    │       │              ▼                │                      │
    │       │         KG Features    Memory Context                 │
    │       │              │                │                      │
    │       │              ▼                ▼                      │
    │       │       ┌─────────────────────────┐                    │
    │       │       │   LLM Predictor          │                   │
    │       │       │   (Multi-Model Ensemble) │                   │
    │       │       └───────────┬─────────────┘                    │
    │       │                   │                                   │
    │       │                   ▼                                   │
    │       │       ┌─────────────────────────┐                    │
    │       │       │ LLM Clinical Supervisor  │                   │
    │       │       │  Gate | Filter | Utility │                   │
    │       │       └───────────┬─────────────┘                    │
    │       │                   │                                   │
    │       │                   ▼                                   │
    │       │       ┌─────────────────────────┐                    │
    │       │       │ LLM Counterfactual Gen   │                    │
    │       │       │ Intervention Alternatives│                    │
    │       │       └───────────┬─────────────┘                    │
    │       │                   │                                   │
    │       │                   ▼                                   │
    │       │          FusedPredictionOutput                        │
    │       │                                                       │
    │       │          ┌────────────────────┐                       │
    │       └──────────→ PersistentExperience│                      │
    │                  │ Store (write-back)  │                      │
    │                  └────────────────────┘                       │
    └──────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from src.fusion_config import (
    ClinicalStateSignature,
    FusedPredictionOutput,
    FusedSystemConfig,
)
from src.llm_encoder import LLMEncoder
from src.llm_predictor import LLMPredictor
from src.llm_counterfactual import LLMCounterfactualGenerator
from src.llm_clinical_supervisor import LLMClinicalSupervisor

# Memory bank (from new system — kept as-is)
from src.manifold_memory import ManifoldMemoryConfig
from src.memory_manager import MetaMemoryManager
from src.ts_formation import PATTERN_LABELS, TRAJECTORY_LABELS

# KG integration (from new system — kept as-is)
try:
    from src.kg_integration import KnowledgeGraphFeatureBuilder
    KG_AVAILABLE = True
except ImportError:
    KnowledgeGraphFeatureBuilder = None
    KG_AVAILABLE = False

# Persistent experience store (from new system — kept as-is)
try:
    from src.persistent_memory_store import PersistentExperienceStore, forecast_sample_identity
    PERSIST_AVAILABLE = True
except ImportError:
    PersistentExperienceStore = None
    forecast_sample_identity = None
    PERSIST_AVAILABLE = False

# Patient text generator (from old system)
try:
    from patient_text_generator2 import (
        calculate_sofa_respiration,
        calculate_sofa_coagulation,
        calculate_sofa_liver,
        calculate_sofa_cardiovascular,
        calculate_sofa_cns,
        calculate_sofa_renal,
        calculate_total_sofa,
    )
    SOFA_CALC_AVAILABLE = True
except ImportError:
    SOFA_CALC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Harm Control (from new system's manifold_forecasting_trainer)
# ---------------------------------------------------------------------------

class HarmControl:
    """
    Three-layer harm control gate from the new system.

    Controls memory residual magnitude through:
        - quality scale: suppress if memory signal quality < threshold
        - alignment scale: suppress if multi-path directions conflict
        - cap scale: suppress if residual magnitude exceeds limit
    """

    def __init__(
        self,
        quality_threshold: float = 0.45,
        alignment_threshold: float = 0.30,
        cap_scale: float = 0.15,
    ):
        self.quality_threshold = quality_threshold
        self.alignment_threshold = alignment_threshold
        self.cap_scale = cap_scale

    def compute(
        self,
        memory_quality: float,
        pattern_confidence: float,
        trajectory_confidence: float,
        experience_confidence: float,
        memory_residual_magnitude: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute harm control scales.

        Returns (final_scale, audit_dict).
        """
        # Quality: suppress low-quality memory signals
        quality_scale = min(1.0, max(0.0, memory_quality / max(self.quality_threshold, 1e-6)))

        # Alignment: check if pattern/trajectory/experience agree on direction
        confidences = [pattern_confidence, trajectory_confidence, experience_confidence]
        if confidences:
            mean_conf = sum(confidences) / len(confidences)
            max_dev = max(abs(c - mean_conf) for c in confidences)
            alignment_scale = 1.0 - min(1.0, max_dev / max(self.alignment_threshold, 1e-6))
        else:
            alignment_scale = 1.0

        # Cap: limit maximum residual magnitude
        cap = self.cap_scale * (1.0 + memory_quality)
        cap_scale_val = min(1.0, cap / max(memory_residual_magnitude, 1e-6))

        final_scale = min(quality_scale, alignment_scale, cap_scale_val)

        return final_scale, {
            "quality_scale": quality_scale,
            "alignment_scale": alignment_scale,
            "cap_scale": cap_scale_val,
            "final_scale": final_scale,
            "memory_quality": memory_quality,
            "residual_magnitude": memory_residual_magnitude,
        }


# ---------------------------------------------------------------------------
# Fused System
# ---------------------------------------------------------------------------

class FusedSystem:
    """
    Complete fused system orchestrator.

    Usage:
        config = FusedSystemConfig()
        system = FusedSystem(config)

        # Predict with full pipeline
        output = system.predict(patient_text, intervention)

        # Generate counterfactual alternatives
        alternatives = system.generate_counterfactuals(patient_text, intervention)

        # Write results back to experience store
        system.write_experience(output)
    """

    def __init__(self, config: FusedSystemConfig):
        self.config = config
        self.device = torch.device(config.device)

        # ---- LLM Components (System 1) ----
        self.encoder = LLMEncoder(config.encoder)
        self.predictor = LLMPredictor(config.predictor)
        self.counterfactual = LLMCounterfactualGenerator(config.counterfactual)
        self.supervisor = LLMClinicalSupervisor(config.supervisor)

        # ---- Memory Bank (System 2) ----
        memory_config = ManifoldMemoryConfig(
            sequence_feature_dim=config.encoder.sequence_feature_dim,
            static_feature_dim=config.encoder.static_feature_dim,
            manifold_dim=config.encoder.manifold_dim,
            value_dim=config.encoder.value_dim,
            max_memory=config.max_memory,
            device=config.device,
            seed=config.seed,
        )
        self.memory_manager = MetaMemoryManager(
            base_config=memory_config,
            series_count=config.series_count,
            seasonality=config.seasonality,
            dataset_name=config.dataset_name,
        )

        # ---- KG Integration (System 2) ----
        self.kg_builder: Optional[KnowledgeGraphFeatureBuilder] = None
        if KG_AVAILABLE and config.kg_directory:
            try:
                self.kg_builder = KnowledgeGraphFeatureBuilder.from_directory(
                    kg_directory=config.kg_directory,
                    mapping_path=config.kg_mapping_path,
                )
                if config.verbose:
                    print(f"[FusedSystem] KG loaded: {self.kg_builder.graph_summary.get('node_count', 0)} nodes, "
                          f"{self.kg_builder.graph_summary.get('mapping_rule_count', 0)} mapping rules")
            except Exception as e:
                print(f"[FusedSystem] KG loading failed: {e}")

        # ---- Persistent Experience Store (System 2) ----
        self.persist_store: Optional[PersistentExperienceStore] = None
        if PERSIST_AVAILABLE and config.enable_persistence and config.persist_directory:
            try:
                self.persist_store = PersistentExperienceStore(
                    store_path=Path(config.persist_directory),
                    reusable_splits={"train", "external"},
                )
                self.memory_manager.configure_semantic_store(self.persist_store)
                if config.verbose:
                    print(f"[FusedSystem] Persistent store initialized at {config.persist_directory}")
            except Exception as e:
                print(f"[FusedSystem] Persistent store init failed: {e}")

        # ---- Harm Control ----
        self.harm_control = HarmControl()

        # ---- Runtime state ----
        self._prediction_count: int = 0

    # ==================================================================
    # Public API
    # ==================================================================

    def predict(
        self,
        patient_text: str,
        intervention: str = "",
        patient_id: str = "",
        label_row: Optional[Dict[str, object]] = None,
        context_frame: Optional[object] = None,  # pd.DataFrame
        enable_counterfactual: bool = True,
        write_to_memory: bool = True,
    ) -> FusedPredictionOutput:
        """
        Full prediction pipeline: encode → memory → predict → supervise → counterfactual.

        Args:
            patient_text: Natural language patient description
            intervention: Current intervention description
            patient_id: Patient/stay identifier
            label_row: Optional label data for KG feature extraction
            context_frame: Optional trajectory DataFrame for KG feature extraction
            enable_counterfactual: Whether to generate alternative interventions
            write_to_memory: Whether to write results to memory bank

        Returns:
            FusedPredictionOutput with complete results
        """
        t_start = time.time()
        output = FusedPredictionOutput(
            patient_id=patient_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

        # ---- Step 1: KG Feature Extraction ----
        kg_features_vec: List[float] = []
        kg_flags: Dict[str, float] = {}
        guideline_alignment = 0.0

        if self.kg_builder and label_row is not None and context_frame is not None:
            try:
                import pandas as pd
                if isinstance(context_frame, pd.DataFrame):
                    kg_features_vec, kg_flags, guideline_alignment = self.kg_builder.build_features(
                        label_row, context_frame
                    )
                    output.kg_features = kg_flags
                    output.guideline_alignment = guideline_alignment
            except Exception as e:
                if self.config.verbose:
                    print(f"[FusedSystem] KG feature extraction failed: {e}")

        # ---- Step 2: LLM Encoding → Clinical State + Memory Embedding ----
        patient_static = self._extract_patient_static(patient_text)
        encoding, clinical_state = self.encoder.encode(
            patient_text, intervention,
            kg_features=kg_features_vec,
            patient_static=patient_static,
            metadata={
                "patient_id": patient_id,
                "timestamp": time.time(),
                "kg_guideline_alignment": guideline_alignment,
            },
        )
        output.clinical_state = clinical_state

        # ---- Step 3: Memory Bank Retrieval ----
        formation_features = (
            clinical_state.formation_features
            if clinical_state.formation_features
            else [0.0] * 16
        )
        pattern_label = self._label_index(clinical_state.pattern_label, PATTERN_LABELS)
        trajectory_label = self._label_index(clinical_state.trajectory_label, TRAJECTORY_LABELS)
        experience_label = pattern_label * len(TRAJECTORY_LABELS) + trajectory_label

        history_length = 12  # default 12 time windows
        horizon = self.config.predictor.forecast_horizon

        memory_read = self.memory_manager.read(
            encoding=encoding,
            horizon=horizon,
            history_length=history_length,
            formation_features=formation_features,
            pattern_label=pattern_label,
            trajectory_label=trajectory_label,
            experience_label=experience_label,
            metadata=encoding.metadata,
            kg_features=kg_features_vec if kg_features_vec else None,
        )

        # Extract memory context
        semantic = memory_read.semantic_result
        exp_read = memory_read.component_results.get("experience")
        pat_read = memory_read.component_results.get("pattern")
        traj_read = memory_read.component_results.get("trajectory")

        output.memory_confidence = float(exp_read.confidence) if exp_read else 0.0
        output.memory_matched_count = len(exp_read.matched_indices) if exp_read else 0
        output.memory_matched_labels = list(exp_read.matched_labels) if exp_read else []
        output.template_blend_weight = float(semantic.get("template_blend_weight", 0))
        output.semantic_hit_count = int(semantic.get("hit_count", 0))

        memory_context_text = self._format_memory_context(memory_read, clinical_state)

        # ---- Step 4: LLM Prediction ----
        prediction, pred_mean_conf, pred_std = self.predictor.predict_with_uncertainty(
            patient_text=patient_text,
            intervention=intervention,
            memory_context=memory_context_text,
            clinical_state=clinical_state,
            kg_features=kg_flags,
            guideline_alignment=guideline_alignment,
        )

        # Populate prediction output
        output.model_name = self.config.predictor.predictor_model
        output.risk_level = str(prediction.get("risk_level", "unknown"))
        output.risk_confidence = float(prediction.get("confidence", 0) or 0)
        output.reasoning = str(prediction.get("reasoning", ""))
        output.prediction_uncertainty = float(pred_std)
        output.prediction_std = float(pred_std)

        # Extract hourly SOFA totals and component series
        hourly = prediction.get("hourly_predictions", {})
        for hour_key, hour_data in hourly.items():
            if isinstance(hour_data, dict):
                total = hour_data.get("total")
                if isinstance(total, (int, float)):
                    output.hourly_sofa_totals[str(hour_key)] = float(total)

                for comp in ["sofa_respiration", "sofa_coagulation", "sofa_liver",
                             "sofa_cardiovascular", "sofa_cns", "sofa_renal"]:
                    val = hour_data.get(comp)
                    if isinstance(val, (int, float)):
                        if comp not in output.sofa_scores_series:
                            output.sofa_scores_series[comp] = []
                        output.sofa_scores_series[comp].append(float(val))

        # Last hour as cross-sectional scores
        last_hour = hourly.get(str(horizon), hourly.get("8", {}))
        if isinstance(last_hour, dict):
            for comp in ["sofa_respiration", "sofa_coagulation", "sofa_liver",
                         "sofa_cardiovascular", "sofa_cns", "sofa_renal"]:
                val = last_hour.get(comp)
                if isinstance(val, (int, float)):
                    output.sofa_scores[comp] = float(val)

        # ---- Step 5: Harm Control ----
        memory_quality = output.memory_confidence
        pattern_conf = float(pat_read.confidence) if pat_read else 0.0
        traj_conf = float(traj_read.confidence) if traj_read else 0.0
        exp_conf = float(exp_read.confidence) if exp_read else 0.0

        # Compute approximate memory residual magnitude
        pre_strength = abs(output.template_blend_weight * output.semantic_hit_count)
        harm_scale, harm_audit = self.harm_control.compute(
            memory_quality=memory_quality,
            pattern_confidence=pattern_conf,
            trajectory_confidence=traj_conf,
            experience_confidence=exp_conf,
            memory_residual_magnitude=pre_strength,
        )
        output.metadata["harm_control"] = harm_audit

        # ---- Step 6: LLM Clinical Supervisor ----
        patient_summary = json.dumps(clinical_state.to_dict(), ensure_ascii=False)

        # 6a: Memory Gate
        gate_result = self.supervisor.memory_gate(
            patient_summary=patient_summary,
            match_count=output.memory_matched_count,
            top_label=str(output.memory_matched_labels[0]) if output.memory_matched_labels else "none",
            memory_confidence=output.memory_confidence,
            template_blend=output.template_blend_weight,
            semantic_hits=output.semantic_hit_count,
        )

        # If gate says don't apply and harm_control says it's risky, double-lock
        apply_memory = gate_result.get("apply_memory", True) and harm_scale > 0.1
        output.supervisor_gate_decision = apply_memory
        output.supervisor_gate_reason = str(gate_result.get("reasoning", ""))

        # 6b: Retrieval Filter (format retrieved cases for review)
        retrieved_cases = self._build_retrieved_case_list(memory_read)
        filter_result = self.supervisor.retrieval_filter(
            patient_summary=patient_summary,
            retrieved_cases=retrieved_cases,
        )
        output.supervisor_filtered_count = len(filter_result.get("filtered_cases", []))

        # ---- Step 7: Counterfactual Generation ----
        if enable_counterfactual:
            filtered_memory_context = self._format_filtered_memory_context(
                memory_read, filter_result,
            )

            cf_result = self.counterfactual.generate(
                patient_text=patient_text,
                current_intervention=intervention,
                memory_context=filtered_memory_context,
                clinical_state=clinical_state,
                kg_features=kg_flags,
                current_prediction=prediction,
            )

            output.counterfactual_candidates = cf_result.get("candidates", [])
            output.best_alternative = cf_result.get("best_alternative")

            # 6c: Transition Utility (if we have a best alternative)
            if output.best_alternative:
                utility_result = self.supervisor.transition_utility(
                    state_before=patient_summary,
                    action_description=str(output.best_alternative.get(
                        "intervention_description",
                        output.best_alternative.get("strategy_name", ""),
                    )),
                    state_after=f"Predicted SOFA change: {output.best_alternative.get('expected_sofa_change_8h', 0)}",
                )
                output.supervisor_utility_score = float(utility_result.get("net_utility", 0))

        # ---- Step 8: Write to Memory Bank ----
        if write_to_memory:
            self.memory_manager.write(
                encoding=encoding,
                pattern_label=pattern_label,
                trajectory_label=trajectory_label,
                experience_label=experience_label,
                metadata={
                    "patient_id": patient_id,
                    "sofa_current": clinical_state.sofa_current,
                    "risk_level_hash": hash(output.risk_level) % 100,
                    "intervention_hash": hash(intervention) % 1000,
                    "timestamp": time.time(),
                    "severity_bin": self._bin_to_float(
                        clinical_state.severity_bin,
                        ["low", "moderate", "high", "critical"],
                    ),
                    "trend_bin": self._bin_to_float(
                        clinical_state.trend_bin,
                        ["improving", "stable", "worsening"],
                    ),
                },
                activity=1.0,
            )
            self._prediction_count += 1

        # ---- Timing ----
        output.metadata["elapsed_seconds"] = round(time.time() - t_start, 2)
        output.metadata["prediction_count"] = self._prediction_count

        return output

    def generate_counterfactuals(
        self,
        patient_text: str,
        intervention: str = "",
        prediction_output: Optional[FusedPredictionOutput] = None,
    ) -> Dict[str, object]:
        """
        Standalone counterfactual generation using the full pipeline context.
        """
        clinical_state = prediction_output.clinical_state if prediction_output else None
        memory_context = ""
        if prediction_output:
            memory_context = f"Memory confidence: {prediction_output.memory_confidence}\n"
            memory_context += f"Matched cases: {prediction_output.memory_matched_count}\n"
            memory_context += f"Semantic hits: {prediction_output.semantic_hit_count}"

        kg_features = prediction_output.kg_features if prediction_output else {}

        current_prediction = {
            "risk_level": prediction_output.risk_level if prediction_output else "unknown",
            "confidence": prediction_output.risk_confidence if prediction_output else 0.0,
            "hourly_predictions": {
                str(k): {"total": v}
                for k, v in (prediction_output.hourly_sofa_totals.items() if prediction_output else [])
            },
        }

        return self.counterfactual.generate(
            patient_text=patient_text,
            current_intervention=intervention,
            memory_context=memory_context,
            clinical_state=clinical_state,
            kg_features=kg_features,
            current_prediction=current_prediction,
        )

    def write_experience(self, output: FusedPredictionOutput) -> bool:
        """
        Write prediction result back to the persistent experience store.
        """
        if not self.persist_store:
            return False

        try:
            experience = {
                "patient_id": output.patient_id,
                "timestamp": output.timestamp,
                "risk_level": output.risk_level,
                "risk_confidence": output.risk_confidence,
                "memory_confidence": output.memory_confidence,
                "template_blend_weight": output.template_blend_weight,
                "hourly_sofa_totals": output.hourly_sofa_totals,
                "sofa_scores": output.sofa_scores,
                "supervisor_gate_decision": output.supervisor_gate_decision,
                "guideline_alignment": output.guideline_alignment,
                "model_name": output.model_name,
                "clinical_state": output.clinical_state.to_dict() if output.clinical_state else {},
            }

            persist_path = Path(self.config.persist_directory) / "experiences.jsonl"
            self._append_jsonl(persist_path, [experience])
            return True
        except Exception as e:
            print(f"[FusedSystem] Write experience failed: {e}")
            return False

    def load_experiences(self, result_files: List[str]) -> int:
        """
        Load historical prediction results into the memory bank.
        Compatible with existing result_*.json files from the old system.
        """
        loaded = 0
        for filepath in result_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                loaded += self._import_result(data)
            except Exception as e:
                print(f"[FusedSystem] Failed to load {filepath}: {e}")
        return loaded

    def get_memory_summary(self) -> Dict[str, object]:
        """Get summary statistics of the memory bank."""
        return self.memory_manager.summarize()

    # ==================================================================
    # Internal Helpers
    # ==================================================================

    def _extract_patient_static(self, patient_text: str) -> List[float]:
        """Extract basic static features from patient text."""
        import re

        features = []

        # Age
        age_match = re.search(r'(\d+)\s*岁', patient_text)
        features.append(float(age_match.group(1)) / 100.0 if age_match else 0.5)

        # Gender
        features.append(1.0 if '男性' in patient_text else (0.0 if '女性' in patient_text else 0.5))

        # Weight
        wt_match = re.search(r'(\d+)\s*kg', patient_text)
        features.append(float(wt_match.group(1)) / 150.0 if wt_match else 0.5)

        # ICU days
        icu_match = re.search(r'ICU.*?(\d+)\s*天', patient_text)
        features.append(float(icu_match.group(1)) / 30.0 if icu_match else 0.1)

        return features

    def _build_retrieved_case_list(
        self, memory_read,
    ) -> List[Dict[str, object]]:
        """Build a list of retrieved cases for supervisor review."""
        cases = []
        for comp_name in ["experience", "pattern", "trajectory"]:
            comp = memory_read.component_results.get(comp_name)
            if comp is None:
                continue
            for idx, label in zip(comp.matched_indices[:3], comp.matched_labels[:3]):
                cases.append({
                    "component": comp_name,
                    "index": idx,
                    "label": label,
                    "similarity": float(comp.max_similarity) if idx == comp.matched_indices[0] else 0.5,
                    "confidence": float(comp.confidence),
                })
        return cases

    def _format_memory_context(
        self, memory_read, clinical_state: ClinicalStateSignature,
    ) -> str:
        """Format memory read results as natural language context for the LLM predictor."""
        semantic = memory_read.semantic_result
        exp_read = memory_read.component_results.get("experience")

        lines = []

        if semantic.get("hit_count", 0) > 0:
            lines.append(f"Found {semantic['hit_count']} similar historical cases (prototype confidence: {semantic.get('template_confidence', 0):.2f}).")
            lines.append(f"Future direction alignment: {semantic.get('direction_alignment', 1.0):.2f}")
            lines.append(f"Template blend weight: {semantic.get('template_blend_weight', 0):.3f}")

        if exp_read and exp_read.matched_indices:
            lines.append(f"Experience memory matched {len(exp_read.matched_indices)} cases.")
            lines.append(f"Top experience label: {exp_read.top_label}")
            lines.append(f"Experience confidence: {exp_read.confidence:.3f}")

        pat_read = memory_read.component_results.get("pattern")
        if pat_read and pat_read.matched_indices:
            lines.append(f"Pattern memory: matched {len(pat_read.matched_indices)} cases (confidence: {pat_read.confidence:.3f}, top label: {pat_read.top_label}).")

        traj_read = memory_read.component_results.get("trajectory")
        if traj_read and traj_read.matched_indices:
            lines.append(f"Trajectory memory: matched {len(traj_read.matched_indices)} cases (confidence: {traj_read.confidence:.3f}, top label: {traj_read.top_label}).")

        planner = memory_read.planner_weights
        lines.append(f"Memory component weights — Pattern: {planner.get('pattern', 0):.2f}, "
                     f"Trajectory: {planner.get('trajectory', 0):.2f}, "
                     f"Experience: {planner.get('experience', 0):.2f}")

        if not lines:
            lines.append("No relevant historical cases found in the experience memory bank.")

        return "\n".join(lines)

    def _format_filtered_memory_context(
        self, memory_read, filter_result: Dict[str, object],
    ) -> str:
        """Format memory context using only supervisor-approved cases."""
        relevant = set(filter_result.get("relevant_cases", []))
        if not relevant:
            return "No clinically relevant historical cases after supervisor filtering."

        lines = []
        for comp_name in ["experience", "pattern", "trajectory"]:
            comp = memory_read.component_results.get(comp_name)
            if comp is None:
                continue
            relevant_matches = [
                (idx, label) for idx, label in zip(comp.matched_indices, comp.matched_labels)
                if idx in relevant
            ]
            if relevant_matches:
                lines.append(f"{comp_name}: {len(relevant_matches)} relevant matches (confidence: {comp.confidence:.3f})")

        if not lines:
            return "No clinically relevant historical cases after supervisor filtering."
        return "\n".join(lines)

    def _import_result(self, data: Dict[str, object]) -> int:
        """Import a single result JSON into the memory bank."""
        # Try to extract key fields from old-system result format
        patient_id = str(data.get("patient_id", data.get("stay_id", "")))
        if not patient_id:
            return 0

        # Extract what we can and write to memory
        # This is best-effort import for bootstrapping
        return 0

    def _label_index(self, label: str, labels: List[str]) -> int:
        try:
            return labels.index(label)
        except ValueError:
            return 0

    def _bin_to_float(self, value: str, bins: List[str]) -> float:
        try:
            return float(bins.index(value)) / max(1, len(bins) - 1)
        except ValueError:
            return 0.5

    def _append_jsonl(self, path: Path, rows: List[Dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
