#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-Based Patient State Encoder — replaces the DL encoder (GRU/Transformer) with
DSPy-driven LLM feature extraction, producing TorchManifoldEncodingOutput compatible
with the manifold memory system.

The LLM reads structured patient text descriptions and extracts:
    1. ClinicalStateSignature (severity/trend/volatility/trajectory/pattern bins)
    2. Numerical clinical feature vectors (formation features, KG features, static info)
    3. These are projected into manifold space (query/key/value) for memory operations

This bridges the LLM world (natural language) and the tensor world (manifold memory).
"""

from __future__ import annotations

import json
import math
import re
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from src.fusion_config import ClinicalStateSignature, LLMEncoderConfig
from src.manifold_memory import (
    ManifoldMemoryConfig,
    TorchManifoldEncodingOutput,
    _normalize_tensor,
)
from src.ts_formation import (
    PATTERN_LABELS,
    TRAJECTORY_LABELS,
    formation_feature_names,
)


# ---------------------------------------------------------------------------
# DSPy-style prompt templates (usable with or without DSPy)
# ---------------------------------------------------------------------------

CLINICAL_STATE_EXTRACTION_PROMPT = """You are a clinical AI that extracts structured patient state from ICU descriptions.

Patient Description:
{patient_text}

Current Intervention:
{intervention}

Extract the following clinical state information as a JSON object. Be precise and use only the provided data.

Output a single JSON with these exact keys:
- severity_bin: one of "low" (SOFA<4), "moderate" (SOFA 4-8), "high" (SOFA 8-12), "critical" (SOFA>=12)
- level_bin: one of "below_baseline", "near_baseline", "above_baseline", "far_above_baseline"
- trend_bin: one of "improving", "stable", "worsening"
- volatility_bin: one of "quiet", "variable", "unstable"
- trajectory_label: one of "stable_regime", "rising_regime", "falling_regime", "seasonal_regime", "shifted_regime"
- pattern_label: one of "flat", "up", "down", "volatile", "spike"
- sofa_current: the current total SOFA score (number)
- sofa_trend_6h: the SOFA change over the last 6 hours (number, positive means worsening)
- reasoning: brief clinical reasoning for the assessment (1-2 sentences)

JSON:"""


FEATURE_EXTRACTION_PROMPT = """You are a clinical AI that converts ICU patient data into numerical feature vectors.

Patient Description:
{patient_text}

Current Intervention:
{intervention}

Clinical State Summary:
{clinical_state}

Based on the patient data, extract these numerical feature vectors. Output ONLY a JSON object:

1. "formation_features": 16 numbers representing:
   [local_slope, medium_slope, local_volatility, volatility_ratio, seasonal_gap_norm,
    seasonal_strength, level_shift_norm, max_zscore, range_norm, phase_sin,
    phase_cos, change_proxy, curvature, autocorr_lag1, stability_score, regime_mix_score]
   - local_slope: recent SOFA trend (positive = worsening), range roughly [-0.5, 0.5]
   - local_volatility: recent variability, range [0, 1]
   - stability_score: how stable the patient is, range [0, 1] (1 = very stable)
   - regime_mix_score: mixing between regimes, range [0, 1]

2. "sofa_components": object with these keys and estimated current values:
   {"sofa_respiration": number, "sofa_coagulation": number, "sofa_liver": number,
    "sofa_cardiovascular": number, "sofa_cns": number, "sofa_renal": number}

3. "vital_signs_summary": object with estimated current values:
   {"map": number, "heart_rate": number, "respiratory_rate": number,
    "temperature": number, "spo2": number, "gcs": number,
    "lactate": number, "creatinine": number, "platelet": number, "bilirubin": number}

4. "intervention_features": object with intervention characterization:
   {"vasopressor_active": 0 or 1, "vasopressor_intensity": number (0-1),
    "mechanical_ventilation": 0 or 1, "antibiotic_active": 0 or 1,
    "fluid_resuscitation_active": 0 or 1}

JSON:"""


# ---------------------------------------------------------------------------
# Lightweight projection network — maps LLM features → manifold space
# ---------------------------------------------------------------------------

class LLMFeatureProjector(nn.Module):
    """
    Projects LLM-extracted clinical features into the manifold memory space.

    Takes a concatenated vector of:
        - formation features (16 dims)
        - sofa components (6 dims)
        - vital signs summary (10 dims)
        - intervention features (5 dims)
        - KG features (N dims, from KnowledgeGraphFeatureBuilder)
        - patient static (M dims)
    And produces query/key/value tensors compatible with TorchManifoldEncodingOutput.
    """

    def __init__(self, config: LLMEncoderConfig):
        super().__init__()
        self.config = config

        # Input: formation(16) + sofa(6) + vitals(10) + intervention(5) + KG + static
        self.clinical_feature_dim = 16 + 6 + 10 + 5
        self.total_input_dim = self.clinical_feature_dim + config.static_feature_dim

        # Project to context embedding (mimics GRU's 3-part context)
        self.context_projector = nn.Sequential(
            nn.Linear(self.total_input_dim, config.projection_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.projection_hidden_dim) if config.use_projection_norm else nn.Identity(),
            nn.Linear(config.projection_hidden_dim, config.context_dim),
            nn.GELU(),
            nn.LayerNorm(config.context_dim) if config.use_projection_norm else nn.Identity(),
        )

        # Query/Key/Value heads (same architecture as GRUManifoldEncoder)
        self.query_head = nn.Linear(config.context_dim, config.manifold_dim)
        self.key_head = nn.Linear(config.context_dim, config.manifold_dim)
        self.value_head = nn.Linear(config.context_dim, config.value_dim)

    def forward(
        self,
        clinical_features: torch.Tensor,
        static_features: torch.Tensor,
    ) -> TorchManifoldEncodingOutput:
        """
        Args:
            clinical_features: [clinical_feature_dim] tensor
            static_features: [static_feature_dim] tensor (KG + patient static)
        Returns:
            TorchManifoldEncodingOutput with query/key/value tensors
        """
        if clinical_features.dim() == 1:
            clinical_features = clinical_features.unsqueeze(0)
        if static_features.dim() == 1:
            static_features = static_features.unsqueeze(0)

        combined = torch.cat([clinical_features, static_features], dim=-1)
        context = self.context_projector(combined)
        query = _normalize_tensor(self.query_head(context))
        key = _normalize_tensor(self.key_head(context))
        value = self.value_head(context)

        return TorchManifoldEncodingOutput(
            query=query.squeeze(0),
            key=key.squeeze(0),
            value=value.squeeze(0),
            input_embedding=context.squeeze(0),
        )


# ---------------------------------------------------------------------------
# LLM Encoder — orchestrates LLM calls + projection
# ---------------------------------------------------------------------------

class LLMEncoder:
    """
    LLM-based patient state encoder.

    Usage:
        encoder = LLMEncoder(config)
        encoding, state_sig = encoder.encode(patient_text, intervention,
                                              kg_features, patient_static)
        # encoding is TorchManifoldEncodingOutput — ready for MetaMemoryManager
    """

    def __init__(
        self,
        config: LLMEncoderConfig,
        llm_call_fn=None,  # injectable for testing / custom backends
    ):
        self.config = config
        self.projector = LLMFeatureProjector(config)
        self.projector.to(torch.device(config.device))

        # LLM call function — defaults to Ollama via DSPy
        self._llm_call = llm_call_fn or self._default_ollama_call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        patient_text: str,
        intervention: str = "",
        kg_features: Optional[List[float]] = None,
        patient_static: Optional[List[float]] = None,
        metadata: Optional[Dict[str, float]] = None,
    ) -> Tuple[TorchManifoldEncodingOutput, ClinicalStateSignature]:
        """
        Full encoding pipeline: LLM extraction → projection → manifold encoding.
        """
        # Step 1: LLM extracts clinical state signature
        state_sig = self.extract_clinical_state(patient_text, intervention)

        # Step 2: LLM extracts numerical feature vectors
        features = self.extract_features(patient_text, intervention, state_sig)

        # Step 3: Build tensors and project to manifold space
        clinical_tensor = self._features_to_tensor(features)

        kg_vec = kg_features if kg_features else []
        static_vec = patient_static if patient_static else []
        if len(kg_vec) + len(static_vec) < self.config.static_feature_dim:
            padding = [0.0] * (self.config.static_feature_dim - len(kg_vec) - len(static_vec))
            full_static = kg_vec + static_vec + padding
        else:
            full_static = (kg_vec + static_vec)[:self.config.static_feature_dim]

        static_tensor = torch.tensor(full_static, dtype=torch.float32, device=torch.device(self.config.device))

        with torch.no_grad():
            encoding = self.projector(clinical_tensor, static_tensor)

        encoding.metadata = metadata or {}
        encoding.metadata.update({
            "severity_bin": self._bin_to_float(state_sig.severity_bin, ["low", "moderate", "high", "critical"]),
            "trend_bin": self._bin_to_float(state_sig.trend_bin, ["improving", "stable", "worsening"]),
            "trajectory_label": float(TRAJECTORY_LABELS.index(state_sig.trajectory_label)
                                      if state_sig.trajectory_label in TRAJECTORY_LABELS else 0),
            "pattern_label": float(PATTERN_LABELS.index(state_sig.pattern_label)
                                   if state_sig.pattern_label in PATTERN_LABELS else 0),
        })

        return encoding, state_sig

    def extract_clinical_state(
        self, patient_text: str, intervention: str = ""
    ) -> ClinicalStateSignature:
        """Extract structured clinical state via LLM."""
        prompt = CLINICAL_STATE_EXTRACTION_PROMPT.format(
            patient_text=patient_text[:4000],
            intervention=intervention[:1000],
        )
        raw = self._llm_call(prompt)
        parsed = self._parse_json(raw)

        sofa = float(parsed.get("sofa_current", 0) or 0)
        return ClinicalStateSignature(
            severity_bin=str(parsed.get("severity_bin", "moderate")),
            level_bin=str(parsed.get("level_bin", "near_baseline")),
            trend_bin=str(parsed.get("trend_bin", "stable")),
            volatility_bin=str(parsed.get("volatility_bin", "quiet")),
            trajectory_label=str(parsed.get("trajectory_label", "stable_regime")),
            pattern_label=str(parsed.get("pattern_label", "flat")),
            sofa_current=sofa,
            sofa_trend_6h=float(parsed.get("sofa_trend_6h", 0) or 0),
        )

    def extract_features(
        self,
        patient_text: str,
        intervention: str = "",
        clinical_state: Optional[ClinicalStateSignature] = None,
    ) -> Dict[str, object]:
        """Extract numerical clinical feature vectors via LLM."""
        state_str = json.dumps(clinical_state.to_dict(), ensure_ascii=False) if clinical_state else "{}"
        prompt = FEATURE_EXTRACTION_PROMPT.format(
            patient_text=patient_text[:4000],
            intervention=intervention[:1000],
            clinical_state=state_str,
        )
        raw = self._llm_call(prompt)
        return self._parse_json(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _features_to_tensor(self, features: Dict[str, object]) -> torch.Tensor:
        """Convert LLM-extracted feature dict to a flat tensor."""
        formation = self._extract_list(features.get("formation_features", []), 16)
        sofa = self._extract_ordered(features.get("sofa_components", {}), [
            "sofa_respiration", "sofa_coagulation", "sofa_liver",
            "sofa_cardiovascular", "sofa_cns", "sofa_renal",
        ])
        vitals = self._extract_ordered(features.get("vital_signs_summary", {}), [
            "map", "heart_rate", "respiratory_rate", "temperature", "spo2",
            "gcs", "lactate", "creatinine", "platelet", "bilirubin",
        ])
        intervention_f = self._extract_ordered(features.get("intervention_features", {}), [
            "vasopressor_active", "vasopressor_intensity",
            "mechanical_ventilation", "antibiotic_active",
            "fluid_resuscitation_active",
        ])

        flat = formation + sofa + vitals + intervention_f
        vec = torch.tensor(flat, dtype=torch.float32, device=torch.device(self.config.device))
        return vec

    def _extract_list(self, val: object, expected_len: int) -> List[float]:
        if isinstance(val, (list, tuple)):
            result = [float(v) for v in val]
        elif isinstance(val, str):
            result = [float(x.strip()) for x in val.strip("[]").split(",") if x.strip()]
        else:
            result = []
        while len(result) < expected_len:
            result.append(0.0)
        return result[:expected_len]

    def _extract_ordered(self, obj: object, keys: List[str]) -> List[float]:
        if isinstance(obj, dict):
            return [float(obj.get(k, 0) or 0) for k in keys]
        return [0.0] * len(keys)

    def _bin_to_float(self, value: str, bins: List[str]) -> float:
        try:
            return float(bins.index(value)) / max(1, len(bins) - 1)
        except ValueError:
            return 0.5

    def _parse_json(self, text: str) -> Dict[str, object]:
        """Robust JSON extraction from LLM output."""
        # Try direct parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to extract JSON block from markdown
        for pattern in [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{.*\}',
        ]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except (json.JSONDecodeError, TypeError):
                    continue

        return {}

    def _default_ollama_call(self, prompt: str) -> str:
        """Default LLM call via Ollama HTTP API (no DSPy dependency required)."""
        import urllib.request

        payload = json.dumps({
            "model": self.config.encoder_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }).encode("utf-8")

        url = f"{self.config.ollama_base_url}/api/generate"
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "")
        except Exception as e:
            print(f"[LLMEncoder] Ollama call failed: {e}")
            return "{}"
