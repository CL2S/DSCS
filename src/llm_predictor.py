#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-Based SOFA Predictor — replaces the DL base predictor with DSPy-driven LLM
forecasting. Uses multi-model ensemble for robustness and uncertainty estimation.

Integrates with:
    - Memory bank readout (from MetaMemoryManager) for experience-enhanced prediction
    - KG features (from KnowledgeGraphFeatureBuilder) for guideline-aware forecasting
    - ClinicalStateSignature for structured patient context
"""

from __future__ import annotations

import json
import math
import re
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

from src.fusion_config import ClinicalStateSignature, LLMPredictorConfig


# ---------------------------------------------------------------------------
# DSPy-style prompt templates
# ---------------------------------------------------------------------------

SOFA_PREDICTION_SYSTEM = """You are an expert ICU clinical forecasting AI. Your task is to predict SOFA scores for the next 8 hours based on patient data, historical experience, and clinical guidelines.

Output ONLY valid JSON. Do not include any text outside the JSON object."""


SOFA_PREDICTION_PROMPT = """## Patient Clinical State
{patient_text}

## Current Intervention
{intervention}

## Clinical State Assessment
- Severity: {severity} (SOFA ≈ {sofa_current})
- Trend: {trend} (6h change: {sofa_trend_6h})
- Volatility: {volatility}
- Pattern: {pattern}, Trajectory: {trajectory}

## Knowledge Graph Flags
{kg_summary}

## Guideline Alignment Score: {guideline_alignment}

## Historical Experience Context
{memory_context}

## Task
Predict the SOFA scores for each of the next 8 hours (hour 1 through hour 8).
For each hour, provide:
1. Individual component scores (sofa_respiration, sofa_coagulation, sofa_liver, sofa_cardiovascular, sofa_cns, sofa_renal)
2. Total SOFA score
3. The reasoning should consider the current trajectory, similar historical cases, and guideline compliance.

Also assess:
- risk_level: "low", "medium", or "high" for septic shock risk in the next 8 hours
- key_concern: the most critical clinical concern (1 sentence)
- confidence: your confidence in this prediction (0.0 to 1.0)

Output as JSON:
{{
  "hourly_predictions": {{
    "1": {{"sofa_respiration": N, "sofa_coagulation": N, "sofa_liver": N, "sofa_cardiovascular": N, "sofa_cns": N, "sofa_renal": N, "total": N}},
    ...
    "8": {{...}}
  }},
  "risk_level": "low|medium|high",
  "key_concern": "string",
  "confidence": 0.0-1.0,
  "reasoning": "brief clinical reasoning"
}}

JSON:"""


# ---------------------------------------------------------------------------
# LLM Predictor
# ---------------------------------------------------------------------------

class LLMPredictor:
    """
    LLM-based SOFA predictor with multi-model ensemble support.

    Usage:
        predictor = LLMPredictor(config, llm_call_fn)
        output = predictor.predict(patient_text, intervention, memory_context,
                                    clinical_state, kg_features)
    """

    def __init__(
        self,
        config: LLMPredictorConfig,
        llm_call_fn=None,
    ):
        self.config = config
        self._llm_call = llm_call_fn or self._default_ollama_call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        patient_text: str,
        intervention: str = "",
        memory_context: str = "",
        clinical_state: Optional[ClinicalStateSignature] = None,
        kg_features: Optional[Dict[str, float]] = None,
        guideline_alignment: float = 0.0,
    ) -> Dict[str, object]:
        """
        Run prediction, optionally with ensemble for uncertainty estimation.
        """
        if self.config.uncertainty_method == "multi_model" and len(self.config.ensemble_models) > 1:
            return self._ensemble_predict(
                patient_text, intervention, memory_context,
                clinical_state, kg_features, guideline_alignment,
            )
        else:
            return self._single_predict(
                self.config.predictor_model,
                patient_text, intervention, memory_context,
                clinical_state, kg_features, guideline_alignment,
            )

    def predict_with_uncertainty(
        self,
        patient_text: str,
        intervention: str = "",
        memory_context: str = "",
        clinical_state: Optional[ClinicalStateSignature] = None,
        kg_features: Optional[Dict[str, float]] = None,
        guideline_alignment: float = 0.0,
    ) -> Tuple[Dict[str, object], float, float]:
        """
        Predict with uncertainty estimates.
        Returns (prediction, mean_confidence, std_confidence).
        """
        if self.config.uncertainty_method == "multi_model":
            result = self._ensemble_predict(
                patient_text, intervention, memory_context,
                clinical_state, kg_features, guideline_alignment,
            )
            confidences = result.get("_individual_confidences", [result.get("confidence", 0.5)])
            return result, mean(confidences) if confidences else 0.5, stdev(confidences) if len(confidences) > 1 else 0.0
        elif self.config.uncertainty_method == "temperature":
            samples = []
            for _ in range(self.config.num_samples):
                sample = self._single_predict(
                    self.config.predictor_model,
                    patient_text, intervention, memory_context,
                    clinical_state, kg_features, guideline_alignment,
                    temperature=0.7,  # higher temp for diversity
                )
                samples.append(sample)
            merged = self._merge_samples(samples)
            confidences = [s.get("confidence", 0.5) for s in samples]
            return merged, mean(confidences) if confidences else 0.5, stdev(confidences) if len(confidences) > 1 else 0.0
        else:
            result = self._single_predict(
                self.config.predictor_model,
                patient_text, intervention, memory_context,
                clinical_state, kg_features, guideline_alignment,
            )
            return result, result.get("confidence", 0.5), 0.0

    # ------------------------------------------------------------------
    # Internal: single model prediction
    # ------------------------------------------------------------------

    def _single_predict(
        self,
        model_name: str,
        patient_text: str,
        intervention: str,
        memory_context: str,
        clinical_state: Optional[ClinicalStateSignature],
        kg_features: Optional[Dict[str, float]],
        guideline_alignment: float,
        temperature: Optional[float] = None,
    ) -> Dict[str, object]:
        """Run prediction with a single model."""
        state = clinical_state or ClinicalStateSignature()
        kg_summary = self._format_kg_summary(kg_features or {})

        prompt = SOFA_PREDICTION_PROMPT.format(
            patient_text=patient_text[:5000],
            intervention=intervention[:1000],
            severity=state.severity_bin,
            sofa_current=state.sofa_current,
            trend=state.trend_bin,
            sofa_trend_6h=state.sofa_trend_6h,
            volatility=state.volatility_bin,
            pattern=state.pattern_label,
            trajectory=state.trajectory_label,
            kg_summary=kg_summary,
            guideline_alignment=f"{guideline_alignment:.2f}",
            memory_context=memory_context if memory_context else "No similar historical cases available.",
        )

        temp = temperature if temperature is not None else self.config.temperature
        raw = self._llm_call(prompt, model=model_name, temperature=temp)
        return self._parse_prediction(raw)

    # ------------------------------------------------------------------
    # Internal: ensemble
    # ------------------------------------------------------------------

    def _ensemble_predict(
        self,
        patient_text: str,
        intervention: str,
        memory_context: str,
        clinical_state: Optional[ClinicalStateSignature],
        kg_features: Optional[Dict[str, float]],
        guideline_alignment: float,
    ) -> Dict[str, object]:
        """Average predictions across multiple models."""
        all_predictions = []
        for model_name in self.config.ensemble_models:
            pred = self._single_predict(
                model_name, patient_text, intervention, memory_context,
                clinical_state, kg_features, guideline_alignment,
            )
            all_predictions.append(pred)

        return self._merge_samples(all_predictions)

    def _merge_samples(self, predictions: List[Dict[str, object]]) -> Dict[str, object]:
        """Merge multiple predictions by averaging SOFA scores."""
        if not predictions:
            return {}
        if len(predictions) == 1:
            return predictions[0]

        merged_hourly: Dict[str, Dict[str, float]] = {}
        sofa_components = ["sofa_respiration", "sofa_coagulation", "sofa_liver",
                           "sofa_cardiovascular", "sofa_cns", "sofa_renal", "total"]

        for hour in range(1, 9):
            hour_key = str(hour)
            merged_hourly[hour_key] = {}
            for comp in sofa_components:
                values = []
                for pred in predictions:
                    hourly = pred.get("hourly_predictions", {})
                    hour_data = hourly.get(hour_key, {})
                    val = hour_data.get(comp)
                    if isinstance(val, (int, float)):
                        values.append(float(val))
                merged_hourly[hour_key][comp] = round(mean(values), 2) if values else 0.0

        risk_levels = [p.get("risk_level", "medium") for p in predictions]
        risk_level = max(set(risk_levels), key=risk_levels.count) if risk_levels else "medium"

        confidences = [float(p.get("confidence", 0.5) or 0.5) for p in predictions]

        reasonings = [p.get("reasoning", "") for p in predictions if p.get("reasoning")]
        merged_reasoning = " | ".join(reasonings[:3])

        return {
            "hourly_predictions": merged_hourly,
            "risk_level": risk_level,
            "confidence": round(mean(confidences), 3),
            "reasoning": merged_reasoning,
            "key_concern": predictions[0].get("key_concern", ""),
            "_individual_confidences": confidences,
            "_model_count": len(predictions),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_kg_summary(self, kg_features: Dict[str, float]) -> str:
        """Format KG flags into a readable summary."""
        active = [(k.replace("kg_", ""), v) for k, v in kg_features.items() if v > 0]
        if not active:
            return "No specific KG flags active."
        lines = [f"- {k}: active" for k, v in sorted(active)]
        return "\n".join(lines)

    def _parse_prediction(self, text: str) -> Dict[str, object]:
        """Robust JSON extraction from LLM output."""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        for pattern in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```', r'\{.*\}']:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except (json.JSONDecodeError, TypeError):
                    continue
        return {}

    def _default_ollama_call(
        self, prompt: str, model: str = "", temperature: Optional[float] = None
    ) -> str:
        """Default Ollama HTTP API call."""
        import urllib.request

        model_name = model or self.config.predictor_model
        temp = temperature if temperature is not None else self.config.temperature

        payload = json.dumps({
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": self.config.max_tokens,
            },
        }).encode("utf-8")

        url = f"{self.config.ollama_base_url}/api/generate"
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "")
        except Exception as e:
            print(f"[LLMPredictor] Ollama call failed for {model_name}: {e}")
            return "{}"
