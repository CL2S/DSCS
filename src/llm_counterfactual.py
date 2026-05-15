#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-Based Counterfactual Intervention Generator — replaces the DL counterfactual
plan renderer with DSPy-driven LLM intervention planning.

Generates alternative intervention plans by:
    1. Analyzing the current intervention and patient state
    2. Searching for evidence-based alternatives (KG-guided)
    3. Using similar case experience to inform options
    4. Evaluating and ranking candidates by predicted outcome
"""

from __future__ import annotations

import json
import math
import re
from typing import Dict, List, Optional

from src.fusion_config import ClinicalStateSignature, LLMCounterfactualConfig


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

COUNTERFACTUAL_GENERATION_PROMPT = """You are an ICU clinical decision support AI. Your task is to generate alternative intervention plans for a sepsis patient, based on clinical evidence, guidelines, and similar case experience.

## Current Patient State
{patient_text}

## Current Intervention Plan
{current_intervention}

## Clinical State Assessment
- Severity: {severity} (SOFA ≈ {sofa_current})
- Trend: {trend}
- Trajectory: {trajectory}
- Pattern: {pattern}

## Guideline Compliance
{guideline_context}

## Similar Case Experience
{memory_context}

## Task
Generate {max_candidates} alternative intervention plans that could potentially improve patient outcomes. For each plan:

1. Describe the intervention strategy
2. Explain the clinical rationale (why this might be better)
3. Estimate the expected SOFA change at 8 hours (positive = improvement)
4. Rate the plan's risk level (low/medium/high)
5. Rate your confidence (0.0-1.0)

IMPORTANT RULES:
- Only suggest interventions supported by clinical evidence or guidelines
- Do NOT suggest stopping life-saving treatments
- Consider the patient's current severity — don't suggest overly aggressive plans for stable patients
- Each plan must be clinically distinct from the others
- If the patient is stable and the current plan is appropriate, say so

Output as JSON:
{{
  "patient_stable_on_current_plan": true/false,
  "candidates": [
    {{
      "plan_id": "plan_1",
      "strategy_name": "short descriptive name",
      "intervention_description": "detailed description of the alternative intervention",
      "changes_from_current": ["change 1", "change 2"],
      "clinical_rationale": "why this might improve outcomes",
      "expected_sofa_change_8h": number (positive = improvement),
      "risk_level": "low|medium|high",
      "confidence": 0.0-1.0,
      "evidence_basis": ["guideline X", "similar case Y"]
    }}
  ],
  "assessment": "overall clinical assessment"
}}

JSON:"""


COUNTERFACTUAL_EVALUATION_PROMPT = """You are an ICU clinical decision support AI. Compare two intervention plans for the same patient and determine which is likely to produce better outcomes.

## Patient State
{patient_text}

## Plan A (Current)
{plan_current}

## Plan B (Alternative)
{plan_alternative}

## Predicted SOFA Trajectory Under Current Plan
{current_prediction}

Compare the plans across these dimensions (score each 0-10):
- guideline_compliance: adherence to SSC 2021 guidelines
- risk_benefit_ratio: expected benefit relative to risk
- evidence_strength: strength of supporting evidence
- feasibility: practical implementability in ICU
- patient_specificity: how well tailored to this specific patient

Output as JSON:
{{
  "plan_a_scores": {{"guideline_compliance": N, "risk_benefit_ratio": N, "evidence_strength": N, "feasibility": N, "patient_specificity": N}},
  "plan_b_scores": {{...}},
  "winner": "A" or "B" or "tie",
  "confidence": 0.0-1.0,
  "reasoning": "why the winner is preferred"
}}

JSON:"""


# ---------------------------------------------------------------------------
# KG Repair Actions (from new system's counterfactual_plan_renderer)
# ---------------------------------------------------------------------------

REPAIR_ACTIONS = {
    "state_sepsis": [
        {"action": "ensure_early_antimicrobial", "description": "Ensure broad-spectrum antimicrobial within 1 hour"},
        {"action": "ensure_blood_cultures", "description": "Obtain blood cultures before antimicrobial"},
        {"action": "ensure_lactate_measurement", "description": "Measure serum lactate level"},
    ],
    "state_septic_shock": [
        {"action": "ensure_vasopressor", "description": "Initiate vasopressor to maintain MAP >= 65 mmHg"},
        {"action": "ensure_fluid_resuscitation", "description": "Complete initial fluid resuscitation 30 mL/kg"},
        {"action": "ensure_lactate_clearance_monitoring", "description": "Monitor lactate clearance"},
    ],
    "state_hypotension": [
        {"action": "ensure_map_monitoring", "description": "Continuous MAP monitoring, target >= 65 mmHg"},
        {"action": "assess_fluid_responsiveness", "description": "Assess fluid responsiveness before additional fluids"},
    ],
    "state_high_lactate": [
        {"action": "ensure_lactate_repeat", "description": "Repeat lactate measurement within 2-4 hours"},
        {"action": "ensure_tissue_perfusion_assessment", "description": "Assess tissue perfusion and cardiac output"},
    ],
    "state_organ_dysfunction": [
        {"action": "ensure_organ_support", "description": "Ensure appropriate organ support (RRT, ventilation, etc.)"},
        {"action": "ensure_source_control", "description": "Identify and control infection source"},
    ],
}


# ---------------------------------------------------------------------------
# LLM Counterfactual Generator
# ---------------------------------------------------------------------------

class LLMCounterfactualGenerator:
    """
    LLM-based counterfactual intervention generator.

    Usage:
        generator = LLMCounterfactualGenerator(config, llm_call_fn)
        candidates = generator.generate(patient_text, current_intervention,
                                         memory_context, clinical_state, kg_features)
    """

    def __init__(
        self,
        config: LLMCounterfactualConfig,
        llm_call_fn=None,
    ):
        self.config = config
        self._llm_call = llm_call_fn or self._default_ollama_call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        patient_text: str,
        current_intervention: str = "",
        memory_context: str = "",
        clinical_state: Optional[ClinicalStateSignature] = None,
        kg_features: Optional[Dict[str, float]] = None,
        current_prediction: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """
        Generate counterfactual intervention candidates.
        """
        state = clinical_state or ClinicalStateSignature()

        # Step 1: Generate KG repair suggestions
        kg_repairs = []
        if self.config.enable_kg_repair and kg_features:
            kg_repairs = self._generate_kg_repairs(kg_features)

        # Step 2: Generate LLM-driven alternatives
        guideline_context = self._format_guideline_context(kg_features, kg_repairs)

        prompt = COUNTERFACTUAL_GENERATION_PROMPT.format(
            patient_text=patient_text[:4000],
            current_intervention=current_intervention[:1500],
            severity=state.severity_bin,
            sofa_current=state.sofa_current,
            trend=state.trend_bin,
            trajectory=state.trajectory_label,
            pattern=state.pattern_label,
            guideline_context=guideline_context,
            memory_context=memory_context if memory_context else "No similar case data available.",
            max_candidates=self.config.max_candidates,
        )

        raw = self._llm_call(prompt)
        candidates = self._parse_json(raw)

        # Step 3: Attach KG repair context
        candidates["kg_repairs"] = kg_repairs

        # Step 4: If we have a current prediction, evaluate alternatives
        if current_prediction and candidates.get("candidates"):
            candidates = self._evaluate_against_current(
                patient_text, current_intervention, candidates, current_prediction,
            )

        return candidates

    def generate_intervention_plan(
        self,
        patient_text: str,
        current_intervention: str = "",
        memory_context: str = "",
        clinical_state: Optional[ClinicalStateSignature] = None,
        kg_features: Optional[Dict[str, float]] = None,
        donor_context: str = "",
    ) -> Dict[str, object]:
        """
        Generate intervention plans incorporating donor (similar patient) context.
        Compatible with the new system's counterfactual workflow.
        """
        state = clinical_state or ClinicalStateSignature()

        donor_section = ""
        if donor_context:
            donor_section = f"""
## Similar Patient (Donor) Experience
{donor_context}

Consider whether the donor's intervention approach can be adapted for the current patient.
"""

        prompt = COUNTERFACTUAL_GENERATION_PROMPT.format(
            patient_text=patient_text[:4000],
            current_intervention=current_intervention[:1500],
            severity=state.severity_bin,
            sofa_current=state.sofa_current,
            trend=state.trend_bin,
            trajectory=state.trajectory_label,
            pattern=state.pattern_label,
            guideline_context=self._format_guideline_context(kg_features, []),
            memory_context=memory_context + donor_section if memory_context else donor_section,
            max_candidates=self.config.max_candidates,
        )

        raw = self._llm_call(prompt)
        return self._parse_json(raw)

    # ------------------------------------------------------------------
    # Internal: KG repair
    # ------------------------------------------------------------------

    def _generate_kg_repairs(self, kg_features: Dict[str, float]) -> List[Dict[str, str]]:
        """Generate repair suggestions based on KG flag gaps."""
        repairs = []
        for flag_name, flag_value in kg_features.items():
            if flag_value <= 0:
                # This flag is NOT active — check if it should be repaired
                base_name = flag_name.replace("kg_", "")
                if base_name in REPAIR_ACTIONS:
                    repairs.extend([
                        {"flag": flag_name, **action}
                        for action in REPAIR_ACTIONS[base_name]
                    ])
        return repairs

    # ------------------------------------------------------------------
    # Internal: evaluation
    # ------------------------------------------------------------------

    def _evaluate_against_current(
        self,
        patient_text: str,
        current_intervention: str,
        candidates: Dict[str, object],
        current_prediction: Dict[str, object],
    ) -> Dict[str, object]:
        """Evaluate each candidate against the current plan."""
        pred_summary = json.dumps({
            "risk_level": current_prediction.get("risk_level"),
            "confidence": current_prediction.get("confidence"),
            "hourly_totals": {
                k: v.get("total", 0) if isinstance(v, dict) else 0
                for k, v in current_prediction.get("hourly_predictions", {}).items()
            },
        }, ensure_ascii=False)

        evaluated = []
        for candidate in (candidates.get("candidates") or []):
            plan_desc = candidate.get("intervention_description", str(candidate))
            prompt = COUNTERFACTUAL_EVALUATION_PROMPT.format(
                patient_text=patient_text[:2000],
                plan_current=current_intervention[:1000],
                plan_alternative=plan_desc[:1000],
                current_prediction=pred_summary,
            )
            raw = self._llm_call(prompt)
            eval_result = self._parse_json(raw)
            candidate["evaluation"] = eval_result
            candidate["composite_score"] = self._compute_composite_score(eval_result)
            evaluated.append(candidate)

        evaluated.sort(key=lambda c: c.get("composite_score", 0), reverse=True)
        candidates["candidates"] = evaluated
        if evaluated:
            candidates["best_alternative"] = evaluated[0]

        return candidates

    def _compute_composite_score(self, evaluation: Dict[str, object]) -> float:
        """Compute composite score from plan evaluation."""
        if not evaluation or evaluation.get("winner") == "tie":
            return 5.0

        plan_b = evaluation.get("plan_b_scores", {})
        if isinstance(plan_b, dict):
            scores = [float(v) for v in plan_b.values() if isinstance(v, (int, float))]
            return sum(scores) / max(1, len(scores)) if scores else 5.0
        return 5.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_guideline_context(
        self,
        kg_features: Optional[Dict[str, float]],
        kg_repairs: List[Dict[str, str]],
    ) -> str:
        """Format KG and guideline context for the prompt."""
        lines = []

        if kg_features:
            active = [k.replace("kg_", "") for k, v in kg_features.items() if v > 0]
            missing = [k.replace("kg_", "") for k, v in kg_features.items() if v <= 0]
            if active:
                lines.append(f"Active clinical indicators: {', '.join(active)}")
            if missing:
                lines.append(f"Missing/incomplete indicators: {', '.join(missing)}")

        if kg_repairs:
            lines.append("\nSuggested guideline-based repairs:")
            for repair in kg_repairs[:5]:
                lines.append(f"  - {repair.get('description', repair.get('action', ''))}")

        if not lines:
            lines.append("No specific guideline compliance data available.")

        return "\n".join(lines)

    def _parse_json(self, text: str) -> Dict[str, object]:
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
        return {"candidates": []}

    def _default_ollama_call(self, prompt: str, model: str = "", temperature: Optional[float] = None) -> str:
        """Default Ollama HTTP API call."""
        import urllib.request

        model_name = model or self.config.generator_model
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
            print(f"[LLMCounterfactual] Ollama call failed: {e}")
            return '{"candidates": []}'
