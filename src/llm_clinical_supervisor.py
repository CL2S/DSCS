#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Clinical Supervisor — replaces heuristic gates and transition utility formulas
with LLM-driven clinical reasoning for three decision points:

    1. Memory Gate: Should memory-based correction be applied?
       → Addresses hard flaw #2 (no silence mechanism, stable_regime 60-73% harmed)

    2. Retrieval Filter: Which retrieved experiences are clinically relevant?
       → Addresses hard flaw #4 (persistent experience amplifies noise, 696x pre_strength)

    3. Transition Utility: Is this state transition clinically beneficial?
       → Addresses hard flaw #3 (heuristic utility unreliable, sign flips across runs)

The LLM Supervisor does NOT make numerical predictions — it makes clinical SEMANTIC
judgments: "Is this patient stable enough to skip memory correction?", "Are these
retrieved cases actually relevant?", "Is this intervention change beneficial?"
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from src.fusion_config import ClinicalStateSignature, LLMSupervisorConfig


# ---------------------------------------------------------------------------
# Prompt templates for the three decision points
# ---------------------------------------------------------------------------

MEMORY_GATE_PROMPT = """You are an ICU clinical safety auditor. Your job is to decide whether historical case experience should be used to adjust the current patient's SOFA prediction.

## Current Patient State
{patient_summary}

## Memory Context
- Similar historical cases found: {match_count}
- Top case label: {top_label}
- Memory confidence: {memory_confidence:.3f}
- Template blend weight: {template_blend:.3f}
- Semantic prototype hit count: {semantic_hits}

## Decision Rule
Historical experience is MOST useful when:
- The patient is in an active deterioration or transition phase
- The current trajectory is uncertain (variable vitals, unclear trend)
- Similar cases had meaningfully different interventions

Historical experience should be SILENCED (not applied) when:
- The patient is stable and the current prediction is reliable
- The matched cases are from very different clinical contexts
- The memory signal is weak or contradictory

Based on the above, should memory-based correction be applied?

Output JSON:
{{
  "apply_memory": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "brief clinical reasoning for the decision",
  "risk_if_applied": "low|medium|high",
  "risk_if_silenced": "low|medium|high"
}}

JSON:"""


RETRIEVAL_FILTER_PROMPT = """You are an ICU clinical relevance auditor. Review a set of retrieved historical cases and determine which are clinically relevant to the current patient.

## Current Patient
{patient_summary}

## Retrieved Historical Cases
{retrieved_cases}

## Filtering Criteria
A retrieved case is CLINICALLY RELEVANT if:
- Similar severity level and organ dysfunction pattern
- Similar trajectory phase (both improving, both worsening, etc.)
- Comparable intervention context (similar treatment intensity)
- The case outcome provides actionable insight for the current patient

A case should be FILTERED OUT if:
- Very different severity (e.g., stable vs. critical)
- Opposite trajectory direction
- Completely different intervention context
- The case has low outcome confidence

For each retrieved case, output a relevance judgment.

Output JSON:
{{
  "relevant_cases": [0, 2, 5],
  "filtered_cases": [1, 3, 4],
  "filter_reasons": {{
    "1": "reason for filtering case 1",
    "3": "reason for filtering case 3"
  }},
  "overall_retrieval_quality": "high|medium|low",
  "recommendation": "use_all|use_filtered|use_none"
}}

JSON:"""


TRANSITION_UTILITY_PROMPT = """You are an ICU clinical outcome assessor. Evaluate whether a proposed intervention change (transition) is likely to be clinically beneficial.

## Patient State Before Transition
{state_before}

## Proposed Action
{action_description}

## Expected State After
{state_after}

## Historical Transition Data
{transition_context}

Assess the utility of this transition:
- utility_positive: Will the patient likely improve? (0.0-1.0)
- harm_risk: Risk of causing harm? (0.0-1.0)
- evidence_support: How well is this supported by evidence? (0.0-1.0)
- net_utility: Overall net clinical utility (-1.0 to 1.0, positive = beneficial)

Output JSON:
{{
  "utility_positive": 0.0-1.0,
  "harm_risk": 0.0-1.0,
  "evidence_support": 0.0-1.0,
  "net_utility": -1.0 to 1.0,
  "recommendation": "apply|caution|avoid",
  "reasoning": "brief clinical reasoning"
}}

JSON:"""


# ---------------------------------------------------------------------------
# LLM Clinical Supervisor
# ---------------------------------------------------------------------------

class LLMClinicalSupervisor:
    """
    LLM-based clinical supervisor for intelligent gating decisions.

    Usage:
        supervisor = LLMClinicalSupervisor(config, llm_call_fn)

        # Decision 1: Should memory be applied?
        gate = supervisor.memory_gate(patient_summary, match_count, ...)

        # Decision 2: Which retrieved cases are relevant?
        filtered = supervisor.retrieval_filter(patient_summary, retrieved_cases)

        # Decision 3: Is this transition beneficial?
        utility = supervisor.transition_utility(state_before, action, state_after)
    """

    def __init__(
        self,
        config: LLMSupervisorConfig,
        llm_call_fn=None,
    ):
        self.config = config
        self._llm_call = llm_call_fn or self._default_ollama_call

    # ------------------------------------------------------------------
    # Decision Point 1: Memory Gate
    # ------------------------------------------------------------------

    def memory_gate(
        self,
        patient_summary: str,
        match_count: int = 0,
        top_label: str = "",
        memory_confidence: float = 0.0,
        template_blend: float = 0.0,
        semantic_hits: int = 0,
    ) -> Dict[str, object]:
        """
        Decide whether memory-based correction should be applied.

        Returns: {apply_memory, confidence, reasoning, risk_if_applied, risk_if_silenced}
        """
        if not self.config.enable_memory_gate:
            return {"apply_memory": True, "confidence": 1.0, "reasoning": "Gate disabled"}

        # Rule-based pre-filter: if no matches, fast-path to False
        if match_count == 0 and semantic_hits == 0:
            return {
                "apply_memory": False,
                "confidence": 0.95,
                "reasoning": "No similar historical cases found. Applying empty memory correction would add noise.",
                "risk_if_applied": "high",
                "risk_if_silenced": "low",
            }

        prompt = MEMORY_GATE_PROMPT.format(
            patient_summary=patient_summary[:3000],
            match_count=match_count,
            top_label=top_label,
            memory_confidence=f"{memory_confidence:.3f}",
            template_blend=f"{template_blend:.3f}",
            semantic_hits=semantic_hits,
        )

        raw = self._llm_call(prompt)
        result = self._parse_json(raw)

        # Override: if LLM confidence is below threshold, default to NOT applying
        if result.get("confidence", 0) < self.config.memory_gate_confidence_threshold:
            result["apply_memory"] = False
            result["reasoning"] = (result.get("reasoning", "") +
                                   " [OVERRIDE: LLM confidence below threshold]")

        return result

    # ------------------------------------------------------------------
    # Decision Point 2: Retrieval Filter
    # ------------------------------------------------------------------

    def retrieval_filter(
        self,
        patient_summary: str,
        retrieved_cases: List[Dict[str, object]],
    ) -> Dict[str, object]:
        """
        Filter retrieved historical cases for clinical relevance.

        Returns: {relevant_cases, filtered_cases, filter_reasons, overall_retrieval_quality}
        """
        if not self.config.enable_retrieval_filter:
            return {
                "relevant_cases": list(range(len(retrieved_cases))),
                "filtered_cases": [],
                "filter_reasons": {},
                "overall_retrieval_quality": "high",
                "recommendation": "use_all",
            }

        if not retrieved_cases:
            return {
                "relevant_cases": [],
                "filtered_cases": [],
                "filter_reasons": {},
                "overall_retrieval_quality": "low",
                "recommendation": "use_none",
            }

        # Format cases for LLM review
        cases_text = self._format_retrieved_cases(retrieved_cases)

        prompt = RETRIEVAL_FILTER_PROMPT.format(
            patient_summary=patient_summary[:2000],
            retrieved_cases=cases_text[:4000],
        )

        raw = self._llm_call(prompt)
        result = self._parse_json(raw)

        # Validate indices
        max_idx = len(retrieved_cases) - 1
        result["relevant_cases"] = [i for i in result.get("relevant_cases", []) if 0 <= i <= max_idx]
        result["filtered_cases"] = [i for i in result.get("filtered_cases", []) if 0 <= i <= max_idx]

        return result

    # ------------------------------------------------------------------
    # Decision Point 3: Transition Utility
    # ------------------------------------------------------------------

    def transition_utility(
        self,
        state_before: str,
        action_description: str,
        state_after: str,
        transition_context: str = "",
    ) -> Dict[str, object]:
        """
        Assess whether a proposed intervention transition is clinically beneficial.
        Replaces the heuristic transition utility formula that was unreliable (sign flips).

        Returns: {utility_positive, harm_risk, evidence_support, net_utility, recommendation}
        """
        if not self.config.enable_utility_assessment:
            return {
                "utility_positive": 0.5,
                "harm_risk": 0.3,
                "evidence_support": 0.5,
                "net_utility": 0.0,
                "recommendation": "caution",
                "reasoning": "Utility assessment disabled",
            }

        prompt = TRANSITION_UTILITY_PROMPT.format(
            state_before=state_before[:2000],
            action_description=action_description[:1500],
            state_after=state_after[:2000],
            transition_context=transition_context[:1000] if transition_context else "No historical transition data available.",
        )

        raw = self._llm_call(prompt)
        result = self._parse_json(raw)

        # Ensure net_utility is in valid range
        nu = result.get("net_utility", 0)
        if isinstance(nu, (int, float)):
            result["net_utility"] = max(-1.0, min(1.0, float(nu)))

        return result

    # ------------------------------------------------------------------
    # Composite assessment
    # ------------------------------------------------------------------

    def comprehensive_assessment(
        self,
        patient_summary: str,
        memory_gate_result: Dict[str, object],
        retrieval_filter_result: Dict[str, object],
        transition_utility_result: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """
        Produce a comprehensive clinical safety assessment combining all three decisions.
        """
        flags = []
        risk_level = "low"

        # Gate assessment
        if not memory_gate_result.get("apply_memory", True):
            flags.append("MEMORY_SILENCED: Patient stable, memory correction suppressed")
        if memory_gate_result.get("risk_if_applied") == "high":
            flags.append("HIGH_RISK_IF_MEMORY_APPLIED")
            risk_level = "high"

        # Retrieval assessment
        rec = retrieval_filter_result.get("recommendation", "use_all")
        if rec == "use_none":
            flags.append("ALL_RETRIEVED_CASES_FILTERED: No clinically relevant matches")
            risk_level = "high"
        elif rec == "use_filtered":
            flags.append(f"RETRIEVAL_FILTERED: {len(retrieval_filter_result.get('filtered_cases', []))} cases removed")

        # Transition assessment
        if transition_utility_result:
            if transition_utility_result.get("recommendation") == "avoid":
                flags.append("TRANSITION_AVOID: High harm risk detected")
                risk_level = "high"
            elif transition_utility_result.get("recommendation") == "caution":
                flags.append("TRANSITION_CAUTION: Uncertainty in utility assessment")

        return {
            "flags": flags,
            "overall_risk_level": risk_level,
            "recommendation_ready": risk_level != "high",
            "memory_applied": memory_gate_result.get("apply_memory", False),
            "cases_used": len(retrieval_filter_result.get("relevant_cases", [])),
            "transition_recommended": (
                transition_utility_result.get("recommendation", "caution") == "apply"
                if transition_utility_result else False
            ),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_retrieved_cases(self, cases: List[Dict[str, object]]) -> str:
        """Format retrieved cases for LLM review."""
        lines = []
        for i, case in enumerate(cases[:10]):  # Cap at 10 for context window
            lines.append(f"Case {i}:")
            lines.append(f"  Label: {case.get('label', 'unknown')}")
            lines.append(f"  Similarity: {case.get('similarity', 0):.3f}")
            lines.append(f"  Risk Level: {case.get('risk_level', 'unknown')}")
            lines.append(f"  Intervention: {str(case.get('intervention', ''))[:200]}")
            if case.get('metadata'):
                meta = case['metadata']
                lines.append(f"  Severity: {meta.get('severity_bin', 'unknown')}")
                lines.append(f"  Trend: {meta.get('trend_bin', 'unknown')}")
            lines.append("")
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
        return {}

    def _default_ollama_call(self, prompt: str, model: str = "", temperature: Optional[float] = None) -> str:
        """Default Ollama HTTP API call."""
        import urllib.request

        model_name = model or self.config.supervisor_model
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
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "")
        except Exception as e:
            print(f"[LLMSupervisor] Ollama call failed: {e}")
            return "{}"
