#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fused System — LLM-driven SOFA prediction integrated with manifold memory bank,
KG integration, persistent experience store, and clinical supervision.

This package fuses two previously independent systems:
    System 1: LLM-based SOFA prediction & evaluation (DSPy + Ollama)
    System 2: Deep learning manifold memory with KG & counterfactual planning

The fusion replaces DL components (encoder, predictor, counterfactual) with
LLM equivalents while keeping the memory/KG/persistence infrastructure intact.
"""

from src.fusion_config import (
    ClinicalStateSignature,
    FusedPredictionOutput,
    FusedSystemConfig,
    LLMEncoderConfig,
    LLMPredictorConfig,
    LLMCounterfactualConfig,
    LLMSupervisorConfig,
)

from src.fused_system import FusedSystem

__all__ = [
    "FusedSystem",
    "FusedSystemConfig",
    "FusedPredictionOutput",
    "ClinicalStateSignature",
    "LLMEncoderConfig",
    "LLMPredictorConfig",
    "LLMCounterfactualConfig",
    "LLMSupervisorConfig",
]
