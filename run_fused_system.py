#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run script for the Fused System — integrates LLM SOFA prediction with
the manifold memory bank, KG, and clinical supervision.

Usage:
    # Interactive mode (manual patient input)
    python run_fused_system.py --mode interactive

    # Single prediction
    python run_fused_system.py --mode single --input "patient text..." --intervention "..."

    # Batch processing from icu_stays_descriptions88.json
    python run_fused_system.py --mode batch --data icu_stays_descriptions88.json

    # Demo with built-in test case
    python run_fused_system.py --mode demo

    # Memory bank summary
    python run_fused_system.py --mode summary

    # Load historical experiences into memory bank
    python run_fused_system.py --mode load --results output/best_result/
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.fusion_config import FusedSystemConfig
from src.fused_system import FusedSystem


# ---------------------------------------------------------------------------
# Demo patient case
# ---------------------------------------------------------------------------

DEMO_PATIENT = """ICU住院编号 30117609

患者基本信息：65岁男性，体重75kg，ICU住院第2天。

当前临床表现：
- 体温：38.5°C，心率：102次/分，呼吸频率：24次/分，血压：95/60 mmHg
- MAP：72 mmHg，SpO2：94%（鼻导管吸氧3L/min）
- GCS：14分（E3V5M6）

实验室检查：
- 白细胞：14.2×10⁹/L，中性粒细胞百分比：88%
- 乳酸：3.8 mmol/L（较6小时前上升0.9）
- 降钙素原：12.5 ng/mL
- 肌酐：1.6 mg/dL（较基线上升0.4）
- 总胆红素：1.8 mg/dL
- 血小板：145×10³/μL
- PaO2/FiO2：285

SOFA评分变化（过去6小时）：
- 呼吸系统：2→2
- 凝血系统：1→1
- 肝脏：1→1
- 心血管系统：1→2
- 中枢神经：0→0
- 肾脏：1→2
- 总SOFA：6→8

当前治疗：
- 头孢吡肟2g q8h（已用36小时）
- 去甲肾上腺素0.08 μg/kg/min
- 乳酸林格液维持输液
- 血糖管理"""

DEMO_INTERVENTION = "去甲肾上腺素0.08 μg/kg/min，头孢吡肟2g q8h，乳酸林格液维持输液，计划评估是否需要升级抗生素"


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def run_interactive(system: FusedSystem):
    """Interactive mode — user inputs patient descriptions one at a time."""
    print("\n" + "=" * 60)
    print("  Fused System — Interactive Mode")
    print("  Type 'quit' to exit, 'demo' for demo case")
    print("=" * 60 + "\n")

    while True:
        print("-" * 40)
        patient_text = input("Patient description (or 'demo'/'quit'): ").strip()
        if patient_text.lower() == 'quit':
            break
        if patient_text.lower() == 'demo':
            patient_text = DEMO_PATIENT

        intervention = input("Current intervention (press Enter to skip): ").strip()
        if not intervention and patient_text == DEMO_PATIENT:
            intervention = DEMO_INTERVENTION

        patient_id = input("Patient ID (press Enter for auto): ").strip()
        if not patient_id:
            import re
            m = re.search(r'ICU住院编号\s*(\d+)', patient_text)
            patient_id = m.group(1) if m else f"manual_{int(time.time())}"

        print("\nRunning prediction pipeline...")
        output = system.predict(
            patient_text=patient_text,
            intervention=intervention,
            patient_id=patient_id,
            enable_counterfactual=True,
        )
        _print_output(output)


# ---------------------------------------------------------------------------
# Single mode
# ---------------------------------------------------------------------------

def run_single(system: FusedSystem, patient_text: str, intervention: str, patient_id: str):
    """Single prediction mode."""
    output = system.predict(
        patient_text=patient_text,
        intervention=intervention,
        patient_id=patient_id or "single_case",
        enable_counterfactual=True,
    )
    _print_output(output)
    return output


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def run_batch(system: FusedSystem, data_path: str, limit: int = 0):
    """Batch processing from JSON data file."""
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle different data formats
    if isinstance(data, list):
        cases = data
    elif isinstance(data, dict):
        cases = list(data.values()) if any(isinstance(v, dict) for v in data.values()) else [data]
    else:
        print("Unexpected data format")
        return

    if limit > 0:
        cases = cases[:limit]

    print(f"\nProcessing {len(cases)} cases...")
    results = []

    for i, case in enumerate(cases):
        if isinstance(case, str):
            patient_text = case
            intervention = ""
            pid = f"batch_{i}"
        elif isinstance(case, dict):
            patient_text = case.get("description", case.get("text", str(case)))
            intervention = case.get("intervention", "")
            pid = case.get("stay_id", case.get("patient_id", f"batch_{i}"))
        else:
            continue

        print(f"[{i+1}/{len(cases)}] Processing {pid}...")
        output = system.predict(
            patient_text=str(patient_text),
            intervention=str(intervention),
            patient_id=str(pid),
            enable_counterfactual=False,  # Skip for speed
        )
        results.append({
            "patient_id": pid,
            "risk_level": output.risk_level,
            "hourly_sofa_totals": output.hourly_sofa_totals,
            "memory_confidence": output.memory_confidence,
        })

    # Save batch results
    out_path = Path("output") / "fused_batch_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


# ---------------------------------------------------------------------------
# Load mode
# ---------------------------------------------------------------------------

def run_load(system: FusedSystem, results_dir: str):
    """Load historical result files into memory bank and persistent store."""
    result_files = glob.glob(os.path.join(results_dir, "result_*.json"))
    if not result_files:
        print(f"No result_*.json files found in {results_dir}")
        return

    print(f"Loading {len(result_files)} result files...")
    loaded = system.load_experiences(result_files)
    print(f"Loaded {loaded} experiences into memory bank.")

    summary = system.get_memory_summary()
    print("\nMemory Bank Summary:")
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _print_output(output):
    """Pretty-print prediction output."""
    print("\n" + "=" * 60)
    print("  PREDICTION RESULT")
    print("=" * 60)

    # Clinical State
    if output.clinical_state:
        cs = output.clinical_state
        print(f"\n📋 Clinical State:")
        print(f"   Severity: {cs.severity_bin} | Trend: {cs.trend_bin} | "
              f"Volatility: {cs.volatility_bin}")
        print(f"   Pattern: {cs.pattern_label} | Trajectory: {cs.trajectory_label}")
        print(f"   SOFA Current: {cs.sofa_current} | 6h Trend: {cs.sofa_trend_6h:+.1f}")

    # Risk Assessment
    print(f"\n⚠️  Risk Assessment:")
    print(f"   Risk Level: {output.risk_level.upper()}")
    print(f"   Confidence: {output.risk_confidence:.2f}")
    if output.prediction_std > 0:
        print(f"   Uncertainty (std): {output.prediction_std:.3f}")

    # SOFA Predictions
    if output.hourly_sofa_totals:
        print(f"\n📈 SOFA Predictions (8-hour forecast):")
        hours = sorted(output.hourly_sofa_totals.keys(), key=lambda x: int(x))
        values = [output.hourly_sofa_totals[h] for h in hours]
        print(f"   Hours: {'  '.join(hours)}")
        print(f"   SOFA:  {'  '.join(f'{v:.1f}' for v in values)}")
        if values:
            print(f"   Trend: {values[0]:.1f} → {values[-1]:.1f} "
                  f"({values[-1]-values[0]:+.1f})")

    # Memory Context
    print(f"\n🧠 Memory Context:")
    print(f"   Confidence: {output.memory_confidence:.3f}")
    print(f"   Matched Cases: {output.memory_matched_count}")
    print(f"   Template Blend: {output.template_blend_weight:.3f}")
    print(f"   Semantic Hits: {output.semantic_hit_count}")

    # Supervisor
    print(f"\n🛡️  Clinical Supervisor:")
    print(f"   Memory Gate: {'APPLIED' if output.supervisor_gate_decision else 'SILENCED'}")
    if output.supervisor_gate_reason:
        reason = output.supervisor_gate_reason[:120]
        print(f"   Reason: {reason}...")
    print(f"   Cases Filtered: {output.supervisor_filtered_count}")

    # KG
    if output.kg_features:
        active_kg = [k for k, v in output.kg_features.items() if v > 0]
        print(f"\n🔗 Knowledge Graph:")
        print(f"   Active Flags: {', '.join(active_kg[:8]) if active_kg else 'none'}")
        print(f"   Guideline Alignment: {output.guideline_alignment:.2f}")

    # Reasoning
    if output.reasoning:
        print(f"\n💡 Clinical Reasoning:")
        for line in output.reasoning.split('. ')[:3]:
            if line.strip():
                print(f"   • {line.strip()}.")

    # Counterfactual
    if output.counterfactual_candidates:
        print(f"\n🔄 Counterfactual Alternatives ({len(output.counterfactual_candidates)} candidates):")
        for i, cand in enumerate(output.counterfactual_candidates[:3]):
            print(f"   {i+1}. {cand.get('strategy_name', 'Unknown')}")
            print(f"      Expected SOFA change: {cand.get('expected_sofa_change_8h', 0):+.1f}")
            print(f"      Risk: {cand.get('risk_level', 'unknown')} | "
                  f"Confidence: {cand.get('confidence', 0):.2f}")
        if output.supervisor_utility_score != 0:
            print(f"   Transition Utility: {output.supervisor_utility_score:+.3f}")

    # Timing
    elapsed = output.metadata.get("elapsed_seconds", 0)
    print(f"\n⏱️  Pipeline completed in {elapsed:.1f}s")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fused System — LLM SOFA Prediction + Memory Bank + KG + Clinical Supervisor"
    )
    parser.add_argument("--mode", choices=["interactive", "single", "batch", "demo", "summary", "load"],
                        default="demo", help="Run mode")
    parser.add_argument("--input", type=str, default="", help="Patient description text")
    parser.add_argument("--intervention", type=str, default="", help="Current intervention")
    parser.add_argument("--patient-id", type=str, default="", help="Patient/stay ID")
    parser.add_argument("--data", type=str, default="icu_stays_descriptions88.json",
                        help="Path to batch data file")
    parser.add_argument("--results", type=str, default="output/best_result/",
                        help="Path to result files directory (for load mode)")
    parser.add_argument("--limit", type=int, default=0, help="Limit batch processing to N cases")
    parser.add_argument("--kg-dir", type=str, default="", help="Path to KG directory")
    parser.add_argument("--persist-dir", type=str, default="output/fused_experiences/",
                        help="Path to persistent experience store")
    parser.add_argument("--no-persist", action="store_true", help="Disable persistent experience store")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Build config
    config = FusedSystemConfig()
    config.verbose = args.verbose

    if args.kg_dir:
        config.kg_directory = args.kg_dir

    if args.persist_dir:
        config.persist_directory = args.persist_dir

    if args.no_persist:
        config.enable_persistence = False

    # Initialize system
    print("Initializing Fused System...")
    print(f"  Encoder model: {config.encoder.encoder_model}")
    print(f"  Predictor model: {config.predictor.predictor_model}")
    print(f"  Ensemble: {', '.join(config.predictor.ensemble_models)}")
    print(f"  Supervisor model: {config.supervisor.supervisor_model}")
    print(f"  KG: {'enabled' if config.kg_directory else 'disabled'}")
    print(f"  Persistence: {'enabled' if config.enable_persistence else 'disabled'}")

    system = FusedSystem(config)

    # Route to mode
    if args.mode == "interactive":
        run_interactive(system)

    elif args.mode == "single":
        if not args.input:
            print("Error: --input required for single mode")
            sys.exit(1)
        run_single(system, args.input, args.intervention, args.patient_id)

    elif args.mode == "batch":
        run_batch(system, args.data, args.limit)

    elif args.mode == "demo":
        print("\nRunning demo with built-in test case...\n")
        output = run_single(system, DEMO_PATIENT, DEMO_INTERVENTION, "30117609_demo")
        system.write_experience(output)

    elif args.mode == "summary":
        summary = system.get_memory_summary()
        print("\nMemory Bank Summary:")
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

    elif args.mode == "load":
        run_load(system, args.results)


if __name__ == "__main__":
    main()
