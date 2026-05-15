"""
Phase A verification experiment: A1 + A2 + A3 combined.

Four groups:
  G1: baseline — no decoupling, no uncertainty guardrail
  G2: a1_only — decoupling=post_projection, scale=0.30
  G3: a1_a2 — decoupling + uncertainty guardrail (threshold=2.0)
  G4: a1_a2_strong — decoupling + uncertainty guardrail (threshold=1.5, stricter)

Usage:
  python scripts/run_phase_a_full.py              # full 512-series
  python scripts/run_phase_a_full.py --smoke      # 256-series quick verification
  python scripts/run_phase_a_full.py --dry-run    # print commands only
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DRY_RUN = False


def _run(cmd: list[str], description: str) -> int:
    print(f"\n{'='*60}")
    print(f"[phase_a] {description}")
    print(f"[phase_a] {' '.join(cmd)}")
    print(f"{'='*60}\n", flush=True)
    if _DRY_RUN:
        return 0
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def parse_args():
    p = argparse.ArgumentParser(description="Phase A: decoupling + uncertainty + diagnostics")
    p.add_argument("--python-exe", default=sys.executable)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--persistent-memory-store", default=r".\output\persistent_memory\eicu_full_1024_kg258")
    p.add_argument("--output-dir", default=r".\output\formal\phase_a")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true", help="256 series quick test")
    return p.parse_args()


def _base(args, max_series: int, output_name: str) -> list[str]:
    return [
        args.python_exe,
        str(PROJECT_ROOT / "run_forecasting_experiment.py"),
        "--dataset-format", "eicu_sepsis3",
        "--eicu-max-series", str(max_series),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--enable-kg",
        "--enable-transition-memory",
        "--transition-signature-match-weight", "0.08",
        "--disable-transition-partial-signature",
        "--persistent-memory-store", args.persistent_memory_store,
        "--persistent-memory-allowed-splits", "train,external",
        "--prime-persistent-memory-before-fit",
        "--memory-gain-audit-max-cases", "0",
        "--memory-gain-audit-top-k", "20",
        "--output-json", str(Path(args.output_dir) / output_name),
    ]


def main() -> int:
    global _DRY_RUN
    args = parse_args()
    _DRY_RUN = bool(args.dry_run)
    max_series = 256 if args.smoke else 512
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    prefix = f"eicu_{max_series}"

    if args.smoke:
        print("[phase_a] SMOKE: 256 series, 8 epochs")
    if args.dry_run:
        print("[phase_a] DRY RUN\n")

    # G1: baseline — shared encoding, no uncertainty guardrail
    g1 = _base(args, max_series, f"{prefix}_baseline.json")
    g1.extend(["--disable-branch-decoupling", "--uncertainty-delta-threshold", "0"])

    # G2: A1 only — decoupling enabled, no uncertainty guardrail
    g2 = _base(args, max_series, f"{prefix}_a1_decouple.json")
    g2.extend(["--uncertainty-delta-threshold", "0"])

    # G3: A1+A2 — decoupling + uncertainty guardrail (default threshold=2.0)
    g3 = _base(args, max_series, f"{prefix}_a1_a2_uncertainty.json")

    # G4: A1+A2 strong — decoupling + stricter uncertainty (threshold=1.5)
    g4 = _base(args, max_series, f"{prefix}_a1_a2_uncertainty_strict.json")
    g4.extend(["--uncertainty-delta-threshold", "1.5"])

    for name, cmd in [
        ("baseline", g1),
        ("a1_decouple", g2),
        ("a1_a2_uncertainty", g3),
        ("a1_a2_uncertainty_strict", g4),
    ]:
        rc = _run(cmd, name)
        if rc != 0:
            print(f"[phase_a] {name} failed (exit {rc})", file=sys.stderr)
            return rc

    print(f"\n[phase_a] Done. Results: {Path(args.output_dir)}")
    print("[phase_a] Key fields to compare:")
    print("  improvement_mae, helped/harmed, memory_harm_quality")
    print("  diagnostic_summary.bottleneck_ranking")
    print("  uncertainty_analysis.forecast_mean_std")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
