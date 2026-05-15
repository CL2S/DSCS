"""
R3 safe transition residual — full experiment pipeline.

Single-command entry point that:
  1. Rebuilds the persistent memory store with current kg-enabled schema.
  2. Runs the four-group ablation:
       a) no transition memory (baseline)
       b) transition + no signature (weight=0.0)
       c) transition + signature (weight=0.08, default safe gate)
       d) transition + signature + aggressive safe gate

Usage (dry-run first to review commands):
  python scripts/run_r3_full_experiment.py --dry-run

Full run:
  python scripts/run_r3_full_experiment.py

Quick smoke test (8 series, 1 epoch):
  python scripts/run_r3_full_experiment.py --eicu-max-series 8 --epochs 1 --batch-size 4 --smoke
"""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


_DRY_RUN = False


def _run(cmd: list[str], description: str, cwd: str | None = None) -> int:
    print(f"\n{'='*60}")
    print(f"[r3] {description}")
    print(f"[r3] {' '.join(cmd)}")
    print(f"{'='*60}\n", flush=True)
    if _DRY_RUN:
        return 0
    return subprocess.run(cmd, cwd=cwd or str(PROJECT_ROOT)).returncode


def parse_args():
    p = argparse.ArgumentParser(description="R3 safe transition residual — full experiment pipeline")
    p.add_argument("--python-exe", default=sys.executable)
    p.add_argument("--eicu-max-series", type=int, default=512)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--persistent-memory-store", default=r".\output\persistent_memory\eicu_full_train_store_kg258")
    p.add_argument("--persistent-memory-allowed-splits", default="train,external")
    p.add_argument("--memory-gain-audit-max-cases", type=int, default=0)
    p.add_argument("--memory-gain-audit-top-k", type=int, default=20)
    p.add_argument("--transition-signature-match-weight", type=float, default=0.08)
    p.add_argument("--output-dir", default=r".\output\formal\r3_safe_transition_gate")
    p.add_argument("--skip-persistent-build", action="store_true", help="Skip store rebuild (use if store already exists)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Small-scale smoke test (overrides series/epochs/batch)")
    return p.parse_args()


def build_store(args) -> int:
    """Step 1: Build persistent store with current kg258 schema."""
    cmd = [
        args.python_exe,
        str(PROJECT_ROOT / "run_forecasting_experiment.py"),
        "--dataset-format", "eicu_sepsis3",
        "--eicu-max-series", str(args.eicu_max_series),
        "--enable-kg",
        "--persistent-memory-store", args.persistent_memory_store,
        "--build-persistent-memory-only",
        "--persistent-memory-build-splits", "train",
        "--persistent-memory-allowed-splits", args.persistent_memory_allowed_splits,
    ]
    return _run(cmd, "Step 1/2: building persistent store (kg258 schema)")


def run_ablations(args) -> int:
    """Step 2: Run the four-group ablation."""
    ablation_script = str(PROJECT_ROOT / "scripts" / "run_mvp3_transition_signature_ablations.py")
    cmd = [
        args.python_exe,
        ablation_script,
        "--eicu-max-series", str(args.eicu_max_series),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--persistent-memory-store", args.persistent_memory_store,
        "--persistent-memory-allowed-splits", args.persistent_memory_allowed_splits,
        "--memory-gain-audit-max-cases", str(args.memory_gain_audit_max_cases),
        "--memory-gain-audit-top-k", str(args.memory_gain_audit_top_k),
        "--transition-signature-match-weight", str(args.transition_signature_match_weight),
        "--output-dir", args.output_dir,
    ]
    return _run(cmd, "Step 2/2: running 4-group ablation")


def main() -> int:
    global _DRY_RUN
    args = parse_args()
    _DRY_RUN = bool(args.dry_run)
    if args.smoke:
        args.eicu_max_series = 8
        args.epochs = 1
        args.batch_size = 4
        args.persistent_memory_store = r".\output\persistent_memory\eicu_smoke_store_kg258"
        args.output_dir = r".\output\formal\r3_safe_transition_gate\smoke"
        print("[r3] Smoke mode: 8 series, 1 epoch, batch 4")
    if args.dry_run:
        print("[r3] DRY RUN — printing commands only\n")

    if not args.skip_persistent_build:
        rc = build_store(args)
        if rc != 0:
            print(f"[r3] Store build failed (exit {rc})", file=sys.stderr)
            return rc

    rc = run_ablations(args)
    if rc != 0:
        print(f"[r3] Ablation run failed (exit {rc})", file=sys.stderr)
        return rc

    print("\n[r3] Pipeline complete.")
    print(f"[r3] Results in: {Path(args.output_dir)}")
    print("[r3] Key comparison command:")
    print(f'  python -c "import json, pathlib; '
          f'files = sorted(pathlib.Path(r\'{args.output_dir}\').glob(\'eicu_*.json\')); '
          f'[print(f.name, json.loads(f.read_text(encoding=\"utf-8\")).get(\"memory_effectiveness\",{{}}).get(\"improvement_mae\")) for f in files]"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
