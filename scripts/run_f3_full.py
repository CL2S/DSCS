"""
F2+F3: Build 1024 persistent store + verification experiment.

F1: Code defaults restored (already done — no R5 penalties in template_blend, no harm boost)
F2: Build persistent store from 1024 series (provides ~1536 extra stays beyond experiment's 512)
F3: Run 3-group experiment at 512 series to verify:
      G1: no_transition        — baseline, no transition, persist ON
      G2: restored_with_persist — restored code + transition + persist from 1024 store
      G3: restored_no_persist   — restored code + transition + persist OFF (ablation)

Usage:
  python scripts/run_f3_full.py                    # full run (build store + experiment)
  python scripts/run_f3_full.py --skip-store-build # skip store build (if already built)
  python scripts/run_f3_full.py --dry-run          # print commands only
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DRY_RUN = False


def _run(cmd: list[str], description: str) -> int:
    print(f"\n{'='*60}")
    print(f"[f3] {description}")
    print(f"[f3] {' '.join(cmd)}")
    print(f"{'='*60}\n", flush=True)
    if _DRY_RUN:
        return 0
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def parse_args():
    p = argparse.ArgumentParser(description="F2+F3: Build 1024 store + 3-group verification experiment")
    p.add_argument("--python-exe", default=sys.executable)
    p.add_argument("--store-max-series", type=int, default=1024, help="Series count for persistent store build")
    p.add_argument("--experiment-max-series", type=int, default=512, help="Series count for experiment")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--persistent-memory-store", default=r".\output\persistent_memory\eicu_full_1024_kg258")
    p.add_argument("--output-dir", default=r".\output\formal\f3_restored_baseline")
    p.add_argument("--skip-store-build", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_store(args) -> int:
    cmd = [
        args.python_exe,
        str(PROJECT_ROOT / "run_forecasting_experiment.py"),
        "--dataset-format", "eicu_sepsis3",
        "--eicu-max-series", str(args.store_max_series),
        "--enable-kg",
        "--persistent-memory-store", args.persistent_memory_store,
        "--build-persistent-memory-only",
        "--persistent-memory-build-splits", "train",
        "--persistent-memory-allowed-splits", "train,external",
    ]
    return _run(cmd, f"Step 1/2: build persistent store ({args.store_max_series} series)")


def run_ablations(args) -> int:
    ablation = str(PROJECT_ROOT / "scripts" / "run_mvp3_transition_signature_ablations.py")
    cmd = [
        args.python_exe, ablation,
        "--eicu-max-series", str(args.experiment_max_series),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--persistent-memory-store", args.persistent_memory_store,
        "--persistent-memory-allowed-splits", "train,external",
        "--memory-gain-audit-max-cases", "0",
        "--memory-gain-audit-top-k", "20",
        "--output-dir", args.output_dir,
    ]
    return _run(cmd, f"Step 2/2: 3-group experiment ({args.experiment_max_series} series)")


def main() -> int:
    global _DRY_RUN
    args = parse_args()
    _DRY_RUN = bool(args.dry_run)
    if args.dry_run:
        print("[f3] DRY RUN\n")

    if not args.skip_store_build:
        rc = build_store(args)
        if rc != 0:
            print(f"[f3] Store build failed (exit {rc})", file=sys.stderr)
            return rc

    rc = run_ablations(args)
    if rc != 0:
        print(f"[f3] Experiment failed (exit {rc})", file=sys.stderr)
        return rc

    print(f"\n[f3] Done. Results: {Path(args.output_dir)}")
    print("[f3] Verify:")
    print("  1. improvement_mae should be close to R4's 0.063")
    print("  2. persistent_memory.loaded_persistent_samples > 0")
    print("  3. val_memory_gain_audit should show consistent val/test behavior")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
