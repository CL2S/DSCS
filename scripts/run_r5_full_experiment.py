"""
R5 per-pattern harm control — full experiment pipeline.

Single-command entry point:
  python scripts/run_r5_full_experiment.py

Smoke test:
  python scripts/run_r5_full_experiment.py --eicu-max-series 8 --epochs 1 --batch-size 4 --smoke
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DRY_RUN = False


def _run(cmd: list[str], description: str) -> int:
    print(f"\n{'='*60}")
    print(f"[r5] {description}")
    print(f"[r5] {' '.join(cmd)}")
    print(f"{'='*60}\n", flush=True)
    if _DRY_RUN:
        return 0
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def parse_args():
    p = argparse.ArgumentParser(description="R5 per-pattern harm control + direction filtering — full experiment")
    p.add_argument("--python-exe", default=sys.executable)
    p.add_argument("--eicu-max-series", type=int, default=512)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--persistent-memory-store", default=r".\output\persistent_memory\eicu_full_train_store_kg258")
    p.add_argument("--persistent-memory-allowed-splits", default="train,external")
    p.add_argument("--output-dir", default=r".\output\formal\r5_harm_control")
    p.add_argument("--skip-persistent-build", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true", help="8 series, 1 epoch, batch 4")
    return p.parse_args()


def build_store(args) -> int:
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
    return _run(cmd, "Step 1/2: building persistent store")


def run_ablations(args) -> int:
    cmd = [
        args.python_exe,
        str(PROJECT_ROOT / "scripts" / "run_mvp3_transition_signature_ablations.py"),
        "--eicu-max-series", str(args.eicu_max_series),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--persistent-memory-store", args.persistent_memory_store,
        "--persistent-memory-allowed-splits", args.persistent_memory_allowed_splits,
        "--memory-gain-audit-max-cases", "0",
        "--memory-gain-audit-top-k", "20",
        "--output-dir", args.output_dir,
    ]
    return _run(cmd, "Step 2/2: running 4-group R5 ablation")


def main() -> int:
    global _DRY_RUN
    args = parse_args()
    _DRY_RUN = bool(args.dry_run)
    if args.smoke:
        args.eicu_max_series = 8
        args.epochs = 1
        args.batch_size = 4
        args.persistent_memory_store = r".\output\persistent_memory\eicu_smoke_store_kg258"
        args.output_dir = r".\output\formal\r5_harm_control\smoke"
        print("[r5] Smoke mode: 8 series, 1 epoch, batch 4")
    if args.dry_run:
        print("[r5] DRY RUN\n")

    if not args.skip_persistent_build:
        rc = build_store(args)
        if rc != 0:
            print(f"[r5] Store build failed (exit {rc})", file=sys.stderr)
            return rc

    rc = run_ablations(args)
    if rc != 0:
        print(f"[r5] Ablation failed (exit {rc})", file=sys.stderr)
        return rc

    print(f"\n[r5] Done. Results in: {Path(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
