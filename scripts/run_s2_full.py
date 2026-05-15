"""
S2: Stabilized training (8 epochs) + transition penalties removed.

Two-stage experiment:
  Stage 1 (smoke):  256 series, 8 epochs — quick verification
  Stage 2 (full):   512 series, 8 epochs — formal 3-group ablation

Usage:
  python scripts/run_s2_full.py --smoke   # quick verification (256 series)
  python scripts/run_s2_full.py           # full experiment (512 series)
  python scripts/run_s2_full.py --dry-run # print commands only
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DRY_RUN = False


def _run(cmd: list[str], description: str) -> int:
    print(f"\n{'='*60}")
    print(f"[s2] {description}")
    print(f"[s2] {' '.join(cmd)}")
    print(f"{'='*60}\n", flush=True)
    if _DRY_RUN:
        return 0
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def parse_args():
    p = argparse.ArgumentParser(description="S2: 8-epoch experiment, transition penalties removed")
    p.add_argument("--python-exe", default=sys.executable)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--persistent-memory-store", default=r".\output\persistent_memory\eicu_full_1024_kg258")
    p.add_argument("--output-dir", default=r".\output\formal\s2_stabilized")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true", help="256 series quick verification")
    return p.parse_args()


def run_ablations(args, max_series: int) -> int:
    ablation = str(PROJECT_ROOT / "scripts" / "run_mvp3_transition_signature_ablations.py")
    cmd = [
        args.python_exe, ablation,
        "--eicu-max-series", str(max_series),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--persistent-memory-store", args.persistent_memory_store,
        "--persistent-memory-allowed-splits", "train,external",
        "--memory-gain-audit-max-cases", "0",
        "--memory-gain-audit-top-k", "20",
        "--output-dir", args.output_dir,
    ]
    return _run(cmd, f"3-group experiment ({max_series} series, {args.epochs} epochs)")


def main() -> int:
    global _DRY_RUN
    args = parse_args()
    _DRY_RUN = bool(args.dry_run)

    max_series = 256 if args.smoke else 512
    if args.smoke:
        print("[s2] SMOKE: 256 series, 8 epochs")
    if args.dry_run:
        print("[s2] DRY RUN\n")

    rc = run_ablations(args, max_series)
    if rc != 0:
        print(f"[s2] Failed (exit {rc})", file=sys.stderr)
        return rc

    print(f"\n[s2] Done. Results: {Path(args.output_dir)}")
    print("[s2] Verify: improvement_mae should approach R4's 0.063")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
