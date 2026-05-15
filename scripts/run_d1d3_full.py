"""
D1-D3 diagnostic experiment — single-command entry point.

D1: Enable persistent memory reuse (--prime-persistent-memory-before-fit)
D2: Add validation set memory_gain_audit
D3: Per-case retrieval source tracing for top harmed/helped

Usage:
  python scripts/run_d1d3_full.py
  python scripts/run_d1d3_full.py --smoke
  python scripts/run_d1d3_full.py --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DRY_RUN = False


def _run(cmd: list[str], description: str) -> int:
    print(f"\n{'='*60}")
    print(f"[d1d3] {description}")
    print(f"[d1d3] {' '.join(cmd)}")
    print(f"{'='*60}\n", flush=True)
    if _DRY_RUN:
        return 0
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def parse_args():
    p = argparse.ArgumentParser(description="D1-D3 diagnostics: persistent reuse + val audit + retrieval trace")
    p.add_argument("--python-exe", default=sys.executable)
    p.add_argument("--eicu-max-series", type=int, default=512)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--persistent-memory-store", default=r".\output\persistent_memory\eicu_full_train_store_kg258")
    p.add_argument("--persistent-memory-allowed-splits", default="train,external")
    p.add_argument("--output-dir", default=r".\output\formal\d1_d3_diagnostics")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true", help="8 series, 1 epoch, batch 4")
    return p.parse_args()


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
    return _run(cmd, "Running 3-group D1-D3 ablation")


def main() -> int:
    global _DRY_RUN
    args = parse_args()
    _DRY_RUN = bool(args.dry_run)
    if args.smoke:
        args.eicu_max_series = 8
        args.epochs = 1
        args.batch_size = 4
        args.persistent_memory_store = r".\output\persistent_memory\eicu_smoke_store_kg258"
        args.output_dir = r".\output\formal\d1_d3_diagnostics\smoke"
        print("[d1d3] Smoke mode: 8 series, 1 epoch, batch 4")
    if args.dry_run:
        print("[d1d3] DRY RUN\n")

    rc = run_ablations(args)
    if rc != 0:
        print(f"[d1d3] Ablation failed (exit {rc})", file=sys.stderr)
        return rc

    print(f"\n[d1d3] Done. Results in: {Path(args.output_dir)}")
    print("[d1d3] Key fields to compare:")
    print("  memory_gain_audit (test set)")
    print("  val_memory_gain_audit (validation set)")
    print("  persistent_memory.reuse_audit (persistent memory usage)")
    print("  transition_gate_audit.transition_case_details[].retrieval_trace (retrieval source)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
