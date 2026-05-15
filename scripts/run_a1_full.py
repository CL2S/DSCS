"""
A1: Branch decoupling verification experiment.

Three groups compare encoding space separation:
  G1: no_decoupling — shared encoding (--disable-branch-decoupling)
  G2: decouple_default — post_projection mode, scale=0.30 (current defaults)
  G3: decouple_strong — post_projection mode, scale=0.50

Usage:
  python scripts/run_a1_full.py              # full 512-series experiment
  python scripts/run_a1_full.py --smoke      # 256-series quick verification
  python scripts/run_a1_full.py --dry-run    # print commands only
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DRY_RUN = False


def _run(cmd: list[str], description: str) -> int:
    print(f"\n{'='*60}")
    print(f"[a1] {description}")
    print(f"[a1] {' '.join(cmd)}")
    print(f"{'='*60}\n", flush=True)
    if _DRY_RUN:
        return 0
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def parse_args():
    p = argparse.ArgumentParser(description="A1: Branch decoupling verification")
    p.add_argument("--python-exe", default=sys.executable)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--persistent-memory-store", default=r".\output\persistent_memory\eicu_full_1024_kg258")
    p.add_argument("--output-dir", default=r".\output\formal\a1_branch_decoupling")
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
        print("[a1] SMOKE: 256 series, 8 epochs")
    if args.dry_run:
        print("[a1] DRY RUN\n")

    # G1: no decoupling (shared encoding)
    g1 = _base(args, max_series, f"{prefix}_no_decoupling.json")
    g1.extend(["--disable-branch-decoupling"])

    # G2: default decoupling (post_projection, scale=0.30)
    g2 = _base(args, max_series, f"{prefix}_decouple_default.json")
    # uses defaults: post_projection, scale=0.30

    # G3: strong decoupling (post_projection, scale=0.50)
    g3 = _base(args, max_series, f"{prefix}_decouple_strong.json")
    g3.extend(["--retrieval-projection-scale", "0.50"])

    for name, cmd in [("no_decoupling", g1), ("decouple_default", g2), ("decouple_strong", g3)]:
        rc = _run(cmd, name)
        if rc != 0:
            print(f"[a1] {name} failed (exit {rc})", file=sys.stderr)
            return rc

    print(f"\n[a1] Done. Results: {Path(args.output_dir)}")
    print("[a1] Key comparison: improvement_mae, memory_harm_quality, trans_utility across 3 groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
