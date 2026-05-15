import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _add_bool_flag(args, flag: str, enabled: bool) -> None:
    if enabled:
        args.append(flag)


def _base_command(args, output_json: Path) -> list[str]:
    command = [
        args.python_exe,
        str(PROJECT_ROOT / "run_forecasting_experiment.py"),
        "--dataset-format",
        "eicu_sepsis3",
        "--eicu-max-series",
        str(args.eicu_max_series),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--persistent-memory-store",
        args.persistent_memory_store,
        "--persistent-memory-allowed-splits",
        args.persistent_memory_allowed_splits,
        "--memory-gain-audit-max-cases",
        str(args.memory_gain_audit_max_cases),
        "--memory-gain-audit-top-k",
        str(args.memory_gain_audit_top_k),
        "--output-json",
        str(output_json),
    ]
    _add_bool_flag(command, "--enable-kg", args.enable_kg)
    if not args.disable_persistent_reuse:
        _add_bool_flag(command, "--prime-persistent-memory-before-fit", True)
    else:
        _add_bool_flag(command, "--disable-persistent-memory-reuse", True)
    _add_bool_flag(command, "--skip-posthoc-diagnostics", args.skip_posthoc_diagnostics)
    return command


def _suite_commands(args) -> list[tuple[str, list[str]]]:
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"eicu_{int(args.eicu_max_series)}"

    # G1: baseline — no transition memory, persistent ON (uses 1024 store, 512 series)
    no_transition = _base_command(args, output_dir / f"{prefix}_no_transition.json")

    # G2: restored + transition + persist ON — target (R4-level config, 1024-store persistent)
    restored_persist = _base_command(args, output_dir / f"{prefix}_restored_with_persist.json")
    restored_persist.extend(
        [
            "--enable-transition-memory",
            "--transition-signature-match-weight", "0.08",
            "--disable-transition-partial-signature",
        ]
    )

    # G3: restored + transition + persist OFF — ablation to measure persistent delta
    restored_no_persist = _base_command(args, output_dir / f"{prefix}_restored_no_persist.json")
    restored_no_persist.extend(
        [
            "--enable-transition-memory",
            "--transition-signature-match-weight", "0.08",
            "--disable-transition-partial-signature",
            "--disable-persistent-memory-reuse",
        ]
    )

    return [
        ("no_transition", no_transition),
        ("restored_with_persist", restored_persist),
        ("restored_no_persist", restored_no_persist),
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standard 3-group ablation: no_transition / with_persist / no_persist."
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--eicu-max-series", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--enable-kg", action="store_true", default=True)
    parser.add_argument("--disable-kg", action="store_false", dest="enable_kg")
    parser.add_argument("--persistent-memory-store", default=r".\output\persistent_memory\eicu_full_1024_kg258")
    parser.add_argument("--persistent-memory-allowed-splits", default="train,external")
    parser.add_argument("--memory-gain-audit-max-cases", type=int, default=0)
    parser.add_argument("--memory-gain-audit-top-k", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        default=r".\output\formal\s2_stabilized",
    )
    parser.add_argument("--disable-persistent-reuse", action="store_true", help="Run without persistent memory (--disable-persistent-memory-reuse)")
    parser.add_argument("--skip-posthoc-diagnostics", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name, command in _suite_commands(args):
        print(f"[f3-ablation] {name}", flush=True)
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=str(PROJECT_ROOT))
        if completed.returncode != 0:
            print(f"[f3-ablation] {name} failed with exit code {completed.returncode}", file=sys.stderr)
            return completed.returncode
    print("\n[f3] All 3 groups complete.", flush=True)
    print("[f3] Quick compare command:")
    print(f'  python -c "import json, pathlib; '
          f'files = sorted(pathlib.Path(r\'{args.output_dir}\').glob(\'eicu_*.json\')); '
          f'[(print(f.name, json.loads(f.read_text(encoding=\\\"utf-8\\\")).get(\"memory_effectiveness\",{{}}).get(\"improvement_mae\"))) for f in files]"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
