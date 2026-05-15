import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = PROJECT_ROOT / "run_forecasting_experiment.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed control experiments to verify donor-ranking changes do not alter factual forecasting metrics."
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "output" / "analysis" / "evaluation_line_isolation"))
    parser.add_argument("--eicu-max-series", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-train-windows-per-series", type=int, default=8)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--forecast-horizon", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    return parser.parse_args()


def _variant_metrics(payload: Dict[str, object]) -> Dict[str, float]:
    factual = payload["evaluation_lines"]["factual_forecasting"]
    enabled = factual["metrics"]["memory_enabled"]
    disabled = factual["metrics"]["memory_disabled"]
    return {
        "hybrid_mae": float(enabled["mae"]),
        "hybrid_rmse": float(enabled["rmse"]),
        "hybrid_smape": float(enabled["smape"]),
        "base_mae": float(disabled["mae"]),
        "base_rmse": float(disabled["rmse"]),
        "base_smape": float(disabled["smape"]),
        "improvement_mae": float(factual["improvement"]["improvement_mae"]),
        "improvement_rmse": float(factual["improvement"]["improvement_rmse"]),
        "improvement_smape": float(factual["improvement"]["improvement_smape"]),
    }


def _compare_metrics(reference: Dict[str, float], current: Dict[str, float]) -> Dict[str, float]:
    return {key: abs(float(current[key]) - float(reference[key])) for key in reference}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    common_args: List[str] = [
        sys.executable,
        str(RUN_SCRIPT),
        "--dataset-format",
        "eicu_sepsis3",
        "--enable-kg",
        "--eicu-max-series",
        str(args.eicu_max_series),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--max-train-windows-per-series",
        str(args.max_train_windows_per_series),
        "--history-length",
        str(args.history_length),
        "--forecast-horizon",
        str(args.forecast_horizon),
        "--encoder-type",
        "transformer",
        "--enable-transition-memory",
        "--skip-posthoc-diagnostics",
        "--seed",
        str(args.seed),
    ]

    variants = {
        "donor_only": [
            "--counterfactual-donor-score-mode",
            "structured",
            "--counterfactual-candidate-policy",
            "donor_only",
        ],
        "generated_best": [
            "--counterfactual-donor-score-mode",
            "structured",
            "--counterfactual-candidate-policy",
            "generated_best",
        ],
        "no_hard_filter": [
            "--counterfactual-donor-score-mode",
            "structured",
            "--counterfactual-candidate-policy",
            "donor_only",
            "--disable-counterfactual-hard-filter",
        ],
    }

    payloads: Dict[str, Dict[str, object]] = {}
    metric_rows: Dict[str, Dict[str, float]] = {}
    for name, variant_args in variants.items():
        output_json = output_dir / f"{name}.json"
        command = common_args + variant_args + ["--output-json", str(output_json)]
        subprocess.run(command, cwd=str(PROJECT_ROOT.parent), check=True)
        with output_json.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        payloads[name] = payload
        metric_rows[name] = _variant_metrics(payload)

    reference_name = "donor_only"
    reference_metrics = metric_rows[reference_name]
    metric_diffs = {
        name: _compare_metrics(reference_metrics, row)
        for name, row in metric_rows.items()
        if name != reference_name
    }
    isolation_pass = all(
        max(diff.values(), default=0.0) <= float(args.tolerance) for diff in metric_diffs.values()
    )

    summary = {
        "control_question": "Do donor-ranking and plausibility-only changes leave factual forecasting unchanged?",
        "reference_variant": reference_name,
        "tolerance": float(args.tolerance),
        "passes_isolation_check": bool(isolation_pass),
        "factual_metrics": metric_rows,
        "absolute_differences_vs_reference": metric_diffs,
        "donor_ranking_snapshots": {
            name: payloads[name]["evaluation_lines"]["counterfactual_donor_ranking"]
            for name in payloads
        },
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
