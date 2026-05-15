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
        description="Run a small KG revision matrix to compare plain KG concatenation against explicit KG residual paths."
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "output" / "analysis" / "kg_revision_matrix"))
    parser.add_argument("--eicu-max-series", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-train-windows-per-series", type=int, default=4)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--forecast-horizon", type=int, default=2)
    parser.add_argument("--encoder-type", default="transformer")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _variant_metrics(payload: Dict[str, object]) -> Dict[str, float]:
    factual = payload["evaluation_lines"]["factual_forecasting"]
    enabled = factual["metrics"]["memory_enabled"]
    disabled = factual["metrics"]["memory_disabled"]
    audit = factual["memory_path_audit"]["factual_path_audit"]
    direct = audit.get("direct_memory_means", {})
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
        "kg_gate": float(direct.get("kg_gate", 0.0)),
        "kg_residual_strength": float(direct.get("kg_residual_strength", 0.0)),
        "kg_feature_density": float(direct.get("kg_feature_density", 0.0)),
        "kg_guideline_alignment": float(direct.get("kg_guideline_alignment", 0.0)),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    common_args: List[str] = [
        sys.executable,
        str(RUN_SCRIPT),
        "--dataset-format",
        "eicu_sepsis3",
        "--eicu-target-field",
        "total_sofa",
        "--eicu-max-series",
        str(args.eicu_max_series),
        "--history-length",
        str(args.history_length),
        "--forecast-horizon",
        str(args.forecast_horizon),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--max-train-windows-per-series",
        str(args.max_train_windows_per_series),
        "--encoder-type",
        str(args.encoder_type),
        "--seed",
        str(args.seed),
    ]
    variants = {
        "no_kg": [],
        "kg_concat_only": ["--enable-kg"],
        "kg_explicit_w006": ["--enable-kg", "--enable-explicit-kg-path", "--kg-residual-weight", "0.06"],
        "kg_explicit_w012": ["--enable-kg", "--enable-explicit-kg-path", "--kg-residual-weight", "0.12"],
        "kg_explicit_only_w006": [
            "--enable-kg",
            "--disable-kg-static-concat",
            "--enable-explicit-kg-path",
            "--kg-residual-weight",
            "0.06",
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

    baseline = metric_rows["kg_concat_only"]
    summary = {
        "question": "Does explicit KG residual fusion improve over plain KG concatenation on the current smoke setup?",
        "reference_variant": "kg_concat_only",
        "variant_metrics": metric_rows,
        "best_variants": {
            "hybrid_mae": min(metric_rows.items(), key=lambda item: item[1]["hybrid_mae"])[0],
            "hybrid_rmse": min(metric_rows.items(), key=lambda item: item[1]["hybrid_rmse"])[0],
            "improvement_mae": max(metric_rows.items(), key=lambda item: item[1]["improvement_mae"])[0],
            "improvement_rmse": max(metric_rows.items(), key=lambda item: item[1]["improvement_rmse"])[0],
        },
        "acceptance_checks": {
            "explicit_kg_used": bool(metric_rows["kg_explicit_only_w006"]["kg_gate"] > 0.0),
            "explicit_kg_only_beats_kg_concat_on_mae": bool(
                metric_rows["kg_explicit_only_w006"]["hybrid_mae"] < metric_rows["kg_concat_only"]["hybrid_mae"]
            ),
            "explicit_kg_only_beats_no_kg_on_mae": bool(
                metric_rows["kg_explicit_only_w006"]["hybrid_mae"] < metric_rows["no_kg"]["hybrid_mae"]
            ),
        },
        "deltas_vs_kg_concat_only": {
            name: {
                key: float(row[key]) - float(baseline[key])
                for key in [
                    "hybrid_mae",
                    "hybrid_rmse",
                    "hybrid_smape",
                    "base_mae",
                    "base_rmse",
                    "base_smape",
                    "improvement_mae",
                    "improvement_rmse",
                    "improvement_smape",
                    "kg_gate",
                    "kg_residual_strength",
                ]
            }
            for name, row in metric_rows.items()
            if name != "kg_concat_only"
        },
        "audit_snapshot": {
            name: payloads[name]["evaluation_lines"]["factual_forecasting"]["memory_path_audit"]["factual_path_audit"]
            for name in payloads
        },
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
