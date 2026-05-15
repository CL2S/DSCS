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
        description="Run a small persistent-memory matrix to verify semantic retrieval activation on the factual forecasting path."
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "output" / "analysis" / "persistent_semantic_matrix"))
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
    audit = factual["memory_path_audit"]["factual_path_audit"]
    semantic = audit.get("semantic_retrieval", {})
    persistent = payload.get("persistent_memory", {})
    return {
        "hybrid_mae": float(enabled["mae"]),
        "hybrid_rmse": float(enabled["rmse"]),
        "improvement_mae": float(factual["improvement"]["improvement_mae"]),
        "experience_archive_use_rate": float(audit.get("experience_archive_use_rate", 0.0)),
        "semantic_hit_rate": float(semantic.get("hit_rate", 0.0)),
        "semantic_mean_top_score": float(semantic.get("mean_top_score", 0.0)),
        "semantic_mean_template_blend_weight": float(semantic.get("mean_template_blend_weight", 0.0)),
        "loaded_persistent_samples": float(persistent.get("loaded_persistent_samples", 0.0)),
        "loaded_persistent_prototypes": float(persistent.get("loaded_persistent_prototypes", 0.0)),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stores_dir = output_dir / "stores"
    stores_dir.mkdir(parents=True, exist_ok=True)

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
        "--enable-kg",
        "--disable-kg-static-concat",
        "--enable-explicit-kg-path",
        "--kg-residual-weight",
        "0.06",
    ]

    variants = {
        "explicit_only_no_persistent": [],
        "explicit_only_store_unprimed": [
            "--persistent-memory-store",
            str(stores_dir / "unprimed_store"),
        ],
        "explicit_only_store_primed": [
            "--persistent-memory-store",
            str(stores_dir / "primed_store"),
            "--prime-persistent-memory-before-fit",
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

    summary = {
        "question": "Does prefit persistent-memory priming activate semantic retrieval on the factual path?",
        "variant_metrics": metric_rows,
        "acceptance_checks": {
            "unprimed_semantic_hit_rate_is_zero": bool(
                metric_rows["explicit_only_store_unprimed"]["semantic_hit_rate"] == 0.0
            ),
            "primed_semantic_hit_rate_positive": bool(
                metric_rows["explicit_only_store_primed"]["semantic_hit_rate"] > 0.0
            ),
            "primed_loads_persistent_prototypes": bool(
                metric_rows["explicit_only_store_primed"]["loaded_persistent_prototypes"] > 0.0
            ),
            "primed_keeps_archive_active": bool(
                metric_rows["explicit_only_store_primed"]["experience_archive_use_rate"] > 0.0
            ),
        },
        "deltas_vs_no_persistent": {
            name: {
                key: float(values[key]) - float(metric_rows["explicit_only_no_persistent"][key])
                for key in [
                    "hybrid_mae",
                    "hybrid_rmse",
                    "improvement_mae",
                    "experience_archive_use_rate",
                    "semantic_hit_rate",
                    "semantic_mean_top_score",
                    "semantic_mean_template_blend_weight",
                ]
            }
            for name, values in metric_rows.items()
            if name != "explicit_only_no_persistent"
        },
        "persistent_memory_snapshot": {
            name: payloads[name].get("persistent_memory", {})
            for name in payloads
        },
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
