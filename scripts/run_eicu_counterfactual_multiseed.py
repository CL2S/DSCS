import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = PROJECT_ROOT / "run_forecasting_experiment.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full eICU counterfactual forecasting evaluation across multiple seeds.")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "output" / "forecasting" / "mem_mod_round15" / "multiseed"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--forecast-horizon", type=int, default=2)
    parser.add_argument("--target-field", default="total_sofa")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_results = []
    for seed in seeds:
        output_json = output_dir / f"eicu_counterfactual_full_seed{seed}.json"
        command = [
            sys.executable,
            str(RUN_SCRIPT),
            "--dataset-format",
            "eicu_sepsis3",
            "--dataset-name",
            f"eicu_counterfactual_full_seed{seed}",
            "--history-length",
            str(args.history_length),
            "--forecast-horizon",
            str(args.forecast_horizon),
            "--eicu-target-field",
            args.target_field,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--max-train-windows-per-series",
            "4",
            "--device",
            args.device,
            "--encoder-type",
            "transformer",
            "--transformer-d-model",
            "96",
            "--transformer-layers",
            "2",
            "--transformer-heads",
            "4",
            "--transformer-ff-dim",
            "192",
            "--transformer-dropout",
            "0.1",
            "--memory-direct-residual-mode",
            "adaptive",
            "--memory-path-coordination-mode",
            "sum",
            "--skip-posthoc-diagnostics",
            "--seed",
            str(seed),
            "--output-json",
            str(output_json),
        ]
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
        payload = _load_json(output_json)
        run_results.append(
            {
                "seed": seed,
                "output_json": str(output_json),
                "hybrid_mae": payload["test_metrics"]["mae"],
                "base_mae": payload["base_only"]["mae"],
                "improvement_mae": payload["memory_effectiveness"]["improvement_mae"],
                "counterfactual_mean_delta": payload["counterfactual_evaluation"]["mean_predicted_delta"],
                "counterfactual_improvement_rate": payload["counterfactual_evaluation"]["predicted_improvement_rate"],
                "donor_found_rate": payload["counterfactual_evaluation"]["donor_found_rate"],
                "donor_similarity_mean": payload["counterfactual_evaluation"]["donor_similarity_mean"],
            }
        )

    summary = {
        "seeds": seeds,
        "run_count": len(run_results),
        "runs": run_results,
        "aggregate": {
            "hybrid_mae_mean": mean(item["hybrid_mae"] for item in run_results),
            "base_mae_mean": mean(item["base_mae"] for item in run_results),
            "improvement_mae_mean": mean(item["improvement_mae"] for item in run_results),
            "counterfactual_mean_delta_mean": mean(item["counterfactual_mean_delta"] for item in run_results),
            "counterfactual_improvement_rate_mean": mean(item["counterfactual_improvement_rate"] for item in run_results),
            "donor_found_rate_mean": mean(item["donor_found_rate"] for item in run_results),
            "donor_similarity_mean": mean(item["donor_similarity_mean"] for item in run_results),
        },
    }
    summary_path = output_dir / "eicu_counterfactual_full_multiseed_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
