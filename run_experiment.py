import argparse
import json

from src.data_utils import DatasetBundle, encode_labels, load_csv_dataset, split_dataset
from src.evaluate import basic_metrics
from src.memory_model import DynamicMemoryClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Run dynamic-memory MVP on a tabular CSV dataset.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--label-col", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--split-mode", choices=["random", "time", "drift"], default="random")
    parser.add_argument("--time-col", default="")
    parser.add_argument("--drift-features", default="")
    parser.add_argument("--drift-high-risk", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--sim-threshold", type=float, default=0.92)
    parser.add_argument("--merge-alpha", type=float, default=0.2)
    parser.add_argument("--decay", type=float, default=0.997)
    parser.add_argument("--forget-threshold", type=float, default=0.1)
    parser.add_argument("--max-memory", type=int, default=256)
    parser.add_argument("--base-epochs", type=int, default=4)
    parser.add_argument("--base-learning-rate", type=float, default=0.08)
    parser.add_argument("--base-l2", type=float, default=1e-4)
    parser.add_argument("--positive-weight-scale", type=float, default=1.0)
    parser.add_argument("--prototype-confidence", type=float, default=0.84)
    parser.add_argument("--memory-temperature", type=float, default=0.12)
    parser.add_argument("--correction-confidence", type=float, default=0.78)
    parser.add_argument("--uncertainty-low", type=float, default=0.2)
    parser.add_argument("--uncertainty-high", type=float, default=0.8)
    parser.add_argument("--correction-scale", type=float, default=0.45)
    parser.add_argument(
        "--optimize-for",
        choices=["f1_macro", "balanced_accuracy", "recall_priority", "clinical_warning"],
        default="f1_macro",
    )
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument("--min-precision", type=float, default=0.0)
    parser.add_argument("--min-specificity", type=float, default=0.0)
    parser.add_argument("--threshold-min", type=float, default=0.15)
    parser.add_argument("--threshold-max", type=float, default=0.85)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def _select_base_threshold(model: DynamicMemoryClassifier, y_true, positive_probs):
    best_score = None
    best_stats = None
    grid_min = int(model.threshold_min * 100)
    grid_max = int(model.threshold_max * 100)
    for threshold_idx in range(grid_min, grid_max + 1):
        threshold = threshold_idx / 100.0
        stats = model._binary_stats(y_true, positive_probs, threshold)
        score = model._threshold_score(stats)
        if best_score is None or score > best_score:
            best_score = score
            best_stats = stats
    return best_stats


def main():
    args = parse_args()
    drift_features = [item.strip() for item in args.drift_features.split(",") if item.strip()]

    x_data, y_raw, feature_columns = load_csv_dataset(args.csv, args.label_col)
    y_data, label_mapping = encode_labels(y_raw)

    bundle = split_dataset(
        x_data=x_data,
        y_data=y_data,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        feature_columns=feature_columns,
        split_mode=args.split_mode,
        time_col=args.time_col,
        drift_features=drift_features or None,
        drift_high_risk=args.drift_high_risk,
    )
    bundle.label_mapping = label_mapping
    bundle.feature_columns = feature_columns

    model = DynamicMemoryClassifier(
        top_k=args.top_k,
        sim_threshold=args.sim_threshold,
        merge_alpha=args.merge_alpha,
        decay=args.decay,
        forget_threshold=args.forget_threshold,
        max_memory=args.max_memory,
        base_epochs=args.base_epochs,
        base_learning_rate=args.base_learning_rate,
        base_l2=args.base_l2,
        positive_weight_scale=args.positive_weight_scale,
        prototype_confidence=args.prototype_confidence,
        memory_temperature=args.memory_temperature,
        correction_confidence=args.correction_confidence,
        uncertainty_low=args.uncertainty_low,
        uncertainty_high=args.uncertainty_high,
        correction_scale=args.correction_scale,
        optimize_for=args.optimize_for,
        min_recall=args.min_recall,
        min_precision=args.min_precision,
        min_specificity=args.min_specificity,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        seed=args.seed,
    )

    model.fit(bundle.x_train, bundle.y_train, bundle.x_val, bundle.y_val)

    val_pred = model.predict(bundle.x_val)
    val_prob = model.predict_proba(bundle.x_val)
    test_pred = model.predict(bundle.x_test)
    test_prob = model.predict_proba(bundle.x_test)

    result = {
        "feature_count": len(bundle.feature_columns),
        "class_count": len(bundle.label_mapping),
        "split_mode": bundle.split_mode,
        "split_summary": bundle.split_summary,
        "memory_size": len(model.memory),
        "decision_threshold": model.decision_threshold,
        "blend_weight": model.anchor_weight + model.correction_weight,
        "anchor_weight": model.anchor_weight,
        "correction_weight": model.correction_weight,
        "threshold_selection_summary": model.threshold_selection_summary,
        "memory_diagnostics": model.memory_diagnostics,
        "val_metrics": basic_metrics(bundle.y_val, val_pred, val_prob),
        "test_metrics": basic_metrics(bundle.y_test, test_pred, test_prob),
    }

    if len(bundle.label_mapping) == 2:
        base_val_prob = model.predict_base_proba(bundle.x_val)
        base_test_prob = model.predict_base_proba(bundle.x_test)
        base_val_stats = _select_base_threshold(model, bundle.y_val, [p[1] for p in base_val_prob])
        base_threshold = base_val_stats["threshold"]
        base_val_pred = [1 if p[1] >= base_threshold else 0 for p in base_val_prob]
        base_test_pred = [1 if p[1] >= base_threshold else 0 for p in base_test_prob]
        result["base_only"] = {
            "decision_threshold": base_threshold,
            "val_metrics": basic_metrics(bundle.y_val, base_val_pred, base_val_prob),
            "test_metrics": basic_metrics(bundle.y_test, base_test_pred, base_test_prob),
        }
        result["memory_effectiveness"] = {
            "delta_f1_macro": result["test_metrics"]["f1_macro"] - result["base_only"]["test_metrics"]["f1_macro"],
            "delta_auc": result["test_metrics"]["auc"] - result["base_only"]["test_metrics"]["auc"],
            "delta_recall_pos": result["test_metrics"]["recall_pos"] - result["base_only"]["test_metrics"]["recall_pos"],
            "delta_precision_pos": result["test_metrics"]["precision_pos"] - result["base_only"]["test_metrics"]["precision_pos"],
            "delta_balanced_accuracy": result["test_metrics"]["balanced_accuracy"] - result["base_only"]["test_metrics"]["balanced_accuracy"],
        }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
