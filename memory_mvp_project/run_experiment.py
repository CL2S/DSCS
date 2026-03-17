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
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--sim-threshold", type=float, default=0.8)
    parser.add_argument("--merge-alpha", type=float, default=0.2)
    parser.add_argument("--decay", type=float, default=0.997)
    parser.add_argument("--forget-threshold", type=float, default=0.1)
    parser.add_argument("--max-memory", type=int, default=5000)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main():
    args = parse_args()

    x_data, y_raw, feature_columns = load_csv_dataset(args.csv, args.label_col)
    y_data, label_mapping = encode_labels(y_raw)

    bundle = split_dataset(
        x_data=x_data,
        y_data=y_data,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
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
    )

    model.fit(bundle.x_train, bundle.y_train)

    val_pred = model.predict(bundle.x_val)
    val_prob = model.predict_proba(bundle.x_val)
    test_pred = model.predict(bundle.x_test)
    test_prob = model.predict_proba(bundle.x_test)

    result = {
        "feature_count": len(bundle.feature_columns),
        "class_count": len(bundle.label_mapping),
        "memory_size": len(model.memory),
        "val_metrics": basic_metrics(bundle.y_val, val_pred, val_prob),
        "test_metrics": basic_metrics(bundle.y_test, test_pred, test_prob),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
