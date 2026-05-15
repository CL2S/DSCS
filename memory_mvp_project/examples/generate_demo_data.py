import argparse
import csv
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--features", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.features < 10:
        raise ValueError("features must be >= 10")

    random.seed(args.seed)
    cols = [f"feat_{i}" for i in range(args.features)] + ["label"]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)

        for _ in range(args.samples):
            feats = [random.gauss(0.0, 1.0) for _ in range(args.features)]
            signal = 0.6 * feats[0] + 0.5 * feats[1] - 0.4 * feats[2] + 0.2 * feats[3]
            label = 1 if signal + random.gauss(0.0, 0.7) > 0 else 0
            writer.writerow([*feats, label])

    print(f"saved demo dataset to {args.output}")


if __name__ == "__main__":
    main()
