import csv
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class DatasetBundle:
    x_train: List[List[float]]
    y_train: List[int]
    x_val: List[List[float]]
    y_val: List[int]
    x_test: List[List[float]]
    y_test: List[int]
    label_mapping: Dict[str, int]
    feature_columns: List[str]


def _is_float(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def load_csv_dataset(csv_path: str, label_col: str) -> Tuple[List[List[float]], List[str], List[str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        if label_col not in reader.fieldnames:
            raise ValueError(f"label column '{label_col}' not found")

        feature_cols = [c for c in reader.fieldnames if c != label_col]
        if len(feature_cols) < 10:
            raise ValueError("feature count is less than 10; expected 10+")

        rows = list(reader)

    categorical_maps: Dict[str, Dict[str, int]] = {c: {} for c in feature_cols}
    x_data: List[List[float]] = []
    y_labels: List[str] = []

    for row in rows:
        feature_vec: List[float] = []
        for col in feature_cols:
            value = row[col].strip()
            if value == "":
                value = "0"

            if _is_float(value):
                feature_vec.append(float(value))
            else:
                cmap = categorical_maps[col]
                if value not in cmap:
                    cmap[value] = len(cmap)
                feature_vec.append(float(cmap[value]))

        x_data.append(feature_vec)
        y_labels.append(str(row[label_col]).strip())

    return x_data, y_labels, feature_cols


def encode_labels(y_labels: List[str]) -> Tuple[List[int], Dict[str, int]]:
    mapping: Dict[str, int] = {}
    y_encoded: List[int] = []
    for y in y_labels:
        if y not in mapping:
            mapping[y] = len(mapping)
        y_encoded.append(mapping[y])
    return y_encoded, mapping


def split_dataset(
    x_data: List[List[float]],
    y_data: List[int],
    test_size: float,
    val_size: float,
    seed: int,
) -> DatasetBundle:
    random.seed(seed)
    idx = list(range(len(x_data)))
    random.shuffle(idx)

    n = len(idx)
    n_test = int(n * test_size)
    n_val = int(n * val_size)

    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]

    def gather(idxs: List[int]):
        return [x_data[i] for i in idxs], [y_data[i] for i in idxs]

    x_train, y_train = gather(train_idx)
    x_val, y_val = gather(val_idx)
    x_test, y_test = gather(test_idx)

    x_train, x_val, x_test = standardize_by_train(x_train, x_val, x_test)

    return DatasetBundle(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        label_mapping={},
        feature_columns=[],
    )


def standardize_by_train(
    x_train: List[List[float]], x_val: List[List[float]], x_test: List[List[float]]
) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    if not x_train:
        return x_train, x_val, x_test

    dim = len(x_train[0])
    means = [0.0] * dim
    stds = [0.0] * dim

    for row in x_train:
        for j, v in enumerate(row):
            means[j] += v
    means = [m / len(x_train) for m in means]

    for row in x_train:
        for j, v in enumerate(row):
            stds[j] += (v - means[j]) ** 2
    stds = [((s / len(x_train)) ** 0.5) for s in stds]
    stds = [s if s > 1e-12 else 1.0 for s in stds]

    def transform(xset: List[List[float]]) -> List[List[float]]:
        out = []
        for row in xset:
            out.append([(row[j] - means[j]) / stds[j] for j in range(dim)])
        return out

    return transform(x_train), transform(x_val), transform(x_test)
