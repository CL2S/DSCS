import csv
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


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
    split_mode: str = "random"
    split_summary: Dict[str, str] = None


def _is_float(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _detect_delimiter(csv_path: str) -> str:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",|;\t")
        return dialect.delimiter
    except csv.Error:
        return ","


def load_csv_dataset(csv_path: str, label_col: str) -> Tuple[List[List[float]], List[str], List[str]]:
    delimiter = _detect_delimiter(csv_path)
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
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
    feature_columns: Optional[List[str]] = None,
    split_mode: str = "random",
    time_col: str = "",
    drift_features: Optional[List[str]] = None,
    drift_high_risk: bool = True,
) -> DatasetBundle:
    split_mode = split_mode.lower()
    if split_mode not in {"random", "time", "drift"}:
        raise ValueError(f"unsupported split mode '{split_mode}'")

    if split_mode == "random":
        train_idx, val_idx, test_idx = _stratified_random_split(x_data, y_data, test_size, val_size, seed)
        split_summary = {"mode": "random", "detail": "stratified_random"}
    elif split_mode == "time":
        if not time_col:
            raise ValueError("time split requires --time-col")
        if not feature_columns or time_col not in feature_columns:
            raise ValueError(f"time column '{time_col}' not found in features")
        train_idx, val_idx, test_idx = _ordered_split(
            x_data=x_data,
            feature_index=feature_columns.index(time_col),
            test_size=test_size,
            val_size=val_size,
            descending=False,
        )
        split_summary = {"mode": "time", "detail": f"time_col={time_col}"}
    else:
        resolved_drift = _resolve_drift_feature_indices(feature_columns or [], drift_features)
        if not resolved_drift:
            raise ValueError("drift split requires at least one valid drift feature")
        drift_scores = _compute_drift_scores(x_data, resolved_drift)
        train_idx, val_idx, test_idx = _ordered_split(
            x_data=x_data,
            feature_values=drift_scores,
            test_size=test_size,
            val_size=val_size,
            descending=drift_high_risk,
        )
        detail = ",".join((feature_columns or [])[idx] for idx in resolved_drift)
        split_summary = {
            "mode": "drift",
            "detail": f"drift_features={detail}",
            "direction": "high_to_low" if drift_high_risk else "low_to_high",
        }

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
        split_mode=split_mode,
        split_summary=split_summary,
    )


def _stratified_random_split(
    x_data: List[List[float]],
    y_data: List[int],
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    rng = random.Random(seed)
    buckets: Dict[int, List[int]] = {}
    for idx, label in enumerate(y_data):
        buckets.setdefault(label, []).append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for label_indices in buckets.values():
        rng.shuffle(label_indices)
        n_label = len(label_indices)
        n_test = int(n_label * test_size)
        n_val = int(n_label * val_size)

        test_idx.extend(label_indices[:n_test])
        val_idx.extend(label_indices[n_test : n_test + n_val])
        train_idx.extend(label_indices[n_test + n_val :])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def _ordered_split(
    x_data: List[List[float]],
    test_size: float,
    val_size: float,
    feature_index: Optional[int] = None,
    feature_values: Optional[List[float]] = None,
    descending: bool = False,
) -> Tuple[List[int], List[int], List[int]]:
    if feature_values is None:
        if feature_index is None:
            raise ValueError("ordered split requires feature index or values")
        feature_values = [row[feature_index] for row in x_data]

    ordered = sorted(range(len(x_data)), key=lambda idx: feature_values[idx], reverse=descending)
    n_total = len(ordered)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)

    test_idx = ordered[:n_test]
    val_idx = ordered[n_test : n_test + n_val]
    train_idx = ordered[n_test + n_val :]
    return train_idx, val_idx, test_idx


def _resolve_drift_feature_indices(feature_columns: List[str], drift_features: Optional[List[str]]) -> List[int]:
    if drift_features:
        return [feature_columns.index(name) for name in drift_features if name in feature_columns]

    default_features = [
        "Age",
        "BMI",
        "GenHlth",
        "DiffWalk",
        "HighBP",
        "HighChol",
        "HeartDiseaseorAttack",
        "Stroke",
    ]
    return [feature_columns.index(name) for name in default_features if name in feature_columns]


def _compute_drift_scores(x_data: List[List[float]], feature_indices: List[int]) -> List[float]:
    means = []
    stds = []
    for idx in feature_indices:
        values = [row[idx] for row in x_data]
        mean = sum(values) / max(1, len(values))
        var = sum((value - mean) ** 2 for value in values) / max(1, len(values))
        std = var ** 0.5
        means.append(mean)
        stds.append(std if std > 1e-12 else 1.0)

    scores = []
    for row in x_data:
        score = 0.0
        for pos, idx in enumerate(feature_indices):
            score += (row[idx] - means[pos]) / stds[pos]
        scores.append(score / max(1, len(feature_indices)))
    return scores


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
