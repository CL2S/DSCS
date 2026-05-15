import math
from typing import Dict, List, Sequence


def _flatten(sequences: Sequence[Sequence[float]]) -> List[float]:
    flattened: List[float] = []
    for sequence in sequences:
        flattened.extend(float(value) for value in sequence)
    return flattened


def mae(y_true: Sequence[Sequence[float]], y_pred: Sequence[Sequence[float]]) -> float:
    truth = _flatten(y_true)
    pred = _flatten(y_pred)
    return sum(abs(left - right) for left, right in zip(truth, pred)) / max(1, len(truth))


def rmse(y_true: Sequence[Sequence[float]], y_pred: Sequence[Sequence[float]]) -> float:
    truth = _flatten(y_true)
    pred = _flatten(y_pred)
    mse = sum((left - right) ** 2 for left, right in zip(truth, pred)) / max(1, len(truth))
    return math.sqrt(mse)


def smape(y_true: Sequence[Sequence[float]], y_pred: Sequence[Sequence[float]]) -> float:
    truth = _flatten(y_true)
    pred = _flatten(y_pred)
    total = 0.0
    count = 0
    for left, right in zip(truth, pred):
        denom = abs(left) + abs(right)
        if denom <= 1e-8:
            continue
        total += 2.0 * abs(left - right) / denom
        count += 1
    return total / max(1, count)


def _seasonal_naive_denominator(insample: Sequence[float], seasonality: int) -> float:
    lag = max(1, seasonality)
    if len(insample) <= lag:
        lag = 1
    if len(insample) <= lag:
        return 1.0
    diffs = [abs(insample[idx] - insample[idx - lag]) for idx in range(lag, len(insample))]
    return sum(diffs) / max(1, len(diffs))


def mase(
    y_true: Sequence[Sequence[float]],
    y_pred: Sequence[Sequence[float]],
    insample_series: Sequence[Sequence[float]],
    seasonality: int,
) -> float:
    series_scores: List[float] = []
    for truth, pred, insample in zip(y_true, y_pred, insample_series):
        denom = _seasonal_naive_denominator(insample, seasonality)
        error = sum(abs(left - right) for left, right in zip(truth, pred)) / max(1, len(truth))
        series_scores.append(error / max(1e-8, denom))
    return sum(series_scores) / max(1, len(series_scores))


def forecasting_metrics(
    y_true: Sequence[Sequence[float]],
    y_pred: Sequence[Sequence[float]],
    insample_series: Sequence[Sequence[float]],
    seasonality: int,
) -> Dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mase": mase(y_true, y_pred, insample_series, seasonality),
    }
