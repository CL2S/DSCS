import math
from dataclasses import dataclass
from typing import Dict, List, Sequence


PATTERN_LABELS = ["flat", "up", "down", "volatile", "spike"]
TRAJECTORY_LABELS = ["stable_regime", "rising_regime", "falling_regime", "seasonal_regime", "shifted_regime"]


def formation_feature_names() -> List[str]:
    return [
        "local_slope",
        "medium_slope",
        "local_volatility",
        "volatility_ratio",
        "seasonal_gap_norm",
        "seasonal_strength",
        "level_shift_norm",
        "max_zscore",
        "range_norm",
        "phase_sin",
        "phase_cos",
        "change_proxy",
        "curvature",
        "autocorr_lag1",
        "stability_score",
        "regime_mix_score",
    ]


@dataclass
class WindowFormation:
    features: List[float]
    feature_map: Dict[str, float]
    pattern_label: int
    trajectory_label: int
    experience_label: int


def _safe_scale(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    mean_value = sum(values) / max(1, len(values))
    variance = sum((value - mean_value) ** 2 for value in values) / max(1, len(values))
    std_value = math.sqrt(max(variance, 1e-8))
    fallback = max(1.0, abs(mean_value), max(abs(value) for value in values))
    return max(std_value, fallback * 1e-2)


def _slope(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return (values[-1] - values[0]) / max(1, len(values) - 1)


def _phase_features(end_index: int, seasonality: int) -> tuple[float, float]:
    if seasonality <= 1:
        return 0.0, 1.0
    phase = 2.0 * math.pi * (end_index % seasonality) / seasonality
    return math.sin(phase), math.cos(phase)


def _max_zscore(values: Sequence[float], center: float, scale: float) -> float:
    if not values:
        return 0.0
    return max(abs((value - center) / max(scale, 1e-6)) for value in values)


def _autocorr_lag1(values: Sequence[float]) -> float:
    if len(values) <= 2:
        return 0.0
    left = values[:-1]
    right = values[1:]
    left_mean = sum(left) / max(1, len(left))
    right_mean = sum(right) / max(1, len(right))
    numerator = sum((l - left_mean) * (r - right_mean) for l, r in zip(left, right))
    left_var = sum((l - left_mean) ** 2 for l in left)
    right_var = sum((r - right_mean) ** 2 for r in right)
    denom = math.sqrt(max(left_var * right_var, 1e-8))
    return float(numerator / denom)


def _curvature(values: Sequence[float], scale: float) -> float:
    if len(values) <= 2:
        return 0.0
    second_diffs = [
        values[index] - 2.0 * values[index - 1] + values[index - 2]
        for index in range(2, len(values))
    ]
    return (sum(second_diffs) / max(1, len(second_diffs))) / max(scale, 1e-6)


def _label_pattern(
    local_slope: float,
    medium_slope: float,
    volatility: float,
    volatility_ratio: float,
    level_shift_norm: float,
    max_z: float,
    range_norm: float,
) -> int:
    if max_z >= 2.6 or (range_norm >= 3.25 and volatility_ratio >= 1.1) or abs(level_shift_norm) >= 1.9:
        return PATTERN_LABELS.index("spike")
    if volatility >= 1.05 and volatility_ratio >= 1.08 and abs(local_slope) < 0.1:
        return PATTERN_LABELS.index("volatile")
    if local_slope >= 0.1 and medium_slope >= 0.05:
        return PATTERN_LABELS.index("up")
    if local_slope <= -0.1 and medium_slope <= -0.05:
        return PATTERN_LABELS.index("down")
    return PATTERN_LABELS.index("flat")


def _label_trajectory(
    local_slope: float,
    medium_slope: float,
    seasonal_gap_norm: float,
    seasonal_strength: float,
    level_shift_norm: float,
    change_proxy: float,
    curvature: float,
    stability_score: float,
) -> int:
    if abs(level_shift_norm) >= 1.05 or change_proxy >= 0.62 or abs(curvature) >= 0.08:
        return TRAJECTORY_LABELS.index("shifted_regime")
    if seasonal_strength >= 0.65 and abs(seasonal_gap_norm) >= 0.45 and abs(local_slope) < 0.08:
        return TRAJECTORY_LABELS.index("seasonal_regime")
    if medium_slope >= 0.04 or (local_slope >= 0.07 and stability_score >= 0.2):
        return TRAJECTORY_LABELS.index("rising_regime")
    if medium_slope <= -0.04 or (local_slope <= -0.07 and stability_score >= 0.2):
        return TRAJECTORY_LABELS.index("falling_regime")
    return TRAJECTORY_LABELS.index("stable_regime")


def build_window_formation(
    context: Sequence[float],
    seasonality: int,
    end_index: int,
) -> WindowFormation:
    scale = _safe_scale(context)
    center = sum(context) / max(1, len(context))
    local_slope = _slope(context) / max(scale, 1e-6)
    mid_point = max(2, len(context) // 2)
    medium_slope = _slope(context[-mid_point:]) / max(scale, 1e-6)
    volatility = math.sqrt(sum((value - center) ** 2 for value in context) / max(1, len(context))) / max(scale, 1e-6)
    tail_width = max(2, len(context) // 3)
    tail_center = sum(context[-tail_width:]) / max(1, tail_width)
    tail_volatility = math.sqrt(
        sum((value - tail_center) ** 2 for value in context[-tail_width:]) / max(1, tail_width)
    ) / max(scale, 1e-6)
    volatility_ratio = tail_volatility / max(volatility, 1e-6)

    seasonal_gap = 0.0
    seasonal_strength = 0.0
    if seasonality > 0 and len(context) > seasonality:
        seasonal_gap = (context[-1] - context[-seasonality]) / max(scale, 1e-6)
        recent_block = context[-seasonality:]
        previous_block = context[-2 * seasonality : -seasonality]
        if previous_block:
            seasonal_strength = 1.0 - min(
                2.0,
                sum(abs(left - right) for left, right in zip(recent_block, previous_block)) / max(1, len(previous_block)) / max(scale, 1e-6),
            ) / 2.0
            seasonal_strength = max(0.0, seasonal_strength)

    chunk = max(1, len(context) // 4)
    left_mean = sum(context[:chunk]) / max(1, chunk)
    right_mean = sum(context[-chunk:]) / max(1, chunk)
    level_shift_norm = (right_mean - left_mean) / max(scale, 1e-6)

    maximum = max(context)
    minimum = min(context)
    range_norm = (maximum - minimum) / max(scale, 1e-6)
    max_z = _max_zscore(context, center, scale)
    phase_sin, phase_cos = _phase_features(end_index=end_index, seasonality=seasonality)

    diffs = [abs(context[index] - context[index - 1]) for index in range(1, len(context))]
    if diffs:
        change_proxy = max(diffs) / max(scale, 1e-6)
    else:
        change_proxy = 0.0

    change_proxy = min(change_proxy, 5.0) / 5.0
    curvature = _curvature(context, scale)
    autocorr_lag1 = _autocorr_lag1(context)
    stability_score = 1.0 / (1.0 + volatility + 0.5 * change_proxy + abs(level_shift_norm))
    regime_mix_score = min(
        4.0,
        abs(local_slope) + 0.75 * abs(medium_slope) + volatility + abs(level_shift_norm) + change_proxy,
    ) / 4.0

    features = [
        local_slope,
        medium_slope,
        volatility,
        volatility_ratio,
        seasonal_gap,
        seasonal_strength,
        level_shift_norm,
        max_z,
        range_norm,
        phase_sin,
        phase_cos,
        change_proxy,
        curvature,
        autocorr_lag1,
        stability_score,
        regime_mix_score,
    ]
    names = formation_feature_names()
    feature_map = {name: float(value) for name, value in zip(names, features)}

    pattern_label = _label_pattern(
        local_slope,
        medium_slope,
        volatility,
        volatility_ratio,
        level_shift_norm,
        max_z,
        range_norm,
    )
    trajectory_label = _label_trajectory(
        local_slope,
        medium_slope,
        seasonal_gap,
        seasonal_strength,
        level_shift_norm,
        change_proxy,
        curvature,
        stability_score,
    )
    experience_label = pattern_label * len(TRAJECTORY_LABELS) + trajectory_label

    return WindowFormation(
        features=features,
        feature_map=feature_map,
        pattern_label=pattern_label,
        trajectory_label=trajectory_label,
        experience_label=experience_label,
    )
