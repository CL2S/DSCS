import math
import os
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from distutils.util import strtobool
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.kg_integration import KnowledgeGraphFeatureBuilder
from src.ts_formation import TRAJECTORY_LABELS, build_window_formation, formation_feature_names


# Adapted from TSForecasting/utils/data_loader.py so the project can consume
# benchmark .tsf files directly without depending on the external repo at runtime.
def convert_tsf_to_dataframe(
    full_file_path_and_name: str,
    replace_missing_vals_with: str = "NaN",
    value_column_name: str = "series_value",
):
    col_names: List[str] = []
    col_types: List[str] = []
    all_data: Dict[str, List[object]] = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("@"):
                if not line.startswith("@data"):
                    line_content = line.split(" ")
                    if line.startswith("@attribute"):
                        if len(line_content) != 3:
                            raise ValueError("Invalid .tsf attribute declaration.")
                        col_names.append(line_content[1])
                        col_types.append(line_content[2])
                    else:
                        if len(line_content) != 2:
                            raise ValueError("Invalid .tsf metadata declaration.")
                        if line.startswith("@frequency"):
                            frequency = line_content[1]
                        elif line.startswith("@horizon"):
                            forecast_horizon = int(line_content[1])
                        elif line.startswith("@missing"):
                            contain_missing_values = bool(strtobool(line_content[1]))
                        elif line.startswith("@equallength"):
                            contain_equal_length = bool(strtobool(line_content[1]))
                else:
                    if not col_names:
                        raise ValueError("Missing attribute section before @data.")
                    found_data_tag = True
                line_count += 1
                continue

            if line.startswith("#"):
                line_count += 1
                continue

            if not col_names:
                raise ValueError("Missing attribute section.")
            if not found_data_tag:
                raise ValueError("Missing @data tag.")

            if not started_reading_data_section:
                started_reading_data_section = True
                found_data_section = True
                all_series = []
                for col in col_names:
                    all_data[col] = []

            full_info = line.split(":")
            if len(full_info) != len(col_names) + 1:
                raise ValueError("Missing attributes/values in .tsf series row.")

            raw_series = full_info[-1].split(",")
            numeric_series: List[float | str] = []
            for value in raw_series:
                if value == "?":
                    numeric_series.append(replace_missing_vals_with)
                else:
                    numeric_series.append(float(value))

            if numeric_series.count(replace_missing_vals_with) == len(numeric_series):
                raise ValueError("All values are missing in a .tsf series.")

            all_series.append(pd.Series(numeric_series).array)

            for idx, col_name in enumerate(col_names):
                if col_types[idx] == "numeric":
                    att_val = int(full_info[idx])
                elif col_types[idx] == "string":
                    att_val = str(full_info[idx])
                elif col_types[idx] == "date":
                    att_val = datetime.strptime(full_info[idx], "%Y-%m-%d %H-%M-%S")
                else:
                    raise ValueError("Unsupported .tsf attribute type.")
                all_data[col_name].append(att_val)

            line_count += 1

    if line_count == 0:
        raise ValueError("Empty .tsf file.")
    if not col_names:
        raise ValueError("Missing .tsf attribute section.")
    if not found_data_section:
        raise ValueError("Missing .tsf data section.")

    all_data[value_column_name] = all_series
    loaded_data = pd.DataFrame(all_data)
    return loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length


DATASET_REGISTRY: Dict[str, Dict[str, int]] = {
    "tourism_monthly_dataset.tsf": {"history_length": 15, "forecast_horizon": 24, "seasonality": 12},
    "nn5_daily_dataset_without_missing_values.tsf": {"history_length": 9, "forecast_horizon": 56, "seasonality": 7},
    "australian_electricity_demand_dataset.tsf": {"history_length": 420, "forecast_horizon": 336, "seasonality": 48},
}

TSF_STATIC_FEATURE_NAMES = [
    "context_mean",
    "context_scale",
    "context_last",
    "context_min",
    "context_max",
    "context_trend",
    "seasonal_gap",
    "history_length",
    "forecast_horizon",
]
TSF_SEQUENCE_FEATURE_NAMES = ["normalized_value"]
EICU_STATIC_FEATURE_NAMES = [
    "age_years",
    "gender_male",
    "gender_female",
    "admissionweight_kg",
    "acutephysiologyscore",
    "apachescore",
    "adult_eligible",
    "pre_baseline_total_sofa",
    "suspected_infection_offset_minutes",
]
EICU_TEXT_HASH_BUCKETS = 4
EICU_LABEL_EXCLUDED_COLUMNS = {
    "patientunitstayid",
    "patienthealthsystemstayid",
    "uniquepid",
}
EICU_TRAJECTORY_EXCLUDED_COLUMNS = {
    "patientunitstayid",
    "bin_index",
    "rel_start_hours",
    "rel_end_hours",
}
EICU_INTERVENTION_KEYWORDS = (
    "antibiotic",
    "vasopressor",
    "pressor",
    "resp_support",
    "fio2",
    "culture",
    "drug",
    "med",
    "treat",
)
EICU_AUX_REGRESSION_TARGET_NAMES = [
    "future_sofa_delta_mean",
    "future_lactate_delta",
]
EICU_AUX_BINARY_TARGET_NAMES = [
    "future_vasopressor_need",
    "future_resp_support_escalation",
]
EICU_AUX_TARGET_NAMES = EICU_AUX_REGRESSION_TARGET_NAMES + EICU_AUX_BINARY_TARGET_NAMES
EICU_CONTEXTUAL_METADATA_KEYS = [
    "hospitalid",
    "wardid",
    "unittype",
    "infection_anchor_type",
    "infection_anchor_value",
    "suspected_infection_from",
]


def frequency_to_seasonality(frequency: Optional[str]) -> int:
    mapping = {
        "yearly": 1,
        "quarterly": 4,
        "monthly": 12,
        "weekly": 52,
        "daily": 7,
        "hourly": 24,
        "half_hourly": 48,
        "10_minutes": 144,
    }
    if frequency is None:
        return 1
    return mapping.get(str(frequency).lower(), 1)


def _clean_series(values: Sequence[object]) -> List[float]:
    cleaned: List[float] = []
    for value in values:
        if isinstance(value, str):
            if value.lower() == "nan":
                continue
            cleaned.append(float(value))
        else:
            numeric = float(value)
            if math.isnan(numeric):
                continue
            cleaned.append(numeric)
    return cleaned


def _safe_scale(values: Sequence[float]) -> Tuple[float, float]:
    mean_value = sum(values) / max(1, len(values))
    variance = sum((value - mean_value) ** 2 for value in values) / max(1, len(values))
    std_value = math.sqrt(max(variance, 1e-8))
    fallback = max(1.0, abs(mean_value), max(abs(value) for value in values))
    scale = max(std_value, fallback * 1e-2)
    return mean_value, scale


def _trend(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return (values[-1] - values[0]) / max(1, len(values) - 1)


def _window_static_features(context: Sequence[float], seasonality: int, horizon: int) -> List[float]:
    mean_value, scale = _safe_scale(context)
    minimum = min(context)
    maximum = max(context)
    seasonal_gap = context[-1] - context[-seasonality] if seasonality > 0 and len(context) > seasonality else 0.0
    return [
        mean_value,
        scale,
        context[-1],
        minimum,
        maximum,
        _trend(context),
        seasonal_gap,
        float(len(context)),
        float(horizon),
    ]


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric):
        return default
    return numeric


def _safe_metadata_value(value: object):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(numeric):
        return None
    if abs(numeric - round(numeric)) < 1e-9:
        return int(round(numeric))
    return float(numeric)


def _eicu_label_static_features(label_row: Dict[str, object]) -> List[float]:
    gender = str(label_row.get("gender", "")).strip().lower()
    return [
        _safe_float(label_row.get("age_years")),
        1.0 if gender == "male" else 0.0,
        1.0 if gender == "female" else 0.0,
        _safe_float(label_row.get("admissionweight_kg")),
        _safe_float(label_row.get("acutephysiologyscore")),
        _safe_float(label_row.get("apachescore")),
        _safe_float(label_row.get("adult_eligible")),
        _safe_float(label_row.get("pre_baseline_total_sofa")),
        _safe_float(label_row.get("suspected_infection_offset_minutes")),
    ]


def _normalize_window(context: Sequence[float], future: Sequence[float]) -> Tuple[List[List[float]], List[float], float, float]:
    center, scale = _safe_scale(context)
    normalized_sequence = [[(value - center) / scale] for value in context]
    normalized_future = [(value - center) / scale for value in future]
    return normalized_sequence, normalized_future, center, scale


@dataclass
class ForecastSample:
    sequence: List[List[float]]
    static: List[float]
    target: List[float]
    metadata: Dict[str, object]
    scale_center: float
    scale_value: float
    raw_context: List[float]
    raw_target: List[float]
    formation_features: List[float]
    pattern_label: int
    trajectory_label: int
    experience_label: int
    patient_static: Optional[List[float]] = None
    intervention_static: Optional[List[float]] = None
    intervention_sequence: Optional[List[List[float]]] = None
    kg_features: Optional[List[float]] = None
    aux_targets: Optional[Dict[str, float]] = None


@dataclass
class ForecastingDataset:
    dataset_name: str
    source_path: str
    frequency: str
    seasonality: int
    forecast_horizon: int
    history_length: int
    train_samples: List[ForecastSample]
    val_samples: List[ForecastSample]
    test_samples: List[ForecastSample]
    series_count: int
    min_series_length: int
    mean_series_length: float
    max_series_length: int
    static_feature_names: List[str]
    sequence_feature_names: List[str]
    formation_feature_names: List[str]
    patient_feature_names: Optional[List[str]] = None
    intervention_feature_names: Optional[List[str]] = None
    intervention_sequence_feature_names: Optional[List[str]] = None
    kg_feature_names: Optional[List[str]] = None
    aux_target_names: Optional[List[str]] = None

    def summary(self) -> Dict[str, float | int | str]:
        return {
            "dataset_name": self.dataset_name,
            "source_path": self.source_path,
            "frequency": self.frequency,
            "seasonality": self.seasonality,
            "forecast_horizon": self.forecast_horizon,
            "history_length": self.history_length,
            "series_count": self.series_count,
            "min_series_length": self.min_series_length,
            "mean_series_length": self.mean_series_length,
            "max_series_length": self.max_series_length,
            "train_windows": len(self.train_samples),
            "val_windows": len(self.val_samples),
            "test_windows": len(self.test_samples),
        }


def build_tsf_forecasting_dataset(
    tsf_path: str,
    dataset_name: str = "",
    history_length: Optional[int] = None,
    forecast_horizon: Optional[int] = None,
    max_train_windows_per_series: int = 24,
) -> ForecastingDataset:
    loaded_data, frequency, file_horizon, _, _ = convert_tsf_to_dataframe(tsf_path)
    file_name = os.path.basename(tsf_path)
    dataset_name = dataset_name or file_name.replace(".tsf", "")
    registry = DATASET_REGISTRY.get(file_name, {})

    horizon = forecast_horizon or file_horizon or registry.get("forecast_horizon")
    if horizon is None:
        raise ValueError(f"Missing forecast horizon for {file_name}.")
    seasonality = registry.get("seasonality", frequency_to_seasonality(frequency))
    history = history_length or registry.get("history_length") or max(2 * seasonality, horizon)

    train_samples: List[ForecastSample] = []
    val_samples: List[ForecastSample] = []
    test_samples: List[ForecastSample] = []
    lengths: List[int] = []

    static_feature_names = list(TSF_STATIC_FEATURE_NAMES)
    sequence_feature_names = list(TSF_SEQUENCE_FEATURE_NAMES)
    formation_names = formation_feature_names()

    for series_idx, row in loaded_data.iterrows():
        values = _clean_series(row["series_value"])
        lengths.append(len(values))
        if len(values) < history + 3 * horizon:
            continue

        series_name = str(row.get("series_name", f"series_{series_idx}"))
        test_target_start = len(values) - horizon
        val_target_start = test_target_start - horizon
        train_last_end = val_target_start - horizon
        if train_last_end < history:
            continue

        val_context = values[val_target_start - history : val_target_start]
        val_target = values[val_target_start:test_target_start]
        val_formation = build_window_formation(val_context, seasonality=seasonality, end_index=val_target_start)
        val_sequence, val_norm_target, val_center, val_scale = _normalize_window(val_context, val_target)
        val_samples.append(
            ForecastSample(
                sequence=val_sequence,
                static=_window_static_features(val_context, seasonality, horizon),
                target=val_norm_target,
                metadata={
                    "stay_id": float(series_idx),
                    "series_index": float(series_idx),
                    "series_name": series_name,
                    "window_end_index": float(val_target_start),
                    "dataset_name": dataset_name,
                    "series_count": float(len(loaded_data)),
                    "seasonality": float(seasonality),
                    "history_length": float(history),
                    "forecast_horizon": float(horizon),
                },
                scale_center=val_center,
                scale_value=val_scale,
                raw_context=val_context,
                raw_target=val_target,
                formation_features=val_formation.features,
                pattern_label=val_formation.pattern_label,
                trajectory_label=val_formation.trajectory_label,
                experience_label=val_formation.experience_label,
            )
        )

        test_context = values[test_target_start - history : test_target_start]
        test_target = values[test_target_start:]
        test_formation = build_window_formation(test_context, seasonality=seasonality, end_index=test_target_start)
        test_sequence, test_norm_target, test_center, test_scale = _normalize_window(test_context, test_target)
        test_samples.append(
            ForecastSample(
                sequence=test_sequence,
                static=_window_static_features(test_context, seasonality, horizon),
                target=test_norm_target,
                metadata={
                    "stay_id": float(series_idx),
                    "series_index": float(series_idx),
                    "series_name": series_name,
                    "window_end_index": float(test_target_start),
                    "dataset_name": dataset_name,
                    "series_count": float(len(loaded_data)),
                    "seasonality": float(seasonality),
                    "history_length": float(history),
                    "forecast_horizon": float(horizon),
                },
                scale_center=test_center,
                scale_value=test_scale,
                raw_context=test_context,
                raw_target=test_target,
                formation_features=test_formation.features,
                pattern_label=test_formation.pattern_label,
                trajectory_label=test_formation.trajectory_label,
                experience_label=test_formation.experience_label,
            )
        )

        candidate_end_indices = list(range(history, train_last_end + 1))
        if not candidate_end_indices:
            continue
        if max_train_windows_per_series > 0 and len(candidate_end_indices) > max_train_windows_per_series:
            step = (len(candidate_end_indices) - 1) / max(1, max_train_windows_per_series - 1)
            candidate_end_indices = sorted(
                {
                    candidate_end_indices[min(len(candidate_end_indices) - 1, int(round(step * idx)))]
                    for idx in range(max_train_windows_per_series)
                }
            )

        for end_index in candidate_end_indices:
            context = values[end_index - history : end_index]
            future = values[end_index : end_index + horizon]
            formation = build_window_formation(context, seasonality=seasonality, end_index=end_index)
            sequence, norm_target, center, scale = _normalize_window(context, future)
            train_samples.append(
                ForecastSample(
                    sequence=sequence,
                    static=_window_static_features(context, seasonality, horizon),
                    target=norm_target,
                    metadata={
                        "stay_id": float(series_idx),
                        "series_index": float(series_idx),
                        "series_name": series_name,
                        "window_end_index": float(end_index),
                        "dataset_name": dataset_name,
                        "series_count": float(len(loaded_data)),
                        "seasonality": float(seasonality),
                        "history_length": float(history),
                        "forecast_horizon": float(horizon),
                    },
                    scale_center=center,
                    scale_value=scale,
                    raw_context=context,
                    raw_target=future,
                    formation_features=formation.features,
                    pattern_label=formation.pattern_label,
                    trajectory_label=formation.trajectory_label,
                    experience_label=formation.experience_label,
                )
            )

    if not lengths:
        raise ValueError(f"No valid series found in {tsf_path}.")

    return ForecastingDataset(
        dataset_name=dataset_name,
        source_path=tsf_path,
        frequency=str(frequency or "unknown"),
        seasonality=seasonality,
        forecast_horizon=int(horizon),
        history_length=int(history),
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        series_count=len(lengths),
        min_series_length=min(lengths),
        mean_series_length=sum(lengths) / max(1, len(lengths)),
        max_series_length=max(lengths),
        static_feature_names=static_feature_names,
        sequence_feature_names=sequence_feature_names,
        formation_feature_names=formation_names,
    )


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _is_intervention_feature(name: str) -> bool:
    normalized = _normalize_name(name)
    return any(keyword in normalized for keyword in EICU_INTERVENTION_KEYWORDS)


def _coerce_numeric_columns(frame: pd.DataFrame, exclude: set[str]) -> Tuple[List[str], List[str]]:
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for column in frame.columns:
        if column in exclude:
            continue
        series = frame[column]
        coerced = pd.to_numeric(series, errors="coerce")
        if coerced.notna().sum() > 0 and coerced.notna().sum() >= max(2, int(0.3 * max(1, len(series)))):
            numeric_cols.append(column)
        else:
            categorical_cols.append(column)
    return numeric_cols, categorical_cols


def _hash_text_features(value: object, feature_prefix: str, bucket_count: int = EICU_TEXT_HASH_BUCKETS) -> List[float]:
    hashed = [0.0] * bucket_count
    if value is None or pd.isna(value):
        return hashed
    text = str(value).strip().lower()
    if not text:
        return hashed
    parts = [part.strip() for part in re.split(r"[|,;/]+", text) if part.strip()]
    tokens: List[str] = []
    for part in parts or [text]:
        tokens.extend(token for token in re.split(r"\s+", part) if token)
    if not tokens:
        tokens = [text]
    for token in tokens:
        digest = hashlib.sha1(f"{feature_prefix}::{token}".encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % bucket_count
        hashed[bucket] += 1.0
    return hashed


def _categorical_feature_names(columns: Sequence[str], bucket_count: int = EICU_TEXT_HASH_BUCKETS) -> List[str]:
    names: List[str] = []
    for column in columns:
        prefix = _normalize_name(column)
        for idx in range(bucket_count):
            names.append(f"{prefix}_hash_{idx}")
    return names


def _encode_numeric_row(row: Dict[str, object], columns: Sequence[str]) -> List[float]:
    return [_safe_float(row.get(column)) for column in columns]


def _encode_categorical_row(row: Dict[str, object], columns: Sequence[str], bucket_count: int = EICU_TEXT_HASH_BUCKETS) -> List[float]:
    features: List[float] = []
    for column in columns:
        features.extend(_hash_text_features(row.get(column), _normalize_name(column), bucket_count=bucket_count))
    return features


def _aggregate_context_features(frame: pd.DataFrame, columns: Sequence[str], prefix: str) -> Tuple[List[float], List[str]]:
    features: List[float] = []
    names: List[str] = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna().tolist() if column in frame.columns else []
        normalized = _normalize_name(f"{prefix}_{column}")
        if not values:
            features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([float(values[-1]), float(sum(values) / len(values)), float(max(values))])
        names.extend([f"{normalized}_last", f"{normalized}_mean", f"{normalized}_max"])
    return features, names


def _tail_values(values: Sequence[float], minimum_width: int = 2) -> List[float]:
    if not values:
        return []
    width = min(len(values), max(minimum_width, len(values) // 2))
    return list(values[-width:])


def _safe_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean_value = sum(values) / max(1, len(values))
    variance = sum((value - mean_value) ** 2 for value in values) / max(1, len(values))
    return math.sqrt(max(variance, 0.0))


def _last_observed_gap_steps(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    valid_positions = [index for index, value in enumerate(series.tolist()) if not pd.isna(value)]
    if not valid_positions:
        return float(len(series))
    return float(len(series) - 1 - valid_positions[-1])


def _numeric_series_from_frame(frame: pd.DataFrame, column: str) -> List[float]:
    if column not in frame.columns:
        return []
    return [float(value) for value in pd.to_numeric(frame[column], errors="coerce").dropna().tolist()]


def _last_valid_frame_value(frame: pd.DataFrame, column: str, default: float = 0.0) -> float:
    values = _numeric_series_from_frame(frame, column)
    return values[-1] if values else float(default)


def _aggregate_frame_value(
    frame: pd.DataFrame,
    column: str,
    reducer: str = "mean",
    default: float = 0.0,
) -> float:
    values = _numeric_series_from_frame(frame, column)
    if not values:
        return float(default)
    if reducer == "max":
        return float(max(values))
    if reducer == "min":
        return float(min(values))
    if reducer == "last":
        return float(values[-1])
    return float(sum(values) / max(1, len(values)))


def _build_eicu_aux_targets(
    context: Sequence[float],
    context_frame: pd.DataFrame,
    future: Sequence[float],
    future_frame: pd.DataFrame,
) -> Dict[str, float]:
    current_sofa = float(context[-1]) if context else _last_valid_frame_value(context_frame, "total_sofa", default=0.0)
    future_sofa_mean = float(sum(float(value) for value in future) / max(1, len(future))) if future else current_sofa
    current_lactate = _last_valid_frame_value(context_frame, "lactate_max", default=0.0)
    future_lactate_mean = _aggregate_frame_value(future_frame, "lactate_max", reducer="mean", default=current_lactate)
    current_resp_support = _aggregate_frame_value(context_frame, "resp_support", reducer="max", default=0.0)
    future_resp_support = _aggregate_frame_value(future_frame, "resp_support", reducer="max", default=current_resp_support)
    future_vasopressor_any = _aggregate_frame_value(future_frame, "vasopressor_any", reducer="max", default=0.0)
    return {
        "future_sofa_delta_mean": float(future_sofa_mean - current_sofa),
        "future_lactate_delta": float(future_lactate_mean - current_lactate),
        "future_vasopressor_need": 1.0 if future_vasopressor_any > 0.0 else 0.0,
        "future_resp_support_escalation": 1.0 if future_resp_support > current_resp_support else 0.0,
    }


def _dynamic_context_feature_names(columns: Sequence[str], prefix: str) -> List[str]:
    names: List[str] = []
    for column in columns:
        normalized = _normalize_name(f"{prefix}_{column}")
        names.extend(
            [
                f"{normalized}_trend",
                f"{normalized}_recent_delta",
                f"{normalized}_tail_mean",
                f"{normalized}_tail_volatility",
                f"{normalized}_missing_ratio",
                f"{normalized}_last_observed_gap_steps",
                f"{normalized}_recent_trend",
                f"{normalized}_recent_acceleration",
                f"{normalized}_observed_ratio",
                f"{normalized}_recent_observed_ratio",
                f"{normalized}_longest_missing_streak",
                f"{normalized}_trailing_missing_streak",
            ]
        )
    return names


def _recent_trend(values: Sequence[float], tail_size: int = 3) -> float:
    if not values:
        return 0.0
    tail = list(values[-max(1, tail_size) :])
    return float(_trend(tail))


def _recent_acceleration(values: Sequence[float]) -> float:
    if len(values) < 3:
        return 0.0
    last_delta = float(values[-1] - values[-2])
    previous_delta = float(values[-2] - values[-3])
    return float(last_delta - previous_delta)


def _missing_streak_stats(series: pd.Series) -> Tuple[float, float]:
    if series.empty:
        return 0.0, 0.0
    missing_flags = [1 if pd.isna(value) else 0 for value in series.tolist()]
    longest = 0
    current = 0
    for flag in missing_flags:
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    trailing = 0
    for flag in reversed(missing_flags):
        if not flag:
            break
        trailing += 1
    return float(longest), float(trailing)


def _dynamic_context_features(frame: pd.DataFrame, columns: Sequence[str], prefix: str) -> Tuple[List[float], List[str]]:
    features: List[float] = []
    names = _dynamic_context_feature_names(columns, prefix=prefix)
    row_count = max(1, len(frame))
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else pd.Series(dtype=float)
        observed = [float(value) for value in series.dropna().tolist()]
        trend = _trend(observed)
        recent_delta = observed[-1] - observed[-2] if len(observed) > 1 else 0.0
        tail = _tail_values(observed)
        tail_mean = float(sum(tail) / max(1, len(tail))) if tail else 0.0
        tail_volatility = _safe_std(tail)
        missing_ratio = 1.0 - (len(observed) / row_count)
        last_gap = _last_observed_gap_steps(series)
        recent_trend = _recent_trend(observed)
        recent_acceleration = _recent_acceleration(observed)
        observed_ratio = len(observed) / row_count
        tail_window = series.tail(min(3, row_count))
        recent_observed_ratio = float(tail_window.notna().mean()) if len(tail_window) > 0 else 0.0
        longest_missing_streak, trailing_missing_streak = _missing_streak_stats(series)
        features.extend(
            [
                float(trend),
                float(recent_delta),
                float(tail_mean),
                float(tail_volatility),
                float(missing_ratio),
                float(last_gap),
                float(recent_trend),
                float(recent_acceleration),
                float(observed_ratio),
                float(recent_observed_ratio),
                float(longest_missing_streak),
                float(trailing_missing_streak),
            ]
        )
    return features, names


def _time_gap_features(frame: pd.DataFrame, prefix: str = "context_time_gap") -> Tuple[List[float], List[str]]:
    names = [
        f"{prefix}_last_hours",
        f"{prefix}_mean_hours",
        f"{prefix}_std_hours",
        f"{prefix}_irregularity",
    ]
    if "rel_end_hours" not in frame.columns:
        return [0.0, 0.0, 0.0, 0.0], names
    rel_end_hours = pd.to_numeric(frame["rel_end_hours"], errors="coerce").dropna().tolist()
    if len(rel_end_hours) <= 1:
        return [0.0, 0.0, 0.0, 0.0], names
    gaps = [
        float(rel_end_hours[index] - rel_end_hours[index - 1])
        for index in range(1, len(rel_end_hours))
    ]
    gap_mean = float(sum(gaps) / max(1, len(gaps)))
    gap_std = float(_safe_std(gaps))
    irregularity = gap_std / max(abs(gap_mean), 1e-6)
    return [float(gaps[-1]), gap_mean, gap_std, float(irregularity)], names


def _multivariate_dynamic_summary(
    frame: pd.DataFrame,
    columns: Sequence[str],
    prefix: str = "context_multivariate",
) -> Tuple[List[float], List[str]]:
    names = [
        f"{prefix}_change_mean",
        f"{prefix}_change_max",
        f"{prefix}_trend_mean_abs",
        f"{prefix}_trend_max_abs",
        f"{prefix}_observed_ratio",
        f"{prefix}_missing_ratio",
    ]
    if not columns:
        return [0.0] * len(names), names
    recent_changes: List[float] = []
    abs_trends: List[float] = []
    total_cells = max(1, len(frame) * len(columns))
    observed_cells = 0
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else pd.Series(dtype=float)
        observed = [float(value) for value in series.dropna().tolist()]
        observed_cells += len(observed)
        if len(observed) > 1:
            recent_changes.append(abs(observed[-1] - observed[-2]))
        abs_trends.append(abs(_trend(observed)))
    observed_ratio = observed_cells / total_cells
    missing_ratio = 1.0 - observed_ratio
    return [
        float(sum(recent_changes) / max(1, len(recent_changes))) if recent_changes else 0.0,
        float(max(recent_changes)) if recent_changes else 0.0,
        float(sum(abs_trends) / max(1, len(abs_trends))) if abs_trends else 0.0,
        float(max(abs_trends)) if abs_trends else 0.0,
        float(observed_ratio),
        float(missing_ratio),
    ], names


def _patient_context_dynamic_features(
    frame: pd.DataFrame,
    physiology_sequence_columns: Sequence[str],
) -> Tuple[List[float], List[str]]:
    dynamic_features, dynamic_names = _dynamic_context_features(
        frame,
        physiology_sequence_columns,
        prefix="physiology_dynamic",
    )
    gap_features, gap_names = _time_gap_features(frame)
    summary_features, summary_names = _multivariate_dynamic_summary(
        frame,
        physiology_sequence_columns,
    )
    return dynamic_features + gap_features + summary_features, dynamic_names + gap_names + summary_names


def _dynamic_behavior_profile(
    frame: pd.DataFrame,
    physiology_sequence_columns: Sequence[str],
    target_field: str,
) -> Dict[str, float]:
    row_count = max(1, len(frame))
    observed_ratios: List[float] = []
    recent_observed_ratios: List[float] = []
    last_gaps: List[float] = []
    longest_missing_streaks: List[float] = []
    trailing_missing_streaks: List[float] = []
    recent_change_abs: List[float] = []
    recent_trend_abs: List[float] = []

    for column in physiology_sequence_columns:
        series = pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else pd.Series(dtype=float)
        observed = [float(value) for value in series.dropna().tolist()]
        observed_ratios.append(float(len(observed) / row_count))
        tail_window = series.tail(min(3, row_count))
        recent_observed_ratios.append(float(tail_window.notna().mean()) if len(tail_window) > 0 else 0.0)
        last_gaps.append(float(_last_observed_gap_steps(series)))
        longest_missing_streak, trailing_missing_streak = _missing_streak_stats(series)
        longest_missing_streaks.append(float(longest_missing_streak))
        trailing_missing_streaks.append(float(trailing_missing_streak))
        if len(observed) > 1:
            recent_change_abs.append(abs(float(observed[-1] - observed[-2])))
        recent_trend_abs.append(abs(float(_recent_trend(observed))))

    target_series = pd.to_numeric(frame[target_field], errors="coerce") if target_field in frame.columns else pd.Series(dtype=float)
    target_observed = [float(value) for value in target_series.dropna().tolist()]
    time_gap_features, _ = _time_gap_features(frame)
    return {
        "observed_ratio_mean": float(sum(observed_ratios) / max(1, len(observed_ratios))) if observed_ratios else 0.0,
        "recent_observed_ratio_mean": (
            float(sum(recent_observed_ratios) / max(1, len(recent_observed_ratios))) if recent_observed_ratios else 0.0
        ),
        "mean_last_observed_gap_steps": float(sum(last_gaps) / max(1, len(last_gaps))) if last_gaps else 0.0,
        "max_last_observed_gap_steps": float(max(last_gaps)) if last_gaps else 0.0,
        "mean_longest_missing_streak": (
            float(sum(longest_missing_streaks) / max(1, len(longest_missing_streaks))) if longest_missing_streaks else 0.0
        ),
        "mean_trailing_missing_streak": (
            float(sum(trailing_missing_streaks) / max(1, len(trailing_missing_streaks))) if trailing_missing_streaks else 0.0
        ),
        "recent_change_mean_abs": (
            float(sum(recent_change_abs) / max(1, len(recent_change_abs))) if recent_change_abs else 0.0
        ),
        "recent_trend_mean_abs": (
            float(sum(recent_trend_abs) / max(1, len(recent_trend_abs))) if recent_trend_abs else 0.0
        ),
        "target_recent_trend": float(_recent_trend(target_observed)),
        "target_recent_acceleration": float(_recent_acceleration(target_observed)),
        "time_gap_last_hours": float(time_gap_features[0]),
        "time_gap_mean_hours": float(time_gap_features[1]),
        "time_gap_std_hours": float(time_gap_features[2]),
        "time_irregularity": float(time_gap_features[3]),
    }


def _align_named_feature_block(
    feature_values: Sequence[float],
    feature_names: Sequence[str],
    expected_feature_names: Sequence[str],
) -> Tuple[List[float], List[str]]:
    if not expected_feature_names:
        return [float(value) for value in feature_values], [str(name) for name in feature_names]
    feature_map = {
        str(name): float(feature_values[index])
        for index, name in enumerate(feature_names)
        if index < len(feature_values)
    }
    aligned_names = [str(name) for name in expected_feature_names]
    aligned_values = [float(feature_map.get(name, 0.0)) for name in aligned_names]
    return aligned_values, aligned_names


def _should_include_dynamic_patient_features(
    feature_schema: Optional[Dict[str, object]],
    physiology_sequence_columns: Sequence[str],
) -> bool:
    if feature_schema is None:
        return True
    explicit_flag = feature_schema.get("include_dynamic_patient_features")
    if explicit_flag is not None:
        return bool(explicit_flag)
    expected_patient_feature_names = list(feature_schema.get("patient_feature_names", []))
    if not expected_patient_feature_names:
        return True
    dynamic_feature_names = set(_patient_context_dynamic_features(pd.DataFrame(), physiology_sequence_columns)[1])
    return any(name in dynamic_feature_names for name in expected_patient_feature_names)


def _normalize_multivariate_window(
    context_frame: pd.DataFrame,
    sequence_columns: Sequence[str],
    target_field: str,
    future: Sequence[float],
) -> Tuple[List[List[float]], List[float], float, float]:
    context_matrix: List[List[float]] = []
    column_means: Dict[str, float] = {}
    column_scales: Dict[str, float] = {}
    for column in sequence_columns:
        values = [float(value) for value in pd.to_numeric(context_frame[column], errors="coerce").fillna(0.0).tolist()]
        mean_value, scale_value = _safe_scale(values if values else [0.0])
        column_means[column] = mean_value
        column_scales[column] = scale_value
    for _, row in context_frame.iterrows():
        step: List[float] = []
        for column in sequence_columns:
            raw_value = _safe_float(row.get(column))
            step.append((raw_value - column_means[column]) / column_scales[column])
        context_matrix.append(step)
    target_center = column_means.get(target_field, 0.0)
    target_scale = column_scales.get(target_field, 1.0)
    normalized_future = [(value - target_center) / target_scale for value in future]
    return context_matrix, normalized_future, target_center, target_scale


def _window_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> List[List[float]]:
    if not columns:
        return [[0.0] for _ in range(len(frame))]
    matrix: List[List[float]] = []
    for _, row in frame.iterrows():
        matrix.append([_safe_float(row.get(column)) for column in columns])
    return matrix


def _build_patient_context_features(
    context: Sequence[float],
    context_frame: pd.DataFrame,
    physiology_sequence_columns: Sequence[str],
    target_field: str,
    seasonality: int,
    forecast_horizon: int,
    include_dynamic_features: bool,
) -> Tuple[List[float], List[str]]:
    base_features = _window_static_features(context, seasonality, forecast_horizon)
    base_names = [f"context_target_{name}" for name in TSF_STATIC_FEATURE_NAMES]
    aggregate_features, aggregate_names = _aggregate_context_features(
        context_frame,
        [column for column in physiology_sequence_columns if column != target_field],
        prefix="physiology",
    )
    if include_dynamic_features:
        dynamic_features, dynamic_names = _patient_context_dynamic_features(
            context_frame,
            physiology_sequence_columns,
        )
    else:
        dynamic_features, dynamic_names = [], []
    return (
        base_features + aggregate_features + dynamic_features,
        base_names + aggregate_names + dynamic_names,
    )


def derive_eicu_sepsis3_feature_schema(
    labels_csv: str,
    trajectory_csv: str,
    target_field: str = "total_sofa",
) -> Dict[str, object]:
    labels_df = pd.read_csv(labels_csv)
    trajectory_df = pd.read_csv(trajectory_csv)
    if target_field not in trajectory_df.columns:
        raise ValueError(f"Unknown eICU target field: {target_field}")

    labels_df = labels_df.drop_duplicates(subset=["patientunitstayid"]).copy()
    label_numeric_columns, label_categorical_columns = _coerce_numeric_columns(labels_df, EICU_LABEL_EXCLUDED_COLUMNS)
    label_patient_numeric = [column for column in label_numeric_columns if not _is_intervention_feature(column)]
    label_intervention_numeric = [column for column in label_numeric_columns if _is_intervention_feature(column)]
    label_patient_categorical = [column for column in label_categorical_columns if not _is_intervention_feature(column)]
    label_intervention_categorical = [column for column in label_categorical_columns if _is_intervention_feature(column)]

    trajectory_numeric_columns, _ = _coerce_numeric_columns(trajectory_df, EICU_TRAJECTORY_EXCLUDED_COLUMNS)
    physiology_sequence_columns = [column for column in trajectory_numeric_columns if not _is_intervention_feature(column)]
    intervention_context_columns = [column for column in trajectory_numeric_columns if _is_intervention_feature(column)]
    if target_field not in physiology_sequence_columns:
        physiology_sequence_columns = [target_field] + [column for column in physiology_sequence_columns if column != target_field]

    return {
        "label_patient_numeric": label_patient_numeric,
        "label_intervention_numeric": label_intervention_numeric,
        "label_patient_categorical": label_patient_categorical,
        "label_intervention_categorical": label_intervention_categorical,
        "physiology_sequence_columns": physiology_sequence_columns,
        "intervention_context_columns": intervention_context_columns,
    }


def _build_serialized_forecast_sample(payload: Dict[str, object], forecast_horizon: int) -> ForecastSample:
    sequence = [[float(value) for value in step] for step in payload.get("sequence", [])]
    patient_static = [float(value) for value in payload.get("patient_static", [])]
    intervention_static = [float(value) for value in payload.get("intervention_static", [])]
    intervention_sequence = [
        [float(value) for value in step]
        for step in payload.get("intervention_sequence", [])
    ]
    metadata = dict(payload.get("metadata", {}))
    raw_context = [float(value) for value in payload.get("raw_context", [])]
    raw_target = [float(value) for value in payload.get("raw_target", [])]
    target = [float(value) for value in payload.get("target", [])]
    if not target:
        target = [0.0] * max(1, int(forecast_horizon))
    if not raw_target:
        raw_target = [0.0] * len(target)
    formation_features = [float(value) for value in payload.get("formation_features", [])]
    pattern_label = int(payload.get("pattern_label", 0))
    trajectory_label = int(payload.get("trajectory_label", 0))
    experience_label = int(
        payload.get(
            "experience_label",
            pattern_label * len(TRAJECTORY_LABELS) + trajectory_label,
        )
    )
    static = [float(value) for value in payload.get("static", [])]
    if not static:
        static = patient_static + intervention_static
    aux_targets = {
        str(key): float(value)
        for key, value in dict(payload.get("aux_targets", {})).items()
        if str(key) in EICU_AUX_TARGET_NAMES
    }
    if aux_targets:
        metadata["aux_targets"] = dict(aux_targets)
    return ForecastSample(
        sequence=sequence,
        static=static,
        target=target,
        metadata=metadata,
        scale_center=float(payload.get("scale_center", 0.0)),
        scale_value=max(1e-6, float(payload.get("scale_value", 1.0))),
        raw_context=raw_context,
        raw_target=raw_target,
        formation_features=formation_features,
        pattern_label=pattern_label,
        trajectory_label=trajectory_label,
        experience_label=experience_label,
        patient_static=patient_static,
        intervention_static=intervention_static,
        intervention_sequence=intervention_sequence,
        kg_features=[float(value) for value in payload.get("kg_features", [])],
        aux_targets=aux_targets or None,
    )


def build_eicu_sepsis3_inference_sample(
    payload: Dict[str, object],
    labels_csv: str,
    trajectory_csv: str,
    dataset_name: str = "eicu_sepsis3_inference",
    history_length: int = 4,
    forecast_horizon: int = 2,
    target_field: str = "total_sofa",
    enable_kg: bool = False,
    kg_directory: str = "",
    append_kg_to_patient_static: bool = True,
    seasonality: int = 4,
    feature_schema: Optional[Dict[str, object]] = None,
) -> ForecastSample:
    sample_type = str(payload.get("sample_type", "")).strip().lower()
    if sample_type == "forecast_sample" or "sequence" in payload:
        return _build_serialized_forecast_sample(payload, forecast_horizon=forecast_horizon)

    label_row = dict(payload.get("label_row", {}))
    context_rows = list(payload.get("context_rows", []))
    if not context_rows:
        raise ValueError("Inference payload must include non-empty context_rows.")
    if len(context_rows) < history_length:
        raise ValueError(
            f"Inference payload only has {len(context_rows)} context rows; history_length={history_length} is required."
        )

    schema = feature_schema or derive_eicu_sepsis3_feature_schema(
        labels_csv=labels_csv,
        trajectory_csv=trajectory_csv,
        target_field=target_field,
    )
    label_patient_numeric = list(schema.get("label_patient_numeric", []))
    label_intervention_numeric = list(schema.get("label_intervention_numeric", []))
    label_patient_categorical = list(schema.get("label_patient_categorical", []))
    label_intervention_categorical = list(schema.get("label_intervention_categorical", []))
    physiology_sequence_columns = list(schema.get("physiology_sequence_columns", []))
    intervention_context_columns = list(schema.get("intervention_context_columns", []))
    if target_field not in physiology_sequence_columns:
        raise ValueError(f"Target field {target_field} is missing from physiology_sequence_columns.")
    include_dynamic_patient_features = _should_include_dynamic_patient_features(
        feature_schema,
        physiology_sequence_columns,
    )

    stay_id = payload.get("stay_id", label_row.get("patientunitstayid", -1))
    if "patientunitstayid" not in label_row and stay_id is not None:
        label_row["patientunitstayid"] = stay_id

    context_frame = pd.DataFrame(context_rows).copy()
    if target_field not in context_frame.columns:
        raise ValueError(f"context_rows must include target field {target_field}.")
    if "bin_index" in context_frame.columns:
        context_frame["bin_index"] = pd.to_numeric(context_frame["bin_index"], errors="coerce")
        if context_frame["bin_index"].notna().sum() > 0:
            context_frame = context_frame.sort_values("bin_index").reset_index(drop=True)
    context_frame = context_frame.tail(history_length).reset_index(drop=True)

    context = [float(value) for value in pd.to_numeric(context_frame[target_field], errors="coerce").fillna(0.0).tolist()]
    future = [float(value) for value in payload.get("future_target", [])]
    if not future:
        future = [float(context[-1])] * max(1, int(forecast_horizon))
    if len(future) != forecast_horizon:
        raise ValueError(f"future_target length={len(future)} does not match forecast_horizon={forecast_horizon}.")

    last_row = context_frame.iloc[-1].to_dict()
    end_index = int(_safe_float(last_row.get("bin_index"), float(len(context_rows) - 1)))
    formation = build_window_formation(context, seasonality=seasonality, end_index=end_index)
    sequence, norm_target, center, scale = _normalize_multivariate_window(
        context_frame=context_frame,
        sequence_columns=physiology_sequence_columns,
        target_field=target_field,
        future=future,
    )
    patient_context_features, patient_context_feature_names = _build_patient_context_features(
        context=context,
        context_frame=context_frame,
        physiology_sequence_columns=physiology_sequence_columns,
        target_field=target_field,
        seasonality=seasonality,
        forecast_horizon=forecast_horizon,
        include_dynamic_features=include_dynamic_patient_features,
    )
    expected_patient_feature_names = list(feature_schema.get("patient_feature_names", [])) if feature_schema else []
    if expected_patient_feature_names:
        context_name_set = set(patient_context_feature_names)
        expected_context_feature_names = [
            str(name)
            for name in expected_patient_feature_names
            if str(name) in context_name_set
        ]
        if expected_context_feature_names:
            patient_context_features, patient_context_feature_names = _align_named_feature_block(
                patient_context_features,
                patient_context_feature_names,
                expected_context_feature_names,
            )
    intervention_context_features, _ = _aggregate_context_features(
        context_frame,
        intervention_context_columns,
        prefix="intervention",
    )
    intervention_sequence = _window_matrix(context_frame, intervention_context_columns)
    label_patient_features = _encode_numeric_row(label_row, label_patient_numeric) + _encode_categorical_row(
        label_row,
        label_patient_categorical,
    )
    label_intervention_features = _encode_numeric_row(label_row, label_intervention_numeric) + _encode_categorical_row(
        label_row,
        label_intervention_categorical,
    )
    kg_builder = KnowledgeGraphFeatureBuilder.from_directory(kg_directory) if enable_kg else None
    kg_features: List[float] = []
    kg_flags: Dict[str, float] = {}
    kg_guideline_alignment = 0.0
    if kg_builder is not None:
        kg_features, kg_flags, kg_guideline_alignment = kg_builder.build_features(
            label_row=label_row,
            context_frame=context_frame,
        )

    patient_static = patient_context_features + label_patient_features
    if append_kg_to_patient_static:
        patient_static = patient_static + kg_features
    intervention_static = label_intervention_features + intervention_context_features
    metadata = {
        "stay_id": float(_safe_float(stay_id, -1.0)),
        "series_index": -1.0,
        "series_name": str(payload.get("series_name", f"stay_{int(_safe_float(stay_id, -1.0))}" if _safe_float(stay_id, -1.0) >= 0 else "inference_case")),
        "window_end_index": float(end_index),
        "window_rel_end_hours": _safe_float(last_row.get("rel_end_hours")),
        "dataset_name": dataset_name,
        "series_count": 1.0,
        "seasonality": float(seasonality),
        "history_length": float(history_length),
        "forecast_horizon": float(forecast_horizon),
        "target_field": target_field,
        "patient_feature_dim": float(len(patient_static)),
        "intervention_feature_dim": float(len(intervention_static)),
        "kg_enabled": 1.0 if kg_builder is not None else 0.0,
        "kg_appended_to_patient_static": 1.0 if append_kg_to_patient_static else 0.0,
        "kg_guideline_alignment": float(kg_guideline_alignment),
        "inference_payload_mode": "raw_window",
        "dynamic_profile": _dynamic_behavior_profile(
            frame=context_frame,
            physiology_sequence_columns=physiology_sequence_columns,
            target_field=target_field,
        ),
        "temporal_feature_version": "phase1_dynamic_v2",
    }
    for key in EICU_CONTEXTUAL_METADATA_KEYS:
        if key in label_row:
            safe_value = _safe_metadata_value(label_row.get(key))
            if safe_value is not None:
                metadata[key] = safe_value
    for key in [
        "sepsis3_label",
        "septic_shock_label_operational",
        "septic_shock_label_relaxed",
        "pre_baseline_total_sofa",
    ]:
        if key in label_row:
            metadata[key] = _safe_float(label_row.get(key))
    if kg_flags:
        metadata["kg_flags"] = kg_flags
    metadata.update({str(key): value for key, value in dict(payload.get("metadata", {})).items()})
    aux_targets = {
        str(key): float(value)
        for key, value in dict(payload.get("aux_targets", metadata.get("aux_targets", {}))).items()
        if str(key) in EICU_AUX_TARGET_NAMES
    }
    if aux_targets:
        metadata["aux_targets"] = dict(aux_targets)

    return ForecastSample(
        sequence=sequence,
        static=patient_static + intervention_static,
        target=norm_target,
        metadata=metadata,
        scale_center=center,
        scale_value=scale,
        raw_context=context,
        raw_target=future,
        formation_features=formation.features,
        pattern_label=formation.pattern_label,
        trajectory_label=formation.trajectory_label,
        experience_label=formation.experience_label,
        patient_static=patient_static,
        intervention_static=intervention_static,
        intervention_sequence=intervention_sequence,
        kg_features=kg_features,
        aux_targets=aux_targets or None,
    )


def build_eicu_sepsis3_forecasting_dataset(
    labels_csv: str,
    trajectory_csv: str,
    dataset_name: str = "eicu_sepsis3_forecasting",
    history_length: int = 4,
    forecast_horizon: int = 2,
    max_train_windows_per_series: int = 24,
    target_field: str = "total_sofa",
    max_series_count: Optional[int] = None,
    enable_kg: bool = False,
    kg_directory: str = "",
    append_kg_to_patient_static: bool = True,
    feature_schema: Optional[Dict[str, object]] = None,
) -> ForecastingDataset:
    labels_df = pd.read_csv(labels_csv)
    trajectory_df = pd.read_csv(trajectory_csv)
    if target_field not in trajectory_df.columns:
        raise ValueError(f"Unknown eICU target field: {target_field}")

    labels_df = labels_df.drop_duplicates(subset=["patientunitstayid"]).copy()
    label_lookup = {
        int(row["patientunitstayid"]): row.to_dict()
        for _, row in labels_df.iterrows()
        if not pd.isna(row.get("patientunitstayid"))
    }

    trajectory_df[target_field] = pd.to_numeric(trajectory_df[target_field], errors="coerce")
    trajectory_df["patientunitstayid"] = pd.to_numeric(trajectory_df["patientunitstayid"], errors="coerce")
    trajectory_df["bin_index"] = pd.to_numeric(trajectory_df["bin_index"], errors="coerce")
    trajectory_df["rel_end_hours"] = pd.to_numeric(trajectory_df["rel_end_hours"], errors="coerce")
    trajectory_df = trajectory_df.dropna(subset=["patientunitstayid", "bin_index", target_field]).copy()
    trajectory_df["patientunitstayid"] = trajectory_df["patientunitstayid"].astype(int)
    trajectory_df = trajectory_df.sort_values(["patientunitstayid", "bin_index"]).reset_index(drop=True)
    if max_series_count and max_series_count > 0:
        allowed_stays = set(trajectory_df["patientunitstayid"].drop_duplicates().tolist()[:max_series_count])
        trajectory_df = trajectory_df[trajectory_df["patientunitstayid"].isin(allowed_stays)].reset_index(drop=True)

    schema = feature_schema or derive_eicu_sepsis3_feature_schema(
        labels_csv=labels_csv,
        trajectory_csv=trajectory_csv,
        target_field=target_field,
    )
    label_patient_numeric = list(schema.get("label_patient_numeric", []))
    label_intervention_numeric = list(schema.get("label_intervention_numeric", []))
    label_patient_categorical = list(schema.get("label_patient_categorical", []))
    label_intervention_categorical = list(schema.get("label_intervention_categorical", []))
    physiology_sequence_columns = list(schema.get("physiology_sequence_columns", []))
    intervention_context_columns = list(schema.get("intervention_context_columns", []))
    if target_field not in physiology_sequence_columns:
        physiology_sequence_columns = [target_field] + [column for column in physiology_sequence_columns if column != target_field]
    include_dynamic_patient_features = _should_include_dynamic_patient_features(
        feature_schema,
        physiology_sequence_columns,
    )

    patient_label_feature_names = (
        [_normalize_name(column) for column in label_patient_numeric]
        + _categorical_feature_names(label_patient_categorical)
    )
    intervention_label_feature_names = (
        [_normalize_name(column) for column in label_intervention_numeric]
        + _categorical_feature_names(label_intervention_categorical)
    )

    patient_context_feature_names = _build_patient_context_features(
        context=[0.0] * max(1, history_length),
        context_frame=pd.DataFrame(columns=list(physiology_sequence_columns) + ["rel_end_hours"]),
        physiology_sequence_columns=physiology_sequence_columns,
        target_field=target_field,
        seasonality=4,
        forecast_horizon=forecast_horizon,
        include_dynamic_features=True,
    )[1]
    intervention_context_feature_names = _aggregate_context_features(
        pd.DataFrame(columns=intervention_context_columns),
        intervention_context_columns,
        prefix="intervention",
    )[1]
    kg_builder = KnowledgeGraphFeatureBuilder.from_directory(kg_directory) if enable_kg else None
    kg_feature_names = list(kg_builder.feature_names) if kg_builder is not None else []

    patient_feature_names = list(patient_context_feature_names + patient_label_feature_names)
    if append_kg_to_patient_static:
        patient_feature_names.extend(kg_feature_names)
    intervention_feature_names = intervention_label_feature_names + intervention_context_feature_names
    static_feature_names = patient_feature_names + intervention_feature_names
    sequence_feature_names = list(physiology_sequence_columns)
    formation_names = formation_feature_names()
    train_samples: List[ForecastSample] = []
    val_samples: List[ForecastSample] = []
    test_samples: List[ForecastSample] = []
    lengths: List[int] = []
    seasonality = 4
    series_total = int(trajectory_df["patientunitstayid"].nunique())

    for series_idx, (stay_id, group) in enumerate(trajectory_df.groupby("patientunitstayid", sort=False)):
        label_row = label_lookup.get(int(stay_id), {})
        series_frame = group.sort_values("bin_index").reset_index(drop=True)
        values = [float(value) for value in series_frame[target_field].tolist()]
        lengths.append(len(values))
        if len(values) < history_length + 3 * forecast_horizon:
            continue

        series_name = f"stay_{int(stay_id)}"
        test_target_start = len(values) - forecast_horizon
        val_target_start = test_target_start - forecast_horizon
        train_last_end = val_target_start - forecast_horizon
        if train_last_end < history_length:
            continue

        label_patient_features = _encode_numeric_row(label_row, label_patient_numeric) + _encode_categorical_row(
            label_row,
            label_patient_categorical,
        )
        label_intervention_features = _encode_numeric_row(label_row, label_intervention_numeric) + _encode_categorical_row(
            label_row,
            label_intervention_categorical,
        )

        def build_sample(target_start: int) -> ForecastSample:
            context_start = target_start - history_length
            context = values[context_start:target_start]
            future = values[target_start : target_start + forecast_horizon]
            formation = build_window_formation(context, seasonality=seasonality, end_index=target_start)
            context_frame = series_frame.iloc[context_start:target_start].reset_index(drop=True)
            future_frame = series_frame.iloc[target_start : target_start + forecast_horizon].reset_index(drop=True)
            sequence, norm_target, center, scale = _normalize_multivariate_window(
                context_frame=context_frame,
                sequence_columns=physiology_sequence_columns,
                target_field=target_field,
                future=future,
            )
            row = series_frame.iloc[target_start]
            patient_context_features, patient_context_feature_names = _build_patient_context_features(
                context=context,
                context_frame=context_frame,
                physiology_sequence_columns=physiology_sequence_columns,
                target_field=target_field,
                seasonality=seasonality,
                forecast_horizon=forecast_horizon,
                include_dynamic_features=include_dynamic_patient_features,
            )
            expected_patient_feature_names = list(feature_schema.get("patient_feature_names", [])) if feature_schema else []
            if expected_patient_feature_names:
                context_name_set = set(patient_context_feature_names)
                expected_context_feature_names = [
                    str(name)
                    for name in expected_patient_feature_names
                    if str(name) in context_name_set
                ]
                if expected_context_feature_names:
                    patient_context_features, patient_context_feature_names = _align_named_feature_block(
                        patient_context_features,
                        patient_context_feature_names,
                        expected_context_feature_names,
                    )
            intervention_context_features, _ = _aggregate_context_features(
                context_frame,
                intervention_context_columns,
                prefix="intervention",
            )
            intervention_sequence = _window_matrix(
                context_frame,
                intervention_context_columns,
            )
            kg_features: List[float] = []
            kg_flags: Dict[str, float] = {}
            kg_guideline_alignment = 0.0
            if kg_builder is not None:
                kg_features, kg_flags, kg_guideline_alignment = kg_builder.build_features(
                    label_row=label_row,
                    context_frame=context_frame,
                )
            patient_static = patient_context_features + label_patient_features
            if append_kg_to_patient_static:
                patient_static = patient_static + kg_features
            intervention_static = label_intervention_features + intervention_context_features
            metadata = {
                "stay_id": float(stay_id),
                "series_index": float(series_idx),
                "series_name": series_name,
                "window_end_index": float(row["bin_index"]),
                "window_rel_end_hours": _safe_float(row.get("rel_end_hours")),
                "dataset_name": dataset_name,
                "series_count": float(series_total),
                "seasonality": float(seasonality),
                "history_length": float(history_length),
                "forecast_horizon": float(forecast_horizon),
                "target_field": target_field,
                "patient_feature_dim": float(len(patient_static)),
                "intervention_feature_dim": float(len(intervention_static)),
                "kg_enabled": 1.0 if kg_builder is not None else 0.0,
                "kg_appended_to_patient_static": 1.0 if append_kg_to_patient_static else 0.0,
                "kg_guideline_alignment": float(kg_guideline_alignment),
                "dynamic_profile": _dynamic_behavior_profile(
                    frame=context_frame,
                    physiology_sequence_columns=physiology_sequence_columns,
                    target_field=target_field,
                ),
                "temporal_feature_version": "phase1_dynamic_v2",
            }
            for key in EICU_CONTEXTUAL_METADATA_KEYS:
                if key in label_row:
                    safe_value = _safe_metadata_value(label_row.get(key))
                    if safe_value is not None:
                        metadata[key] = safe_value
            for key in [
                "sepsis3_label",
                "septic_shock_label_operational",
                "septic_shock_label_relaxed",
                "pre_baseline_total_sofa",
            ]:
                if key in label_row:
                    metadata[key] = _safe_float(label_row.get(key))
            if kg_flags:
                metadata["kg_flags"] = kg_flags
            aux_targets = _build_eicu_aux_targets(
                context=context,
                context_frame=context_frame,
                future=future,
                future_frame=future_frame,
            )
            metadata["aux_targets"] = dict(aux_targets)
            return ForecastSample(
                sequence=sequence,
                static=patient_static + intervention_static,
                target=norm_target,
                metadata=metadata,
                scale_center=center,
                scale_value=scale,
                raw_context=context,
                raw_target=future,
                formation_features=formation.features,
                pattern_label=formation.pattern_label,
                trajectory_label=formation.trajectory_label,
                experience_label=formation.experience_label,
                patient_static=patient_static,
                intervention_static=intervention_static,
                intervention_sequence=intervention_sequence,
                kg_features=kg_features,
                aux_targets=aux_targets,
            )

        val_samples.append(build_sample(val_target_start))
        test_samples.append(build_sample(test_target_start))

        candidate_end_indices = list(range(history_length, train_last_end + 1))
        if max_train_windows_per_series > 0 and len(candidate_end_indices) > max_train_windows_per_series:
            step = (len(candidate_end_indices) - 1) / max(1, max_train_windows_per_series - 1)
            candidate_end_indices = sorted(
                {
                    candidate_end_indices[min(len(candidate_end_indices) - 1, int(round(step * idx)))]
                    for idx in range(max_train_windows_per_series)
                }
            )
        for end_index in candidate_end_indices:
            train_samples.append(build_sample(end_index))

    if not lengths:
        raise ValueError(f"No valid eICU series found in {trajectory_csv}.")

    return ForecastingDataset(
        dataset_name=dataset_name,
        source_path=f"trajectory={trajectory_csv};labels={labels_csv}",
        frequency="6h",
        seasonality=seasonality,
        forecast_horizon=int(forecast_horizon),
        history_length=int(history_length),
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        series_count=len(lengths),
        min_series_length=min(lengths),
        mean_series_length=sum(lengths) / max(1, len(lengths)),
        max_series_length=max(lengths),
        static_feature_names=static_feature_names,
        sequence_feature_names=sequence_feature_names,
        formation_feature_names=formation_names,
        patient_feature_names=patient_feature_names,
        intervention_feature_names=intervention_feature_names,
        intervention_sequence_feature_names=list(intervention_context_columns),
        kg_feature_names=kg_feature_names,
        aux_target_names=list(EICU_AUX_TARGET_NAMES),
    )
