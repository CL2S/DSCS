import csv
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from src.data_utils import standardize_by_train


SOFA_FIELDS = [
    "sofa_respiration",
    "sofa_coagulation",
    "sofa_liver",
    "sofa_cardiovascular",
    "sofa_cns",
    "sofa_renal",
    "sofa_total",
]


@dataclass
class TemporalPatientRecord:
    stay_id: int
    subject_id: int
    input_description: str
    output_summary: str
    sofa_scores: List[Dict[str, float]]
    sofa_scores_post_icu: List[Dict[str, float]]
    extra_labels: Dict[str, float] = field(default_factory=dict)


@dataclass
class TemporalDataset:
    x_data: List[List[float]]
    y_data: List[int]
    feature_names: List[str]
    label_names: List[str]
    patient_ids: List[int]
    records: List[TemporalPatientRecord]
    metadata: List[Dict[str, float]]
    task_type: str
    target_mode: str
    sequence_data: List[List[List[float]]]
    static_data: List[List[float]]
    sequence_feature_names: List[str]


def load_temporal_patient_json(json_path: str) -> List[TemporalPatientRecord]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    records: List[TemporalPatientRecord] = []
    for item in data:
        records.append(
            TemporalPatientRecord(
                stay_id=int(item["stay_id"]),
                subject_id=int(item["subject_id"]),
                input_description=str(item.get("input_description", "")),
                output_summary=str(item.get("output_summary", "")),
                sofa_scores=list(item.get("sofa_scores", [])),
                sofa_scores_post_icu=list(item.get("sofa_scores_post_icu", [])),
                extra_labels=dict(item.get("extra_labels", {})),
            )
        )
    return records


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_text(value: str) -> str:
    return str(value or "").strip()


def _sex_text(raw_gender: str) -> str:
    lowered = _safe_text(raw_gender).lower()
    if lowered.startswith("m"):
        return "male"
    if lowered.startswith("f"):
        return "female"
    return "unknown sex"


def _format_age_text(raw_age: str, age_years: str) -> str:
    text = _safe_text(raw_age)
    if text:
        if ">" in text:
            return f"age > {text.replace('>', '').strip()} years"
        return f"age {text} years"
    years = _safe_float(age_years, 0.0)
    return f"age {int(years)} years" if years > 0 else "age unknown"


def _format_weight_text(value: str) -> str:
    weight = _safe_float(value, 0.0)
    return f"weight {weight:.1f} kg" if weight > 0 else "weight unknown"


def _build_eicu_input_description(label_row: Dict[str, str]) -> str:
    age_text = _format_age_text(label_row.get("age_raw", ""), label_row.get("age_years", ""))
    sex_text = _sex_text(label_row.get("gender", ""))
    weight_text = _format_weight_text(label_row.get("admissionweight_kg", ""))
    anchor_type = _safe_text(label_row.get("infection_anchor_type", ""))
    anchor_value = _safe_text(label_row.get("infection_anchor_value", ""))
    antibiotic_name = _safe_text(label_row.get("antibiotic_name", ""))
    unit_type = _safe_text(label_row.get("unittype", ""))
    diagnosis = _safe_text(label_row.get("apacheadmissiondx", ""))
    return (
        f"ICU stay {label_row.get('patientunitstayid', '')}. "
        f"Health-system stay {label_row.get('patienthealthsystemstayid', '')}. "
        f"{age_text}. Sex {sex_text}. {weight_text}. "
        f"Unit type: {unit_type or 'unknown'}. "
        f"Admission diagnosis: {diagnosis or 'unknown'}. "
        f"Infection anchor type: {anchor_type or 'unknown'}. "
        f"Infection anchor detail: {anchor_value or 'unknown'}. "
        f"Related antibiotic: {antibiotic_name or 'unknown'}."
    )


def _build_eicu_output_summary(
    label_row: Dict[str, str],
    pre_steps: Sequence[Dict[str, float]],
    post_steps: Sequence[Dict[str, float]],
) -> str:
    pre_last = float(pre_steps[-1].get("sofa_total", 0.0)) if pre_steps else 0.0
    post_max = max(float(step.get("sofa_total", 0.0)) for step in post_steps) if post_steps else 0.0
    delta = post_max - pre_last
    return (
        f"Suspected infection offset {label_row.get('suspected_infection_offset_minutes', '')} minutes. "
        f"Pre-infection final SOFA total {pre_last:.1f}; post-infection 24h max SOFA total {post_max:.1f}; "
        f"delta {delta:.1f}. "
        f"Sepsis-3 label {label_row.get('sepsis3_label', '')}. "
        f"Operational septic shock label {label_row.get('septic_shock_label_operational', '')}."
    )


def load_eicu_sepsis3_csv(labels_csv: str, trajectory_csv: str) -> List[TemporalPatientRecord]:
    label_rows: Dict[str, Dict[str, str]] = {}
    with open(labels_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_rows[str(row["patientunitstayid"])] = row

    trajectory_rows: Dict[str, List[Dict[str, str]]] = {}
    with open(trajectory_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trajectory_rows.setdefault(str(row["patientunitstayid"]), []).append(row)

    records: List[TemporalPatientRecord] = []
    for stay_id, label_row in label_rows.items():
        rows = sorted(trajectory_rows.get(stay_id, []), key=lambda item: _safe_int(item.get("bin_index", "0")))
        if not rows:
            continue

        pre_steps: List[Dict[str, float]] = []
        post_steps: List[Dict[str, float]] = []
        for row in rows:
            step = {
                "sofa_respiration": _safe_float(row.get("resp_score", "0")),
                "sofa_coagulation": _safe_float(row.get("coag_score", "0")),
                "sofa_liver": _safe_float(row.get("liver_score", "0")),
                "sofa_cardiovascular": _safe_float(row.get("cardio_score", "0")),
                "sofa_cns": _safe_float(row.get("cns_score", "0")),
                "sofa_renal": _safe_float(row.get("renal_score", "0")),
                "sofa_total": _safe_float(row.get("total_sofa", "0")),
            }
            if _safe_int(row.get("bin_index", "0")) < 0:
                pre_steps.append(step)
            else:
                post_steps.append(step)

        if not pre_steps or not post_steps:
            continue

        extra_labels = {
            "sepsis3_label": _safe_float(label_row.get("sepsis3_label", "0")),
            "sepsis3_label_baseline0": _safe_float(label_row.get("sepsis3_label_baseline0", "0")),
            "septic_shock_label_operational": _safe_float(label_row.get("septic_shock_label_operational", "0")),
            "septic_shock_label_relaxed": _safe_float(label_row.get("septic_shock_label_relaxed", "0")),
            "pre_baseline_total_sofa": _safe_float(label_row.get("pre_baseline_total_sofa", "0")),
            "post_24h_max_total_sofa": _safe_float(label_row.get("post_24h_max_total_sofa", "0")),
            "delta_sofa_post_vs_pre": _safe_float(label_row.get("delta_sofa_post_vs_pre", "0")),
        }

        subject_id = _safe_int(label_row.get("patienthealthsystemstayid", "0"), _safe_int(stay_id))
        records.append(
            TemporalPatientRecord(
                stay_id=_safe_int(stay_id),
                subject_id=subject_id,
                input_description=_build_eicu_input_description(label_row),
                output_summary=_build_eicu_output_summary(label_row, pre_steps, post_steps),
                sofa_scores=pre_steps,
                sofa_scores_post_icu=post_steps,
                extra_labels=extra_labels,
            )
        )
    return records


def load_temporal_records(
    json_path: str = "",
    dataset_format: str = "legacy_json",
    eicu_sepsis3_labels_csv: str = "",
    eicu_sepsis3_trajectory_csv: str = "",
) -> List[TemporalPatientRecord]:
    if dataset_format == "eicu_sepsis3":
        if not eicu_sepsis3_labels_csv or not eicu_sepsis3_trajectory_csv:
            raise ValueError("eICU Sepsis-3 dataset format requires both labels CSV and trajectory CSV.")
        return load_eicu_sepsis3_csv(eicu_sepsis3_labels_csv, eicu_sepsis3_trajectory_csv)
    if not json_path:
        raise ValueError("legacy_json dataset format requires --json input.")
    return load_temporal_patient_json(json_path)


def _extract_age(text: str) -> float:
    match = re.search(r"(\d+)\u5c81", text) or re.search(
        r"\bage\s*(?:>|>=)?\s*(\d+)\s*years?\b",
        text,
        flags=re.IGNORECASE,
    )
    return float(match.group(1)) if match else 0.0


def _extract_weight(text: str) -> float:
    match = re.search(r"\u4f53\u91cd\s*([0-9.]+)kg", text) or re.search(
        r"\bweight\s*([0-9.]+)\s*kg\b",
        text,
        flags=re.IGNORECASE,
    )
    return float(match.group(1)) if match else 0.0


def _extract_sex_flags(text: str) -> Tuple[float, float]:
    lowered = text.lower()
    if "\u5973" in text or "female" in lowered:
        return 0.0, 1.0
    if "\u7537" in text or "male" in lowered:
        return 1.0, 0.0
    return 0.0, 0.0


def _get_vector(step: Dict[str, float], fields: Sequence[str]) -> List[float]:
    return [float(step.get(field, 0.0)) for field in fields]


def _build_sequence_matrix(steps: Sequence[Dict[str, float]], fields: Sequence[str]) -> List[List[float]]:
    return [_get_vector(step, fields) for step in steps]


def _flatten_steps(steps: Sequence[Dict[str, float]], fields: Sequence[str], prefix: str) -> Tuple[List[float], List[str]]:
    values: List[float] = []
    names: List[str] = []
    for idx, step in enumerate(steps):
        vector = _get_vector(step, fields)
        for field, value in zip(fields, vector):
            names.append(f"{prefix}_t{idx}_{field}")
            values.append(value)
    return values, names


def _build_sequence_stats(
    steps: Sequence[Dict[str, float]],
    fields: Sequence[str],
    prefix: str,
) -> Tuple[List[float], List[str]]:
    values: List[float] = []
    names: List[str] = []
    vectors = [_get_vector(step, fields) for step in steps]
    count = len(vectors)

    for field_idx, field in enumerate(fields):
        series = [vector[field_idx] for vector in vectors]
        first = series[0]
        last = series[-1]
        mean_value = sum(series) / max(1, count)
        min_value = min(series)
        max_value = max(series)
        span = max_value - min_value
        slope = (last - first) / max(1, count - 1)
        volatility = sum(abs(series[idx] - series[idx - 1]) for idx in range(1, len(series))) / max(1, len(series) - 1)
        names.extend(
            [
                f"{prefix}_{field}_first",
                f"{prefix}_{field}_last",
                f"{prefix}_{field}_mean",
                f"{prefix}_{field}_min",
                f"{prefix}_{field}_max",
                f"{prefix}_{field}_span",
                f"{prefix}_{field}_slope",
                f"{prefix}_{field}_volatility",
            ]
        )
        values.extend([first, last, mean_value, min_value, max_value, span, slope, volatility])
    return values, names


def _build_delta_features(
    steps: Sequence[Dict[str, float]],
    fields: Sequence[str],
    prefix: str,
) -> Tuple[List[float], List[str]]:
    values: List[float] = []
    names: List[str] = []
    vectors = [_get_vector(step, fields) for step in steps]
    for idx in range(1, len(vectors)):
        for field_idx, field in enumerate(fields):
            names.append(f"{prefix}_delta_t{idx}_{field}")
            values.append(vectors[idx][field_idx] - vectors[idx - 1][field_idx])
    return values, names


def _count_threshold_crossings(series: Sequence[float], threshold: float) -> int:
    return sum(1 for value in series if value >= threshold)


def _build_prototype_features(
    steps: Sequence[Dict[str, float]],
    fields: Sequence[str],
    prefix: str,
) -> Tuple[List[float], List[str]]:
    values: List[float] = []
    names: List[str] = []
    vectors = [_get_vector(step, fields) for step in steps]
    total_series = [vector[-1] for vector in vectors]

    total_first = total_series[0]
    total_last = total_series[-1]
    total_mean = sum(total_series) / len(total_series)
    total_min = min(total_series)
    total_max = max(total_series)
    total_span = total_max - total_min
    total_slope = (total_last - total_first) / max(1, len(total_series) - 1)
    total_volatility = sum(abs(total_series[idx] - total_series[idx - 1]) for idx in range(1, len(total_series))) / max(
        1, len(total_series) - 1
    )
    names.extend(
        [
            f"{prefix}_total_first",
            f"{prefix}_total_last",
            f"{prefix}_total_mean",
            f"{prefix}_total_min",
            f"{prefix}_total_max",
            f"{prefix}_total_span",
            f"{prefix}_total_slope",
            f"{prefix}_total_volatility",
            f"{prefix}_total_high10_count",
            f"{prefix}_total_high12_count",
        ]
    )
    values.extend(
        [
            total_first,
            total_last,
            total_mean,
            total_min,
            total_max,
            total_span,
            total_slope,
            total_volatility,
            float(_count_threshold_crossings(total_series, 10.0)),
            float(_count_threshold_crossings(total_series, 12.0)),
        ]
    )

    organ_fields = list(fields[:-1])
    for field_idx, field in enumerate(organ_fields):
        series = [vector[field_idx] for vector in vectors]
        mean_value = sum(series) / len(series)
        max_value = max(series)
        last_value = series[-1]
        burden_count = sum(1 for value in series if value >= 2.0)
        active_count = sum(1 for value in series if value >= 1.0)
        slope = (series[-1] - series[0]) / max(1, len(series) - 1)
        names.extend(
            [
                f"{prefix}_{field}_mean",
                f"{prefix}_{field}_max",
                f"{prefix}_{field}_last",
                f"{prefix}_{field}_active_count",
                f"{prefix}_{field}_burden_count",
                f"{prefix}_{field}_slope",
            ]
        )
        values.extend([mean_value, max_value, last_value, float(active_count), float(burden_count), slope])

    return values, names


def _build_static_features(record: TemporalPatientRecord) -> Tuple[List[float], List[str]]:
    age = _extract_age(record.input_description)
    weight = _extract_weight(record.input_description)
    male_flag, female_flag = _extract_sex_flags(record.input_description)
    return [age, weight, male_flag, female_flag], ["age_years", "weight_kg", "sex_male", "sex_female"]


def _trajectory_shape_label(
    pre_steps: Sequence[Dict[str, float]],
    post_steps: Sequence[Dict[str, float]],
    balanced: bool = False,
) -> str:
    baseline_total = float(pre_steps[-1].get("sofa_total", 0.0))
    post_totals = [float(step.get("sofa_total", 0.0)) for step in post_steps]
    mean_delta = (sum(post_totals) / len(post_totals)) - baseline_total
    last_delta = post_totals[-1] - baseline_total
    span = max(post_totals) - min(post_totals)

    if mean_delta >= 2.0 or last_delta >= 2.0:
        return "worsen"
    if mean_delta <= -2.0 or last_delta <= -2.0:
        return "improve"
    if balanced:
        return "mixed"
    if span >= 4.0:
        return "volatile"
    return "stable"


def _patient_row_from_steps(
    record: TemporalPatientRecord,
    pre_steps: Sequence[Dict[str, float]],
    post_steps: Sequence[Dict[str, float]],
    sofa_fields: Sequence[str],
    worsening_delta: float,
    feature_mode: str,
    target_mode: str,
    label_to_id: Dict[str, int],
) -> Tuple[List[float], List[str], int, Dict[str, float]]:
    if feature_mode == "prototype":
        pre_values, pre_names = _build_prototype_features(pre_steps, sofa_fields, "pre")
    elif feature_mode == "stats":
        pre_values, pre_names = _build_sequence_stats(pre_steps, sofa_fields, "pre")
    else:
        flat_values, flat_names = _flatten_steps(pre_steps, sofa_fields, "pre")
        stat_values, stat_names = _build_sequence_stats(pre_steps, sofa_fields, "pre")
        delta_values, delta_names = _build_delta_features(pre_steps, sofa_fields, "pre")
        pre_values = flat_values + stat_values + delta_values
        pre_names = flat_names + stat_names + delta_names

    static_values, static_names = _build_static_features(record)

    last_pre_total = float(pre_steps[-1].get("sofa_total", 0.0))
    first_post_total = float(post_steps[0].get("sofa_total", 0.0))
    max_post_total = max(float(step.get("sofa_total", 0.0)) for step in post_steps)
    mean_post_total = sum(float(step.get("sofa_total", 0.0)) for step in post_steps) / len(post_steps)

    handcrafted_values = [
        float(len(pre_steps)),
        float(last_pre_total >= 10.0),
    ]
    handcrafted_names = [
        "history_length",
        "pre_last_total_ge_10",
    ]

    row = pre_values + static_values + handcrafted_values
    names = pre_names + static_names + handcrafted_names

    if target_mode == "trajectory_shape":
        label_name = _trajectory_shape_label(pre_steps, post_steps, balanced=False)
    elif target_mode == "trajectory_shape_balanced":
        label_name = _trajectory_shape_label(pre_steps, post_steps, balanced=True)
    elif target_mode in {
        "sepsis3_label",
        "sepsis3_label_baseline0",
        "septic_shock_label_operational",
        "septic_shock_label_relaxed",
    }:
        target_value = float(record.extra_labels.get(target_mode, 0.0))
        label_name = "positive" if target_value >= 0.5 else "negative"
    else:
        label_name = "worsen" if max_post_total >= last_pre_total + worsening_delta else "non_worsen"

    if label_name not in label_to_id:
        label_to_id[label_name] = len(label_to_id)
    label = label_to_id[label_name]

    metadata = {
        "stay_id": float(record.stay_id),
        "subject_id": float(record.subject_id),
        "history_last_total": last_pre_total,
        "future_first_total": first_post_total,
        "future_max_total": max_post_total,
        "future_mean_total": mean_post_total,
        "target_label_id": float(label),
        "trajectory_label": label_name,
    }
    for key, value in record.extra_labels.items():
        metadata[key] = float(value)
    return row, names, label, metadata


def build_patient_forecast_dataset(
    json_path: str = "",
    dataset_format: str = "legacy_json",
    eicu_sepsis3_labels_csv: str = "",
    eicu_sepsis3_trajectory_csv: str = "",
    sofa_fields: Sequence[str] = SOFA_FIELDS,
    worsening_delta: float = 2.0,
    feature_mode: str = "full",
    target_mode: str = "binary_worsening",
) -> TemporalDataset:
    records = load_temporal_records(
        json_path=json_path,
        dataset_format=dataset_format,
        eicu_sepsis3_labels_csv=eicu_sepsis3_labels_csv,
        eicu_sepsis3_trajectory_csv=eicu_sepsis3_trajectory_csv,
    )
    x_data: List[List[float]] = []
    y_data: List[int] = []
    patient_ids: List[int] = []
    metadata: List[Dict[str, float]] = []
    feature_names: List[str] = []
    label_to_id: Dict[str, int] = {}
    sequence_data: List[List[List[float]]] = []
    static_data: List[List[float]] = []

    for record in records:
        if not record.sofa_scores or not record.sofa_scores_post_icu:
            continue
        sequence_matrix = _build_sequence_matrix(record.sofa_scores, sofa_fields)
        static_vector, _ = _build_static_features(record)
        row, names, label, meta = _patient_row_from_steps(
            record=record,
            pre_steps=record.sofa_scores,
            post_steps=record.sofa_scores_post_icu,
            sofa_fields=sofa_fields,
            worsening_delta=worsening_delta,
            feature_mode=feature_mode,
            target_mode=target_mode,
            label_to_id=label_to_id,
        )
        if not feature_names:
            feature_names = names
        x_data.append(row)
        y_data.append(label)
        patient_ids.append(record.stay_id)
        metadata.append(meta)
        sequence_data.append(sequence_matrix)
        static_data.append(static_vector)

    return TemporalDataset(
        x_data=x_data,
        y_data=y_data,
        feature_names=feature_names,
        label_names=[name for name, _ in sorted(label_to_id.items(), key=lambda item: item[1])],
        patient_ids=patient_ids,
        records=records,
        metadata=metadata,
        task_type="patient_forecast",
        target_mode=target_mode,
        sequence_data=sequence_data,
        static_data=static_data,
        sequence_feature_names=list(sofa_fields),
    )


def build_window_forecast_dataset(
    json_path: str = "",
    dataset_format: str = "legacy_json",
    eicu_sepsis3_labels_csv: str = "",
    eicu_sepsis3_trajectory_csv: str = "",
    sofa_fields: Sequence[str] = SOFA_FIELDS,
    history_window: int = 4,
    forecast_horizon: int = 3,
    worsening_delta: float = 2.0,
    feature_mode: str = "prototype",
    target_mode: str = "binary_worsening",
) -> TemporalDataset:
    records = load_temporal_records(
        json_path=json_path,
        dataset_format=dataset_format,
        eicu_sepsis3_labels_csv=eicu_sepsis3_labels_csv,
        eicu_sepsis3_trajectory_csv=eicu_sepsis3_trajectory_csv,
    )
    x_data: List[List[float]] = []
    y_data: List[int] = []
    patient_ids: List[int] = []
    metadata: List[Dict[str, float]] = []
    feature_names: List[str] = []
    label_to_id: Dict[str, int] = {}
    sequence_data: List[List[List[float]]] = []
    static_data: List[List[float]] = []

    for record in records:
        full_sequence = list(record.sofa_scores) + list(record.sofa_scores_post_icu)
        if len(full_sequence) < history_window + forecast_horizon:
            continue

        for end_idx in range(history_window, len(full_sequence) - forecast_horizon + 1):
            history_steps = full_sequence[end_idx - history_window : end_idx]
            future_steps = full_sequence[end_idx : end_idx + forecast_horizon]
            sequence_matrix = _build_sequence_matrix(history_steps, sofa_fields)
            static_vector, _ = _build_static_features(record)
            row, names, label, meta = _patient_row_from_steps(
                record=record,
                pre_steps=history_steps,
                post_steps=future_steps,
                sofa_fields=sofa_fields,
                worsening_delta=worsening_delta,
                feature_mode=feature_mode,
                target_mode=target_mode,
                label_to_id=label_to_id,
            )
            meta.update(
                {
                    "window_end_index": float(end_idx - 1),
                    "forecast_horizon": float(forecast_horizon),
                    "history_window": float(history_window),
                }
            )
            if not feature_names:
                feature_names = names
            x_data.append(row)
            y_data.append(label)
            patient_ids.append(record.stay_id)
            metadata.append(meta)
            sequence_data.append(sequence_matrix)
            static_data.append(static_vector)

    return TemporalDataset(
        x_data=x_data,
        y_data=y_data,
        feature_names=feature_names,
        label_names=[name for name, _ in sorted(label_to_id.items(), key=lambda item: item[1])],
        patient_ids=patient_ids,
        records=records,
        metadata=metadata,
        task_type="window_forecast",
        target_mode=target_mode,
        sequence_data=sequence_data,
        static_data=static_data,
        sequence_feature_names=list(sofa_fields),
    )


def temporal_cv_splits(num_items: int, folds: int, seed: int) -> List[Tuple[List[int], List[int], List[int]]]:
    if folds < 3:
        raise ValueError("fold count must be at least 3")
    indices = list(range(num_items))
    rng = random.Random(seed)
    rng.shuffle(indices)
    fold_bins: List[List[int]] = [[] for _ in range(folds)]
    for idx, item_idx in enumerate(indices):
        fold_bins[idx % folds].append(item_idx)

    splits: List[Tuple[List[int], List[int], List[int]]] = []
    for fold_idx in range(folds):
        test_idx = fold_bins[fold_idx]
        val_idx = fold_bins[(fold_idx + 1) % folds]
        train_idx: List[int] = []
        for inner_idx, bucket in enumerate(fold_bins):
            if inner_idx not in {fold_idx, (fold_idx + 1) % folds}:
                train_idx.extend(bucket)
        splits.append((train_idx, val_idx, test_idx))
    return splits


def grouped_temporal_cv_splits(
    patient_ids: Sequence[int],
    folds: int,
    seed: int,
) -> List[Tuple[List[int], List[int], List[int]]]:
    if folds < 3:
        raise ValueError("fold count must be at least 3")

    unique_ids = list(dict.fromkeys(patient_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_ids)
    fold_bins: List[List[int]] = [[] for _ in range(folds)]
    for idx, patient_id in enumerate(unique_ids):
        fold_bins[idx % folds].append(patient_id)

    index_by_patient: Dict[int, List[int]] = {}
    for idx, patient_id in enumerate(patient_ids):
        index_by_patient.setdefault(patient_id, []).append(idx)

    splits: List[Tuple[List[int], List[int], List[int]]] = []
    for fold_idx in range(folds):
        test_patients = set(fold_bins[fold_idx])
        val_patients = set(fold_bins[(fold_idx + 1) % folds])
        train_patients = set(unique_ids) - test_patients - val_patients

        train_idx = [idx for patient_id in train_patients for idx in index_by_patient[patient_id]]
        val_idx = [idx for patient_id in val_patients for idx in index_by_patient[patient_id]]
        test_idx = [idx for patient_id in test_patients for idx in index_by_patient[patient_id]]
        splits.append((train_idx, val_idx, test_idx))
    return splits


def gather_standardized_split(
    x_data: List[List[float]],
    y_data: List[int],
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
) -> Tuple[List[List[float]], List[int], List[List[float]], List[int], List[List[float]], List[int]]:
    x_train = [x_data[idx] for idx in train_idx]
    y_train = [y_data[idx] for idx in train_idx]
    x_val = [x_data[idx] for idx in val_idx]
    y_val = [y_data[idx] for idx in val_idx]
    x_test = [x_data[idx] for idx in test_idx]
    y_test = [y_data[idx] for idx in test_idx]
    x_train, x_val, x_test = standardize_by_train(x_train, x_val, x_test)
    return x_train, y_train, x_val, y_val, x_test, y_test
