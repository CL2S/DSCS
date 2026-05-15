import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch

from src.ts_formation import PATTERN_LABELS, TRAJECTORY_LABELS
from src.tsf_data import ForecastSample


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def forecast_sample_identity(sample: ForecastSample) -> str:
    identity = {
        "dataset_name": sample.metadata.get("dataset_name", ""),
        "series_name": sample.metadata.get("series_name", ""),
        "window_end_index": sample.metadata.get("window_end_index", 0.0),
        "history_length": sample.metadata.get("history_length", 0.0),
        "forecast_horizon": sample.metadata.get("forecast_horizon", 0.0),
        "seasonality": sample.metadata.get("seasonality", 0.0),
        "raw_context": [round(float(value), 8) for value in sample.raw_context],
    }
    digest = hashlib.sha1(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return f"exp_{digest[:16]}"


@dataclass
class SemanticPrototypeHit:
    prototype_id: str
    dataset_name: str
    experience_label: int
    pattern_label: int
    trajectory_label: int
    score: float
    support: int
    quality_score: float
    group_key: str
    future_curve: List[float]
    future_direction_type: str = "flat"
    outcome_entropy: float = 0.0


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(float(left_value) * float(right_value) for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(float(value) * float(value) for value in left))
    right_norm = math.sqrt(sum(float(value) * float(value) for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return numerator / (left_norm * right_norm)


def _mean_abs(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(abs(float(value)) for value in values) / max(1, len(values)))


DEFAULT_REUSABLE_SPLITS = {"train", "external"}


def _canonical_identifier(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _first_identifier(*values: object) -> str:
    for value in values:
        text = _canonical_identifier(value)
        if text:
            return text
    return ""


def _identifier_aliases(value: object) -> Set[str]:
    text = _canonical_identifier(value)
    if not text:
        return set()
    aliases = {text}
    lowered = text.lower()
    if lowered.startswith("stay_"):
        suffix = text[5:]
        if suffix:
            aliases.add(suffix)
    try:
        numeric = float(text)
        if numeric.is_integer():
            integer_text = str(int(numeric))
            aliases.add(integer_text)
            aliases.add(f"stay_{integer_text}")
    except (TypeError, ValueError):
        pass
    return {alias for alias in aliases if alias}


def _hash_identifier(value: object) -> str:
    text = _canonical_identifier(value)
    if not text:
        return ""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _normalize_identifier_sets(values: Optional[Sequence[object]]) -> Tuple[Set[str], Set[str]]:
    raw_values: Set[str] = set()
    hashed_values: Set[str] = set()
    for value in values or []:
        for text in _identifier_aliases(value):
            raw_values.add(text)
            hashed_values.add(text)
            hashed_values.add(_hash_identifier(text))
    return raw_values, hashed_values


def _infer_split(source: object) -> str:
    text = str(source or "").lower()
    if "test" in text:
        return "test"
    if "val" in text or "valid" in text:
        return "val"
    if "external" in text:
        return "external"
    if "train" in text or "fit" in text:
        return "train"
    return "unknown"


def _split_counts(rows: Sequence[Dict[str, object]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        split = str(row.get("split") or _infer_split(row.get("source", ""))).lower()
        counts[split] = counts.get(split, 0) + 1
    return counts


class PersistentExperienceStore:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.manifest_path = self.root_dir / "manifest.json"
        self.entries_path = self.root_dir / "experience_entries.jsonl"
        self.prototypes_path = self.root_dir / "prototypes.jsonl"
        self.events_path = self.root_dir / "events.jsonl"
        self.index_dir = self.root_dir / "indexes"
        self.cache_dir = self.root_dir / "cache"
        self.semantic_index_path = self.index_dir / "semantic_index.json"
        self.ensure_store()

    def ensure_store(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            _safe_json_dump(
                self.manifest_path,
                {
                    "schema_version": 1,
                    "formation_version": 1,
                    "store_name": "forecasting_persistent_experience_memory",
                    "created_at": _utc_now(),
                    "updated_at": _utc_now(),
                    "entry_count": 0,
                    "prototype_count": 0,
                    "datasets": {},
                    "available_model_caches": [],
                },
            )
        for path in [self.entries_path, self.prototypes_path, self.events_path]:
            if not path.exists():
                path.write_text("", encoding="utf-8")
        if not self.semantic_index_path.exists():
            _safe_json_dump(
                self.semantic_index_path,
                {"by_dataset": {}, "by_experience_label": {}, "updated_at": _utc_now()},
            )

    def _load_jsonl(self, path: Path) -> List[Dict[str, object]]:
        if not path.exists():
            return []
        rows: List[Dict[str, object]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def load_manifest(self) -> Dict[str, object]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _update_manifest(
        self,
        entries: Optional[List[Dict[str, object]]] = None,
        prototypes: Optional[List[Dict[str, object]]] = None,
    ) -> Dict[str, object]:
        manifest = self.load_manifest()
        entry_rows = entries if entries is not None else self._load_jsonl(self.entries_path)
        prototype_rows = prototypes if prototypes is not None else self._load_jsonl(self.prototypes_path)
        datasets: Dict[str, int] = {}
        for row in entry_rows:
            dataset_name = str(row.get("dataset_name", "unknown"))
            datasets[dataset_name] = datasets.get(dataset_name, 0) + 1
        manifest.update(
            {
                "updated_at": _utc_now(),
                "entry_count": len(entry_rows),
                "prototype_count": len(prototype_rows),
                "datasets": datasets,
                "available_model_caches": sorted(path.stem for path in self.cache_dir.glob("*.pt")),
            }
        )
        _safe_json_dump(self.manifest_path, manifest)
        return manifest

    def _experience_id(self, sample: ForecastSample) -> str:
        return forecast_sample_identity(sample)

    def _direction_type(self, values: Sequence[float]) -> str:
        if len(values) <= 1:
            return "flat"
        delta = values[-1] - values[0]
        peak_index = max(range(len(values)), key=lambda index: values[index])
        trough_index = min(range(len(values)), key=lambda index: values[index])
        if peak_index > trough_index and delta > 0:
            return "rise_drop_rise"
        if peak_index < trough_index and delta < 0:
            return "drop_rise_drop"
        if delta > 0:
            return "mostly_up"
        if delta < 0:
            return "mostly_down"
        return "flat"

    def _severity_bucket(self, sample: ForecastSample) -> str:
        sofa_value = sample.metadata.get("pre_baseline_total_sofa")
        try:
            severity_value = float(sofa_value)
        except (TypeError, ValueError):
            severity_value = float(sum(sample.raw_context) / max(1, len(sample.raw_context))) if sample.raw_context else 0.0
        if severity_value < 4.0:
            return "low"
        if severity_value < 8.0:
            return "moderate"
        if severity_value < 12.0:
            return "high"
        return "critical"

    def _kg_state_signature(self, kg_flags: Dict[str, object], kg_features: Sequence[float]) -> str:
        if not kg_flags and not kg_features:
            return "none"
        shock = 1 if any(float(kg_flags.get(name, 0.0)) > 0.5 for name in ["state_septic_shock", "treat_vasopressor"]) else 0
        infection = 1 if any(
            float(kg_flags.get(name, 0.0)) > 0.5
            for name in ["state_sepsis", "state_suspected_infection", "exam_blood_culture", "treat_antibiotic"]
        ) else 0
        respiratory = 1 if any(
            float(kg_flags.get(name, 0.0)) > 0.5
            for name in ["state_hypoxemia", "treat_respiratory_support"]
        ) else 0
        lactate = 1 if any(float(kg_flags.get(name, 0.0)) > 0.5 for name in ["exam_lactate", "monitor_lactate_repeat"]) else 0
        dense = 1 if sum(1 for value in kg_features if abs(float(value)) > 1e-5) >= max(2, len(kg_features) // 4) else 0
        return f"i{infection}s{shock}r{respiratory}l{lactate}d{dense}"

    def _intervention_signature(
        self,
        intervention_static: Sequence[float],
        intervention_sequence: Sequence[Sequence[float]],
    ) -> str:
        static_values = [float(value) for value in intervention_static]
        flattened_sequence = [float(value) for step in intervention_sequence for value in step]
        combined = static_values + flattened_sequence
        if not combined:
            return "none"
        density = float(sum(1 for value in combined if abs(value) > 1e-5) / max(1, len(combined)))
        magnitude = _mean_abs(combined)
        first_step_mean = _mean_abs(intervention_sequence[0]) if intervention_sequence else _mean_abs(static_values)
        last_step_mean = _mean_abs(intervention_sequence[-1]) if intervention_sequence else _mean_abs(static_values)
        delta = last_step_mean - first_step_mean
        if density < 0.05 and magnitude < 1e-3:
            intensity = "none"
        elif density < 0.18:
            intensity = "light"
        elif density < 0.42:
            intensity = "moderate"
        else:
            intensity = "intensive"
        if delta > 0.05:
            trend = "escalating"
        elif delta < -0.05:
            trend = "deescalating"
        else:
            trend = "stable"
        return f"{intensity}_{trend}"

    def _entry_from_sample(self, sample: ForecastSample, source: str) -> Dict[str, object]:
        experience_id = self._experience_id(sample)
        metadata = dict(sample.metadata or {})
        split = str(metadata.get("split") or metadata.get("source_split") or _infer_split(source)).lower()
        if split == "validation":
            split = "val"
        source_stay_id = _first_identifier(
            metadata.get("stay_id"),
            metadata.get("patientunitstayid"),
            metadata.get("source_stay_id"),
        )
        source_patient_id = _first_identifier(
            metadata.get("patient_id"),
            metadata.get("uniquepid"),
            metadata.get("patienthealthsystemstayid"),
            metadata.get("source_patient_id"),
        )
        is_allowed_for_reuse = bool(metadata.get("is_allowed_for_reuse", split in DEFAULT_REUSABLE_SPLITS))
        raw_future = [float(value) for value in sample.raw_target]
        kg_flags = {
            str(key): float(value)
            for key, value in dict(metadata.get("kg_flags", {})).items()
        }
        intervention_static = [float(value) for value in (sample.intervention_static or [])]
        intervention_sequence = [
            [float(value) for value in step]
            for step in (sample.intervention_sequence or [])
        ]
        kg_features = [float(value) for value in (sample.kg_features or [])]
        severity_bucket = self._severity_bucket(sample)
        kg_state_signature = self._kg_state_signature(kg_flags, kg_features)
        intervention_signature = self._intervention_signature(intervention_static, intervention_sequence)
        return {
            "experience_id": experience_id,
            "schema_version": 1,
            "formation_version": 1,
            "dataset_name": str(metadata.get("dataset_name", "")),
            "series_name": str(metadata.get("series_name", "")),
            "window_end_index": float(metadata.get("window_end_index", 0.0)),
            "seasonality": int(float(metadata.get("seasonality", 1.0))),
            "history_length": int(float(metadata.get("history_length", len(sample.raw_context)))),
            "forecast_horizon": int(float(metadata.get("forecast_horizon", len(raw_future)))),
            "raw_context": [float(value) for value in sample.raw_context],
            "raw_future": raw_future,
            "scale_center": float(sample.scale_center),
            "scale_value": float(sample.scale_value),
            "normalized_context": [float(step[0]) for step in sample.sequence],
            "normalized_future": [float(value) for value in sample.target],
            "static_features": [float(value) for value in sample.static],
            "patient_static_features": [float(value) for value in (sample.patient_static or sample.static)],
            "intervention_static_features": intervention_static,
            "intervention_sequence": intervention_sequence,
            "kg_features": kg_features,
            "kg_flags": kg_flags,
            "severity_bucket": severity_bucket,
            "kg_state_signature": kg_state_signature,
            "intervention_signature": intervention_signature,
            "kg_appended_to_patient_static": float(metadata.get("kg_appended_to_patient_static", 0.0)),
            "kg_guideline_alignment": float(metadata.get("kg_guideline_alignment", 0.0)),
            "formation_features": [float(value) for value in sample.formation_features],
            "pattern_label": int(sample.pattern_label),
            "trajectory_label": int(sample.trajectory_label),
            "experience_label": int(sample.experience_label),
            "pattern_name": PATTERN_LABELS[sample.pattern_label],
            "trajectory_name": TRAJECTORY_LABELS[sample.trajectory_label],
            "shape_signature": {
                "local_slope": float(sample.formation_features[0]),
                "medium_slope": float(sample.formation_features[1]),
                "seasonal_strength": float(sample.formation_features[5]),
                "change_proxy": float(sample.formation_features[11]),
                "curvature": float(sample.formation_features[12]),
                "autocorr_lag1": float(sample.formation_features[13]),
            },
            "target_signature": {
                "future_mean": float(sum(raw_future) / max(1, len(raw_future))),
                "future_peak_step": int(max(range(len(raw_future)), key=lambda index: raw_future[index])) if raw_future else 0,
                "future_peak_value": float(max(raw_future)) if raw_future else 0.0,
                "future_trough_step": int(min(range(len(raw_future)), key=lambda index: raw_future[index])) if raw_future else 0,
                "future_trough_value": float(min(raw_future)) if raw_future else 0.0,
                "future_direction_type": self._direction_type(raw_future),
            },
            "support": 1,
            "quality_score": float(sample.formation_features[14]),
            "source": source,
            "split": split,
            "source_dataset": str(metadata.get("dataset_name", "")),
            "source_run_id": str(metadata.get("source_run_id", "")),
            "source_hospital_id": _first_identifier(metadata.get("hospitalid"), metadata.get("hospital_id")),
            "source_unit_type": _first_identifier(metadata.get("unittype"), metadata.get("unit_type")),
            "source_patient_id_hash": _hash_identifier(source_patient_id),
            "source_stay_id": source_stay_id,
            "source_stay_id_hash": _hash_identifier(source_stay_id),
            "is_allowed_for_reuse": bool(is_allowed_for_reuse and split in DEFAULT_REUSABLE_SPLITS),
            "created_at": _utc_now(),
        }

    def _build_semantic_index(self, entries: Sequence[Dict[str, object]]) -> None:
        by_dataset: Dict[str, List[str]] = {}
        by_experience_label: Dict[str, List[str]] = {}
        for entry in entries:
            exp_id = str(entry["experience_id"])
            dataset_name = str(entry.get("dataset_name", "unknown"))
            label_key = str(int(entry.get("experience_label", -1)))
            by_dataset.setdefault(dataset_name, []).append(exp_id)
            by_experience_label.setdefault(label_key, []).append(exp_id)
        _safe_json_dump(
            self.semantic_index_path,
            {
                "by_dataset": by_dataset,
                "by_experience_label": by_experience_label,
                "updated_at": _utc_now(),
            },
        )

    def _prototype_group_key(self, entry: Dict[str, object]) -> str:
        return "|".join(
            [
                str(entry.get("dataset_name", "")),
                str(entry.get("pattern_label", -1)),
                str(entry.get("trajectory_label", -1)),
                str(entry.get("severity_bucket", "unknown")),
                str(entry.get("kg_state_signature", "none")),
                str(entry.get("intervention_signature", "none")),
                str(entry.get("target_signature", {}).get("future_direction_type", "flat")),
                str(entry.get("forecast_horizon", -1)),
                str(entry.get("seasonality", -1)),
            ]
        )

    def _prototype_from_members(self, group_key: str, members: Sequence[Dict[str, object]]) -> Dict[str, object]:
        support = len(members)
        member_count_by_split = _split_counts(members)
        source_splits = sorted(member_count_by_split)
        reusable_members = [
            bool(member.get("is_allowed_for_reuse", str(member.get("split") or _infer_split(member.get("source", ""))).lower() in DEFAULT_REUSABLE_SPLITS))
            and str(member.get("split") or _infer_split(member.get("source", ""))).lower() in DEFAULT_REUSABLE_SPLITS
            for member in members
        ]
        formation_dim = len(members[0]["formation_features"])
        future_dim = len(members[0]["normalized_future"])
        kg_dim = len(members[0].get("kg_features", []))
        mean_formation = [
            float(sum(float(member["formation_features"][index]) for member in members) / support)
            for index in range(formation_dim)
        ]
        mean_future = [
            float(sum(float(member["normalized_future"][index]) for member in members) / support)
            for index in range(future_dim)
        ]
        mean_kg = [
            float(sum(float(member.get("kg_features", [0.0] * kg_dim)[index]) for member in members) / support)
            for index in range(kg_dim)
        ] if kg_dim > 0 else []
        intervention_density_mean = float(
            sum(
                1.0
                if str(member.get("intervention_signature", "none")).startswith(("moderate", "intensive"))
                else 0.0
                for member in members
            ) / support
        )
        future_direction_type = str(members[0].get("target_signature", {}).get("future_direction_type", "flat"))
        return {
            "prototype_id": f"proto_{hashlib.sha1(group_key.encode('utf-8')).hexdigest()[:16]}",
            "group_key": group_key,
            "member_experience_ids": [str(member["experience_id"]) for member in members[:32]],
            "source_splits": source_splits,
            "member_count_by_split": member_count_by_split,
            "is_allowed_for_reuse": all(reusable_members),
            "dataset_name": str(members[0]["dataset_name"]),
            "seasonality": int(members[0].get("seasonality", 1)),
            "forecast_horizon": int(members[0].get("forecast_horizon", future_dim)),
            "pattern_label": int(members[0]["pattern_label"]),
            "trajectory_label": int(members[0]["trajectory_label"]),
            "experience_label": int(members[0]["experience_label"]),
            "pattern_name": str(members[0]["pattern_name"]),
            "trajectory_name": str(members[0]["trajectory_name"]),
            "severity_bucket": str(members[0].get("severity_bucket", "unknown")),
            "kg_state_signature": str(members[0].get("kg_state_signature", "none")),
            "intervention_signature": str(members[0].get("intervention_signature", "none")),
            "future_direction_type": future_direction_type,
            "support": support,
            "prototype_formation_center": mean_formation,
            "prototype_future_mean_curve": mean_future,
            "prototype_kg_center": mean_kg,
            "intervention_density_mean": intervention_density_mean,
            "quality_score": float(sum(float(member["quality_score"]) for member in members) / support),
            "last_updated_at": _utc_now(),
        }

    def rebuild_prototypes(self) -> int:
        entries = self._load_jsonl(self.entries_path)
        grouped: Dict[str, List[Dict[str, object]]] = {}
        for entry in entries:
            grouped.setdefault(self._prototype_group_key(entry), []).append(entry)

        prototype_rows = [
            self._prototype_from_members(group_key, members)
            for group_key, members in grouped.items()
        ]

        self.prototypes_path.write_text("", encoding="utf-8")
        _append_jsonl(self.prototypes_path, prototype_rows)
        self._build_semantic_index(entries)
        self._update_manifest(entries=entries, prototypes=prototype_rows)
        _append_jsonl(
            self.events_path,
            [
                {
                    "event_id": f"evt_{hashlib.sha1((_utc_now() + 'prototype_rebuild').encode('utf-8')).hexdigest()[:16]}",
                    "event_type": "prototype_rebuild",
                    "prototype_count": len(prototype_rows),
                    "timestamp": _utc_now(),
                }
            ],
        )
        return len(prototype_rows)

    def _incremental_update_prototypes(
        self,
        new_rows: Sequence[Dict[str, object]],
        entries: Sequence[Dict[str, object]],
    ) -> int:
        prototype_rows = self._load_jsonl(self.prototypes_path)
        if entries and not prototype_rows and len(new_rows) < len(entries):
            return self.rebuild_prototypes()

        grouped_new: Dict[str, List[Dict[str, object]]] = {}
        for row in new_rows:
            grouped_new.setdefault(self._prototype_group_key(row), []).append(row)

        prototype_by_key = {str(row.get("group_key", "")): row for row in prototype_rows}
        for group_key, members in grouped_new.items():
            existing = prototype_by_key.get(group_key)
            if existing is None:
                prototype_rows.append(self._prototype_from_members(group_key, members))
                prototype_by_key[group_key] = prototype_rows[-1]
                continue

            old_support = max(0, int(existing.get("support", 0)))
            add_support = len(members)
            new_support = max(1, old_support + add_support)

            def weighted_vector(old_key: str, member_key: str) -> List[float]:
                old_values = [float(value) for value in existing.get(old_key, [])]
                if not old_values:
                    return [
                        float(sum(float(member.get(member_key, [])[index]) for member in members) / add_support)
                        for index in range(len(members[0].get(member_key, [])))
                    ]
                return [
                    float(
                        (
                            old_support * float(old_values[index])
                            + sum(float(member.get(member_key, [0.0] * len(old_values))[index]) for member in members)
                        )
                        / new_support
                    )
                    for index in range(len(old_values))
                ]

            existing["prototype_formation_center"] = weighted_vector("prototype_formation_center", "formation_features")
            existing["prototype_future_mean_curve"] = weighted_vector("prototype_future_mean_curve", "normalized_future")
            if existing.get("prototype_kg_center") or members[0].get("kg_features", []):
                existing["prototype_kg_center"] = weighted_vector("prototype_kg_center", "kg_features")

            add_intervention_density = float(
                sum(
                    1.0 if str(member.get("intervention_signature", "none")).startswith(("moderate", "intensive")) else 0.0
                    for member in members
                )
                / add_support
            )
            existing["intervention_density_mean"] = float(
                (
                    old_support * float(existing.get("intervention_density_mean", 0.0))
                    + add_support * add_intervention_density
                )
                / new_support
            )
            existing["quality_score"] = float(
                (
                    old_support * float(existing.get("quality_score", 0.0))
                    + sum(float(member.get("quality_score", 0.0)) for member in members)
                )
                / new_support
            )
            existing["support"] = new_support
            member_ids = list(existing.get("member_experience_ids", []))
            member_ids.extend(str(member["experience_id"]) for member in members)
            existing["member_experience_ids"] = member_ids[:32]
            split_counts = dict(existing.get("member_count_by_split", {}))
            for split, count in _split_counts(members).items():
                split_counts[split] = int(split_counts.get(split, 0)) + int(count)
            existing["member_count_by_split"] = split_counts
            existing["source_splits"] = sorted(split_counts)
            existing["is_allowed_for_reuse"] = bool(existing.get("is_allowed_for_reuse", True)) and all(
                bool(member.get("is_allowed_for_reuse", False)) for member in members
            )
            existing["last_updated_at"] = _utc_now()

        self.prototypes_path.write_text("", encoding="utf-8")
        _append_jsonl(self.prototypes_path, prototype_rows)
        self._build_semantic_index(entries)
        self._update_manifest(entries=list(entries), prototypes=prototype_rows)
        return len(prototype_rows)

    def upsert_samples(
        self,
        samples: Sequence[ForecastSample],
        source: str = "train",
        prototype_update_mode: str = "incremental",
        progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
        progress_interval: int = 1000,
    ) -> Dict[str, int]:
        existing_entries = self._load_jsonl(self.entries_path)
        existing_ids = {str(entry["experience_id"]) for entry in existing_entries}
        new_rows: List[Dict[str, object]] = []
        skipped = 0
        total_samples = len(samples)
        progress_every = max(1, int(progress_interval))
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "scan_start",
                    "source": source,
                    "total": total_samples,
                    "existing_entry_count": len(existing_entries),
                }
            )
        for index, sample in enumerate(samples, start=1):
            row = self._entry_from_sample(sample, source=source)
            if row["experience_id"] in existing_ids:
                skipped += 1
                if progress_callback is not None and (index == total_samples or index % progress_every == 0):
                    progress_callback(
                        {
                            "stage": "scan",
                            "source": source,
                            "processed": index,
                            "total": total_samples,
                            "inserted": len(new_rows),
                            "skipped": skipped,
                        }
                    )
                continue
            existing_ids.add(str(row["experience_id"]))
            new_rows.append(row)
            if progress_callback is not None and (index == total_samples or index % progress_every == 0):
                progress_callback(
                    {
                        "stage": "scan",
                        "source": source,
                        "processed": index,
                        "total": total_samples,
                        "inserted": len(new_rows),
                        "skipped": skipped,
                    }
                )

        if new_rows:
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "append_start",
                        "source": source,
                        "inserted": len(new_rows),
                        "path": str(self.entries_path),
                    }
                )
            _append_jsonl(self.entries_path, new_rows)
            if progress_callback is not None:
                progress_callback({"stage": "append_done", "source": source, "inserted": len(new_rows)})

        entries = existing_entries + new_rows
        prototype_rebuilt = False
        if new_rows:
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "prototype_start",
                        "source": source,
                        "mode": prototype_update_mode,
                        "new_rows": len(new_rows),
                    }
                )
            if prototype_update_mode == "rebuild":
                prototype_count = self.rebuild_prototypes()
                prototype_rebuilt = True
            else:
                prototype_count = self._incremental_update_prototypes(new_rows, entries)
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "prototype_done",
                        "source": source,
                        "mode": prototype_update_mode,
                        "prototype_count": prototype_count,
                        "prototype_rebuilt": prototype_rebuilt,
                    }
                )
        else:
            prototype_count = len(self._load_jsonl(self.prototypes_path))
        manifest = self._update_manifest(entries=entries)
        _append_jsonl(
            self.events_path,
            [
                {
                    "event_id": f"evt_{hashlib.sha1((_utc_now() + 'upsert').encode('utf-8')).hexdigest()[:16]}",
                    "event_type": "upsert_samples",
                    "inserted_count": len(new_rows),
                    "skipped_count": skipped,
                    "source": source,
                    "prototype_update_mode": prototype_update_mode,
                    "prototype_rebuilt": prototype_rebuilt,
                    "timestamp": _utc_now(),
                }
            ],
        )
        summary = {
            "inserted_count": len(new_rows),
            "skipped_count": skipped,
            "entry_count": int(manifest["entry_count"]),
            "prototype_count": prototype_count,
            "prototype_update_mode": prototype_update_mode,
            "prototype_rebuilt": prototype_rebuilt,
        }
        if progress_callback is not None:
            progress_callback({"stage": "done", "source": source, **summary})
        return summary

    def load_samples(
        self,
        dataset_name: Optional[str] = None,
        seasonality: Optional[int] = None,
        forecast_horizon: Optional[int] = None,
        allowed_splits: Optional[Sequence[str]] = None,
        exclude_stay_ids: Optional[Sequence[object]] = None,
        exclude_patient_ids: Optional[Sequence[object]] = None,
        strict_no_test: bool = True,
        return_audit: bool = False,
    ):
        entries = self._load_jsonl(self.entries_path)
        loaded: List[ForecastSample] = []
        allowed = {str(split).lower() for split in (allowed_splits or sorted(DEFAULT_REUSABLE_SPLITS))}
        exclude_stay_raw, exclude_stay_hash = _normalize_identifier_sets(exclude_stay_ids)
        exclude_patient_raw, exclude_patient_hash = _normalize_identifier_sets(exclude_patient_ids)
        audit: Dict[str, object] = {
            "scanned_total": len(entries),
            "matched_scope": 0,
            "loaded_total": 0,
            "loaded_by_split": {},
            "allowed_splits": sorted(allowed),
            "strict_no_test": bool(strict_no_test),
            "exclude_stay_count": len(exclude_stay_raw),
            "exclude_patient_count": len(exclude_patient_raw),
            "excluded_scope": 0,
            "excluded_test_split": 0,
            "excluded_disallowed_split": 0,
            "excluded_not_allowed_for_reuse": 0,
            "excluded_same_stay": 0,
            "excluded_same_patient": 0,
        }
        for entry in entries:
            if dataset_name is not None and str(entry.get("dataset_name", "")) != dataset_name:
                audit["excluded_scope"] = int(audit["excluded_scope"]) + 1
                continue
            if seasonality is not None and int(entry.get("seasonality", -1)) != int(seasonality):
                audit["excluded_scope"] = int(audit["excluded_scope"]) + 1
                continue
            if forecast_horizon is not None and int(entry.get("forecast_horizon", -1)) != int(forecast_horizon):
                audit["excluded_scope"] = int(audit["excluded_scope"]) + 1
                continue
            audit["matched_scope"] = int(audit["matched_scope"]) + 1
            split = str(entry.get("split") or _infer_split(entry.get("source", ""))).lower()
            if strict_no_test and split == "test":
                audit["excluded_test_split"] = int(audit["excluded_test_split"]) + 1
                continue
            if allowed and split not in allowed:
                audit["excluded_disallowed_split"] = int(audit["excluded_disallowed_split"]) + 1
                continue
            is_allowed = bool(entry.get("is_allowed_for_reuse", split in DEFAULT_REUSABLE_SPLITS))
            if not is_allowed:
                audit["excluded_not_allowed_for_reuse"] = int(audit["excluded_not_allowed_for_reuse"]) + 1
                continue
            entry_stay_id = _first_identifier(entry.get("source_stay_id"), entry.get("series_name"))
            entry_stay_hash = _canonical_identifier(entry.get("source_stay_id_hash") or _hash_identifier(entry_stay_id))
            entry_stay_aliases = _identifier_aliases(entry_stay_id)
            entry_stay_hashes = {_hash_identifier(alias) for alias in entry_stay_aliases}
            if entry_stay_hash:
                entry_stay_hashes.add(entry_stay_hash)
            if entry_stay_aliases.intersection(exclude_stay_raw) or entry_stay_hashes.intersection(exclude_stay_hash):
                audit["excluded_same_stay"] = int(audit["excluded_same_stay"]) + 1
                continue
            entry_patient_hash = _canonical_identifier(entry.get("source_patient_id_hash"))
            if entry_patient_hash and entry_patient_hash in exclude_patient_hash:
                audit["excluded_same_patient"] = int(audit["excluded_same_patient"]) + 1
                continue
            metadata = {
                "stay_id": float(len(loaded) + 1000000),
                "series_index": float(len(loaded) + 1000000),
                "series_name": str(entry.get("series_name", "persistent_series")),
                "window_end_index": float(entry.get("window_end_index", 0.0)),
                "dataset_name": str(entry.get("dataset_name", "")),
                "series_count": 0.0,
                "seasonality": float(entry.get("seasonality", 1)),
                "history_length": float(entry.get("history_length", len(entry.get("raw_context", [])))),
                "forecast_horizon": float(entry.get("forecast_horizon", len(entry.get("raw_future", [])))),
                "experience_id": str(entry.get("experience_id", "")),
                "persistent_source": True,
                "persistent_split": split,
                "persistent_source_name": str(entry.get("source", "")),
                "source_stay_id": entry_stay_id,
                "source_stay_id_hash": entry_stay_hash,
                "source_patient_id_hash": entry_patient_hash,
                "kg_appended_to_patient_static": float(entry.get("kg_appended_to_patient_static", 0.0)),
                "kg_guideline_alignment": float(entry.get("kg_guideline_alignment", 0.0)),
            }
            kg_flags = {
                str(key): float(value)
                for key, value in dict(entry.get("kg_flags", {})).items()
            }
            if kg_flags:
                metadata["kg_flags"] = kg_flags
            patient_static = [float(value) for value in entry.get("patient_static_features", [])]
            intervention_static = [float(value) for value in entry.get("intervention_static_features", [])]
            intervention_sequence = [
                [float(value) for value in step]
                for step in entry.get("intervention_sequence", [])
            ]
            kg_features = [float(value) for value in entry.get("kg_features", [])]
            if not patient_static:
                patient_static = [float(value) for value in entry.get("static_features", [])]
            static_features = patient_static + intervention_static if (patient_static or intervention_static) else [
                float(value) for value in entry.get("static_features", [])
            ]
            loaded.append(
                ForecastSample(
                    sequence=[[float(value)] for value in entry.get("normalized_context", [])],
                    static=static_features,
                    target=[float(value) for value in entry.get("normalized_future", [])],
                    metadata=metadata,
                    scale_center=float(entry.get("scale_center", 0.0)),
                    scale_value=float(entry.get("scale_value", 1.0)),
                    raw_context=[float(value) for value in entry.get("raw_context", [])],
                    raw_target=[float(value) for value in entry.get("raw_future", [])],
                    formation_features=[float(value) for value in entry.get("formation_features", [])],
                    pattern_label=int(entry.get("pattern_label", 0)),
                    trajectory_label=int(entry.get("trajectory_label", 0)),
                    experience_label=int(entry.get("experience_label", 0)),
                    patient_static=patient_static,
                    intervention_static=intervention_static,
                    intervention_sequence=intervention_sequence,
                    kg_features=kg_features,
                )
            )
            loaded_by_split = dict(audit["loaded_by_split"])
            loaded_by_split[split] = int(loaded_by_split.get(split, 0)) + 1
            audit["loaded_by_split"] = loaded_by_split
        audit["loaded_total"] = len(loaded)
        if return_audit:
            return loaded, audit
        return loaded

    def summary(self) -> Dict[str, object]:
        manifest = self.load_manifest()
        return {
            "store_path": str(self.root_dir),
            "entry_count": int(manifest.get("entry_count", 0)),
            "prototype_count": int(manifest.get("prototype_count", 0)),
            "datasets": manifest.get("datasets", {}),
        }

    def _model_cache_paths(self, model_fingerprint: str) -> tuple[Path, Path]:
        return (
            self.cache_dir / f"{model_fingerprint}.pt",
            self.cache_dir / f"{model_fingerprint}.meta.json",
        )

    def load_model_cache(self, model_fingerprint: str) -> Optional[Dict[str, object]]:
        tensor_path, _ = self._model_cache_paths(model_fingerprint)
        if not tensor_path.exists():
            return None
        payload = torch.load(tensor_path, map_location="cpu")
        if not isinstance(payload, dict):
            return None
        return payload

    def save_model_cache(self, model_fingerprint: str, payload: Dict[str, object]) -> Dict[str, object]:
        tensor_path, meta_path = self._model_cache_paths(model_fingerprint)
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, tensor_path)
        banks = payload.get("banks", {})
        meta = {
            "model_fingerprint": model_fingerprint,
            "schema_version": int(payload.get("schema_version", 1)),
            "created_at": payload.get("created_at", _utc_now()),
            "dataset_name": payload.get("dataset_name", ""),
            "seasonality": payload.get("seasonality", 1),
            "forecast_horizon": payload.get("forecast_horizon", 0),
            "pattern_memory_size": len(banks.get("pattern", [])),
            "trajectory_memory_size": len(banks.get("trajectory", [])),
            "experience_memory_size": len(banks.get("experience_hot", [])),
            "experience_archive_size": len(banks.get("experience_archive", [])),
            "store_path": str(tensor_path),
        }
        _safe_json_dump(meta_path, meta)
        self._update_manifest()
        return meta

    def load_prototypes(
        self,
        dataset_name: Optional[str] = None,
        seasonality: Optional[int] = None,
        forecast_horizon: Optional[int] = None,
        allowed_splits: Optional[Sequence[str]] = None,
        strict_no_test: bool = True,
        return_audit: bool = False,
    ):
        prototypes = self._load_jsonl(self.prototypes_path)
        filtered: List[Dict[str, object]] = []
        allowed = {str(split).lower() for split in (allowed_splits or sorted(DEFAULT_REUSABLE_SPLITS))}
        audit: Dict[str, object] = {
            "scanned_total": len(prototypes),
            "matched_scope": 0,
            "loaded_total": 0,
            "allowed_splits": sorted(allowed),
            "strict_no_test": bool(strict_no_test),
            "excluded_scope": 0,
            "excluded_test_split": 0,
            "excluded_disallowed_split": 0,
            "excluded_not_allowed_for_reuse": 0,
        }
        for prototype in prototypes:
            if dataset_name is not None and str(prototype.get("dataset_name", "")) != dataset_name:
                audit["excluded_scope"] = int(audit["excluded_scope"]) + 1
                continue
            if seasonality is not None:
                try:
                    prototype_seasonality = int(prototype.get("seasonality", -1))
                except Exception:
                    prototype_seasonality = None
                if prototype_seasonality != int(seasonality):
                    audit["excluded_scope"] = int(audit["excluded_scope"]) + 1
                    continue
            if forecast_horizon is not None:
                try:
                    prototype_horizon = int(prototype.get("forecast_horizon", -1))
                except Exception:
                    prototype_horizon = None
                if prototype_horizon != int(forecast_horizon):
                    audit["excluded_scope"] = int(audit["excluded_scope"]) + 1
                    continue
            audit["matched_scope"] = int(audit["matched_scope"]) + 1
            source_splits = [
                str(split).lower()
                for split in prototype.get("source_splits", [])
            ] or ["unknown"]
            if strict_no_test and "test" in source_splits:
                audit["excluded_test_split"] = int(audit["excluded_test_split"]) + 1
                continue
            if allowed and not set(source_splits).issubset(allowed):
                audit["excluded_disallowed_split"] = int(audit["excluded_disallowed_split"]) + 1
                continue
            if not bool(prototype.get("is_allowed_for_reuse", not bool(strict_no_test))):
                audit["excluded_not_allowed_for_reuse"] = int(audit["excluded_not_allowed_for_reuse"]) + 1
                continue
            filtered.append(prototype)
        audit["loaded_total"] = len(filtered)
        if return_audit:
            return filtered, audit
        return filtered

    def semantic_retrieve(
        self,
        formation_features: Sequence[float],
        pattern_label: int,
        trajectory_label: int,
        experience_label: int,
        kg_features: Optional[Sequence[float]] = None,
        kg_flags: Optional[Dict[str, object]] = None,
        intervention_static: Optional[Sequence[float]] = None,
        intervention_sequence: Optional[Sequence[Sequence[float]]] = None,
        future_direction_type: str = "flat",
        severity_bucket: str = "unknown",
        dataset_name: Optional[str] = None,
        seasonality: Optional[int] = None,
        forecast_horizon: Optional[int] = None,
        top_k: int = 3,
    ) -> List[SemanticPrototypeHit]:
        candidates = self.load_prototypes(
            dataset_name=dataset_name,
            seasonality=seasonality,
            forecast_horizon=forecast_horizon,
        )
        hits: List[SemanticPrototypeHit] = []
        query_kg_features = [float(value) for value in (kg_features or [])]
        query_kg_signature = self._kg_state_signature(
            {
                str(key): float(value)
                for key, value in dict(kg_flags or {}).items()
            },
            query_kg_features,
        )
        query_intervention_signature = self._intervention_signature(
            [float(value) for value in (intervention_static or [])],
            [[float(value) for value in step] for step in (intervention_sequence or [])],
        )
        for candidate in candidates:
            prototype_center = [float(value) for value in candidate.get("prototype_formation_center", [])]
            formation_similarity = max(0.0, _cosine_similarity(formation_features, prototype_center))
            prototype_kg_center = [float(value) for value in candidate.get("prototype_kg_center", [])]
            kg_similarity = max(0.0, _cosine_similarity(query_kg_features, prototype_kg_center)) if query_kg_features and prototype_kg_center else 0.0
            pattern_match = 1.0 if int(candidate.get("pattern_label", -1)) == int(pattern_label) else 0.0
            trajectory_match = 1.0 if int(candidate.get("trajectory_label", -1)) == int(trajectory_label) else 0.0
            experience_match = 1.0 if int(candidate.get("experience_label", -1)) == int(experience_label) else 0.0
            severity_match = 1.0 if str(candidate.get("severity_bucket", "unknown")) == str(severity_bucket) else 0.0
            kg_signature_match = 1.0 if str(candidate.get("kg_state_signature", "none")) == query_kg_signature else 0.0
            intervention_match = 1.0 if str(candidate.get("intervention_signature", "none")) == query_intervention_signature else 0.0
            direction_match = 1.0 if str(candidate.get("future_direction_type", "flat")) == str(future_direction_type) else 0.0
            support = max(1, int(candidate.get("support", 1)))
            quality_score = float(candidate.get("quality_score", 0.0))
            support_bonus = min(1.0, math.log1p(float(support)) / math.log(33.0))
            score = (
                0.34 * formation_similarity
                + 0.12 * kg_similarity
                + 0.11 * experience_match
                + 0.08 * pattern_match
                + 0.08 * trajectory_match
                + 0.07 * severity_match
                + 0.07 * kg_signature_match
                + 0.06 * intervention_match
                + 0.03 * direction_match
                + 0.02 * support_bonus
                + 0.02 * max(0.0, quality_score)
            )
            hits.append(
                SemanticPrototypeHit(
                    prototype_id=str(candidate.get("prototype_id", "")),
                    dataset_name=str(candidate.get("dataset_name", "")),
                    experience_label=int(candidate.get("experience_label", -1)),
                    pattern_label=int(candidate.get("pattern_label", -1)),
                    trajectory_label=int(candidate.get("trajectory_label", -1)),
                    score=float(score),
                    support=support,
                    quality_score=quality_score,
                    group_key=str(candidate.get("group_key", "")),
                    future_curve=[float(value) for value in candidate.get("prototype_future_mean_curve", [])],
                    future_direction_type=str(candidate.get("future_direction_type", "flat")),
                    outcome_entropy=float(candidate.get("outcome_entropy", 0.0)),
                )
            )
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[: max(1, int(top_k))]
