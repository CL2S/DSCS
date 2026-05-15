from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_MAPPING_PATH = Path(__file__).resolve().parents[1] / "input" / "knowledge" / "06_schema_design" / "eicu_to_kg_node_mapping.csv"
GUIDELINE_ALIGNMENT_FEATURE = "kg_guideline_alignment"


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


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    return "" if text == "nan" else text


def _normalize_lookup_key(value: object) -> str:
    text = _safe_text(value)
    for token in ("_", "-", "/", "(", ")", ","):
        text = text.replace(token, " ")
    return " ".join(text.split())


def _split_candidates(value: object) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [item.strip() for item in text.split("|") if item.strip()]


def _string_or_empty(value: object) -> str:
    return "" if value is None or _safe_text(value) == "" else str(value).strip()


def _parse_aliases(value: object) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    sanitized = text.replace("[", "").replace("]", "").replace('"', "").replace("'", "")
    return [item.strip() for item in sanitized.split(",") if item.strip()]


def _bool_flag(value: object) -> float:
    return 1.0 if _safe_float(value) > 0.0 else 0.0


def _series_available(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    return 1.0 if pd.to_numeric(frame[column], errors="coerce").notna().any() else 0.0


def _series_repeat_available(frame: pd.DataFrame, column: str, minimum_count: int = 2) -> float:
    if column not in frame.columns:
        return 0.0
    count = int(pd.to_numeric(frame[column], errors="coerce").notna().sum())
    return 1.0 if count >= minimum_count else 0.0


def _series_any_positive(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return 1.0 if float(values.max()) > 0.0 else 0.0


def _series_min_below(frame: pd.DataFrame, column: str, threshold: float) -> float:
    if column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return 0.0
    return 1.0 if float(values.min()) < threshold else 0.0


def _series_max_above_or_equal(frame: pd.DataFrame, column: str, threshold: float) -> float:
    if column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return 0.0
    return 1.0 if float(values.max()) >= threshold else 0.0


def _load_lookup_entry(
    lookup: Dict[str, Dict[str, str]],
    name: object,
    node_id: object,
    node_type: object,
) -> None:
    node_name = str(name or "").strip()
    if not node_name:
        return
    key = _normalize_lookup_key(node_name)
    if not key or key in lookup:
        return
    lookup[key] = {
        "node_id": str(node_id or ""),
        "node_name": node_name,
        "node_type": str(node_type or "").strip(),
    }


@dataclass
class KnowledgeGraphMappingRule:
    feature_name: str
    feature_group: str
    kg_node_name: str
    kg_node_candidates: List[str]
    kg_node_type: str
    relation_type: str
    source_table: str
    source_column: str
    rule_type: str
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    text_token: str
    description: str
    resolved_node_id: str = ""
    resolved_node_name: str = ""
    resolved_node_type: str = ""


@dataclass
class KnowledgeGraphFeatureBuilder:
    kg_directory: str
    feature_names: List[str]
    graph_summary: Dict[str, object]
    mapping_rules: List[KnowledgeGraphMappingRule]

    @classmethod
    def from_directory(cls, kg_directory: str, mapping_path: str = "") -> "KnowledgeGraphFeatureBuilder":
        kg_path = Path(kg_directory)
        if not kg_path.exists():
            raise FileNotFoundError(f"KG directory not found: {kg_directory}")

        nodes_path = kg_path / "nodes.csv"
        edges_path = kg_path / "edges.csv"
        guideline_path = kg_path / "guideline_relations.csv"
        build_summary_path = kg_path / "build_summary.json"
        resolved_mapping_path = Path(mapping_path).expanduser().resolve() if mapping_path else DEFAULT_MAPPING_PATH.resolve()
        if not nodes_path.exists() or not edges_path.exists():
            raise FileNotFoundError(f"Missing KG core files under: {kg_directory}")
        if not resolved_mapping_path.exists():
            raise FileNotFoundError(f"KG mapping file not found: {resolved_mapping_path}")

        nodes_df = pd.read_csv(
            nodes_path,
            usecols=["node_id", "entity_id", "name", "aliases", "primary_type"],
        )
        edges_df = pd.read_csv(edges_path, usecols=["relation_type"])
        guideline_df = pd.read_csv(guideline_path) if guideline_path.exists() else pd.DataFrame()
        build_summary: Dict[str, object] = {}
        if build_summary_path.exists():
            build_summary = json.loads(build_summary_path.read_text(encoding="utf-8"))

        node_lookup: Dict[str, Dict[str, str]] = {}
        for _, row in nodes_df.iterrows():
            node_id = row.get("node_id") or row.get("entity_id") or ""
            _load_lookup_entry(node_lookup, row.get("name"), node_id, row.get("primary_type"))
            for alias in _parse_aliases(row.get("aliases")):
                _load_lookup_entry(node_lookup, alias, node_id, row.get("primary_type"))
        if not guideline_df.empty:
            for _, row in guideline_df.iterrows():
                _load_lookup_entry(node_lookup, row.get("source_name"), row.get("source_id"), row.get("source_type"))
                _load_lookup_entry(node_lookup, row.get("target_name"), row.get("target_id"), row.get("target_type"))

        mapping_df = pd.read_csv(resolved_mapping_path)
        mapping_rules: List[KnowledgeGraphMappingRule] = []
        ordered_feature_names: List[str] = []
        for _, row in mapping_df.iterrows():
            feature_name = str(row.get("feature_name", "")).strip()
            if not feature_name:
                continue
            if feature_name not in ordered_feature_names:
                ordered_feature_names.append(feature_name)
            candidates = _split_candidates(row.get("kg_node_candidates")) or [str(row.get("kg_node_name", "")).strip()]
            resolved = {}
            for candidate in candidates:
                resolved = node_lookup.get(_normalize_lookup_key(candidate), {})
                if resolved:
                    break
            mapping_rules.append(
                KnowledgeGraphMappingRule(
                    feature_name=feature_name,
                    feature_group=str(row.get("feature_group", "")).strip(),
                    kg_node_name=str(row.get("kg_node_name", "")).strip(),
                    kg_node_candidates=candidates,
                    kg_node_type=str(row.get("kg_node_type", "")).strip(),
                    relation_type=str(row.get("relation_type", "")).strip(),
                    source_table=str(row.get("source_table", "")).strip(),
                    source_column=str(row.get("source_column", "")).strip(),
                    rule_type=str(row.get("rule_type", "")).strip(),
                    lower_bound=_parse_optional_float(row.get("lower_bound")),
                    upper_bound=_parse_optional_float(row.get("upper_bound")),
                    text_token=_string_or_empty(row.get("text_token")),
                    description=_string_or_empty(row.get("description")),
                    resolved_node_id=str(resolved.get("node_id", "")),
                    resolved_node_name=str(resolved.get("node_name", "")),
                    resolved_node_type=str(resolved.get("node_type", "")),
                )
            )

        relation_types = {_safe_text(name) for name in edges_df.get("relation_type", pd.Series(dtype=str)).tolist()}
        feature_names = ordered_feature_names + [GUIDELINE_ALIGNMENT_FEATURE]
        resolved_rule_count = sum(1 for rule in mapping_rules if rule.resolved_node_id)
        unresolved_features = sorted({rule.feature_name for rule in mapping_rules if not rule.resolved_node_id})
        graph_summary = {
            "kg_directory": str(kg_path.resolve()),
            "mapping_path": str(resolved_mapping_path),
            "node_count": int(len(nodes_df)),
            "relation_type_count": int(len(relation_types)),
            "guideline_relation_count": int(len(guideline_df)),
            "mapping_rule_count": int(len(mapping_rules)),
            "mapping_feature_count": int(len(ordered_feature_names)),
            "resolved_mapping_rule_count": int(resolved_rule_count),
            "unresolved_mapping_rule_count": int(len(mapping_rules) - resolved_rule_count),
            "unresolved_features": unresolved_features,
            "has_sepsis_node": bool(node_lookup.get("sepsis")),
            "has_septic_shock_node": bool(node_lookup.get("septic shock")),
            "has_guideline_treatment_relations": "treatment" in relation_types,
            "build_summary": build_summary,
        }
        return cls(
            kg_directory=str(kg_path.resolve()),
            feature_names=feature_names,
            graph_summary=graph_summary,
            mapping_rules=mapping_rules,
        )

    def _evaluate_rule(
        self,
        rule: KnowledgeGraphMappingRule,
        label_row: Dict[str, object],
        context_frame: pd.DataFrame,
    ) -> float:
        if rule.source_table == "labels":
            raw_value = label_row.get(rule.source_column)
            numeric_value = _safe_float(raw_value, default=float("nan"))
            if rule.rule_type == "label_positive":
                return _bool_flag(raw_value)
            if rule.rule_type == "label_gt":
                return 1.0 if not math.isnan(numeric_value) and numeric_value > float(rule.lower_bound or 0.0) else 0.0
            if rule.rule_type == "label_ge":
                return 1.0 if not math.isnan(numeric_value) and numeric_value >= float(rule.lower_bound or 0.0) else 0.0
            if rule.rule_type == "label_between":
                lower = float(rule.lower_bound or 0.0)
                upper = float(rule.upper_bound or 0.0)
                return 1.0 if not math.isnan(numeric_value) and lower < numeric_value < upper else 0.0
            if rule.rule_type == "label_between_inclusive":
                lower = float(rule.lower_bound or 0.0)
                upper = float(rule.upper_bound or 0.0)
                return 1.0 if not math.isnan(numeric_value) and lower <= numeric_value <= upper else 0.0
            if rule.rule_type == "text_contains":
                token = _safe_text(rule.text_token)
                return 1.0 if token and token in _safe_text(raw_value) else 0.0
            return 0.0

        if rule.source_table != "trajectory":
            return 0.0
        if rule.rule_type == "series_present":
            return _series_available(context_frame, rule.source_column)
        if rule.rule_type == "series_any_positive":
            return _series_any_positive(context_frame, rule.source_column)
        if rule.rule_type == "series_min_lt":
            return _series_min_below(context_frame, rule.source_column, float(rule.upper_bound or 0.0))
        if rule.rule_type == "series_max_ge":
            return _series_max_above_or_equal(context_frame, rule.source_column, float(rule.lower_bound or 0.0))
        if rule.rule_type == "series_count_ge":
            return _series_repeat_available(context_frame, rule.source_column, int(rule.lower_bound or 2.0))
        return 0.0

    def mapping_records(self) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        for rule in self.mapping_rules:
            records.append(
                {
                    "feature_name": rule.feature_name,
                    "feature_group": rule.feature_group,
                    "kg_node_name": rule.kg_node_name,
                    "kg_node_candidates": "|".join(rule.kg_node_candidates),
                    "kg_node_type": rule.kg_node_type,
                    "relation_type": rule.relation_type,
                    "source_table": rule.source_table,
                    "source_column": rule.source_column,
                    "rule_type": rule.rule_type,
                    "lower_bound": rule.lower_bound,
                    "upper_bound": rule.upper_bound,
                    "text_token": rule.text_token,
                    "description": rule.description,
                    "resolved_node_id": rule.resolved_node_id,
                    "resolved_node_name": rule.resolved_node_name,
                    "resolved_node_type": rule.resolved_node_type,
                }
            )
        return records

    def build_features(
        self,
        label_row: Dict[str, object],
        context_frame: pd.DataFrame,
    ) -> Tuple[List[float], Dict[str, float], float]:
        base_feature_names = [name for name in self.feature_names if name != GUIDELINE_ALIGNMENT_FEATURE]
        sample_flags: Dict[str, float] = {}
        for feature_name in base_feature_names:
            feature_rules = [rule for rule in self.mapping_rules if rule.feature_name == feature_name]
            feature_value = max(
                [self._evaluate_rule(rule, label_row, context_frame) for rule in feature_rules] or [0.0]
            )
            sample_flags[feature_name] = float(feature_value)

        expected_count = 0.0
        aligned_count = 0.0
        if sample_flags.get("kg_state_sepsis", 0.0) > 0.0:
            expected_count += 4.0
            aligned_count += sample_flags.get("kg_exam_sofa", 0.0)
            aligned_count += sample_flags.get("kg_exam_lactate", 0.0)
            aligned_count += sample_flags.get("kg_exam_blood_culture", 0.0)
            aligned_count += sample_flags.get("kg_treat_early_antimicrobial", 0.0)
        if sample_flags.get("kg_state_septic_shock", 0.0) > 0.0 or sample_flags.get("kg_state_hypotension", 0.0) > 0.0:
            expected_count += 3.0
            aligned_count += sample_flags.get("kg_treat_vasopressor", 0.0)
            aligned_count += sample_flags.get("kg_monitor_map65", 0.0)
            aligned_count += sample_flags.get("kg_monitor_lactate_repeat", 0.0)
        guideline_alignment = aligned_count / max(1.0, expected_count) if expected_count > 0.0 else 0.0
        sample_flags[GUIDELINE_ALIGNMENT_FEATURE] = float(guideline_alignment)

        feature_vector = [sample_flags.get(feature_name, 0.0) for feature_name in base_feature_names]
        feature_vector.append(float(guideline_alignment))
        return feature_vector, sample_flags, float(guideline_alignment)


def _parse_optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    numeric = _safe_float(text, default=float("nan"))
    return None if math.isnan(numeric) else float(numeric)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm <= 1e-8 or right_norm <= 1e-8:
        return 0.0
    return float(dot / (left_norm * right_norm))
