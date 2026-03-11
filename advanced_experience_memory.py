#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前沿记忆增强经验库（Advanced Experience Memory Bank）

增强点（融合基础经验阈值）：
1) 混合检索：dense + intervention symbolic + 阈值一致性 + 质量 + 时间衰减
2) 记忆分层：episodic / semantic / procedural
3) 规则巩固：支持度 + 质量 + 阈值覆盖度
4) 置信度感知：基于案例质量进行召回加权
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Dict, List, Optional


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _extract_patient_id(text: str) -> str:
    m = re.search(r"ICU住院编号\s*(\d+)", text or "")
    if m:
        return m.group(1)
    return f"unknown_{abs(hash(text or '')) % 100000}"


def _extract_age_gender(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    ag = re.search(r"(\d+)岁(男性|女性)", text or "")
    if ag:
        out["age"] = int(ag.group(1))
        out["gender"] = "M" if ag.group(2) == "男性" else "F"
    return out


def _parse_risk_level(result: Dict[str, Any]) -> str:
    risk = result.get("risk_level")
    if isinstance(risk, str) and risk:
        return risk.lower()

    pred = result.get("prediction", {}) if isinstance(result.get("prediction"), dict) else {}
    ia = pred.get("intervention_analysis", {}) if isinstance(pred.get("intervention_analysis"), dict) else {}
    risk = ia.get("risk_level")
    return str(risk).lower() if isinstance(risk, str) and risk else "unknown"


def _intervention_type(text: str) -> str:
    s = (text or "").lower()
    if "去甲肾上腺素" in s or "norepinephrine" in s:
        return "norepinephrine"
    if "多巴胺" in s or "dopamine" in s:
        return "dopamine"
    if "抗生素" in s or "antibiotic" in s:
        return "antibiotic"
    if "机械通气" in s or "ventilation" in s:
        return "ventilation"
    if "补液" in s or "fluid" in s:
        return "fluid"
    return "other"


def _hash_embedding(text: str, dim: int = 128) -> List[float]:
    vec = [0.0] * dim
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", (text or "").lower())
    if not tokens:
        return vec
    for tok in tokens:
        h = hashlib.md5(tok.encode("utf-8")).hexdigest()
        idx = int(h[:8], 16) % dim
        sign = 1.0 if int(h[8:10], 16) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _parse_series_values(text: str, key: str) -> List[float]:
    pattern = rf"{re.escape(key)}.*?\[(.*?)\]"
    m = re.search(pattern, text or "")
    if not m:
        return []
    content = m.group(1)
    vals = []
    for p in content.split(','):
        p = p.strip()
        if not p or p.lower() == 'nan':
            continue
        try:
            vals.append(float(p))
        except Exception:
            continue
    return vals


def _extract_clinical_state(input_description: str, intervention: str, result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """从描述中抽取基础经验阈值状态（qSOFA/SOFA/休克相关）"""
    text = input_description or ""

    rr_vals = _parse_series_values(text, "呼吸频率变化")
    sbp_vals = _parse_series_values(text, "收缩压变化")
    map_vals = _parse_series_values(text, "平均动脉压变化")
    gcs_vals = _parse_series_values(text, "格拉斯哥昏迷评分变化")
    lac_vals = _parse_series_values(text, "乳酸变化")
    pfr_vals = _parse_series_values(text, "氧合指数变化")
    plt_vals = _parse_series_values(text, "血小板计数变化")
    bili_vals = _parse_series_values(text, "总胆红素变化")
    crea_vals = _parse_series_values(text, "血清肌酐变化")
    urine_vals = _parse_series_values(text, "尿量变化")

    ne_vals = _parse_series_values(text, "去甲肾上腺素")
    if not ne_vals:
        ne_vals = _parse_series_values(text, "血管活性药物使用剂量")

    rr = rr_vals[-1] if rr_vals else None
    sbp = sbp_vals[-1] if sbp_vals else None
    map_v = map_vals[-1] if map_vals else None
    gcs = gcs_vals[-1] if gcs_vals else None
    lac = lac_vals[-1] if lac_vals else None
    pfr = pfr_vals[-1] if pfr_vals else None
    plt = plt_vals[-1] if plt_vals else None
    bili = bili_vals[-1] if bili_vals else None
    crea = crea_vals[-1] if crea_vals else None
    urine = urine_vals[-1] if urine_vals else None
    ne = ne_vals[-1] if ne_vals else None

    q_rr = 1 if rr is not None and rr >= 22 else 0
    q_sbp = 1 if sbp is not None and sbp <= 100 else 0
    q_mental = 1 if gcs is not None and gcs < 15 else 0
    qsofa_score = q_rr + q_sbp + q_mental

    state = {
        "qsofa_score": qsofa_score,
        "qsofa_high_risk": 1 if qsofa_score >= 2 else 0,
        "map_low": 1 if map_v is not None and map_v < 65 else 0,
        "lactate_gt2": 1 if lac is not None and lac > 2 else 0,
        "lactate_gt4": 1 if lac is not None and lac > 4 else 0,
        "pfr_lt200": 1 if pfr is not None and pfr < 200 else 0,
        "platelet_lt100": 1 if plt is not None and plt < 100 else 0,
        "bilirubin_high": 1 if bili is not None and bili > 34.2 else 0,
        "creatinine_high": 1 if crea is not None and crea > 177 else 0,
        "urine_very_low": 1 if urine is not None and urine < 500 else 0,
        "norepinephrine_gt01": 1 if ne is not None and ne > 0.1 else 0,
        "intervention_type": _intervention_type(intervention),
    }

    if isinstance(result, dict):
        series = result.get("predicted_sofa_scores_series")
        if isinstance(series, dict) and "sofa_total" in series and isinstance(series.get("sofa_total"), list):
            vals = [v for v in series.get("sofa_total", []) if isinstance(v, (int, float))]
            if len(vals) >= 2:
                state["sofa_delta_72h_ge2"] = 1 if (vals[-1] - vals[0]) >= 2 else 0

    return state


def _guideline_consistency(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    keys = [
        "qsofa_high_risk", "map_low", "lactate_gt2", "lactate_gt4", "pfr_lt200",
        "platelet_lt100", "bilirubin_high", "creatinine_high", "urine_very_low",
        "norepinephrine_gt01", "sofa_delta_72h_ge2",
    ]
    common = 0
    matches = 0
    for k in keys:
        if k in a and k in b:
            common += 1
            if a[k] == b[k]:
                matches += 1
    if common == 0:
        return 0.0
    return matches / common


@dataclass
class EpisodicMemory:
    case_id: str
    patient_id: str
    timestamp: float
    input_description: str
    intervention: str
    intervention_type: str
    risk_level: str
    quality_score: float
    total_confidence: float
    embedding: List[float]
    tags: List[str]
    clinical_state: Dict[str, Any] = field(default_factory=dict)


class AdvancedExperienceMemoryBank:
    def __init__(self, storage_path: str = "./output/advanced_experience_memory.json"):
        self.storage_path = storage_path
        self.schema_version = 2
        self.episodic_memories: Dict[str, EpisodicMemory] = {}
        self.semantic_memory: Dict[str, Any] = {
            "risk_distribution": defaultdict(int),
            "intervention_effectiveness": defaultdict(lambda: {"count": 0, "avg_quality": 0.0}),
        }
        self.procedural_rules: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for m in data.get("episodic_memories", []):
                if "clinical_state" not in m or not isinstance(m.get("clinical_state"), dict):
                    m["clinical_state"] = {}
                self.episodic_memories[m["case_id"]] = EpisodicMemory(**m)
            sem = data.get("semantic_memory", {})
            self.semantic_memory["risk_distribution"] = defaultdict(int, sem.get("risk_distribution", {}))
            self.semantic_memory["intervention_effectiveness"] = defaultdict(
                lambda: {"count": 0, "avg_quality": 0.0},
                sem.get("intervention_effectiveness", {}),
            )
            self.procedural_rules = data.get("procedural_rules", [])
        except Exception:
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        data = {
            "schema_version": self.schema_version,
            "updated_at": datetime.now().isoformat(),
            "episodic_memories": [asdict(v) for v in self.episodic_memories.values()],
            "semantic_memory": {
                "risk_distribution": dict(self.semantic_memory["risk_distribution"]),
                "intervention_effectiveness": dict(self.semantic_memory["intervention_effectiveness"]),
            },
            "procedural_rules": self.procedural_rules,
        }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _update_semantic(self, memory: EpisodicMemory) -> None:
        self.semantic_memory["risk_distribution"][memory.risk_level] += 1
        stat = self.semantic_memory["intervention_effectiveness"][memory.intervention_type]
        c = stat.get("count", 0)
        avg = _safe_float(stat.get("avg_quality", 0.0))
        new_avg = (avg * c + memory.quality_score) / (c + 1)
        stat["count"] = c + 1
        stat["avg_quality"] = round(new_avg, 4)

    def _consolidate_rules(self) -> None:
        grouped: Dict[str, List[EpisodicMemory]] = defaultdict(list)
        for m in self.episodic_memories.values():
            grouped[f"{m.intervention_type}:{m.risk_level}"].append(m)

        rules: List[Dict[str, Any]] = []
        for key, cases in grouped.items():
            if len(cases) < 3:
                continue
            avg_quality = sum(c.quality_score for c in cases) / len(cases)
            if avg_quality < 0.6:
                continue

            intervention_type, risk_level = key.split(":", 1)
            threshold_keys = [
                "qsofa_high_risk", "map_low", "lactate_gt2", "lactate_gt4",
                "pfr_lt200", "platelet_lt100", "creatinine_high", "norepinephrine_gt01",
            ]
            profile = {}
            covered = 0
            for tk in threshold_keys:
                vals = [c.clinical_state.get(tk) for c in cases if tk in c.clinical_state]
                if not vals:
                    continue
                covered += 1
                positive_ratio = sum(1 for v in vals if v == 1) / len(vals)
                profile[tk] = 1 if positive_ratio >= 0.6 else 0

            threshold_coverage = covered / len(threshold_keys) if threshold_keys else 0.0
            rule_confidence = min(0.98, 0.45 + 0.35 * avg_quality + 0.2 * threshold_coverage)

            rules.append({
                "rule_id": f"rule_{intervention_type}_{risk_level}",
                "if": {
                    "intervention_type": intervention_type,
                    "risk_level": risk_level,
                    "clinical_threshold_profile": profile,
                },
                "then": {
                    "recommendation": f"在{risk_level}风险且使用{intervention_type}时，优先参考高质量病例并校验关键阈值状态。",
                    "expected_quality": round(avg_quality, 3),
                    "threshold_coverage": round(threshold_coverage, 3),
                },
                "support": len(cases),
                "confidence": round(rule_confidence, 3),
            })
        self.procedural_rules = sorted(rules, key=lambda x: (x["confidence"], x["support"]), reverse=True)[:30]

    def add_experience_from_result(self, result_file: str) -> Optional[str]:
        if not os.path.exists(result_file):
            return None
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                r = json.load(f)

            input_description = r.get("input_description", "")
            intervention = r.get("intervention", "")
            patient_id = str(r.get("patient_id") or _extract_patient_id(input_description))
            confidence = _safe_float(r.get("total_confidence", 0.0))
            risk_level = _parse_risk_level(r)
            quality = max(0.0, min(1.0, confidence))

            extra = _extract_age_gender(input_description)
            clinical_state = _extract_clinical_state(input_description, intervention, r)
            tags = [f"risk_{risk_level}", f"intv_{_intervention_type(intervention)}"]
            if extra.get("age"):
                tags.append("elderly" if extra["age"] >= 65 else "non_elderly")
            if extra.get("gender"):
                tags.append(extra["gender"].lower())
            if clinical_state.get("qsofa_high_risk") == 1:
                tags.append("qsofa_high")
            if clinical_state.get("lactate_gt2") == 1:
                tags.append("lactate_high")

            now_ts = time.time()
            case_id = f"mem_{patient_id}_{int(now_ts)}"
            merged_text = f"{input_description}\n干预:{intervention}\n风险:{risk_level}"
            m = EpisodicMemory(
                case_id=case_id,
                patient_id=patient_id,
                timestamp=now_ts,
                input_description=input_description,
                intervention=intervention,
                intervention_type=_intervention_type(intervention),
                risk_level=risk_level,
                quality_score=quality,
                total_confidence=confidence,
                embedding=_hash_embedding(merged_text),
                tags=tags,
                clinical_state=clinical_state,
            )
            self.episodic_memories[case_id] = m
            self._update_semantic(m)
            self._consolidate_rules()
            self._save()
            return case_id
        except Exception:
            return None

    def _recency_weight(self, ts: float, risk_level: str = "unknown") -> float:
        if risk_level in ("high", "critical"):
            half_life_days = 14.0
        elif risk_level in ("low", "moderate"):
            half_life_days = 45.0
        else:
            half_life_days = 30.0
        delta_days = max(0.0, (time.time() - ts) / 86400.0)
        return math.exp(-math.log(2) * delta_days / half_life_days)

    def get_recommendations(self, input_description: str, intervention: str, top_k: int = 5) -> Dict[str, Any]:
        q_risk = "unknown"
        q_it = _intervention_type(intervention)
        q_emb = _hash_embedding(f"{input_description}\n干预:{intervention}")
        q_state = _extract_clinical_state(input_description, intervention)

        scored = []
        for m in self.episodic_memories.values():
            dense = _cosine(q_emb, m.embedding)
            symbolic_intervention = 1.0 if m.intervention_type == q_it else 0.0
            symbolic_threshold = _guideline_consistency(q_state, m.clinical_state)
            symbolic = 0.5 * symbolic_intervention + 0.5 * symbolic_threshold
            if q_risk != "unknown" and m.risk_level == q_risk:
                symbolic = min(1.0, symbolic + 0.1)

            quality = m.quality_score
            recency = self._recency_weight(m.timestamp, m.risk_level)
            guideline = symbolic_threshold

            score = 0.35 * dense + 0.20 * symbolic + 0.25 * guideline + 0.15 * quality + 0.05 * recency
            scored.append((score, m, {
                "dense": dense,
                "symbolic": symbolic,
                "guideline": guideline,
                "quality": quality,
                "recency": recency,
            }))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        similar_experiences = []
        for s, m, br in top:
            similar_experiences.append({
                "case_id": m.case_id,
                "similarity": round(float(s), 4),
                "patient_id": m.patient_id,
                "risk_level": m.risk_level,
                "intervention": m.intervention,
                "quality_score": m.quality_score,
                "total_confidence": m.total_confidence,
                "score_breakdown": {k: round(float(v), 4) for k, v in br.items()},
                "clinical_state": m.clinical_state,
            })

        suggestions: List[str] = []
        if similar_experiences:
            best = similar_experiences[0]
            suggestions.append(
                f"最相似案例风险等级为 {best['risk_level']}，置信度 {best['total_confidence']:.2f}，建议优先核对阈值状态一致性后参考SOFA轨迹。"
            )
            if isinstance(best.get("clinical_state"), dict):
                cs = best["clinical_state"]
                if cs.get("qsofa_high_risk") == 1:
                    suggestions.append("历史最相似案例属于qSOFA高风险，请提高对血流动力学恶化的警惕。")
                if cs.get("lactate_gt2") == 1:
                    suggestions.append("历史相似案例存在乳酸升高信号，建议关注组织灌注与复苏充分性。")

        top_rules = [r for r in self.procedural_rules if r.get("if", {}).get("intervention_type") == q_it][:3]
        for r in top_rules:
            suggestions.append(r.get("then", {}).get("recommendation", ""))

        return {
            "similar_experiences": similar_experiences,
            "suggestions": [s for s in suggestions if s],
            "relevant_rules": top_rules,
        }

    def batch_import_from_directory(self, directory: str, pattern: str = "result_*.json") -> int:
        if not os.path.isdir(directory):
            return 0
        count = 0
        for name in os.listdir(directory):
            if not fnmatch.fnmatch(name, pattern):
                continue
            path = os.path.join(directory, name)
            if self.add_experience_from_result(path):
                count += 1
        return count

    def get_statistics(self) -> Dict[str, Any]:
        risk_dist = dict(self.semantic_memory["risk_distribution"])
        return {
            "memory_type": "advanced_hybrid_memory_guideline_enhanced",
            "episodic_count": len(self.episodic_memories),
            "procedural_rules_count": len(self.procedural_rules),
            "risk_distribution": risk_dist,
            "top_interventions": sorted(
                [
                    {"intervention_type": k, **v}
                    for k, v in dict(self.semantic_memory["intervention_effectiveness"]).items()
                ],
                key=lambda x: x.get("count", 0),
                reverse=True,
            )[:5],
        }
