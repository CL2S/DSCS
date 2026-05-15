from typing import Dict, List, Optional, Sequence


STATE_LABELS = {
    "kg_state_sepsis": "脓毒症",
    "kg_state_septic_shock": "脓毒性休克",
    "kg_state_organ_dysfunction": "器官功能障碍",
    "kg_state_hypotension": "低血压",
    "kg_state_high_lactate": "高乳酸",
}

REPAIR_ACTION_LABELS = {
    "early_antimicrobial": "补足早期抗菌治疗时效窗口",
    "antimicrobial_course": "补足抗菌疗程覆盖",
    "vasopressor_support": "补足血管活性药支持信号",
    "respiratory_support": "补足呼吸支持信号",
}

SEARCH_ACTION_LABELS = {
    "template_earlier_antimicrobial": "尝试更早抗菌模板",
    "template_antimicrobial_immediate": "尝试立即抗菌模板",
    "template_antimicrobial_2h": "尝试 2 小时内抗菌模板",
    "template_vasopressor_low": "尝试低强度升压支持模板",
    "template_vasopressor_bridge": "尝试桥接强度升压模板",
    "template_vasopressor_moderate": "尝试中等强度升压模板",
    "template_vasopressor_standard": "尝试标准强度升压支持模板",
    "template_respiratory_support": "尝试呼吸支持模板",
    "template_respiratory_support_low": "尝试低强度呼吸支持模板",
    "template_combo_sepsis_hemodynamic": "尝试脓毒症血流动力学组合模板",
}

CANDIDATE_SOURCE_LABELS = {
    "raw_intervention_store": "原始干预库",
    "donor_original": "相似病例原始方案",
    "generated_kg_repaired": "相似病例方案 + KG 修补",
    "generated_template_antimicrobial_fast": "安全修补后模板搜索：更早抗菌",
    "generated_template_antimicrobial_immediate": "参数化模板搜索：立即抗菌",
    "generated_template_antimicrobial_2h": "参数化模板搜索：2 小时内抗菌",
    "generated_template_vasopressor_bridge": "参数化模板搜索：桥接强度升压支持",
    "generated_template_vasopressor_low": "安全修补后模板搜索：低强度升压支持",
    "generated_template_vasopressor_moderate": "参数化模板搜索：中等强度升压支持",
    "generated_template_vasopressor_standard": "安全修补后模板搜索：标准强度升压支持",
    "generated_template_sepsis_bundle": "安全修补后模板搜索：脓毒症 bundle",
    "generated_template_respiratory_support": "安全修补后模板搜索：呼吸支持",
    "generated_template_respiratory_support_low": "参数化模板搜索：低强度呼吸支持",
    "generated_strategy_sepsis_hemodynamic_combo": "安全偏离 donor：脓毒症血流动力学组合策略",
}

STRATEGY_FAMILY_LABELS = {
    "donor_original": "donor 原始方案",
    "kg_repair": "KG 安全修补",
    "template": "固定模板",
    "antimicrobial_timing": "抗菌时机参数化",
    "vasopressor_intensity": "升压强度参数化",
    "respiratory_intensity": "呼吸支持强度参数化",
    "sepsis_hemodynamic_combo": "脓毒症血流动力学组合",
}

PATTERN_LABELS = {
    "flat": "平稳",
    "up": "上升",
    "down": "下降",
    "volatile": "波动",
    "spike": "尖峰/突变",
}

TRAJECTORY_LABELS = {
    "stable_regime": "稳定阶段",
    "rising_regime": "上升阶段",
    "falling_regime": "下降阶段",
    "seasonal_regime": "周期阶段",
    "shifted_regime": "阶段转换",
}

AUXILIARY_FORECAST_LABELS = {
    "future_sofa_delta_mean": "未来 SOFA 增量",
    "future_lactate_delta": "未来乳酸增量",
    "future_vasopressor_need": "后续血管活性药需求概率",
    "future_resp_support_escalation": "后续呼吸支持升级概率",
}

SELECTION_CONFLICT_LABELS = {
    "future_sofa_worse": "未来 SOFA 方向未优于当前方案",
    "future_lactate_worse": "未来乳酸方向未优于当前方案",
    "vasopressor_risk_higher": "后续升压药需求风险更高",
    "resp_support_risk_higher": "后续呼吸支持升级风险更高",
}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _boolish(value: object, threshold: float = 0.5) -> bool:
    return _safe_float(value) >= threshold


def _round(value: object, digits: int = 3) -> float:
    return round(_safe_float(value), digits)


def _mean(values: Sequence[object]) -> float:
    numeric = [_safe_float(value) for value in values]
    if not numeric:
        return 0.0
    return sum(numeric) / max(1, len(numeric))


def _state_summary(active_state_flags: Sequence[str]) -> List[str]:
    return [STATE_LABELS.get(str(flag), str(flag)) for flag in active_state_flags]


def _source_label(source: object) -> str:
    normalized = str(source or "").strip()
    return CANDIDATE_SOURCE_LABELS.get(normalized, normalized or "未知来源")


def _pattern_label(name: object) -> str:
    normalized = str(name or "").strip()
    return PATTERN_LABELS.get(normalized, normalized)


def _trajectory_label(name: object) -> str:
    normalized = str(name or "").strip()
    return TRAJECTORY_LABELS.get(normalized, normalized)


def _repair_labels(actions: Sequence[object]) -> List[str]:
    return [REPAIR_ACTION_LABELS.get(str(action), str(action)) for action in actions]


def _search_labels(actions: Sequence[object]) -> List[str]:
    return [SEARCH_ACTION_LABELS.get(str(action), str(action)) for action in actions]


def _strategy_family_label(name: object) -> str:
    normalized = str(name or "").strip()
    return STRATEGY_FAMILY_LABELS.get(normalized, normalized or "未标注")


def _parameter_profile_summary(profile: Dict[str, object]) -> str:
    items: List[str] = []
    if "antibiotic_offset_minutes" in profile:
        items.append(f"抗菌起始时机={_round(profile.get('antibiotic_offset_minutes'))} 分钟")
    if "antibiotic_course_ge_72h" in profile:
        items.append(f"抗菌疗程覆盖标志={_round(profile.get('antibiotic_course_ge_72h'))}")
    if "vasopressor_intensity" in profile:
        items.append(f"升压强度={_round(profile.get('vasopressor_intensity'))}")
    if "respiratory_support_intensity" in profile:
        items.append(f"呼吸支持强度={_round(profile.get('respiratory_support_intensity'))}")
    return "；".join(items) if items else "无显式参数偏移"


def _horizon_labels(prediction: Sequence[object], step_hours: int = 6) -> List[str]:
    return [f"+{(index + 1) * step_hours}h" for index in range(len(prediction))]


def _trajectory_rows(prediction: Sequence[object], step_hours: int = 6) -> List[Dict[str, object]]:
    labels = _horizon_labels(prediction, step_hours=step_hours)
    return [
        {
            "horizon": label,
            "predicted_value": _round(value),
        }
        for label, value in zip(labels, prediction)
    ]


def _monitoring_summary(action_profile: Dict[str, object]) -> str:
    items: List[str] = []
    if _boolish(action_profile.get("exam_blood_culture")):
        items.append("已包含血培养")
    if _boolish(action_profile.get("exam_lactate")):
        items.append("已包含乳酸检测")
    if _boolish(action_profile.get("monitor_map65")):
        items.append("已包含 MAP>=65 监测")
    if _boolish(action_profile.get("monitor_lactate_repeat")):
        items.append("已包含乳酸复测")
    return "；".join(items) if items else "未见新增监测/检查信号"


def _antimicrobial_status(action_profile: Dict[str, object]) -> str:
    if not _boolish(action_profile.get("treat_early_antimicrobial")):
        return "未见早期抗菌治疗信号"
    return (
        "已建议/包含早期抗菌；"
        f"时效评分={_round(action_profile.get('antibiotic_timeliness'))}；"
        f"疗程标志={_round(action_profile.get('antibiotic_duration_flag'))}"
    )


def _vasopressor_status(action_profile: Dict[str, object]) -> str:
    if not _boolish(action_profile.get("treat_vasopressor")):
        return "未见血管活性药支持信号"
    return (
        "已建议/包含血管活性药支持；"
        f"last={_round(action_profile.get('vasopressor_intensity_last'))}；"
        f"mean={_round(action_profile.get('vasopressor_intensity_mean'))}；"
        f"max={_round(action_profile.get('vasopressor_intensity_max'))}"
    )


def _respiratory_status(action_profile: Dict[str, object]) -> str:
    if not _boolish(action_profile.get("treat_respiratory_support")):
        return "未见呼吸支持信号"
    return (
        "已建议/包含呼吸支持；"
        f"last={_round(action_profile.get('respiratory_intensity_last'))}；"
        f"mean={_round(action_profile.get('respiratory_intensity_mean'))}；"
        f"max={_round(action_profile.get('respiratory_intensity_max'))}"
    )


def _intervention_detail_rows(
    current_plan_summary: Dict[str, object],
    recommended_plan_summary: Dict[str, object],
    repair_actions: Sequence[object],
) -> List[Dict[str, object]]:
    current_profile = dict(current_plan_summary.get("action_profile", {}))
    recommended_profile = dict(recommended_plan_summary.get("action_profile", {}))
    repairs = set(str(action) for action in repair_actions)

    return [
        {
            "domain": "抗感染",
            "current_status": _antimicrobial_status(current_profile),
            "recommended_status": _antimicrobial_status(recommended_profile),
            "change_summary": "需补足早期抗菌时效窗口" if "early_antimicrobial" in repairs else "未见新增抗菌修补",
        },
        {
            "domain": "血流动力学支持",
            "current_status": _vasopressor_status(current_profile),
            "recommended_status": _vasopressor_status(recommended_profile),
            "change_summary": "需补足血管活性药支持" if "vasopressor_support" in repairs else "未见新增血流动力学修补",
        },
        {
            "domain": "呼吸支持",
            "current_status": _respiratory_status(current_profile),
            "recommended_status": _respiratory_status(recommended_profile),
            "change_summary": "需补足呼吸支持" if "respiratory_support" in repairs else "未见新增呼吸支持修补",
        },
        {
            "domain": "监测与检查",
            "current_status": _monitoring_summary(current_profile),
            "recommended_status": _monitoring_summary(recommended_profile),
            "change_summary": "依据 donor/KG 保留或补足必要监测信号",
        },
    ]


def _confidence_level(selected_candidate: Dict[str, object]) -> str:
    donor = dict(selected_candidate.get("donor", {}))
    similarity = _safe_float(donor.get("donor_similarity"))
    guideline = _safe_float(donor.get("donor_guideline_compatibility"))
    penalties = _safe_float(donor.get("donor_missing_care_penalty")) + _safe_float(donor.get("donor_contraindication_penalty"))
    predicted_delta = _safe_float(selected_candidate.get("predicted_delta"))
    if similarity >= 0.75 and guideline >= 0.7 and penalties <= 0.25 and predicted_delta > 0.0:
        return "高"
    if similarity >= 0.45 and guideline >= 0.45 and penalties <= 0.6:
        return "中"
    return "低"


def _warning_lines(case_detail: Dict[str, object]) -> List[str]:
    selected_candidate = dict(case_detail.get("selected_candidate", {}))
    donor = dict(selected_candidate.get("donor", {}))
    warnings: List[str] = []
    if _safe_float(selected_candidate.get("predicted_delta")) <= 0.0:
        warnings.append("单步模拟未显示该候选方案能带来更优的短期预测。")
    if _safe_float(donor.get("donor_similarity")) < 0.4:
        warnings.append("相似病例相似度偏低，类比证据较弱。")
    if not _boolish(donor.get("donor_hard_filter_valid")):
        reason = str(donor.get("donor_hard_filter_reason", "")).strip()
        warnings.append(f"原始 donor 未通过硬过滤：{reason or '未说明原因'}。")
    if _safe_float(donor.get("donor_missing_care_penalty")) > 0.3:
        warnings.append("候选方案仍存在照护缺口处罚。")
    if _safe_float(donor.get("donor_contraindication_penalty")) > 0.0:
        warnings.append("候选方案仍存在禁忌或冲突处罚。")
    warnings.append("该输出属于相似病例检索与候选方案重排，不代表因果疗效证明。")
    return warnings


def _rationale_lines(case_detail: Dict[str, object]) -> List[str]:
    selected_candidate = dict(case_detail.get("selected_candidate", {}))
    donor = dict(selected_candidate.get("donor", {}))
    lines = [
        f"方案来源：{_source_label(selected_candidate.get('candidate_source'))}",
        f"模型 proxy 改善：{_round(selected_candidate.get('predicted_delta'))}",
        f"病例相似度：{_round(donor.get('donor_similarity'))}",
        f"KG 相似度：{_round(donor.get('donor_kg_similarity'))}",
        f"指南一致性：{_round(donor.get('donor_guideline_compatibility'))}",
        f"状态匹配度：{_round(donor.get('donor_state_match'))}",
        f"donor 总分：{_round(donor.get('donor_total_score'))}",
    ]
    repair_actions = _repair_labels(selected_candidate.get("repair_actions", []))
    if repair_actions:
        lines.append("KG 修补动作：" + "；".join(repair_actions))
    search_actions = _search_labels(selected_candidate.get("search_actions", []))
    if search_actions:
        lines.append("模板搜索动作：" + "；".join(search_actions))
    if str(selected_candidate.get("candidate_anchor_relation", "donor_centered")) == "safe_deviation":
        lines.append("该候选属于安全偏离 donor 的策略候选，而不是仅对 donor 原方案做局部修补。")
    strategy_family = str(selected_candidate.get("candidate_strategy_family", "")).strip()
    if strategy_family:
        lines.append("策略族：" + _strategy_family_label(strategy_family))
    parameter_summary = _parameter_profile_summary(dict(selected_candidate.get("candidate_parameter_profile", {})))
    if parameter_summary != "无显式参数偏移":
        lines.append("关键参数：" + parameter_summary)
    safety_rationale = [str(item) for item in selected_candidate.get("candidate_safety_rationale", []) if str(item).strip()]
    if safety_rationale:
        lines.append("安全边界依据：" + "；".join(safety_rationale))
    return lines


def _plan_highlight(plan_summary: Dict[str, object]) -> str:
    action_profile = dict(plan_summary.get("action_profile", {}))
    highlights: List[str] = []
    if _boolish(action_profile.get("treat_early_antimicrobial")):
        highlights.append("早期抗菌")
    if _boolish(action_profile.get("treat_vasopressor")):
        highlights.append("血管活性药")
    if _boolish(action_profile.get("treat_respiratory_support")):
        highlights.append("呼吸支持")
    if _boolish(action_profile.get("exam_lactate")):
        highlights.append("乳酸监测")
    if _boolish(action_profile.get("exam_blood_culture")):
        highlights.append("血培养")
    return "、".join(highlights) if highlights else "未见强干预信号"


def _dedupe_top_donors(top_donor_candidates: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    deduped: Dict[float, Dict[str, object]] = {}
    for candidate in top_donor_candidates:
        stay_id = _safe_float(candidate.get("stay_id"), -1.0)
        current = deduped.get(stay_id)
        if current is None or _safe_float(candidate.get("donor_total_score")) > _safe_float(current.get("donor_total_score")):
            deduped[stay_id] = dict(candidate)
    return sorted(
        deduped.values(),
        key=lambda item: _safe_float(item.get("donor_total_score")),
        reverse=True,
    )


def _top_similar_case_rows(top_donor_candidates: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for rank, candidate in enumerate(_dedupe_top_donors(top_donor_candidates), start=1):
        rows.append(
            {
                "rank": rank,
                "stay_id": _safe_float(candidate.get("stay_id"), -1.0),
                "similarity": _round(candidate.get("donor_similarity")),
                "kg_similarity": _round(candidate.get("donor_kg_similarity")),
                "guideline_compatibility": _round(candidate.get("donor_guideline_compatibility")),
                "state_match": _round(candidate.get("donor_state_match")),
                "total_score": _round(candidate.get("donor_total_score")),
                "hard_filter_status": "通过" if _boolish(candidate.get("donor_hard_filter_valid")) else "未通过",
                "overlap_status": "通过" if _boolish(candidate.get("donor_overlap_valid")) else "未通过",
                "plan_highlight": _plan_highlight(dict(candidate.get("plan_summary", {}))),
            }
        )
    return rows


def _predicted_outcome_section(
    base_prediction: Sequence[object],
    factual_prediction: Sequence[object],
    counterfactual_prediction: Sequence[object],
    predicted_delta: object,
    prediction_uncertainty: Optional[Dict[str, object]] = None,
    step_hours: int = 6,
) -> Dict[str, object]:
    base_mean = _mean(base_prediction)
    factual_mean = _mean(factual_prediction)
    counterfactual_mean = _mean(counterfactual_prediction)
    change_vs_current = counterfactual_mean - factual_mean
    if change_vs_current < 0.0:
        interpretation = "推荐方案对应更低的未来 SOFA 均值，方向上更优。"
    elif change_vs_current > 0.0:
        interpretation = "推荐方案对应更高的未来 SOFA 均值，未显示改善。"
    else:
        interpretation = "推荐方案与当前方案的未来 SOFA 均值基本持平。"
    section = {
        "target_name": "未来 SOFA 预测轨迹",
        "baseline_prediction": _trajectory_rows(base_prediction, step_hours=step_hours),
        "current_plan_prediction": _trajectory_rows(factual_prediction, step_hours=step_hours),
        "recommended_plan_prediction": _trajectory_rows(counterfactual_prediction, step_hours=step_hours),
        "summary": {
            "baseline_mean": _round(base_mean),
            "current_plan_mean": _round(factual_mean),
            "recommended_plan_mean": _round(counterfactual_mean),
            "recommended_minus_current": _round(change_vs_current),
            "model_proxy_delta": _round(predicted_delta),
            "interpretation": interpretation,
        },
    }
    uncertainty_payload = dict(prediction_uncertainty or {})
    if uncertainty_payload:
        current_plan_uncertainty = dict(uncertainty_payload.get("current_plan", {}))
        recommended_plan_uncertainty = dict(uncertainty_payload.get("recommended_plan", {}))
        guardrail_summary = dict(uncertainty_payload.get("guardrail_summary", {}))
        section["uncertainty"] = {
            "current_plan_mean_std": _round(dict(current_plan_uncertainty.get("forecast", {})).get("summary", {}).get("mean_std")),
            "recommended_plan_mean_std": _round(dict(recommended_plan_uncertainty.get("forecast", {})).get("summary", {}).get("mean_std")),
            "delta_lower_bound": _round(guardrail_summary.get("delta_lower_bound")),
            "delta_upper_bound": _round(guardrail_summary.get("delta_upper_bound")),
        }
    return section


def _clinical_risk_forecast_section(clinical_risk_forecasts: Optional[Dict[str, object]]) -> List[Dict[str, object]]:
    payload = dict(clinical_risk_forecasts or {})
    current_plan = dict(payload.get("current_plan", {}))
    recommended_plan = dict(payload.get("recommended_plan", {}))
    rows: List[Dict[str, object]] = []
    for task_name, label in AUXILIARY_FORECAST_LABELS.items():
        current_task = dict(current_plan.get(task_name, {}))
        recommended_task = dict(recommended_plan.get(task_name, {}))
        if not current_task and not recommended_task:
            continue
        rows.append(
            {
                "task_name": task_name,
                "label": label,
                "current_mean": _round(current_task.get("mean")),
                "current_lower": _round(current_task.get("lower")),
                "current_upper": _round(current_task.get("upper")),
                "recommended_mean": _round(recommended_task.get("mean")),
                "recommended_lower": _round(recommended_task.get("lower")),
                "recommended_upper": _round(recommended_task.get("upper")),
            }
        )
    return rows


def _candidate_option_rows(candidate_options: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    ranked = sorted(
        [dict(option) for option in candidate_options],
        key=lambda item: (
            _safe_float(item.get("candidate_selection_score")),
            _safe_float(item.get("predicted_delta")),
            _safe_float(item.get("donor_similarity")),
        ),
        reverse=True,
    )
    rows: List[Dict[str, object]] = []
    for rank, option in enumerate(ranked[:3], start=1):
        rows.append(
            {
                "rank": rank,
                "candidate_source": str(option.get("candidate_source", "")),
                "candidate_source_label": _source_label(option.get("candidate_source", "")),
                "candidate_layer": str(option.get("candidate_layer", "")),
                "selection_score": _round(option.get("candidate_selection_score")),
                "predicted_delta": _round(option.get("predicted_delta")),
                "donor_stay_id": _safe_float(option.get("donor_stay_id"), -1.0),
                "similarity": _round(option.get("donor_similarity")),
                "guideline_compatibility": _round(option.get("donor_guideline_compatibility")),
                "penalty_sum": _round(
                    _safe_float(option.get("donor_missing_care_penalty"))
                    + _safe_float(option.get("donor_contraindication_penalty"))
                ),
                "repair_actions": _repair_labels(option.get("repair_actions", [])),
                "search_actions": _search_labels(option.get("search_actions", [])),
            }
        )
    return rows


def _selection_diagnostics(selected_candidate: Dict[str, object], guardrail: Dict[str, object]) -> Dict[str, object]:
    components = dict(selected_candidate.get("selection_components", {}))
    guardrail_summary = dict(guardrail.get("selection_summary", {}))
    conflicts = [
        SELECTION_CONFLICT_LABELS.get(str(item), str(item))
        for item in components.get("multiobjective_conflicts", [])
        if str(item).strip()
    ]
    return {
        "candidate_selection_score": _round(selected_candidate.get("candidate_selection_score")),
        "base_rule_score": _round(selected_candidate.get("stage2_base_selection_score")),
        "pre_neighborhood_score": _round(selected_candidate.get("stage3_pre_neighborhood_selection_score")),
        "multiobjective_support": _round(components.get("multiobjective_support")),
        "uncertainty_penalty": _round(components.get("uncertainty_penalty")),
        "positive_unstable_penalty": _round(components.get("positive_unstable_penalty")),
        "delta_lower_bound": _round(components.get("delta_lower_bound")),
        "neighbor_consistency": _round(guardrail_summary.get("neighbor_consistency")),
        "neighbor_exchangeability_mean": _round(guardrail_summary.get("neighbor_exchangeability_mean")),
        "neighbor_action_alignment_mean": _round(guardrail_summary.get("neighbor_action_alignment_mean")),
        "neighbor_hard_pass_rate": _round(guardrail_summary.get("neighbor_hard_pass_rate")),
        "neighbor_overlap_valid_rate": _round(guardrail_summary.get("neighbor_overlap_valid_rate")),
        "conflicts": conflicts,
        "auxiliary_reasons": [str(item) for item in guardrail_summary.get("auxiliary_reasons", []) if str(item).strip()],
    }


def _evidence_summary(
    top_donor_candidates: Sequence[Dict[str, object]],
    candidate_options: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    donors = _dedupe_top_donors(top_donor_candidates)[:3]
    donor_count = len(donors)
    mean_similarity = _mean(candidate.get("donor_similarity") for candidate in donors)
    mean_guideline = _mean(candidate.get("donor_guideline_compatibility") for candidate in donors)
    mean_state_match = _mean(candidate.get("donor_state_match") for candidate in donors)
    hard_filter_pass_rate = _mean(1.0 if _boolish(candidate.get("donor_hard_filter_valid")) else 0.0 for candidate in donors)
    overlap_pass_rate = _mean(1.0 if _boolish(candidate.get("donor_overlap_valid")) else 0.0 for candidate in donors)
    option_rows = _candidate_option_rows(candidate_options)

    if donor_count >= 2 and mean_similarity >= 0.75 and mean_guideline >= 0.55:
        consistency = "强"
    elif donor_count >= 2 and mean_similarity >= 0.5:
        consistency = "中"
    else:
        consistency = "弱"

    notes = [
        f"支持该病例判断的近邻 donor 数量为 {donor_count} 个。",
        f"近邻 donor 平均相似度为 {_round(mean_similarity)}，平均指南一致性为 {_round(mean_guideline)}。",
        f"近邻 donor 平均状态匹配度为 {_round(mean_state_match)}，硬过滤通过率为 {_round(hard_filter_pass_rate)}。",
        f"近邻 donor 轨迹重叠通过率为 {_round(overlap_pass_rate)}，整体证据一致性评估为“{consistency}”。",
    ]
    return {
        "donor_support_count": donor_count,
        "mean_similarity": _round(mean_similarity),
        "mean_guideline_compatibility": _round(mean_guideline),
        "mean_state_match": _round(mean_state_match),
        "hard_filter_pass_rate": _round(hard_filter_pass_rate),
        "overlap_pass_rate": _round(overlap_pass_rate),
        "consistency_level": consistency,
        "notes": notes,
        "top_candidate_options": option_rows,
    }


def _clinical_assessment_lines(
    query: Dict[str, object],
    current_plan_summary: Dict[str, object],
    selected_candidate: Dict[str, object],
) -> List[str]:
    active_states = _state_summary(query.get("active_state_flags", []))
    current_profile = dict(current_plan_summary.get("action_profile", {}))
    donor = dict(selected_candidate.get("donor", {}))
    dynamic_profile = dict(query.get("dynamic_profile", {}))
    lines = [
        f"当前病例的关键状态包括：{'、'.join(active_states or ['未识别'])}。",
        f"当前病程模式为 {_pattern_label(query.get('pattern_name'))} / {_trajectory_label(query.get('trajectory_name'))}，最近观测窗口为 {list(query.get('raw_context', []))}。",
    ]
    if dynamic_profile:
        lines.append(
            "最近监测行为摘要为：整体观测覆盖率约 "
            f"{_round(dynamic_profile.get('observed_ratio_mean'))}，最近窗口覆盖率约 {_round(dynamic_profile.get('recent_observed_ratio_mean'))}，"
            f"时间间隔不规则度约 {_round(dynamic_profile.get('time_irregularity'))}。"
        )
        trailing_missing = _safe_float(dynamic_profile.get("mean_trailing_missing_streak"))
        if trailing_missing > 0.0:
            lines.append(f"最近窗口仍存在连续缺失片段，平均尾部缺失步数约 {_round(trailing_missing)}。")
    care_gaps: List[str] = []
    if "脓毒症" in active_states and not _boolish(current_profile.get("treat_early_antimicrobial")):
        care_gaps.append("早期抗菌覆盖不足")
    if ("低血压" in active_states or "高乳酸" in active_states) and not _boolish(current_profile.get("treat_vasopressor")):
        care_gaps.append("血流动力学支持偏弱")
    if not _boolish(current_profile.get("exam_lactate")):
        care_gaps.append("乳酸监测不足")
    if care_gaps:
        lines.append("基于当前状态与现有干预，主要可见缺口为：" + "；".join(care_gaps) + "。")
    else:
        lines.append("现有方案已覆盖部分关键照护，但系统仍在相似病例中检索更稳妥的替代方案。")
    lines.append(
        f"当前主参考 donor 为 stay_id={_safe_float(donor.get('stay_id'), -1.0)}，病例相似度 {_round(donor.get('donor_similarity'))}，状态匹配度 {_round(donor.get('donor_state_match'))}。"
    )
    pool_tags = [str(item) for item in donor.get("donor_pool_tags", []) if str(item).strip()]
    pool_match_score = _safe_float(donor.get("donor_pool_match_score"))
    if pool_tags:
        lines.append(
            "该 donor 的场景可比性标签为："
            + "、".join(pool_tags)
            + f"，场景匹配分约 {_round(pool_match_score)}。"
        )
    return lines


def _risk_delta_sentences(clinical_risk_forecasts: Sequence[Dict[str, object]]) -> List[str]:
    lines: List[str] = []
    for row in clinical_risk_forecasts:
        current_mean = _safe_float(row.get("current_mean"))
        recommended_mean = _safe_float(row.get("recommended_mean"))
        delta = recommended_mean - current_mean
        if abs(delta) < 0.01:
            continue
        change_text = "略低" if delta < 0.0 else "略高"
        lines.append(
            f"{row.get('label', '')}在推荐方案下较当前方案{change_text}（{_round(current_mean)} -> {_round(recommended_mean)}）。"
        )
    return lines


def _short_term_trend_lines(
    predicted_outcome: Dict[str, object],
    clinical_risk_forecasts: Sequence[Dict[str, object]],
    rolling_horizon: Optional[Dict[str, object]] = None,
) -> List[str]:
    summary = dict(predicted_outcome.get("summary", {}))
    uncertainty = dict(predicted_outcome.get("uncertainty", {}))
    lines = [
        f"当前方案的未来 SOFA 均值预测为 {summary.get('current_plan_mean', 0.0)}，推荐方案为 {summary.get('recommended_plan_mean', 0.0)}。",
        f"模型单步 proxy 改善值为 {summary.get('model_proxy_delta', 0.0)}，对应解读为：{summary.get('interpretation', '')}",
    ]
    if uncertainty:
        lines.append(
            f"不确定性区间显示改善下界为 {uncertainty.get('delta_lower_bound', 0.0)}，上界为 {uncertainty.get('delta_upper_bound', 0.0)}。"
        )
    lines.extend(_risk_delta_sentences(clinical_risk_forecasts)[:3])
    rollout = dict(rolling_horizon or {})
    rollout_summary = dict(rollout.get("summary", {}))
    if rollout_summary:
        cumulative_delta = _safe_float(rollout_summary.get("discounted_cumulative_delta"))
        stable = bool(rollout_summary.get("stable_candidate_source", False))
        stability_text = "保持稳定" if stable else "发生切换"
        direction_text = "仍为正向" if cumulative_delta > 0.0 else "未显示正向"
        lines.append(
            f"两步滚动模拟的折扣累计改善为 {_round(cumulative_delta)}，短期连续决策信号{direction_text}，方案来源{stability_text}。"
        )
    return lines


def _selection_summary_lines(selection_diagnostics: Dict[str, object]) -> List[str]:
    lines: List[str] = []
    selection_score = _safe_float(selection_diagnostics.get("candidate_selection_score"))
    base_score = _safe_float(selection_diagnostics.get("base_rule_score"))
    support = _safe_float(selection_diagnostics.get("multiobjective_support"))
    penalty = _safe_float(selection_diagnostics.get("uncertainty_penalty"))
    lower_bound = _safe_float(selection_diagnostics.get("delta_lower_bound"))
    positive_unstable_penalty = _safe_float(selection_diagnostics.get("positive_unstable_penalty"))
    pre_neighborhood_score = _safe_float(selection_diagnostics.get("pre_neighborhood_score"))
    neighbor_consistency = _safe_float(selection_diagnostics.get("neighbor_consistency"))
    neighbor_exchangeability = _safe_float(selection_diagnostics.get("neighbor_exchangeability_mean"))
    neighbor_action_alignment = _safe_float(selection_diagnostics.get("neighbor_action_alignment_mean"))
    neighbor_hard_rate = _safe_float(selection_diagnostics.get("neighbor_hard_pass_rate"))
    neighbor_overlap_rate = _safe_float(selection_diagnostics.get("neighbor_overlap_valid_rate"))
    conflicts = list(selection_diagnostics.get("conflicts", []))

    lines.append(
        f"主排序分为 { _round(selection_score) }，其中基础规则分为 { _round(base_score) }，多目标支持项为 { _round(support) }，不确定性惩罚为 { _round(penalty) }。"
    )
    lines.append(
        f"在加入邻域一致性之前，候选分为 { _round(pre_neighborhood_score) }；邻域一致性并入后，最终分更新为 { _round(selection_score) }。"
    )
    lines.append(f"多目标改善下界为 { _round(lower_bound) }。")
    lines.append(
        f"近邻 donor 一致性为 { _round(neighbor_consistency) }，其中可交换性均值 { _round(neighbor_exchangeability) }，动作对齐均值 { _round(neighbor_action_alignment) }，硬过滤通过率 { _round(neighbor_hard_rate) }，overlap 通过率 { _round(neighbor_overlap_rate) }。"
    )
    if conflicts:
        lines.append("多目标冲突包括：" + "；".join(conflicts) + "。")
    if positive_unstable_penalty > 0.0:
        lines.append(f"该候选还触发了 { _round(positive_unstable_penalty) } 的“正收益但不稳定”惩罚。")
    for item in selection_diagnostics.get("auxiliary_reasons", []):
        lines.append(str(item))
    return lines


def _not_recommended_lines(
    guardrail: Dict[str, object],
    warnings: Sequence[str],
) -> List[str]:
    reasons = [str(item) for item in guardrail.get("reasons", []) if str(item).strip()]
    if reasons:
        return reasons
    if str(guardrail.get("status", "review_only")) == "recommendation_ready":
        return ["当前候选方案已通过门控，但仍需结合床旁检查、病原学结果和医师判断决定是否采用。"]
    return [str(item) for item in warnings if str(item).strip()]


def _review_points(
    guardrail: Dict[str, object],
    warnings: Sequence[str],
    detailed_actions: Sequence[Dict[str, object]],
) -> List[str]:
    points: List[str] = list(_not_recommended_lines(guardrail, warnings))
    selection_summary = dict(guardrail.get("selection_summary", {}))
    auxiliary_reasons = [str(item) for item in selection_summary.get("auxiliary_reasons", []) if str(item).strip()]
    points.extend(auxiliary_reasons)
    for row in detailed_actions:
        if str(row.get("current_status", "")) != str(row.get("recommended_status", "")):
            points.append(f"请确认“{row.get('domain', '')}”对应的床旁条件是否支持从当前方案调整到建议方案。")
    points.append("请将本结果与病原学、容量反应性、器官支持强度和最新化验结果一起复核。")
    deduped: List[str] = []
    for point in points:
        if point not in deduped:
            deduped.append(point)
    return deduped


def _method_boundary_lines() -> List[str]:
    return [
        "该系统本质上是“相似病例检索 + 候选干预模拟 + 风险重排”，不是严格的因果疗效估计器。",
        "模型输出反映的是在当前训练分布下的短期 proxy 预测差异，不等同于真实临床获益。",
        "如果后续引入 OPE、DR 或时间变化混杂校正，应单独给出实验设计、适用前提和边界说明。",
    ]


def _actionability_status_label(status: object) -> str:
    return "可作为建议候选" if str(status or "") == "recommendation_ready" else "仅供医生复核"


def _actionability_note(status: object) -> str:
    if str(status or "") == "recommendation_ready":
        return "已通过当前门控，可作为机器生成的候选方案供医生进一步确认。"
    return "未通过当前门控，不建议直接采纳，只能作为复核参考。"


def _report_text(
    patient_summary: Dict[str, object],
    recommended_plan: Dict[str, object],
    rationale: Sequence[str],
    warnings: Sequence[str],
) -> str:
    lines = [
        f"Patient: stay_id={patient_summary.get('stay_id')} ({patient_summary.get('series_name', '')})",
        "Active states: " + ", ".join(patient_summary.get("active_states", []) or ["none"]),
        f"Confidence: {recommended_plan.get('confidence', '低')}",
        "Recommended plan:",
    ]
    lines.extend(f"- {line}" for line in recommended_plan.get("care_actions", []))
    lines.append("Rationale:")
    lines.extend(f"- {line}" for line in rationale)
    lines.append("Warnings:")
    lines.extend(f"- {line}" for line in warnings)
    return "\n".join(lines)


def render_counterfactual_case_report(case_detail: Dict[str, object]) -> Dict[str, object]:
    query = dict(case_detail.get("query", {}))
    selected_candidate = dict(case_detail.get("selected_candidate", {}))
    recommended_plan = {
        "candidate_source": str(selected_candidate.get("candidate_source", "")),
        "candidate_source_label": _source_label(selected_candidate.get("candidate_source", "")),
        "confidence": _confidence_level(selected_candidate),
        "care_actions": [
            row["recommended_status"]
            for row in _intervention_detail_rows(
                dict(query.get("current_plan", {})),
                dict(selected_candidate.get("plan_summary", {})),
                selected_candidate.get("repair_actions", []),
            )
        ],
        "repair_actions": _repair_labels(selected_candidate.get("repair_actions", [])),
        "search_actions": _search_labels(selected_candidate.get("search_actions", [])),
        "predicted_delta": _round(selected_candidate.get("predicted_delta")),
    }
    patient_summary = {
        "stay_id": _safe_float(query.get("stay_id"), -1.0),
        "series_name": str(query.get("series_name", "")),
        "pattern_name": _pattern_label(query.get("pattern_name", "")),
        "trajectory_name": _trajectory_label(query.get("trajectory_name", "")),
        "experience_label": int(_safe_float(query.get("experience_label"), -1.0)),
        "active_states": _state_summary(query.get("active_state_flags", [])),
        "recent_context": [float(value) for value in query.get("raw_context", [])],
    }
    rationale = _rationale_lines(case_detail)
    warnings = _warning_lines(case_detail)
    report = {
        "patient_summary": patient_summary,
        "top_similar_cases": _top_similar_case_rows(case_detail.get("top_donor_candidates", [])),
        "recommended_plan": recommended_plan,
        "rationale": rationale,
        "warnings": warnings,
    }
    report["report_text"] = _report_text(patient_summary, recommended_plan, rationale, warnings)
    return report


def render_physician_case_report(
    case_detail: Dict[str, object],
    base_prediction: Optional[Sequence[object]] = None,
    factual_prediction: Optional[Sequence[object]] = None,
    counterfactual_prediction: Optional[Sequence[object]] = None,
    guardrail: Optional[Dict[str, object]] = None,
    prediction_uncertainty: Optional[Dict[str, object]] = None,
    clinical_risk_forecasts: Optional[Dict[str, object]] = None,
    rolling_horizon: Optional[Dict[str, object]] = None,
    step_hours: int = 6,
) -> Dict[str, object]:
    query = dict(case_detail.get("query", {}))
    selected_candidate = dict(case_detail.get("selected_candidate", {}))
    donor = dict(selected_candidate.get("donor", {}))
    current_plan_summary = dict(query.get("current_plan", {}))
    recommended_plan_summary = dict(selected_candidate.get("plan_summary", {}))
    repair_actions = _repair_labels(selected_candidate.get("repair_actions", []))
    search_actions = _search_labels(selected_candidate.get("search_actions", []))
    guardrail = dict(guardrail or {})

    detailed_actions = _intervention_detail_rows(
        current_plan_summary=current_plan_summary,
        recommended_plan_summary=recommended_plan_summary,
        repair_actions=selected_candidate.get("repair_actions", []),
    )
    predicted_outcome = _predicted_outcome_section(
        base_prediction=base_prediction or [],
        factual_prediction=factual_prediction or [],
        counterfactual_prediction=counterfactual_prediction or [],
        predicted_delta=selected_candidate.get("predicted_delta"),
        prediction_uncertainty=prediction_uncertainty,
        step_hours=step_hours,
    )
    risk_rows = _clinical_risk_forecast_section(clinical_risk_forecasts)
    warnings = _warning_lines(case_detail)
    evidence_summary = _evidence_summary(
        top_donor_candidates=case_detail.get("top_donor_candidates", []),
        candidate_options=case_detail.get("candidate_options", []),
    )
    selection_diagnostics = _selection_diagnostics(selected_candidate, guardrail)

    physician_view = {
        "case_overview": {
            "stay_id": _safe_float(query.get("stay_id"), -1.0),
            "series_name": str(query.get("series_name", "")),
            "pattern_name": _pattern_label(query.get("pattern_name", "")),
            "trajectory_name": _trajectory_label(query.get("trajectory_name", "")),
            "active_states": _state_summary(query.get("active_state_flags", [])),
            "recent_context": [float(value) for value in query.get("raw_context", [])],
            "dynamic_profile": dict(query.get("dynamic_profile", {})),
        },
        "recommended_intervention_plan": {
            "actionability": str(guardrail.get("status", "review_only")),
            "actionability_label": _actionability_status_label(guardrail.get("status", "review_only")),
            "actionability_note": _actionability_note(guardrail.get("status", "review_only")),
            "candidate_source": str(selected_candidate.get("candidate_source", "")),
            "candidate_source_label": _source_label(selected_candidate.get("candidate_source", "")),
            "confidence": _confidence_level(selected_candidate),
            "candidate_layer": str(selected_candidate.get("candidate_layer", "safety")),
            "candidate_anchor_relation": str(selected_candidate.get("candidate_anchor_relation", "donor_centered")),
            "strategy_family": _strategy_family_label(selected_candidate.get("candidate_strategy_family", "")),
            "parameter_profile": dict(selected_candidate.get("candidate_parameter_profile", {})),
            "parameter_summary": _parameter_profile_summary(dict(selected_candidate.get("candidate_parameter_profile", {}))),
            "safety_rationale": [str(item) for item in selected_candidate.get("candidate_safety_rationale", []) if str(item).strip()],
            "derived_from_similar_patient": {
                "stay_id": _safe_float(donor.get("stay_id"), -1.0),
                "similarity": _round(donor.get("donor_similarity")),
                "kg_similarity": _round(donor.get("donor_kg_similarity")),
                "guideline_compatibility": _round(donor.get("donor_guideline_compatibility")),
                "state_match": _round(donor.get("donor_state_match")),
                "total_score": _round(donor.get("donor_total_score")),
                "pool_match_score": _round(donor.get("donor_pool_match_score")),
                "pool_tags": [str(item) for item in donor.get("donor_pool_tags", []) if str(item).strip()],
            },
            "repair_actions": repair_actions,
            "search_actions": search_actions,
            "detailed_actions": detailed_actions,
            "selection_diagnostics": selection_diagnostics,
        },
        "clinical_assessment": _clinical_assessment_lines(
            query=query,
            current_plan_summary=current_plan_summary,
            selected_candidate=selected_candidate,
        ),
        "similar_patient_comparison": _top_similar_case_rows(case_detail.get("top_donor_candidates", [])),
        "predicted_outcome_for_current_patient": predicted_outcome,
        "clinical_risk_forecasts": risk_rows,
        "short_term_trend_assessment": _short_term_trend_lines(
            predicted_outcome=predicted_outcome,
            clinical_risk_forecasts=risk_rows,
            rolling_horizon=rolling_horizon,
        ),
        "not_recommended_reasons": _not_recommended_lines(guardrail, warnings),
        "review_points": _review_points(guardrail, warnings, detailed_actions),
        "selection_rationale": _rationale_lines(case_detail),
        "evidence_strength": evidence_summary,
        "candidate_option_comparison": evidence_summary.get("top_candidate_options", []),
        "selection_summary": _selection_summary_lines(selection_diagnostics),
        "method_boundary": _method_boundary_lines(),
        "rolling_horizon": dict(rolling_horizon or {}),
    }
    physician_view["markdown"] = render_physician_markdown(physician_view)
    return physician_view


def render_physician_markdown(physician_view: Dict[str, object]) -> str:
    case_overview = dict(physician_view.get("case_overview", {}))
    plan = dict(physician_view.get("recommended_intervention_plan", {}))
    derived = dict(plan.get("derived_from_similar_patient", {}))
    selection_diagnostics = dict(plan.get("selection_diagnostics", {}))
    predicted = dict(physician_view.get("predicted_outcome_for_current_patient", {}))
    summary = dict(predicted.get("summary", {}))
    uncertainty = dict(predicted.get("uncertainty", {}))
    evidence = dict(physician_view.get("evidence_strength", {}))
    rollout = dict(physician_view.get("rolling_horizon", {}))
    rollout_summary = dict(rollout.get("summary", {}))

    lines = [
        "# 相似患者干预建议单",
        "",
        "## 1. 病情判断",
        f"- 患者 stay_id：{case_overview.get('stay_id')}",
        f"- 序列标识：{case_overview.get('series_name', '')}",
        f"- 当前病程模式：{case_overview.get('pattern_name', '')} / {case_overview.get('trajectory_name', '')}",
        f"- 当前关键状态：{'、'.join(case_overview.get('active_states', []) or ['无'])}",
        f"- 最近观测窗口：{case_overview.get('recent_context', [])}",
    ]
    for item in physician_view.get("clinical_assessment", []):
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## 2. 干预建议",
            f"- 输出状态：{plan.get('actionability_label', plan.get('actionability', ''))}",
            f"- 使用建议：{plan.get('actionability_note', '')}",
            f"- 推荐方案来源：{plan.get('candidate_source_label', '')}",
            f"- 推荐方案置信度：{plan.get('confidence', '')}",
            f"- 候选生成关系：{plan.get('candidate_anchor_relation', '')}",
            f"- 候选策略族：{plan.get('strategy_family', '')}",
            f"- 主参考相似患者：stay_id={derived.get('stay_id')}, 相似度={derived.get('similarity')}, 指南一致性={derived.get('guideline_compatibility')}, 状态匹配={derived.get('state_match')}",
            f"- KG 修补动作：{'；'.join(plan.get('repair_actions', []) or ['无'])}",
            f"- 模板搜索动作：{'；'.join(plan.get('search_actions', []) or ['无'])}",
            f"- 参数化偏移：{plan.get('parameter_summary', '')}",
            f"- 候选主排序分：{selection_diagnostics.get('candidate_selection_score', '')}",
            f"- 多目标支持项 / 不确定性惩罚：{selection_diagnostics.get('multiobjective_support', '')} / {selection_diagnostics.get('uncertainty_penalty', '')}",
            "",
            "| 干预域 | 当前方案 | 推荐方案 | 变化说明 |",
            "|---|---|---|---|",
        ]
    )
    for item in plan.get("safety_rationale", []):
        lines.append(f"- 安全偏离说明：{item}")
    for row in plan.get("detailed_actions", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("domain", "")),
                    str(row.get("current_status", "")),
                    str(row.get("recommended_status", "")),
                    str(row.get("change_summary", "")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 3. 短期趋势判断",
            f"- 当前方案未来 SOFA 均值：{summary.get('current_plan_mean', '')}",
            f"- 推荐方案未来 SOFA 均值：{summary.get('recommended_plan_mean', '')}",
            f"- 推荐减当前：{summary.get('recommended_minus_current', '')}",
            f"- 单步模型 proxy 改善：{summary.get('model_proxy_delta', '')}",
            f"- 解释：{summary.get('interpretation', '')}",
        ]
    )
    if uncertainty:
        lines.extend(
            [
                f"- 当前方案预测标准差：{uncertainty.get('current_plan_mean_std', '')}",
                f"- 推荐方案预测标准差：{uncertainty.get('recommended_plan_mean_std', '')}",
                f"- 改善下界：{uncertainty.get('delta_lower_bound', '')}",
                f"- 改善上界：{uncertainty.get('delta_upper_bound', '')}",
            ]
        )
    if rollout_summary:
        lines.extend(
            [
                f"- 两步滚动折扣累计改善：{_round(rollout_summary.get('discounted_cumulative_delta'))}",
                f"- 两步滚动原始累计改善：{_round(rollout_summary.get('raw_cumulative_delta'))}",
                f"- 方案来源是否稳定：{'是' if bool(rollout_summary.get('stable_candidate_source', False)) else '否'}",
            ]
        )
    for item in physician_view.get("short_term_trend_assessment", []):
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "| 时间点 | 当前方案预测 | 推荐方案预测 |",
            "|---|---:|---:|",
        ]
    )
    current_rows = list(predicted.get("current_plan_prediction", []))
    recommended_rows = list(predicted.get("recommended_plan_prediction", []))
    for current_row, recommended_row in zip(current_rows, recommended_rows):
        lines.append(
            f"| {current_row.get('horizon', '')} | {current_row.get('predicted_value', '')} | {recommended_row.get('predicted_value', '')} |"
        )

    risk_rows = list(physician_view.get("clinical_risk_forecasts", []))
    if risk_rows:
        lines.extend(
            [
                "",
                "| 辅助风险指标 | 当前方案 | 推荐方案 |",
                "|---|---:|---:|",
            ]
        )
        for row in risk_rows:
            current_text = f"{row.get('current_mean', '')} [{row.get('current_lower', '')}, {row.get('current_upper', '')}]"
            recommended_text = f"{row.get('recommended_mean', '')} [{row.get('recommended_lower', '')}, {row.get('recommended_upper', '')}]"
            lines.append(f"| {row.get('label', '')} | {current_text} | {recommended_text} |")

    lines.extend(
        [
            "",
            "## 4. 为什么暂不建议直接采纳",
        ]
    )
    for item in physician_view.get("selection_summary", []):
        lines.append(f"- {item}")
    for item in physician_view.get("not_recommended_reasons", []):
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## 5. 相似患者证据",
            f"- 支持 donor 数量：{evidence.get('donor_support_count', 0)}",
            f"- 平均相似度：{evidence.get('mean_similarity', 0.0)}",
            f"- 平均指南一致性：{evidence.get('mean_guideline_compatibility', 0.0)}",
            f"- 平均状态匹配度：{evidence.get('mean_state_match', 0.0)}",
            f"- 证据一致性：{evidence.get('consistency_level', '')}",
        ]
    )
    for item in evidence.get("notes", []):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "| 排名 | stay_id | 相似度 | KG相似度 | 指南一致性 | 状态匹配 | 总分 | 硬过滤 | 重叠过滤 | 方案要点 |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in physician_view.get("similar_patient_comparison", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("rank", "")),
                    str(row.get("stay_id", "")),
                    str(row.get("similarity", "")),
                    str(row.get("kg_similarity", "")),
                    str(row.get("guideline_compatibility", "")),
                    str(row.get("state_match", "")),
                    str(row.get("total_score", "")),
                    str(row.get("hard_filter_status", "")),
                    str(row.get("overlap_status", "")),
                    str(row.get("plan_highlight", "")),
                ]
            )
            + " |"
        )

    candidate_options = list(physician_view.get("candidate_option_comparison", []))
    if candidate_options:
        lines.extend(
            [
                "",
                "## 6. Top-3 备选方案对照",
                "| 排名 | 方案来源 | 层级 | 选择分数 | proxy改善 | donor | 相似度 | 指南一致性 | 惩罚和 | 修补/搜索动作 |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in candidate_options:
            action_text = "；".join(row.get("repair_actions", []) + row.get("search_actions", [])) or "无"
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("rank", "")),
                        str(row.get("candidate_source_label", "")),
                        str(row.get("candidate_layer", "")),
                        str(row.get("selection_score", "")),
                        str(row.get("predicted_delta", "")),
                        str(row.get("donor_stay_id", "")),
                        str(row.get("similarity", "")),
                        str(row.get("guideline_compatibility", "")),
                        str(row.get("penalty_sum", "")),
                        action_text,
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## 7. 医生复核重点",
        ]
    )
    for item in physician_view.get("review_points", []):
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## 8. 方法边界说明",
        ]
    )
    for item in physician_view.get("method_boundary", []):
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## 9. 选择理由",
        ]
    )
    for item in physician_view.get("selection_rationale", []):
        lines.append(f"- {item}")
    return "\n".join(lines)
