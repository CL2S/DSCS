# 可直接执行的改造清单（函数级别）

> 目标：在不推翻现有 `AdvancedExperienceMemoryBank` 架构前提下，融合《基础经验.md》中的阈值知识，增强检索准确性与高危稳健性。

## 1) `advanced_experience_memory.py`

### A. 新增阈值解析函数
- [x] 新增 `_parse_series_values(text, key)`：解析描述文本中的序列数值。
- [x] 新增 `_extract_clinical_state(input_description, intervention, result=None)`：抽取 qSOFA / MAP / 乳酸 / NE 剂量等阈值状态。
- [x] 新增 `_guideline_consistency(a, b)`：计算查询与历史病例的阈值一致性分。

### B. 扩展 Episodic 记忆结构
- [x] 在 `EpisodicMemory` 增加 `clinical_state` 字段（默认空字典，向后兼容旧数据）。
- [x] 在 `_load()` 里兼容旧存量数据（若缺少 `clinical_state` 自动补空）。

### C. 写入阶段增强
- [x] 在 `add_experience_from_result()` 中写入阈值状态到 `clinical_state`。
- [x] 将关键阈值标签（如 `qsofa_high`、`lactate_high`）加入 `tags`。

### D. 规则巩固增强
- [x] 在 `_consolidate_rules()` 中加入 `clinical_threshold_profile` 与 `threshold_coverage`。
- [x] 规则置信度改为 `质量 + 阈值覆盖度` 组合。

### E. 检索与评分增强
- [x] `get_recommendations()` 增加 `q_state` 提取。
- [x] 原 `symbolic` 拆分为 `intervention匹配 + 阈值一致性`。
- [x] 综合得分改为：`dense + symbolic + guideline + quality + recency`。
- [x] 返回 `score_breakdown` 与 `clinical_state`，增强可解释性。

### F. 时间衰减增强
- [x] `_recency_weight()` 从固定半衰期改为风险等级自适应半衰期。

---

## 2) `experience_integration.py`

### G. 高危置信度门控
- [x] 新增 `_clinical_risk_gate(input_description, intervention)`：检测 qSOFA>=2、MAP<65、lactate>4、NE>0.1。
- [x] 在 `adjust_model_confidence()` 中加入上调限幅（高危时经验上调最多 +0.08）。

---

## 3) 兼容性与回退
- [x] 保持 `get_recommendations(input_description, intervention, top_k=5)` 既有调用签名。
- [x] 保持 `ExperienceIntegration` 现有调用链不变。
- [x] 保持新旧经验库自动回退机制。

---

## 4) 建议的实验对照（下一步）
- [ ] A/B：仅原混合记忆 vs 融合阈值先验后的记忆。
- [ ] 指标：Top-K召回一致性、risk-level匹配率、高危病例误上调率。
- [ ] 子集：按 `qSOFA>=2`、`lactate>2/4`、`NE>0.1` 分层评估。
