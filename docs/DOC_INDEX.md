# 文档索引

> 最后更新：2026-05-12

---

## 活跃顶层文档

| 文件 | 说明 |
|---|---|
| [00_ROADMAP.md](00_ROADMAP.md) | **修改路线图**：Phase A-D 四阶段计划，每项任务的方案、文件、收益预估和完成状态 |
| [01_DIAGNOSIS.md](01_DIAGNOSIS.md) | **硬伤诊断报告**：经验记忆方法完整架构、R0-S2 实验历程、7 项硬伤、能力边界 |
| [02_USER_ANALYSIS.md](02_USER_ANALYSIS.md) | **方法缺陷分析（正式稿）**：12 项问题—成因—影响—改进方案—预期收益 |
| [03_LLM_AGENT_PLAN.md](03_LLM_AGENT_PLAN.md) | **LLM Clinical Supervisor 方案**：用 LLM 做 Memory Gate / Retrieval Filter / Transition Utility，突破纯数值架构天花板 |

---

## 分类导航

### `01_core/` — 核心系统

- [SYSTEM_ARCHITECTURE_DETAILED_GUIDE.md](01_core/SYSTEM_ARCHITECTURE_DETAILED_GUIDE.md) — 系统架构、模块关系、主链路
- [PERSISTENT_EXPERIENCE_MEMORY_GUIDE.md](01_core/PERSISTENT_EXPERIENCE_MEMORY_GUIDE.md) — Memory 存储结构、重建逻辑、persistent 层
- [EXPERIMENT_AND_OPERATION_GUIDE.md](01_core/EXPERIMENT_AND_OPERATION_GUIDE.md) — 实验运行、命令参数、输出解读

### `02_eicu/` — eICU 专项

- [EICU_FORECASTING_AND_EVALUATION_GUIDE.md](02_eicu/EICU_FORECASTING_AND_EVALUATION_GUIDE.md) — 数据集、样本构造、预测任务、评估框架
- [EICU_COUNTERFACTUAL_SIMILAR_PATIENT_METHOD_GUIDE_20260413.md](02_eicu/EICU_COUNTERFACTUAL_SIMILAR_PATIENT_METHOD_GUIDE_20260413.md) — 九条修改后的方法总览、创新点、局限性
- [EICU_CURRENT_ISSUES_AND_REMEDIATION_STATUS_20260410.md](02_eicu/EICU_CURRENT_ISSUES_AND_REMEDIATION_STATUS_20260410.md) — 当前问题与修正状态
- [EICU_MEMORY_KG_REMEDIATION_PLAN_20260409.md](02_eicu/EICU_MEMORY_KG_REMEDIATION_PLAN_20260409.md) — Memory + KG 修正计划
- [EICU_NINE_STEP_MODIFICATION_TRACKER_20260413.md](02_eicu/EICU_NINE_STEP_MODIFICATION_TRACKER_20260413.md) — 九步修改追踪
- [EICU_NEXT_STAGE_MODIFICATION_TRACKER_20260415.md](02_eicu/EICU_NEXT_STAGE_MODIFICATION_TRACKER_20260415.md) — 后续修改追踪

### `03_knowledge_graph/` — 知识图谱

- [KG_01_OVERVIEW.md](03_knowledge_graph/KG_01_OVERVIEW.md) — KG 本体、LMKG 底座、产物
- [KG_02_INTEGRATION.md](03_knowledge_graph/KG_02_INTEGRATION.md) — eICU→KG 映射、病例级特征、接入点
- [KG_03_COUNTERFACTUAL_EVAL.md](03_knowledge_graph/KG_03_COUNTERFACTUAL_EVAL.md) — Donor 检索、评分、评估指标
- [KG_04_STATUS_AND_PLAN.md](03_knowledge_graph/KG_04_STATUS_AND_PLAN.md) — 当前状态、局限、下一步
- [KG_05_METHOD_SUMMARY_AND_REVISION_PLAN.md](03_knowledge_graph/KG_05_METHOD_SUMMARY_AND_REVISION_PLAN.md)
- [KG_06_AUTO_ITERATION_LOG.md](03_knowledge_graph/KG_06_AUTO_ITERATION_LOG.md)
- [KG_07_TRANSITION_UTILITY_ACTION_ITERATION.md](03_knowledge_graph/KG_07_TRANSITION_UTILITY_ACTION_ITERATION.md)
- [KG_08_EARLY_FUSION_LARGE_SCALE_VALIDATION.md](03_knowledge_graph/KG_08_EARLY_FUSION_LARGE_SCALE_VALIDATION.md)

### `03_paper_materials/` — 论文素材

- [EICU_PAPER_MASTER_OUTLINE_20260417.md](03_paper_materials/EICU_PAPER_MASTER_OUTLINE_20260417.md) — 论文大纲
- [README.md](03_paper_materials/README.md)

### `04_frcdg_fusion/` — FRCDG 融合

- [FRCDG_MEMORY_EICU_FUSION_PLAN_20260417.md](04_frcdg_fusion/FRCDG_MEMORY_EICU_FUSION_PLAN_20260417.md)
- [README.md](04_frcdg_fusion/README.md)

### `05_experience_memory/` — 经验记忆

- [EXPERIENCE_MEMORY_DETAILED_GUIDE_20260427.md](05_experience_memory/EXPERIENCE_MEMORY_DETAILED_GUIDE_20260427.md) — 经验记忆库详细说明
- [README.md](05_experience_memory/README.md)

### `90_assets/` — 素材资源

图示、公式源码、论文插图等。

### `99_archive/` — 历史归档

| 文件 | 说明 |
|---|---|
| `modifyv3_original_r0_r2_plan.md` | R0-R2 原始修改方案（2026-05-10） |
| `modifyv4_draft.md` | modifyv4 草稿 |
| `sepsis_kg_mvp_overview.md` | Sepsis KG MVP 概述（旧版） |
| `eicu_counterfactual_diagnosis.md` | eICU 反事实方法诊断（旧版，已被 01_DIAGNOSIS.md 取代） |
| `frcdg_refactoring_plan_v3.md` | FRCDG 代码重构计划 v3 |
