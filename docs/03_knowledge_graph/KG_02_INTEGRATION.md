# KG 与 eICU / 模型接入

## 1. eICU 变量如何进入 KG 层

把每个病例窗口映射成一组病例级 KG 特征。

核心文件：

- [eicu_to_kg_node_mapping.csv](/input/knowledge/06_schema_design/eicu_to_kg_node_mapping.csv)
- [kg_integration.py](/src/kg_integration.py)

典型映射包括：

- `sepsis3_label -> kg_state_sepsis`
- `septic_shock_label_operational -> kg_state_septic_shock`
- `MAP < 65 -> kg_state_hypotension`
- `lactate >= 2 -> kg_state_high_lactate`
- `3 小时内抗菌药 -> kg_treat_early_antimicrobial`
- `有升压药 -> kg_treat_vasopressor`

## 2. kg_feature_vector 是什么

`kg_feature_vector` 是窗口级知识特征向量，不是图神经网络嵌入。

它来自两部分：

- 状态与检查相关 flag
- 基于指南的 alignment 分数

这些特征会被写进 [ForecastSample](/src/tsf_data.py)：

- `kg_features`
- `metadata["kg_flags"]`
- `metadata["kg_guideline_alignment"]`

## 3. 进入模型的位置

当前 KG 主要在两个位置接入。

第一处是 patient state 表示：

- `kg_features` 会拼进 `patient_static`
- 因此它会进入 encoder，影响病例状态 embedding

第二处是 counterfactual donor 约束：

- donor 选择时会显式计算 `kg_similarity`
- 还会看 `guideline_compatibility`
- 还会看 `state_match`
- 还会看缺失护理与禁忌惩罚

## 4. memory bank 和 intervention store 的关系

这两个概念要分开看。

`memory bank` 负责存窗口级经验状态：

- pattern
- trajectory
- experience

`intervention store` 负责存可迁移干预模板：

- `intervention_static`
- `intervention_sequence`
- donor 对应的患者 embedding
- donor 对应的 KG 特征和 metadata

所以当前反事实路径不是直接从 memory bank 取干预，而是：

1. memory 先缩小 donor 检索范围
2. intervention store 再提供具体 donor 干预

## 5. 新增的两层接入

本轮已经新增两层机制。

第一层是 `knowledge-safe write`：

- 写入 memory bank / intervention store 前先算知识质量
- 低质量条目会降低 `write_confidence`

第二层是 `KG-guided candidate generation`：

- 不再只评估原始 donor 干预
- 可以额外评估按 KG 规则最小修复后的干预候选

## 6. 对应代码入口

- [tsf_data.py](/src/tsf_data.py)
- [kg_integration.py](/src/kg_integration.py)
- [manifold_forecasting_trainer.py](/src/manifold_forecasting_trainer.py)
- [run_forecasting_experiment.py](/run_forecasting_experiment.py)
