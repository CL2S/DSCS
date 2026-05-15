# Memory MVP Project
## 2026-04 Stage-1/2 Update

本轮在反事实预测链路中补了两类能力。

1. `knowledge-safe write + donor hard filter`
   - memory bank 和 intervention store 在写入前会计算 `knowledge_quality_score`、`knowledge_feasibility_score`、`write_confidence`
   - donor 检索时新增硬约束，优先过滤明显缺失关键救治的候选

2. `KG-guided candidate generation`
   - 反事实阶段新增 `generated_best` 模式
   - 该模式会同时评估原始 donor 干预和 `generated_kg_repaired` 候选
   - `generated_kg_repaired` 会按 sepsis / shock / lactate 相关规则对 donor 干预做最小修复，再交给模型比较效果

关于 memory bank 的训练与保留逻辑，当前代码是：

- 训练开始前先构建一次初始 online memory bank
- 每个 epoch 结束后都会基于当前 encoder 重新构建一次 online memory bank
- 每个 epoch 同时做验证，记录 `best_state`
- 训练结束后先恢复 `best_state`，再重建最终 online memory bank

所以最终留下来的不是“最后一个 epoch 的 memory bank”，而是“验证集最优模型状态对应的 memory bank”。

本仓库是一个围绕“记忆增强时序建模”构建的研究与工程系统，目标不是实现单一模型，而是把以下几类能力放到同一框架中：

- 时序窗口的结构化表示
- 基于历史经验的检索与复用
- 患者状态与干预信息的分离建模
- 反事实 donor 检索与干预迁移
- 脓毒症知识图谱与临床规则的接入

当前系统包含三条主线：

- `tabular`：面向表格分类任务的动态记忆模型
- `temporal`：面向 ICU / Sepsis 时序任务的 patient 级与 window 级实验
- `forecasting`：面向多步预测与反事实分析的主线系统，也是当前最完整的一条线

如果你第一次进入仓库，建议按这个顺序阅读：

1. [DOC_INDEX.md](docs/DOC_INDEX.md)
2. [SYSTEM_ARCHITECTURE_DETAILED_GUIDE.md](docs/01_core/SYSTEM_ARCHITECTURE_DETAILED_GUIDE.md)
3. [PERSISTENT_EXPERIENCE_MEMORY_GUIDE.md](docs/01_core/PERSISTENT_EXPERIENCE_MEMORY_GUIDE.md)
4. [EICU_FORECASTING_AND_EVALUATION_GUIDE.md](docs/02_eicu/EICU_FORECASTING_AND_EVALUATION_GUIDE.md)
5. [Sepsis_KG_MVP_详细说明.md](docs/03_knowledge_graph/Sepsis_KG_MVP_详细说明.md)

主要入口脚本：

- [run_experiment.py](run_experiment.py)
- [run_temporal_experiment.py](run_temporal_experiment.py)
- [run_forecasting_experiment.py](run_forecasting_experiment.py)

核心源码目录：

- `src/`
- `input/`
- `output/`
- `docs/`

知识图谱相关脚本与资源位于：

- `input/knowledge/`
- [build_sepsis_kg_mvp.py](input/knowledge/scripts/build_sepsis_kg_mvp.py)
- [build_sepsis_kg_guideline_enhanced.py](input/knowledge/scripts/build_sepsis_kg_guideline_enhanced.py)

本仓库文档现在以“系统说明”为主，不再把每一轮修改过程写成主文档。历史材料只保留在归档目录，不作为当前方法介绍的主体。
