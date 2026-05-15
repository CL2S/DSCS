# 背景

随着 eICU 预测框架逐步引入经验记忆、持久化语义原型、显式知识图谱支路与 epoch 级误差反馈，`docs/02_eicu` 下先后累积了多份补充说明文档。这些文档记录了各轮修改的局部背景与验证结果，但也带来了两个问题：

1. 问题、证据、拟解决方案与修改进度被分散在多份 follow-up 文档中，难以快速判断当前系统的真实瓶颈。
2. 方法主文容易再次被局部迭代记录污染，不利于后续按论文结构组织材料。

因此，本文件的定位是将当前阶段的主要问题、对应证据、拟解决方案和修改进度统一收敛到一处，使 `docs/02_eicu` 重新回到“方法主文 + 数据说明 + 问题进展总表”的简洁结构。

# 本轮修改目标

1. 新增一份统一的问题与进展总表文档，系统总结当前模型的真实瓶颈。
2. 为每个问题补充拟解决方案与当前修改进度，避免只给出批评、不提供后续路径。
3. 压缩 `docs/02_eicu` 目录，将与方法主文无关的多份跟踪型文档合并或删除。
4. 保留最少且最清晰的文档集合，方便后续论文写作与实验追踪。

# 修改内容

## 1. 新增问题、方案与进度总表

新增 `EICU_CURRENT_ISSUES_AND_REMEDIATION_STATUS_20260410.md`，统一汇总以下三类内容：

1. 当前问题本身，即哪些模块已经被证明存在结构性缺陷。
2. 每个问题的实验证据，即结论来自哪类结果文件而非主观印象。
3. 每个问题对应的拟解决方案与当前进度，即已经完成、部分完成还是尚未启动。

## 2. 合并并下线零散补充文档

以下 6 份文档的核心信息已经并入本文件，不再单独保留：

1. `EICU_EVALUATION_LINES_REFACTOR_20260409.md`
2. `EICU_EVALUATION_LINES_VALIDATION_SUPPLEMENT_20260409.md`
3. `EICU_PERSISTENT_SEMANTIC_REMEDIATION_SUPPLEMENT_20260409.md`
4. `EICU_EPOCH_FEEDBACK_INTERPRETABILITY_20260409.md`
5. `EICU_EPOCH_FEEDBACK_INTERPRETABILITY_MIDSCALE_20260409.md`
6. `EICU_EPOCH_FEEDBACK_REGULARIZATION_FOLLOWUP_20260409.md`

这些文档此前分别记录评估线拆分、persistent semantic 修复、epoch feedback 闭环与 regularization follow-up。其历史价值仍然存在，但继续保留为独立主文档只会增加目录噪声。

## 3. 保留的文档结构

压缩后，`docs/02_eicu` 目录仅保留以下三份核心文档：

1. `EICU_MEMORY_KG_REMEDIATION_PLAN_20260409.md`
   角色：方法主文，偏论文写法。
2. `EICU_FORECASTING_AND_EVALUATION_GUIDE.md`
   角色：数据与评估背景说明。
3. `EICU_CURRENT_ISSUES_AND_REMEDIATION_STATUS_20260410.md`
   角色：当前问题、证据、拟解决方案与修改进度总表。

# 影响范围

本轮修改仅影响文档结构，不改动模型代码与实验脚本。其影响主要体现在：

1. 文档层级从“多轮 follow-up 并列堆积”改为“主文 + 数据指南 + 进展总表”。
2. 问题讨论与方法描述被明确分离，便于后续写作时分别抽取 `Methods` 与 `Limitations/Discussion`。
3. 读者不再需要跨多份补充说明拼接当前状态，只需查看本文件即可掌握现状。

# 运行方式

本文件中的判断主要基于以下结果文件与现有代码实现：

```bash
memory_mvp_project/output/analysis/evaluation_line_isolation/summary.json
memory_mvp_project/output/analysis/kg_revision_matrix/summary.json
memory_mvp_project/output/analysis/persistent_semantic_matrix_v3/summary.json
memory_mvp_project/output/analysis/epoch_feedback_interpretability_midscale_v3/summary.json
memory_mvp_project/src/manifold_forecasting_trainer.py
memory_mvp_project/src/memory_manager.py
memory_mvp_project/src/persistent_memory_store.py
```

其中：

1. `evaluation_line_isolation` 用于验证 factual forecasting 与 donor ranking 已经被干净隔离。
2. `kg_revision_matrix` 用于判断显式 KG 支路与静态拼接策略的效果。
3. `persistent_semantic_matrix_v3` 用于判断 persistent semantic 修复后的边际收益。
4. `epoch_feedback_interpretability_midscale_v3` 用于判断 epoch feedback 与 regularization 在中规模设置下的真实表现。

# 结果与分析

## 一、当前问题总表

| 编号 | 当前问题 | 关键证据 | 拟解决方案 | 当前进度 |
|---|---|---|---|---|
| 1 | `factual forecasting` 的净收益仍不稳定，解释性改动尚未稳定转化为 held-out 误差改善 | 在 `epoch_feedback_interpretability_midscale_v3` 中，best checkpoint 下 baseline 与 feedback 的 `hybrid_mae`、`hybrid_rmse`、`improvement_mae` 完全一致；final epoch 下 feedback 的 `hybrid_mae` 还略差于 baseline | 把训练目标从“只做可解释性约束”改成“可解释性约束 + utility-aware branch optimization”；扩大中规模实验预算；加入更严格的 slice-based factual 对照 | `部分完成`。评估框架已修好，但训练闭环还没有带来稳定 factual gain |
| 2 | `epoch feedback` 当前更像防崩机制，而不是有效增益机制 | final epoch 下，feedback 将 `path_alignment` 从 `0.6983` 降到 `0.5201`，`test_hard_mean_memory_delta_strength` 从 `0.0636` 降到 `0.0383`，说明它在压弱记忆影响 | 从“惩罚负对齐”升级为“奖励有用记忆增量”；引入 utility-conditioned gate 更新，而不是继续靠平均惩罚压缩分支 | `部分完成`。error attribution、hard-case reweighting 与 regularization 已接入，但优化方向仍偏保守 |
| 3 | archive 在训练后期仍会塌缩，而且这是系统共性，不是 feedback 独有问题 | `final_epoch_archive_collapse_is_general = true`，且 baseline 与 feedback 在 final epoch 的 `experience_archive_use_rate` 都为 `0.0` | 做 `slice-aware archive retention`，只对 hard cases 与 KG-active cases 强制保留 blended retrieval；推迟 bank 压缩时机；增加 archive-specific curriculum | `部分完成`。已有 `archive_retention_loss`，但没有真正阻止 final epoch collapse |
| 4 | KG consistency 分支目前没有被真正激活 | 中规模结果里 `kg_underused_rate`、`validation_kg_active_kg_underused_rate` 和 `test_kg_active_kg_underused_rate` 基本全为 `0.0` | 构造 KG-hard-case 子集；重定义 KG active 样本判据；提高 KG target 的分辨率，而不是只在当前分布上做无效约束 | `未完成`。KG 路径已可审计，但 KG consistency training 还没有形成有效训练压力 |
| 5 | persistent semantic 已经“能用”，但仍未稳定变成主训练流程中的长期收益来源 | `persistent_semantic_matrix_v3` 中，primed 组 `semantic_hit_rate=1.0`、`loaded_persistent_prototypes=7`，相对 no-persistent 的 `hybrid_mae` 改善约 `0.0143`；但这一收益仍依赖预热与特定设置 | 把 semantic 原型从弱模板混合升级为显式 residual branch；增加 prototype refresh schedule；统一预热与主训练流程 | `部分完成`。store 结构、prime 流程与 prototype 粒度已修，但 semantic 仍偏弱融合 |
| 6 | semantic 当前仍主要通过 planner weight 与 template blend 发挥作用，控制力不足 | 代码层面，semantic 命中主要影响 planner 权重、prior 与 template blend，并未成为独立主残差头 | 增加 `semantic_residual_head` 与独立 gate；将 semantic 路径与 online experience 路径拆开后再统一协调 | `未完成`。当前结构仍停留在“弱混合”阶段 |
| 7 | checkpoint 选择口径仍过于单一，只优化 `best_val_mae`，没有显式纳入机制质量 | 当前默认 `checkpoint_selection_mode = best_val_mae`；而你的研究目标已经不只是最小 MAE，而是“可解释性预测” | 增加 `best_interpretability_tradeoff` 或多目标 checkpoint 口径，将 `path_alignment`、`archive_use`、`kg_gate` 等指标纳入模型选择 | `未完成`。目前仍以纯误差最优为主 |
| 8 | 文档体系此前严重碎片化，不利于后续研究写作与阶段判断 | `docs/02_eicu` 原先存在 8 份并列文档，其中 6 份本质上是连续 follow-up 日志 | 将问题和进展统一收口到本文件，仅保留三份核心文档 | `已完成` |

## 二、已经完成的关键修复

以下事项已经可以视为阶段性完成，而不应继续作为“未解决问题”重复讨论：

1. `factual forecasting`、`counterfactual donor ranking` 与 `clinical plausibility` 的评估线已经拆开，且 donor 配置变化不会污染 factual 指标。
2. 显式 KG 路径已经接入主预测并可审计，不再停留在“图谱被拼到了静态特征里但无法证明是否被使用”的阶段。
3. persistent store 的样本结构保存与恢复已经修好，不再因为 `patient_static / intervention_static / kg_features` 缺失而导致复用失败。
4. persistent semantic 的 prototype 粒度已经从极粗模板改善到可工作状态，至少证明这条路径并非天然无效。
5. epoch-end error attribution 已经进入训练闭环，说明“解释信号进入训练”在工程上是可行的。

## 三、当前优先级排序

若只按对“可解释性预测模型”最关键的程度排序，当前问题优先级建议如下：

1. `archive collapse` 与后期记忆退化。
2. `epoch feedback` 只会压弱记忆，不会提升有用记忆。
3. semantic 路径仍然过弱，缺少显式残差头。
4. checkpoint 选择目标与研究目标不一致。
5. KG consistency 分支没有被真正激活。

该排序的理由很直接：如果 archive 和 semantic 都不能稳定参与，而 checkpoint 仍只按 `val_mae` 选，你最后得到的仍然会是一个“少用记忆、少暴露机制”的安全模型，而不是一个真正具有结构解释力的预测器。

## 四、推荐的后续改造顺序

建议按以下顺序推进，而不是并行堆更多新模块：

1. 先修 `archive collapse`
   - 目标：让 final epoch 不再出现 `experience_archive_use_rate = 0.0`
   - 做法：slice-aware retention、delayed compression、hard-case archive floor
2. 再修 `semantic residual`
   - 目标：让 semantic 不再只是弱模板混合，而是真正可审计的残差分支
   - 做法：新增 `semantic_residual_head` 和独立 gate
3. 再改 `epoch feedback` 的优化方向
   - 目标：从“压小记忆”转为“提升有效记忆”
   - 做法：引入 utility-aware reward、branch usefulness target
4. 最后改 `checkpoint` 策略
   - 目标：使模型选择标准与“可解释性预测”目标一致
   - 做法：联合 `val_mae`、`path_alignment`、`archive_use`、`kg_gate` 形成多目标选择口径

# 已知问题

1. 本文件是当前状态总表，但并不替代方法主文；如果后续方法结构再次变化，仍需同步维护 `EICU_MEMORY_KG_REMEDIATION_PLAN_20260409.md`。
2. 现有问题排序基于当前已有实验结果，若后续在更大规模实验上出现结论反转，应更新本文件中的优先级。
3. 某些问题的证据已经较强，例如 archive collapse；某些问题的证据仍偏间接，例如 semantic 弱混合的上限约束，需要结合后续代码修改进一步确认。

# 下一步计划

1. 先在代码层推进 `slice-aware archive retention`，专门解决 final epoch collapse。
2. 随后新增 `semantic_residual_head`，把 semantic 从弱混合提升为显式残差分支。
3. 在训练器中加入更明确的 usefulness target，使 epoch feedback 不再通过“压低 memory delta”换稳定。
4. 完成多目标 checkpoint 选择后，再进入下一轮中规模实验，重新判断“可解释性预测”是否真的优于当前 baseline。
