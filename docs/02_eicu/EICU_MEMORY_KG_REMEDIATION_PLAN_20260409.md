# 背景

本研究面向 eICU 脓毒症时序数据上的多步病情预测任务，目标是在给定短期观测窗口与临床干预上下文的条件下，对未来病情严重度轨迹进行预测。与仅依赖黑箱时序编码器的方案相比，当前系统尝试将三类额外信息显式纳入预测过程：其一为病例级经验记忆，其二为可持久化的语义原型库，其三为由临床规则、状态负荷与照护行为构成的知识图谱特征。上述设计并非单纯追求更高的点预测精度，而是希望在预测过程中保留一定的结构可解释性，即模型能够回答“预测主要受何种历史经验、何种图谱约束以及何种路径协调机制所驱动”。

在方法演化过程中，系统已经完成三条评估线的拆分：`factual forecasting` 用于检验真实预测误差，`counterfactual donor ranking` 用于检验反事实候选排序，`clinical plausibility` 用于检验临床规则一致性。因而，当前文档不再沿用工程迭代日志式的叙述方式，而将重点转向方法本身，包括任务定义、模型组成、训练目标、关键超参数及其在实验框架中的作用。与“当前问题、拟解决方案和修改进度”直接相关的内容，已单独整理至 `EICU_CURRENT_ISSUES_AND_REMEDIATION_STATUS_20260410.md`，以避免方法主文继续被多轮 follow-up 信息污染。

# 本轮修改目标

1. 将原有偏工程实现与修复记录的文档重写为偏论文风格的方法说明。
2. 系统介绍当前预测框架的组成，包括基础时序编码器、经验记忆模块、语义原型检索、显式知识图谱支路、转移记忆与误差反馈机制。
3. 明确列出模型训练所使用的关键参数及其默认设定，避免方法描述停留在抽象层面。
4. 以研究问题为主线说明当前框架希望解决的核心矛盾，即“预测性能、结构可解释性与记忆机制有效利用”之间的统一。

# 修改内容

## 1. 任务形式化与数据表示

设每个患者样本记为 `(X_hist, S_patient, S_intervention, G, Y_future)`。其中：

1. `X_hist` 表示长度为 `history_length` 的历史时序观测窗口。
2. `S_patient` 表示患者静态上下文特征。
3. `S_intervention` 表示干预相关静态与序列特征。
4. `G` 表示知识图谱相关特征，包括规则对齐、状态负荷与照护负荷等信息。
5. `Y_future` 表示长度为 `forecast_horizon` 的未来目标轨迹。

在当前 eICU 数据构造中，默认 `history_length=4`、`forecast_horizon=2`，且可通过 `append_kg_to_patient_static` 控制是否将 `G` 直接拼接入患者静态特征。每个样本还保留 `pattern_label`、`trajectory_label`、`experience_label`、`formation_features`、`patient_static`、`intervention_static` 与 `kg_features` 等字段，以支持后续的记忆检索、原型聚合与路径审计。

## 2. 模型总体结构

当前模型可以概括为“基础时序预测器 + 显式结构化残差”的组合式架构。其预测流程并非一次性从时序编码直接回归未来轨迹，而是依次叠加若干结构化修正项。

### 2.1 基础时序编码器

模型首先根据 `encoder_type` 选择基础编码器。

1. `GRU` 编码器为默认配置，主要参数包括：`gru_hidden_dim=64`、`gru_layers=1`、`gru_dropout=0.1`、`gru_bidirectional=True`。
2. `Transformer` 编码器作为替代配置，主要参数包括：`transformer_d_model=96`、`transformer_layers=2`、`transformer_heads=4`、`transformer_ff_dim=192`、`transformer_dropout=0.1`、`transformer_max_length=256`。

基础编码结果与干预嵌入共同进入 `base_regressor`，形成未经记忆与图谱修正的原始预测。

### 2.2 经验记忆与直接残差路径

模型在流形空间中维护 `pattern`、`trajectory` 与 `experience` 三类记忆组件。记忆读出首先通过 `memory_projectors` 映射到统一隐空间，再经 `memory_residual_heads` 产生候选残差。为避免记忆分支无条件覆盖基础预测，系统设置 `direct_residual_gate` 对直接经验残差进行门控，并通过 `memory_path_coordinator` 协调直接路径与融合路径的相对权重。

这意味着记忆机制在模型中并非“额外特征拼接”，而是具备独立路径、独立门控和独立审计指标的预测修正分支。

### 2.3 语义原型与持久化经验库

除在线热记忆外，系统还维护持久化经验库，并通过语义原型聚合提升跨轮次经验复用能力。持久化样本首先写入 `PersistentExperienceStore`，随后按模式标签、轨迹标签、季节性、预测步长、严重度桶、图谱状态签名与干预签名等上下文信息构建语义原型。训练与预测时，`MemoryManager` 可从原型库中检索 `top_k` 个语义命中，并据此调整：

1. `experience` 组件的先验读出。
2. 经验规划器的路径权重。
3. 原型模板曲线与在线经验读出的融合比例。

当前默认 `persistent_semantic_top_k=3`，且可选择在训练前使用 `prime_persistent_memory_before_fit` 对持久化记忆进行预热。

### 2.4 显式知识图谱支路

知识图谱特征既可作为静态特征直接拼接，也可经独立支路显式进入预测。为避免“将离散规则标记粗暴拼入静态编码器”导致的表示污染，当前框架支持显式图谱残差路径：`kg_features` 经过 `kg_projector` 编码后，由 `kg_residual_head` 产生图谱残差，再由 `kg_gate` 结合 `kg_guideline_alignment` 进行门控。该设计的核心目的不只是提升误差指标，更在于使知识图谱对预测的贡献能够被单独观测、开关与归因。

### 2.5 转移记忆与分桶残差

模型还支持基于状态转移的记忆检索机制，用于捕捉干预变化后的潜在轨迹偏移。对应模块包括 `transition_context_projector`、转移模板融合与事实路径修正。与此同时，`bucket_heads` 将未来轨迹按短期、中期、长期结构划分，以缓解统一残差头难以同时处理不同时间尺度误差的问题。

## 3. 训练目标与误差反馈机制

模型训练采用多项损失的线性组合。记 `L_fusion` 为主预测损失，`L_base` 为基础预测辅助损失，则总体目标可概括为：

`L = L_fusion + lambda_aux * L_base + lambda_align * L_align + lambda_temp * L_temp + L_feedback`

其中：

1. `L_align` 用于约束预测轨迹与真实未来轨迹在方向与结构上的一致性。
2. `L_temp` 用于抑制预测序列的非平滑波动。
3. `L_feedback` 为基于 epoch 级错例归因的附加约束，包括 `kg_consistency_loss`、`path_alignment_loss`、`archive_retention_loss` 与 `memory_delta_floor_loss`。

误差反馈机制的核心思想并非“每个 epoch 任意调整优化器超参数”，而是对高误差样本进行结构化归因后，再动态调节局部训练压力。具体而言：

1. `hard_example_weight` 提高困难样本在主损失中的权重。
2. `kg_consistency_weight` 约束知识图谱活跃样本上 `kg_gate` 不应长期处于过低水平。
3. `path_alignment_weight` 压制记忆直接路径与融合路径之间的负对齐。
4. `archive_retention_weight` 与 `archive_retention_target` 用于抑制训练后期 archive 使用塌缩。
5. `memory_delta_floor_weight` 与 `memory_delta_floor` 用于避免模型通过整体缩小记忆影响来获得表面稳定性。

上述权重可在 epoch 结束后依据验证集错例分析结果动态更新，更新速度由 `epoch_feedback_momentum` 控制，困难样本筛选比例由 `feedback_top_error_rate` 控制。

## 4. 关键模型与训练参数

### 4.1 训练与优化参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `epochs` | 6 | 训练轮数 |
| `batch_size` | 16 | 小批量大小 |
| `learning_rate` | 1e-3 | AdamW 学习率 |
| `weight_decay` | 1e-4 | 参数衰减 |
| `grad_clip` | 1.0 | 梯度裁剪上限 |
| `aux_base_loss_weight` | 0.3 | 基础预测辅助损失权重 |
| `align_loss_weight` | 0.02 | 结构对齐损失权重 |
| `temporal_smoothness_weight` | 0.02 | 时序平滑损失权重 |

### 4.2 记忆与解释性反馈参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `memory_direct_residual_weight` | 0.1 | 直接经验残差幅度 |
| `memory_direct_residual_mode` | `fixed` | 直接残差控制方式 |
| `memory_path_coordination_mode` | `sum` | 记忆路径协调方式 |
| `enable_epoch_feedback` | `False` | 是否启用 epoch 级误差反馈 |
| `hard_example_weight` | 0.35 | 困难样本重加权强度 |
| `kg_consistency_weight` | 0.06 | 图谱一致性约束强度 |
| `path_alignment_weight` | 0.05 | 路径对齐约束强度 |
| `archive_retention_weight` | 0.04 | archive 保留约束强度 |
| `memory_delta_floor_weight` | 0.03 | 记忆作用下限约束强度 |
| `archive_retention_target` | 0.10 | archive 使用目标值 |
| `memory_delta_floor` | 0.05 | 记忆影响下限 |
| `epoch_feedback_momentum` | 0.35 | 动态权重更新动量 |
| `feedback_top_error_rate` | 0.35 | 错例分析中纳入困难样本的比例 |
| `checkpoint_selection_mode` | `best_val_mae` | 模型选择口径 |

### 4.3 经验记忆与流形表示参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `top_k` | 6 | 在线记忆检索数量 |
| `sim_threshold` | 0.92 | 记忆合并相似度阈值 |
| `merge_alpha` | 0.2 | 原型更新幅度 |
| `decay` | 0.997 | 记忆活性衰减率 |
| `forget_threshold` | 0.08 | 记忆遗忘阈值 |
| `max_memory` | 256 | 记忆容量上限 |
| `memory_temperature` | 0.15 | 检索分布温度 |
| `manifold_dim` | 32 | 流形查询空间维度 |
| `manifold_value_dim` | 48 | 流形值空间维度 |
| `manifold_fusion_hidden_dim` | 64 | 多分支融合隐层维度 |

### 4.4 图谱与转移路径参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `enable_kg` | `False` | 是否启用知识图谱特征 |
| `disable_kg_static_concat` | `False` | 是否禁用图谱静态拼接 |
| `enable_explicit_kg_path` | `False` | 是否启用显式图谱支路 |
| `kg_residual_weight` | 0.12 | 图谱残差幅度 |
| `kg_alignment_floor` | 0.05 | 图谱对齐下限 |
| `enable_transition_memory` | `False` | 是否启用转移记忆 |
| `transition_top_k` | 6 | 转移候选数 |
| `transition_state_weight` | 0.65 | 状态相似度权重 |
| `transition_action_weight` | 0.35 | 干预相似度权重 |
| `transition_score_weight` | 0.12 | 转移残差强度 |
| `transition_template_blend_weight` | 0.10 | 转移模板融合权重 |

## 5. 评估协议与可解释性输出

当前实验协议明确区分三条评估线。

1. `factual forecasting` 仅以真实未来标签上的 `MAE`、`RMSE`、`SMAPE` 及相对基础模型的改进量为准。
2. `counterfactual donor ranking` 仅评价 donor 选择、候选排序与代理改善率，不作为真实预测收益证据。
3. `clinical plausibility` 仅评价规则一致性、可行性与临床合理性，不替代事实预测指标。

为了增强结构可解释性，训练与后处理阶段会输出以下审计量：

1. `direct_memory_strength`、`coordinated_memory_strength` 与 `path_alignment`，用于衡量记忆路径之间的协同关系。
2. `kg_gate`、`kg_residual_strength`、`kg_guideline_alignment`，用于衡量图谱支路是否实际参与预测。
3. `experience_archive_use_rate`、`semantic_retrieval.hit_rate`、`retrieval_source_rate`，用于衡量长期经验是否被真正调用。
4. `epoch_feedback_history` 与错例归因摘要，用于观察训练期可解释性反馈是否改变了模型的局部行为。

# 影响范围

本轮文档重写不改变现有代码实现，也不改变既有实验脚本的输入输出结构。其影响主要体现在以下三个层面：

1. 将原有“问题修复记录式”表述改写为“方法说明式”表述，使文档更适合作为论文方法节的中文草稿。
2. 将原本分散在脚本参数、训练器配置与补充说明文档中的方法细节进行集中整理。
3. 为后续撰写研究报告、论文初稿或答辩材料提供统一的方法描述模板。

# 运行方式

若需要复现当前方法，可使用如下实验入口命令，并依据研究目标选择相应配置。

```bash
python memory_mvp_project\run_forecasting_experiment.py ^
  --dataset-format eicu_sepsis3 ^
  --eicu-target-field total_sofa ^
  --history-length 4 ^
  --forecast-horizon 2 ^
  --eicu-max-series 32 ^
  --max-train-windows-per-series 8 ^
  --encoder-type transformer ^
  --epochs 4 ^
  --batch-size 16 ^
  --learning-rate 1e-3 ^
  --weight-decay 1e-4 ^
  --enable-kg ^
  --disable-kg-static-concat ^
  --enable-explicit-kg-path ^
  --enable-epoch-feedback ^
  --kg-residual-weight 0.12 ^
  --hard-example-weight 0.35 ^
  --kg-consistency-weight 0.06 ^
  --path-alignment-weight 0.05 ^
  --archive-retention-weight 0.04 ^
  --memory-delta-floor-weight 0.03
```

若研究重点转向持久化经验库，则可进一步加入：

```bash
--persistent-memory-store <store_dir> --prime-persistent-memory-before-fit --persistent-semantic-top-k 3
```

若研究重点转向转移记忆，则可进一步加入：

```bash
--enable-transition-memory --transition-top-k 6 --transition-score-weight 0.12
```

# 结果与分析

从方法论角度看，当前框架已经具备三个与传统黑箱时序预测器明显不同的特征。

第一，预测结果不再仅由一个统一的隐变量表示直接回归得到，而是由基础预测、经验残差、图谱残差、转移残差及多尺度桶修正共同构成。这使得模型天然具备“路径级”解释能力。

第二，知识图谱与记忆系统已经从“输入特征补丁”转为“显式预测支路”。这类结构设计比简单特征拼接更容易支持审计，因为模型是否调用图谱、是否调用 archive、是否使用语义原型，都可以通过独立门控与读出强度进行观察。

第三，误差反馈机制将“解释”从后验分析推进到训练过程本身。模型并非只在训练完成后报告错例，而是在 epoch 级别对困难样本进行归因，并将归因结果转化为后续训练的局部约束。这为构建“可解释性驱动训练”提供了必要的技术基础。

在现阶段实验中，这套方法已经能够回答若干关键方法问题，例如：图谱是否真实进入主预测、长期经验是否被实际调用、训练后期是否发生 archive 塌缩、路径协调是否从负对齐转向正对齐。相较于单纯比较一个误差指标，当前框架更接近一种“结构可解释预测器”的研究原型。

# 已知问题

1. 当前文档虽然已按方法学重写，但对应实验结论仍主要来自小规模与中等规模验证，尚不足以形成最终规模上的稳定统计结论。
2. 文中描述的方法模块已经完整存在于代码中，但不同模块的有效性并不均衡，尤其是长期经验库与语义原型的稳定贡献仍需进一步验证。
3. 现有框架可以提供结构性解释，但尚不能等同于严格的因果识别模型；其解释仍主要属于“机制归因”而非“干预效应识别”。

# 下一步计划

1. 在保持本文方法表述不变的前提下，补充更大样本规模下的系统实验，以验证各分支模块的外部稳定性。
2. 继续细化“训练期解释性反馈”与“测试期事实预测收益”之间的关系，明确哪些解释性指标能够真正转化为泛化误差改进。
3. 将知识图谱支路、持久化语义原型与转移记忆进一步统一到更明确的结构方程式表述中，为后续因果启发建模做准备。
4. 在后续论文写作中，将本文件与评估协议文档、实验补充文档配套使用，分别承担方法主文、实验设置与附录说明的角色。
