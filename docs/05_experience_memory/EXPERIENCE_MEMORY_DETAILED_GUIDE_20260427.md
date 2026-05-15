# 当前经验记忆库构建与工作机制详细说明

## 背景

本项目的经验记忆库不是单一文件，也不是简单的向量数据库。当前实现把“经验”拆成两个层次：

1. **持久化经验库**：由 [src/persistent_memory_store.py](../../src/persistent_memory_store.py) 维护，负责跨实验、跨进程保存历史样本、经验原型和模型相关 neural cache。
2. **在线 memory bank**：由 [src/manifold_memory.py](../../src/manifold_memory.py)、[src/memory_components.py](../../src/memory_components.py)、[src/memory_manager.py](../../src/memory_manager.py) 和 [src/manifold_forecasting_trainer.py](../../src/manifold_forecasting_trainer.py) 在训练/推理时动态构建，负责实际参与预测、检索和反事实推理。

**persistent experience store 负责经验沉淀和复用，online memory bank 负责当前模型状态下的可微读出和预测融合**。二者不是互相替代关系，而是“持久化经验供给当前运行，当前运行再形成模型相关在线记忆”的关系。

重点回答以下问题：

- 输入数据如何被处理成经验样本。
- 经验以什么形式存储，存储在哪些文件中。
- 经验内容如何新增、去重、聚合、更新和缓存。
- 模型训练和推理时如何调用经验。
- 经验记忆库如何提升 factual forecasting 和 counterfactual reasoning。
- 当前机制的优势、边界和后续改进方向。

## 修改内容

新增 `docs/05_experience_memory/` 子目录，并新增本说明文档。文档依据当前代码实现整理，主要参考：

- [src/tsf_data.py](../../src/tsf_data.py)：把 TSF 或 eICU 数据构造成 `ForecastSample`。
- [src/persistent_memory_store.py](../../src/persistent_memory_store.py)：持久化经验条目、原型、索引和 neural cache。
- [src/manifold_memory.py](../../src/manifold_memory.py)：编码器、注意力读出和基础 memory item。
- [src/memory_components.py](../../src/memory_components.py)：pattern、trajectory、experience 三类组件记忆，以及 hot/archive 机制。
- [src/memory_manager.py](../../src/memory_manager.py)：三类记忆的统一规划、语义原型检索和先验融合。
- [src/manifold_forecasting_trainer.py](../../src/manifold_forecasting_trainer.py)：训练、最终记忆重建、预测融合、反事实 donor 选择、干预候选生成和 transition memory。
- [run_forecasting_experiment.py](../../run_forecasting_experiment.py)：实验入口中持久化经验库的加载、复用、导出和输出记录。

## 运行方式

经验记忆库通常通过 `run_forecasting_experiment.py` 的参数启用：

```powershell
python run_forecasting_experiment.py `
  --persistent-memory-store output/memory_store/eicu `
  --persistent-memory-scope dataset `
  --persistent-semantic-top-k 3
```

关键参数含义：

- `--persistent-memory-store`：指定持久化经验库目录。为空时不启用持久化经验库。
- `--persistent-memory-scope dataset`：只复用同一 `dataset_name + seasonality + forecast_horizon` 的经验。
- `--persistent-memory-scope all`：允许跨数据集复用经验，风险更高，需要额外解释分布差异。
- `--persistent-semantic-top-k`：语义原型检索时取前几个 prototype。
- `--prime-persistent-memory-before-fit`：训练前先把当前训练样本写入 store，用于冷启动。
- `--disable-persistent-memory-reuse`：禁用读取旧经验，只保留写出能力。
- `--disable-persistent-memory-export`：禁用写出当前训练经验。

## 结果与分析

### 1. 总体工作流

当前经验记忆库的完整链路可以概括为：

```text
原始数据
  -> ForecastSample 样本构造
  -> formation 特征与 pattern/trajectory/experience 标签
  -> 持久化经验条目 experience_entries.jsonl
  -> 原型聚合 prototypes.jsonl
  -> 当前训练运行加载 persistent samples/prototypes
  -> 构建在线 pattern/trajectory/experience memory bank
  -> factual 预测时融合 memory readout
  -> counterfactual 时从 intervention store 检索 donor 干预
  -> 输出预测、反事实候选、donor 质量指标和 memory diagnostics
```

这里最重要的解释点是：**经验库既保存历史病例窗口，也保存它们在时序形态、知识图谱状态、干预形态和未来趋势上的抽象原型；当前模型运行时再把这些经验编码成可读写的在线记忆。**

### 2. 输入数据如何被处理

#### 2.1 统一样本对象：`ForecastSample`

无论输入是普通 TSF 时序数据，还是 eICU Sepsis-3 数据，最终都会被整理成 [src/tsf_data.py](../../src/tsf_data.py) 中的 `ForecastSample`。这是经验记忆库的最小经验单位。

每个 `ForecastSample` 包含：

- `sequence`：归一化后的历史时序窗口，模型编码器直接读取。
- `static`：静态特征，当前 eICU 中通常是 `patient_static + intervention_static`。
- `target`：归一化后的未来预测目标。
- `metadata`：样本身份和上下文信息，例如 stay、窗口位置、数据集名、季节性、预测步长、KG 标记、辅助任务标签。
- `scale_center` / `scale_value`：用于把归一化预测恢复到原始量纲。
- `raw_context` / `raw_target`：原始历史窗口和未来真实值。
- `formation_features`：时序形态特征。
- `pattern_label` / `trajectory_label` / `experience_label`：由 formation 模块产生的离散经验标签。
- `patient_static`：患者状态、人口学、动态上下文等患者侧特征。
- `intervention_static`：标签表和历史窗口聚合出的干预侧静态特征。
- `intervention_sequence`：历史窗口内逐时间步干预特征。
- `kg_features`：知识图谱构造出的病例级向量。
- `aux_targets`：多任务辅助预测目标，例如未来 SOFA 变化、乳酸变化、升压药需求、呼吸支持升级。

答辩时可以说：**经验不是只存一条时间序列，而是存一个带有临床状态、历史形态、干预上下文、KG 语义和未来结局的窗口化病例经验。**

#### 2.2 eICU 数据处理流程

eICU 主线由 `build_eicu_sepsis3_forecasting_dataset(...)` 构造。其输入通常包括：

- `labels_csv`：病例级标签和静态信息。
- `trajectory_csv`：按时间 bin 排列的动态轨迹。
- `target_field`：默认是 `total_sofa`。
- `history_length`：历史窗口长度，当前默认常用 4。
- `forecast_horizon`：未来预测长度，当前默认常用 2。
- `enable_kg` 和 `kg_directory`：是否启用知识图谱特征。

具体处理步骤如下：

1. 读取标签表和轨迹表。
2. 对 `patientunitstayid`、`bin_index`、目标字段等关键列做数值化和缺失过滤。
3. 按 `patientunitstayid` 分组，按 `bin_index` 排序。
4. 对每个患者构造多个滑动窗口：
   - 历史窗口：`target_start - history_length` 到 `target_start`。
   - 未来窗口：`target_start` 到 `target_start + forecast_horizon`。
5. 使用历史窗口计算归一化参数：
   - `scale_center` 为历史窗口中心。
   - `scale_value` 为安全尺度。
   - `sequence` 和 `target` 均用历史窗口尺度归一化。
6. 使用 `build_window_formation(...)` 计算时序形态特征和标签。
7. 将 label 表特征拆为患者侧和干预侧。
8. 将轨迹表特征拆为生理动态序列和干预动态序列。
9. 可选接入 KG，生成 `kg_features`、`kg_flags` 和 `kg_guideline_alignment`。
10. 生成辅助任务标签，用于约束反事实候选是否临床合理。

因此，输入数据不是直接进入 memory，而是先被结构化为**带标签、带归一化尺度、带临床上下文的预测窗口**。

#### 2.3 formation 标签的作用

`formation_features` 是连接原始时间序列和记忆检索的关键。它提供：

- 局部斜率、长期趋势、波动程度、季节性、变化点、曲率、自相关等形态特征。
- `pattern_label`：偏局部形态的类别。
- `trajectory_label`：偏未来或整体轨迹形态的类别。
- `experience_label`：通常可理解为 pattern 和 trajectory 的组合标签。

这些标签在后续有三类用途：

- 写入 `pattern_memory`、`trajectory_memory`、`experience_memory`。
- 构建持久化 prototypes 的分组键。
- 反事实 donor 检索时限定相似经验空间。

### 3. 数据如何储存

#### 3.1 持久化目录结构

`PersistentExperienceStore` 初始化后会形成如下目录结构：

```text
<persistent-memory-store>/
  manifest.json
  experience_entries.jsonl
  prototypes.jsonl
  events.jsonl
  indexes/
    semantic_index.json
  cache/
    <model_fingerprint>.pt
    <model_fingerprint>.meta.json
```

这些文件分别承担不同职责。

#### 3.2 `manifest.json`

`manifest.json` 是经验库元信息，记录：

- `schema_version`：存储格式版本。
- `formation_version`：formation 特征版本。
- `store_name`：当前为 `forecasting_persistent_experience_memory`。
- `created_at` / `updated_at`：创建和更新时间。
- `entry_count`：经验条目数量。
- `prototype_count`：经验原型数量。
- `datasets`：各数据集条目数量。
- `available_model_caches`：当前已有 neural cache 指纹。

它的作用是快速回答“这个经验库里现在有多少经验、来自哪些数据集、有没有可复用的模型缓存”。

#### 3.3 `experience_entries.jsonl`

这是最核心的持久化文件。每一行是一个经验样本，一个 JSON 对象，对应一个 `ForecastSample` 的持久化表达。

主要字段包括：

- 身份字段：
  - `experience_id`
  - `dataset_name`
  - `series_name`
  - `window_end_index`
  - `seasonality`
  - `history_length`
  - `forecast_horizon`
- 原始和归一化数据：
  - `raw_context`
  - `raw_future`
  - `normalized_context`
  - `normalized_future`
  - `scale_center`
  - `scale_value`
- 特征字段：
  - `static_features`
  - `patient_static_features`
  - `intervention_static_features`
  - `intervention_sequence`
  - `kg_features`
  - `kg_flags`
  - `formation_features`
- 标签字段：
  - `pattern_label`
  - `trajectory_label`
  - `experience_label`
  - `pattern_name`
  - `trajectory_name`
- 语义摘要字段：
  - `severity_bucket`
  - `kg_state_signature`
  - `intervention_signature`
  - `shape_signature`
  - `target_signature`
- 质量和来源字段：
  - `support`
  - `quality_score`
  - `source`
  - `created_at`

其中 `experience_id` 由 `forecast_sample_identity(...)` 生成，使用数据集名、序列名、窗口位置、历史长度、预测长度、季节性和 `raw_context` 的 SHA1 摘要。这保证同一个窗口重复写入时可以被识别并跳过。

#### 3.4 `prototypes.jsonl`

`prototypes.jsonl` 保存经验原型。它不是单个病例，而是多个经验条目的聚合中心。

当前分组键由以下字段组合：

```text
dataset_name
pattern_label
trajectory_label
severity_bucket
kg_state_signature
intervention_signature
future_direction_type
forecast_horizon
seasonality
```

每个 prototype 保存：

- `prototype_id`
- `group_key`
- `member_experience_ids`
- `dataset_name`
- `pattern_label`
- `trajectory_label`
- `experience_label`
- `severity_bucket`
- `kg_state_signature`
- `intervention_signature`
- `future_direction_type`
- `support`
- `prototype_formation_center`
- `prototype_future_mean_curve`
- `prototype_kg_center`
- `intervention_density_mean`
- `quality_score`
- `last_updated_at`

答辩时可以强调：**prototype 是经验库的“语义压缩层”，让系统不用每次扫描所有历史样本，也能基于形态、病情、KG 状态和干预模式找到相近经验簇。**

#### 3.5 `indexes/semantic_index.json`

当前 semantic index 是轻量索引，包含：

- `by_dataset`：按数据集映射到经验 ID。
- `by_experience_label`：按经验标签映射到经验 ID。
- `updated_at`。

当前实际 prototype 检索仍主要通过 `load_prototypes(...)` 加载后过滤和打分；这个索引用于保存基础分组关系和后续扩展。

#### 3.6 `events.jsonl`

`events.jsonl` 是操作日志，记录：

- `upsert_samples`
- `prototype_rebuild`

这可以回答“经验库内容如何更迭、有没有记录每次重建”。

#### 3.7 `cache/<model_fingerprint>.pt`

neural cache 保存的是当前模型参数条件下的最终在线 memory bank，而不是参数无关的原始样本。

其主要内容是：

- `pattern`：pattern memory 的 memory items。
- `trajectory`：trajectory memory 的 memory items。
- `experience_hot`：experience memory 热区。
- `experience_archive`：experience memory 归档区。

对应的 `.meta.json` 保存：

- `model_fingerprint`
- `dataset_name`
- `seasonality`
- `forecast_horizon`
- 各类 memory size
- cache 文件路径

`model_fingerprint` 由训练配置、memory 配置、动态 loss 权重、模型 state_dict 等共同哈希得到。只有模型状态一致时才复用，避免把旧模型编码出的 memory bank 错用到新模型上。

### 4. 内容如何更迭

#### 4.1 新经验写入

`upsert_samples(samples, source=...)` 是持久化经验写入入口：

1. 读取已有 `experience_entries.jsonl`。
2. 为每个样本计算 `experience_id`。
3. 已存在则跳过。
4. 新样本追加写入 JSONL。
5. 触发 `rebuild_prototypes()`。
6. 更新 `manifest.json`。
7. 追加 `events.jsonl` 事件。

这是一种 append-first 的写法，避免覆盖已有经验。

#### 4.2 原型重建

每次写入后会重建 prototypes。重建不是增量合并，而是从所有 entries 重新分组计算：

- `prototype_formation_center`：formation 特征均值。
- `prototype_future_mean_curve`：归一化未来曲线均值。
- `prototype_kg_center`：KG 特征均值。
- `support`：组内样本数。
- `quality_score`：组内质量分均值。

这保证 prototype 始终反映当前完整经验库。

#### 4.3 在线 memory bank 更迭

在线 memory bank 在训练期间会多次重建：

1. `fit(...)` 开始时先根据 memory seed samples 构建一次 memory bank。
2. 每个 epoch 训练结束后，根据当前 encoder 重新构建 memory bank。
3. 验证集选择最优模型状态。
4. 训练结束后恢复 `best_state`。
5. 再基于 `best_state` 恢复或重建最终 memory bank。

因此最终留下的不是最后一个 epoch 的 memory，而是**验证集最优模型状态对应的 memory bank**。

#### 4.4 在线写入规则

在线 memory item 的基本结构是：

- `key`：归一化 key 向量，用于相似度检索。
- `value`：value 向量，用于读出和预测融合。
- `label`：pattern、trajectory 或 experience 标签。
- `activity`：活跃度。
- `support`：被合并支持数。
- `metadata`：病例和质量相关元信息。

写入时先判断是否可以与已有记忆合并：

- 如果 label 内存数量不足，会强制保留新记忆，避免早期 memory 被少数样本垄断。
- 如果存在高相似记忆，则按 `merge_alpha` 做 key/value 融合，并增加 `support`。
- 如果没有合适合并对象，则新增 memory item。
- 写入后根据活跃度、support、患者多样性和每类最大容量进行 trim。

#### 4.5 hot/archive 机制

`ExperienceMemory` 比 pattern 和 trajectory 更复杂，它把经验记忆拆成：

- `memory_bank`：热区，存当前最常用、优先读出的经验。
- `archive_bank`：归档区，存被压缩但仍可能有用的经验。

读取时先读 hot memory。如果 hot 置信度不足，或者 archive 置信度明显更好，会按上限权重混合 archive readout。这样可以防止经验库只保留最近或最强的少量经验，提升困难样本上的回忆能力。

### 5. 经验如何被调用

#### 5.1 实验入口中的持久化复用

[run_forecasting_experiment.py](../../run_forecasting_experiment.py) 中的调用顺序是：

1. 如果设置 `--persistent-memory-store`，创建 `PersistentExperienceStore`。
2. 如果设置 `--prime-persistent-memory-before-fit`，先把当前训练样本写入 store。
3. 如果没有禁用 reuse，则读取：
   - `persistent_samples`
   - `persistent_prototypes`
4. 调用 `trainer.configure_semantic_store(...)` 接入 prototype 检索。
5. 调用 `trainer.configure_neural_cache(...)` 接入模型相关 cache。
6. 训练时使用：

```text
memory_seed_samples = dedupe(persistent_samples + current_train_samples)
```

这说明 persistent samples 会和当前训练样本共同构建在线 memory bank。

#### 5.2 三类在线记忆

`MetaMemoryManager` 管理三类 memory：

- `pattern_memory`：偏局部形态模式，强调 pattern label。
- `trajectory_memory`：偏时序轨迹趋势，强调 trajectory label。
- `experience_memory`：综合经验标签，允许更宽的相似经验复用，并带 hot/archive。

三类 memory 的作用不同：

- pattern memory 帮助模型识别“当前窗口像哪种局部形态”。
- trajectory memory 帮助模型判断“未来走势或轨迹类型可能是什么”。
- experience memory 帮助模型复用“历史上类似状态和干预下的整体结局经验”。

#### 5.3 semantic retrieval 如何进入预测

当 `MetaMemoryManager.read(...)` 被调用时，会先做 semantic retrieval：

1. 从 persistent store 加载符合 dataset、seasonality、forecast_horizon 的 prototypes。
2. 对每个 prototype 计算综合得分：
   - formation 相似度。
   - KG 向量相似度。
   - pattern/trajectory/experience label 匹配。
   - 病情严重度匹配。
   - KG 状态签名匹配。
   - 干预签名匹配。
   - 未来方向匹配。
   - support bonus。
   - quality score。
3. 取 top-k prototype。
4. 构造：
   - `semantic_prior`：调整 experience label 先验。
   - `template_curve`：按 prototype 得分加权的未来曲线模板。
   - `template_confidence`：语义模板置信度。
   - `template_blend_weight`：模板曲线融合权重。
5. 提升 `experience` 分支 planner weight。

因此 semantic retrieval 不是直接替代模型预测，而是通过**先验、模板曲线和分支权重**影响模型。

#### 5.4 在线读出和融合

模型前向时大致有两条路径：

1. factual path：当前样本自己的历史状态、静态信息、干预上下文、KG residual。
2. retrieval path：当前样本查询在线 memory bank，得到 pattern/trajectory/experience readout。

记忆读出过程：

- encoder 输出 query/key/value/input embedding。
- query 与 memory key 做点积相似度。
- 加入 label prior 和 rerank bias。
- 取 top-k。
- softmax 得到注意力权重。
- 加权 memory value 得到 readout。
- 根据最大相似度、label mass、分数 margin、注意力熵计算 memory confidence。

融合过程：

- pattern、trajectory、experience readout 分别投影到 latent 空间。
- 根据 planner weights、confidence 和 learned gate 得到三类 memory gate。
- experience readout 可进一步混合 semantic template curve 编码。
- 生成两类 memory residual：
  - fusion residual：通过融合网络和 bucket head 生成。
  - direct memory residual：由 experience readout 直接生成。
- 两条 residual 通过 `memory_path_coordinator` 或 sum 模式协调。
- 最终预测为：

```text
fusion_prediction = base_prediction + coordinated_memory_residual + transition_residual
```

### 6. 经验记忆库如何服务反事实推理

#### 6.1 反事实问题的形式

在当前系统中，反事实推理不是抽象地问“如果不同会怎样”，而是更具体地做：

> 对当前病例窗口，寻找历史上相似患者或相似状态下的 donor 干预方案，把该干预方案或修复后的干预方案施加到当前样本，预测未来结局变化。

也就是说，反事实推理依赖三件事：

- 当前病例 factual 预测。
- 历史经验中的 donor 干预。
- 模型对替换干预后的 counterfactual 预测。

#### 6.2 intervention store

训练结束后，trainer 会基于 memory seed samples 构建 `intervention_store_entries`。这部分代码集中在 [src/manifold_forecasting_trainer.py](../../src/manifold_forecasting_trainer.py) 的 `InterventionStoreEntry`、`InterventionComponentRecord`、`_build_intervention_store(...)`、`_store_intervention_plan_components(...)`、`_rebuild_intervention_store_cache(...)` 和 `export_inference_bundle(...)`。

需要先明确一个边界：**干预库不是 `PersistentExperienceStore` 目录中的独立 JSONL 文件**。当前实现中，干预库首先是 trainer 内存中的运行时结构；如果设置 `--output-inference-bundle`，它会随 `trainer.export_inference_bundle()` 一起保存到 `.pt` inference bundle 中。也就是说：

- `PersistentExperienceStore` 负责经验样本、prototype 和 neural cache。
- `intervention store` 负责反事实 donor 干预方案检索。
- `intervention store` 的持久化位置是 inference bundle 的 `trainer_bundle` 内部，而不是 `experience_entries.jsonl`。

##### 6.2.1 干预库的总体结构

trainer 初始化时会维护以下干预库字段：

```text
intervention_store_entries: List[InterventionStoreEntry]
intervention_plan_store: Dict[str, Dict[str, object]]
intervention_method_store: Dict[str, InterventionComponentRecord]
intervention_dose_store: Dict[str, InterventionComponentRecord]
intervention_timing_store: Dict[str, InterventionComponentRecord]
intervention_context_store: Dict[str, InterventionComponentRecord]
intervention_store_by_label: Dict[int, List[int]]
intervention_store_by_pattern: Dict[int, List[int]]
intervention_store_by_trajectory: Dict[int, List[int]]
intervention_store_by_hospital: Dict[str, List[int]]
intervention_store_by_unit_type: Dict[str, List[int]]
intervention_store_by_infection_anchor: Dict[str, List[int]]
intervention_store_embedding_cache: Dict[int, torch.Tensor]
```

可以把它理解为三层：

1. **病例级 donor 条目层**：`intervention_store_entries`，每个样本一条 donor 候选。
2. **干预方案组件层**：`plan_store + method/dose/timing/context_store`，把干预向量拆成可复用的组件编码。
3. **快速检索索引层**：按 label、pattern、trajectory、hospital、unit、infection anchor 建立倒排索引，并按 experience label 缓存 patient embedding tensor。

##### 6.2.2 病例级 `InterventionStoreEntry` 存什么

`InterventionStoreEntry` 是反事实 donor 的主表条目。每个条目包含：

- `stay_id`
- `experience_id`
- `experience_label`
- `pattern_label`
- `trajectory_label`
- `patient_embedding`
- `intervention_plan_code`
- `intervention_static`
- `intervention_sequence`
- `kg_features`
- `kg_guideline_score`
- `metadata`

这些字段的含义如下：

- `stay_id`：donor 对应的 ICU stay。
- `experience_id`：窗口级经验 ID，默认由 `forecast_sample_identity(sample)` 生成。
- `experience_label` / `pattern_label` / `trajectory_label`：用于按照经验类别和时序形态筛选 donor。
- `patient_embedding`：当前模型 encoder 对该样本状态的 latent embedding，用于与 query 患者做向量相似度检索。
- `intervention_plan_code`：组件化干预方案编码，包含完整 plan code 以及 method/dose/timing/context 四类组件 code。
- `intervention_static`：病例级或窗口聚合后的干预静态向量。
- `intervention_sequence`：历史窗口内逐时间步干预向量。
- `kg_features`：donor 病例的 KG 向量，用于 donor 与 query 的 KG 相似度。
- `kg_guideline_score`：donor 方案的指南一致性得分。
- `metadata`：写入时的上下文信息，包括 KG flags、quality、hospital、unit type、infection anchor、intervention_plan_code 等。

其中 `patient_embedding` 来自当前模型编码器，代表患者状态；`intervention_static` 和 `intervention_sequence` 代表 donor 的干预方案；`kg_guideline_score` 和 `kg_flags` 用于判断该方案是否符合脓毒症相关知识约束。

##### 6.2.3 干预库如何从样本构建

`_build_intervention_store(samples)` 在 `fit(...)` 结束阶段被调用，输入是 `memory_samples`，也就是去重后的 `persistent_samples + current_train_samples` 或当前训练样本。构建步骤是：

1. 清空旧的 `intervention_store_entries` 和组件库。
2. 分 batch 对样本调用 `_encode_batch(...)`。
3. 对每个样本提取：
   - `intervention_static`
   - `intervention_sequence`
   - `kg_features`
   - `kg_guideline_alignment`
   - `pattern_label`
   - `trajectory_label`
   - `experience_label`
4. 调用 `_store_intervention_plan_components(...)` 生成组件化 plan code。
5. 调用 `_write_metadata(sample)` 生成 donor 元数据，并把 `intervention_plan_code` 写入 metadata。
6. 用 encoder 的 `input_embedding` 作为 `patient_embedding`。
7. 组装 `InterventionStoreEntry`，追加到 `intervention_store_entries`。
8. 最后调用 `_rebuild_intervention_store_cache()` 建立倒排索引和 embedding cache。

因此干预库的来源不是单独人工录入的治疗规则，而是**从历史窗口样本中抽取“当时患者状态 + 当时干预方案 + 后续结局经验标签 + KG 语义”的 donor 条目**。

##### 6.2.4 为什么要同时存原始向量和组件 code

当前代码同时保存两种表达：

```text
intervention_static / intervention_sequence
intervention_plan_code
```

前者是模型可直接使用的数值向量，后者是可复用、可解释、可压缩的结构化编码。

这样设计有三个原因：

- **推理效率**：反事实 forward 时可以直接把 `intervention_static` 和 `intervention_sequence` 放回 `ForecastSample`。
- **可解释性**：输出 case report 时可以通过 `intervention_plan_code` 和组件库还原 method/dose/timing/context。
- **去重和复用**：相同的 method、dose、timing 或 context 组件会生成稳定 hash code，只在对应组件 store 中保存一次。

##### 6.2.5 组件库如何划分 method/dose/timing/context

`_intervention_feature_component_type(feature_name, is_sequence)` 按特征名自动划分组件类型：

- `timing`：包含 `offset`、`minute`、`hour`、`time`、`timing`、`duration`、`course`、`window` 等关键词。
- `dose`：包含 `dose`、`dosage`、`amount`、`rate`、`score`、`intensity`、`level`、`mean`、`max`、`last` 等关键词，且不属于 `any` 或 `flag`。
- `method`：包含 `antibiotic`、`antimicrobial`、`vasopressor`、`resp`、`vent`、`blood_culture`、`lactate`、`map`、`sofa`、`monitor`、`exam`、`treat`、`any`、`flag` 等关键词。
- `context`：无法归入上述类别的静态特征。
- 对 sequence 特征，如果没有命中明确规则，默认归入 `dose`。

随后 `_intervention_component_index_groups()` 会把 `intervention_feature_names` 和 `intervention_sequence_feature_names` 分别划入四类组件。每类组件记录：

- 静态特征索引。
- 时序特征索引。
- 特征名。
- 对应数值。

这意味着干预方案不是一个不可解释的长向量，而是被拆成“做了什么、剂量/强度如何、何时做、上下文是什么”四个部分。

##### 6.2.6 `InterventionComponentRecord` 存什么

每个组件记录包含：

```text
component_code
component_type
static_values_by_index
sequence_values_by_index
static_feature_names_by_index
sequence_feature_names_by_index
summary
```

其中：

- `component_code`：由组件内容 SHA1 生成，例如 `int_method_xxx`、`int_dose_xxx`。
- `component_type`：`method`、`dose`、`timing` 或 `context`。
- `static_values_by_index`：静态干预向量中属于该组件的索引和值。
- `sequence_values_by_index`：时序干预矩阵中属于该组件的列索引和每个时间步的值。
- `static_feature_names_by_index`：静态索引对应的原始特征名。
- `sequence_feature_names_by_index`：时序索引对应的原始特征名。
- `summary`：组件摘要，包括特征数量、序列长度、静态强度、序列强度、active action profile。

`component_code` 是稳定编码。代码会先把组件内容整理成 `code_payload`，其中数值 round 到 8 位小数，再用 `json.dumps(..., sort_keys=True)` 做 SHA1。因此相同组件会得到相同 code，不会重复注册。

##### 6.2.7 `intervention_plan_store` 存什么

每个完整干预方案会被编码成一个 plan：

```text
plan_payload = {
  "schema_version": 2,
  "static_dim": 静态干预向量维度,
  "sequence_dim": 时序干预每步维度,
  "sequence_length": 时序长度,
  "components": {
    "method": int_method_xxx,
    "dose": int_dose_xxx,
    "timing": int_timing_xxx,
    "context": int_context_xxx
  }
}
```

再由 `plan_payload` 生成稳定 `plan_code`，形如 `int_plan_xxx`。最终 donor entry 里的 `intervention_plan_code` 是：

```text
{
  "plan": int_plan_xxx,
  "method": int_method_xxx,
  "dose": int_dose_xxx,
  "timing": int_timing_xxx,
  "context": int_context_xxx
}
```

这相当于给每个 donor 干预方案保存了一个“组合索引”：完整方案由四个组件 code 拼装而来。

##### 6.2.8 干预方案如何还原

`_reconstruct_intervention_plan_from_code(intervention_plan_code)` 可以从 plan code 还原原始数值向量：

1. 读取 `plan_code` 对应的 `plan_record`。
2. 根据 `static_dim`、`sequence_dim`、`sequence_length` 创建全 0 的静态向量和时序矩阵。
3. 依次读取 method、dose、timing、context 四类组件。
4. 把每个组件中的 `static_values_by_index` 写回静态向量对应位置。
5. 把 `sequence_values_by_index` 写回时序矩阵对应位置。
6. 返回 `reconstructed_static` 和 `reconstructed_sequence`。

`_materialize_intervention_entry(entry)` 会在重建 cache 时保证 entry 可用：

- 如果 entry 已经有 `intervention_plan_code`，就尝试用组件库还原 `intervention_static` 和 `intervention_sequence`。
- 如果 entry 没有 plan code 但有原始向量，就反向注册组件，生成 plan code。

因此 inference bundle 重新加载后，即使需要依赖组件 code，也可以恢复成模型实际需要的干预向量。

##### 6.2.9 干预库如何被序列化保存

`export_inference_bundle()` 会把干预库写入 `trainer_bundle`：

```text
trainer_bundle = {
  ...
  "intervention_component_stores": {
    "schema_version": 2,
    "plan_store": ...,
    "method_store": ...,
    "dose_store": ...,
    "timing_store": ...,
    "context_store": ...
  },
  "intervention_store_entries": [
    InterventionStoreEntry as dict,
    ...
  ],
  ...
}
```

如果 `run_forecasting_experiment.py` 设置了 `--output-inference-bundle`，外层会调用 `torch.save(...)` 保存：

```text
{
  "schema_version": 1,
  "bundle_type": "eicu_counterfactual_inference",
  "dataset_summary": ...,
  "input_sources": ...,
  "dataset_semantics": ...,
  "persistent_memory": ...,
  "trainer_bundle": trainer.export_inference_bundle()
}
```

因此，干预库落盘后不是人眼直接阅读的 JSONL，而是在 `.pt` bundle 中作为 Python dict/list/tensor payload 保存。重新加载时，`load_inference_bundle(...)` 会：

1. 加载模型参数和 normalizer。
2. 调用 `_load_intervention_component_stores(...)` 恢复组件库。
3. 调用 `_intervention_store_entry_from_dict(...)` 恢复 donor 条目。
4. 调用 `_rebuild_intervention_store_cache()` 重建 label/pattern/trajectory/hospital/unit/infection 索引和 embedding cache。

##### 6.2.10 干预库如何建立检索索引

`_rebuild_intervention_store_cache()` 会把每条 donor entry 的索引挂到多个倒排表：

- `intervention_store_by_label[experience_label]`
- `intervention_store_by_pattern[pattern_label]`
- `intervention_store_by_trajectory[trajectory_label]`
- `intervention_store_by_hospital[hospitalid]`
- `intervention_store_by_unit_type[unittype]`
- `intervention_store_by_infection_anchor[infection_anchor_type]`

同时，对每个 `experience_label`，还会把该标签下 donor 的 `patient_embedding` 堆叠成 tensor，放入 `intervention_store_embedding_cache`。当前反事实检索函数主要根据候选索引临时构造 embedding tensor 做相似度；这个 cache 为按 label 快速检索提供基础。

##### 6.2.11 干预库在反事实推理中如何被使用

`predict_counterfactual(...)` 调用 `_counterfactual_ranked_donors_from_manager_result(...)`，后者会从干预库中选择 donor：

1. 根据 memory manager 的 experience top label 和 matched labels 得到候选 experience labels。
2. 加入当前样本的 experience label。
3. 从 `intervention_store_by_label` 取同经验标签 donor。
4. 可选加入同 pattern、同 trajectory donor。
5. 可选加入同 hospital、同 unit type、同 infection anchor donor。
6. 候选不足时按配置做 global backfill。
7. 排除与 query 同一个 `stay_id` 的 donor。
8. 用 query 的 `encoding.input_embedding` 与 donor 的 `patient_embedding` 做 cosine/点积相似度。
9. 结合 KG 相似度、指南兼容性、hard filter、overlap filter、neighbor consistency、transition utility 等进行重排。
10. 选出排名靠前的 donor，将其 `intervention_static` 和 `intervention_sequence` 作为反事实干预候选来源。

因此干预库的核心作用是：**提供可被模型直接施加到当前 query 上的历史干预方案，并且这些方案带有患者状态 embedding、经验标签、KG 语义和上下文索引，便于做可信 donor 检索。**

##### 6.2.12 干预库与经验记忆库的关系

干预库和经验记忆库共享同一批 `memory_samples`，但职责不同：

- online memory bank 主要存 key/value 经验，用于预测时读出历史状态和未来趋势。
- intervention store 主要存 donor 干预方案，用于反事实时替换当前样本的治疗输入。
- persistent prototypes 影响“像哪些经验”，intervention store 影响“借用哪个历史干预方案”。
- transition store 进一步补充“某种状态 + 某种行动 historically 对未来曲线的效用”。

可以向导师解释为：

> 经验记忆库解决“当前病例像哪些历史经验”；干预库解决“这些相似历史经验中有哪些实际执行过的干预可以作为反事实 donor”；transition store 则进一步估计“状态-行动-未来变化”的经验效用。

#### 6.3 donor 候选池

反事实 donor 不是从全库盲选，而是先构造候选池。候选池来源包括：

- 同 experience label。
- 同 pattern label。
- 同 trajectory label。
- 同医院。
- 同病区或 unit type。
- 同感染锚点。
- 必要时全局 backfill。

这样做的目的：

- 保证 donor 与当前样本具有时序形态相似性。
- 保证 donor 与当前样本具有临床场景相似性。
- 避免候选过少时完全找不到 donor。

#### 6.4 donor 打分

每个 donor 会计算多类分数：

- `embedding_similarity`：患者 embedding 相似度。
- `kg_similarity`：KG 特征相似度。
- `guideline_compatibility`：干预是否覆盖当前病情所需的指南相关 care。
- `state_match`：供体状态是否与当前状态匹配。
- `missing_care_penalty`：关键 care 缺失惩罚。
- `contraindication_penalty`：不适合当前状态的干预惩罚。
- `transition_score`：如果启用 transition memory，评估 action 到 outcome 的经验效用。
- `action_change_score`：干预变化是否有足够信息量。
- `overlap_score`：状态和行动集合是否有足够重叠。
- `neighbor_consistency`：候选 donor 周围邻居是否支持相同决策。
- `learned_reranker_score`：可选学习式重排器分数。
- `pool_match_reward`：同医院、同病区、同感染锚点等候选池匹配奖励。

最终 donor score 是这些项的加权组合，再经过 hard filter、overlap filter 和 neighbor consistency 修正。

#### 6.5 干预候选生成

系统支持三种候选策略：

- `donor_only`：直接使用 donor 原始干预。
- `generated_best`：同时评估 donor 原始干预和 KG 修复后的 donor 干预。
- `safe_search`：在安全修复基础上加入受约束的模板搜索。

候选生成后，并不是只看 donor 分数，而是把每个候选真正放回模型中做一次 counterfactual forward，得到候选未来预测。

#### 6.6 候选选择

对每个候选，系统计算：

```text
predicted_delta = mean(factual_prediction) - mean(counterfactual_prediction)
```

如果目标是 SOFA，`predicted_delta > 0` 表示 counterfactual 预测比 factual 更低，代表可能改善。候选选择还会综合：

- 预测改善幅度。
- 可行性。
- 临床安全修复。
- 辅助任务信号。
- donor 邻域一致性。
- KG 约束和惩罚。

最终输出 selected candidate、top donor candidates、candidate options 和各类 donor 质量指标。

### 7. 为什么经验记忆库能提升反事实推理效果

#### 7.1 它让 donor 不是任意相似，而是“经验相似”

普通相似患者检索可能只看 embedding 距离。当前系统额外引入：

- pattern label。
- trajectory label。
- experience label。
- severity bucket。
- KG state signature。
- intervention signature。
- future direction。

因此 donor 更接近“相似病情 + 相似时序形态 + 相似治疗上下文 + 有历史结局支持”的病例。

#### 7.2 它提供可复用的未来结局模板

prototype 的 `prototype_future_mean_curve` 给模型一个经验模板。该模板不会直接覆盖预测，而是经过 `prototype_curve_encoder` 编码后按 `template_blend_weight` 融合进 experience readout。这样模型既能保留当前样本特异性，也能借用历史经验的未来趋势。

#### 7.3 它减少小样本下的反事实不稳定

反事实推理最容易不稳定的地方是 donor 少、候选干预稀疏、模型对未见组合外推。经验库通过以下方式缓解：

- persistent samples 扩大 memory seed。
- prototypes 提供跨样本聚合中心。
- hot/archive 保留低频但有价值的经验。
- semantic prior 防止检索完全被当前小批训练数据支配。
- transition memory 为 action-outcome 提供经验效用估计。

#### 7.4 它让反事实评价更可解释

输出中保留了大量可解释字段：

- donor 相似度。
- donor KG 相似度。
- 指南兼容度。
- 缺失 care 惩罚。
- 禁忌惩罚。
- overlap 是否通过。
- 邻域一致性。
- 候选来源。
- repair actions。
- search actions。
- predicted delta。

这些字段可以回答“为什么选这个 donor，而不是另一个 donor”。

#### 7.5 它支持知识图谱约束

经验库记录 KG 特征和 KG flags，反事实 donor 打分时使用：

- sepsis 状态。
- septic shock 状态。
- hypotension。
- high lactate。
- organ dysfunction。
- antibiotic。
- blood culture。
- lactate exam。
- vasopressor。
- MAP/lactate monitoring。

因此反事实不是单纯寻找数值相似，而是会考虑当前病情下哪些 care 应该出现、哪些干预可能不合适。

### 8. 如何判断经验库是否生效

实验输出中重点看：

- `persistent_memory.enabled`：是否启用持久化经验库。
- `persistent_memory.loaded_persistent_samples`：加载了多少历史样本。
- `persistent_memory.loaded_persistent_prototypes`：加载了多少 prototypes。
- `persistent_memory.neural_cache.loaded`：是否命中 neural cache。
- `persistent_memory.neural_cache.exported`：是否写出 neural cache。
- `memory_diagnostics.memory_seed_sample_count`：最终构建 online memory 的样本数。
- `memory_diagnostics.semantic_retrieval.hit_rate`：semantic retrieval 命中率。
- `memory_diagnostics.semantic_retrieval.mean_top_score`：prototype 匹配质量。
- `memory_diagnostics.semantic_retrieval.mean_template_blend_weight`：语义模板实际融合强度。
- `memory_diagnostics.experience_archive_use_rate`：archive 是否实际被使用。
- `memory_diagnostics.gate_means.experience_gate`：experience 分支是否参与预测。
- `counterfactual.donor_found_rate`：是否能找到 donor。
- `counterfactual.donor_exact_experience_match_rate`：donor 与 query 经验标签精确匹配率。
- `counterfactual.donor_guideline_compatibility_mean`：donor 干预指南兼容性。
- `counterfactual.donor_overlap_valid_rate`：donor 与 query 可交换性。
- `counterfactual.predicted_improvement_rate`：预测改善比例。

如果 `loaded_persistent_samples` 和 `loaded_persistent_prototypes` 大于 0，且 `semantic_retrieval.enabled = true`，说明持久化经验确实进入了当前运行。如果 `semantic_hit_rate` 很低，说明 prototype 条件过严、数据集 scope 不匹配或经验库覆盖不足。

### 9. 导师可能追问与回答要点

#### 9.1 你们的经验记忆库存的到底是什么？

回答要点：

> 存的不是单个向量，而是窗口化病例经验。每条经验包括原始历史窗口、未来真实轨迹、归一化序列、患者静态特征、干预特征、KG 特征、formation 形态特征、pattern/trajectory/experience 标签、病情严重度、干预签名、未来趋势签名和质量分。另有 prototype 层把同类经验聚合为语义中心，neural cache 层保存当前模型状态下的最终在线 memory bank。

#### 9.2 为什么不用普通向量数据库？

回答要点：

> 当前任务不只是最近邻检索，还需要训练期可微融合、标签先验、时序形态约束、KG 约束、hot/archive 管理和模型指纹 cache。普通向量数据库可以替代一部分近邻检索，但不能直接表达 pattern/trajectory/experience 三分支、prototype future curve、memory confidence、direct residual 和 counterfactual donor 评分链路。

#### 9.3 经验库如何避免重复样本？

回答要点：

> 通过 `forecast_sample_identity(...)` 生成 `experience_id`。它由数据集名、序列名、窗口结束位置、历史长度、预测长度、季节性和 raw context 哈希得到。写入时如果 ID 已存在就跳过。

#### 9.4 内容会不会越存越乱？

回答要点：

> 持久化 entries 是 append 去重的；每次 upsert 后会从全量 entries 重建 prototypes，保证语义中心一致。在线 memory bank 则有相似度合并、每标签容量限制、患者多样性限制、活跃度/质量/supported priority、hot/archive 压缩。两层都不是无限无约束增长。

#### 9.5 persistent memory 和 online memory 有什么区别？

回答要点：

> persistent memory 是参数无关经验沉淀，存 JSONL/prototype/cache 文件；online memory 是当前模型运行时的 key/value/label/activity/support 记忆，直接参与注意力检索和预测融合。persistent memory 提供样本和 prototype，online memory 在当前 encoder 下重新编码后使用。

#### 9.6 为什么训练后要重建最终 memory bank？

回答要点：

> 因为 memory key/value 来自当前 encoder。训练过程中 encoder 参数会变，每个 epoch 后都要用当前 encoder 重建 memory。最终模型选择验证集最优 `best_state` 后，必须基于这个 best_state 重建或恢复对应 memory bank，否则 memory 与模型参数不匹配。

#### 9.7 semantic prototype 如何影响模型？

回答要点：

> 它通过三条路径影响：第一，调整 experience label prior；第二，提高 experience 分支 planner weight；第三，把 prototype future curve 编码为 template value，按置信度融合进 experience readout。它不是硬替换预测，而是软先验和软模板。

#### 9.8 反事实 donor 为什么可信？

回答要点：

> donor 不只按 embedding 相似度选，还要经过经验标签、pattern/trajectory、医院/病区/感染锚点、KG 相似度、指南兼容度、hard filter、overlap filter、邻域一致性和可选 transition utility 共同约束。最终还会把候选干预真正输入模型预测，再根据 predicted delta 和安全可行性选择。

#### 9.9 经验记忆是否会造成数据泄漏？

回答要点：

> 需要看实验配置。默认 `persistent-memory-scope dataset` 只按数据集、seasonality、horizon 过滤；如果 store 中已经包含测试窗口，就可能有泄漏风险。因此正式实验应使用只由训练集或历史训练轮次构成的 store，并在报告中记录 `source`、`entry_count` 和 `loaded_persistent_samples`。当前代码提供机制，但实验设计上必须控制 store 的来源。

#### 9.10 如何做消融证明它有效？

建议消融：

- 关闭 persistent memory reuse。
- 关闭 semantic retrieval，只使用 online memory。
- 关闭 neural cache，只比较恢复效率。
- 关闭 KG features。
- 关闭 experience archive。
- 关闭 transition memory。
- `donor_only`、`generated_best`、`safe_search` 三种候选策略对比。
- factual line 比较 `use_memory=True` 和 `use_memory=False`。
- counterfactual line 比较 donor quality、predicted improvement、guideline compatibility、overlap valid rate。

### 10. 当前边界

当前实现仍有几个边界需要主动说明：

- 持久化经验库不是完整病历数据库，只保存任务相关窗口经验。
- semantic index 当前较轻量，prototype 检索仍主要是加载后过滤打分。
- prototype 是均值中心，可能掩盖同一组内的多峰分布。
- 如果 persistent store 混入测试样本，实验会有数据泄漏风险，需要严格控制来源。
- counterfactual 的 predicted improvement 是模型估计，不是随机对照试验因果效应。
- donor 干预来自历史观测数据和规则修复，不能直接等同于临床推荐。
- transition memory 默认可配置启用，其效用依赖样本覆盖和 action 表达质量。

## 已知问题

- 目前文档层面已经可以解释完整链路，但 formal paper 中还需要补充更严格的公式化定义。
- 对经验库有效性的最终证明仍需要系统消融实验，而不是只看单次运行指标。
- 如果要面向论文答辩，建议额外整理一张图，把 persistent store、online memory bank、semantic retrieval、intervention store、counterfactual candidate selection 画在同一流程中。

## 下一步计划

建议后续优先补充以下内容：

1. 为经验库增加数据来源审计字段，明确 entry 来自 train/validation/test/历史外部经验。
2. 对 prototype 检索做消融，报告 semantic hit rate、template blend weight 与 factual MAE 的关系。
3. 对 donor 选择做分层评估，分别统计同标签、同医院、同 KG 状态、不同候选策略下的改善率。
4. 将 prototype 从均值中心扩展为多中心或 medoid，减少多峰经验被平均的问题。
5. 增加正式论文公式：经验条目、prototype 聚合、semantic score、memory readout、counterfactual donor score 和 candidate selection score。

### 11. 组合式反事实干预输入输出框架

本节合并记录最近一次围绕“患者输入、干预库组件组合、反事实输出和医学解释”的代码修改。它是在前面经验库和干预库机制之上的一个可运行验证框架，目标不是马上求出全局最优干预方案，而是先验证完整链路能否跑通：

```text
手工患者模板
  -> 按 bundle schema 补齐缺失特征
  -> 构造 ForecastSample
  -> 从 intervention store 检索相似 donor
  -> 抽取 donor 的 method/dose/timing/context 组件
  -> 生成有限组合式反事实候选
  -> 逐个候选送入模型预测
  -> 选择一个预测上有所改善的候选
  -> 输出 JSON 与医生可读 Markdown 报告
```

#### 11.1 新增脚本入口

新增脚本为：

```text
scripts/generate_compositional_intervention_plan.py
```

脚本复用了现有 `scripts/infer_eicu_counterfactual_plan.py` 中的 bundle 加载、trainer 重建、样本构造和维度校验逻辑。它加载训练阶段导出的 `.pt` inference bundle 后，会恢复 trainer 内部的干预组件库，包括完整干预方案库以及 method、dose、timing、context 等组件子库。

该脚本的核心职责是：

- 接收一个患者 JSON 输入。
- 将患者输入转成模型内部使用的 `ForecastSample`。
- 从当前 bundle 中恢复干预库和相似病例 donor 检索能力。
- 基于 donor 的干预组件生成小规模组合式候选。
- 对每个候选进行反事实预测。
- 输出被选中候选、全部候选、相似 donor、医学解释和安全复核信息。

#### 11.2 患者输入模板

新增患者模板为：

```text
examples/eicu_compositional_patient_template.json
```

模板采用便于手工实验的结构，主要包括：

- `stay_id`：手工病例编号。
- `series_name`：本次推理病例名称。
- `label_row`：患者静态信息、感染状态、脓毒症和休克标签等。
- `context_rows`：最近若干时间窗中的 SOFA、乳酸、血管活性药和呼吸支持等轨迹。
- `future_target`：用于构造样本的未来目标占位。
- `metadata`：实验备注。

由于训练 bundle 的特征列可能比手工模板多，脚本会读取 bundle 中保存的 `sequence_feature_names` 和相关 schema，对模板中缺失的轨迹字段自动补 0。这样做的目的是降低框架验证门槛，但需要注意：自动补 0 只适合快速跑通流程，不应被解释为严肃临床缺失值处理方案。

#### 11.3 干预组件如何被组合

候选生成不再只依赖完整 donor 方案，而是把干预库中的方案拆成可替换组件。当前初步框架使用几类候选：

- 当前方案经过知识约束修补后的候选。
- 直接借用相似 donor 的完整干预方案。
- 保留当前患者方案主体，只替换 donor 的某一类组件，例如方法、剂量或时机。
- 保留当前方案主体，替换 method+dose、method+timing 或 method+dose+timing。
- 从多个 top donor 中分别抽取 method、dose、timing 组件，做有限交叉组合。

这一步的意义是：干预库不仅能提供“历史上某个患者完整用了什么方案”，还可以提供“历史方案中的某个治疗方法、剂量强度或时机模式”，从而支持后续更灵活的反事实干预生成。

当前组合搜索是保守的小规模候选枚举，不做全局优化。这样设计是为了先验证可行性，避免在框架尚未稳定时把主要复杂度放在搜索算法上。

#### 11.4 候选如何被预测和选择

每个候选都会被重新构造成一个 counterfactual `ForecastSample`，保持患者状态、历史轨迹和上下文不变，只替换干预向量或干预序列。随后脚本调用 trainer 的前向预测，得到：

- 当前方案下的未来 SOFA 预测。
- 组合式反事实候选下的未来 SOFA 预测。
- 辅助风险预测，例如未来 SOFA 增量、乳酸变化、后续血管活性药需求概率、呼吸支持升级概率。

脚本用当前方案未来 SOFA 均值减去候选方案未来 SOFA 均值作为 proxy 改善。如果该值为正，说明模型估计候选方案下未来 SOFA 更低，方向上更好。候选最终排序还会结合已有的 candidate selection score、邻域一致性、安全惩罚和 donor 质量信息。

#### 11.5 输出形式

脚本支持两类输出：

- JSON：完整保存输入病例、当前预测、反事实预测、被选中候选、全部候选、donor 信息、guardrail 和医学解释。
- Markdown：面向医生或导师快速查看的中文解释报告。

示例运行命令为：

```powershell
python scripts\generate_compositional_intervention_plan.py `
  --bundle-path output\analysis\phase8_rollout_bundle.pt `
  --input-json examples\eicu_compositional_patient_template.json `
  --output-json output\analysis\compositional_smoke.json `
  --output-md output\analysis\compositional_smoke.md `
  --device cpu `
  --max-donors 3 `
  --max-candidates 12
```

一次烟测结果显示，该模板病例生成了 12 个候选，其中 10 个候选的 proxy 改善为正。最终选中的候选来源为 `composite_current_plus_dose`，当前方案未来 SOFA 预测为 `[8.965, 9.549]`，组合式反事实方案未来 SOFA 预测为 `[8.748, 9.401]`，平均 proxy 改善约为 `0.182`。

#### 11.6 医学解释和边界

医学报告会解释：

- 当前病例的关键状态，例如脓毒症、脓毒性休克、高乳酸。
- 推荐候选来自哪类组合策略。
- 当前方案和反事实方案的短期 SOFA 预测差异。
- 辅助风险指标是否同步改善。
- 主要相似 donor 的相似度、指南一致性和状态匹配度。
- 为什么该结果只能作为医生复核参考，而不能直接等同于临床治疗建议。

需要特别强调：当前输出是“相似病例检索 + 干预组件组合 + 模型预测重排”的结果，不是严格因果效应估计，也不是随机对照试验证明的疗效。后续如果要把它作为论文实验，需要继续补充不确定性估计、OPE/DR 或时间变化混杂校正、批量病例评估和消融实验。
