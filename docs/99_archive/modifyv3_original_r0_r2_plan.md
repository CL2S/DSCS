下面给你一份**面向代码修改的详细方案**。我会按“先保住实验可信度 → 再提升预测性能 → 再补验证与解释 → 最后考虑大模型”的顺序写。核心原则是：

**不要先急着把 LLM 接进预测器。你现在更应该先把经验库从“相似病例检索系统”升级成“状态转移建模 + 自检修复 + 证据验证”的闭环系统。**

你的当前文档已经有很完整的基础：持久化经验库、在线 memory bank、prototype、semantic retrieval、intervention store、donor rerank、KG 约束和组合式反事实候选生成都已经存在。现在要做的不是继续堆模块，而是让这些模块更强地服务 factual forecasting 和 counterfactual reasoning。

---

# 〇、可行性审查与必要修订

## 0.0 总体判断

这个修改方案**总体可行，但不能按原文全部并行推进**。它覆盖了数据审计、transition memory、memory loss、prototype 多中心、自检修复、donor verification、证据级解释、site adaptation 和 LLM controller，方向是完整的；但如果直接一起做，会出现三个问题：

1. **工程面过大**：P1 到 P8 同时展开会改动 `persistent_memory_store.py`、`memory_manager.py`、`manifold_forecasting_trainer.py`、数据构造、推理脚本和报告层，难以定位性能变化来自哪里。
2. **当前代码已经有部分 transition memory**：`EndToEndForecastingManifoldTrainer` 里已有 `TransitionStoreEntry`、`_build_transition_store(...)`、`_retrieve_transition_readout(...)`、`transition_residual`、`transition_store_size` 等运行时结构。因此后续不是“从零新增 transition memory”，而是**把现有 runtime transition store 做成 split-safe、可持久化、可评估、可门控的主线模块**。
3. **现有证据要求先控害再增强**：目前 `phase8_rollout_smoke_16` 中 memory-enabled factual MAE 为 `1.9801`，base-only MAE 为 `1.8515`，`improvement_mae = -0.1286`。这说明当前 memory 对普通预测没有稳定增益，甚至会伤害预测。因此第一轮代码修改不应先扩大 memory 影响力，而应先做审计、消融、gate 降权和安全回退。

所以，本方案需要从“扩展系统能力”改成“先让 memory 可审计、可降权、可证明有效，再逐步扩展能力”。

## 0.1 当前方案中可直接保留的部分

以下部分方向正确，建议保留：

- P0 数据泄漏审计和标准消融框架。
- P1 显式 clinical state 与 transition memory 主线。
- P2 memory residual supervision、contrastive retrieval、gate calibration 的思想。
- P3 prototype 多中心或 medoid 化。
- P5 donor / prototype / candidate verification。
- P6 证据级反事实贡献评分，用于解释和排查 retrieval 是否依赖了正确临床证据。
- P8 LLM 只做审查、修复建议和叙事，不进入数值预测主干。

## 0.2 必须修订的部分

### A. P1 不能写成“新增 transition memory”

当前代码已经有运行时 transition store。后续应修改为：

```text
升级现有 transition store：
  runtime transition store
  -> 增加 clinical_state_signature
  -> 增加 split/source/stay 审计字段
  -> 支持导出到 inference bundle 或 persistent store
  -> 支持 transition utility 的离线评估
  -> 支持 factual path 中的安全门控
```

不要再单独创建一套与 trainer 内部 transition store 平行的 `TransitionMemory` 黑盒类，否则会出现两套 transition 逻辑并存，后续很难维护。

### B. P2 的 gate calibration 不能用测试集结果训练

`memory_gain = abs(base_error) - abs(final_error)` 这个思路可以保留，但只能在训练集或验证集上生成 gate target。不能用 test 误差反过来训练或调参，否则会造成评估泄漏。

更稳妥的实现顺序是：

```text
训练阶段：
  记录 base prediction 与 memory prediction
  用 train/val 构造 memory_gain 标签
  训练 gate calibrator 或校准阈值

推理阶段：
  只使用 similarity、support、entropy、overlap、guideline、uncertainty 等可观测质量信号
  不使用真实 target
```

### C. P2 应先做“memory harm control”，再做更强 memory loss

由于当前 factual 预测证据是 memory 伤害 MAE，下一步最小修改应优先做：

- memory gate 上限；
- low-confidence memory fallback；
- base-only 与 memory-enabled 的逐病例差异输出；
- 按 donor similarity、overlap、experience label match、prototype support 分层报告 memory 是否有益。

只有当这些分层结果显示某些子群 memory 确实有益，再引入更强的 residual supervision。

### D. P4/P5 不应早于 P0/P2

Probe repair 和 verification 很重要，但它们依赖稳定的审计字段、prototype 结构和分层指标。若过早加入，会变成“系统很复杂但不知道哪里有效”。建议放到第二轮或第三轮。

### E. P7/P8 暂时不是预测性能优先项

site-aware adaptation 和 LLM meta-controller 更适合论文后续增强和系统完整性，不适合作为当前“为什么 memory 预测变差、如何提升预测性能”的第一批代码修改。

## 0.3 修订后的执行优先级

建议把原 P0-P8 改成以下顺序：

| 修订优先级 | 后续修改目标 | 原方案对应 | 为什么排在这里 |
|---|---|---|---|
| R0 | 数据来源审计 + 标准消融 + 当前评估集隔离 | P0 | 没有这个，任何 memory 提升都不可信 |
| R1 | memory harm control：gate 上限、低置信回退、逐病例差异输出 | P2 | 当前证据显示 memory 会伤害 factual MAE，必须先控害 |
| R2 | clinical state signature 接入现有 transition store | P1 | 让 transition store 从“隐式向量”变成可解释状态转移 |
| R3 | transition residual 安全接入 factual forecasting | P1/P2 | 只在高支持、高一致性状态下让 transition 影响预测 |
| R4 | prototype 多中心/medoid + outcome entropy | P3 | 解决均值 prototype 抹平 ICU 多峰走向的问题 |
| R5 | donor/prototype/candidate verification | P5 | 避免错误经验长期复用 |
| R6 | probe repair 与 evidence contribution | P4/P6 | 做解释、排错和后续论文增强 |
| R7 | site-aware adaptation 与 LLM meta-controller | P7/P8 | 放到系统稳定后作为增强项 |

---

# 一、优先级总表

| 优先级 | 修改方向                                  | 主要目的                         | 难度 | 灵感来源                                 |
| --- | ------------------------------------- | ---------------------------- | -- | ------------------------------------ |
| P0  | 数据泄漏审计 + 标准消融框架                       | 保证实验可信，否则所有提升都可能不成立          | 中  | ICU distribution shift、你的经验库文档       |
| P1  | 显式 Clinical State + Transition Memory | 把“相似经验”升级为“状态—干预—结果”预测       | 高  | 你已有 Markov transition 方案、MemMA       |
| P2  | 加强 memory 对预测 loss 的约束                | 让 memory 不只是软提示，而是真正学会修正预测   | 中高 | Memory Decoder、你的 memory residual 设计 |
| P3  | Prototype 从均值中心升级为多中心/medoid          | 解决 ICU 异质性被均值抹平的问题           | 中  | ByteRover、你的 prototype 边界分析          |
| P4  | Probe QA / Probe Task 自检与修复           | 让经验库能发现并修复写入和检索失败            | 高  | MemMA                                |
| P5  | Knowledge / Donor Verification Layer  | 避免错误经验、错误 donor、错误 KG 约束污染系统 | 中高 | SHARP、LMKG                           |
| P6  | 证据级反事实贡献评分                            | 解释“哪条临床证据真正影响预测/反事实选择”       | 中  | Counterfactual Multi-Agent Diagnosis |
| P7  | Site-aware Memory Adaptation          | 解决跨医院、跨数据集经验迁移失效             | 中高 | ICU distribution shift               |
| P8  | LLM Meta-Thinker，可选接入                 | 用 LLM 做规划、审查、修复，不直接做数值预测     | 中  | MemMA、SHARP、ByteRover                |

---

# 二、P0：先做数据泄漏审计和标准消融框架

这是第一优先级。你的文档已经明确提到：如果 persistent store 混入测试窗口，经验记忆实验会有数据泄漏风险。这个问题不先解决，后面所有结果都很难说服导师或审稿人。

## 0.1 修改内容

### A. 给每条经验加入更严格的来源字段

建议在 `experience_entries.jsonl` 中新增字段：

```json
{
  "split": "train | val | test | external",
  "source_run_id": "...",
  "source_dataset": "eICU",
  "source_hospital_id": "...",
  "source_unit_type": "...",
  "source_patient_id_hash": "...",
  "source_time_range": "...",
  "is_allowed_for_reuse": true
}
```

当前你已有 `source`、`dataset_name`、`series_name`、`window_end_index` 等字段，但还不够保证“只复用训练来源经验”。

### B. 在 `PersistentExperienceStore.load_samples(...)` 加硬过滤

新增参数：

```python
allowed_splits=("train", "external")
exclude_patient_ids=None
exclude_stay_ids=None
strict_no_test=True
```

加载 persistent samples 时，默认拒绝：

```text
split == test
当前 evaluation stay_id
当前 evaluation patient_id
```

### C. 在 `run_forecasting_experiment.py` 输出泄漏审计报告

建议输出：

```json
"persistent_memory_leakage_audit": {
  "loaded_total": 12000,
  "loaded_train": 12000,
  "loaded_val": 0,
  "loaded_test": 0,
  "excluded_same_stay": 18,
  "excluded_same_patient": 24,
  "strict_no_test": true
}
```

## 0.2 为什么必须先做

ICU sepsis distribution shift 论文强调，跨中心、跨站点的性能很容易因数据分布和部署场景变化而失效；如果经验库来源不清楚，模型可能不是“学会泛化”，而是在隐式复用评估集经验。该文比较了不同 target data regime 和部署策略，也明确说明 routine fine-tuning 并不总是可靠，部署策略必须按数据来源和目标环境来定。

## 0.3 验收标准

修改后必须能回答：

```text
本次实验加载的 persistent memory 中，有多少条来自 train？
有没有 test？
有没有当前 stay_id？
有没有当前 patient？
```

---

# 三、P1：构建显式 Clinical State，并升级现有 Transition Memory 主线

这是最重要的性能改进方向。当前系统已经能回答“当前病例像谁”，但还没有足够强地回答：

[
P(z_{t+1} \mid z_t, a_t)
]

也就是：**在当前状态下，采取某类干预，下一状态通常会怎样变化。**

你之前的修改方案已经提出要把当前窗口升级为显式临床状态 `z_t`，并用 `(z_t, a_t, z_{t+1}, utility_t, support, confidence, guideline_score)` 形式组织 transition memory。这是非常正确的方向。

但需要注意当前代码事实：`src/manifold_forecasting_trainer.py` 中已经存在运行时 `TransitionStoreEntry`、`_build_transition_store(...)`、transition cache、transition readout 和 `transition_residual`。所以这里的任务不是另起炉灶新增一套 transition memory，而是**升级现有 transition store**：

```text
现有 runtime transition store
  -> 增加显式 clinical state signature
  -> 增加 split/source/stay 审计字段
  -> 增加 transition utility 质量统计
  -> 增加低置信门控和 base-only fallback
  -> 再考虑持久化 transition_entries.jsonl
```

---

## 1.1 新增 `ClinicalStateBuilder`

建议新增文件：

```text
src/clinical_state.py
```

核心类：

```python
@dataclass
class ClinicalState:
    z_cont: np.ndarray
    state_flags: Dict[str, bool]
    trend_features: Dict[str, float]
    intervention_coverage: Dict[str, float]
    state_signature: str
    severity_bucket: str
    site_signature: str
```

### `z_cont` 来自

* 当前 encoder embedding
* formation features
* KG features
* patient_static
* intervention_static summary

### `state_flags` 包含

```text
sepsis
septic_shock
hypotension
high_lactate
organ_dysfunction
vasopressor_on
antibiotic_covered
blood_culture_done
respiratory_support
renal_dysfunction
```

### `trend_features` 包含

```text
sofa_delta
lactate_delta
map_delta
vasopressor_delta
resp_support_delta
urine_output_delta
```

### `state_signature` 示例

```text
shock_high_lactate_vaso_on_abx_covered_sofa_rising
```

## 1.2 在 `ForecastSample.metadata` 中保存 clinical state

在 `src/tsf_data.py` 或构造 eICU dataset 的流程中，生成：

```python
sample.metadata["clinical_state"] = clinical_state.to_dict()
sample.metadata["state_signature"] = clinical_state.state_signature
```

你当前 `ForecastSample` 已经包含 sequence、static、target、metadata、formation features、patient_static、intervention_static、kg_features 和 auxiliary targets，因此加 clinical state 是自然扩展。

---

## 1.3 持久化现有 transition store，而不是重复新增并行实现

当前 trainer 内已有 runtime transition store。后续如果要跨实验复用，才在 `PersistentExperienceStore` 中新增：

```text
transition_entries.jsonl
transition_prototypes.jsonl
```

单条 transition entry 建议从现有 `TransitionStoreEntry` 扩展而来：

```json
{
  "transition_id": "...",
  "experience_id": "...",
  "dataset_name": "eICU",
  "split": "train",
  "site_signature": "...",

  "state_t_signature": "...",
  "state_t_vector": [...],
  "action_signature": "...",
  "action_vector": [...],
  "state_next_signature": "...",
  "state_next_vector": [...],

  "delta_target": [...],
  "delta_sofa": -1.0,
  "delta_lactate": -0.4,
  "delta_vasopressor_need": -0.2,
  "delta_resp_support": 0.0,

  "short_term_clinical_utility": 0.73,
  "guideline_score": 0.82,
  "support": 1,
  "quality_score": 0.91,
  "created_at": "..."
}
```

---

## 1.4 定义 short-term clinical utility

不要直接叫 reward，论文中更稳妥地叫：

```text
short_term_clinical_utility
```

建议先用组合代理指标：

[
U_t =
-w_1 \Delta SOFA
-w_2 \Delta Lactate
-w_3 \Delta VasopressorNeed
-w_4 \Delta RespSupport
+w_5 ShockRemission
]

代码中可以实现为：

```python
utility = (
    - w_sofa * delta_sofa_norm
    - w_lactate * delta_lactate_norm
    - w_vaso * delta_vaso_need
    - w_resp * delta_resp_support
    + w_shock * shock_remission_flag
)
```

注意：
这里是短期代理效用，不是严格临床疗效，也不是因果 reward。你的文档中已经明确 counterfactual predicted improvement 只是模型估计，不能等同于随机对照试验证明的疗效。

---

## 1.5 抽象现有 transition readout 接口

当前 transition retrieval 已经在 trainer 内部实现。后续可以把它逐步抽象成更清晰的接口，但不建议一开始就新增一套独立黑盒类。更稳妥的顺序是：

```text
第一步：保留 trainer 内部实现，只补诊断输出和门控。
第二步：当接口稳定后，再抽成 TransitionMemory 或 TransitionStoreReader。
第三步：保证原有 predict_counterfactual 和 factual forward 共用同一套 readout 逻辑。
```

目标接口可以是：

```python
class TransitionStoreReader:
    def add_transition(...)
    def read(state_query, action_query=None, top_k=20)
    def estimate_utility(state_query, action_query)
    def estimate_next_delta(state_query, action_query)
```

输出：

```python
{
  "transition_readout": tensor,
  "expected_delta": tensor,
  "expected_utility": float,
  "confidence": float,
  "support": int,
  "matched_transition_ids": [...]
}
```

---

## 1.6 接入 factual forecasting

在 `src/manifold_forecasting_trainer.py` 中，你当前最终预测类似：

```text
fusion_prediction = base_prediction + coordinated_memory_residual + transition_residual
```

现在要让 `transition_residual` 不再只是附属项，而是明确来自现有 transition readout。示意如下，实际实现时应优先复用 `_retrieve_transition_readout(...)` 和 `_transition_factual_residual(...)`：

```python
transition_result = self._retrieve_transition_readout(
    state_query=clinical_state_embedding,
    action_query=current_intervention_embedding
)

transition_residual = transition_gate * transition_head(
    transition_result["transition_readout"]
)
```

同时增加一个辅助损失：

```python
L_transition_delta = SmoothL1(predicted_delta, true_delta)
L_transition_utility = MSE(predicted_utility, true_utility)
```

总损失：

[
L = L_{forecast}

* \lambda_1 L_{transition_delta}
* \lambda_2 L_{transition_utility}
  ]

## 1.7 灵感来源

* 你的已有方案中已经明确提出 `z_t, a_t, z_{t+1}, utility_t, support, confidence, guideline_score` 的 transition memory，这是当前方法最应该实现的主线。
* MemMA 的核心启发是 memory 不能只看存储和检索，而要看 construction、retrieval、utilization 的闭环；transition memory 就是把 utilization 中的未来变化反过来结构化存储。

---

# 四、P2：加强 memory 对预测 loss 的直接约束

你现在的 memory signal 主要通过 semantic prior、template curve、planner weight、readout residual、direct residual 等方式影响预测。问题是这些路径可能太软，被 base model 吞掉。

要提高性能，需要让 memory path 有更明确的训练目标。

但根据当前已有结果，`phase8_rollout_smoke_16` 中 memory-enabled factual MAE 比 base-only 更差。因此这一节的第一目标应先改为：

```text
先控制 memory 对 factual forecasting 的伤害，
再让 memory 学会稳定修正 base model。
```

也就是说，P2 的第一批代码不应直接增加复杂 loss，而应先增加：

- base-only 与 memory-enabled 的逐病例误差差异输出；
- memory gate 上限；
- low-confidence memory fallback；
- 按 similarity / support / entropy / overlap / guideline 分层的 memory gain 报告。

---

## 2.1 新增 memory residual supervision

当前模型有：

```text
base_prediction
memory_residual
final_prediction
```

建议显式定义：

[
r^* = y - \hat{y}_{base}
]

让 memory branch 学 residual：

```python
residual_target = target - base_prediction.detach()
memory_residual_pred = memory_residual_head(memory_readout)

L_memory_residual = SmoothL1(memory_residual_pred, residual_target)
```

这样 memory 的任务不再是模糊地“帮一下预测”，而是明确学习：

> 当前 base model 没预测好的部分，能否由历史经验修正？

---

## 2.2 新增 prototype template consistency loss

当 semantic retrieval 置信度高时，prototype future curve 不应只是融合进 readout，还应对预测方向有弱监督：

```python
if template_confidence > tau:
    L_template = SmoothL1(
        predicted_future_direction,
        prototype_future_direction
    )
```

或者更稳妥：

```python
L_template_direction = CE(
    predicted_direction_class,
    prototype_future_direction_type
)
```

不要直接强制预测等于 prototype curve，因为 prototype 只是经验模板，不是当前患者真实未来。

---

## 2.3 新增 contrastive retrieval loss

目的：让 embedding 的“相似”更接近“未来走势相似”，而不只是输入形态相似。

正样本：

```text
same experience_label
same future_direction
similar KG state
similar intervention signature
```

负样本：

```text
same pattern but opposite future_direction
same severity but different intervention response
```

损失：

```python
L_contrastive = supervised_contrastive_loss(
    query_embedding,
    positive_embeddings,
    negative_embeddings
)
```

这一步很关键，因为当前 performance 不强，很可能是“检索到的相似”不是“对预测有用的相似”。

---

## 2.4 新增 gate calibration loss

你需要知道 memory gate 什么时候该开、什么时候该关。

可以定义：

```python
memory_gain = abs(base_error) - abs(final_error)
```

如果 memory_gain > 0，说明 memory 有帮助；如果 < 0，说明 memory 伤害预测。

训练一个 gate target：

```python
gate_target = 1 if memory_gain > margin else 0
L_gate = BCE(memory_gate, gate_target)
```

这可以减少 memory 在低置信场景下乱干预。

实现时必须加一个边界：`gate_target` 只能来自训练集或验证集，不能来自测试集。推理时不能使用真实 target，也不能使用 test error。更安全的实现是先训练一个小的 gate calibrator，用以下可观测质量信号预测 memory 是否可能有益：

```text
prototype_support
prototype_outcome_entropy
semantic_top_score
donor_similarity
donor_overlap_valid
guideline_compatibility
memory_uncertainty
```

如果这些信号低于阈值，factual forecasting 应回退到 base-only 或只允许很小的 memory residual。

---

## 2.5 灵感来源

Memory Decoder 的核心思想是让小型 memory module 学会模拟检索器的领域分布，并在推理时与 base model 做分布插值；它的启发不是让你直接复制语言模型结构，而是提醒你：**memory module 必须有明确的对齐目标，而不是只靠最终任务损失弱监督。**

---

# 五、P3：把 prototype 从均值中心升级为多中心或 medoid

你当前 prototype 保存的是 `prototype_future_mean_curve`、`prototype_formation_center`、`prototype_kg_center` 等均值中心。文档中也已经承认：均值中心可能掩盖同一组内的多峰分布。

在 ICU 场景，这个问题很严重。因为同一类 sepsis/shock 状态下，患者可能走向改善、稳定或恶化三个方向，均值会把这些全部抹平。

---

## 3.1 修改 prototype 数据结构

当前：

```json
{
  "prototype_id": "...",
  "prototype_future_mean_curve": [...],
  "support": 42
}
```

建议改为：

```json
{
  "prototype_id": "...",
  "group_key": "...",
  "subtype": "improving | stable | worsening | mixed",
  "medoid_experience_id": "...",
  "member_experience_ids": [...],

  "future_curve_medoid": [...],
  "future_curve_mean": [...],
  "future_curve_std": [...],

  "outcome_entropy": 0.37,
  "state_center": [...],
  "kg_center": [...],
  "intervention_center": [...],

  "support": 18,
  "confidence": 0.82,
  "maturity": "draft | validated | core"
}
```

---

## 3.2 重建 prototype 时做 outcome-aware clustering

在 `rebuild_prototypes()` 中，先按原 group_key 分组，再在组内做二次聚类：

聚类输入：

```text
normalized_future
delta_target
clinical_state_vector
intervention_vector
kg_features
```

聚类目标：

```text
同一大类下拆分 improvement / stable / deterioration
```

简单实现：

```python
if group_size >= min_cluster_size:
    clusters = KMeans(n_clusters=min(3, estimated_modes))
else:
    use single prototype
```

更稳实现：

```python
cluster by future_direction_type first
then medoid within each direction
```

---

## 3.3 检索时引入 uncertainty gating

如果 prototype 的 outcome_entropy 高，说明这个 prototype 内部不一致，应降低其模板融合权重：

```python
template_blend_weight *= (1 - outcome_entropy)
```

这样可以避免高不确定 prototype 强行影响预测。

---

## 3.4 灵感来源

ByteRover 的 Context Tree 强调每个知识条目要有 relations、provenance、narrative、snippets 和 lifecycle metadata，而不是普通碎片；它还用 importance、maturity、recency 管理知识生命周期。你的 prototype 也需要从“均值向量”升级成“有成熟度、有不确定性、有代表样本的经验对象”。

---

# 六、P4：加入 Probe Task 自检与 Memory Repair

这是把你的系统从“强 memory pipeline”升级为“memory 闭环系统”的关键。

MemMA 明确指出，现有 memory 系统有两个问题：前向路径 strategic blindness，后向路径 sparse and delayed feedback。它的解决方式是在 session 后生成 probe QA，验证当前 memory，并把失败转成修复动作。

你可以不直接用自然语言 QA，而是先做结构化 probe task，更适合你的时序预测系统。

---

## 4.1 新增 `memory_probe.py`

建议新增：

```text
src/memory_probe.py
```

核心类：

```python
class MemoryProbeGenerator:
    def generate_retrieval_probes(sample, memory_store): ...
    def generate_transition_probes(sample, transition_store): ...
    def generate_donor_probes(sample, intervention_store): ...
```

---

## 4.2 设计四类 probe

### A. 经验检索 probe

问题：

```text
给定当前 sample 的 state_signature，能否检索到同 experience_label 且 future_direction 相同的经验？
```

失败说明：

```text
prototype 分组过粗、semantic score 不合理、经验条目写入不足
```

### B. 未来趋势 probe

问题：

```text
给定当前 history + KG state，prototype top-k 中是否包含正确 future_direction_type？
```

失败说明：

```text
prototype future curve 对当前状态无效
```

### C. transition probe

问题：

```text
给定 z_t 和 a_t，transition memory 是否能估计 z_t+1 的方向？
```

失败说明：

```text
transition entry 缺失或 state/action signature 不稳定
```

### D. donor probe

问题：

```text
给定 query patient，intervention store 是否能找到至少一个 guideline-compatible donor？
```

失败说明：

```text
donor pool 太窄、索引不足、KG filter 过严
```

---

## 4.3 Probe 失败后生成 repair action

新增 repair action：

```text
INSERT_ENTRY
MERGE_ENTRY
SPLIT_PROTOTYPE
DEGRADE_PROTOTYPE
PROMOTE_PROTOTYPE
ADD_ALIAS_SIGNATURE
RELABEL_EXPERIENCE
REWEIGHT_DONOR
```

示例：

```python
if probe_type == "future_direction" and failed:
    if prototype.outcome_entropy > threshold:
        repair = SPLIT_PROTOTYPE
    elif support < min_support:
        repair = DEGRADE_PROTOTYPE
```

---

## 4.4 在 `events.jsonl` 记录修复历史

新增事件：

```json
{
  "event_type": "probe_repair",
  "probe_id": "...",
  "failed_reason": "...",
  "repair_action": "SPLIT_PROTOTYPE",
  "affected_ids": ["proto_xxx"],
  "created_at": "..."
}
```

---

## 4.5 灵感来源

MemMA 在 backward path 中通过 probe QA、in-situ verification、evidence-grounded repair 和 semantic consolidation，把下游失败转成当前 memory 的局部修复信号。你这里可以把自然语言 QA 改成结构化 probe task，更符合 ICU 时序预测。

---

# 七、P5：新增 Knowledge / Donor Verification Layer

你现在有 KG similarity、guideline compatibility、missing care penalty、contraindication penalty 等，但它们更多是打分项，不是完整验证层。

需要新增一个明确的 verification stage，避免错误经验、错误 prototype、错误 donor 被长期复用。

---

## 5.1 新增 `memory_verifier.py`

建议包含：

```python
class MemoryVerifier:
    def verify_experience_entry(entry): ...
    def verify_prototype(prototype): ...
    def verify_transition(entry): ...
    def verify_donor_candidate(query, donor): ...
    def verify_counterfactual_candidate(query, candidate): ...
```

每个函数输出：

```json
{
  "verification_status": "accept | review | reject",
  "verification_score": 0.83,
  "failed_checks": [],
  "evidence_chain": [],
  "recommended_action": "keep | lower_weight | require_review | reject"
}
```

---

## 5.2 经验条目验证

检查：

```text
是否来自允许 split
是否有 raw_context/raw_future
是否有 KG flags
是否有 intervention signature
是否有 target_signature
是否质量分低于阈值
```

---

## 5.3 prototype 验证

检查：

```text
support 是否足够
outcome_entropy 是否过高
组内未来方向是否冲突
medoid 是否真实存在
KG state 是否一致
intervention signature 是否混乱
```

---

## 5.4 donor 验证

检查：

```text
是否同一 stay_id
是否跨 split 泄漏
KG state 是否明显不匹配
guideline score 是否过低
contraindication penalty 是否过高
overlap 是否足够
neighbor consistency 是否支持
```

---

## 5.5 反事实候选验证

检查：

```text
候选是否只是数值组合，没有历史支持
method/dose/timing/context 是否组合后失真
predicted improvement 是否伴随风险指标恶化
auxiliary targets 是否一致改善
```

---

## 5.6 灵感来源

SHARP 把知识图谱三元组验证从静态分类改造成“schema-aware planning + internal KG tools + external evidence”的动态核验过程，并证明去掉 schema-aware planning、KG tools 或 external tools 都会明显降性能。你这里要借鉴的是：**经验库也需要结构约束和证据链，而不是只靠相似度。**

LMKG 的启发是：医学知识层要有细粒度实体和关系类型、来源和管理机制，而不是只用一组 KG vector。它构建了大规模医疗 KG，并强调细粒度实体/关系类型对下游医疗任务的支撑价值。

---

# 八、P6：加入证据级反事实贡献评分

你现在的反事实主要是 action-level：

```text
替换 donor 干预方案 → 预测未来 SOFA 变化
```

但还缺 evidence-level：

```text
去掉某个关键证据 → 预测或 donor ranking 是否改变
```

这会显著增强解释性，也能帮助你判断模型到底依赖哪些临床证据。

---

## 6.1 新增 `evidence_counterfactual.py`

建议支持 evidence group：

```text
lactate_group
map_group
vasopressor_group
antibiotic_group
infection_evidence_group
sofa_component_group
respiratory_support_group
renal_function_group
kg_flag_group
```

---

## 6.2 定义 Sepsis Evidence Contribution Score

对于 factual forecasting：

[
ECS_g = |\hat y(x) - \hat y(x_{\setminus g})|
]

对于 risk 分类：

[
ECS_g = |P(risk|x) - P(risk|x_{\setminus g})|
]

对于 donor ranking：

[
ECS^{donor}*g = 1 - KendallTau(rank(x), rank(x*{\setminus g}))
]

输出：

```json
{
  "evidence_group": "lactate_group",
  "forecast_delta": 0.42,
  "risk_probability_delta": 0.18,
  "donor_rank_shift": 0.31,
  "interpretation": "lactate evidence strongly supports high-risk forecast"
}
```

---

## 6.3 用 ECS 反过来检查 memory retrieval

如果当前样本的最高 ECS 是 lactate/MAP，但 semantic retrieval 返回的 prototype 主要由 respiratory features 驱动，说明 retrieval 可能不对。

可以新增一致性指标：

```python
retrieval_evidence_alignment = overlap(
    top_evidence_groups,
    prototype_key_features
)
```

---

## 6.4 灵感来源

Counterfactual Multi-Agent Diagnosis 论文通过 counterfactual case editing 改动关键临床发现，并用 Counterfactual Probability Gap 衡量诊断置信变化；核心思想是：不要只问模型“哪些证据重要”，而要看“改掉证据后模型是否真的改主意”。这非常适合迁移到 sepsis 风险解释和 donor 选择解释。

---

# 九、P7：把 site-aware 从“索引过滤”升级成“适配方法”

你现在已有 hospital、unit type、infection anchor 的 donor 索引和 pool match reward，这很好。但这还只是 retrieval 层面的 site-aware，不是模型层面的 domain adaptation。

---

## 7.1 拆分三类 memory bank

建议把 persistent memory 拆成：

```text
shared_memory
source_site_memory
target_site_memory
```

每条 entry 增加：

```json
{
  "memory_scope": "shared | source | target",
  "site_id": "...",
  "site_distribution_signature": "...",
  "target_regime": "none | small | medium | large"
}
```

---

## 7.2 定义 site weighting

检索 prototype/donor 时，不再只靠同 hospital reward，而是用：

[
w_{site} = \exp(-D(source, target)) \cdot f(n_{target})
]

其中 (D) 可以是：

* feature mean/std distance
* MMD
* CORAL distance
* label prevalence difference

---

## 7.3 加入 domain alignment loss

对 encoder embedding 加：

```python
L_mmd = MMD(source_embeddings, target_embeddings)
L_coral = CORAL(source_embeddings, target_embeddings)
```

只在跨数据集或跨医院实验启用。

---

## 7.4 按 target data regime 选择策略

实现规则：

```text
target data none:
  use source/shared memory, stronger uncertainty penalty

target data small:
  source memory + small target calibration + donor verification

target data medium:
  add MMD/CORAL domain adaptation

target data large:
  target memory dominant, source memory only as archive/reference
```

---

## 7.5 灵感来源

ICU sepsis distribution shift 论文比较了 generalization、fine-tuning/retraining、target training、supervised DA 和 fusion training，并指出 fine-tuning 并不总是最好；small、medium、large target data regimes 下最优策略不同。因此你的经验库也应按目标医院数据量调整复用策略。

---

# 十、P8：大模型应该怎么接入，而不是盲目接入

我不建议你把 LLM 直接接进 SOFA 数值预测主干。更合理的接法是：

```text
LLM = Meta-Thinker / Verifier / Repair Agent / Narrative Generator
```

---

## 8.1 LLM Meta-Thinker 的输入输出

新增：

```text
src/llm_meta_controller.py
```

输入：

```json
{
  "current_sample_summary": "...",
  "top_prototypes": [...],
  "top_donors": [...],
  "memory_diagnostics": {...},
  "failed_probes": [...]
}
```

输出必须是 JSON：

```json
{
  "construction_guidance": {
    "retain": [],
    "merge": [],
    "resolve_conflict": [],
    "lower_quality": []
  },
  "retrieval_gap": {
    "missing_evidence": ["infection evidence", "post-fluid response"],
    "next_memory_layer": "trajectory_memory | kg_layer | intervention_store"
  },
  "repair_actions": [
    {
      "action": "SPLIT_PROTOTYPE",
      "target_id": "proto_xxx",
      "reason": "mixed future directions under same state signature"
    }
  ]
}
```

---

## 8.2 LLM 不参与最终临床推荐

LLM 输出只能作为：

```text
候选 repair proposal
候选 verification explanation
候选 narrative
```

不直接决定：

```text
最终治疗方案
最终 predicted delta
最终 clinical recommendation
```

---

## 8.3 灵感来源

MemMA 的 Meta-Thinker 用于指导 construction 和 retrieval，并通过 probe QA 失败修复 memory；SHARP 用 agent 做 schema-aware planning 和多源证据验证；ByteRover 强调 memory operation 应成为 agent 的内部工具，而不是外部黑盒服务。你应把 LLM 放在这些位置，而不是放到数值预测器里。

---

# 十一、建议的代码修改顺序

## 第一阶段：先保证实验可信，并量化 memory 是否伤害预测

### 修改文件

```text
src/persistent_memory_store.py
run_forecasting_experiment.py
docs/05_experience_memory/
scripts/run_eicu_phase0_baseline.py
```

### 完成事项

1. 增加 split/source/site 审计字段。
2. load persistent memory 时禁止 test/当前 stay 进入。
3. 输出 leakage audit。
4. 建立标准消融脚本。
5. 输出逐病例 `memory_gain = base_abs_error - memory_abs_error`。
6. 输出分层 memory gain：按 pattern、trajectory、donor similarity、overlap、prototype support、guideline compatibility 分层。
7. 增加 memory gate 上限和 low-confidence fallback 的实验开关。

### 标准消融

```text
Base
+ online memory
+ persistent samples
+ semantic prototypes
+ gate clamp / fallback
+ KG features
+ transition memory
+ self-repair
+ verification
```

---

## 第二阶段：实现 Clinical State，并升级现有 Transition Memory

### 修改文件

```text
src/clinical_state.py
src/persistent_memory_store.py
src/memory_components.py
src/memory_manager.py
src/manifold_forecasting_trainer.py
```

### 完成事项

1. 生成 clinical state。
2. 把 `state_signature`、`severity_bucket`、`site_signature` 写入 `ForecastSample.metadata`。
3. 扩展现有 `TransitionStoreEntry` 的 metadata，而不是新建平行 transition store。
4. 增加 transition readout 的 support、confidence、utility 分层诊断。
5. 在 factual forecasting 中只允许高置信 transition residual 生效。
6. donor reranking 接入 transition utility，但必须保留关闭开关做消融。

---

## 第三阶段：加强 memory training objective

### 修改文件

```text
src/manifold_forecasting_trainer.py
src/manifold_memory.py
src/memory_components.py
```

### 完成事项

1. memory residual supervision。
2. template consistency loss。
3. contrastive retrieval loss。
4. gate calibration loss。

---

## 第四阶段：prototype 多中心化

### 修改文件

```text
src/persistent_memory_store.py
src/memory_manager.py
```

### 完成事项

1. group 内二次聚类。
2. 保存 medoid。
3. 保存 outcome entropy。
4. 检索时根据 entropy 降低 template weight。

---

## 第五阶段：自检与修复

### 修改文件

```text
src/memory_probe.py
src/memory_repair.py
src/persistent_memory_store.py
```

### 完成事项

1. 生成 structured probe。
2. 运行 probe verification。
3. 失败转 repair action。
4. repair 写入 events。
5. repair 后重建 prototypes。

---

## 第六阶段：verification layer

### 修改文件

```text
src/memory_verifier.py
src/manifold_forecasting_trainer.py
scripts/generate_compositional_intervention_plan.py
```

### 完成事项

1. entry verification。
2. prototype verification。
3. donor verification。
4. candidate verification。
5. 输出 evidence chain。

---

## 第七阶段：证据级反事实解释

### 修改文件

```text
src/evidence_counterfactual.py
src/manifold_forecasting_trainer.py
scripts/generate_compositional_intervention_plan.py
```

### 完成事项

1. 定义 evidence group。
2. 做 ablation/perturbation。
3. 计算 ECS。
4. 输出解释报告。

---

## 第八阶段：site-aware adaptation

### 修改文件

```text
src/domain_adaptation.py
src/persistent_memory_store.py
src/manifold_forecasting_trainer.py
```

### 完成事项

1. source/shared/target memory 分层。
2. MMD/CORAL alignment。
3. site weighting。
4. target regime 策略选择。

---

# 十二、最终推荐的最小可行修改版本

你不需要一口气全做。最小可行版本建议只做这六件：

## MVP-1：split-safe persistent memory

确保无泄漏。范围不只包括 persistent samples，也包括 runtime memory、intervention store、transition store 和 inference bundle 中可复用条目的来源审计。

## MVP-2：memory harm control

先处理当前已经观察到的 factual 预测变差问题。具体包括 gate 上限、低置信回退、逐病例 memory gain 输出和分层 memory gain 报告。验收标准不是“结构更复杂”，而是至少能回答：

```text
哪些病例 memory 有帮助？
哪些病例 memory 伤害预测？
伤害是否集中在低 support、低 overlap、高 entropy 或跨 site 场景？
```

## MVP-3：clinical state signature + 现有 transition store 升级

把当前窗口变成显式临床状态，而不是纯 embedding；同时把现有 runtime transition store 升级为可诊断、可门控、可后续持久化的状态转移模块。

## MVP-4：safe transition residual

让 transition residual 只在高 support、高 confidence、低 uncertainty 的情况下影响 factual forecasting。默认必须保留 base-only fallback。

## MVP-5：prototype 多中心化

解决均值 prototype 抹平异质性。

## MVP-6：memory residual supervision

在完成前面审计和控害后，再让 memory 明确学习 base model 的误差修正。否则过早加 residual loss，可能只是把错误 memory 更强地注入预测。

这六件做完，你的方法主线会明显从：

```text
相似经验增强预测
```

升级为：

```text
临床状态转移经验增强预测
```

这比简单“接入大模型”更可能带来真实性能提升。

---

# 十三、最后的判断

**你下一步代码修改的核心，不是把系统变得更复杂，而是把经验库对预测的作用变得更直接、更可验证。**

最关键的三条改法是：

1. **Clinical State + Transition Memory**
   解决“相似病例不等于病程转移”的问题。

2. **Memory-specific Loss**
   解决“memory 只是软提示，预测器不一定真的用它”的问题。

3. **Probe Repair + Verification**
   解决“经验库错了也会继续被复用”的问题。

这些修改分别对应：

* MemMA 的 memory cycle 与 self-evolution；
* Memory Decoder 的 memory 对齐思想；
* SHARP 的 schema-aware verification；
* Counterfactual Diagnosis 的证据反事实检验；
* ICU distribution shift 的 site-aware 部署策略。

---

# 本轮 R0 / MVP-1 修改记录：split-safe persistent memory（2026-05-09）

## 1. 修改目标

本轮从修改方案第一条开始执行，目标不是提升模型指标，而是先保证后续所有经验库实验具备可信的数据边界。具体目标是：持久化经验库写入时记录来源；读取时默认只复用训练集或外部来源经验；实验输出中给出泄漏审计，说明到底加载了多少条经验、排除了多少条 test / val / 当前评估 stay 或 patient 经验。

## 2. 代码修改内容

### 2.1 `src/persistent_memory_store.py`

已完成以下修改：

- 每条写入 `experience_entries.jsonl` 的经验新增来源审计字段，包括 `split`、`source_dataset`、`source_run_id`、`source_hospital_id`、`source_unit_type`、`source_patient_id_hash`、`source_stay_id`、`source_stay_id_hash`、`is_allowed_for_reuse`。
- `split` 默认从样本 metadata 读取；如果旧数据没有该字段，则从 `source` 推断。例如 `train_prefit_prime`、`train` 推断为 `train`，`test` 推断为 `test`，`val` 或 `valid` 推断为 `val`。
- `load_samples(...)` 新增安全读取逻辑：默认只允许 `train, external`；默认严格拒绝 test；默认可传入当前 validation/test 的 stay 或 patient 标识并排除同 stay / 同 patient 经验。
- 对旧版 store 做了 stay 标识兼容：`123.0`、`123`、`stay_123` 会作为同一 stay 的别名参与排除，避免旧经验只有 `series_name=stay_xxx` 时绕过审计。
- `load_samples(..., return_audit=True)` 会返回审计报告，字段包括 `scanned_total`、`matched_scope`、`loaded_total`、`loaded_by_split`、`excluded_test_split`、`excluded_disallowed_split`、`excluded_same_stay`、`excluded_same_patient` 等。
- `rebuild_prototypes()` 现在会把 prototype 的成员 split 汇总到 `source_splits` 和 `member_count_by_split`，并标记 `is_allowed_for_reuse`。
- `load_prototypes(...)` 也加入 split-safe 过滤，避免语义 prototype 从 test 或未知来源中被复用。

### 2.2 `run_forecasting_experiment.py`

已完成以下修改：

- 新增命令行控制：
  - `--persistent-memory-allowed-splits`：默认 `train,external`。
  - `--disable-persistent-memory-strict-no-test`：仅用于调试或复现实验，关闭后才允许非严格模式。
  - `--disable-persistent-memory-exclude-current-eval`：仅用于消融，关闭当前验证/测试 stay 与 patient 排除。
- 实验启动时，会自动从当前 `val_samples + test_samples` 提取 stay / patient 标识，用作 persistent memory 读取排除列表。
- 输出 JSON 的 `persistent_memory` 中新增：
  - `allowed_splits`
  - `strict_no_test`
  - `exclude_current_eval`
  - `excluded_eval_stay_count`
  - `excluded_eval_patient_count`
  - `reuse_audit`
  - `prototype_audit`
- 推理 bundle 中也记录了 persistent memory 的安全配置，便于后续复现实验。

## 3. 已运行验证

### 3.1 语法验证

命令：

```powershell
python -m py_compile .\src\persistent_memory_store.py .\run_forecasting_experiment.py
```

结果：通过，无语法错误。

### 3.2 CLI 参数验证

命令：

```powershell
python .\run_forecasting_experiment.py --help
```

结果：通过，新的 persistent memory 参数已出现在帮助信息中。

### 3.3 最小泄漏审计烟雾测试

构造了 4 条持久化经验：

- 1 条 `train`，应当加载；
- 1 条 `test`，应当被严格排除；
- 1 条 `val`，应当因不在默认允许 split 中被排除；
- 1 条 `train` 但 stay 与当前评估 stay 重合，应当被排除。

观察到的核心结果：

```json
{
  "loaded_total": 1,
  "loaded_by_split": {"train": 1},
  "excluded_test_split": 1,
  "excluded_disallowed_split": 1,
  "excluded_same_stay": 1,
  "strict_no_test": true
}
```

说明本轮新增的 split 过滤、test 拒绝和当前评估 stay 排除在最小构造样例中按预期工作。

## 4. 后续需要运行的正式验证实验

### 4.1 安全持久化记忆基线实验

目的：验证在默认安全过滤下，persistent memory 不会加载 test / val / 当前评估 stay 经验。

命令模板：

```powershell
python .\run_forecasting_experiment.py `
  --dataset-format eicu_sepsis3 `
  --eicu-max-series 128 `
  --epochs 3 `
  --batch-size 16 `
  --persistent-memory-store .\output\persistent_memory\eicu_safe_store `
  --prime-persistent-memory-before-fit `
  --persistent-memory-allowed-splits train,external `
  --output-json .\output\formal\r0_split_safe_memory\eicu_safe_memory.json
```

预期结果：

- `persistent_memory.strict_no_test = true`
- `persistent_memory.allowed_splits = ["train", "external"]`
- `persistent_memory.reuse_audit.loaded_by_split` 中只应出现 `train` 或 `external`
- `persistent_memory.reuse_audit.excluded_test_split` 可以大于等于 0，但最终 `loaded_by_split.test` 不应存在
- `persistent_memory.reuse_audit.excluded_same_stay` 如果大于 0，说明当前评估 stay 排除机制实际生效

### 4.2 安全过滤消融实验

目的：对比“默认安全过滤”和“关闭当前评估排除”时，加载经验数量和指标是否发生异常变化，用来检查是否存在 stay 级泄漏风险。

命令模板：

```powershell
python .\run_forecasting_experiment.py `
  --dataset-format eicu_sepsis3 `
  --eicu-max-series 128 `
  --epochs 3 `
  --batch-size 16 `
  --persistent-memory-store .\output\persistent_memory\eicu_safe_store `
  --persistent-memory-allowed-splits train,external `
  --disable-persistent-memory-exclude-current-eval `
  --output-json .\output\formal\r0_split_safe_memory\eicu_no_eval_exclusion.json
```

预期结果：

- 如果关闭当前评估排除后 `loaded_persistent_samples` 明显增加，并且 MAE 异常改善，则提示之前可能存在 stay/patient 级泄漏风险。
- 如果加载数量和指标变化很小，则说明当前 store 本身较干净，但仍应保留默认安全过滤作为正式实验设置。

### 4.3 与 base-only / memory-disabled 对照

目的：确认本轮修改不是为了直接提高性能，而是为了让后续性能比较可信。

命令模板：

```powershell
python .\run_forecasting_experiment.py `
  --dataset-format eicu_sepsis3 `
  --eicu-max-series 128 `
  --epochs 3 `
  --batch-size 16 `
  --persistent-memory-store .\output\persistent_memory\eicu_safe_store `
  --disable-persistent-memory-reuse `
  --output-json .\output\formal\r0_split_safe_memory\eicu_memory_reuse_disabled.json
```

预期结果：

- `persistent_memory.reuse_disabled = true`
- `loaded_persistent_samples = 0`
- 该结果作为后续 memory-enabled 的干净对照组。

## 5. 当前结论

第一条 R0/MVP-1 的核心框架已经落地。现在 persistent memory 不再是简单“有就加载”，而是先经过 split、test、当前评估 stay/patient 的安全门控，并且每次实验会输出审计证据。后续所有性能提升实验都应先检查 `persistent_memory.reuse_audit` 和 `prototype_audit`，确认没有 test 或当前评估对象泄漏，再讨论 MAE、RMSE 或反事实指标是否真实改善。

---

# 本轮 R1 / MVP-2 修改记录：memory harm control（2026-05-09）

## 1. 背景

在当前方法中，全量经验库与持久化增量更新链路已经具备可用性，但这并不自动等于预测性能提升。前面的 smoke 结果仍显示 memory-enabled 的 factual MAE 可能差于 base-only，这说明更大的经验库如果缺少“控害”机制，低质量检索、方向冲突残差或过强记忆修正都可能把预测拉偏。

因此，本轮继续执行修改方案中的下一条：先建立 memory harm control。目标不是直接追求最优性能，而是让记忆残差在进入最终预测前经过质量、方向和幅度三道约束。

## 2. 本轮修改目标

本轮修改完成三件事：

- 对记忆残差引入整体质量门控：综合 pattern、trajectory、experience 三路记忆的门控权重与置信度，而不是只看单一路径。
- 对记忆残差引入方向一致性约束：当融合残差与直接经验残差明显反向时，将其降权甚至关闭。
- 对记忆残差引入幅度上限：防止经验库残差相对 base prediction 过大，导致预测被记忆路径主导。

## 3. 修改内容

### 3.1 `src/manifold_forecasting_trainer.py`

新增或完善了以下逻辑：

- 在融合表示阶段记录 `memory_weighted_confidence`，计算方式是三路记忆 gate 与三路 confidence 的加权和。它代表当前样本整体记忆证据质量。
- 新增 `_apply_memory_harm_control(...)`，在最终预测前处理 `coordinated_memory_residual`。
- harm control 由三部分共同决定：
  - quality scale：整体记忆质量低于阈值时降低残差；
  - alignment scale：记忆路径之间方向冲突时降低残差；
  - cap scale：残差幅度超过 base prediction 允许比例时降低残差。
- 最终采用三者最小值作为残差缩放系数，保守地阻止低质记忆伤害预测。
- 在单样本、batch forward、诊断统计、factual memory audit 中加入 harm control 审计字段。
- 模型 fingerprint/runtime settings 纳入 harm control 配置，避免不同控害配置误复用同一套神经缓存或 checkpoint 设置。

### 3.2 `run_forecasting_experiment.py`

新增命令行参数：

```powershell
--disable-memory-harm-control
--memory-quality-floor 0.18
--memory-min-path-alignment -0.20
--memory-residual-cap-ratio 0.35
```

并将这些设置写入输出 JSON 的 `memory_config`，便于后续实验复现和横向比较。

## 4. 运行与验证方式

### 4.1 语法验证

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  .\src\manifold_forecasting_trainer.py `
  .\run_forecasting_experiment.py `
  .\src\persistent_memory_store.py `
  .\src\tsf_data.py
```

预期结果：无语法错误，无 traceback。

### 4.2 控害逻辑单元验证

```powershell
.\.venv\Scripts\python.exe -c "import torch, json; from src.manifold_memory import ManifoldMemoryConfig; from src.manifold_forecasting_trainer import EndToEndForecastingManifoldTrainer, ForecastingTrainerConfig; mc=ManifoldMemoryConfig(sequence_feature_dim=1, static_feature_dim=1, device='cpu'); tc=ForecastingTrainerConfig(forecast_horizon=2, device='cpu'); tr=EndToEndForecastingManifoldTrainer(mc, tc, static_feature_dim=1, kg_feature_dim=0, intervention_feature_dim=0, intervention_sequence_dim=0, formation_feature_dim=16); residual=torch.tensor([10.0,-10.0]); base=torch.tensor([1.0,1.0]); out,summary,aux=tr._apply_memory_harm_control(base,residual,{'memory_weighted_confidence':0.01,'experience_confidence':0.01,'path_alignment':-0.5}); print(json.dumps({'out':out.tolist(),'summary':summary,'strength':float(aux['coordinated_strength_tensor'].item())}, sort_keys=True))"
```

本轮实际结果显示：

```json
{
  "out": [0.0, -0.0],
  "strength": 0.0,
  "summary": {
    "memory_harm_alignment_scale": 0.0,
    "memory_harm_cap_scale": 0.03500000014901161,
    "memory_harm_confidence_scale": 0.05555555555555556,
    "memory_harm_control_enabled": true,
    "memory_harm_control_scale": 0.0,
    "memory_harm_post_strength": 0.0,
    "memory_harm_pre_strength": 10.0,
    "memory_harm_quality": 0.01
  }
}
```

这说明在低质量、强冲突、过大残差的极端情况下，控害模块确实会把危险记忆残差压到 0。

### 4.3 最小端到端 smoke

```powershell
.\.venv\Scripts\python.exe .\run_forecasting_experiment.py `
  --dataset-format tsf `
  --tsf .\TSForecasting\tsf_data\sample.tsf `
  --dataset-name r1_harm_smoke `
  --history-length 8 `
  --forecast-horizon 4 `
  --max-train-windows-per-series 1 `
  --epochs 1 `
  --batch-size 64 `
  --max-memory 64 `
  --skip-posthoc-diagnostics `
  --output-json .\output\r1_harm_control_smoke.json
```

本轮实际结果：

- 命令成功运行，输出文件为 `output\r1_harm_control_smoke.json`。
- `memory_config.memory_harm_control_enabled = true`。
- factual path audit 中出现：
  - `memory_harm_control_scale`
  - `memory_harm_confidence_scale`
  - `memory_harm_quality`
  - `memory_harm_alignment_scale`
  - `memory_harm_cap_scale`
  - `memory_harm_pre_strength`
  - `memory_harm_post_strength`
- 本次 smoke 的 `memory_harm_quality` 均值约为 `0.3645`，高于默认阈值 `0.18`，且方向冲突不明显，因此平均裁剪比例为 `1.0`。

## 5. 结果与分析

本轮修改验证了控害框架可运行，但不能声称已经提升性能。最小 TSF smoke 仍显示 memory-enabled 指标差于 base-only：

- memory-enabled MAE：约 `52.2491`
- base-only MAE：约 `50.2403`
- improvement_mae：约 `-2.0089`

这个结果的含义是：当前修改首先提供“防止明显坏记忆进入预测”的安全机制和可观测字段，而不是直接证明记忆有效。真正判断是否提升，需要在 eICU 数据上做多 seed、base-only、memory-enabled、harm-control-disabled 三组对照。

## 6. 下一步实验命令

建议后续至少跑三组对照：

### 6.1 默认 harm control

```powershell
.\.venv\Scripts\python.exe .\run_forecasting_experiment.py `
  --dataset-format eicu_sepsis3 `
  --eicu-max-series 512 `
  --epochs 5 `
  --batch-size 16 `
  --persistent-memory-store .\output\persistent_memory\eicu_full_train_store `
  --persistent-memory-allowed-splits train,external `
  --output-json .\output\formal\r1_harm_control\eicu_harm_control_on.json
```

预期结果：输出中 `memory_harm_control_enabled=true`，并能看到 harm control 审计字段；如果坏记忆主要来自低质量或过大残差，则 memory-enabled 与 base-only 的差距应缩小。

### 6.2 关闭 harm control

```powershell
.\.venv\Scripts\python.exe .\run_forecasting_experiment.py `
  --dataset-format eicu_sepsis3 `
  --eicu-max-series 512 `
  --epochs 5 `
  --batch-size 16 `
  --persistent-memory-store .\output\persistent_memory\eicu_full_train_store `
  --persistent-memory-allowed-splits train,external `
  --disable-memory-harm-control `
  --output-json .\output\formal\r1_harm_control\eicu_harm_control_off.json
```

预期结果：如果控害有效，关闭后 memory residual 更容易变大，memory-enabled 预测相对 base-only 的负迁移可能更明显。

### 6.3 不复用持久化经验库的 base 对照

```powershell
.\.venv\Scripts\python.exe .\run_forecasting_experiment.py `
  --dataset-format eicu_sepsis3 `
  --eicu-max-series 512 `
  --epochs 5 `
  --batch-size 16 `
  --persistent-memory-store .\output\persistent_memory\eicu_full_train_store `
  --disable-persistent-memory-reuse `
  --output-json .\output\formal\r1_harm_control\eicu_base_no_persistent_reuse.json
```

预期结果：用于区分“模型本身训练不足”与“经验库复用带来的额外影响”。如果 base 对照也很差，优先处理训练预算和特征；如果 only memory-enabled 变差，继续加强检索质量和残差控害。

## 7. 已知问题

- 当前控害阈值仍是启发式默认值，尚未在 eICU 多 seed 实验上校准。
- `path_alignment = 0` 时当前不会因方向约束被惩罚，因为它代表没有明确反向证据；如果后续发现零对齐也有风险，可以把阈值策略改为需要正向对齐才放行。
- 本轮没有解决“如何让记忆真正贡献正向增益”的全部问题，只解决了低质量记忆伤害预测时缺乏刹车的问题。

## 8. 下一步计划

后续可以继续修改方案的下一条：做逐病例 memory delta 审计和子群体差异输出。也就是不仅看总体 MAE，还要输出哪些患者、哪些 pattern/trajectory、哪些 SOFA 严重度区间中 memory 是正贡献，哪些是负贡献，从而为进一步优化检索和反事实干预组合提供依据。

---

# 本轮 R1 / MVP-2 补充修改记录：逐病例 memory gain 审计（2026-05-09）

## 1. 背景

上一轮已经完成 memory harm control 的核心门控：低质量、方向冲突或幅度过大的记忆残差会被降权。但原方案中第二条 R1 / MVP-2 还包含一个关键要求：不能只看总体 MAE，而要输出逐病例差异，回答“哪些病例 memory 帮了忙，哪些病例 memory 造成了伤害”。

因此，本轮补齐逐病例 memory gain 审计。这个审计只比较同一测试病例上的 base-only 预测和 memory-enabled 预测，不依赖反事实 donor，也不依赖临床可行性 reranking，属于 factual forecasting 层面的纯粹证据。

## 2. 本轮修改目标

本轮目标是让每次实验额外输出：

- 每个测试病例的 base-only MAE；
- 每个测试病例的 memory-enabled MAE；
- 每个病例的 `memory_gain_mae = base_mae - memory_enabled_mae`；
- memory 帮助率、伤害率和 memory gain 分布；
- 最受益病例与最受害病例；
- 按 pattern、trajectory、pattern+trajectory、基线水平分层的 memory gain 报告。

## 3. 修改内容

### 3.1 `run_forecasting_experiment.py`

新增命令行参数：

```powershell
--memory-gain-audit-top-k
--memory-gain-audit-max-cases
```

其中：

- `--memory-gain-audit-top-k` 控制输出多少个最受益和最受害病例，默认 `12`。
- `--memory-gain-audit-max-cases` 控制 `case_details` 中最多保存多少条逐病例记录，默认 `512`；设置为 `0` 表示保存全部测试病例。

新增 helper：

- `_case_error_summary(...)`：计算单病例 MAE、RMSE、bias。
- `_build_memory_gain_audit(...)`：构建逐病例 memory gain 审计。
- `_memory_gain_group_summary(...)`：构建分层 memory gain 报告。
- `_baseline_level_bin(...)`：按基线水平分层，便于后续迁移到 SOFA 严重度分析。

输出 JSON 新增顶层字段：

```json
"memory_gain_audit": {
  "sample_count": ...,
  "case_detail_count": ...,
  "outcome_counts": {
    "helped": ...,
    "harmed": ...,
    "unchanged": ...
  },
  "helped_rate": ...,
  "harmed_rate": ...,
  "mean_memory_gain_mae": ...,
  "memory_gain_mae_distribution": {...},
  "memory_delta_strength_distribution": {...},
  "top_helped_cases": [...],
  "top_harmed_cases": [...],
  "subgroup_slices": {...},
  "case_details": [...]
}
```

同时也把该审计挂到：

```json
evaluation_lines.factual_forecasting.memory_path_audit.memory_gain_audit
layered_evaluation_baseline.factual_layer.memory_gain_audit_summary
```

这样后续看总输出、factual 专线输出、分层基线输出时，都能找到同一套证据。

## 4. 运行与验证方式

### 4.1 语法验证

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  .\run_forecasting_experiment.py `
  .\src\manifold_forecasting_trainer.py
```

本轮实际结果：通过，无语法错误。

### 4.2 最小端到端 smoke

```powershell
.\.venv\Scripts\python.exe .\run_forecasting_experiment.py `
  --dataset-format tsf `
  --tsf .\TSForecasting\tsf_data\sample.tsf `
  --dataset-name r1_memory_gain_audit_smoke `
  --history-length 8 `
  --forecast-horizon 4 `
  --max-train-windows-per-series 1 `
  --epochs 1 `
  --batch-size 64 `
  --max-memory 64 `
  --skip-posthoc-diagnostics `
  --memory-gain-audit-max-cases 20 `
  --memory-gain-audit-top-k 5 `
  --output-json .\output\r1_memory_gain_audit_smoke.json
```

本轮实际结果：

```json
{
  "sample_count": 299.0,
  "case_detail_count": 20.0,
  "helped_rate": 0.06688963210702341,
  "harmed_rate": 0.9331103678929766,
  "mean_memory_gain_mae": -2.0088608577229845,
  "top_helped": 5,
  "top_harmed": 5,
  "nested_available": true
}
```

说明：

- `memory_gain_audit` 顶层字段可以正常读取。
- `evaluation_lines.factual_forecasting.memory_path_audit.memory_gain_audit` 中也可以读到同一审计。
- 本次小样本中，memory 帮助 20 例，伤害 279 例，平均 memory gain 为负，说明当前 smoke 仍不支持“记忆提升预测”的结论。

## 5. 结果与分析

本轮修改提高的是可解释性和可审计性，而不是直接提升指标。现在输出已经能回答：

- 总体 memory 是否有益：看 `mean_memory_gain_mae`。
- 有多少病例受益：看 `helped_rate` 和 `outcome_counts.helped`。
- 有多少病例受害：看 `harmed_rate` 和 `outcome_counts.harmed`。
- 哪些病例最受益：看 `top_helped_cases`。
- 哪些病例最受害：看 `top_harmed_cases`。
- 哪些子群体更容易受益或受害：看 `subgroup_slices`。

这对后续优化非常重要，因为如果总体 MAE 变差，但某些 pattern 或 SOFA 水平区间明显受益，就说明不应该完全放弃经验库，而应该做条件启用、检索质量筛选或子群体门控。

## 6. 后续正式验证实验

### 6.1 eICU 中等规模默认审计

```powershell
.\.venv\Scripts\python.exe .\run_forecasting_experiment.py `
  --dataset-format eicu_sepsis3 `
  --eicu-max-series 512 `
  --epochs 5 `
  --batch-size 16 `
  --persistent-memory-store .\output\persistent_memory\eicu_full_train_store `
  --persistent-memory-allowed-splits train,external `
  --memory-gain-audit-max-cases 0 `
  --memory-gain-audit-top-k 20 `
  --output-json .\output\formal\r1_memory_gain_audit\eicu_512_memory_gain_audit.json
```

预期结果：

- `memory_gain_audit.sample_count` 等于测试病例数。
- `case_detail_count` 等于测试病例数，因为 `--memory-gain-audit-max-cases 0` 表示保存全部病例。
- 能看到每个病例的 `memory_gain_mae`。
- 如果经验库仍整体伤害预测，`mean_memory_gain_mae` 应小于 0，且 `harmed_rate` 高。

### 6.2 关闭 harm control 的对照实验

```powershell
.\.venv\Scripts\python.exe .\run_forecasting_experiment.py `
  --dataset-format eicu_sepsis3 `
  --eicu-max-series 512 `
  --epochs 5 `
  --batch-size 16 `
  --persistent-memory-store .\output\persistent_memory\eicu_full_train_store `
  --persistent-memory-allowed-splits train,external `
  --disable-memory-harm-control `
  --memory-gain-audit-max-cases 0 `
  --memory-gain-audit-top-k 20 `
  --output-json .\output\formal\r1_memory_gain_audit\eicu_512_memory_gain_no_harm_control.json
```

预期结果：

- 如果 harm control 有效，关闭后 `harmed_rate` 可能升高，或 `top_harmed_cases` 中的负 gain 绝对值更大。
- 如果两者几乎一致，说明当前主要问题不是残差控害，而是检索质量、特征表达或训练不足。

### 6.3 base 对照：不复用持久化经验库

```powershell
.\.venv\Scripts\python.exe .\run_forecasting_experiment.py `
  --dataset-format eicu_sepsis3 `
  --eicu-max-series 512 `
  --epochs 5 `
  --batch-size 16 `
  --persistent-memory-store .\output\persistent_memory\eicu_full_train_store `
  --disable-persistent-memory-reuse `
  --memory-gain-audit-max-cases 0 `
  --memory-gain-audit-top-k 20 `
  --output-json .\output\formal\r1_memory_gain_audit\eicu_512_no_persistent_reuse_gain_audit.json
```

预期结果：

- 用于区分“持久化经验库复用造成的影响”和“模型自身 runtime memory 造成的影响”。
- 如果不复用持久化经验库后 `mean_memory_gain_mae` 改善，说明 persistent memory 检索质量需要继续筛选。
- 如果仍然变差，说明问题可能在 runtime memory 残差路径或训练目标本身。

## 7. 已知问题

- `baseline_level_bin` 当前是通用数值分层，迁移到 eICU 后可以进一步改成更明确的 SOFA 严重度分层。
- 默认 `case_details` 只保存 512 条，正式实验若要全量逐病例输出，需要显式设置 `--memory-gain-audit-max-cases 0`。
- 本轮只做 factual memory gain 审计，还没有把这些负贡献病例反向用于训练或检索过滤。

## 8. 下一步计划

下一步建议进入 MVP-3：实现更明确的 clinical state signature，并把 transition memory 的检索从单纯模式相似扩展为“病情状态 + 干预动作 + 后续变化”的结构化匹配。这样才能让反事实干预组合不只是找相似病例，而是找“类似状态下，采取某类干预后确实改善”的经验。
---

# 本轮 R2 / MVP-3 修改记录：clinical state signature + 现有 transition store 升级（2026-05-10）

## 背景

上一轮已经完成 `memory harm control` 和逐病例 `memory_gain_audit`，可以回答 memory 在 factual forecasting 中总体是否有益、哪些病例受益、哪些病例受损。本轮继续执行修改方案中的 R2 / MVP-3：不新建一套平行的 `TransitionMemory` 黑盒，而是在现有 `EndToEndForecastingManifoldTrainer` 的 runtime transition store 上增加显式 clinical state signature、来源审计字段和检索侧签名一致性加权。

## 本轮修改目标

1. 让 transition store 中每条 `TransitionStoreEntry` 不只保存连续 `state_vector`，还保存可读、可分组、可审计的 `clinical_state_signature`。
2. 在 transition 检索评分中加入轻量 `clinical_state_signature` 一致性加分，避免只依赖连续向量相似度。
3. 在 `memory_diagnostics` 中输出 transition signature 覆盖率、唯一签名数、来源 split 分布和复用允许率，便于后续判断 transition memory 是否 split-safe、是否具备临床状态分层能力。
4. 保持现有 transition store、bundle 导出、bundle 加载和 factual/counterfactual 路径兼容，不引入独立新存储类。

## 修改内容

### 1. `src/manifold_forecasting_trainer.py`

- 新增 `transition_signature_match_weight` 运行时参数，默认 `0.08`。
- 新增 `_clinical_state_signature_payload(...)` 和 `_clinical_state_signature(...)`：
  - 按当前窗口末值构造 `severity_bin`；
  - 按归一化当前水平构造 `level_bin`；
  - 按近期变化构造 `trend_bin`；
  - 按波动和 change proxy 构造 `volatility_bin`；
  - 合并 `trajectory_label` 和 KG 状态旗标，形成稳定字符串签名。
- 新增 `_transition_source_audit_metadata(...)`，为 transition entry 写入：
  - `source_split`
  - `source_dataset`
  - `source_stay_id`
  - `source_stay_id_hash`
  - `source_patient_id_hash`
  - `is_allowed_for_reuse`
- 扩展 `_build_transition_store(...)`：
  - 每条 `TransitionStoreEntry.metadata` 新增 `clinical_state_signature`；
  - 新增 `clinical_state_signature_schema_version`；
  - 新增 `clinical_state_signature_features`；
  - 新增 `clinical_state_vector_dim`；
  - 同步维护 `transition_signature_cache` 和 `transition_signature_counts`。
- 扩展 `_rebuild_transition_store_cache(...)`：
  - 从已有 bundle 或已有 entry metadata 重建 signature cache；
  - 兼容旧 bundle 中没有签名字段的情况。
- 扩展 `_retrieve_transition_readout(...)`：
  - 查询样本生成 `query_clinical_state_signature`；
  - 对签名完全一致的候选增加 `transition_signature_match_weight`；
  - 输出 `matched_clinical_state_signature_count` 和 `matched_clinical_state_signature_weight`。
- 新增 `_transition_store_signature_audit(...)`，并写入 `memory_diagnostics.transition_store_signature_audit`。

### 2. `run_forecasting_experiment.py`

- 新增命令行参数：

```powershell
--transition-signature-match-weight
```

- 将该参数写入 trainer：

```python
trainer.transition_signature_match_weight = float(args.transition_signature_match_weight)
```

- 在输出 JSON 的 `memory_config` 中记录 `transition_signature_match_weight`，保证同一组实验参数可复现。

## 影响范围

- 影响 `--enable-transition-memory` 打开时的 transition candidate scoring。
- 不改变默认未启用 transition memory 时的主预测路径。
- 不改变 `ForecastSample` 数据结构，不破坏 TSF 和 eICU 数据加载。
- 旧 inference bundle 仍可加载；旧 entry 缺少 `clinical_state_signature` 时签名加权自动退化为 0。
- 当前签名是第一版规则型离散签名，主要用于审计和轻量筛选，不等同于最终临床状态模型。

## 运行方式

### 编译检查

```powershell
python -m py_compile .\run_forecasting_experiment.py .\src\manifold_forecasting_trainer.py
```

### 本轮已执行的 smoke 验证

```powershell
python .\run_forecasting_experiment.py `
  --dataset-format tsf `
  --tsf .\TSForecasting\tsf_data\sample.tsf `
  --dataset-name mvp3_transition_signature_smoke `
  --history-length 8 `
  --forecast-horizon 4 `
  --max-train-windows-per-series 1 `
  --epochs 1 `
  --batch-size 64 `
  --max-memory 64 `
  --enable-transition-memory `
  --skip-posthoc-diagnostics `
  --memory-gain-audit-max-cases 20 `
  --memory-gain-audit-top-k 5 `
  --transition-signature-match-weight 0.08 `
  --output-json .\output\mvp3_transition_signature_smoke.json
```

### 建议正式 eICU 验证实验

默认启用 signature matching：

```powershell
python .\run_forecasting_experiment.py `
  --dataset-format eicu_sepsis3 `
  --eicu-max-series 512 `
  --epochs 5 `
  --batch-size 16 `
  --enable-kg `
  --enable-transition-memory `
  --persistent-memory-store .\output\persistent_memory\eicu_full_train_store `
  --persistent-memory-allowed-splits train,external `
  --memory-gain-audit-max-cases 0 `
  --memory-gain-audit-top-k 20 `
  --transition-signature-match-weight 0.08 `
  --output-json .\output\formal\mvp3_transition_signature\eicu_512_signature_transition.json
```

关闭 signature matching 的对照实验：

```powershell
python .\run_forecasting_experiment.py `
  --dataset-format eicu_sepsis3 `
  --eicu-max-series 512 `
  --epochs 5 `
  --batch-size 16 `
  --enable-kg `
  --enable-transition-memory `
  --persistent-memory-store .\output\persistent_memory\eicu_full_train_store `
  --persistent-memory-allowed-splits train,external `
  --memory-gain-audit-max-cases 0 `
  --memory-gain-audit-top-k 20 `
  --transition-signature-match-weight 0.0 `
  --output-json .\output\formal\mvp3_transition_signature\eicu_512_no_signature_transition.json
```

关闭 transition memory 的 base 对照：

```powershell
python .\run_forecasting_experiment.py `
  --dataset-format eicu_sepsis3 `
  --eicu-max-series 512 `
  --epochs 5 `
  --batch-size 16 `
  --enable-kg `
  --persistent-memory-store .\output\persistent_memory\eicu_full_train_store `
  --persistent-memory-allowed-splits train,external `
  --memory-gain-audit-max-cases 0 `
  --memory-gain-audit-top-k 20 `
  --output-json .\output\formal\mvp3_transition_signature\eicu_512_no_transition_memory.json
```

### 快速读取审计结果

```powershell
python -c "import json; d=json.load(open('output/mvp3_transition_signature_smoke.json',encoding='utf-8')); a=d['memory_diagnostics']['transition_store_signature_audit']; print(a)"
```

正式 eICU 结果建议重点比较以下字段：

```text
memory_effectiveness.improvement_mae
memory_gain_audit.mean_memory_gain_mae
memory_gain_audit.harmed_rate
memory_diagnostics.transition_store_signature_audit.signature_coverage_rate
memory_diagnostics.transition_store_signature_audit.unique_signature_count
memory_diagnostics.transition_store_signature_audit.source_split_counts
memory_diagnostics.transition_store_signature_audit.reuse_allowed_rate
```

## 结果与分析

本轮已完成编译验证：

```text
python -m py_compile .\run_forecasting_experiment.py .\src\manifold_forecasting_trainer.py
结果：通过
```

本轮已完成 TSF smoke 验证，输出文件为：

```text
.\output\mvp3_transition_signature_smoke.json
```

关键审计结果：

```json
{
  "entry_count": 299,
  "signature_coverage_rate": 1.0,
  "unique_signature_count": 21,
  "source_split_counts": {
    "train_runtime": 299
  },
  "signature_match_weight": 0.08
}
```

说明：

- transition store 中 299 条 entry 均生成了 clinical state signature。
- signature 已形成 21 个不同临床状态桶，说明该字段不是常量占位。
- 当前 smoke 使用 TSF sample 数据，不是 eICU 临床数据，所以 `source_split_counts` 为 `train_runtime`，KG 相关状态为 `kg_none` 是预期行为。
- smoke 的总体 factual memory gain 仍为负，`mean_memory_gain_mae = -1.9545`，说明本轮改动主要完成 transition state 审计和轻量检索加权，并不声称已经解决 memory 伤害问题。

## 已知问题

- 当前 clinical state signature 是规则型离散签名，阈值主要为了审计和 smoke 可运行，尚未针对 eICU 分布做校准。
- TSF sample 不包含 sepsis KG 状态，因此不能验证 `state_sepsis`、`state_septic_shock`、`state_high_lactate` 等 KG 旗标对签名的实际分层效果。
- 本轮只做了小型 smoke，不能作为性能结论；正式结论必须看 eICU 512 或更大规模对照实验。
- `mem` 虚拟环境缺少 `torch`，本轮 smoke 使用系统 `python`，其 `torch` 版本为 `2.6.0+cu124`。

## 下一步计划

1. 跑 eICU 三组对照：signature matching、signature weight 置零、关闭 transition memory。
2. 如果 signature matching 能降低 `harmed_rate` 或提升 `mean_memory_gain_mae`，再把签名从完全匹配升级为部分匹配评分。
3. 如果 signature 分布过粗或过细，根据 eICU 的 SOFA、lactate、MAP、vasopressor 和 respiratory support 分层重新校准阈值。
4. 下一轮可以继续推进 R3：只在高 support、高一致性、正 utility 的 transition 状态下接入 factual residual。

---

# 本轮补丁：persistent memory schema 过滤与三组消融一键运行（2026-05-10）

## 背景

执行 MVP-3 正式 eICU 命令时出现如下错误：

```text
RuntimeError: The size of tensor a (244) must match the size of tensor b (258) at non-singleton dimension 1
```

根因是旧的 `.\output\persistent_memory\eicu_full_train_store` 中保存的 `patient_static_features` 为 244 维，而当前命令启用了 `--enable-kg`，当前 eICU 数据集构造出的 `patient_static` 为 258 维。`persistent_samples + dataset.train_samples` 混合作为 `memory_seed_samples` 后，归一化使用当前 run 的 `static_mean/static_std`，因此旧样本维度不匹配。

## 本轮修改目标

1. 不让旧 schema 的 persistent memory 样本进入 neural memory bank、intervention store 和 transition store。
2. 保留 persistent store 的加载审计，明确报告被过滤样本数量和原因。
3. 提供一个脚本，把三组 MVP-3 消融实验合并成一条命令顺序执行。

## 修改内容

### 1. `run_forecasting_experiment.py`

新增 `_filter_memory_seed_samples_for_schema(...)`：

- 按当前数据集 schema 检查：
  - `patient_static` 维度；
  - `intervention_static` 维度；
  - `intervention_sequence` 单步维度。
- 对不兼容样本直接排除，避免进入 `_build_memory_bank(...)`。
- 输出 `memory_seed_schema_audit` 到：

```text
persistent_memory.memory_seed_schema_audit
```

该审计字段包含：

```json
{
  "input_total": "...",
  "kept_total": "...",
  "excluded_total": "...",
  "expected_patient_dim": "...",
  "excluded_patient_static_dim": "...",
  "excluded_persistent_source": "...",
  "examples": [...]
}
```

### 2. `scripts/run_mvp3_transition_signature_ablations.py`

新增一键三组消融脚本，顺序运行：

1. `signature_transition`：启用 transition memory，并设置 `--transition-signature-match-weight 0.08`。
2. `no_signature_transition`：启用 transition memory，但设置 `--transition-signature-match-weight 0.0`。
3. `no_transition_memory`：关闭 transition memory。

## 影响范围

- 只影响 `memory_seed_samples` 的构建，不改变 train/val/test 数据本身。
- 对 schema 兼容的 persistent memory 样本不做排除。
- 对旧 schema persistent store 会自动回退为“只用当前 train runtime 样本作为 memory seed”，避免运行中断。
- semantic prototype 仍可加载；本补丁只过滤进入神经 memory bank 的样本。

## 运行方式

### 原命令现在可直接重跑

```powershell
python .\run_forecasting_experiment.py `
  --dataset-format eicu_sepsis3 `
  --eicu-max-series 512 `
  --epochs 5 `
  --batch-size 16 `
  --enable-kg `
  --enable-transition-memory `
  --persistent-memory-store .\output\persistent_memory\eicu_full_train_store `
  --persistent-memory-allowed-splits train,external `
  --memory-gain-audit-max-cases 0 `
  --memory-gain-audit-top-k 20 `
  --transition-signature-match-weight 0.08 `
  --output-json .\output\formal\mvp3_transition_signature\eicu_512_signature_transition.json
```

### 三组消融合并为一条命令

```powershell
python .\scripts\run_mvp3_transition_signature_ablations.py `
  --eicu-max-series 512 `
  --epochs 5 `
  --batch-size 16 `
  --persistent-memory-store .\output\persistent_memory\eicu_full_train_store `
  --persistent-memory-allowed-splits train,external `
  --memory-gain-audit-max-cases 0 `
  --memory-gain-audit-top-k 20 `
  --transition-signature-match-weight 0.08 `
  --output-dir .\output\formal\mvp3_transition_signature
```

该命令会生成：

```text
.\output\formal\mvp3_transition_signature\eicu_512_signature_transition.json
.\output\formal\mvp3_transition_signature\eicu_512_no_signature_transition.json
.\output\formal\mvp3_transition_signature\eicu_512_no_transition_memory.json
```

如需先检查实际展开的三个命令：

```powershell
python .\scripts\run_mvp3_transition_signature_ablations.py `
  --eicu-max-series 512 `
  --epochs 5 `
  --batch-size 16 `
  --dry-run
```

## 结果与分析

已执行编译检查：

```powershell
python -m py_compile .\run_forecasting_experiment.py .\scripts\run_mvp3_transition_signature_ablations.py
```

结果：通过。

已执行三组消融脚本 dry-run：

```powershell
python .\scripts\run_mvp3_transition_signature_ablations.py --eicu-max-series 8 --epochs 1 --batch-size 4 --dry-run
```

结果：三组命令可正常展开。

已执行小规模 eICU smoke：

```powershell
python .\run_forecasting_experiment.py `
  --dataset-format eicu_sepsis3 `
  --eicu-max-series 8 `
  --epochs 1 `
  --batch-size 4 `
  --enable-kg `
  --enable-transition-memory `
  --persistent-memory-store .\output\persistent_memory\eicu_full_train_store `
  --persistent-memory-allowed-splits train,external `
  --memory-gain-audit-max-cases 8 `
  --memory-gain-audit-top-k 3 `
  --transition-signature-match-weight 0.08 `
  --skip-posthoc-diagnostics `
  --output-json .\output\formal\mvp3_transition_signature\eicu_8_signature_transition_smoke.json
```

关键审计结果：

```json
{
  "input_total": 57771,
  "kept_total": 24,
  "excluded_total": 57747,
  "expected_patient_dim": 258,
  "excluded_patient_static_dim": 57747,
  "excluded_persistent_source": 57747,
  "excluded_runtime_source": 0
}
```

说明：

- 旧 persistent store 中 57747 条样本均为 244 维 patient_static，与当前 258 维 schema 不兼容。
- 当前 24 条 train runtime 样本维度兼容，因此保留。
- 原始 244/258 维度错误已被消除。

## 已知问题

- 当前 persistent store 是旧 schema，若要真正复用大规模 persistent memory，需要用当前 `--enable-kg` 和当前 feature schema 重新构建 persistent store。
- 本补丁先保证实验不崩溃；它不会把 244 维旧样本自动补齐到 258 维，因为没有可靠的特征名对齐映射时自动补齐会引入隐性错误。
- 小规模 smoke 的性能指标不能作为正式结论。

## 下一步计划

1. 先运行一键三组消融脚本，确认 512 规模下 signature transition 是否优于 no-signature 和 no-transition。
2. 如需复用 persistent memory 的全部 57747 条样本，应先用当前 schema 重新 build store。
3. 拿到三组 JSON 后，再决定是否进入 R3：safe transition residual factual path。

---

# MVP-3 三组消融结果分析与下一步计划（2026-05-10）

## 背景

已完成三组 eICU 512 消融实验：

```text
eicu_512_signature_transition.json
eicu_512_no_signature_transition.json
eicu_512_no_transition_memory.json
```

三组实验均使用当前 `--enable-kg` schema。由于旧 persistent store 中的样本为 244 维，而当前 patient static 为 258 维，旧 persistent 样本被 schema 过滤；本轮实际验证的是当前 run 的 train runtime memory，而不是旧 57747 条 persistent memory 的完整复用。

## 本轮结果汇总

| 实验 | test MAE | base MAE | improvement MAE | mean memory gain | helped rate | harmed rate |
|---|---:|---:|---:|---:|---:|---:|
| signature transition | 1.3842 | 1.4085 | 0.0244 | 0.0244 | 52.93% | 47.07% |
| no-signature transition | 1.3860 | 1.4085 | 0.0226 | 0.0226 | 52.93% | 46.88% |
| no transition memory | 1.3965 | 1.4085 | 0.0120 | 0.0120 | 15.63% | 14.45% |

关键结论：

- transition memory 明确优于 no-transition：MAE 额外改善约 `0.0124`。
- signature matching 相比 no-signature transition 有小幅额外改善：MAE 额外改善约 `0.0018`。
- signature matching 没有明显降低 harmed rate，主要收益体现在整体误差均值略好。
- no-transition memory 的大多数病例 unchanged，说明 transition path 是本轮 memory 真正产生影响的主要来源。

## 分层分析

### 1. baseline level 分层

signature transition 下：

- `level_lt_4`：mean gain `0.0774`，helped rate `69.93%`。
- `level_ge_12`：mean gain `0.0684`，helped rate `77.78%`。
- `level_8_to_12`：mean gain `0.0148`，helped rate `54.55%`。
- `level_4_to_8`：mean gain `-0.0034`，helped rate `42.61%`，harmed rate `57.39%`。

解释：transition memory 更适合低 SOFA 和高 SOFA 两端，对中间严重度尤其 `4-8` 区间不稳定。

### 2. pattern 分层

signature transition 下：

- `down`：mean gain `0.0811`，helped rate `75.68%`。
- `spike`：mean gain `0.0400`，helped rate `57.80%`。
- `up`：mean gain `0.0218`，helped rate `54.55%`。
- `flat`：mean gain `0.0012`，helped rate `44.68%`，harmed rate `55.32%`。

解释：transition memory 对有明确趋势或突变的轨迹更有价值；对 flat pattern 很容易产生过度修正。

### 3. trajectory 分层

signature transition 下：

- `rising_regime`：mean gain `0.2385`，helped rate `66.67%`。
- `shifted_regime`：mean gain `0.0288`，helped rate `57.83%`。
- `stable_regime`：mean gain `-0.0100`，harmed rate `76.00%`。

解释：stable regime 是下一轮门控必须保护的子群，不能让 transition residual 在稳定病程中自由生效。

### 4. 极端病例

signature transition 的最大受益病例：

```text
stay_159009: +0.9828, spike|shifted_regime, level_lt_4
stay_170932: +0.9813, spike|shifted_regime, level_lt_4
stay_141470: +0.7401, spike|shifted_regime, level_lt_4
```

最大受损病例：

```text
stay_163209: -0.9642, spike|shifted_regime, level_lt_4
stay_167698: -0.9547, spike|shifted_regime, level_lt_4
stay_162673: -0.5933, spike|shifted_regime, level_lt_4
```

解释：同一个 `spike|shifted_regime|level_lt_4` 子群同时包含最大正收益和最大负收益，说明仅靠 pattern/trajectory/level 分层还不够，下一轮必须加入 per-case transition gate 和 residual cap。

## 已知限制

1. 旧 persistent memory 样本被过滤：

```text
input_total = 57771
kept_total = 1536
excluded_total = 56235
expected_patient_dim = 258
excluded_patient_static_dim = 56235
```

因此本轮不能声称“旧 persistent memory 大规模复用有效”，只能说明当前 schema 下 runtime transition memory 有正收益。

2. signature matching 收益较小：

signature 比 no-signature 只提升约 `0.0018 MAE`，说明当前 signature 是可用但偏弱的辅助信号，还不足以单独作为强门控。

3. harmed rate 仍偏高：

signature transition helped rate 为 `52.93%`，harmed rate 为 `47.07%`。总体 MAE 是正收益，但逐病例风险仍高。

## 下一步修改计划

### 第一优先级：R3 safe transition residual factual path

现在可以进入预测性能提升相关修改，但不能直接放大 memory 权重。下一轮应实现安全门控：

```text
transition residual 生效条件：
  confidence >= threshold
  support_strength >= threshold
  expected_utility > 0
  stable_regime 降权或禁用
  flat pattern 降权
  residual magnitude <= cap_ratio * base uncertainty / scale
```

建议新增参数：

```text
--transition-min-confidence
--transition-min-support
--transition-min-expected-utility
--transition-stable-regime-penalty
--transition-flat-pattern-penalty
--transition-residual-cap-ratio
```

目标不是让 transition 影响更多病例，而是减少 harmed cases，尤其是 stable/flat 和中等 SOFA 区间。

### 第二优先级：按当前 schema 重建 persistent store

如果论文或实验需要证明 persistent memory 复用有效，必须重新构建当前 schema 的 store，建议不要覆盖旧目录，使用新目录：

```powershell
python .\run_forecasting_experiment.py `
  --dataset-format eicu_sepsis3 `
  --eicu-max-series 512 `
  --enable-kg `
  --persistent-memory-store .\output\persistent_memory\eicu_full_train_store_kg258 `
  --build-persistent-memory-only `
  --persistent-memory-build-splits train `
  --persistent-memory-allowed-splits train,external
```

然后用新 store 重跑三组消融：

```powershell
python .\scripts\run_mvp3_transition_signature_ablations.py `
  --eicu-max-series 512 `
  --epochs 5 `
  --batch-size 16 `
  --persistent-memory-store .\output\persistent_memory\eicu_full_train_store_kg258 `
  --persistent-memory-allowed-splits train,external `
  --memory-gain-audit-max-cases 0 `
  --memory-gain-audit-top-k 20 `
  --transition-signature-match-weight 0.08 `
  --output-dir .\output\formal\mvp3_transition_signature_kg258
```

### 第三优先级：signature 从完全匹配升级为部分匹配

当前 signature 有小幅收益，但完全匹配太硬。后续可把签名拆成组件打分：

```text
severity match
level match
trend match
volatility match
trajectory match
KG active state overlap
```

这样可以避免“签名字符串不完全一致就没有任何加分”的问题。

## 应该立即做的事

1. 先不要继续加 LLM 或新模块。
2. 下一轮直接改 R3 safe transition gate，目标是降低 harmed rate。
3. 同时准备当前 schema 的 persistent store 重建，用新目录 `eicu_full_train_store_kg258`。
4. R3 完成后，固定用同一套三组消融命令比较：
   - no transition
   - transition no signature
   - transition signature
   - transition signature + safe gate
