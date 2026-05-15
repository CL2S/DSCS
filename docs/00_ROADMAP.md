# 经验记忆增强预测系统 — 修改路线图

> 创建：2026-05-12 | 基于 `01_DIAGNOSIS.md` 的 7 项硬伤诊断

---

## 当前状态

| 指标 | R4 峰值 | S2 当前 | 差距 |
|---|---|---|---|
| improvement_mae | 0.0632 | 0.0333 | -47% |
| helped_rate | 61.1% | 58.0% | -3pp |
| harmed_rate | 38.9% | 42.0% | +3pp |
| stable_regime harmed | 73.3% | 66.7% | -6pp（仍严重） |
| persistent_loaded | 0 | 1536 | 首次可用 |
| trans_utility | +0.004 | -0.059 | 翻转（不可靠） |

已完成：R0-R4 的 split-safe 持久化、harm control、连续 transition gate、val audit、retrieval trace。

---

## 四阶段路线

```
Phase A: 拆耦 + 不确定性      ← 当前阶段，1-2 周
Phase B: 增强 factual 底座    ← 2-3 周
Phase C: 提升 retrieval 质量  ← 2-3 周
Phase D: 收敛与论文化          ← 1-2 周
```

---

## Phase A：拆耦 + 不确定性

### A1：拆分 factual forecasting 与 retrieval 的表示空间

**对应硬伤：** #1（训练随机性主导）、#2（无沉默机制）、#4（持久化放大噪声）

**当前问题：** `EndToEndForecastingManifoldTrainer` 的 `_encode_sample` 同时服务 factual prediction 和 memory retrieval。两个任务对隐空间的要求不同：预测需要捕获时序动态，检索需要捕获病例间全局相似性。共享 encoder 导致表示折中。

**修改方案：**
- 保留 shared encoder 底层，添加两个独立 projection head：
  - `factual_head` → 输出用于预测的 embedding
  - `retrieval_head` → 输出用于 prototype/experience 检索的 embedding
- retrieval head 可用 contrastive loss 单独训练（以"预测修正价值"为相似度目标）

**涉及文件：** `src/manifold_forecasting_trainer.py`, `run_forecasting_experiment.py`

**预期收益：** 减少表示冲突，降低 memory residual 的噪声幅度，提升训练稳定性。

**实际修改（2026-05-12）：** 发现架构已有 `_split_encoding_spaces`，但被弱化（projection_scale=0.05, head_delta=0.05, pre_projection mode, 初始化为零）。修改为 projection_scale=0.30, head_delta=0.30, post_projection mode。新增 `branch_decoupling_enabled` 开关，暴露 CLI `--branch-decoupling-mode`, `--retrieval-projection-scale`, `--disable-branch-decoupling`。

**状态：** ✅ 代码完成，待实验验证

---

### A2：MC Dropout 不确定性估计

**对应硬伤：** #1（微小改善不可靠）

**当前问题：** 预测只输出点估计。当两个候选方案的预测差值很小时，无法判断差异是否在噪声范围内。

**修改方案：**
- 在 `predict()` 中新增 `num_samples` 参数
- 多次前向（dropout 开启），收集样本
- 输出均值 + 标准差
- 在 counterfactual 比较时输出预测区间

**涉及文件：** `src/manifold_forecasting_trainer.py`

**预期收益：** 能区分"模型确定"和"不确定"的预测，guardrail 可据此降低不确定推荐的信度。

**实际修改（2026-05-12）：** 基础设施（`predict_with_uncertainty`、`analyze_uncertainty`）已存在。修改为：(1) `analyze_uncertainty` 新增 `per_sample_rows` 输出每样本不确定性；(2) `_phase0_report_decision` 新增 `uncertain_delta` 检查——当 `|predicted_delta| < prediction_std * 2.0` 时标记推荐不可靠；(3) CLI `--uncertainty-delta-threshold`（默认 2.0，设 0 禁用）。

**状态：** ✅ 代码完成，待实验验证

---

### A3：分层评估体系

**对应硬伤：** #6（检索与预测脱节）

**当前已有：** factual 层的 memory_gain_audit、val_audit、transition_gate_audit、retrieval_trace

**待补：**
- **Retrieval 层：** donor 临床邻近性分布、prototype 方向匹配率、persistent vs runtime 来源比例
- **Candidate 层：** 可行率、违规率、repair 频率、候选方案多样性
- **综合报告：** 自动生成"瓶颈在哪一层"的诊断摘要

**涉及文件：** `run_forecasting_experiment.py`, `src/evaluate.py`

**预期收益：** 每次实验自动定位瓶颈，加速迭代。

**实际修改（2026-05-12）：** 新增 `_build_diagnostic_summary` 函数，自动读取所有分层数据（memory_gain_audit, val_audit, factual_path_audit, counterfactual_summary, uncertainty_analysis）生成综合诊断摘要。输出包含：各层 verdict、最差子群、瓶颈排序（bottleneck_ranking），位于 JSON 顶层 `diagnostic_summary`。

**状态：** ✅ 代码完成，待实验验证

---

## Phase B：增强 factual 底座

### B1：细化 ICU 动态特征

**对应硬伤：** #2（时序表示粗粒度）

**修改方案：**
- 新增特征：干预时序（距上次抗生素/升压药的小时数）、治疗响应速度（干预后 6h 的 SOFA 变化率）、状态骤变检测（24h 内最大 SOFA 跳变）
- 在 `ForecastSample` 中增加 `dynamic_features` 字段
- 不影响现有 formation features

**涉及文件：** `src/tsf_data.py`, `src/manifold_forecasting_trainer.py`

**状态：** ⬜ 未开始

---

### B2：Memory Residual 直接监督

**对应硬伤：** #1（残差方向不可控）、#7（harm control 只管幅度）

**当前问题：** memory residual 没有直接的训练信号。模型不知道残差是否正确。

**修改方案：**
- 定义 `residual_target = y_true - base_prediction.detach()`
- 新增 `L_memory_residual = SmoothL1(memory_residual, residual_target)`
- 将 memory path 的训练目标从"帮一下预测"改为"学习修正 base model 的误差"

**涉及文件：** `src/manifold_forecasting_trainer.py`

**预期收益：** memory residual 产生可学习的、有方向性的修正，而非无差别噪声。

**状态：** ⬜ 未开始

---

### B3：Learnable Memory Gate

**对应硬伤：** #2（无沉默机制）、#7（只能管幅度）

**修改方案：**
- 新增一个小型 MLP：输入（encoding, memory_quality, prototype_confidence, state_signature）
- 输出 gate ∈ [0, 1]，表示"记忆修正应该被应用的程度"
- 训练目标：gate * residual 的预测误差最小化 + gate 接近 0 时的正则化（鼓励沉默）
- 替代当前的 heuristic harm control 中的 quality gate

**涉及文件：** `src/manifold_forecasting_trainer.py`

**预期收益：** 模型学会对 stable_regime 等不需要修正的情况自动降低 gate，解决硬伤 #2。

**状态：** ⬜ 未开始

---

## Phase C：提升 retrieval 质量

### C1：持久化经验 relevance 筛选
### C2：Contrastive retrieval embedding
### C3：学习式 donor reranker

**状态：** ⬜ 全部未开始（Phase A 完成后细化）

---

## Phase D：收敛与论文化

### D1：系统定位为"检索增强候选干预模拟框架"
### D2：医生报告证据强度增强
### D3：多 seed 稳定性实验

**状态：** ⬜ 全部未开始

---

## 修改记录

| 日期 | 阶段 | 任务 | 状态 | 结果 |
|---|---|---|---|---|
| 05-12 | — | 路线图创建 | ✅ | — |
| 05-12 | A1 | 分支解耦增强 | ✅ | projection_scale 0.05→0.30, post_projection mode, CLI 暴露 |
| 05-12 | A2 | 不确定性 guardrail | ✅ | uncertainty→counterfactual 决策, per_sample_rows, uncertain_delta 检查 |
| 05-12 | A3 | 分层诊断摘要 | ✅ | diagnostic_summary, bottleneck_ranking 自动生成 |
| — | — | — | — | — |
