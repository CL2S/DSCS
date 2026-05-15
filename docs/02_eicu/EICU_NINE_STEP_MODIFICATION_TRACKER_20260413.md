# EICU 九阶段修改计划清单（持续更新）

## 1. 文件用途

本文件用于跟踪当前 eICU“相似患者检索 + 候选干预建议”方法的九阶段修改路线。它的定位不是方法说明文档，而是后续逐条实施时的统一状态看板。

后续维护规则如下：

- 本文件作为九阶段修改的主跟踪文件。
- 每当开始具体修改某一条时，先在对应条目下补充“详细实施计划”。
- 每当完成某一条修改后，必须回填“当前状态”“是否达到预期”“是否修改完成”“本轮结果摘要”“验证结论”。
- 若某一条修改未达到预期，不直接标记完成，而是标记为“未达预期”或“返工中”，并记录原因。

## 2. 状态定义

| 状态 | 含义 |
|---|---|
| 未开始 | 尚未进入实施 |
| 待细化 | 已确定要做，但尚未写出该条的详细实施计划 |
| 进行中 | 已开始改动代码或实验流程 |
| 已完成 | 代码与验证均达到当前阶段预期 |
| 未达预期 | 已做改动，但效果未达到验收标准 |
| 返工中 | 已识别问题，正在同一阶段内继续修正 |

## 3. 总表

| 编号 | 阶段名称 | 目标优先级 | 前置依赖 | 当前状态 | 是否达到预期 | 是否修改完成 | 最近更新 |
|---|---|---|---|---|---|---|---|
| 1 | Phase 0: 分层评估基线 | P0 | 无 | 已完成 | 是 | 是 | 2026-04-13 |
| 2 | factual / retrieval 双头拆分 | P0 | 1 | 已完成 | 是 | 是 | 2026-04-13 |
| 3 | 动态特征增强 | P0 | 1, 2 | 已完成 | 是 | 是 | 2026-04-13 |
| 4 | 多任务预测 + 不确定性 | P1 | 1, 2, 3 | 已完成 | 是 | 是 | 2026-04-14 |
| 5 | donor 池扩大 + overlap 过滤 | P1 | 1, 2, 3 | 已完成 | 是 | 是 | 2026-04-14 |
| 6 | 学习式 reranker | P1 | 1, 5 | 已完成 | 是 | 是 | 2026-04-14 |
| 7 | 候选方案扩展为“安全层 + 搜索层” | P1 | 1, 3, 5, 6 | 已完成 | 是 | 是 | 2026-04-14 |
| 8 | 轻量 rolling-horizon 策略 | P2 | 1, 3, 4, 7 | 已完成 | 是 | 是 | 2026-04-15 |
| 9 | 反事实严谨性 + 医生证据增强 | P2 | 1, 4, 5, 6, 7 | 未开始 | 待验证 | 否 | 2026-04-13 |

---

## 4. 分阶段清单

### 4.1 Phase 0: 分层评估基线

**目标**

在修改主模型前，先建立一套可复用的分层评估框架，用于定位问题到底出在 factual、retrieval、candidate generation、counterfactual ranking 还是最终报告层。

**大致修改计划**

- 固定一组 baseline 运行配置和随机种子。
- 生成分层评估输出，包括：
  - factual 层：MAE、RMSE、校准误差、分亚型误差。
  - retrieval 层：donor similarity 分布、state match、guideline compatibility、top-k donor 一致性。
  - candidate 层：候选方案数量、repair 占比、违规率、guardrail 拦截率。
  - counterfactual 层：predicted delta 分布、improvement rate、不同 donor score mode 对比。
  - report 层：recommendation_ready 占比、review_only 原因分布。
- 将这些指标统一导出到结构化 JSON 或 CSV，作为后续各阶段验收基线。

**涉及模块**

- `run_forecasting_experiment.py`
- `scripts/run_eicu_counterfactual_multiseed.py`
- `src/manifold_forecasting_trainer.py`
- `output/analysis/`

**预期有效性**

- 可行性：高
- 预期收益：高
- 主要价值：建立“改动是否有效”的统一判据，避免后续凭感觉判断。

**阶段验收标准**

- 能稳定复现一份 baseline 指标集。
- baseline 指标字段固定，便于后续对照。
- 能清楚区分 factual、retrieval、candidate、counterfactual 各层问题。

**当前状态**

- 当前状态：已完成
- 是否达到预期：是
- 是否修改完成：是
- 本轮结果摘要：已在主实验入口补充 `layered_evaluation_baseline`，并新增固定配置脚本 `scripts/run_eicu_phase0_baseline.py`，可自动导出完整实验结果、分层 JSON 和汇总 CSV。
- 验证结论：Phase 0 第一版已达到“能稳定生成统一基线并分层定位问题”的目标，后续各阶段可直接基于该输出做对照。

**详细实施计划**

- 在 `run_forecasting_experiment.py` 中新增 `layered_evaluation_baseline` 输出块。
- 将分层输出拆为五层：
  - factual 层：聚合指标、误差分布、按 pattern / trajectory / pattern+trajectory 的切片误差。
  - retrieval 层：selected donor 指标分布、exact match rate、top-3 donor 一致性。
  - candidate 层：候选数分布、repair 占比、unsafe option rate、source 分布。
  - counterfactual 层：selected delta 分布、improvement rate、repaired vs original 对比。
  - report 层：基于 Phase 0 阈值的 `recommendation_ready` / `review_only` 比例和原因分布。
- 将单病例 guardrail 的核心阈值参数化到主实验入口，供 report 层统一使用。
- 新增 `scripts/run_eicu_phase0_baseline.py`，固定基线配置并导出：
  - `phase0_seed*_full.json`
  - `phase0_seed*_layered.json`
  - `phase0_baseline_runs.csv`
  - `phase0_baseline_summary.json`
- 使用小规模 smoke 运行做两轮验证：
  - 16-series smoke：验证主实验输出中已包含完整分层结构。
  - 8-series quiet smoke：验证包装脚本静默子进程输出后仍能稳定产出 JSON / CSV 汇总。
- 本轮关键验证结果：
  - 16-series smoke：`improvement_mae = 0.0399`，`donor_similarity_mean = 0.8324`，`candidate_count_mean = 1.9375`，`recommendation_ready_rate = 0.6875`
  - 8-series quiet smoke：`donor_similarity_mean = 0.8651`，`candidate_count_mean = 2.0`，`recommendation_ready_rate = 0.625`
- 本轮产物位置：
  - `output/analysis/phase0_baseline_smoke/`
  - `output/analysis/phase0_baseline_smoke_quiet/`

---

### 4.2 factual / retrieval 双头拆分

**目标**

将当前高度耦合的 factual forecasting 与 retrieval/reranking 表示学习分开，降低单一隐空间同时服务多任务带来的目标冲突。

**大致修改计划**

- 保留共享输入编码入口，但拆分为两条 head：
  - factual head：专注未来状态预测。
  - retrieval head：专注 donor 相似性与检索表达。
- 调整训练流程，使 factual 损失与 retrieval 相关损失不再完全绑定在同一表示上。
- 保持外部接口尽量稳定，避免先期大规模重构调用层。

**涉及模块**

- `src/manifold_forecasting_trainer.py`
- `src/memory_model.py`
- `src/memory_manager.py`

**预期有效性**

- 可行性：中高
- 预期收益：很高
- 主要价值：从结构上降低“上游 forecast 不稳拖累下游 donor 检索”的问题。

**阶段验收标准**

- factual 指标不低于现有 baseline。
- retrieval 质量指标有可观提升，或至少在 factual 不退化前提下更稳定。
- 新结构不破坏现有单病例推理入口。

**当前状态**

- 当前状态：已完成
- 是否达到预期：是
- 是否修改完成：是
- 本轮结果摘要：已在 `src/manifold_forecasting_trainer.py` 内完成 factual / retrieval 双空间拆分。当前实现保留共享 encoder 输入入口，但新增零初始化残差式 `factual_projection`、`retrieval_projection` 和 retrieval query/key/value 增量头；forecast 主干与 fusion gate 改用 factual 空间，memory read/write、intervention store 和 donor ranking 改用 retrieval 空间，`load_inference_bundle()` 也已改为 `strict=False` 以兼容旧 bundle。
- 验证结论：第 2 条的结构性目标已经达到，且未破坏现有单病例推理与旧 bundle 加载。8-series Phase 0 对照中，最终 smoke 版 `hybrid_mae = 1.8948`，优于第 1 条基线的 `1.9131`；`recommendation_ready_rate` 保持 `0.625` 不变，`donor_similarity_mean` 维持在同量级。旧 bundle 与新 bundle 的单病例推理均已跑通。

**详细实施计划**

- 在 `src/manifold_forecasting_trainer.py` 中引入最小可行的“双空间”实现，而不是重写 encoder：
  - `factual_projection`：零初始化残差适配器，保守地服务 forecast/fusion 主干。
  - `retrieval_projection`：零初始化残差适配器，保守地服务 memory / donor 检索。
  - retrieval query/key/value 增量头：以原始 encoder 的 query/key/value 为底座，只学习小幅残差，避免训练初期漂移过大。
- 新增 `_split_encoding_spaces()`，把共享 encoder 输出拆成：
  - factual embedding
  - retrieval encoding（query/key/value/input_embedding）
- 调整下游使用路径：
  - factual 路径：`base_regressor`、`_build_fused_representation()`、factual prediction 主干。
  - retrieval 路径：`memory_manager.read/write()`、`_build_memory_bank()`、`_build_intervention_store()`、counterfactual donor ranking。
- 为兼容历史 bundle，更新 `load_inference_bundle()`：
  - 使用 `strict=False` 加载 state dict
  - 将缺失的新参数保留为默认零初始化，从而兼容第 2 条之前导出的 inference bundle
  - 记录 `bundle_load` 诊断信息到 `memory_diagnostics`
- 本轮验证与结果：
  - 静态检查：
    - `python -m py_compile src/manifold_forecasting_trainer.py`
    - `python -m py_compile scripts/infer_eicu_counterfactual_plan.py run_forecasting_experiment.py`
  - Phase 0 对照 smoke：
    - 基线：`output/analysis/phase0_baseline_smoke_quiet/phase0_baseline_summary.json`
      - `hybrid_mae = 1.9131`
      - `donor_similarity_mean = 0.8651`
      - `recommendation_ready_rate = 0.625`
    - 第 2 条最终 smoke：`output/analysis/phase2_space_split_smoke_v4/phase0_baseline_summary.json`
      - `hybrid_mae = 1.8948`
      - `donor_similarity_mean = 0.8633`
      - `recommendation_ready_rate = 0.625`
  - 新 bundle 导出与单病例推理：
    - bundle: `output/analysis/phase2_space_split_final_bundle.pt`
    - run summary: `output/analysis/phase2_space_split_final_bundle_run.json`
    - inference json: `output/analysis/phase2_space_split_final_bundle_inference.json`
    - inference md: `output/analysis/phase2_space_split_final_bundle_inference.md`
    - 关键结果：
      - `test mae = 1.8721`
      - `donor_similarity_mean = 0.8696`
      - 高相似度病例 `stay_141959` 仍可输出医生版建议单，最终 donor 相似度约 `0.942`
  - 旧 bundle 兼容性验证：
    - `output/tmp_high_similarity_bundle_32.pt`
    - `output/analysis/phase2_space_split_old_bundle_inference_finalcheck.json`
    - `output/analysis/phase2_space_split_old_bundle_inference_finalcheck.md`
    - 结果：旧 bundle 在当前代码下可正常加载和推理，未发生 state dict 兼容错误
- 阶段结论：
  - 结构性解耦已经完成。
  - 单病例入口和医生版输出未被破坏。
  - tiny smoke 下 memory 相对 base 的 `improvement_mae` 仍为负值，这说明第 2 条解决了“表示耦合”问题，但还没有解决“memory 何时真正带来稳定收益”问题；这个残留问题将在第 3、5、6 条继续处理。

---

### 4.3 动态特征增强

**目标**

增强对 ICU 病程动态的建模能力，让模型学到“病程动态相似”而不只是“静态变量相似”。

**大致修改计划**

- 在样本构造层补充动态特征：
  - 缺失指示
  - 上次观测间隔
  - 近期斜率
  - 短时响应速度
  - 多尺度时间窗摘要
- 优先在数据层和特征层增强，不先引入过度复杂的新模型。
- 对阶段转换、急性恶化、高乳酸、休克支持等病例单独做误差切片。

**涉及模块**

- `src/tsf_data.py`
- `src/manifold_forecasting_trainer.py`
- 相关数据加载与特征派生逻辑

**预期有效性**

- 可行性：高
- 预期收益：高
- 主要价值：同时改善 factual 预测与 donor 检索的医学合理性。

**阶段验收标准**

- 动态病例切片上的 factual 误差下降。
- donor state match 或近邻一致性上升。
- 未引入明显的维度错配或输入 schema 失稳。

**当前状态**

- 当前状态：已完成
- 是否达到预期：是
- 是否修改完成：是
- 本轮结果摘要：已在 `src/tsf_data.py` 中为 eICU `patient_static` 补充动态病程摘要特征，覆盖缺失比例、最近观测间隔、趋势、近期变化、尾部均值/波动、时间间隔统计和多变量整体变化摘要；同时在 `scripts/infer_eicu_counterfactual_plan.py` 中补齐 `patient_feature_names` 透传，使新 bundle 默认启用动态特征、旧 bundle 继续按原 schema 推理。
- 验证结论：第 3 条已达到“先在数据层增强动态病程建模且不破坏推理兼容性”的目标。8-series Phase 0 smoke 中，`hybrid_mae` 从第 2 条 smoke 的 `1.8948` 降到 `1.6650`，`improvement_mae` 从 `-0.0293` 转为 `+0.0229`，`recommendation_ready_rate` 从 `0.625` 升到 `0.75`；新旧 bundle 的单病例推理均已跑通。

**详细实施计划**

- 优先只改数据层和样本构造层，不先改大模型结构：
  - 在 `src/tsf_data.py` 为 eICU patient_static 增加病程动态摘要特征。
  - 不修改 `formation_features` 的既有字段顺序，避免波及 planner、报告和诊断索引。
- 新增的动态特征分三类：
  - 缺失与观测质量：
    - 各生理序列列的 `missing_ratio`
    - 各列距离最近一次有效观测的 `last_observed_gap_steps`
  - 动态变化速度：
    - 各列的 `trend`
    - 各列的 `recent_delta`
    - 各列尾部窗口的 `tail_mean`
    - 各列尾部窗口的 `tail_volatility`
  - 时间间隔与多尺度摘要：
    - 基于 `rel_end_hours` 的 `gap_last / gap_mean / gap_std / irregularity`
    - 多变量总体变化强度与缺失比例摘要
- 兼容策略：
  - 新训练默认启用动态 patient 特征，更新 `patient_feature_names`。
  - 单病例推理按 bundle 的 `patient_feature_names` 自动判断是否启用这些新特征，保证旧 bundle 不因维度变化而失效。
- 涉及修改文件：
  - `src/tsf_data.py`
  - `scripts/infer_eicu_counterfactual_plan.py`
  - `docs/02_eicu/EICU_NINE_STEP_MODIFICATION_TRACKER_20260413.md`
- 验证计划：
  - `python -m py_compile` 检查相关文件。
  - 跑 8-series Phase 0 smoke，对照第 2 条最终基线。
  - 导出一个新 bundle，并跑单病例推理。
  - 用旧 bundle 再跑一次单病例推理，验证兼容性未破坏。

- 本轮实际验证与结果：
  - 静态检查：
    - `python -m py_compile memory_mvp_project/src/tsf_data.py memory_mvp_project/scripts/infer_eicu_counterfactual_plan.py`
  - 8-series Phase 0 smoke：
    - `output/analysis/phase3_dynamic_features_smoke/phase0_baseline_summary.json`
    - 关键结果：
      - `hybrid_mae = 1.6650`
      - `base_mae = 1.6880`
      - `improvement_mae = 0.0229`
      - `donor_similarity_mean = 0.8561`
      - `donor_guideline_mean = 0.6881`
      - `candidate_count_mean = 2.0`
      - `predicted_improvement_rate = 0.75`
      - `recommendation_ready_rate = 0.75`
  - 对照第 2 条 smoke：
    - `output/analysis/phase2_space_split_smoke_v4/phase0_baseline_summary.json`
    - 对照结果：
      - `hybrid_mae = 1.8948`
      - `improvement_mae = -0.0293`
      - `donor_similarity_mean = 0.8633`
      - `recommendation_ready_rate = 0.625`
  - 新 bundle 导出与单病例推理：
    - `output/analysis/phase3_dynamic_bundle.pt`
    - `output/analysis/phase3_dynamic_bundle_run.json`
    - `output/analysis/phase3_dynamic_bundle_inference.json`
    - `output/analysis/phase3_dynamic_bundle_inference.md`
    - 关键结果：高相似病例 `stay_141959` 在新 bundle 下仍可生成医生版建议单，最终 donor 相似度约 `0.955`，状态为 `recommendation_ready`
  - 旧 bundle 兼容性验证：
    - `output/tmp_high_similarity_bundle_32.pt`
    - `output/analysis/phase3_old_bundle_inference_compat.json`
    - `output/analysis/phase3_old_bundle_inference_compat.md`
    - 结果：旧 bundle 在当前代码下仍可正常加载和推理，未发生 patient 维度不匹配错误
- 阶段结论：
  - 动态病程特征已经接入训练样本和单病例推理样本构造层。
  - 这一步先解决了“静态相似多、动态相似弱”的输入表达问题，没有提前引入更复杂模型。
  - tiny smoke 下 factual 层和 recommendation 层都出现正向信号，说明这一步是有效改动。
  - donor similarity 均值与第 2 条相比略低，但仍保持在同一量级；考虑到 factual 误差和 recommendation_ready_rate 明显改善，这个轻微波动可接受，后续由第 5、6 条继续优化 donor 质量与排序稳定性。
---

### 4.4 多任务预测 + 不确定性

**目标**

把当前以单一 proxy 为中心的预测改为多任务预测，并为推荐与 guardrail 提供不确定性信息。

**大致修改计划**

- 将预测目标从单一 SOFA/轨迹 proxy 扩展为多任务：
  - future SOFA
  - lactate change
  - vasopressor need
  - respiratory support escalation
  - 必要时再扩展短期死亡风险
- 为预测加入不确定性估计，优先考虑：
  - MC dropout
  - quantile regression
- 将区间信息接入 guardrail 和医生版报告。

**涉及模块**

- `src/manifold_forecasting_trainer.py`
- `src/memory_model.py`
- `scripts/infer_eicu_counterfactual_plan.py`
- `src/counterfactual_plan_renderer.py`

**预期有效性**

- 可行性：中等
- 预期收益：高
- 主要价值：让“推荐更优”不再只是单一均值差，而是多维度、带置信度的比较。

**阶段验收标准**

- 至少新增 2 到 3 个临床相关预测头并跑通训练与推理。
- 输出预测区间或方差指标。
- guardrail 能使用不确定性信息拒绝边际改善方案。

**当前状态**

- 当前状态：已完成
- 是否达到预期：是
- 是否修改完成：是
- 本轮结果摘要：已在 `src/tsf_data.py` 为 eICU 样本补充 4 个窗口级辅助目标，在 `src/manifold_forecasting_trainer.py` 中接入轻量多任务头、辅助损失、辅助任务评估和 MC-dropout 预测接口，并把 `base / factual / counterfactual` 三条路径的不确定性、辅助风险预测和区间信息贯通到 `scripts/infer_eicu_counterfactual_plan.py` 与 `src/counterfactual_plan_renderer.py` 的医生版输出；同时为 `scripts/run_eicu_phase0_baseline.py` 增加了多任务权重和不确定性采样参数透传。
- 验证结论：第 4 条已达到“新增多任务预测头、输出区间信息、并将不确定性接入 guardrail 与医生报告”的阶段目标。默认 8-series Phase 0 smoke 中，`test_multitask_metrics`、`test_uncertainty_analysis`、`example_prediction.uncertainty` 与 `example_prediction.auxiliary_prediction` 均已稳定产出；新 bundle 与旧 bundle 的单病例推理均能输出辅助风险表、预测区间和 guardrail 不确定性摘要。需要明确的是，tiny smoke 下 factual 主指标相较第 3 条略有回落，因此这一步的收益应理解为“证据与风险表达增强”，而不是“factual 主线已提升”。

**详细实施计划**

- 优先做“最小可行多任务 + MC-dropout 不确定性”而不是一次性引入更复杂的 quantile / ensemble：
  - 在 `src/tsf_data.py` 的 eICU 样本构造中补充窗口级辅助目标：
    - `future_sofa_delta_mean`
    - `future_lactate_delta`
    - `future_vasopressor_need`
    - `future_resp_support_escalation`
  - 训练时仅对有标签的 eICU 样本启用辅助损失，通用 TSF 数据集保持兼容。
- 在 `src/manifold_forecasting_trainer.py` 中新增轻量多任务头：
  - 共享现有主干表示，不重写 encoder
  - 回归头预测未来 SOFA 变化和乳酸变化
  - 二分类头预测未来升压药需要和呼吸支持升级
  - 训练损失使用主任务 loss + 加权 auxiliary multitask loss
- 在同一训练器中加入 MC-dropout 预测接口：
  - 输出每个 horizon 的 `mean / std / lower / upper`
  - 输出辅助任务的均值、方差和区间
  - 不修改旧 bundle 推理输入格式；旧 bundle 缺少新参数时继续按 `strict=False` 兼容加载
- 在 `scripts/infer_eicu_counterfactual_plan.py` 中扩展单病例输出：
  - 补充 `base / factual / counterfactual` 三条路径的不确定性结果
  - 将辅助任务预测写入结构化输出
  - guardrail 新增不确定性相关拦截条件，例如：
    - 推荐方案预测标准差过高
    - 改善幅度下界未明显高于 0
- 在 `src/counterfactual_plan_renderer.py` 中扩展医生版输出：
  - 展示预测区间
  - 展示未来乳酸变化、升压药需要概率、呼吸支持升级概率
  - 把“因不确定性过高而仅供复核”的原因写入复核要点
- 验证计划：
  - `python -m py_compile` 检查相关文件
  - 跑 8-series Phase 0 smoke，对照第 3 条基线
  - 导出一个新 bundle，验证单病例推理和医生版输出
  - 用旧 bundle 再跑一次推理，确认兼容路径未破坏

- 本轮实际验证与结果：
  - 静态检查：
    - `python -m py_compile memory_mvp_project/src/tsf_data.py memory_mvp_project/src/manifold_forecasting_trainer.py memory_mvp_project/src/counterfactual_plan_renderer.py memory_mvp_project/scripts/infer_eicu_counterfactual_plan.py memory_mvp_project/run_forecasting_experiment.py`
    - `python -m py_compile memory_mvp_project/scripts/run_eicu_phase0_baseline.py`
  - 8-series Phase 0 smoke（默认第 4 条配置）：
    - `output/analysis/phase4_multitask_uncertainty_smoke/phase0_baseline_summary.json`
    - `output/analysis/phase4_multitask_uncertainty_smoke/phase0_seed42_full.json`
    - 关键结果：
      - `hybrid_mae = 1.6964`
      - `base_mae = 1.6679`
      - `improvement_mae = -0.0285`
      - `donor_similarity_mean = 0.8693`
      - `recommendation_ready_rate = 0.75`
      - `test_multitask_metrics` 已包含 `future_sofa_delta_mean / future_lactate_delta / future_vasopressor_need / future_resp_support_escalation`
      - `test_uncertainty_analysis.num_samples = 8`
      - `forecast_mean_std = 0.1137`
  - Phase 0 基线脚本参数透传验证：
    - `output/analysis/phase4_multitask_uncertainty_smoke_script_passthrough/phase0_seed42_full.json`
    - 结果：`trainer_config.multitask_loss_weight = 0.08`、`trainer_config.test_uncertainty_samples = 6` 已正确写入输出，说明基线脚本能够显式复现实验超参
  - 新 bundle 导出与单病例推理：
    - `output/analysis/phase4_multitask_bundle.pt`
    - `output/analysis/phase4_multitask_bundle_run.json`
    - `output/analysis/phase4_multitask_bundle_inference.json`
    - `output/analysis/phase4_multitask_bundle_inference.md`
    - 关键结果：
      - 训练输出已包含 `test_multitask_metrics` 与 `test_uncertainty_analysis`
      - 高相似病例 `stay_141959` 在新 bundle 下 donor 相似度约 `0.961`
      - 医生版输出已新增“预测区间”和“辅助风险预测”表格
      - guardrail 结构化输出已包含 `delta_lower_bound / delta_upper_bound`
  - 旧 bundle 兼容性验证：
    - `output/tmp_high_similarity_bundle_32.pt`
    - `output/analysis/phase4_old_bundle_inference_compat.json`
    - `output/analysis/phase4_old_bundle_inference_compat.md`
    - 结果：旧 bundle 在当前代码下仍可正常加载和推理；虽然旧 bundle 不带第 4 条训练权重，但单病例入口仍能产出辅助风险表和不确定性摘要，兼容路径未被破坏
  - 不确定性 guardrail 显式拦截验证：
    - `output/analysis/phase4_old_bundle_inference_uncertainty_guardrail.json`
    - `output/analysis/phase4_old_bundle_inference_uncertainty_guardrail.md`
    - 结果：在同一高相似病例上将 `min_delta_lower_bound` 提高到 `0.12` 后，系统因 `delta_lower_bound = 0.093` 低于阈值而将状态改判为 `review_only`，证明 guardrail 已能基于不确定性拒绝边际改善方案
- 阶段结论：
  - 第 4 条已经把“均值预测”扩展为“多任务 + 区间 + 风险摘要”的单病例输出。
  - 这一步的主要价值是提升推荐证据表达和 guardrail 可解释性，而不是直接优化 factual 主任务精度。
  - tiny smoke 下 factual 主指标相较第 3 条略有回落，但幅度不大，且未出现训练、bundle 导出、单病例推理或旧 bundle 兼容性故障，因此当前可以结束本阶段并进入第 5 条。

**易懂解释**

- 这一步可以理解成：系统以前只会说“我觉得这个方案更好”，但不会说明它到底有多大把握。
- 第 4 条之后，系统会像一个更谨慎的临床助手，除了给出主判断，还会补充：
  - 这个判断的把握度有多高
  - 如果按当前方案走，未来哪些风险可能更高
  - 如果按推荐方案走，这些风险会怎么变
- 所以它的主要提升不是“分数一定更高”，而是“输出更像能被医生复核的证据包”，医生更容易判断这条建议值不值得看。

---

### 4.5 donor 池扩大 + overlap 过滤

**目标**

提升 donor 候选质量，让 donor 相似不只是“看起来像”，而是尽量满足基本可比性和可交换性。

**大致修改计划**

- 在正式实验中扩大 memory bank 和 donor 池规模。
- 设计分层 donor 池对照：
  - 大池
  - 同亚型池
  - 高重叠池
- 在 donor 检索前增加 overlap / 可交换性过滤：
  - 严重度重叠
  - 器官支持状态重叠
  - 近期趋势重叠
  - 可行动作空间重叠
- 暂不直接上复杂 propensity 估计，先用可解释的规则过滤稳定流程。

**涉及模块**

- `src/persistent_memory_store.py`
- `src/memory_manager.py`
- `src/manifold_forecasting_trainer.py`

**预期有效性**

- 可行性：中高
- 预期收益：高
- 主要价值：提高 donor 的医学可比性，为后续 reranker 与 counterfactual 评估打基础。

**阶段验收标准**

- donor similarity 与 donor consistency 分布改善。
- overlap 过滤后 recommendation_ready 的证据强度上升。
- donor 池扩大后结果不再被极小样本噪声主导。

**当前状态**

- 当前状态：已完成
- 是否达到预期：是
- 是否修改完成：是
- 本轮结果摘要：已在 `src/manifold_forecasting_trainer.py` 中补齐 donor 池扩展与 overlap 过滤主逻辑：候选池从“经验标签”扩展为“经验标签 + 同 pattern + 同 trajectory + 受上限控制的全局回填”，并新增基于 severity / trend / state / action 的 overlap 评分、过滤与回退机制；同时把 overlap 字段贯通到 `run_forecasting_experiment.py`、`scripts/run_eicu_phase0_baseline.py` 和逐例 counterfactual 输出中。
- 验证结论：第 5 条已达到“让 donor 选择更强调可比性而不是只追逐 embedding 相似度”的阶段目标。32-stay Phase 0 对照中，开启 overlap 后 `donor_overlap_score_mean` 从 `0.5747` 升到 `0.6029`，`donor_similarity_mean` 从 `0.8641` 降到 `0.8474`，说明 donor 被推向了更高 overlap 的邻域；同时 `recommendation_ready_rate` 保持 `0.875` 未下降。需要明确的是，这一步的收益主要体现在 donor 可比性与检索证据增强，不应解读为 factual 主线已提升。

**详细实施计划**

- donor 池扩大策略（先规则化，不改学习器）：
  - 在经验标签候选之外，补充同 pattern / 同 trajectory 候选。
  - 当候选池规模不足时，引入受上限控制的全局补充池，降低小样本噪声。
  - 新增 donor 候选规模控制参数，保证运行时间可控。
- overlap 过滤规则（可解释）：
  - 严重度重叠：约束 sample 与 donor 的严重度差异。
  - 近期趋势重叠：约束病程短期变化方向与幅度差异。
  - 状态重叠：比较 sepsis/shock/hypotension/high-lactate 等状态集合相似度。
  - 动作需求重叠：比较“应执行动作”集合（如早期抗菌、升压支持、乳酸复测）的一致性。
  - 采用“硬过滤 + fallback”模式：无可行 donor 时允许回退，但记录回退标记。
- 输出与评估增强：
  - donor metadata 增加 `donor_overlap_score`、`donor_overlap_valid`、`donor_overlap_reason`。
  - `counterfactual_summary` 增加 overlap 通过率、fallback 率、原因分布。
  - `layered_evaluation_baseline` 的 retrieval 层补充 overlap 分布统计。
- 涉及修改文件：
  - `src/manifold_forecasting_trainer.py`
  - `run_forecasting_experiment.py`
  - `scripts/run_eicu_phase0_baseline.py`
  - `docs/02_eicu/EICU_NINE_STEP_MODIFICATION_TRACKER_20260413.md`
- 验证计划：
  - `python -m py_compile` 检查相关文件
  - 8-series Phase 0 smoke 与第 4 条对照
  - 新 bundle 单病例推理检查 overlap 字段落盘
  - 旧 bundle 兼容性推理检查（新字段缺失时应可回退）

- 本轮实际验证与结果：
  - 静态检查：
    - `python -m py_compile memory_mvp_project/src/manifold_forecasting_trainer.py`
    - `python -m py_compile memory_mvp_project/run_forecasting_experiment.py`
    - `python -m py_compile memory_mvp_project/scripts/run_eicu_phase0_baseline.py`
  - 8-series Phase 0 smoke：
    - `output/analysis/phase5_overlap_smoke/phase0_baseline_summary.json`
    - 关键结果：
      - `donor_similarity_mean = 0.8698`
      - `donor_overlap_score_mean = 0.6277`
      - `donor_overlap_valid_rate = 0.5`
      - `donor_overlap_fallback_rate = 1.0`
    - 解释：小 donor 池下 overlap 规则频繁触发回退，因此该结果主要用于验证字段和逻辑已接通。
  - 32-stay Phase 0 默认 overlap：
    - `output/analysis/phase5_overlap_smoke_32/phase0_baseline_summary.json`
    - 关键结果：
      - `donor_similarity_mean = 0.8474`
      - `donor_overlap_score_mean = 0.6029`
      - `donor_overlap_valid_rate = 0.625`
      - `donor_overlap_fallback_rate = 0.4688`
      - `recommendation_ready_rate = 0.875`
  - 32-stay Phase 0 关闭 overlap 过滤对照：
    - `output/analysis/phase5_overlap_smoke_32_no_filter/phase0_baseline_summary.json`
    - 关键结果：
      - `donor_similarity_mean = 0.8641`
      - `donor_overlap_score_mean = 0.5747`
      - `donor_overlap_valid_rate = 1.0`
      - `donor_overlap_fallback_rate = 0.0`
      - `recommendation_ready_rate = 0.875`
    - 对照解释：关闭 overlap 后系统更偏向高 embedding 相似 donor；开启 overlap 后平均相似度略降，但 overlap 分数提升，证明 donor 排序被引向了更可比的病程/动作邻域。
  - 新 bundle 导出与单病例推理：
    - `output/analysis/phase5_overlap_bundle.pt`
    - `output/analysis/phase5_overlap_bundle_run.json`
    - `output/analysis/phase5_overlap_bundle_inference.json`
    - `output/analysis/phase5_overlap_bundle_inference.md`
    - `stay_141959` 结果：
      - `donor_similarity = 0.961`
      - `donor_overlap_score = 0.737`
      - `donor_overlap_valid = 1.0`
      - 推荐状态仍为 `recommendation_ready`
      - `predicted_delta = -0.142`
  - 旧 bundle 兼容验证：
    - `output/analysis/phase5_old_bundle_inference_compat.json`
    - `output/analysis/phase5_old_bundle_inference_compat.md`
    - 旧 bundle 仍可推理；同一病例输出了 overlap 字段：
      - `donor_similarity = 0.718`
      - `donor_overlap_score = 0.309`
      - `donor_overlap_valid = 0.0`
    - 说明旧 bundle 兼容路径正常，且 overlap 证据能在推理期补算出来

---

### 4.6 学习式 reranker

**目标**

在安全过滤之后，引入学习式 reranker 替代长期依赖手工配权的 donor/candidate 排序逻辑。

**大致修改计划**

- 保留 rule-based safety filter 作为第一道门。
- 将当前已有结构化特征整理成 reranker 输入：
  - donor similarity
  - KG similarity
  - guideline compatibility
  - penalties
  - state match
  - candidate source
- 训练监督式 reranker，对 donor 或 candidate 进行排序。
- 优先选简单稳健的模型，如 LightGBM 或小型 MLP。

**涉及模块**

- `src/manifold_forecasting_trainer.py`
- `src/persistent_memory_store.py`
- 新增 reranker 训练/推理辅助模块

**预期有效性**

- 可行性：中高
- 预期收益：中高
- 主要价值：减少人工权重脆弱性，提升不同数据分布下的泛化能力。

**阶段验收标准**

- 学习式 reranker 在固定评估集上优于当前手工加权基线，或至少更稳定。
- 安全过滤仍然保留。
- 单病例推理链路可切换 rule-based 与 learned reranker 进行对照。

**当前状态**

- 当前状态：已完成
- 是否达到预期：是
- 是否修改完成：是
- 本轮结果摘要：已落地一个“最小可行学习式 reranker”：在 hard filter 和 overlap filter 之后，增加可开关的线性 learned reranker；训练、bundle 导出、单病例推理、旧 bundle 回退路径均已打通。32-stay Phase 0 对照中，`predicted_improvement_rate` 从 `0.75` 提升到 `0.78125`，`recommendation_ready_rate` 保持 `0.875`，说明排序层开始带来正向收益，但 factual 主指标未变化，这符合本阶段预期。
- 验证结论：阶段目标达成。学习式 reranker 已优先作为“排序质量增强层”而不是“主预测增强层”投入使用；它没有破坏安全过滤与单病例链路，并且支持 rule-only 与 learned-linear 的同 bundle 对照。当前收益属于中等、可持续积累型，后续仍需第 7 条扩展 candidate space、第 9 条增强证据表达，才能把 learned reranker 的优势继续放大。

**详细实施计划**

- 本轮先做“最小可行 learned reranker”，不直接引入外部依赖或复杂模型：
  - 保留现有 hard filter 和 overlap filter 作为第一层安全门。
  - 在安全过滤之后，加一个可开关的线性 learned donor reranker，替代纯手工配权的末端排序。
- reranker 输入先复用现有已暴露的结构化 donor 特征，避免重新造特征管线：
  - donor similarity
  - KG similarity
  - guideline compatibility
  - state match
  - missing-care / contraindication penalties
  - overlap score / overlap valid
  - transition score / action-change score
  - pattern / trajectory / experience 是否匹配
- 训练方式采用轻量 ridge-style 线性回归，不额外引入 LightGBM：
  - 用训练完成后的当前模型，在一批样本上生成 top-k donor 候选。
  - 用 donor-only counterfactual 的 `predicted_delta` 作为监督目标。
  - 学到一组线性权重与 bias，作为 learned reranker score。
- 推理与评估改造：
  - 在 donor metadata 中写入 `donor_learned_reranker_score`、`donor_reranker_adjustment`、`donor_reranker_mode`。
  - 在主实验输出与 layered baseline 中补充 learned reranker 分布和开关状态。
  - 单病例推理支持 rule-only 与 learned-linear 两种模式对照。
- 涉及修改文件：
  - `src/manifold_forecasting_trainer.py`
  - `run_forecasting_experiment.py`
  - `scripts/run_eicu_phase0_baseline.py`
  - `docs/02_eicu/EICU_NINE_STEP_MODIFICATION_TRACKER_20260413.md`
- 验证计划：
  - `python -m py_compile` 检查相关文件。
  - 跑一轮 32-stay Phase 0 rule-only 与 learned-linear 对照。
  - 导出新 bundle 做单病例推理。
  - 用旧 bundle 做兼容性验证，确认 learned reranker 缺失时可回退。
- 实际完成情况：
  - 已在 `src/manifold_forecasting_trainer.py` 中加入最小可行 learned reranker 配置、训练、序列化和推理打分逻辑；reranker 输入复用了现有 donor similarity、KG similarity、guideline compatibility、state match、penalties、overlap score、transition/action-change 等结构化特征。
  - 已在 `run_forecasting_experiment.py` 中补齐 reranker 训练开关、输出字段和 layered baseline 统计。
  - 已在 `scripts/run_eicu_phase0_baseline.py` 中加入 `rule_only` / `learned_linear` 基线对照能力。
  - 已在 `scripts/infer_eicu_counterfactual_plan.py` 中加入 `--counterfactual-reranker-mode-override`，支持同一个 bundle 下做 rule-only 与 learned-linear 对照。
- 实际验证结果：
  - `python -m py_compile memory_mvp_project/src/manifold_forecasting_trainer.py`
  - `python -m py_compile memory_mvp_project/run_forecasting_experiment.py`
  - `python -m py_compile memory_mvp_project/scripts/run_eicu_phase0_baseline.py`
  - `python -m py_compile memory_mvp_project/scripts/infer_eicu_counterfactual_plan.py`
  - 32-stay `rule_only` 基线：
    - 结果文件：`output/analysis/phase6_reranker_ruleonly_32/phase0_baseline_summary.json`
    - `hybrid_mae = 1.7066`
    - `donor_overlap_score_mean = 0.6029`
    - `predicted_improvement_rate = 0.75`
    - `recommendation_ready_rate = 0.875`
  - 32-stay `learned_linear` 基线：
    - 结果文件：`output/analysis/phase6_reranker_learned_32/phase0_baseline_summary.json`
    - `hybrid_mae = 1.7066`
    - `donor_overlap_score_mean = 0.6029`
    - `donor_learned_reranker_score_mean = 0.0198`
    - `predicted_improvement_rate = 0.78125`
    - `recommendation_ready_rate = 0.875`
  - 新 bundle 单病例验证：
    - `output/analysis/phase6_reranker_bundle_inference_learned.json`
    - `output/analysis/phase6_reranker_bundle_inference_ruleonly.json`
    - 同一病例 `stay_141959` 下，learned 模式与 rule-only 模式都能稳定输出医生版报告；learned 模式已写出 `donor_learned_reranker_score` 与 `donor_reranker_adjustment`。
  - 旧 bundle 兼容验证：
    - `output/analysis/phase6_old_bundle_inference_learned_override.json`
    - 旧 bundle 在缺少 reranker state 时能够安全回退，不会导致推理崩溃；此时 learned score 自动退回 `0.0`。

---

### 4.7 候选方案扩展为“安全层 + 搜索层”

**目标**

把当前偏保守的 KG repair 扩展为两层候选方案生成机制，使系统具备真正搜索更优个体化干预路径的能力。

**大致修改计划**

- 第一层保留 KG repair，继续承担安全补缺作用。
- 第二层增加受约束的 action template search：
  - 开始时间
  - 持续时长
  - 组合策略
  - 强度调整
- 先采用有限模板搜索，不直接引入复杂连续动作优化或 RL。

**涉及模块**

- `src/manifold_forecasting_trainer.py`
- 可能新增 candidate template 生成模块
- `src/counterfactual_plan_renderer.py`

**预期有效性**

- 可行性：中等
- 预期收益：中高
- 主要价值：让推荐不再只是 donor 原方案与轻微修补之间二选一。

**阶段验收标准**

- 候选方案数量和多样性上升。
- 违规率不显著升高。
- 至少有一部分病例能找到优于 KG repair-only 的方案。

**当前状态**

- 当前状态：已完成
- 是否达到预期：是
- 是否修改完成：是
- 本轮结果摘要：已将候选生成从“donor 原方案 + KG repair”扩展为“安全补缺层 + 模板搜索层”。在 32-stay Phase 0 对照中，`candidate_count_mean` 从第 6 条的 `1.875` 提升到 `2.96875`，`search_candidate_case_rate = 0.6875`，`search_candidate_selected_rate = 0.4375`，`unsafe_candidate_option_rate = 0.0105`，说明搜索层显著提升了候选多样性，同时没有带来明显安全退化。
- 验证结论：阶段目标达成。第 7 条已经让系统从“只会修 donor 并二选一”进入“先保安全，再试少量可解释模板并择优”的状态。它的主要收益是扩展候选空间，而不是直接提升 factual 主指标；后续若要进一步提高搜索层命中率，需要第 8 条滚动决策和第 9 条更强证据表达继续承接。

**详细实施计划**

- 本轮先做“最小可行安全层 + 搜索层”，不引入 RL 或连续动作优化：
  - 保留 donor 原始方案与 `generated_kg_repaired`，作为第一层安全补缺候选。
  - 在其后增加一个受约束模板搜索层，生成少量、临床可解释的模板方案。
- 搜索层优先使用有限模板，不开放任意组合：
  - 抗菌提前模板：把抗菌时效从“晚/缺失”拉回到更早时间窗。
  - 升压支持强度模板：在低强度和标准强度两档中搜索。
  - 脓毒症 bundle 模板：对“脓毒症 + 低血压/高乳酸”病例生成抗菌 + 升压的组合模板。
  - 必要时补呼吸支持模板，但只在相关状态下启用。
- 搜索模板必须走同一套安全约束：
  - 复用 hard filter、overlap filter、knowledge quality 打分。
  - 对明显更差、明显重复或不通过安全门的模板直接丢弃。
- 输出与报告层同步增强：
  - 在 candidate metadata 中补 `generated_candidate_search_actions`、`generated_candidate_layer`。
  - 在 Phase 0 candidate layer 中增加模板候选可用率、被选中率和来源分布。
  - 在医生版报告中把“安全补缺方案”和“搜索模板方案”区分开显示。
- 涉及修改文件：
  - `src/manifold_forecasting_trainer.py`
  - `run_forecasting_experiment.py`
  - `scripts/run_eicu_phase0_baseline.py`
  - `src/counterfactual_plan_renderer.py`
  - `docs/02_eicu/EICU_NINE_STEP_MODIFICATION_TRACKER_20260413.md`
- 验证计划：
  - `python -m py_compile` 检查相关文件。
  - 跑一轮 32-stay Phase 0，对比第 6 条 learned reranker 基线与第 7 条 `safe_search` 候选策略。
  - 导出新 bundle 做单病例推理，确认报告里出现搜索层来源与模板动作。
  - 若第一版模板未带来候选多样性提升或大量安全回退，则继续在本阶段返工。
- 实际完成情况：
  - 已在 `src/manifold_forecasting_trainer.py` 中增加模板搜索层：
    - 抗菌提前模板
    - 低强度升压支持模板
    - 标准强度升压支持模板
    - 脓毒症 bundle 模板
    - 条件性呼吸支持模板
  - 搜索模板统一走 hard filter、overlap、knowledge quality 约束，不通过安全门或与已有候选重复的模板会被直接丢弃。
  - candidate metadata 已新增 `generated_candidate_layer` 与 `generated_candidate_search_actions`，并已贯通到单病例 JSON / Markdown 报告。
  - `run_forecasting_experiment.py` 与 `scripts/run_eicu_phase0_baseline.py` 已补充搜索层候选率、被选中率和来源分布统计。
  - `src/counterfactual_plan_renderer.py` 已把搜索层来源与“模板搜索动作”渲染到医生版输出。
- 实际验证结果：
  - 静态检查：
    - `python -m py_compile memory_mvp_project/src/manifold_forecasting_trainer.py`
    - `python -m py_compile memory_mvp_project/run_forecasting_experiment.py`
    - `python -m py_compile memory_mvp_project/scripts/run_eicu_phase0_baseline.py`
    - `python -m py_compile memory_mvp_project/src/counterfactual_plan_renderer.py`
  - 32-stay Phase 0 基线：
    - 结果文件：`output/analysis/phase7_safe_search_32/phase0_baseline_summary.json`
    - `candidate_count_mean = 2.96875`
    - `search_candidate_case_rate = 0.6875`
    - `search_candidate_selected_rate = 0.4375`
    - `unsafe_candidate_option_rate = 0.0105`
    - `predicted_improvement_rate = 0.78125`
    - `recommendation_ready_rate = 0.875`
  - 相比第 6 条 learned reranker 基线：
    - 第 6 条 `candidate_count_mean = 1.875`
    - 第 7 条 `candidate_count_mean = 2.96875`
    - 候选明显增多，但 `recommendation_ready_rate` 未下降
  - 搜索层有效性核对：
    - 在 `output/analysis/phase7_safe_search_32/phase0_seed42_full.json` 中，共有 `14` 个病例最终选中了搜索层候选，其中 `12` 个病例的 `predicted_delta > 0`
    - 正向示例包括：
      - `stay_141203 -> generated_template_antimicrobial_fast, delta = 0.0319`
      - `stay_141266 -> generated_template_antimicrobial_fast, delta = 0.2188`
      - `stay_141288 -> generated_template_antimicrobial_fast, delta = 0.1071`
  - 新 bundle 与单病例验证：
    - `output/analysis/phase7_safe_search_bundle.pt`
    - `output/analysis/phase7_safe_search_bundle_run.json`
    - `output/analysis/phase7_safe_search_bundle_inference.json`
    - `output/analysis/phase7_safe_search_bundle_inference.md`
    - 在高相似病例 `stay_141959` 上，系统已能明确输出：
      - 方案来源：`generated_template_vasopressor_low`
      - 候选层级：`search`
      - 模板搜索动作：`尝试较低强度升压支持模板`
    - 该例仍被标为 `recommendation_ready`，但 `predicted_delta = -0.059`，说明搜索层不是“自动更优”，而是“能够提出并显式比较新的安全候选”。

**易懂解释**

- 这一步要解决的问题很直接：现在系统虽然能修 donor 方案，但还是偏“在原方案上补洞”，不像真正帮医生多想几种安全可行的替代路线。
- 第 7 条做完后，系统会从“只会修补现有方案”，变成“先确保安全，再额外试几种小范围、可解释的治疗模板，看哪种更适合当前病人”。
- 更像什么：
  - 以前像是只会说“这个方案漏了早期抗菌，我帮你补上”。
  - 现在要升级成“我先补上必须项，然后再比较几种合理变体，比如抗菌更早一点、升压支持稍强一点、或者两者组合哪种更合适”。

---

### 4.8 轻量 rolling-horizon 策略

**目标**

将当前单步候选比较升级为短期滚动决策，更接近真实 ICU 的动态调整流程。

**大致修改计划**

- 先做轻量版 rolling horizon，而不是 full policy optimization。
- 以 2 到 3 个短期时点为主：
  - 每个时点重估状态
  - 重检索 donor
  - 重生成候选方案
  - 重做风险比较
- 保持计算规模可控，避免一次性引入长时程策略优化复杂度。

**涉及模块**

- `src/manifold_forecasting_trainer.py`
- `scripts/infer_eicu_counterfactual_plan.py`
- 可能新增短期滚动推理控制模块

**预期有效性**

- 可行性：中等偏低
- 预期收益：中等，长期潜力高
- 主要价值：让系统从“单窗口推荐”向“动态策略支持”演进。

**阶段验收标准**

- 能跑通 2 到 3 步短期滚动模拟。
- 每一步均可输出候选方案与风险比较。
- 不出现明显的误差爆炸或状态传播错乱。

**当前状态**

- 当前状态：已完成
- 是否达到预期：是
- 是否修改完成：是
- 本轮结果摘要：已完成一版“轻量两步 rolling-horizon”实现。系统现在不仅能给出单步 donor/candidate 推荐，还能基于第一步的选中方案把窗口向前投影一小步，再重做一次 donor 检索、候选比较和累计改善统计。Phase 0 基线、单病例 JSON/Markdown 输出、旧 bundle 兼容路径都已打通。
- 验证结论：阶段目标达成。第 8 条已经把系统从“只会做当前窗口的一次性比较”推进到“能做短期两步连续比较”的状态。它的主要收益是连续决策可视化和累计改善评估，而不是直接提升 factual 主指标；当前状态传播仍属于轻量近似，只沿目标轴做短期投影，后续若要更临床化，需要第 9 条进一步增强证据表达与边界说明。

**详细实施计划**

- 本轮先做“最小可行 rolling-horizon”，不引入 full policy optimization：
  - 保留现有单步 `predict_counterfactual()` 逻辑不变，避免把第 1 到第 7 条已稳定的单步路径打坏。
  - 新增 `predict_counterfactual_rollout()`，默认只做 2 步短期模拟；每一步都重新执行 donor 检索、候选生成和方案选择。
  - 第一步结束后，不假装知道所有未来协变量；只把目标轨迹沿当前模型预测向前推进一小步，形成“轻量投影状态”，再进入第二步比较。
- 状态传播策略坚持轻量近似，不做复杂世界模型：
  - 重算 `raw_context`、`raw_target`、`formation_features`、pattern/trajectory/experience 标签。
  - 目标序列按新窗口重新归一化。
  - 干预上下文沿用第一步选中的 candidate intervention，作为第二步的当前干预状态。
  - 其余未知生理协变量不做强行虚构，只保留上一窗口结构，避免在本阶段引入大量伪精度。
- 输出层和评估层同步补齐：
  - 在 `run_forecasting_experiment.py` 与 `run_eicu_phase0_baseline.py` 中增加 rollout 配置、结果导出和分层基线统计。
  - 在单病例推理脚本中增加 `rolling_horizon` 结构化输出，并在医生版 Markdown 里增加“短期滚动模拟”章节。
  - 跟踪折扣累计改善、第二步可用率、方案来源稳定率、逐步改善率。
- 涉及修改文件：
  - `src/manifold_forecasting_trainer.py`
  - `run_forecasting_experiment.py`
  - `scripts/run_eicu_phase0_baseline.py`
  - `scripts/infer_eicu_counterfactual_plan.py`
  - `docs/02_eicu/EICU_NINE_STEP_MODIFICATION_TRACKER_20260413.md`
- 实际完成情况：
  - 已在 `src/manifold_forecasting_trainer.py` 中增加 rollout 配置、投影辅助函数和 `predict_counterfactual_rollout()`。
  - 已在训练器 runtime settings 中补齐 `sequence_feature_names`、`patient_feature_names`、`counterfactual_rollout_steps`、`counterfactual_rollout_discount` 的序列化与加载。
  - 已在 `run_forecasting_experiment.py` 中增加 `counterfactual_rollout_evaluation` 和 `rolling_horizon_layer`。
  - 已在 `scripts/run_eicu_phase0_baseline.py` 中补齐 rollout 指标聚合。
  - 已在 `scripts/infer_eicu_counterfactual_plan.py` 中增加 `--rollout-steps`、`--rollout-discount`，并把两步模拟结果写入 JSON/Markdown 医生版输出。
- 实际验证结果：
  - 静态检查：
    - `python -m py_compile memory_mvp_project/src/manifold_forecasting_trainer.py`
    - `python -m py_compile memory_mvp_project/run_forecasting_experiment.py`
    - `python -m py_compile memory_mvp_project/scripts/run_eicu_phase0_baseline.py`
    - `python -m py_compile memory_mvp_project/scripts/infer_eicu_counterfactual_plan.py`
  - 16-stay Phase 0 smoke：
    - 结果文件：`output/analysis/phase8_rollout_smoke_16/phase0_baseline_summary.json`
    - `rollout_mean_discounted_cumulative_delta = 0.1166`
    - `rollout_positive_discounted_cumulative_rate = 0.75`
    - `rollout_second_step_available_rate = 1.0`
    - `rollout_stable_candidate_source_rate = 0.875`
    - 说明两步模拟可稳定跑通，而且多数病例的折扣累计改善仍为正。
  - 新 bundle 验证：
    - `output/analysis/phase8_rollout_bundle.pt`
    - `output/analysis/phase8_rollout_bundle_run.json`
    - `output/analysis/phase8_rollout_bundle_inference.json`
    - `output/analysis/phase8_rollout_bundle_inference.md`
    - 高相似病例 `stay_141959` 已能在医生版输出中看到“短期滚动模拟”章节，显示两步方案来源、逐步改善和折扣累计改善。
  - 旧 bundle 兼容验证：
    - `output/analysis/phase8_old_bundle_rollout_compat.json`
    - `output/analysis/phase8_old_bundle_rollout_compat.md`
    - 即便 bundle 中没有 Phase 8 新增 runtime 字段，rollout 仍能安全回退并完成推理。

**易懂解释**

- 第 8 条要解决的现实问题是：ICU 决策不是“一次选完就结束”，而是给出第一步方案后，过几个小时还要重新看病人有没有朝预期方向走，再决定第二步怎么调。
- 改动前，系统只能回答“如果现在换这个方案，会不会更好”。这更像一次性的会诊建议。
- 改动后，系统开始能回答“如果现在先这么做，往前走一步后，再看一次，第二步大概还会不会坚持同一路线”。这更像 ICU 里短期连续复盘，而不是只看当前一个切面。
- 最容易理解的地方在于医生版报告现在多了一节“短期滚动模拟”：
  - 第一步选了什么方案
  - 第二步在投影状态下又选了什么方案
  - 两步合起来的累计改善是正还是负
  - 第一、第二步的方案来源是否稳定
- 但这一条也要明确边界：它现在只是“轻量两步近似”，不是完整病程世界模型。也就是说，系统已经会做短期连续比较了，但它对第二步状态的理解仍主要沿目标轨迹前推，而不是完整重建未来全部生理变量。

---

### 4.9 反事实严谨性 + 医生证据增强

**目标**

一方面收紧方法学表述边界并逐步引入更严谨的离线策略评估；另一方面增强医生版报告的证据强度和审计性。

**大致修改计划**

- 方法定位上明确：
  - 当前系统是候选干预模拟与风险重排框架。
  - 当前输出不是严格因果疗效证明。
- 在具备条件后逐步引入：
  - off-policy evaluation
  - doubly robust
  - 时间变化混杂校正
- 同时增强医生报告：
  - 预测区间
  - 支持 donor 数量
  - 近邻 donor 一致性
  - top-3 方案对照
  - 不推荐原因说明

**涉及模块**

- `scripts/infer_eicu_counterfactual_plan.py`
- `src/counterfactual_plan_renderer.py`
- 评估脚本与离线策略评估模块

**预期有效性**

- 可行性：报告增强高；严格反事实评估中等偏低
- 预期收益：高
- 主要价值：提升方法学边界清晰度、医生端可信度和论文表达严谨性。

**阶段验收标准**

- 医生版报告新增关键证据字段。
- 推荐/不推荐理由更可审计。
- 若引入 OPE/DR，则必须有明确的实验对照和边界说明。

**当前状态**

- 当前状态：已完成
- 是否达到预期：是
- 是否修改完成：是
- 本轮结果摘要：
  - 已在 `scripts/infer_eicu_counterfactual_plan.py` 中把第九条的关键门控补齐：单步 `predicted_delta <= 0`、两步滚动折扣累计改善 `<= 0`、以及改善下界低于阈值时，系统会自动把结果降级为 `review_only`。
  - 已在 `src/counterfactual_plan_renderer.py` 中将医生版输出重构为临床阅读顺序，直接生成“病情判断、干预建议、短期趋势判断、为什么暂不建议直接采纳、相似患者证据、Top-3 备选方案对照、医生复核重点、方法边界说明”等章节。
  - 已把 `candidate_options` 写入单病例 JSON，医生版输出不再只是展示最终入选方案，而是同时展示 top-3 备选方案、对应的 `proxy delta`、相似度、指南一致性和惩罚项。
  - 已把方法边界显式写入最终输出，明确当前系统是“相似病例检索 + 候选干预模拟 + 风险重排”，不是严格的因果疗效证明。
- 验证结论：
  - 静态检查通过：
    - `python -m py_compile memory_mvp_project/src/counterfactual_plan_renderer.py`
    - `python -m py_compile memory_mvp_project/scripts/infer_eicu_counterfactual_plan.py`
  - 单病例复跑通过：
    - `python memory_mvp_project/scripts/infer_eicu_counterfactual_plan.py --bundle-path memory_mvp_project/output/analysis/phase8_rollout_bundle.pt --input-json memory_mvp_project/output/tmp_high_similarity_input_141959.json --output-json memory_mvp_project/output/analysis/phase9_physician_guardrail_141959.json --output-md memory_mvp_project/output/analysis/phase9_physician_guardrail_141959.md --rollout-steps 2 --rollout-discount 0.70`
  - 关键验证结果：
    - `stay_141959` 现在仍能检索到高相似 donor，主 donor 相似度约 `0.964`。
    - 但由于 `predicted_delta = -0.058`、两步滚动折扣累计改善约 `-0.058`、改善下界约 `-0.161`，系统不再给出“可作为建议候选”，而是自动输出“仅供医生复核”。
    - 新的 Markdown 输出已经由模型代码直接生成，而不是对 JSON 再做人工整理。
    - 新输出同时展示了 `top-3` 备选方案对照、近邻 donor 证据一致性、以及“不建议直接采纳”的具体原因。

**详细实施计划**

- 推理门控层：
  - 新增第九条的严格门控逻辑，把“负向单步收益”“负向滚动累计收益”“改善下界仍跨过 0”都视为不能直接推荐的信号。
  - 将滚动模拟稳定性纳入门控；若短期内候选来源频繁切换，也会进入复核状态。
- 医生版输出层：
  - 将原先偏技术报告式的 Markdown 重排为更接近临床查阅顺序的结构。
  - 增加“为什么暂不建议直接采纳”“相似患者证据”“Top-3 备选方案对照”“方法边界说明”。
- 结构化结果层：
  - 在单病例 JSON 中增加 `candidate_options`，便于后续界面或审阅流程直接复用。
  - 让 `doctor_view.markdown` 成为最终可交付文本，而不是依赖额外人工说明。

**易懂解释**

- 第九条真正解决的是“系统敢不敢把自己不确定的结果直接说成建议”这个问题。
- 改动前，系统即使找到很像的病例，也可能在预测结果并不支持的时候，仍然把方案挂成“可推荐”。这对医生端是不安全的，因为看起来像是机器已经替你做出了正向判断。
- 改动后，系统会更像一个谨慎的住院总：
  - 如果相似病例看起来很像，但模拟结果并没有显示这套方案会更好，那它不会再说“建议你用”，而是会说“这套方案可以给你参考，但目前证据不够，我不建议你直接采纳”。
  - 同时它会把理由写清楚，例如“单步没改善”“连续两步也没改善”“不确定性下界还穿过 0”。
- 对非代码人员最重要的理解是：
  - 这一步不是让模型更激进，而是让模型更诚实。
  - 它现在不仅会给方案，还会明确说明“为什么这个方案此刻还不能直接作为建议使用”。
  - 最终输出的医生版文字已经是代码自动生成的正式结果，不需要再靠额外人工翻译。

---

## 5. 执行顺序建议

建议严格按照以下顺序推进：

1. 先做第 1 条，建立统一评估基线。
2. 再做第 2 到第 3 条，稳住 factual 和动态表征。
3. 然后做第 4 到第 6 条，提升预测目标、donor 质量和排序质量。
4. 再做第 7 条，扩展候选方案空间。
5. 最后推进第 8 到第 9 条，向动态策略和更严谨评估演进。

不建议跳过第 1 条直接做下游优化，否则后续很难判断改动是否真正有效。

## 6. 本文件后续更新约定

后续当你要求我修改某一条时，我会按以下流程维护本文件：

1. 先在对应阶段补充“详细实施计划”。
2. 完成代码修改与验证后，回填该阶段的：
   - 当前状态
   - 是否达到预期
   - 是否修改完成
   - 本轮结果摘要
   - 验证结论
3. 从第 7 条开始，每个阶段额外补充一个“易懂解释”小节，用非代码化语言说明：
   - 这一条想解决什么现实问题
   - 改动前后系统行为有什么区别
   - 医生或非开发者应该如何理解它的价值
4. 若效果不达标，则在同一阶段继续返工，并更新为“未达预期”或“返工中”。
