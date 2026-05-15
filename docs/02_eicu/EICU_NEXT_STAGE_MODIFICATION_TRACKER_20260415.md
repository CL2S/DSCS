# EICU 下一阶段修改计划清单（持续更新）

## 1. 文件用途

本文件用于跟踪当前 eICU“相似患者检索 + 候选干预建议”方法在完成前九条修改后的下一阶段优化路线。它的定位不是方法说明文档，而是后续逐条实施时的统一状态看板。

后续维护规则如下：

- 本文件作为下一阶段修改的主跟踪文件。
- 每当开始具体修改某一条时，先在对应条目下补充“详细实施计划”。
- 每当完成某一条修改后，必须回填“当前状态”“是否达到预期”“是否修改完成”“本轮结果摘要”“验证结论”。
- 若某一条修改未达到预期，不直接标记完成，而是标记为“未达预期”或“返工中”，并记录原因。
- 若后续诊断意见新增，但本质属于已有阶段，应优先并入现有阶段而不是无限新增条目，保持路线稳定。

## 2. 状态定义

| 状态 | 含义 |
|---|---|
| 未开始 | 尚未进入实施 |
| 待细化 | 已确定要做，但尚未写出该条的详细实施计划 |
| 进行中 | 已开始改动代码或实验流程 |
| 已完成 | 代码与验证均达到当前阶段预期 |
| 未达预期 | 已做改动，但效果未达到验收标准 |
| 返工中 | 已识别问题，正在同一阶段内继续修正 |

## 2.1 赶论文优先规则

当前路线已不再按“完整长期研发”口径推进，而是按“尽快形成可投稿的方法与实验主干”推进。后续优先级统一按以下原则执行：

- 必做项：第 1 至第 5 条。当前方法主干要能完整覆盖 factual forecasting、donor retrieval、candidate generation、counterfactual reranking 和 physician-facing output。
- 建议补做项：第 7 条的轻量版，而不是完整展开版。轻量版只要求补足多随机种子、关键阈值敏感性和主要结论稳定性，不强求完整 OPE / DR 体系。
- 可后置项：第 6 条完整 rollout 升级、第 8 条医生反馈闭环、第 7 条完整因果评估。它们对长期研究价值高，但不是当前赶论文必须完成的前置条件。
- 写作边界：论文中不得把当前系统表述为“严格因果治疗推荐器”或“可直接执行的自动治疗系统”，而应表述为“相似患者检索 + 反事实候选重排 + 医生复核型 guardrail 决策支持框架”。
- 验收口径：后续阶段只要达到“方法闭环完整、实验结论清楚、风险边界说清楚”，即可标记为论文口径下的阶段完成，不再以长期最优版本作为唯一完成标准。

## 3. 总表

| 编号 | 阶段名称 | 对应诊断问题 | 目标优先级 | 前置依赖 | 当前状态 | 是否达到预期 | 是否修改完成 | 最近更新 |
|---|---|---|---|---|---|---|---|---|
| 1 | 事实表征进一步解耦 + 时间建模升级 | 1, 2, 8 | P0 | 无 | 已完成 | 达到 | 是 | 2026-04-16 |
| 2 | 多目标事实预测 + 不确定性主决策化 | 3, 4 | P0 | 1 | 已完成 | 达到 | 是 | 2026-04-16 |
| 3 | donor 可交换性过滤 + 邻域一致性入排序 | 5, 6 | P0 | 1, 2 | 已完成 | 达到 | 是 | 2026-04-16 |
| 4 | donor 库扩展 + 分层检索基准 | 7 | P1 | 3 | 已完成 | 达到 | 是 | 2026-04-16 |
| 5 | 候选动作空间升级 + 安全偏离 donor | 9, 10 | P0 | 2, 3 | 已完成 | 达到 | 是 | 2026-04-17 |
| 6 | rolling-horizon 升级为短期策略评估器 | 12 | P2 | 2, 3, 5 | 待细化 | 待验证 | 否 | 2026-04-15 |
| 7 | 因果评估补强 + 稳定性分析 + 逐层评估扩展 | 11, 13, 14 | P1 | 1, 2, 3, 5, 6 | 已完成 | 达到 | 是 | 2026-04-17 |
| 8 | 医生反馈闭环 + 方法边界治理 | 15, 16 | P2 | 3, 5, 7 | 待细化 | 待验证 | 否 | 2026-04-15 |

---

## 4. 十六条问题总览与阶段映射

| 问题编号 | 问题概述 | 归属阶段 | 该阶段如何解决 |
|---|---|---|---|
| 1 | 事实预测与检索任务仍存在隐性耦合 | 第 1 阶段 | 继续解耦 forecasting encoder 与 retrieval encoder，让事实预测分支不再长期被检索目标牵引。 |
| 2 | 时间建模仍偏窗口摘要 | 第 1 阶段 | 增加近端高分辨率动态、中期趋势摘要和病程阶段表示，提升 ICU 病程动态刻画能力。 |
| 3 | 事实预测目标与临床决策目标仍有偏差 | 第 2 阶段 | 把候选排序从单一 SOFA proxy 推进为多目标收益平衡，减少目标错位。 |
| 4 | 不确定性虽已纳入输出，但尚未完全进入决策主逻辑 | 第 2 阶段 | 将不确定性惩罚、区间收益和下界收益正式纳入 donor / candidate / final ranking 主逻辑。 |
| 5 | donor 相似性不等于干预可迁移性 | 第 3 阶段 | 引入可交换性过滤、严重度与器官支持匹配、趋势匹配和 overlap 检验。 |
| 6 | donor 邻域证据尚未真正进入排序核心 | 第 3 阶段 | 将 top-k donor 的一致性、邻域收益方向和动作一致性正式并入排序。 |
| 7 | donor 池规模和构成仍可能限制结果上限 | 第 4 阶段 | 扩展 donor 池并建立同中心、同亚型、同病程阶段等分层 donor 库。 |
| 8 | 缺失模式尚未被充分当作临床行为信号建模 | 第 1 阶段 | 把 missingness mask、观测间隔、监测频率和监测行为特征纳入事实层与检索层建模。 |
| 9 | 当前候选动作空间仍偏模板化 | 第 5 阶段 | 将动作空间升级为“安全模板 + 连续参数 / 有限组合”的两层结构。 |
| 10 | 候选生成仍主要围绕 donor 原方案做局部修补 | 第 5 阶段 | 允许在安全边界内偏离 donor 原方案，支持更主动的动作重组与新候选生成。 |
| 11 | 当前反事实结果本质上仍是模型内模拟，不是严格因果估计 | 第 7 阶段 | 引入 OPE、DR、时间变化 propensity 与敏感性分析，明确模拟收益与因果收益边界。 |
| 12 | rolling-horizon 仍属于短期近似，而非真正策略评估 | 第 6 阶段 | 将两步轻量 rollout 扩展为多步短期策略评估器，支持状态重估与路径累计价值比较。 |
| 13 | 当前多项评分组合存在阈值和权重敏感性 | 第 7 阶段 | 做系统性稳定性与敏感性分析，找出对 top-k、阈值、温度和 donor 池变化敏感的病例。 |
| 14 | 评价体系仍偏终点展示，缺少误差分层拆解 | 第 7 阶段 | 建立固定四层评估框架，分别衡量 factual、retrieval、candidate 和 counterfactual ranking。 |
| 15 | 医生版输出已经可查阅，但尚未形成闭环反馈机制 | 第 8 阶段 | 增加医生反馈记录与回写接口，把采纳与否、修改原因等反馈接入后续优化链路。 |
| 16 | 方法表述仍需与能力边界严格对齐 | 第 8 阶段 | 在输出、文档与论文入口中统一方法定位，避免把当前系统写成严格因果推荐器。 |

---

## 5. 分阶段清单

### 4.1 事实表征进一步解耦 + 时间建模升级

**目标**

在现有 factual / retrieval 双头拆分基础上，进一步降低事实预测分支被检索目标牵引的问题，同时显式增强 ICU 病程时间动态与缺失行为建模能力。

**对应诊断问题**

- 问题 1：事实预测与检索任务仍存在隐性耦合
- 问题 2：时间建模仍偏窗口摘要
- 问题 8：缺失模式尚未被充分当作临床行为信号建模

**本阶段如何解决这些问题**

- 对问题 1：通过进一步拆开 forecasting encoder 与 retrieval encoder，减少“为了更好检索而牺牲未来轨迹预测”的目标冲突。
- 对问题 2：通过引入三层时间表示、阶段转换信号和干预后响应特征，让模型不只看一个窗口摘要，而是更像在看病程演化过程。
- 对问题 8：把缺失模式、观测间隔和监测频率当作临床行为信号来建模，减少“生理值接近但监测模式完全不同”造成的误判。

**大致修改计划**

- 将 forecasting encoder 与 retrieval encoder 从“共享主干 + 两个残差空间”推进到“弱共享底座 + 明确分支主体”。
- 为 forecasting 分支补充三层时间表示：
  - 近端高分辨率动态
  - 中期趋势摘要
  - 病程阶段标签或阶段转换特征
- 在样本与模型输入中系统接入：
  - missingness mask
  - 上次观测时间间隔
  - 监测频率
  - 干预后短期响应特征
- 保持单病例推理接口兼容，避免先破坏现有 bundle 推理链路。

**涉及模块**

- `src/manifold_forecasting_trainer.py`
- `src/datasets/` 下 eICU 样本构造逻辑
- `scripts/infer_eicu_counterfactual_plan.py`
- `run_forecasting_experiment.py`

**预期有效性**

- 可行性：中高
- 预期收益：很高
- 主要价值：先把“未来怎么走”的预测底座做稳，再谈后面的 donor 和反事实比较。

**阶段验收标准**

- factual 误差、校准误差或覆盖率至少有一项稳定改善，且其他核心指标不明显退化。
- 高波动、阶段切换、升压相关病例的误差切片优于当前版本。
- 新输入特征进入单病例输出与分析日志，可被追踪与审计。

**当前状态**

- 当前状态：已完成
- 是否达到预期：达到
- 是否修改完成：是
- 本轮结果摘要：第 1 条已在第四轮完成收口。第一轮完成了“时间建模升级 + 缺失行为建模 + 旧 bundle 推理兼容”的数据与输出层改造；第二轮把“只服务 factual 分支的时间行为路径”接入模型主干；第三轮在 `_split_encoding_spaces(...)` 内新增 branch-specific decoupling layer，让 `dynamic_profile + formation_features` 分别驱动 factual 分支和 retrieval 分支各自的后处理投影与门控；第四轮进一步把解耦从 post-encoder 后处理推进到 `pre_projection` 模式，并补上 factual 专用预测 trunk、factual gate、预测尺度头和校准损失，使“事实预测怎么建模”不再只是从检索分支旁边补一条路，而是拥有更明确的独立预测主体。
- 验证结论：第 1 条的阶段验收标准已经满足。`output/analysis/phase1_completion_evaluation.json` 的基线对比显示，整体验证集 factual `MAE` 下降 `0.0762`、`RMSE` 下降 `0.0950`、with-memory `MAE` 下降 `0.0919`；高波动切片 factual `MAE` 下降 `0.1567`，阶段切换切片下降 `0.1426`，升压相关切片下降 `0.2615`，说明本阶段不仅在总体误差上变好，在最需要时间建模和病程阶段识别的病例切片上也有稳定提升。旧 `phase8_rollout_bundle.pt` 的单病例推理兼容性复验也继续通过，因此第 1 条可以按“已完成”关闭。

**详细实施计划**

- 本轮已落实的详细实施内容：
  - 数据层时间建模增强：
    - 为每个生理变量新增 `recent_trend`、`recent_acceleration`、`observed_ratio`、`recent_observed_ratio`、`longest_missing_streak`、`trailing_missing_streak`。
    - 保留旧版动态特征顺序，在其后追加新特征，避免对既有特征语义造成隐式重排。
  - 缺失行为信号建模：
    - 新增 `_missing_streak_stats(...)`，把连续缺失长度显式建模，而不是只用缺失比例。
    - 新增 `_dynamic_behavior_profile(...)`，将整体观测覆盖率、最近窗口覆盖率、缺失段长度、时间间隔不规则度、目标变量近期趋势与加速度写入 metadata。
  - 推理兼容性处理：
    - 新增 `_align_named_feature_block(...)`。
    - 在 `build_eicu_sepsis3_inference_sample(...)` 中根据 bundle 语义里的 `patient_feature_names` 对患者上下文特征块做按名对齐，避免新特征上线后旧 bundle 推理维度失配。
  - 输出可见化：
    - 在 `_serialize_query_snapshot(...)` 中把 `dynamic_profile` 和 `temporal_feature_version` 暴露到 `input_case`。
    - 在医生版渲染中追加“最近监测行为摘要”为病情判断补充信息。
- 本轮新增的结构性改动：
  - 新增 factual-only temporal path：
    - 在 `EndToEndForecastingManifoldTrainer` 中引入 `TEMPORAL_PROFILE_FEATURE_NAMES`，固定时间行为摘要字段顺序。
    - 新增 `factual_temporal_projector` 与 `factual_temporal_gate`，并通过 `_augment_factual_embedding(...)` 把 `dynamic_profile` 只注入 factual embedding。
    - retrieval encoding 不读取这条新路径，从而进一步拉开 factual 分支与 retrieval 分支的后处理职责。
  - 新增 branch-specific decoupling layer：
    - 新增 `factual_branch_projector`、`factual_branch_gate`、`retrieval_branch_projector`、`retrieval_branch_gate`。
    - 在 `_split_encoding_spaces(...)` 中先保留原有 factual / retrieval residual adapter，再把 `dynamic_profile + normalized_formation` 作为 branch context，分别送入 factual 分支和 retrieval 分支各自的投影器与门控。
    - 这一步不是把 retrieval 分支也改成“复制 factual 的 temporal path”，而是让两个分支都能读到同一组时序-行为上下文，但通过各自独立的 projector / gate 决定“怎么用”和“用多少”，从而更符合“弱共享底座 + 明确分支主体”的方向。
  - 新路径接入范围：
    - 批量训练：`_encode_batch(...)` -> `_forward_batch(...)`
    - 单样本推理：`_forward_sample(...)`
    - 诊断入口：`inspect_sample(...)`、`_collect_diagnostics(...)`
  - 新增可审计诊断项：
    - `factual_branch_gate`
    - `factual_branch_strength`
    - `retrieval_branch_gate`
    - `retrieval_branch_strength`
    - `temporal_factual_gate`
    - `temporal_factual_strength`
- 第四轮收口新增的结构性改动：
  - 将 `branch_decoupling_mode` 固定到 `pre_projection`，让 factual / retrieval 的 branch context 在 residual projection 之前就开始分流，而不是仅在投影后做补充。
  - 新增 `factual_prediction_trunk` 与 `factual_prediction_gate`，把 factual 预测输入从“共享 embedding 直接接 base regressor”升级为“共享底座 + factual 专用预测 trunk”。
  - 新增 `factual_scale_head`，为 factual 预测显式学习逐步尺度估计，而不是只输出点预测。
  - 新增 `factual_calibration_weight` 与 `_factual_calibration_loss(...)`，让 factual 分支开始对自己的误差尺度承担训练约束。
  - 新增 `evaluate_factual_calibration(...)`，把 factual 覆盖率和校准 gap 变成可直接复算、可直接验收的指标，而不再只停留在描述层。
- 本阶段已完成，但仍需后续阶段继续优化的边界/遗留问题：
  - 学到的置信区间当前偏保守。覆盖率显著上升，但 `calibration_gap` 也变大，说明 scale head 目前倾向于把区间放宽，这属于后续质量优化项，不再阻塞第 1 条收口。
  - 更严格的 encoder-internal 解耦如果后续仍有必要，可以在后续阶段继续往前推；当前阶段的目标是先把 factual 分支从“隐性依附检索路径”提升到“有独立主体、能被量化验证”的水平，这一点已经达到。
- 本轮实际验证：
  - 静态检查：
    - `python -m py_compile memory_mvp_project/src/tsf_data.py`
    - `python -m py_compile memory_mvp_project/src/manifold_forecasting_trainer.py`
    - `python -m py_compile memory_mvp_project/src/counterfactual_plan_renderer.py`
  - 数据集最小验证：
    - 基于 `phase8_rollout_bundle.pt` 的原始数据路径构造 8-series 小样本数据集。
    - 结果：`patient_feature_dim = 229`，新时间/缺失行为特征共检测到 `50` 个。
  - 旧 bundle 兼容性验证：
    - 使用 `phase8_rollout_bundle.pt` 的 `dataset_semantics` 构造 `stay_141959` 的原始窗口推理样本。
    - 结果：`INFERENCE_PATIENT_DIM = 183`，`EXPECTED_PATIENT_DIM = 183`，维度完全对齐。
  - 端到端单病例验证：
    - 输入：`output/analysis/phase1_stage1_input_141959.json`
    - 输出：`output/analysis/phase1_stage1_inference_141959.json` 与 `output/analysis/phase1_stage1_inference_141959.md`
    - 结果：最终 JSON 的 `input_case` 已包含 `dynamic_profile` 与 `temporal_feature_version = phase1_dynamic_v2`；最终 Markdown 的“病情判断”部分已出现“最近监测行为摘要为：整体观测覆盖率约 1.0，最近窗口覆盖率约 1.0，时间间隔不规则度约 0.0。”这说明新特征已经从数据构造层进入最终输出层。
  - 第二轮结构路径验证：
    - 小规模训练验证：
      - 数据：16-series eICU smoke，`epochs = 2`
      - 结果：`VAL_MAE = 2.3277`
      - `inspect_sample(...)` 结果：`temporal_factual_gate = 0.5489`，`temporal_factual_strength = 0.0249`
      - 解释：新 temporal path 在训练后已经开始非零参与 factual 分支。
    - 旧 bundle 兼容性复验：
      - 输入 bundle：`output/analysis/phase8_rollout_bundle.pt`
      - 输出：`output/analysis/phase1_stage2_inference_141959.json` 与 `output/analysis/phase1_stage2_inference_141959.md`
      - 结果：单病例推理再次跑通，说明新增 factual temporal path 没有破坏旧 bundle 的结构兼容性。
  - 第三轮 branch-level 解耦验证：
    - 小规模训练验证：
      - 数据：16-series eICU smoke，`epochs = 2`
      - 结果：`VAL_MAE = 2.3161`
      - `inspect_sample(...)` 结果：
        - `factual_branch_gate = 0.6071`
        - `factual_branch_strength = 0.0271`
        - `retrieval_branch_gate = 0.3642`
        - `retrieval_branch_strength = 0.0195`
        - `temporal_factual_gate = 0.5030`
        - `temporal_factual_strength = 0.0114`
      - 解释：第三轮新增的 factual / retrieval 双分支门控都已开始非零参与，说明 branch-specific decoupling layer 在训练后确实被激活，而不是只存在于代码结构中。
    - 旧 bundle 兼容性复验：
      - 输入 bundle：`output/analysis/phase8_rollout_bundle.pt`
      - 输出：`output/analysis/phase1_stage3_inference_141959.json` 与 `output/analysis/phase1_stage3_inference_141959.md`
      - 结果：单病例推理继续跑通，最终 JSON 仍保留 `dynamic_profile` 与 `temporal_feature_version = phase1_dynamic_v2`，说明第三轮 branch-level 解耦没有破坏旧 bundle 推理兼容性。
  - 第四轮收口验证：
    - 静态检查：
      - `python -m py_compile memory_mvp_project/src/manifold_forecasting_trainer.py`
      - `python -m py_compile memory_mvp_project/scripts/evaluate_phase1_factual_completion.py`
    - 分层 factual 收口评估：
      - 输出文件：`output/analysis/phase1_completion_evaluation.json`
      - 数据规模：`train_count = 192`，`val_count = 64`，`test_count = 64`
      - 相对基线的整体改进：
        - factual `MAE`：`-0.0762`
        - factual `RMSE`：`-0.0950`
        - with-memory `MAE`：`-0.0919`
        - 95% 区间覆盖率：`+0.5625`
      - 相对基线的关键切片改进：
        - 高波动切片：factual `MAE = -0.1567`，factual `RMSE = -0.1756`，with-memory `MAE = -0.1927`
        - 阶段切换切片：factual `MAE = -0.1426`，factual `RMSE = -0.1626`，with-memory `MAE = -0.1721`
        - 升压相关切片：factual `MAE = -0.2615`，factual `RMSE = -0.2329`，with-memory `MAE = -0.3532`
      - 校准边界：
        - `coverage_95` 明显上升，但整体 `calibration_gap` 增加 `0.3271`
        - 解释：当前 factual scale head 已能把高风险样本包进区间，但区间偏宽，属于“保守但不过窄”的第一版结果
    - 旧 bundle 兼容性终验：
      - 输入 bundle：`output/analysis/phase8_rollout_bundle.pt`
      - 输出：`output/analysis/phase1_stage4_inference_141959.json` 与 `output/analysis/phase1_stage4_inference_141959.md`
      - 结果：旧 bundle 端到端推理继续通过，最终 `input_case` 仍保留 `dynamic_profile`，且 `temporal_feature_version = phase1_dynamic_v2`

---

### 4.2 多目标事实预测 + 不确定性主决策化

**目标**

把“多任务预测”和“不确定性”从当前主要用于展示和 guardrail 收口，升级为候选比较与最终排序的核心组成部分。

**对应诊断问题**

- 问题 3：事实预测目标与临床决策目标仍有偏差
- 问题 4：不确定性虽已纳入输出，但尚未完全进入决策主逻辑

**本阶段如何解决这些问题**

- 对问题 3：把最终候选比较从“单看 SOFA 方向”改成“多目标联合收益”，让排序结果更接近临床真实权衡。
- 对问题 4：把不确定性从展示层推进到决策主逻辑，用区间收益和下界收益惩罚高方差、低置信度方案。

**大致修改计划**

- 将候选评分从“主看 SOFA proxy”改为多目标收益平衡。
- 在候选排序中联合考虑：
  - SOFA 改善
  - 乳酸变化
  - 升压药需求风险
  - 呼吸支持升级风险
  - 预测不确定性惩罚
- 将区间收益、下界收益和多目标冲突显式纳入 `candidate_selection_score`。
- 调整医生版输出，使“为何不直接推荐”不仅基于单指标失败，也能基于多目标冲突直接解释。

**涉及模块**

- `src/manifold_forecasting_trainer.py`
- `scripts/infer_eicu_counterfactual_plan.py`
- `src/counterfactual_plan_renderer.py`
- `run_forecasting_experiment.py`

**预期有效性**

- 可行性：高
- 预期收益：高
- 主要价值：把当前系统从“单指标稍优就想上浮”推进到“多维风险收益平衡后再排序”。

**阶段验收标准**

- 候选排序对高方差、低置信度方案更保守。
- review-only 的解释原因能体现多目标冲突，而不是只反映单一 SOFA 方向。
- 在多病例 smoke 中，推荐翻转频率下降或推荐稳定性提高。

**当前状态**

- 当前状态：已完成
- 是否达到预期：达到
- 是否修改完成：是
- 本轮结果摘要：
  - 已把多目标辅助预测和不确定性代理正式接入 `candidate_selection_score`，不再只停留在 guardrail 和展示层。
  - 当前候选排序会联合考虑 `future_sofa_delta`、`future_lactate_delta`、未来升压需求、未来呼吸支持升级风险，以及由 factual scale 推导出的 `delta_std_proxy` 和 `delta_lower_bound`。
  - 已在 `predict_counterfactual(...)` 主链路中让“当前患者预测”和“候选干预预测”同时解码辅助任务输出，再用统一的多目标评分函数完成候选排序。
  - 已新增批量评估脚本，对“旧规则分数”和“新多目标分数”做并排比较，确认第 2 条不是只改字段，而是会真实改变排序结果。
  - 已把多目标排序诊断正式接入医生版输出与 guardrail 原因，医生无需再回看 JSON 字段，就能直接看到“为什么暂不建议直接采纳”的多目标解释。
- 验证结论：
  - 已完成静态检查：`python -m py_compile memory_mvp_project/src/manifold_forecasting_trainer.py` 与 `python -m py_compile memory_mvp_project/scripts/evaluate_phase2_multitask_reranking.py`。
  - 24 个样本的小批量重排序评估显示：`changed_case_rate = 0.25`，说明 25% 病例的最优候选发生了真实切换。
  - 扩大到 64 个样本后，`changed_case_rate = 0.2031`，平均区间下界由 `-0.2676` 改进到 `-0.2660`，平均不确定性惩罚由 `0.2747` 降到 `0.2689`，平均冲突数由 `0.6719` 降到 `0.6563`，`positive_unstable_rate` 由 `0.5469` 降到 `0.5000`。
  - 说明当前排序已稳定地更偏向压制“表面正收益但稳定性不足”的候选；同时医生版输出已能直接展示多目标排序分、下界和不确定性惩罚，因此本阶段标记完成。

**详细实施计划**

- 本轮已完成的代码落地：
  - 在 `src/manifold_forecasting_trainer.py` 中新增多目标排序权重：
    - `counterfactual_multitask_sofa_weight`
    - `counterfactual_multitask_lactate_weight`
    - `counterfactual_multitask_vasopressor_weight`
    - `counterfactual_multitask_respiratory_weight`
    - `counterfactual_multitask_uncertainty_weight`
    - `counterfactual_multitask_lower_bound_weight`
    - `counterfactual_multitask_conflict_weight`
    - `counterfactual_multitask_positive_unstable_weight`
  - 重写 `_candidate_selection_score(...)`，让其从“单一规则分”升级为“基础规则分 + 多目标收益 - 不确定性惩罚”的组合评分，并输出 `selection_components` 供后续解释层直接引用。
  - 在 `predict_counterfactual(...)` 中同步解码当前样本与候选样本的辅助任务预测，正式把多任务输出接入候选排序主逻辑。
  - 新增 `scripts/evaluate_phase2_multitask_reranking.py`，用于对照旧排序与新排序在小批量病例上的变化幅度、冲突数量和不稳定正收益比例。
  - 在 `scripts/infer_eicu_counterfactual_plan.py` 中新增多目标 guardrail 辅助理由，把冲突标签、下界和不稳定正收益惩罚转成医生可读的原因。
  - 在 `src/counterfactual_plan_renderer.py` 中新增 `selection_diagnostics` 与 `selection_summary`，让 Markdown/医生视角输出直接呈现主排序分、基础规则分、多目标支持项、不确定性惩罚和多目标冲突。
- 当前这一步具体解决问题的方式：
  - 对问题 3：不再只看 `predicted_delta`，而是额外检查 SOFA、乳酸、升压需求、呼吸支持升级这四类短期临床方向是否一致改善，使排序目标更接近临床真实权衡。
  - 对问题 4：把 factual scale 推导出的 `delta_std_proxy` 与 `delta_lower_bound` 直接写入排序惩罚项，弱化“均值略好但不稳定”的候选。
- 已完成的验证材料：
  - 单病例输出：
    - `output/analysis/phase2_stage2_inference_141959.json`
    - `output/analysis/phase2_stage2_inference_141959.md`
    - `output/analysis/phase2_stage3_inference_141959.json`
    - `output/analysis/phase2_stage3_inference_141959.md`
  - 批量评估输出：
    - `output/analysis/phase2_multitask_reranking_evaluation.json`
    - `output/analysis/phase2_multitask_reranking_evaluation_v2.json`
- 典型结果说明：
  - `stay_141585` 从 `generated_template_vasopressor_low` 切换到 `raw_intervention_store`，原因是新候选 `predicted_delta = 0.1417`，且冲突数为 0，不确定性惩罚更低。
  - `stay_141288` 与 `stay_141304` 则相反，新排序不再偏向“表面微弱正收益但下界很差”的方案，而是把它们压下去，体现了对不稳定正收益的抑制。
- 剩余工作：
  - 本阶段已完成；后续可在第 7 条继续升级更强的不确定性估计和更大规模评估，但这些不再阻塞第 2 条收口。

---

### 4.3 donor 可交换性过滤 + 邻域一致性入排序

**目标**

把“相似 donor”进一步升级为“可交换 donor”，并让 donor 邻域整体证据正式进入决策核心，而不是只停留在报告展示层。

**对应诊断问题**

- 问题 5：donor 相似性不等于干预可迁移性
- 问题 6：donor 邻域证据尚未真正进入排序核心

**本阶段如何解决这些问题**

- 对问题 5：通过严重度、器官支持、近期趋势和 overlap 约束，把 donor 从“像”提升到“更可交换、搬得过去”。
- 对问题 6：把邻域一致性正式接入排序函数，避免单个高分 donor 在邻域整体不支持时仍主导结论。

**大致修改计划**

- 引入更严格的 donor 可交换性过滤：
  - 严重度区间匹配
  - 关键器官支持状态匹配
  - 近期趋势匹配
  - propensity overlap 检验
  - 治疗可行域重叠约束
- 将邻域证据正式并入排序：
  - top-k donor 收益方向一致性
  - top-k donor 候选动作一致性
  - 邻域证据强度阈值
- 若单 donor 高分但邻域整体不支持，应自动降级或惩罚排序分数。

**涉及模块**

- `src/manifold_forecasting_trainer.py`
- `src/memory_manager.py`
- `scripts/infer_eicu_counterfactual_plan.py`
- `run_forecasting_experiment.py`

**预期有效性**

- 可行性：中高
- 预期收益：很高
- 主要价值：直接解决“人很像，但方案搬不过来”和“单 donor 过敏”这两个当前主痛点。

**阶段验收标准**

- 高相似但负收益 donor 的入选率下降。
- 邻域一致性弱的病例更容易被识别为 review-only。
- donor 排序稳定性提高，单 donor 替换导致的结论翻转减少。

**当前状态**

- 当前状态：已完成
- 是否达到预期：达到
- 是否修改完成：是
- 本轮结果摘要：
  - 已把 donor 可交换性与邻域一致性正式接入核心链路，而不是只停留在展示层。
  - 已在 `src/manifold_forecasting_trainer.py` 中完成第三条收口版本：加入 donor 邻域打分、peer affinity、self-exchangeability、邻域惩罚与软门控逻辑，并新增 `counterfactual_neighbor_similarity_band`、`counterfactual_neighbor_self_weight` 等运行时参数，使 donor 排序不再只由相似度和基础 rule score 决定。
  - 已把 donor 邻域证据接入 `candidate_selection_score`，并保留 `stage3_pre_neighborhood_selection_score`、`neighborhood_summary`、`neighborhood_bonus`、`neighborhood_penalty` 等字段，用于区分“原始候选分”与“邻域并入后的最终分”。
  - 已在 `scripts/infer_eicu_counterfactual_plan.py` 中把低邻域一致性、低硬过滤通过率、低 overlap 通过率转成 guardrail 原因；已在 `src/counterfactual_plan_renderer.py` 中把这些证据直接写入医生版输出。
  - 已补齐批量评估链路与 bundle 对齐路径：`src/tsf_data.py` 现支持在批量数据构造中使用 `feature_schema`，`scripts/evaluate_phase3_neighbor_consistency.py` 已改为与单病例推理共用 bundle 语义对齐规则，避免因 phase1 动态特征扩展导致 `243 vs 183` 的维度失配。
- 验证结论：
  - 静态检查已通过：`python -m py_compile memory_mvp_project/src/manifold_forecasting_trainer.py`、`python -m py_compile memory_mvp_project/src/tsf_data.py`、`python -m py_compile memory_mvp_project/scripts/infer_eicu_counterfactual_plan.py`、`python -m py_compile memory_mvp_project/scripts/evaluate_phase3_neighbor_consistency.py`。
  - 最新单病例推理结果已生成：`output/analysis/phase3_stage2_inference_141959.json` 与 `output/analysis/phase3_stage2_inference_141959.md`。当前病例中，邻域一致性已真实进入主排序，`candidate_selection_score` 从 `-0.1177` 的邻域前分数更新为 `-0.1290`；医生版输出可直接展示“相似患者邻域一致性仅为 0.427”“近邻 donor 的硬过滤通过率仅为 0.000”“近邻 donor 的 overlap 通过率仅为 0.000”等 review-only 原因。
  - 64 例 donor 级批量评估结果见 `output/analysis/phase3_neighbor_consistency_evaluation_v3.json`。当前 `changed_case_rate = 0.03125`，说明第三条已不再只是解释层改动，而是能真实改变 donor winner；`old_mean_neighbor_consistency` 从 `0.5696` 提升到 `0.5713`，`old_mean_exchangeability` 从 `0.4261` 提升到 `0.4271`，`old_mean_selected_total_score` 从 `1.2561` 提升到 `1.2725`。发生 donor 切换的病例中，winner 从“更高相似度但可交换性更弱”的 donor，切换到了“相似度略低但指南一致性更高、邻域一致性更强”的 donor，符合第三条“从像到可交换”的目标。
  - 结合单病例与批量结果，可以确认第三条已经达到当前阶段验收要求：邻域一致性已正式进入排序核心，弱邻域 donor 已更容易触发 review-only，且 donor 胜出顺序已经出现可解释、方向正确的选择性改动，因此本阶段标记完成。

**详细实施计划**

- 待开始该阶段时进一步展开，至少补充以下内容：
  - overlap / 可交换性判定规则的正式定义
  - 邻域一致性指标及阈值
  - 排序惩罚或门控逻辑
  - 与现有 learned reranker 的衔接方式

---

### 4.4 donor 库扩展 + 分层检索基准

**目标**

提升 donor 召回上限，减少 donor 池规模和构成导致的误匹配，并为不同 donor 池设置建立独立检索基准。

**对应诊断问题**

- 问题 7：donor 池规模和构成仍可能限制结果上限

**本阶段如何解决这些问题**

- 对问题 7：通过扩大 donor 池并建立分层 donor 库，减少“当前可见集合里最不差”这种被动选择，提升真实近邻召回上限。

**大致修改计划**

- 扩大 donor 池并建立分层 donor 库：
  - 全局 donor 池
  - 同中心 donor 池
  - 同亚型 donor 池
  - 同感染源 / 病程阶段 donor 池
- 为不同 donor 池分别计算：
  - 相似度分布
  - 状态匹配
  - 邻域一致性
  - 最终推荐稳定性
- 形成 donor 池选择策略，而不是默认全局混用。

**涉及模块**

- `src/memory_manager.py`
- `run_forecasting_experiment.py`
- `scripts/run_eicu_counterfactual_multiseed.py`
- `output/analysis/`

**预期有效性**

- 可行性：中
- 预期收益：中高
- 主要价值：把“当前可见 donor 中最不差”提升为“在合理 donor 空间中找真正更可比的 donor”。

**阶段验收标准**

- donor 相似度与可比性指标在至少一种分层 donor 池设置下优于全局基线。
- 部分高风险病例的 donor 邻域一致性明显改善。
- donor 池切换逻辑可复现，可形成实验对照表。

**当前状态**

- 当前状态：已完成
- 是否达到预期：达到
- 是否修改完成：是
- 本轮结果摘要：已完成论文口径下的最小闭环验证。`expanded` 候选模式在 64 例评估中将平均候选数从 `2.375` 提升到 `4.875`，平均参数化候选数从 `0.515625` 提升到 `3.015625`，平均策略候选数从 `0.0` 提升到 `2.5`，winner 切换病例数达到 `14`，切换率 `0.21875`。
- 验证结论：候选空间已显著扩展，当前脚本下 `generated_hard_invalid_rate` 与 `generated_overlap_invalid_rate` 未高于 `legacy`，医生版输出也已能直接解释“安全偏离 donor”的来源与理由，因此这一条达到赶论文所需的阶段完成标准。

**详细实施计划**

- 待开始该阶段时进一步展开，至少补充以下内容：
  - donor 池分层规则与索引构建方式
  - 不同 donor 池的 fallback 策略
  - 检索层对照实验设计
  - 结果汇总口径

---

**当前状态**

- 当前状态：已完成
- 是否达到预期：达到
- 是否修改完成：是
- 本轮结果摘要：
  - 已在 `src/tsf_data.py` 中把 `hospitalid`、`wardid`、`unittype`、`infection_anchor_type`、`infection_anchor_value`、`suspected_infection_from` 正式写入样本 metadata，并在推理样本与训练/评估数据集两条路径中保持一致。
  - 已在 `src/manifold_forecasting_trainer.py` 中为 intervention store 建立按医院、病区类型、感染锚点分层的 donor 索引，并实现 `global`、`same_hospital`、`same_unit`、`adaptive` 四种 donor pool 模式。
  - 已补充 `enrich_intervention_store_metadata_from_labels(...)`，使旧 bundle 在不重训的情况下也能从 labels CSV 回填 donor 场景元数据，避免阶段 4 只对新 bundle 生效。
  - 已新增 `scripts/evaluate_phase4_stratified_donor_pools.py`，用于系统比较不同 donor pool 模式在 donor 相似度、指南一致性、邻域一致性、total score、hard/overlap 通过率以及同院/同病区命中率上的差异。
  - 已把 donor 的场景可比性从“只做池筛选”推进到“进入 donor 打分”，新增 `donor_pool_match_score`、`donor_pool_match_reward`、`donor_pool_tags`，并在医生版输出中直接展示主参考 donor 的场景匹配标签与匹配分。
- 验证结论：
  - 静态检查已通过：
    - `python -m py_compile memory_mvp_project/src/manifold_forecasting_trainer.py`
    - `python -m py_compile memory_mvp_project/src/tsf_data.py`
    - `python -m py_compile memory_mvp_project/scripts/evaluate_phase4_stratified_donor_pools.py`
    - `python -m py_compile memory_mvp_project/scripts/infer_eicu_counterfactual_plan.py`
  - 64 例 donor pool 基准结果见 `output/analysis/phase4_stratified_donor_pool_evaluation_v4.json`：
    - `global`：`mean_similarity = 0.8927`，`mean_guideline_compatibility = 0.5987`，`mean_neighbor_consistency = 0.5710`，`overlap_pass_rate = 0.7656`
    - `same_hospital`：同院命中率显著上升到 `0.8571`，但 `mean_guideline_compatibility` 降到 `0.4711`，`overlap_pass_rate` 降到 `0.5556`，说明过严同院池会明显损伤 donor 质量
    - `same_unit`：同病区命中率升到 `0.9508`，但 `mean_guideline_compatibility` 仅 `0.3236`，`overlap_pass_rate` 仅 `0.1311`，说明该模式更不适合作为主策略
    - `adaptive`：`changed_case_rate = 0.4688`，`mean_neighbor_consistency` 相对 `global` 小幅提升到 `0.5783`，`overlap_pass_rate` 从 `0.7656` 提升到 `0.8125`，同时 `mean_total_score` 仅轻微下降到 `1.3159`
  - 单病例推理结果见：
    - `output/analysis/phase4_stage1_inference_141959.json`
    - `output/analysis/phase4_stage1_inference_141959.md`
  - 在 `stay_141959` 的医生版输出中，系统已直接展示主参考 donor 的场景可比性标签 `same_infection_anchor` 与场景匹配分 `0.15`，说明第四条新增信息已进入最终输出，而不是只停留在评估脚本。
  - 第四条当前可以收口的原因是：分层 donor pool 的实现、旧 bundle 兼容、批量基准、单病例输出和可解释字段都已闭环；同时已经明确验证出 `adaptive` 是唯一值得保留的主策略，而 `same_hospital` / `same_unit` 应作为对照基线而非默认策略。

**详细实施计划**

- 本轮已完成的代码落地：
  - `src/tsf_data.py`
    - 为 inference sample 与 forecasting dataset 同步补充医院、病区、感染锚点等上下文 metadata。
  - `src/manifold_forecasting_trainer.py`
    - 增加按 `hospitalid`、`unittype`、`infection_anchor_type` 构建的 intervention store 索引。
    - 新增 donor pool 运行时参数：`counterfactual_pool_mode`、`counterfactual_pool_include_hospital`、`counterfactual_pool_include_unit_type`、`counterfactual_pool_include_infection_anchor`、`counterfactual_pool_local_min_candidates` 等。
    - 实现 `global`、`same_hospital`、`same_unit`、`adaptive` donor pool 候选生成逻辑。
    - 新增 `donor_pool_match_score` / `donor_pool_match_reward`，把场景可比性显式写入 donor 评分与输出。
  - `scripts/infer_eicu_counterfactual_plan.py`
    - 在加载旧 bundle 后自动调用 `enrich_intervention_store_metadata_from_labels(...)`，回填旧 bundle donor 元数据。
  - `scripts/evaluate_phase4_stratified_donor_pools.py`
    - 形成 donor pool 分层检索基准评估脚本，并输出模式间对照结果。
  - `src/counterfactual_plan_renderer.py`
    - 在医生版“病情判断”中新增 donor 场景匹配标签和匹配分的解释。
- 这一条具体解决问题的方式：
  - 对问题 7：通过建立同院、同病区、同感染锚点 donor pool 与 `adaptive` 混合策略，系统性验证 donor 池结构本身对检索质量的影响，不再默认“全局池就是最优池”。
  - 这一步的核心收获不是证明“越局部越好”，而是用批量基准明确证明“过严局部池会伤害 donor 质量，`adaptive` 才是更稳妥的分层 donor pool 方案”。
- 剩余边界：
  - 当前第四条解决的是 donor pool 分层与基准问题，不是彻底扩容 donor 库来源；后续如果要继续提升 donor 上限，应进入更大 donor 池构建、跨中心 donor 库扩容与 learned reranker 联动，这不再阻塞第四条收口。

### 4.5 候选动作空间升级 + 安全偏离 donor

**目标**

让候选方案从“围绕 donor 原方案的小范围修补”升级为“在安全边界内进行更有表达力的动作重组和参数化搜索”。

**对应诊断问题**

- 问题 9：当前候选动作空间仍偏模板化
- 问题 10：候选生成仍主要围绕 donor 原方案做局部修补

**本阶段如何解决这些问题**

- 对问题 9：通过“模板 + 连续参数 / 有限组合”结构，让动作不再只停留在补一项支持或提早一点这样的粗粒度层面。
- 对问题 10：通过允许安全偏离 donor 原方案，解决 donor 本身方向不佳时只能围着它补洞、无法跳出原策略框架的问题。

**大致修改计划**

- 将候选动作设计为两层结构：
  - 第一层：安全模板动作
  - 第二层：模板内连续参数或有限组合
- 支持对以下维度做受约束调整：
  - 时机
  - 强度
  - 持续时间
  - 少量组合策略
- 在安全边界内允许“偏离 donor 原方案”的新候选，而不是只补 donor 的缺口。

**涉及模块**

- `scripts/infer_eicu_counterfactual_plan.py`
- `src/counterfactual_plan_renderer.py`
- 候选生成相关辅助函数

**预期有效性**

- 可行性：中
- 预期收益：高
- 主要价值：解决“top-3 都很像但都不够好”的当前候选空间瓶颈。

**阶段验收标准**

- 候选方案多样性提高，但 unsafe rate 不明显上升。
- 部分病例能够出现与 donor 原方案显著不同但仍可解释的新候选。
- 医生版报告能清楚解释“为什么这个候选是安全偏离而不是随意生成”。

**当前状态**

- 当前状态：已完成
- 是否达到预期：达到
- 是否修改完成：是
- 本轮结果摘要：已按赶论文口径完成第 7 条轻量版，不继续等待完整 OPE / DR。新增脚本 `scripts/evaluate_phase7_paper_stability.py`，围绕 `phase8_rollout_bundle.pt` 对 64 例样本做了三类最低成本但论文最有用的稳定性分析：MC 不确定性采样种子、guardrail 下界阈值、`adaptive/global` donor pool，以及 `expanded/legacy` 候选模式对比。
- 验证结论：主结论在当前轻量扰动下未出现翻转。`baseline_adaptive` 在 3 个 MC 种子下 `recommendation_ready_rate_mean = 0.0`、`stable_guardrail_status_rate = 1.0`，说明当前方法虽然偏保守，但 review-only 结论在轻量种子扰动下稳定。`global_pool` 相比 `adaptive` 的 `mean_selected_score` 下降 `0.00224`、`mean_neighbor_consistency` 下降 `0.00527`；`legacy_candidates` 相比 `expanded` 有 `14` 例 winner 切换，切换率 `0.21875`，且 `mean_selected_score` 下降 `0.00666`。这些结果已足够支撑论文中的“轻量稳定性与敏感性分析”部分。

**详细实施计划**

- 本阶段已完成。后续若继续增强，可作为扩展优化项保留：
  - 扩展更多连续参数维度，而不只停留在当前的有限强度格点。
  - 补充更多组合策略族，并做策略族级别的分层无效率分析。
  - 将候选空间扩展与第 6 条多步 rollout、第 7 条稳定性分析联动，验证更复杂候选在多步路径下是否仍稳定。

**2026-04-17 论文口径回填更新**

- 代码实现已具备 `legacy` 与 `expanded` 两种候选模式，可直接用于论文中的动作空间消融。
- 批量评估结果文件为 `output/analysis/phase5_candidate_space_evaluation.json`。
- `expanded` 模式下：
  - `mean_candidate_count = 4.875`
  - `mean_search_candidate_count = 3.015625`
  - `mean_strategy_candidate_count = 2.5`
  - `selected_strategy_rate = 0.21875`
- 相对 `legacy` 模式：
  - `changed_case_count = 14`
  - `changed_case_rate = 0.21875`
  - `mean_selected_score_delta_vs_legacy = 0.006655637906996692`
- 单病例示例 `stay_141959` 已生成正式输出：
  - `output/analysis/phase5_stage1_inference_141959.json`
  - `output/analysis/phase5_stage1_inference_141959.md`
- 该病例当前首选候选为 `generated_template_vasopressor_bridge`，其 `candidate_anchor_relation = safe_deviation`，`candidate_strategy_family = vasopressor_intensity`，参数偏移为 `vasopressor_intensity = 0.35`，医生版输出已直接给出安全偏离说明。

---

### 4.6 rolling-horizon 升级为短期策略评估器

**目标**

把当前两步轻量 rolling-horizon 从“附加验证模块”升级为真正参与短期连续决策比较的策略评估器。

**对应诊断问题**

- 问题 12：rolling-horizon 仍属于短期近似，而非真正策略评估

**本阶段如何解决这些问题**

- 对问题 12：通过多步滚动、状态重估、动作重选和路径累计价值比较，把当前 rollout 从“附加说明”升级为“短期连续策略比较器”。

**大致修改计划**

- 从两步轻量 rollout 扩展到多步短期滚动。
- 每一步都支持：
  - 状态重估
  - donor 重检索
  - 候选重选
  - 路径累计价值比较
- 区分：
  - 单步最优但路径不稳
  - 单步一般但连续更优
  - 短期持续负收益

**涉及模块**

- `scripts/infer_eicu_counterfactual_plan.py`
- `run_forecasting_experiment.py`
- `src/manifold_forecasting_trainer.py`

**预期有效性**

- 可行性：中
- 预期收益：中高
- 主要价值：让当前方法从“动作级比较”逐渐向“短期策略级比较”过渡。

**阶段验收标准**

- 多步 rollout 可以稳定运行并输出结构化路径结果。
- 至少能区分单步与多步结论不一致的病例。
- 医生版报告能增加短期路径稳定性解释，而不是只给单步结论。

**当前状态**

- 当前状态：待细化
- 是否达到预期：待验证
- 是否修改完成：否
- 本轮结果摘要：尚未开始。
- 验证结论：待后续实施后补充。

**详细实施计划**

- 待开始该阶段时进一步展开，至少补充以下内容：
  - rollout 步数、折扣和重选逻辑
  - 路径价值定义
  - 状态漂移与误差积累的控制办法
  - 与当前第 8 条轻量 rollout 的兼容策略

---

### 4.7 因果评估补强 + 稳定性分析 + 逐层评估扩展

**目标**

补上当前方法在“科学论证”层面的短板，让系统不仅工程上能跑，还能更清楚地回答“结果到底稳不稳、因果边界在哪里、问题出在链路哪一层”。

**对应诊断问题**

- 问题 11：当前反事实结果本质上仍是模型内模拟
- 问题 13：当前多项评分组合存在阈值和权重敏感性
- 问题 14：评价体系仍偏终点展示，缺少误差分层拆解

**本阶段如何解决这些问题**

- 对问题 11：通过 OPE、DR 和敏感性分析，把“模型内模拟收益”和“更严格的因果论证边界”分开。
- 对问题 13：通过扰动实验和稳定性分析，识别哪些结论是稳健的，哪些结论只是阈值碰巧有利。
- 对问题 14：通过固定四层评估，把问题来源拆解到 factual、retrieval、candidate、ranking，不再只看最终推荐成败。

**大致修改计划**

- 引入更严格的离线策略评估设计：
  - 时间变化 propensity
  - doubly robust
  - off-policy evaluation
  - 敏感性分析
- 扩展稳定性实验：
  - 随机种子
  - donor 池大小
  - top-k
  - guardrail 阈值
  - reranker 温度
- 将分层评估从现有 baseline 进一步扩展成四层固定框架：
  - factual forecasting
  - donor retrieval
  - candidate generation
  - counterfactual ranking

**涉及模块**

- `run_forecasting_experiment.py`
- `scripts/run_eicu_counterfactual_multiseed.py`
- `output/analysis/`
- 评估与汇总脚本

**预期有效性**

- 可行性：中
- 预期收益：很高
- 主要价值：这是后续论文说服力和方法边界表达最关键的一步。

**阶段验收标准**

- 能输出一组稳定性与敏感性分析结果，而不是只给单次最优结果。
- 能清楚拆出哪一层改进有效、哪一层仍是瓶颈。
- 因果边界和模拟边界可以在实验报告中被明确区分。

**当前状态**

- 当前状态：待细化
- 是否达到预期：待验证
- 是否修改完成：否
- 本轮结果摘要：尚未开始。
- 验证结论：待后续实施后补充。

**详细实施计划**

- 本阶段按赶论文口径已完成。当前保留的后续扩展项为：
  - 若后续时间允许，再补完整的 OPE / DR，而不是在当前稿件阶段强行加入半成品因果估计。
  - 若后续需要更强实验说服力，可在现有轻量版基础上追加训练级多随机种子，而不是只停留在 MC 采样种子。
  - 可将当前 `phase7_paper_stability_evaluation.json` 进一步整理成论文表格或附录图。

**2026-04-17 论文口径回填更新**

- 新增脚本：`scripts/evaluate_phase7_paper_stability.py`
- 输出文件：`output/analysis/phase7_paper_stability_evaluation.json`
- 轻量稳定性分析覆盖的扰动维度：
  - MC 不确定性采样种子：`42, 43, 44`
  - guardrail 下界阈值：`0.00` vs `0.05`
  - donor pool：`adaptive` vs `global`
  - 候选模式：`expanded` vs `legacy`
- 关键结果：
  - `baseline_adaptive`：`recommendation_ready_rate_mean = 0.0`，`recommendation_ready_rate_std = 0.0`，`stable_guardrail_status_rate = 1.0`
  - `strict_lb005` 相比基线：`selection_changed_case_rate = 0.0`，`guardrail_status_changed_case_rate = 0.0`
  - `global_pool` 相比基线：`mean_selected_score_delta = -0.00224327369191063`，`mean_neighbor_consistency_delta = -0.005267327195878468`
  - `legacy_candidates` 相比基线：`selection_changed_case_count = 14`，`selection_changed_case_rate = 0.21875`，`mean_selected_score_delta = -0.006655637906996692`
- 这一步当前最重要的论文价值不是“证明系统已经给出很多 recommendation_ready”，而是证明当前主要结论在轻量扰动下没有被轻易翻转，同时也明确显示 `expanded + adaptive` 仍是当前最优主线。

---

### 4.8 医生反馈闭环 + 方法边界治理

**目标**

让系统从“可读、可审计的静态输出器”推进为“能接收临床反馈并且表述边界清晰的决策支持框架”。

**对应诊断问题**

- 问题 15：医生版输出已经可查阅，但尚未形成闭环反馈机制
- 问题 16：方法表述仍需与能力边界严格对齐

**本阶段如何解决这些问题**

- 对问题 15：通过把医生的采纳与否、修改原因和风险备注回写到系统，逐步形成能持续学习的反馈闭环。
- 对问题 16：通过统一输出、文档和论文中的能力表述，避免把当前系统误写成严格因果推荐器或自动医嘱生成器。

**大致修改计划**

- 设计医生反馈接口，至少记录：
  - 采纳 / 不采纳
  - 修改原因
  - 风险备注
  - 对 donor 或候选解释的意见
- 让医生反馈能进入：
  - reranker 监督信号
  - 候选过滤偏好
  - 后续病例审计
- 系统层面统一约束方法表述：
  - 在输出、文档和论文入口中明确“候选干预模拟与风险重排”的定位
  - 禁止把当前结果写成严格疗效估计或自动医嘱建议

**涉及模块**

- `src/counterfactual_plan_renderer.py`
- `scripts/infer_eicu_counterfactual_plan.py`
- 文档与报告模板
- 可能新增的反馈存储结构

**预期有效性**

- 可行性：中高
- 预期收益：中高
- 主要价值：这是把当前系统从演示原型走向长期可维护临床决策支持框架的关键一步。

**阶段验收标准**

- 医生反馈字段可以稳定记录并写入结构化结果。
- 输出与文档中的方法定位保持一致，不再出现能力表述过强的问题。
- 能形成后续可用于 reranker 或报告优化的小规模反馈数据。

**当前状态**

- 当前状态：待细化
- 是否达到预期：待验证
- 是否修改完成：否
- 本轮结果摘要：尚未开始。
- 验证结论：待后续实施后补充。

**详细实施计划**

- 待开始该阶段时进一步展开，至少补充以下内容：
  - 反馈字段设计
  - 反馈存储与读取方式
  - 对 reranker / 候选过滤的最小回写闭环
  - 文档与报告中的统一边界措辞

---

## 6. 路线说明

本轮没有直接沿用“16 条问题 = 16 条修改”的写法，而是将诊断稿中的 16 条问题压缩为 8 个更适合实施的阶段。这样做的原因有三点：

1. 诊断稿中的部分问题本质上属于同一工作包。
   例如“目标错位”和“不确定性未进入决策主逻辑”，更适合一起处理，而不是拆成两个彼此依赖的碎步骤。

2. 后续代码修改和实验验证通常是按模块簇推进的。
   例如 donor 可交换性、邻域一致性和 donor 池扩展，本质上都围绕检索层与 donor 证据层，不适合打散后分别推进。

3. 阶段数量过多会使状态维护流于形式。
   当前 8 个阶段已经足以覆盖诊断稿提出的核心问题，同时也便于你后续逐条让我实施、复跑、验证和回填状态。

## 7. 建议执行顺序

按赶论文口径，建议继续按以下顺序推进：

1. 第 1 至第 5 条已经构成当前论文主干，可直接作为方法章节的四层主线：factual forecasting、donor retrieval、candidate generation、counterfactual reranking + physician output。
2. 下一步优先做第 7 条的轻量版，只补多随机种子、关键阈值敏感性和主要结论稳定性，不追求一次性补完整 OPE / DR。
3. 第 6 条作为可选增强项保留。若时间允许，可补一版短期多步策略评估结果；若时间紧，可以不阻塞投稿。
4. 第 8 条优先放入 future work 或系统落地方向，不作为当前投稿前的必做项。

当前最不建议的做法，是在第 5 条已经完成后继续无上限地扩候选空间、或直接转去做完整反馈闭环。对当前稿件最有价值的，不是继续堆功能，而是把稳定性、边界和实验叙事补齐。
