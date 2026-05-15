# 背景

本轮工作围绕“记忆增强状态转移框架”的自动化迭代展开，目标是在不改变数据集切分与核心实验入口的前提下，持续执行“修改代码 -> 运行验证 -> 分析问题 -> 提出下一步修改 -> 再次修改代码”的闭环。

为保证结果可比较，本轮统一使用以下 quick setting 作为评估环境：

- 数据集：`eicu_sepsis3`
- `eicu-max-series=128`
- `epochs=4`
- `batch-size=16`
- `max-train-windows-per-series=16`
- `counterfactual-donor-score-mode=structured`
- `counterfactual-candidate-policy=generated_best`

提升判据定义为：

- 以同轮 `base_only.mae` 为基线
- 以 `memory_effectiveness.improvement_mae / base_only.mae` 作为相对提升率
- 目标阈值为 `10%`
- 最多执行 `20` 轮


# 本轮修改目标

1. 将原本人工试错的 transition-memory 修改流程改成可自动执行的多轮迭代。
2. 解决当前方法里已经暴露出的结构性问题，尤其是：
   - counterfactual candidate 评估时的状态泄漏
   - donor 路径缺少显式 action-delta 信号
   - factual transition residual 缺少系统化搜索入口
3. 在 20 轮内尽量逼近或达到 `10%` 的 MAE 相对提升目标。
4. 输出完整的过程记录，沉淀哪些修改有效、哪些无效、问题卡在哪里。


# 修改内容

本轮代码修改分为三部分。

第一部分是修正 plan 评估逻辑。

- 在 `manifold_forecasting_trainer.py` 中新增 `_build_plan_evaluation_flags`
- 该逻辑在评估 donor plan 或 generated candidate 时，保留当前样本的状态标志，只从 donor/candidate 继承与干预计划相关的 action/exam/monitor 标志
- 解决了“使用 donor 的状态去评估当前病人方案”的状态泄漏问题

第二部分是补强 donor 侧的 action-delta 信号。

- 新增 `_action_change_score`
- 按照当前病人的 KG 状态，比较“当前方案”与“候选方案”在以下层面的变化：
  - guideline compatibility
  - missing care penalty
  - contraindication penalty
  - 关键 care feature 的新增/移除
- 将 `action_change_score` 接入：
  - donor score
  - candidate selection score
  - counterfactual 汇总统计

第三部分是建立可自动搜索的实验控制与执行脚本。

- 在 `run_forecasting_experiment.py` 中新增参数：
  - `--transition-factual-residual-mode`
  - `--transition-positive-only`
  - `--transition-action-change-weight`
  - `--transition-candidate-action-change-weight`
- 在 `manifold_forecasting_trainer.py` 中新增 `_transition_factual_residual`
- 支持三种 factual residual 形式：
  - `additive`
  - `delta_to_base`
  - `delta_to_fusion_base`
- 新增自动迭代脚本：
  - `memory_mvp_project/tools/auto_transition_iteration.ps1`
- 该脚本会：
  - 最多运行 20 轮候选配置
  - 每轮输出一个 JSON 文件
  - 自动计算相对 MAE 提升率
  - 记录最佳轮次与全部轮次结果到 `summary.json`

另外，本轮中间尝试过“中心化 residual / residual_curve”方案，但验证后发现会显著拉坏 RMSE 和 sMAPE，因此已回退，没有作为最终保留方案。


# 影响范围

本轮影响的文件如下：

- `memory_mvp_project/src/manifold_forecasting_trainer.py`
- `memory_mvp_project/run_forecasting_experiment.py`
- `memory_mvp_project/tools/auto_transition_iteration.ps1`
- `memory_mvp_project/output/auto_iteration/*.json`
- `memory_mvp_project/output/auto_iteration/summary.json`

本轮新增的行为能力如下：

- 能自动执行最多 20 轮 quick search
- 能结构化比较不同 transition heuristic 的效果
- 能自动沉淀最佳轮次与全部轮次的指标


# 运行方式

自动迭代脚本运行命令如下：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\memory_mvp_project\tools\auto_transition_iteration.ps1 -MaxRounds 20 -TargetImprovementRatio 0.10
```

输出目录：

- `memory_mvp_project/output/auto_iteration/`

关键输出文件：

- 每轮结果：`round_XX_*.json`
- 汇总结果：`summary.json`


# 结果与分析

本轮共执行 `20` 轮，未达到 `10%` 的目标提升。

最佳结果出现在第 `13` 轮：

- 轮次名称：`both_topk_4`
- `hybrid_mae = 1.4335836898424608`
- `hybrid_rmse = 2.0539761713476135`
- `improvement_mae = 0.059828174468756323`
- `improvement_ratio_mae = 0.040061402951522641`

换句话说：

- 当前最佳相对 MAE 提升约为 `4.01%`
- 没有达到目标的 `10%`

本轮搜索中最有代表性的几类结果如下：

1. 默认加法型 factual residual 是稳定正增益的。

- 第 1 轮 `both_additive_default`：相对 MAE 提升约 `2.68%`
- 第 2 轮 `factual_only_additive`：结果与第 1 轮几乎一致

说明：

- 当前主增益几乎全部来自 factual transition path
- donor path 对 MAE 的贡献非常有限

2. 提高 `transition_template_blend_weight` 到 `0.12` 有进一步增益。

- 第 10 轮 `both_blend_012`：相对 MAE 提升约 `3.45%`
- 第 12 轮 `factual_only_blend_012`：相对 MAE 提升约 `3.45%`

说明：

- 在当前 quick setting 下，template blend 轻微加大是有益的
- donor on/off 对这组配置几乎没有差别

3. 将 `transition_top_k` 从 `6` 降到 `4` 是本轮最有效的改动。

- 第 13 轮 `both_topk_4`：相对 MAE 提升约 `4.01%`

说明：

- 当前 transition retrieval 在 top-k 过大时会被噪声样本稀释
- 更小的近邻集合更适合这批 quick 数据

4. `positive_only` 明显是负收益。

- 第 3、4 轮均为负提升

说明：

- 单纯依赖 `expected_utility > 0` 来门控 factual residual 过于粗暴
- 会错误压掉本来有帮助的 transition template

5. `delta_to_base` 和 `delta_to_fusion_base` 都没有超过最佳加法型方案。

- 第 5~8 轮均为正增益，但都低于 `topk_4`

说明：

- 把 template 当成“校正项”在逻辑上更稳
- 但在当前实现和当前 quick setting 下，没有超越简单的 additive blending

6. donor 路径仍然偏弱。

从 20 轮结果看：

- donor 开关对 MAE 影响很小
- `donor_action_change_score_mean` 是稳定正值，说明 action-delta 信号不是完全无用
- 但 `donor_transition_action_gain_mean` 依然长期偏负，说明候选 donor plan 相对当前方案并没有形成稳定的真实优势

综合判断如下：

- 已经解决了状态泄漏问题，这属于“必须修”的逻辑问题
- 已经把 donor path 从纯相似度排序推进到了“带 action-delta 的受限策略评估”
- 但当前 donor 路径依然不是主要增益来源
- factual transition path 仍是当前系统最有效的部分


# 已知问题

1. 20 轮自动迭代后，最佳相对 MAE 提升只有 `4.01%`，距离 `10%` 仍有明显差距。

2. 最优轮次虽然提升了 MAE 和 RMSE，但 `sMAPE` 并没有同步改善，说明当前方法对相对误差的控制仍然偏弱。

3. donor 路径当前仍然缺少足够强的“真实动作收益”信号。

- `action_change_score` 只能提供弱辅助
- `transition_action_gain` 仍然偏负

4. 当前 transition utility 仍然过于单一，主要是基于 SOFA 变化的单代理定义，缺乏更细粒度的临床 response utility。

5. 本轮所有搜索都基于 quick setting。

- 结果适合作为方向判断
- 不足以直接代表更大样本、更长训练、更严格验证下的最终结论


# 下一步计划

下一阶段不建议继续在当前 donor 权重上做小修小补，而应转向更有信息量的结构改动。

优先建议如下：

1. 保留当前最优 quick 配置作为阶段性 best setting：

- `transition_factual_residual_mode=additive`
- `transition_top_k=4`
- `transition_template_blend_weight=0.10`
- donor path 保持弱辅助，不作为主增益来源

2. 将 `transition_utility` 从单一 SOFA 代理扩展成组合 utility。

建议至少加入：

- SOFA 变化
- qSOFA 变化
- lactate 相关信号
- shock / hypotension 缓解代理

3. 将 `action_change_score` 拆成分项子分数，而不是单一聚合值。

建议拆为：

- 抗菌药相关 gain
- 升压药相关 gain
- 呼吸支持相关 gain
- 监测/检查相关 gain

4. donor 路径下一步应从“排序修正”升级到“候选集合过滤 + 轻量价值重排”。

也就是：

- 先按 feasibility / guideline / state consistency 过滤
- 再在保留下来的候选上做 transition rerank

5. 在 quick setting 得到稳定候选方案后，再把最优配置扩展到更大样本验证。

建议后续验证顺序：

1. `eicu-max-series=256`
2. 保持最优 quick heuristic
3. 再判断是否值得进入下一轮结构改造
