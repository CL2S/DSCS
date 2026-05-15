# 背景

上一轮自动迭代后，transition memory 的最佳相对 MAE 提升约为 `4.01%`，没有达到目标的 `10%`。从结果看，真正有效的部分仍然是 factual transition path，donor path 对最终 forecasting 指标贡献很弱。

进一步分析后，本轮判断最值得优先修改的点有两个：

1. `transition_utility` 定义过粗，主要是单一“当前值到未来均值”的变化，和实际预测目标对齐不足。
2. `action` 表征过于粗糙，只有少量 flag 和简单强度，难以让记忆库区分“干预细节不同但结果差很多”的样本。

# 本轮修改目标

本轮目标不是直接重写为完整强化学习系统，而是在当前架构内做最小但更有判别力的升级：

1. 把 transition utility 改成多分量 response utility。
2. 把 intervention action 向量改成更细粒度的时机、强度、趋势表征。
3. 重新进行 20 轮自动搜索，验证这些改动是否能超过上一轮的最优结果。
4. 如果后续尝试出现退化，回退到本轮实测最优状态，不保留负收益改动。

# 修改内容

本轮保留的代码修改如下。

## 1. transition utility 改成多分量曲线 utility

文件：
- `memory_mvp_project/src/manifold_forecasting_trainer.py`

新增逻辑：

- 新增 `utility_vector` 到 `TransitionStoreEntry`
- 新增 `_transition_utility_vector_from_sample`
- 新增 `_transition_query_preference_vector`
- 新增 `transition_utility_vector_cache`
- 新增 `transition_label_utility_vector_mean`

新的 utility 不再只看未来均值，而是拆成以下 6 个分量：

1. 当前值到第一步未来值的改善
2. 当前值到未来均值的改善
3. 当前值到未来最差值的改善
4. 当前值到未来末端值的改善
5. 未来序列整体下降趋势
6. 未来波动惩罚

然后根据当前样本的临床状态生成 query preference，对不同状态下的 utility 分量做加权汇总。这样做的目的是让 septic shock、hypotension、organ dysfunction 等状态在 transition memory 中关注不同类型的改善，而不是所有病人都只看一个统一分数。

## 2. action 表征改成更细粒度 profile

文件：
- `memory_mvp_project/src/manifold_forecasting_trainer.py`

新增逻辑：

- 新增 `_intervention_action_profile`
- 重写 `_intervention_action_vector`

新的 action 向量不再只包含少数二值 flag，而是加入了：

1. 抗菌药时机 `antibiotic_timeliness`
2. 抗菌药疗程标志 `antibiotic_duration_flag`
3. 升压药最近强度、平均强度、最大强度、趋势
4. 呼吸支持最近强度、平均强度、最大强度、趋势
5. 血培养、乳酸检查、MAP 监测、乳酸复测等检查/监测 flag
6. intervention static / sequence 的总体强度

## 3. action_change_score 加入分项 gain

文件：
- `memory_mvp_project/src/manifold_forecasting_trainer.py`

`_action_change_score` 不再只依赖 guideline compatibility 和 missing care 变化，还加入了：

1. antimicrobial gain
2. vasopressor gain
3. respiratory gain
4. monitoring gain

并按 sample 的 sepsis / shock / hypotension / high lactate 状态做加权。这样 donor rerank 至少能显式感知“候选方案相对当前方案改进了哪些关键干预”。

## 4. 自动搜索脚本收缩到更有希望的配置空间

文件：
- `memory_mvp_project/tools/auto_transition_iteration.ps1`

脚本搜索空间从上一轮的较分散配置，收缩到更有希望的区域，重点覆盖：

1. `top_k=4/5`
2. `blend_weight=0.12/0.14`
3. `factual_only` 与 `both`
4. 小范围 state/action 权重变化
5. action change on/off 对照

## 5. 已尝试但未保留的探索

本轮后半段还尝试了两项更激进的改动：

1. 把 transition retrieval 从 label-restricted 改成全库检索加标签偏置
2. 把 factual residual gate 从单一 `confidence` 改成 `confidence + residual_scale + utility` 组合门控

这两项在定点验证中都没有带来更好结果，因此已经回退，没有保留在最终代码状态里。

# 影响范围

涉及文件：

- `memory_mvp_project/src/manifold_forecasting_trainer.py`
- `memory_mvp_project/tools/auto_transition_iteration.ps1`

本轮没有改数据集构建逻辑，没有改知识图谱生成逻辑，也没有改 persistent memory store 的持久化结构。

# 运行方式

静态检查：

```powershell
python -m py_compile memory_mvp_project\src\manifold_forecasting_trainer.py
python -m py_compile memory_mvp_project\run_forecasting_experiment.py
```

20 轮自动搜索：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\memory_mvp_project\tools\auto_transition_iteration.ps1 -MaxRounds 20 -TargetImprovementRatio 0.10
```

结果汇总文件：

- `memory_mvp_project/output/auto_iteration/summary.json`

# 结果与分析

本轮 20 轮自动搜索后，最优结果为：

- 最优轮次：`Round 11`
- 配置：`both_topk_5_blend_012`
- `hybrid_mae = 1.4299798161210615`
- `hybrid_rmse = 2.0555464605460565`
- `improvement_mae = 0.063432048190155355`
- `improvement_ratio_mae = 0.042474584343422987`

即：

- 本轮最佳相对 MAE 提升约为 `4.25%`
- 相比上一轮最优 `4.01%` 有小幅提升
- 仍未达到目标的 `10%`

对结果的判断如下：

1. 多分量 utility 和更细 action profile 是有效的，但增益有限。
2. 最优点从 `top_k=4` 移到 `top_k=5`，说明 richer utility/action 后，检索到略更多模板反而更稳。
3. `both` 和 `factual_only` 的结果几乎一致，说明 donor path 仍然不是主增益来源。
4. donor 侧虽然 `donor_action_change_score_mean` 更稳定，但 `transition_action_gain` 依然偏弱，说明 action-delta 信号有了，但还不够强到改写最终预测指标。
5. 后续两次激进尝试之所以被回退，是因为它们没有继续提高最优点，反而在定点验证中带来了退化。

# 已知问题

当前没有达到 `10%` 的根本原因，已经比较明确：

1. transition memory 仍主要在做 template-level residual 修正，而不是进入预测主干。
2. donor path 的 transition score 仍然不够强，尚未形成可稳定拉动 forecasting 指标的第二增益源。
3. 当前 utility 虽然比单一均值变化更好，但仍然是代理目标，不是真正的多目标临床 outcome utility。
4. quick setting 下样本规模较小，容易把方法增益压缩在 `3%~5%` 范围内。
5. 当前 action profile 虽然细了一层，但还没有细到 action subtype policy 层面，例如抗菌药类别差异、升压药剂量轨迹模式等还没有展开。

# 下一步计划

下一轮如果继续冲击 `10%`，不建议再主要靠调权重，应该转向更结构性的修改：

1. 把 transition utility 从当前 6 分量代理，继续扩成 hemodynamic / infection-control / respiratory / monitoring 四类 utility。
2. 让 factual path 使用 query-conditioned template selection，而不只是 fixed top-k softmax。
3. 把 donor 路径里的 action change 再拆成更明确的 subtype gains，而不是继续只调总权重。
4. 在 `eicu-max-series=256` 上复跑当前最优配置，先确认 `4%+` 增益是否稳定。
5. 如果大样本仍然停留在 `5%` 以下，就要承认当前“后融合式经验记忆”已接近上限，需要改成更早期融合或显式状态转移预测主干。
