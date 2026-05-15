# 背景

在上一轮迭代中，transition memory 在 quick setting 下的最优相对 MAE 提升约为 `4.25%`，但存在明显平台期。进一步分析后，判断最值得优先处理的主瓶颈有两项：

1. transition memory 进入预测流程过晚，主要以 residual 形式在末端修正。
2. transition retrieval 仍偏向相似模板平均，容易稀释高价值经验。

因此本轮决定优先尝试：

1. 让 transition memory 更早进入 forecasting 主干。
2. 在 retrieval 阶段加入 utility-aware reranking 和 anchor-style template 选择。
3. 在更大样本上验证这些结构性修改是否真的有效，而不是继续只在 quick setting 调权重。

# 本轮修改目标

本轮目标是验证下面这条判断是否成立：

“如果把 transition memory 提前接入主干，并在检索时减少平均化稀释，那么在更大样本上，memory 增益应该优于当前后融合 baseline。”

验证方式不是只跑一个配置，而是做大样本对照：

1. baseline：保守配置，不启用新的 trunk 注入，也不启用新的 utility-aware / anchor 强化。
2. modified：启用主干注入和强化检索。
3. tuned：启用较弱版本的强化项，检查是否只是默认强度过大。

# 修改内容

## 1. 新增主干级 transition context 注入能力

文件：
- `memory_mvp_project/src/manifold_forecasting_trainer.py`

新增模块：

- `transition_context_projector`
- `transition_trunk_gate`
- `_transition_context_tensor`
- `_transition_trunk_adjustment`

新的逻辑是在 `transition_readout` 生成后，将：

1. `template_curve`
2. `residual_curve`
3. confidence / utility / advantage / improvement_rate / support_strength / transition_score

编码成一个 transition context，然后投影到 fusion hidden space，并在 bucket residual 之前注入到 fused representation。

这一步的目的，是让 transition memory 更早影响 residual branch，而不只是最后再加一条 transition residual。

## 2. 新增 utility-aware retrieval 和 anchor-style template

文件：
- `memory_mvp_project/src/manifold_forecasting_trainer.py`

在 `_retrieve_transition_readout` 中新增：

1. `transition_utility_alignment_weight`
2. `transition_anchor_blend_weight`

修改点：

- 在 state/action similarity 之外，加上 utility alignment 参与排序。
- 不再只用 softmax 平均的 template curve，而是支持“weighted template + top-1 anchor”的混合。

目的是减轻纯平均模板把高价值经验冲淡的问题。

## 3. 运行入口新增实验参数

文件：
- `memory_mvp_project/run_forecasting_experiment.py`

新增参数：

- `--transition-utility-alignment-weight`
- `--transition-anchor-blend-weight`
- `--transition-trunk-weight`
- `--disable-transition-trunk-path`

这些参数用于控制新结构的强度，便于大样本对照。

## 4. 根据大样本结果回退默认配置

虽然代码保留了上述新能力，但根据本轮实测结果，默认配置已经回退到安全值：

- `transition_utility_alignment_weight = 0.0`
- `transition_anchor_blend_weight = 0.0`
- `transition_trunk_weight = 0.0`
- `enable_transition_trunk_path = False`

原因是这些新结构在 `256` 样本的大样本验证中没有打赢 baseline，不能作为默认行为保留。

# 影响范围

本轮影响的文件：

- `memory_mvp_project/src/manifold_forecasting_trainer.py`
- `memory_mvp_project/run_forecasting_experiment.py`

未修改：

- 数据集构建逻辑
- KG 特征构建逻辑
- persistent memory store 持久化结构

# 运行方式

静态检查：

```powershell
python -m py_compile memory_mvp_project\src\manifold_forecasting_trainer.py
python -m py_compile memory_mvp_project\run_forecasting_experiment.py
```

大样本 baseline：

```powershell
python memory_mvp_project\run_forecasting_experiment.py `
  --dataset-format eicu_sepsis3 `
  --enable-kg `
  --eicu-max-series 256 `
  --epochs 4 `
  --batch-size 16 `
  --max-train-windows-per-series 16 `
  --counterfactual-donor-score-mode structured `
  --counterfactual-candidate-policy generated_best `
  --enable-transition-memory `
  --transition-top-k 5 `
  --transition-template-blend-weight 0.12 `
  --transition-utility-alignment-weight 0.0 `
  --transition-anchor-blend-weight 0.0 `
  --disable-transition-trunk-path `
  --output-json memory_mvp_project\output\large_transition_baseline_256.json
```

大样本 modified：

```powershell
python memory_mvp_project\run_forecasting_experiment.py `
  --dataset-format eicu_sepsis3 `
  --enable-kg `
  --eicu-max-series 256 `
  --epochs 4 `
  --batch-size 16 `
  --max-train-windows-per-series 16 `
  --counterfactual-donor-score-mode structured `
  --counterfactual-candidate-policy generated_best `
  --enable-transition-memory `
  --transition-top-k 5 `
  --transition-template-blend-weight 0.12 `
  --transition-utility-alignment-weight 0.12 `
  --transition-anchor-blend-weight 0.35 `
  --transition-trunk-weight 0.18 `
  --output-json memory_mvp_project\output\large_transition_modified_256.json
```

两组保守 tuned 配置也已运行：

- `memory_mvp_project/output/large_transition_modified_256_tuned1.json`
- `memory_mvp_project/output/large_transition_modified_256_tuned2.json`

# 结果与分析

## 1. baseline 结果

文件：
- `memory_mvp_project/output/large_transition_baseline_256.json`

关键结果：

- `hybrid_mae = 1.3717488425879554`
- `hybrid_rmse = 2.004640009477895`
- `improvement_mae = +0.024608453387285456`
- `improvement_rmse = -0.018175545776302426`

即：

- 在 `256` 样本上，保守 baseline 仍然有轻微 MAE 正增益
- 但 RMSE 已经开始承压

## 2. modified 结果

文件：
- `memory_mvp_project/output/large_transition_modified_256.json`

关键结果：

- `hybrid_mae = 1.5283373988230546`
- `hybrid_rmse = 2.1747153250813964`
- `improvement_mae = -0.1319801028478138`
- `improvement_rmse = -0.18825086137980374`

结论：

- 当前实现形式下，强度较大的“主干注入 + 强化检索”在大样本上明显退化
- 不是轻微波动，而是确定性的负收益

## 3. tuned 结果

### tuned1

文件：
- `memory_mvp_project/output/large_transition_modified_256_tuned1.json`

关键结果：

- `hybrid_mae = 1.387274902458057`
- `improvement_mae = +0.009082393517183762`

### tuned2

文件：
- `memory_mvp_project/output/large_transition_modified_256_tuned2.json`

关键结果：

- `hybrid_mae = 1.407736194009171`
- `improvement_mae = -0.011378898033930218`

结论：

- 即使大幅降低强度，新结构也没有超过 baseline
- tuned1 仍明显弱于 baseline 的 `+0.0246`
- tuned2 直接转为负增益

## 4. 综合判断

本轮最重要的结论不是“方向完全错误”，而是：

1. “提前主干融合”这个想法本身不能直接按当前实现落地。
2. 当前 transition readout 的质量，还不足以支撑它被更早、更强地注入主干。
3. 在 quick setting 下看起来合理的强化检索，在 `256` 样本上会稳定放大噪声。
4. 因此，当前这些新能力只能保留为实验开关，不能作为默认配置。

# 已知问题

从这轮大样本结果看，当前主瓶颈进一步被验证为：

1. transition readout 还不够纯，不能承受更强的主干注入。
2. utility-aware rerank 在当前实现下会放大启发式偏差，而不是真正提升可泛化预测。
3. anchor-style template 仍然受单条样本噪声影响，当前数据质量不足以支撑这种更尖锐的模板偏好。
4. donor path 仍然不是 forecasting 的有效第二增益源，这一点在更大样本下也没有改变。

# 下一步计划

基于本轮结果，下一步不建议继续在“更早主干融合”这条线上直接加大强度。更合理的路线是：

1. 先提升 transition readout 质量，再谈更早融合。
2. 优先改 retrieval purity，而不是直接改注入深度。
3. 把 transition selection 改成更严格的候选过滤，再做小范围主干注入验证。
4. 如果后续还做大样本实验，默认应从当前保守 baseline 出发，而不是从激进主干版本出发。
