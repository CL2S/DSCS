# KG 驱动的反事实机制与评估

## 1. donor 是什么

当前系统里的 donor 不是抽象近邻点，而是历史病例中的一个具体窗口样本。它提供的是：

- 一个相似状态下的历史干预模板
- 对应的 `intervention_static`
- 对应的 `intervention_sequence`

## 2. 当前反事实流程

当前反事实路径是：

1. 对当前样本做 factual 编码
2. 通过 memory 读出经验标签，缩小 donor 搜索范围
3. 在 intervention store 中检索 donor
4. 按 embedding、KG、指南规则对 donor 重排
5. 选择一个 donor
6. 用 donor 干预替换当前样本干预
7. 再前向一次，得到 counterfactual prediction

这意味着当前系统本质上仍然是 donor-based counterfactual，而不是显式因果图或治疗规划器。

## 3. 当前有哪些模式

### `legacy`

更偏向：

- 保住 donor 与 query 的整体相似性
- 在此基础上加少量 KG 与规则修正

### `structured`

更偏向：

- 强调状态匹配
- 强调指南一致性
- 更显式地惩罚缺失护理

### `generated_best`

这是本轮新增的候选干预模式：

1. 先选出一个 donor
2. 保留 donor 原始干预
3. 再按 KG 规则生成 `generated_kg_repaired`
4. 在这些候选里选一个最优单一方案

注意：现在仍然是“单 donor + 单方案选择”，不是多 donor 融合。

## 4. 当前主要指标

当前 donor / counterfactual 评估重点看三层。

第一层，donor 是否像：

- `donor_similarity_mean`
- `donor_exact_experience_match_rate`

第二层，donor 是否更临床一致：

- `donor_kg_similarity_mean`
- `donor_guideline_compatibility_mean`
- `donor_state_match_mean`
- `donor_missing_care_penalty_mean`
- `donor_contraindication_penalty_mean`

第三层，替换后效果 proxy 是否更好：

- `mean_predicted_delta`
- `predicted_improvement_rate`

## 5. 本轮 smoke 结果

本轮 64 stay smoke 对照结果显示：

- 开启 hard filter 后，指南一致性明显提高，缺失护理惩罚下降
- 开启 `generated_best` 后，确实有一部分样本选择了 `generated_kg_repaired`
- 在这批 smoke 结果里，`generated_best` 的改善 proxy 略有提升

结论是：

- KG 已经不仅用于“解释”，而是已经参与 donor 筛选与候选干预选择
- 但它目前仍是约束层，不是完整治疗规划器

## 6. 如何看结果

推荐按这个顺序解读：

1. 先看 donor 是否找到且是否仍像 query
2. 再看 donor 是否更符合临床规则
3. 最后看替换后的改善 proxy

如果只看 `predicted_improvement_rate`，容易误判；因为 donor 太不合理时，proxy 再高也不值得信。
