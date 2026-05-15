# FRCDG 与 Memory MVP 面向 eICU 动态数据的融合与修改方案

## 1. 目标重定义

本次融合不再把任务定义为“在静态表格样本上修改若干特征，使当前分类标签翻转”。

新的目标定义为：

- 输入一个 ICU 动态患者窗口样本。
- 基于历史相似患者检索 donor。
- 在 donor 原始干预、KG 修补干预和小范围安全搜索干预之间生成候选方案。
- 对候选方案作用于当前患者后的短期结局进行反事实预测。
- 输出带有风险门控、可解释证据与候选对比的结果。

适用数据集：

- `eICU`
- `MIMIC-III/MIMIC-IV`
- 其他具有患者动态轨迹、静态特征、干预轨迹和短期预测目标的 ICU 时序数据集

因此，本次融合的主方向不是“把 FRCDG 直接搬到 eICU 上跑”，而是“保留 FRCDG 中可解释的特征编辑思想，把它改造成适合动态时序反事实任务的候选生成组件”。

## 2. 当前两套方法的角色定位

### 2.1 FRCDG 的核心价值

`FRCDG` 当前方法主线是：

1. 用 VAE 梯度得到全局特征重要性排序。
2. 用表格 Transformer 集成模型做分类预测。
3. 按特征排序做逐步贪心修改，直到预测概率翻转。

这套方法的优点是：

- 解释清楚。
- 容易生成“改了哪些特征”的对照结果。
- 对静态结构化表格任务实现简单。

但它的问题也很明确：

- 建模对象是静态样本，不是动态病程。
- 输出目标是当前标签翻转，不是未来短期结局改善。
- 没有 donor 检索。
- 没有干预方案层的表示。
- 没有不确定性约束。
- 没有滚动 horizon 评估。
- 医学规则只限制“能不能改”，不限制“改法是否临床合理”。

### 2.2 memory_mvp_project 的核心价值

`memory_mvp_project` 当前方法主线是：

1. 对患者动态窗口做时序表征。
2. 通过 memory bank / donor retrieval 找相似且可比的历史患者。
3. 基于 donor 原方案、KG 修补方案和安全搜索方案构造候选。
4. 对候选方案做短期 proxy 反事实预测。
5. 用不确定性、门控规则和 physician report 做最终输出。

这套方法适合动态 ICU 数据，但当前不足在于：

- 对“逐特征差异解释”的表达不如 FRCDG 直接。
- 候选方案解释更偏 plan-level，而不是 feature-level。
- 如果论文想保留“counterfactual feature revision”的视觉表现，FRCDG 的表达方式仍然有利用价值。

## 3. 融合原则

### 3.1 主体框架以 dynamic forecasting / donor-based counterfactual 为主

融合后的主体框架必须以 `memory_mvp_project` 为主，而不是以 `FRCDG` 为主。

原因：

- 目标数据集已经明确是 `eICU` 一类动态数据。
- 目标任务是短期结局反事实评估，不是当前静态分类翻转。
- donor 检索、候选生成、门控与滚动评估已经在 `memory_mvp_project` 中形成闭环。

### 3.2 FRCDG 作为“候选生成解释模块”嵌入

FRCDG 不应再作为顶层 pipeline，而应转化为融合系统中的一个子模块：

- 作为 feature ranking prior
- 作为 donor-to-query 差异定位工具
- 作为稀疏候选生成器
- 作为论文中的局部可解释反事实编辑模块

## 4. 方法层面的关键调整

### 4.1 从静态特征转为动态窗口表示

必须先把 FRCDG 的输入对象从：

- 单条静态表格记录

改为：

- 患者时间窗 `sequence`
- 患者静态信息 `patient_static`
- 当前干预 `intervention_static`
- 干预序列 `intervention_sequence`
- KG 派生特征 `kg_features`

调整结论：

- FRCDG 原始的 `feature_extract.py` 和 `transformer_predict.py` 不能直接作为顶层模型继续沿用。
- 若保留 FRCDG 的特征编辑思想，编辑对象只能定义在“窗口级摘要特征”或“干预相关特征”上，而不能继续无差别编辑全部输入字段。

### 4.2 重新定义反事实对象

在动态 ICU 任务中，反事实对象必须从“状态特征”与“干预特征”中分离。

建议拆分为三类：

1. 不可改变量
   - 年龄
   - 性别
   - 基础共病负担
   - 历史已发生事实

2. 状态变量
   - 当前生命体征/实验室窗口摘要
   - SOFA、乳酸、血流动力学状态

3. 干预变量
   - 抗菌药时机
   - 升压支持强度
   - 呼吸支持强度
   - 监测与复查行为

融合后只能对第 3 类和少量可操作的状态派生摘要做候选调整。

不能继续像 FRCDG 静态版那样直接修改任意输入列，否则方法学上会变成“篡改患者状态”，而不是“模拟干预变化”。

### 4.3 输出目标从 label flip 改成未来短期结局改善

融合后目标从：

- `P(y=1)` 降到阈值以下

改成：

- 未来 `SOFA` 轨迹改善
- 未来乳酸变化改善
- 后续升压药需求风险下降
- 后续呼吸支持升级风险下降
- 综合 proxy 改善的下界大于 0

因此候选评价指标不再是单一分类概率，而是多目标组合分数。

## 5. 推荐融合架构

推荐采用如下六层结构：

1. `Dynamic Patient Encoder`
   - 直接沿用 `memory_mvp_project` 的时序编码与 manifold memory 主干。

2. `Donor Retrieval Layer`
   - 沿用现有 donor ranking、hard filter、overlap filter、neighbor consistency。

3. `FR-guided Difference Localization`
   - 新增一个“FRCDG 风格差异定位模块”。
   - 对 query 与 top donor 做差异分析，生成 donor-query 差异排序。

4. `Candidate Generation Layer`
   - 保留三类候选：
   - `donor_original`
   - `generated_kg_repaired`
   - `generated_sparse_edit`

5. `Counterfactual Prediction and Reranking`
   - 沿用多目标评分、不确定性惩罚、下界惩罚与 neighborhood bonus。

6. `Rollout and Physician Guardrail`
   - 沿用 rollout 和 physician-facing report。

这里的新增点是第 3 层和第 4 层中的 `generated_sparse_edit`。

## 6. 新增的 FR-guided 稀疏候选生成模块

### 6.1 模块目标

生成一个新候选来源：

- `generated_sparse_edit`

它不是直接改原始静态特征，而是：

- 围绕 donor 原方案或 KG 修补方案
- 在安全边界内
- 对少量干预参数或窗口级可操作摘要做稀疏修改

### 6.2 候选生成方式

建议这样实现：

1. 先取 top-1 或 top-k donor。
2. 以 donor 原始方案或 `generated_kg_repaired` 方案作为 anchor。
3. 用 FR 风格排序器对以下对象排序：
   - 干预静态特征差异
   - 干预序列摘要差异
   - 临床可操作状态摘要差异
4. 只对前 `N=1~3` 个可操作维度做小步编辑。
5. 每次编辑都通过 hard filter 与 KG feasibility 校验。
6. 将编辑后方案送入反事实预测模型打分。

### 6.3 排序信号构成

排序信号不再使用纯 VAE 全局梯度，而应改成混合信号：

- 全局先验重要性
- 当前患者相对 donor 的差异幅度
- KG 相关性权重
- 当前任务目标相关性权重

建议公式：

`priority = a * global_importance + b * abs(query - donor) + c * kg_relevance + d * target_sensitivity`

其中：

- `global_importance` 可以保留 FRCDG 风格
- `abs(query - donor)` 用于个体化
- `kg_relevance` 强制与 sepsis/shock/lactate 关键位点对齐
- `target_sensitivity` 由当前 counterfactual 预测器估计

## 7. 候选来源设计

融合后候选来源建议固定为四类：

1. `donor_original`
   - donor 原始干预

2. `generated_kg_repaired`
   - donor 原方案 + KG 修补

3. `generated_sparse_edit`
   - donor/KG 方案基础上的少量稀疏编辑

4. `generated_template_search`
   - 现有 safe search 模板搜索候选

推荐顺序：

- 先保留前两类为稳定基线
- 再引入第 3 类作为论文创新点
- 第 4 类作为候选空间扩展

## 8. 代码级修改路线

### 8.1 第一阶段：建立融合目录与方法文档

本阶段已完成：

- 新建 `memory_mvp_project/docs/04_frcdg_fusion/`
- 生成本方案文档

### 8.2 第二阶段：抽取 FRCDG 中真正可复用的部分

建议只保留以下思想，不直接复用整个旧代码：

- 全局特征排序思想
- 稀疏逐步编辑思想
- 特征变化可视化形式

不建议直接复用：

- 静态 VAE + 静态表格输入接口
- 静态 Transformer 分类器主干
- “概率 < 0.45 即成功”的停止规则

### 8.3 第三阶段：在 `memory_mvp_project` 中新增融合模块

建议新增文件：

- `memory_mvp_project/src/frcdg_dynamic_adapter.py`
  - 负责把 FRCDG 思路改造成动态任务可用的排序与稀疏编辑模块

- `memory_mvp_project/src/frcdg_candidate_generator.py`
  - 负责生成 `generated_sparse_edit` 候选

- `memory_mvp_project/scripts/evaluate_frcdg_fusion_candidates.py`
  - 负责对融合候选做单独评估

### 8.4 第四阶段：在现有 counterfactual 主链中接入新候选

主要改动点应放在：

- 候选生成函数附近
- 候选重排函数附近
- physician report 渲染函数附近

目标是让 `generated_sparse_edit` 作为合法候选来源进入：

- 候选池
- selection score
- report 输出

## 9. 实验设计建议

实验不能再沿用 FRCDG 的静态表格指标，必须改成动态时序任务指标。

建议至少做四组对照：

1. `Baseline donor-only`
   - 只有 `donor_original`

2. `donor + KG repair`
   - `donor_original + generated_kg_repaired`

3. `donor + KG repair + sparse edit`
   - 新融合版本

4. `donor + KG repair + sparse edit + safe search`
   - 完整版本

建议报告指标：

- `predicted_improvement_rate`
- `mean_predicted_delta`
- `delta_lower_bound > 0` 的比例
- `generated_candidate_selected_rate`
- `search_candidate_selected_rate`
- `review_only` 比例
- top donor 相似度/可比性统计
- physician guardrail 通过率

## 10. 论文写法建议

论文中应明确说明：

- FRCDG 原始方法针对静态表格反事实生成。
- 本研究面向动态 ICU 数据，不能直接使用静态任务定义。
- 因此我们将其改造为“动态 donor-guided sparse intervention editing”模块。

建议把创新点写成：

1. 将特征排序式反事实思想从静态表格迁移到动态 ICU 反事实候选生成。
2. 将 donor retrieval、KG repair 与 sparse edit 统一到同一候选空间。
3. 通过多目标重排、不确定性下界与 rolling horizon 提高推荐可靠性。

## 11. 风险与注意事项

最需要避免的错误有三类：

1. 继续把动态 ICU 任务写成“静态标签翻转”
   - 这会让 FRCDG 与 memory 框架的方法目标不一致。

2. 不区分状态变量和干预变量
   - 会导致反事实编辑失去临床含义。

3. 把 FRCDG 整个旧分类器主干硬接入动态主系统
   - 会产生两个任务定义冲突的预测头，方法会变得不自洽。

## 12. 推荐实施顺序

推荐严格按以下顺序推进：

1. 明确动态任务定义与特征分层。
2. 新增 FR-guided 差异排序与 sparse edit 模块。
3. 把 `generated_sparse_edit` 接入现有候选池。
4. 用现有多目标评分与 guardrail 做统一重排。
5. 增加针对 eICU 的消融实验与论文写法整理。

## 13. 当前结论

当前最合理的融合方向不是：

- “把 FRCDG 静态反事实生成直接迁到 eICU”

而是：

- “以 memory_mvp_project 为动态主框架”
- “把 FRCDG 改造成 donor-guided、KG-aware、面向动态 ICU 任务的稀疏候选生成模块”

这是方法上最稳、工程上最可落地、论文上也最容易讲清楚的方案。
