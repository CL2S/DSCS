# eICU 相似患者检索与反事实候选干预论文主稿大纲

## 1. 文档定位

本文件不是修改说明，也不是方法跟踪板，而是面向论文写作的“母稿型大纲”。

它服务三个目标：

1. 把现有两轮大修改统一压缩成一条清晰的论文主线。
2. 把已经具备证据支持的内容与不建议写强的内容分开。
3. 先把可能用得上的内容尽量写全，后续再按投稿目标删减。

这份大纲默认服务一篇“方法型 / 系统型 / 原型验证型”的论文，而不是严格因果推断论文，也不是可直接自动开立医嘱的临床系统论文。

---

## 2. 建议论文定位

### 2.1 推荐定位

最建议的定位是：

- 面向脓毒症 ICU 场景的相似患者检索与反事实候选干预重排框架
- 带有多目标风险建模、不确定性门控和医生可读输出的临床决策支持原型

更具体地说，论文要强调的是：

- 如何把相似患者检索、候选动作生成、反事实短期预测和医生可读 guardrail 放进一个闭环框架
- 如何通过两轮迭代把一个“相似病例抄方案”的原型收口为一个“谨慎、可审计、可解释”的 review-oriented decision support system

### 2.2 不建议的定位

不建议把论文写成下面这些方向：

- 严格因果治疗推荐系统
- 自动治疗决策系统
- 可直接替代医生决策的 ICU 强推荐模型
- 证明真实临床获益的疗效型论文

原因很直接：

- 当前收益主要还是模型内短期 proxy 模拟
- 当前第 7 条完成的是轻量稳定性分析，不是完整 OPE / DR
- 当前 guardrail 的主结论是“很多病例仍应 review_only”，这更支持“谨慎决策支持”，不支持“自动推荐”

### 2.3 论文类型建议

从投稿策略上，更适合以下类型：

- 医学人工智能应用方法论文
- 临床决策支持系统论文
- 医疗时序建模与可解释推荐系统论文
- 医疗 AI 系统原型与 retrospective evaluation 论文

---

## 3. 两轮大修改如何映射到论文

### 3.1 论文里不写“我们做了很多修改”

论文不应该按“第 1 条修改、第 2 条修改”来写。

论文里应该把两轮修改压缩成四层方法主线：

1. factual forecasting and patient representation
2. donor retrieval and donor evidence modeling
3. candidate generation under safe deviation constraints
4. counterfactual reranking, uncertainty-aware guardrail, and physician-facing output

### 3.2 第一轮九阶段修改在论文中的作用

第一轮九阶段修改更像是 MVP 闭环成型阶段。

在论文中，它对应的是：

- 系统从“相似患者检索器”进化为“相似患者 + 候选干预 + 反事实评估 + 医生版报告”的完整框架
- 形成了可展示的端到端输出形态
- 确立了 doctor-facing report 作为最终输出，而不只是 JSON 字段

### 3.3 第二轮下一阶段优化在论文中的作用

第二轮大修改更像是“把系统收口为能发论文的形态”。

在论文中，它对应的是：

- 事实预测底座更稳
- donor 选择更强调可交换性而不是单纯相似性
- 候选动作空间从 donor-centered 修补升级为 safe deviation
- 稳定性分析开始补齐，从“单次结果”推进到“轻量敏感性验证”

### 3.4 建议在论文中使用的叙事

可以把两轮修改归纳成一句话：

> We iteratively evolved an initial similar-patient retrieval prototype into a guarded counterfactual candidate reranking framework that supports physician review rather than direct automated treatment recommendation.

这句话的作用是：

- 交代系统是迭代出来的
- 但不陷入研发流水账
- 同时把“谨慎、医生复核导向”的定位写清楚

---

## 4. 题目候选

下面给出三组题目方向。

### 4.1 稳妥型题目

1. `A Similar-Patient Retrieval and Counterfactual Candidate Reranking Framework for Sepsis Intervention Support in eICU`
2. `Counterfactual Candidate Reranking over Similar ICU Patients for Sepsis Decision Support`
3. `From Similar Patients to Reviewable Intervention Suggestions: A Guarded Counterfactual Framework for Sepsis Care in eICU`

### 4.2 强调临床可解释性的题目

1. `Physician-Review-Oriented Counterfactual Intervention Support via Similar-Patient Retrieval in eICU Sepsis`
2. `Evidence-Grounded Similar-Patient Retrieval and Guarded Counterfactual Planning for Sepsis Decision Support`
3. `Retrieval, Counterfactual Ranking, and Physician-Facing Guardrails for Sepsis Intervention Support in eICU`

### 4.3 强调方法创新的题目

1. `Uncertainty-Aware Counterfactual Candidate Reranking with Safe Deviation Search over Similar ICU Patients`
2. `Safe-Deviation Candidate Generation and Donor-Aware Counterfactual Ranking for Sepsis Intervention Support`
3. `A Multi-Objective Similar-Patient Counterfactual Framework with Donor Exchangeability and Physician Guardrails`

### 4.4 当前最推荐题目

当前最推荐的是：

`A Similar-Patient Retrieval and Counterfactual Candidate Reranking Framework for Sepsis Intervention Support in eICU`

原因：

- 不夸大因果性
- 不夸大自动化程度
- 同时覆盖 retrieval、counterfactual、candidate reranking 三个关键点

---

## 5. 摘要写作骨架

### 5.1 一句话版本

We propose an eICU sepsis decision-support framework that retrieves medically comparable patients, generates donor-centered and safe-deviation intervention candidates, and reranks them with multi-objective counterfactual forecasting, uncertainty penalties, and physician-facing guardrails.

### 5.2 标准摘要结构

#### 背景句

- ICU sepsis management requires timely yet individualized intervention decisions.
- Existing systems often focus either on outcome prediction or similar-patient retrieval, but rarely integrate retrieval, candidate generation, counterfactual evaluation, and clinician-facing evidence presentation into one auditable workflow.

#### 方法句

- We build a framework that first learns patient representations and factual short-term forecasts, then retrieves donor patients under similarity and exchangeability constraints, generates intervention candidates including safe deviations from donor plans, and reranks candidates using multi-objective counterfactual forecasting with uncertainty-aware guardrails.

#### 实验句

- Experiments on eICU Sepsis-3 windows evaluate the framework from four layers: factual forecasting, donor retrieval, candidate generation, and counterfactual ranking.
- We further include candidate-space ablation, donor-pool sensitivity, and lightweight stability analyses over uncertainty seeds and guardrail settings.

#### 结果句

这里可先放当前已有结果骨架：

- factual forecasting improves overall MAE and improves more clearly in high-volatility, regime-shift, and vasopressor-related slices
- expanded candidate search increases candidate diversity and changes final winners in `21.9%` of evaluated cases without increasing current proxy invalid-rate metrics
- donor exchangeability and adaptive donor pools improve selected-score and neighborhood evidence relative to weaker baselines
- lightweight stability analysis shows that the current system is conservative but stable, with `review_only` conclusions robust under the tested perturbations

#### 结论句

- The framework is better viewed as a physician-review-oriented decision-support system than an automated treatment recommender.
- Its main value lies in integrating comparable donor evidence, constrained counterfactual candidate search, and auditable guardrails into one end-to-end pipeline.

### 5.3 可直接扩写的摘要草稿

可以直接作为主摘要初稿使用：

> Sepsis management in the ICU requires timely but individualized intervention decisions, while existing retrospective AI systems often focus on either short-term risk prediction or similar-patient retrieval alone. We present a similar-patient retrieval and counterfactual candidate reranking framework for sepsis intervention support in eICU. Given a new patient window, the framework first performs factual short-term forecasting and patient representation learning, retrieves medically comparable donor patients under donor similarity, exchangeability, and neighborhood evidence constraints, generates donor-centered as well as safe-deviation intervention candidates, and reranks these candidates using multi-objective counterfactual forecasting, uncertainty-aware penalties, and physician-facing guardrails. We evaluate the method from four layers: factual forecasting, donor retrieval, candidate generation, and counterfactual ranking. Relative to earlier baselines, the refined factual module improves overall forecasting error and improves more clearly in high-volatility, regime-shift, and vasopressor-related slices. The expanded candidate space increases average candidate diversity and changes the selected winner in 21.9% of evaluated cases without increasing the current proxy invalid-rate measures. Lightweight stability analysis further shows that the current system is conservative but stable: across tested uncertainty seeds and selected threshold perturbations, review-only conclusions do not flip. These results suggest that the proposed framework is better positioned as an auditable physician-review-oriented decision-support system rather than a fully automated treatment recommender.

---

## 6. 核心贡献写法

建议把贡献压缩成 4 条，不要写太多。

### 6.1 推荐贡献表述

1. 我们提出一个端到端闭环框架，把 factual forecasting、similar-patient donor retrieval、candidate generation、counterfactual reranking、physician-facing guardrail 串联为一个统一系统。
2. 我们把 donor 选择从单纯相似度扩展为 donor similarity、exchangeability、neighborhood consistency 和 donor pool stratification 的联合证据建模。
3. 我们提出 safe-deviation candidate generation，使候选动作不再只是 donor-centered 修补，而是在受约束边界内进行参数化偏离与小规模策略搜索。
4. 我们通过多目标风险、预测不确定性和医生可读输出，把系统收口为 review-oriented clinical decision support，而非直接自动推荐。

### 6.2 不建议写成贡献的内容

- 完整因果推断
- 临床疗效验证
- 自动治疗建议
- 泛化到所有 ICU 场景

---

## 7. 论文结构总览

推荐结构：

1. Introduction
2. Related Work
3. Problem Formulation
4. Method
5. Experimental Setup
6. Results
7. Discussion
8. Limitations and Ethical Considerations
9. Conclusion
10. Appendix

---

## 8. 引言详细大纲

### 8.1 第一段：临床问题

要点：

- 脓毒症 ICU 场景中的干预决策具有强时间敏感性和个体异质性
- 实际临床中，医生会参考“像这个患者的既往病例”
- 但纯经验式相似病例参考存在两个问题：
  - 哪些病例真正可比并不清楚
  - 相似病例的干预方案并不能直接迁移

可写句：

- In sepsis care, treatment decisions such as antimicrobial timing, vasopressor support, and monitoring intensity are both time-sensitive and highly patient-specific.
- Clinicians often reason by analogy to prior patients, yet retrospective similarity alone does not guarantee intervention transferability.

### 8.2 第二段：现有方法不足

要点：

- 纯风险预测模型只告诉你风险高低，不告诉你“可以考虑什么候选方案”
- 纯相似患者检索器只能给出类似病例，不能完成结构化方案比较
- 很多反事实系统不强调 donor 可交换性、不确定性和医生可审阅输出

### 8.3 第三段：本文思路

要点：

- 不是直接自动治疗推荐
- 而是先找 donor，再生候选，再做 counterfactual reranking，再做 guardrail

### 8.4 第四段：贡献总结

用一段话概括第 6 节的四条贡献即可。

### 8.5 引言最后一句

要明确文章定位：

- We frame the proposed system as a physician-review-oriented decision-support framework, rather than a strict causal treatment recommender.

---

## 9. Related Work 大纲

建议分四块。

### 9.1 Sepsis outcome prediction

写法：

- 现有工作大量关注 sepsis mortality / deterioration / SOFA forecasting
- 但多数不直接输出候选干预及其比较

### 9.2 Similar-patient retrieval in healthcare

写法：

- 医疗中已有基于 EHR 或 ICU 时序的类似病例检索
- 但核心问题是“相似”不等于“可迁移”

### 9.3 Counterfactual or off-policy decision support

写法：

- 强调这类方法要么代价重、要么依赖严格行为策略建模
- 你当前工作不主张已经完成严格因果估计，而是把它作为边界和 future direction

### 9.4 Clinician-facing explanation and auditability

写法：

- 很多模型给分数，不给证据
- 你这里的独特点是把 donor evidence、top-3 candidates、guardrail reasons 直接渲染成医生视角文本

---

## 10. Problem Formulation 大纲

这一节要把问题定义说清楚。

### 10.1 输入

给定一个患者窗口：

- 生理时序
- 静态特征
- 当前已实施干预
- KG/临床规则映射信号

### 10.2 输出

系统输出不是单个“推荐动作”，而是：

- donor evidence
- top-ranked candidate plan
- counterfactual short-term forecasts
- uncertainty-aware guardrail decision
- physician-facing report

### 10.3 目标

目标不是证明真实因果疗效，而是：

- 在 retrospective 数据下构造一个更合理、更可审阅的候选比较框架
- 让“像”“可比”“可迁移”“短期更优”四件事分层处理

---

## 11. Method 章节详细大纲

## 11.1 总体框架

建议先给一张系统图。

图中模块：

1. Patient window encoder
2. Factual forecasting branch
3. Retrieval branch
4. Donor scoring and neighborhood evidence
5. Candidate generation
6. Counterfactual reranking
7. Uncertainty and guardrail
8. Physician-facing renderer

### 11.2 Patient Representation and Factual Forecasting

这一节写两轮修改后的第 1 条和第 2 条底座能力。

必须写的点：

- factual / retrieval 分支进一步解耦
- 时间动态与 missingness 行为建模
- factual 预测不仅看主目标，也带不确定性

可以写的技术点：

- local / medium slopes, regime features, dynamic_profile
- factual branch trunk and gate
- factual scale head and calibration-oriented training

不要写得像调参日志。

推荐写法：

- We separate factual forecasting from retrieval-oriented representation learning while retaining a weakly shared bottom encoder.
- We further augment the factual branch with temporal trend, acceleration, observation-coverage, and missingness-behavior summaries.

### 11.3 Donor Retrieval with Exchangeability and Neighborhood Evidence

这一节对应第二轮第 3 条和第 4 条。

必须写的点：

- donor 不只按 embedding similarity 排序
- 加入 guideline compatibility、state match、missing-care penalty、contraindication penalty
- neighborhood consistency、exchangeability、action alignment、hard/overlap pass rates
- donor pool 有 `global / same_hospital / same_unit / adaptive`

推荐写法：

- We do not equate donor similarity with intervention transferability.
- Donor selection is therefore conditioned on both pointwise compatibility and neighborhood-level supporting evidence.

### 11.4 Candidate Generation under Safe Deviation Constraints

这一节是论文很重要的亮点，对应第 5 条。

必须写的点：

- `legacy` vs `expanded`
- donor-centered 候选与 safe-deviation 候选共存
- 参数化模板搜索
- 每个候选都有 `candidate_anchor_relation` 和 `candidate_safety_rationale`

推荐写法：

- Instead of only repairing donor plans, we allow constrained safe deviations within interpretable search templates.
- This expands the action space while preserving auditability and bounded deviations from existing intervention scales.

### 11.5 Multi-objective Counterfactual Candidate Reranking

这一节对应第 2 条。

必须写的点：

- 不只看 `predicted_delta`
- 同时看 future SOFA, lactate, vasopressor need, respiratory escalation
- uncertainty penalty 和 lower-bound 进入排序

推荐写法：

- Candidate ranking is driven by a multi-objective score rather than a single proxy delta.
- Uncertainty is incorporated directly into ranking and not only used as an after-the-fact display attribute.

### 11.6 Guardrail and Physician-Facing Report

这一节要突出你和普通“输出 JSON”方法的差异。

必须写的点：

- `review_only` vs `recommendation_ready`
- guardrail reasons
- 医生视角输出顺序

推荐写法：

- The system is intentionally designed to abstain.
- High donor similarity alone is insufficient; negative or unstable short-term evidence leads to a physician-review-only output.

---

## 12. 方法章节中如何融合两轮修改

### 12.1 不按时间写

论文里不要写：

- 第一轮我们做了 A
- 第二轮我们做了 B

### 12.2 按模块写

论文里应该写成：

- 底层表示与 factual forecasting
- donor retrieval 与 donor evidence
- candidate generation
- counterfactual reranking and guardrail

### 12.3 在消融实验里体现演化

两轮修改真正该出现的位置是 Results/Ablation：

- 先证明第一轮让系统成型
- 再证明第二轮让系统更稳、更谨慎、更可解释

---

## 13. Experimental Setup 大纲

### 13.1 数据集与队列定义

需要写清楚：

- eICU Sepsis-3 cohort
- 使用窗口化样本
- history length = 4
- forecast horizon = 2
- 当前任务更偏短期 deterioration / intervention support

### 13.2 训练 / 验证 / 测试

目前已有数字可写：

- 在第 1 条收口评估中：`train_count = 192`，`val_count = 64`，`test_count = 64`

你后续如果想统一全文，就沿用这一套 64 例验证/测试口径写主要实验。

### 13.3 比较设置

建议分三类：

1. 系统内部对照
   - legacy vs expanded
   - global vs adaptive
   - with / without neighborhood evidence
2. 阶段性消融
   - factual base vs improved factual
   - rule-only tendencies vs multi-objective reranking
3. 轻量稳定性分析
   - seeds
   - threshold sensitivity

### 13.4 评价指标

建议按四层写。

#### Factual

- MAE
- RMSE
- calibration coverage
- sliced MAE on high-volatility / regime-shift / vasopressor-related cases

#### Retrieval

- donor similarity
- neighborhood consistency
- exchangeability mean
- action alignment

#### Candidate

- candidate count
- parameterized candidate count
- strategy candidate count
- invalid-rate proxies

#### Counterfactual / Guardrail

- selected score
- predicted delta
- lower bound
- recommendation_ready_rate
- review_only_rate
- stable_guardrail_status_rate

---

## 14. Results 章节详细大纲

建议拆成 6 个小节。

### 14.1 Overall system result

这里先给总体结论，不要一下子堆所有数字。

可以写：

- the framework forms a complete and auditable pipeline from donor retrieval to physician-facing reporting
- the current system is conservative but stable
- the major improvements lie in candidate-space expressiveness and safer abstention behavior

### 14.2 Factual forecasting improvements

可直接引用的现有数字：

- factual `MAE` 下降 `0.0762`
- factual `RMSE` 下降 `0.0950`
- high-volatility slice `MAE` 下降 `0.1567`
- regime-shift slice `MAE` 下降 `0.1426`
- vasopressor-related slice `MAE` 下降 `0.2615`

应当同时坦诚写：

- 覆盖率上升，但 calibration gap 变大
- 说明区间更保守，但不是更锋利

### 14.3 Donor retrieval and donor pool analysis

这里可用第 3 条和第 4 条结果。

可写数字：

- 第 3 条：
  - `changed_case_rate = 0.03125`
  - `mean_neighbor_consistency: 0.5696 -> 0.5713`
  - `mean_exchangeability: 0.4261 -> 0.4271`
  - `mean_selected_total_score: 1.2561 -> 1.2725`
- 第 4 条：
  - `global`: `mean_similarity = 0.8927`, `mean_guideline_compatibility = 0.5987`, `overlap_pass_rate = 0.7656`
  - `same_hospital`: guideline compatibility 降到 `0.4711`
  - `same_unit`: guideline compatibility 降到 `0.3236`
  - `adaptive`: `changed_case_rate = 0.4688`, `mean_neighbor_consistency = 0.5783`, `overlap_pass_rate = 0.8125`

写法重点：

- `adaptive` 不是最严格，但最平衡
- donor 纠偏是选择性介入，不是大规模翻盘

### 14.4 Candidate-space expansion results

这里写第 5 条，是论文最容易出彩的部分之一。

可写数字：

- `mean_candidate_count: 2.375 -> 4.875`
- `mean_parameterized_candidate_count: 0.515625 -> 3.015625`
- `mean_strategy_candidate_count: 0.0 -> 2.5`
- `changed_case_rate = 0.21875`
- invalid-rate proxies 没升高

解释重点：

- 候选空间扩展是真实改变 winner 的
- 不是只在可视化层面增加候选
- safe-deviation 候选已经开始成为 winner

### 14.5 Lightweight stability analysis

这里写第 7 条轻量版。

可写数字：

- baseline adaptive:
  - `recommendation_ready_rate_mean = 0.0`
  - `recommendation_ready_rate_std = 0.0`
  - `stable_guardrail_status_rate = 1.0`
- strict lower bound threshold:
  - selection / status flip 都是 `0.0`
- `global_pool` 相对 baseline:
  - `mean_selected_score_delta = -0.00224`
  - `mean_neighbor_consistency_delta = -0.00527`
- `legacy_candidates` 相对 baseline:
  - `selection_changed_case_rate = 0.21875`
  - `mean_selected_score_delta = -0.00666`

写法重点：

- 当前系统非常保守，但这种保守是稳定的
- 轻量稳定性分析支持“当前结论不是碰巧出现”

### 14.6 Qualitative case study

主案例建议用 `stay_141959`。

这个案例能说明：

- donor 非常相似
- 但推荐依然被 guardrail 拦下
- safe-deviation 候选虽然更像临床上会考虑的桥接方案，但仍然因为证据不足被降为 review_only

还可以加一个正向但谨慎的例子，如之前出现过的：

- `stay_146133`
- `stay_145951`

在正文中，你可以只放 1 个主案例，把其他放附录。

---

## 15. 如何写主案例 `stay_141959`

建议按医生阅读顺序写，而不是按 JSON 字段写。

### 15.1 病情判断

- 当前患者表现出高乳酸、低血压、器官功能障碍和 sepsis 信号
- 病程模式更接近 spike / shifted regime
- 当前主要缺口在于血流动力学支持偏弱

### 15.2 donor 证据

- donor `stay_141227`
- donor similarity 约 `0.964`
- state match `1.0`
- 说明系统确实找到了一个非常像的病例

### 15.3 候选干预

在第 5 条后，这个病例的候选已经不只是 donor-centered 方案，而是：

- `generated_template_vasopressor_bridge`
- `generated_template_vasopressor_low`
- `generated_strategy_sepsis_hemodynamic_combo`

### 15.4 为什么最后仍是 review_only

这一步是文章里最能体现系统边界意识的地方。

虽然：

- 表面上 `predicted_delta` 可能接近 0 或略正

但：

- lower bound 仍然为负
- uncertainty penalty 偏高
- neighborhood support 不够强
- 因此系统输出 review-only

### 15.5 这个案例在论文中的意义

它能说明三件事：

1. 系统不是简单的“相似就推荐”
2. 系统已经能给出 safe-deviation 候选
3. 系统会把“像但不够稳”的方案拦下来

---

## 16. Discussion 章节详细大纲

### 16.1 系统真正做对了什么

建议写：

- It separates similarity from transferability.
- It separates candidate generation from candidate acceptance.
- It turns abstention into a first-class system behavior rather than a failure mode.

### 16.2 为什么当前结果很多是 review-only

不要回避，应该主动解释：

- 当前系统 guardrail 比早期严格
- 当前证据要求 donor、counterfactual proxy、uncertainty、neighborhood support 同时过关
- 这会显著降低“误放行”的概率，但也会减少 recommendation_ready

### 16.3 为什么这仍然有价值

可以这样写：

- 在临床决策支持中，稳定地告诉医生“当前不该直接采纳”本身就是有价值输出
- 这比输出一个看似积极但不稳的建议更可取

### 16.4 当前最重要的局限

必须写清楚：

- 反事实收益仍是模型内模拟
- 不是严格因果效应
- 仍主要针对短期 proxy
- 当前轻量稳定性分析不等于完整稳健性证明

---

## 17. Limitations and Ethical Considerations 大纲

这一节要写得坦诚，不要弱化。

### 17.1 Technical limitations

- counterfactual estimates remain model-based rather than causal
- rollout is still short-horizon
- donor evidence may still be sparse in some neighborhoods
- recommendation_ready is currently rare because the system is conservative

### 17.2 Clinical limitations

- 不可直接作为医嘱
- 需要医生复核
- retrospective evaluation 不能替代 prospective validation

### 17.3 Ethical considerations

- abstention is preferable to overconfident recommendation
- outputs should be interpreted as decision support rather than treatment orders
- retrospective cohorts may encode historical practice bias

---

## 18. Conclusion 大纲

结论不宜写太大。

推荐写法：

- 本文提出并验证了一个相似患者检索 + 反事实候选重排 + 医生复核型 guardrail 框架
- 它的主要贡献在于把 donor evidence、safe-deviation candidate generation、multi-objective uncertainty-aware ranking、physician-facing reporting 串成闭环
- 当前最合理的系统定位是审慎型临床决策支持，而非自动治疗推荐

---

## 19. 建议放入正文的表和图

### 图 1：系统总框架图

内容：

- patient window
- factual branch
- retrieval branch
- donor evidence
- candidate generation
- reranking
- guardrail
- doctor view

### 图 2：单病例医生版输出示意图

建议用 `stay_141959`

### 表 1：数据集与任务定义

内容：

- cohort
- window length
- forecast horizon
- target definitions

### 表 2：四层指标体系

内容：

- factual
- retrieval
- candidate
- counterfactual / guardrail

### 表 3：事实预测改进结果

可用：

- MAE
- RMSE
- slice-level gains

### 表 4：donor 层与 donor pool 对比

可用：

- global
- same_hospital
- same_unit
- adaptive

### 表 5：候选空间扩展对比

可用：

- legacy vs expanded

### 表 6：轻量稳定性分析

可用：

- baseline adaptive
- strict lower bound
- global pool
- legacy candidates

### 表 7：案例研究

建议列：

- current plan
- selected candidate
- donor evidence
- guardrail decision
- why review-only

---

## 20. 已有结果与文件映射表

下面这些文件已经可以直接作为论文素材来源。

### 方法总览与系统叙事

- [EICU_COUNTERFACTUAL_SIMILAR_PATIENT_METHOD_GUIDE_20260413.md](e:/worktable/日常/脓毒症记忆/记忆代码/DSCS/temp_repo/memory_mvp_project/docs/02_eicu/EICU_COUNTERFACTUAL_SIMILAR_PATIENT_METHOD_GUIDE_20260413.md)

### 第一轮九阶段修改跟踪

- [EICU_NINE_STEP_MODIFICATION_TRACKER_20260413.md](e:/worktable/日常/脓毒症记忆/记忆代码/DSCS/temp_repo/memory_mvp_project/docs/02_eicu/EICU_NINE_STEP_MODIFICATION_TRACKER_20260413.md)

### 第二轮下一阶段修改跟踪

- [EICU_NEXT_STAGE_MODIFICATION_TRACKER_20260415.md](e:/worktable/日常/脓毒症记忆/记忆代码/DSCS/temp_repo/memory_mvp_project/docs/02_eicu/EICU_NEXT_STAGE_MODIFICATION_TRACKER_20260415.md)

### 关键实验结果

- factual completion: [phase1_completion_evaluation.json](e:/worktable/日常/脓毒症记忆/记忆代码/DSCS/temp_repo/memory_mvp_project/output/analysis/phase1_completion_evaluation.json)
- multitask reranking: [phase2_multitask_reranking_evaluation_v2.json](e:/worktable/日常/脓毒症记忆/记忆代码/DSCS/temp_repo/memory_mvp_project/output/analysis/phase2_multitask_reranking_evaluation_v2.json)
- neighbor consistency: [phase3_neighbor_consistency_evaluation_v3.json](e:/worktable/日常/脓毒症记忆/记忆代码/DSCS/temp_repo/memory_mvp_project/output/analysis/phase3_neighbor_consistency_evaluation_v3.json)
- donor pool: [phase4_stratified_donor_pool_evaluation_v4.json](e:/worktable/日常/脓毒症记忆/记忆代码/DSCS/temp_repo/memory_mvp_project/output/analysis/phase4_stratified_donor_pool_evaluation_v4.json)
- candidate space: [phase5_candidate_space_evaluation.json](e:/worktable/日常/脓毒症记忆/记忆代码/DSCS/temp_repo/memory_mvp_project/output/analysis/phase5_candidate_space_evaluation.json)
- paper stability: [phase7_paper_stability_evaluation.json](e:/worktable/日常/脓毒症记忆/记忆代码/DSCS/temp_repo/memory_mvp_project/output/analysis/phase7_paper_stability_evaluation.json)

### 关键案例

- physician guardrail case: [phase9_physician_guardrail_141959.md](e:/worktable/日常/脓毒症记忆/记忆代码/DSCS/temp_repo/memory_mvp_project/output/analysis/phase9_physician_guardrail_141959.md)
- stage5 case: [phase5_stage1_inference_141959.md](e:/worktable/日常/脓毒症记忆/记忆代码/DSCS/temp_repo/memory_mvp_project/output/analysis/phase5_stage1_inference_141959.md)

---

## 21. 论文里建议强调的正面结论

建议重点强调：

- 系统闭环完整
- donor evidence 不再只看相似度
- candidate generation 已经从 donor-centered 修补升级为 safe deviation
- guardrail 具备明确的 abstention 机制
- 医生版输出已经由代码稳定生成
- 轻量稳定性分析支持当前主要结论不轻易翻转

---

## 22. 论文里必须主动交代的负面结论

一定要主动写，不要等审稿人提。

- 当前 recommendation_ready 很少，系统整体偏保守
- 当前收益主要是 proxy 层面的短期模拟
- 当前第 7 条只是轻量稳定性分析，不是完整因果论证
- 当前第 6 条完整 rollout 和第 8 条反馈闭环还未作为论文主干完成

主动写这些内容的好处是：

- 文章边界更可信
- 你不会被迫解释一个你其实没做完的强结论

---

## 23. 当前最推荐的写稿顺序

建议不要一上来写摘要，先按下面顺序推进。

1. 先确定题目和论文定位
2. 再写 Method 全章节
3. 再写 Results 的表与图结构
4. 再写 Introduction
5. 最后写 Abstract 和 Conclusion

原因：

- 你现在最强的是方法闭环和实验素材，不是故事性包装
- 先把方法和结果结构固定，摘要和引言会更稳

---

## 24. 后续我可以继续为你生成的内容

如果你继续推进，我建议按下面顺序继续生成：

1. 摘要候选稿 3 版
2. 引言正式初稿
3. Method 正式初稿
4. Results 正式初稿
5. 图表清单与 caption 草案
6. Cover letter / response strategy 草案

---

## 25. 当前一句话总结

这篇论文最合理的写法，不是“我们做了很多修改把模型调好了”，而是：

> 我们构建了一个面向 eICU 脓毒症短期干预支持的相似患者检索与反事实候选重排框架，它强调 donor 可比性、safe deviation 候选生成、不确定性感知排序和医生复核型 guardrail，并在 retrospective 分层实验中展示出更完整、更可审计、且更稳定的决策支持能力。
