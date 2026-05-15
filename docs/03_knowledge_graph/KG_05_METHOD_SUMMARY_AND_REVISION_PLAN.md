# 方法概括与下一轮修改方案汇总

## 背景

`memory_mvp_project` 当前已经形成一条较完整的 forecasting / counterfactual 主线。系统的核心不是单一预测器，而是一个“记忆增强时序建模框架”：先编码当前病例状态，再从历史经验中检索相似记忆，最后结合干预信息和知识图谱完成 factual 预测与 counterfactual 分析。

知识图谱部分目前已经完成脓毒症子图裁剪、指南增强、eICU 到 KG 特征映射，以及 donor 检索时的规则约束；记忆部分已经完成 online memory、persistent experience memory、semantic prototype retrieval 和 neural cache。

导师进一步提出的要求是：不仅要把记忆库作为检索或解释工具，还要进一步用于提升模型的预测能力；同时建议考虑强化学习、马尔科夫链等方法，使系统从“单步 donor 检索增强”进一步走向“状态转移建模与受限策略优化”。

## 本轮修改目标

1. 用一份文档概括当前方法，包括研究目的、研究目标、核心方法、创新点和当前边界。
2. 汇总导师提出的新修改要求，并给出一条与当前仓库连续、可逐步实现的修改方案。
3. 明确下一轮修改不应另起炉灶，而应在现有记忆库、forecasting 主线和 KG 约束机制上增量扩展。

## 修改内容

### 1. 当前研究目的

当前研究的总目标不是单纯做一个脓毒症分类器或单一 forecasting 模型，而是构建一个面向 ICU / Sepsis 场景的记忆增强时序学习框架。这个框架希望回答两个核心问题：

1. 在患者状态高度异质、时间演化明显的临床任务中，能否通过复用历史经验提升预测质量。
2. 在反事实分析和干预迁移中，能否通过医学知识约束把经验复用限制在更临床合理的范围内。

换句话说，本项目想解决的是“如何让模型不仅看当前输入，还能看历史相似经验，并且在医学规则约束下做出更稳定、更可解释的预测与反事实评估”。

### 2. 当前研究目标

围绕上面的研究目的，当前系统的具体目标主要有四个：

1. 对 ICU / Sepsis 时序窗口进行 factual forecasting，预测未来病情变化。
2. 通过 donor 检索与干预迁移进行 counterfactual 分析，评估不同干预方案的潜在影响。
3. 把患者状态、干预信息、时序形态和知识先验拆开建模，提高系统可解释性。
4. 让知识图谱真正参与 donor 选择与规则审计，而不是只作为静态展示资源。

### 3. 当前方法的总体结构

当前方法可以概括为六层：

1. 数据层  
   负责 eICU / forecasting 数据读取、窗口切分、标准化与样本构造。
2. 表示层  
   用 manifold encoder 将时序窗口、静态特征、干预特征编码为统一表示。
3. 记忆层  
   构建 `pattern_memory`、`trajectory_memory`、`experience_memory` 三类 online memory bank。
4. 持久化经验层  
   用 persistent experience store 保存经验条目、semantic prototypes 和 neural cache，使经验可以跨运行复用。
5. 预测与反事实层  
   一方面输出 factual prediction；另一方面从 intervention store 中检索 donor，生成 counterfactual prediction。
6. 知识约束层  
   用脓毒症 KG 提供 `kg_feature_vector`、`kg_similarity`、`guideline_compatibility`、`state_match` 以及相关 penalty，对 donor 选择和候选修复进行约束。

### 4. 当前方法的核心机制

当前方法的关键流程可以简化为：

1. 读取当前病例窗口，得到患者状态、干预状态、形态特征和 KG 特征。
2. 用 manifold encoder 得到统一 embedding。
3. 从多类 memory 中检索历史经验，获得 memory readout。
4. 通过 manager / planner 对多类 memory 的贡献做动态加权。
5. 输出 factual 预测。
6. 在 counterfactual 路径下检索 donor 干预，并结合相似度、KG 相容性与规则惩罚进行重排。
7. 对 donor 干预做最小规则修复，得到 `generated_best` 等候选，再生成 counterfactual 预测。

### 5. 当前方法的核心创新点

当前系统的创新点主要不在单一模型结构，而在多个模块的组合方式：

1. **记忆增强时序建模**  
   不是只依赖当前输入，而是把历史经验作为显式 memory bank 引入预测过程。
2. **多粒度记忆设计**  
   用 `pattern / trajectory / experience` 三类 memory 同时覆盖局部形态、轨迹动态和综合经验，而不是只做单一近邻检索。
3. **患者状态与干预状态分离建模**  
   把 `patient state` 和 `intervention state` 拆开编码，再在预测或反事实阶段重组，使 donor 迁移更清晰。
4. **persistent memory + online memory 一体化**  
   当前运行中的 online memory 与跨运行沉淀的 persistent experience store 被统一在同一套 retrieval 框架中。
5. **知识图谱真正进入下游决策**  
   KG 不是静态背景图，而是直接参与 donor 排序、候选修复和规则惩罚。
6. **knowledge-safe write 与 KG-guided candidate generation**  
   当前系统已经开始在写入和候选生成阶段引入医学知识安全约束，使经验复用更临床一致。

### 6. 当前方法的边界

当前系统虽然已经具备较强的 donor 检索与规则修复能力，但仍然有明确边界：

1. 它目前仍然主要是 donor 检索增强器和临床一致性约束层。
2. 它还不是完整的多步治疗规划器。
3. 它还没有显式建模患者状态从 `t` 到 `t+1` 的状态转移概率。
4. 它还没有完整 action space、剂量约束、时间窗约束和严格因果识别层。

因此，当前方法更接近“经验增强预测 + 规则约束反事实”，而不是“显式序列决策模型”。

### 7. 导师提出的修改要求

导师提出的新要求可以概括为三点：

1. 记忆库不仅要支持检索和解释，还要进一步提升预测能力。
2. 方法层面应考虑引入强化学习思想，而不是停留在静态 donor 重排。
3. 应考虑马尔科夫链或状态转移建模，让系统具备更明确的病程演化表示。

### 8. 对应的修改方案

针对上述要求，建议下一轮方法升级为：

**基于记忆库的马尔科夫状态转移与受限策略优化框架**  
英文可表述为：
**Memory-Guided Markov Transition Policy**

其核心思路不是抛弃现有系统，而是在现有 memory、forecasting 和 KG 机制上增加“状态转移层”和“受限策略层”。

#### 8.1 状态表示升级

把当前时间窗从“仅用于相似检索的 embedding”升级为“显式临床状态表示” `z_t`。  
`z_t` 可以由以下部分组成：

1. 当前 manifold embedding。
2. KG 状态标志，如 `sepsis / septic_shock / hypotension / high_lactate`。
3. 关键临床趋势特征，如 SOFA 变化、MAP 趋势、乳酸变化。
4. 当前干预覆盖信息，如是否已给抗菌药、液体复苏、升压药。

这样做的作用是把“当前病例”从一个纯向量近邻问题，提升为一个具备临床语义的状态节点。

#### 8.2 记忆库升级为状态转移记忆

在现有 persistent experience store 之上，新增一层 transition memory，存储：

`(z_t, a_t, z_t+1, reward_t, support, confidence, guideline_score)`

其中：

1. `z_t` 表示当前状态。
2. `a_t` 表示当前动作或动作组合。
3. `z_t+1` 表示下一时间窗状态。
4. `reward_t` 表示状态改善程度，可先用 SOFA 降低、乳酸下降、休克缓解等代理指标定义。

这样记忆库不再只是“我见过什么样的相似病例”，而是“我见过在什么状态下采取什么动作，通常会转移到什么下一状态”。

#### 8.3 用马尔科夫状态转移建模病程演化

在抽象状态空间上估计：

`P(z_t+1 | z_t, a_t)`

这里的马尔科夫链不是在原始高维病历空间上直接建模，而是在经过时序编码、知识映射和状态抽象后的临床状态空间上建模。  
这样更符合现有系统结构，也更容易结合记忆库中的经验频次、支持度和质量分数。

#### 8.4 用离线强化学习思想做受限策略改进

强化学习部分不建议直接做自由探索，而建议采用**离线、约束、候选受限**的方式：

1. 先由当前 donor 检索器给出候选动作或候选干预序列。
2. 再由 transition memory 估计每个候选在当前状态下的预期转移效果。
3. 最后结合 KG 规则、指南相容性、禁忌惩罚和可行性约束进行重排。

这一步更准确地说是“受限策略改进”，而不是完全开放的 RL。  
它的优点是更安全、可解释，也更符合医疗场景下不能自由探索的事实。

#### 8.5 对现有系统的具体增量改造

如果按工程实现拆解，下一轮建议优先做以下增量：

1. 在 persistent memory 中新增 transition-level entry，而不是只存单个经验条目和 prototype。
2. 为 forecasting 样本增加 `clinical_state_id` 或连续状态向量。
3. 在 donor 候选打分中加入 transition value，例如 `P(improve | z_t, a_t)` 或 `E[reward | z_t, a_t]`。
4. 将当前 `generated_best` 从“单步规则修复”扩展为“单步转移价值重排”。
5. 在验证阶段比较：
   - 仅 factual 预测
   - memory 增强预测
   - KG 约束 donor 重排
   - transition memory + constrained policy 改进

#### 8.6 预期方法收益

如果该方案实现顺利，预期收益主要有四个：

1. 记忆库将直接参与“预测未来状态如何演化”，而不仅是辅助检索相似样本。
2. 系统会从“静态 donor 排序”升级到“基于状态转移的候选评估”。
3. 马尔科夫状态转移会使病程演化建模更显式，便于解释和分析。
4. 强化学习思想会以离线、受约束的方式进入系统，既提高方法深度，也不脱离当前仓库现实。

## 影响范围

本轮文档汇总不修改训练代码和实验脚本，但如果后续按方案实施，主要会影响以下模块：

1. [persistent_memory_store.py](/src/persistent_memory_store.py)
2. [memory_manager.py](/src/memory_manager.py)
3. [manifold_forecasting_trainer.py](/src/manifold_forecasting_trainer.py)
4. [run_forecasting_experiment.py](/run_forecasting_experiment.py)
5. 知识图谱相关说明文档与实验评估文档

## 运行方式

本轮只进行了方法总结与方案设计文档更新，没有新增训练命令、实验脚本或推理入口，因此当前无需新增运行命令。

如果后续进入实现阶段，建议按以下顺序推进：

1. 先补 transition memory 的数据结构与落盘格式。
2. 再补单步 transition score 的检索与重排。
3. 最后再扩展到两步或多步 rollout 与离线策略评估。

## 结果与分析

本轮汇总后的结论是：

1. 当前方法的主线已经比较清晰，研究定位应表述为“记忆增强时序预测与知识约束反事实分析框架”。
2. 当前系统的真正优势在于多记忆、持久化经验、患者状态与干预分离、以及 KG 进入 donor 决策。
3. 导师提出的“记忆库提升预测能力”“强化学习”“马尔科夫链”三点，与当前系统并不冲突，反而正好对应当前系统尚未完成的状态转移建模和多步决策部分。
4. 因此，最合理的下一步不是推翻现有架构，而是把记忆库升级为状态转移记忆，并把 donor 重排升级为受限策略改进。

## 已知问题

当前方案仍然有几个需要提前说明的问题：

1. 马尔科夫假设在真实临床病程中只是近似成立，不应宣称为严格生理机制模型。
2. 离线强化学习依赖历史数据分布，若 action coverage 不足，策略改进容易受偏。
3. 如果 reward 定义过于粗糙，可能导致“预测改善”和“临床真实改善”之间存在偏差。
4. 当前 KG 仍然偏规则层，不等于完整因果图，因此方法表述应避免夸大为严格因果决策系统。

## 下一步计划

建议按以下优先级推进下一轮实现：

1. 先定义 `clinical_state` 和 transition-level memory entry。
2. 先做单步 `transition score`，不要一开始就做完整多步 RL。
3. 把 transition score 先接入现有 donor rerank，再比较是否能提升 factual / counterfactual 指标。
4. 若单步结果有效，再扩展到两步 rollout、候选动作序列和离线策略评估。
5. 最后再考虑更完整的 action space、剂量约束与因果识别层。
