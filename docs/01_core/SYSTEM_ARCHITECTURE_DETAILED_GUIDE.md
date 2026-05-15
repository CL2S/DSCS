# 系统架构详解

## 系统定位

`memory_mvp_project` 不是单一模型仓库，而是一个围绕“记忆增强时序学习”搭建的统一研究框架。当前系统面向三类任务：

- 表格分类
- ICU / Sepsis 时序建模
- 多步时序预测与反事实分析

三类任务共享同一套核心思想：

1. 把当前输入编码成可比较的表示。
2. 从历史经验中检索相似状态。
3. 用当前输入与检索结果共同生成最终输出。

## 总体结构

系统可以分成六层：

1. 数据层
   - 负责原始数据读取、窗口切分、样本构造与标准化
2. 表示层
   - 把时序窗口、静态变量和干预信息编码为统一向量表示
3. 记忆层
   - 用多种 memory bank 存储历史经验，并支持读写、合并、衰减与压缩
4. 调度层
   - 根据场景特征决定 `pattern / trajectory / experience` 三类记忆的使用权重
5. 预测层
   - 负责 factual 预测、counterfactual donor 检索和干预迁移
6. 知识层
   - 用脓毒症知识图谱增强样本表示与 donor 排序

## 顶层目录

核心目录如下：

- `src/`
  - 主要源码
- `input/`
  - 输入数据与知识资源
- `output/`
  - 结果文件、分析文件和中间产物
- `docs/`
  - 系统文档

主要入口脚本：

- [run_experiment.py](/run_experiment.py)
- [run_temporal_experiment.py](/run_temporal_experiment.py)
- [run_forecasting_experiment.py](/run_forecasting_experiment.py)

## 源码分工

### 数据与样本构造

- [data_utils.py](/src/data_utils.py)
  - 表格数据读取与编码
- [temporal_data.py](/src/temporal_data.py)
  - patient/window 级时序数据组织
- [tsf_data.py](/src/tsf_data.py)
  - forecasting 样本构造，定义 `ForecastSample`
- [ts_formation.py](/src/ts_formation.py)
  - 计算 formation 特征、pattern 标签、trajectory 标签与 experience 标签

### 表示与 memory 基础设施

- [manifold_memory.py](/src/manifold_memory.py)
  - manifold 编码器与 attention memory reader
- [memory_components.py](/src/memory_components.py)
  - `PatternMemory`、`TrajectoryMemory`、`ExperienceMemory`
- [memory_manager.py](/src/memory_manager.py)
  - 负责多 memory 的统一调度

### 训练与推理

- [memory_model.py](/src/memory_model.py)
  - 早期 tabular / temporal 分类路线
- [manifold_trainer.py](/src/manifold_trainer.py)
  - temporal manifold 训练器
- [manifold_forecasting_trainer.py](/src/manifold_forecasting_trainer.py)
  - 当前 forecasting 主线训练器

### 指标与持久化

- [forecasting_metrics.py](/src/forecasting_metrics.py)
- [evaluate.py](/src/evaluate.py)
- [persistent_memory_store.py](/src/persistent_memory_store.py)

### 知识图谱接入

- [kg_integration.py](/src/kg_integration.py)

## 核心样本对象

当前 forecasting 主线的基础样本是 [ForecastSample](/src/tsf_data.py#L279)。它同时包含：

- `sequence`
  - 生理时序窗口
- `patient_static`
  - 患者静态与上下文特征
- `intervention_static`
  - 干预静态特征
- `intervention_sequence`
  - 干预时序特征
- `formation_features`
  - 结构化时序形态特征
- `pattern_label / trajectory_label / experience_label`
  - 检索与调度用的弱语义标签
- `kg_features`
  - 知识图谱映射后的特征向量

这个样本结构是系统的重要设计点，因为它把“患者状态”“干预状态”“时序形态”“知识先验”明确拆开了。

## Memory 体系

系统不是单一 memory，而是三类 memory 并行：

- `pattern_memory`
  - 存局部形态相似经验
- `trajectory_memory`
  - 存较长程的动态轨迹经验
- `experience_memory`
  - 存 pattern 与 trajectory 组合后的综合经验

这三类 memory 由 [MetaMemoryManager](/src/memory_manager.py#L22) 协同管理。它们的作用不是简单投票，而是：

- 在不同时间尺度上约束近邻搜索
- 提供不同粒度的 readout
- 共同形成最终的 memory 引导表示

## Forecasting 主线的数据流

当前最重要的数据流发生在 [manifold_forecasting_trainer.py](/src/manifold_forecasting_trainer.py)：

1. 从 `ForecastSample` 读取 `sequence / patient_static / intervention_static / intervention_sequence`
2. 用 manifold encoder 生成 `query / key / value / input_embedding`
3. 用 `MetaMemoryManager.read(...)` 检索 pattern、trajectory、experience 三类经验
4. 用 planner、confidence 和 scenario features 计算融合门控
5. 形成 factual prediction
6. 在 counterfactual 路径下，从 intervention store 检索 donor 并替换干预
7. 输出 factual / counterfactual 结果与可解释元数据

## 反事实路径

counterfactual 分支不是重新训练一个因果模型，而是在当前 factual 模型上做 donor 干预迁移：

1. 先根据当前样本的经验标签与 memory 读出缩小 donor 范围
2. 再在 intervention store 中检索候选 donor
3. 用相似度、KG 特征和规则惩罚对 donor 重排
4. 把 donor 的 `intervention_static` 与 `intervention_sequence` 替换到当前样本
5. 通过同一模型得到 counterfactual prediction

这一设计的价值在于：反事实路径与 factual 路径共享表示空间，因此 donor 选择可以被同一套患者状态表征约束。

## 知识图谱接入点

知识图谱不是独立运行，而是作为知识层进入系统，主要有两个接入位置：

- 样本级接入
  - eICU 变量先映射为 `kg_feature_vector`
- donor 级接入
  - donor 排序时加入 `kg_similarity`、`guideline_compatibility`、`state_match` 与惩罚项

因此，KG 在系统中的角色是：

- 先验特征层
- donor 合理性约束层
- 结果解释层

## 核心创新点

当前系统的主要创新点不是单个模块，而是这些模块的组合：

- 用 formation 特征把时序窗口映射到可解释的弱语义空间
- 用多 memory 而不是单 memory 建模不同时间尺度经验
- 把患者状态与干预信息拆开编码，再在预测端重组
- 把 persistent memory 与在线 memory 统一到一套 retrieval 框架
- 把脓毒症知识图谱接入 donor 选择，而不是只做静态展示图

## 适合如何理解这个系统

最合适的理解方式不是“一个大模型”，而是“一个围绕经验复用建立的时序决策框架”。

在这个框架里：

- encoder 负责描述当前病例
- memory 负责找相似经验
- intervention store 负责提供 donor 干预模板
- KG 负责把 donor 选择拉回到临床知识约束下

这也是当前文档体系的组织方式。
