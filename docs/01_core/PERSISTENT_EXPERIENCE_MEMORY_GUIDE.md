# 持久化经验记忆指南

## 模块定位

系统中的 online memory 只能在当前进程内存在，实验结束后会消失；而反事实与 forecasting 主线又需要复用历史经验。为了解决这个问题，仓库引入了 persistent experience memory。

这个模块的目标不是简单“存 checkpoint”，而是把经验拆成三层：

- 参数无关的经验条目
- 可检索的经验原型
- 与当前模型状态绑定的 neural cache

## 三层结构

### 第一层：经验条目

由 [persistent_memory_store.py](/src/persistent_memory_store.py) 维护。

它保存与模型参数无关、可重复使用的经验信息，例如：

- `raw_context`
- `raw_future`
- `normalized_context`
- `normalized_future`
- `formation_features`
- `pattern_label`
- `trajectory_label`
- `experience_label`

这一层的作用是：

- 跨运行保存经验
- 为 prototype 构造提供原始材料
- 为语义检索提供基础数据

### 第二层：经验原型

经验条目会被聚合为 prototypes，用于 semantic retrieval。

prototype 里保存的不是单个样本，而是某一类经验的中心表示，例如：

- `prototype_formation_center`
- `prototype_future_mean_curve`
- `support`

这一层的作用是：

- 用 formation 语义快速找到相近经验簇
- 给 `experience` 分支提供先验
- 支持 template curve 融合

### 第三层：neural cache

neural cache 是与当前模型参数绑定的缓存，按 `model_fingerprint` 区分。

它保存的是当前模型条件下的最终 memory bank，而不是参数无关的原始经验。其作用是：

- 在同一模型条件下快速恢复最终 memory bank
- 避免重复构建完全相同的在线记忆

## 文件结构

持久化 store 目录下通常包含：

- `manifest.json`
- `experience_entries.jsonl`
- `prototypes.jsonl`
- `events.jsonl`
- `indexes/semantic_index.json`
- `cache/<model_fingerprint>.pt`
- `cache/<model_fingerprint>.meta.json`

## 与 memory bank 的关系

需要区分两件事：

- `persistent experience store`
  - 存的是可复用经验
- `online memory bank`
  - 存的是当前模型在当前运行中的读写记忆

两者的关系是：

1. persistent store 提供样本与 prototype
2. trainer 在当前运行中读取这些样本
3. 然后构建 pattern、trajectory、experience 三类 memory bank
4. 若 neural cache 可用，则直接恢复最终 memory bank

因此，persistent memory 是 online memory 的输入来源与恢复机制，不是在线 memory 的替代品。

## 关键实现

### Store 层

- [persistent_memory_store.py](/src/persistent_memory_store.py)

关键职责：

- `upsert_samples(...)`
- `load_samples(...)`
- `rebuild_prototypes(...)`
- `load_prototypes(...)`
- `semantic_retrieve(...)`
- `load_model_cache(...)`
- `save_model_cache(...)`

### Manager 层

- [memory_manager.py](/src/memory_manager.py)

关键职责：

- 接入 semantic retrieval
- 调整 experience prior
- 调整 planner weights
- 生成 semantic template curve

### Trainer 层

- [manifold_forecasting_trainer.py](/src/manifold_forecasting_trainer.py)

关键职责：

- 配置 semantic store
- 配置 neural cache
- 恢复或重建最终 memory bank

## 读取流程

在 forecasting 主线里，持久化经验的使用流程可以概括为：

1. 加载 persistent samples
2. 加载 prototypes
3. 用 prototypes 做 semantic retrieval
4. 用 samples 构建或恢复 online memory bank
5. 在当前训练或推理中参与 factual 与 counterfactual 路径

## 为什么这个设计重要

这套设计解决了三个问题：

- 经验不会因进程结束而丢失
- 语义检索不依赖当前模型参数
- 当前模型下的最终 memory bank 又可以被快速恢复

换句话说，它把“经验沉淀”和“当前模型高效运行”拆开了。

## 当前边界

当前 persistent memory 的边界也很清楚：

- 它不是完整病历数据库
- 它不是通用向量数据库
- 它主要服务于本仓库的 forecasting / counterfactual 任务

目前 neural cache 缓存的是最终 memory bank，而不是每个样本的当前模型编码，因此它优化的是恢复成本，不是全训练过程的全部检索成本。

## 如何判断是否生效

结果文件里通常看这几个字段：

- `loaded_persistent_samples`
- `loaded_persistent_prototypes`
- `persistent_memory.neural_cache.loaded`
- `memory_diagnostics.semantic_retrieval`

如果这些字段有值，说明持久化经验已经实际进入当前运行，而不是只是落盘未使用。
