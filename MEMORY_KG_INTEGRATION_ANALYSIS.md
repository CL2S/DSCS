# 记忆库（Memory Bank）与知识图谱（KG）融合可行性分析

> 创建：2026-05-14 | 基于现有 SOFA 预测系统和新增经验记忆增强预测系统的对比分析

---

## 一、两套系统的定位对比

### 1.1 现有系统（老系统）：LLM 驱动的 SOFA 预测评估

**代码位置**: [main.py](main.py), [experiment.py](experiment.py), [advanced_experience_memory.py](advanced_experience_memory.py)

**核心特征**:
- **推理引擎**: DSPy + Ollama 大模型（deepseek-r1:32b, gemma3:12b, qwen3:30b）
- **记忆库类型**: 三层混合记忆（情景/语义/程序），基于哈希向量 + 符号规则
- **检索方式**: `S = 0.45·dense + 0.35·symbolic + 0.15·quality + 0.05·recency`
- **知识图谱**: 无结构化 KG，仅有经验库中的统计分布
- **预测方式**: 纯 LLM 生成式预测，输出自然语言 + 结构化 JSON
- **界面**: Web (SPA+PWA) + GUI (Tkinter) + CLI

**长处**:
- LLM 零样本推理能力强，无需大量训练数据
- 输出可解释（自然语言 reasoning），便于临床审阅
- 三层记忆设计理论上合理（实例+统计+规则）
- 工程化程度高，有完整的数据库和前端

**短板**:
- 哈希向量表达能力弱于专用 embedding 模型
- 记忆检索依赖手写公式权重，不可学习
- LLM 推理不稳定，同一输入可能得到不同输出
- 评估器也是 LLM，存在"自己评自己"的循环偏差风险
- 无数据泄漏防护机制（persistent experience 可能跨 split 污染）

### 1.2 新系统：深度学习 + 流形记忆的预测增强

**代码位置**: [src/](src/), [docs/](docs/)

**核心特征**:
- **推理引擎**: PyTorch GRU/Transformer 编码器 + 注意力记忆检索
- **记忆库类型**: Pattern/Trajectory/Experience 三组件 + Semantic Prototype Store
- **检索方式**: 可学习的注意力机制 + label prior + cosine similarity
- **知识图谱**: 完整的静态医学 KG（LMKG + SSC 指南增强），eICU 变量 → KG 节点映射
- **预测方式**: Base predictor + Memory residual + Transition residual = Fused prediction
- **门控机制**: Harm control (quality/alignment/cap) + Continuous transition gate + learnable gate (待实现)
- **界面**: CLI 实验脚本 + 诊断报告

**长处**:
- 端到端可训练，记忆检索和预测可联合优化
- Split-safe 持久化经验库，有完整的泄漏审计
- 多层门控（harm control + transition gate），细粒度控制记忆影响
- Clinical State Signature 显式建模临床状态（severity/trend/volatility）
- 结构化 KG 特征，有 mapping rules 将 eICU 变量映射到 KG 节点
- Transition memory 建模状态转移，有 utility 评估
- Counterfactual 干预规划能力
- 经过 R0-S2 + Phase A 多轮实验验证，有详细的硬伤诊断

**短板**:
- 训练随机性主导性能（同一配置 pre_strength 从 3.66 到 41.80）
- 无"记忆沉默"机制，stable_regime 患者 60-73% 被 harm
- Transition utility 是启发式公式，符号随训练随机翻转
- 持久化经验缺乏 relevance 筛选，放大噪声
- 检索以"形态相似度"为主，与"预测修正价值"脱节
- Harm control 只能管幅度不能管方向
- 中等以上严重度患者系统性受损

---

## 二、两系统各组件的对应关系

```
老系统（LLM 驱动）              新系统（DL 驱动）              融合方向
──────────────────────────────────────────────────────────────────────
advanced_experience_memory  ←→  MetaMemoryManager             架构融合的关键接口
  EpisodicMemory           ←→  ExperienceMemory (hot+archive) 经验记忆
  SemanticMemory           ←→  SemanticPrototype Store        语义原型
  ProceduralMemory         ←→  KG guideline_relations         KG 规则替代
  _hash_embedding          ←→  GRU/Transformer Encoder        DL encoder 替代哈希
  get_recommendations()    ←→  ManagerReadResult              LLM 检索增强

无对应                      ←→  PatternMemory                 新增组件
无对应                      ←→  TrajectoryMemory              新增组件
无对应                      ←→  TransitionMemory              新增组件
无对应                      ←→  ClinicalStateSignature        新增组件
无对应                      ←→  HarmControl (3-layer gate)    新增组件

experience_knowledge_base  ←→  PersistentExperienceStore      持久化
  基础相似检索              ←→  split-safe + leak audit       安全增强

_clinical_risk_gate()      ←→  LLM Clinical Supervisor        LLM 门控替代规则
  qSOFA 规则门控            ←→  语义判断 "该不该修正"          LLM ≥ 规则

无对应                      ←→  KG Feature Builder             增量能力
无对应                      ←→  Counterfactual Plan Renderer   增量能力
```

---

## 三、融合可行性：按层次分析

### 3.1 KG 层融合 — 可行性：高（★ ★ ★ ★ ★）

这是最容易融合的一层，两系统几乎天然互补。

**现状**:
- 新系统的 `KnowledgeGraphFeatureBuilder` 已有完整的 KG 基础设施：LMKG 底座 + SSC 2021 指南增强 + eICU→KG 节点映射表 + 规则引擎（label_positive/series_present/series_min_lt 等 12 种规则类型）
- 老系统完全没有结构化 KG，仅有经验库中的统计分布

**融合方案**：

```
老系统现状:
  advanced_experience_memory.py
    └── SemanticMemory: risk_distribution + intervention_effectiveness
         (仅统计，无医学本体约束)

融合后:
  advanced_experience_memory.py
    ├── SemanticMemory: risk_distribution + intervention_effectiveness (保留)
    └── KG Layer (新增):
         ├── KnowledgeGraphFeatureBuilder: eICU→KG 特征映射
         ├── kg_features 向量: 54 维 KG flag (sepsis/shock/hypotension/antibiotic/vasopressor...)
         ├── guideline_alignment 指数: SSC 指南遵从度评分
         └── KG consistency check: donor 方案的临床约束验证
```

**具体做法**：
- 将 `src/kg_integration.py` 的 `KnowledgeGraphFeatureBuilder` 和映射表直接引入老系统
- 在 `experience_integration.py` 的检索步骤中，用 KG features 替代或增强当前的符号匹配项（当前 `S_symbolic` 仅基于干预类型和风险标签匹配）
- KG 的 guideline alignment 可作为 ProceduralMemory 中规则巩固的外部验证信号

**收益**：
- 为老系统的经验检索增加医学本体约束，减少"语义近但临床策略不一致"的误召回
- guideline alignment 可以提供规则来源的可追溯性，增强可解释性

---

### 3.2 Memory Bank 层融合 — 可行性：中高（★ ★ ★ ★ ☆）

这是最有价值的融合层，但需要解决架构冲突。

**核心冲突**：

| 维度 | 老系统 | 新系统 | 冲突程度 |
|------|--------|--------|----------|
| **编码方式** | 哈希向量（无参数） | GRU/Transformer（可学习） | 高 |
| **记忆单元** | EpisodicMemory dataclass | ManifoldMemoryItem (key+value+label) | 中 |
| **检索方式** | 混合评分公式（手写权重） | Attention read + label prior | 高 |
| **写入策略** | 每个病例一条 | 合并（merge_alpha）+ support 计数 | 中 |
| **记忆持久化** | 无（session-only） | JSONL split-safe | 低（直接引入） |
| **门控机制** | 无 | Harm control + Transition gate | 低（直接引入） |

**融合方案**：

```
老系统记忆检索流程 (当前):
  input_description + intervention
    → _hash_embedding (哈希向量)
    → 混合评分 S = 0.45·dense + 0.35·symbolic + 0.15·quality + 0.05·recency
    → Top-K 相似经验
    → 增强预测上下文

融合后记忆检索流程 (建议):
  input_description + intervention
    ├─→ [路径A: LLM Semantic Path] (老系统增强)
    │     → LLM 提取临床摘要 (Clinical State Signature)
    │     → KG Feature Builder 生成 kg_features
    │     → 混合检索 (改进评分, 加入 KG 维度)
    │     → Top-K 经验 + KG 规则建议
    │
    └─→ [路径B: DL Manifold Path] (新系统能力)
          → GRU/Transformer Encoder
          → Pattern/Trajectory/Experience Memory read
          → Semantic Prototype Store 检索
          → Memory residual (可选注入)
```

**关键改造点**：

1. **Encoding 层统一**（最困难）:
   - 短期方案：双路径并行，各自编码，在检索结果层合并
   - 长期方案：用 DL encoder 替代哈希向量，但需要解决训练数据问题（老系统没有 eICU 时序数据训练 encoder）

2. **PersistentExperienceStore 直接引入**（最直接）:
   - 老系统的 `advanced_experience_memory.py` 完全没有持久化
   - 新系统的 `PersistentExperienceStore` 是成熟的 split-safe 方案，可以直接引入
   - 包含：JSONL 写入/加载、来源追踪（train/val/test/hospital/stay）、泄漏审计、reuse_audit 报告

3. **门控机制迁移**（中难度）:
   - 新系统的三道 harm control (quality/alignment/cap) 可以作为老系统经验注入的后处理层
   - 替代当前 `_clinical_risk_gate()` 中简单的 qSOFA 二值判断

---

### 3.3 预测层融合 — 可行性：低（★ ★ ☆ ☆ ☆）

短期不建议直接融合预测层。

**原因**：
- 老系统的预测完全依赖 LLM（DSPy 编排），输出的是自然语言 reasoning + 结构化 JSON
- 新系统的预测是 DL 模型（base predictor + memory residual + transition residual），输出的是数值向量
- 两者的输出格式、不确定性特征、校准方式完全不同

**替代方案**：
- 保持老系统的 LLM 预测作为主输出（因为有自然语言可解释性优势）
- 新系统的 memory residual 作为**可选的后处理修正信号**（类似 ensemble）
- 或者反过来：新系统的 DL prediction 作为主输出，老系统的 LLM 作为 reasoning 生成器

---

### 3.4 LLM Clinical Supervisor — 可行性：高（★ ★ ★ ★ ★）

这是新系统文档中 [docs/03_LLM_AGENT_PLAN.md](docs/03_LLM_AGENT_PLAN.md) 提出的方案，恰好可以利用老系统的 LLM 基础设施（DSPy + Ollama）。

**三个 LLM 决策点**：

| 决策点 | 老系统已有的能力 | 融合方式 |
|--------|----------------|---------|
| **Memory Gate**: 该不该用记忆修正？ | DSPy 签名定义 + Ollama 调用 | 新增 DSPy 签名 `MemoryGateDecision` |
| **Retrieval Filter**: 检索结果哪些可信？ | 经验库混合评分（可增强为 LLM 二次筛选） | 用 LLM 对 Top-K 做 rerank |
| **Transition Utility**: 效用是正还是负？ | 老系统无此概念，但 LLM 天然适合做语义判断 | 新增 DSPy 签名 `TransitionUtilityAssessment` |

**具体实现**：

```python
# 在 experience_integration.py 中新增
class LLMClinicalSupervisor:
    def memory_gate(self, patient_state: ClinicalStateSignature,
                    memory_context: dict) -> MemoryGateDecision:
        """用 LLM 判断当前患者是否需要经验修正"""
        # 通过 DSPy 调用 Ollama 模型
        # 输出: {apply: bool, confidence: float, reason: str}

    def retrieval_filter(self, query: str, top_k_results: list,
                         patient_state: ClinicalStateSignature) -> list:
        """用 LLM 对检索结果做 relevance 二次筛选"""

    def transition_utility(self, state_before, action, state_after) -> float:
        """用 LLM 评估状态转移的临床效用"""
```

**收益**：
- 直接解决新系统硬伤 #2（无沉默机制）—— LLM 可以识别 stable_regime
- 缓解硬伤 #4（持久化噪声）—— LLM 可以做 relevance 筛选
- 缓解硬伤 #3（utility 不可靠）—— LLM 语义判断替代启发式公式
- 老系统的 LLM 基础设施（DSPy、Ollama、模型管理）可以立刻复用

---

## 四、推荐的融合路线

### 第一阶段：立即可行（1-2 周）

**目标**: KG 集成 + 持久化经验库引入

| 任务 | 老系统改动 | 新系统改动 | 风险 |
|------|-----------|-----------|------|
| KG Feature Builder 引入 | 在 `experience_knowledge_base.py` 中集成 `kg_integration.py` | 提取为独立模块 | 低 |
| 持久化经验库引入 | 在 `advanced_experience_memory.py` 中集成 `PersistentExperienceStore` | 提取为独立模块 | 低 |
| 检索评分公式改进 | 在混合评分中加入 KG 维度: `+ 0.10·S_kg` | 无 | 低 |

**预期效果**: 经验检索质量提升（加入医学本体约束），经验库具备持久化能力

### 第二阶段：架构增强（2-3 周）

**目标**: LLM Clinical Supervisor

| 任务 | 老系统改动 | 新系统改动 | 风险 |
|------|-----------|-----------|------|
| Clinical State Signature | 新增患者状态结构化提取（DSPy 签名） | 提供签名格式规范 | 低 |
| LLM Memory Gate | 新增 DSPy 签名，调用 Ollama 做门控 | 提供 gate 接口规范 | 中 |
| LLM Retrieval Filter | LLM 对 Top-K 做 rerank | 提供 relevance label 数据 | 中 |

**预期效果**: 解决"记忆沉默"问题，减少 stable_regime 的 false positive 修正

### 第三阶段：深层融合（3-4 周）

**目标**: DL Encoder 替代哈希 + 双路径融合

| 任务 | 老系统改动 | 新系统改动 | 风险 |
|------|-----------|-----------|------|
| DL Encoder 训练 | 用新系统的 GRU/Transformer encoder 替代 `_hash_embedding` | 提供预训练 encoder 权重 | 高 |
| 双路径检索融合 | LLM Path + DL Path → 合并结果 | 提供融合协议 | 高 |
| Harm Control 迁移 | 经验注入后应用 quality/alignment/cap 门控 | 提取为独立模块 | 中 |

**预期效果**: 检索精度质变，记忆修正方向可控

---

## 五、风险与注意事项

### 5.1 架构风险

1. **复杂度爆炸**: 两系统融合后总代码量可能超过 15,000 行，维护成本显著增加。建议通过明确的接口抽象（如 `MemoryBankInterface`, `KGProvider`, `GateController`）控制耦合。

2. **LLM + DL 双引擎延迟**: LLM 推理（Ollama 调用）和 DL forward pass 的延迟特征完全不同。LLM 可能耗时数秒（尤其是 32B 模型），而 DL forward 是毫秒级。在实时场景中需要异步处理。

3. **训练 vs 推理的不一致**: 新系统的 DL 组件需要训练，而老系统的 LLM 组件是零样本推理。融合后可能面临"一部分可学习、一部分不可学习"的尴尬。

### 5.2 数据风险

1. **数据源不统一**: 老系统使用 `ai_clinician_dataset.csv`（MIMIC-III 衍生），新系统使用 eICU Sepsis-3 处理后数据。两个数据集的患者分布、特征空间可能不同。

2. **KG 映射覆盖度**: 新系统的 KG mapping rules 基于 eICU 列名设计。如果老系统数据集的特征名不同，需要重新做映射。

### 5.3 验证风险

1. **没有统一的评估基准**: 老系统的评估依赖 LLM 评估器（置信度评分），新系统的评估基于 ground truth 对比（MAE/RMSE）。融合后需要新的评估体系。

2. **Ablation 复杂度**: 融合后的成分太多（KG features + pattern memory + trajectory memory + experience memory + prototype + transition + LLM gate + LLM filter），做 ablation study 需要大量实验。

---

## 六、结论

**两套系统的融合是可行的，且有明确的分层路径：**

1. **KG 层融合（优先级最高）**: 新系统的 `KnowledgeGraphFeatureBuilder` + eICU→KG 映射表可以直接引入老系统，为经验检索增加医学本体约束。改动小、风险低、收益明确。

2. **Memory Bank 层融合（优先级中）**: 新系统的 `PersistentExperienceStore`（split-safe 持久化）和 Harm Control（三层门控）可以直接引入，但 encoding 层的统一需要更长时间。

3. **LLM Clinical Supervisor（优先级中高）**: 新系统文档中提出的 LLM 决策方案恰好可以利用老系统的 DSPy + Ollama 基础设施，是两系统最自然的结合点——用 LLM 做临床语义判断，用 DL 做数值预测。

4. **预测层融合（优先级低）**: 短期内不建议。两系统的预测机制本质不同（LLM 生成 vs DL 回归），强行融合得不偿失。

**建议的第一步行列**：将 KG Feature Builder 和 PersistentExperienceStore 提取为独立模块，在老系统的 `experience_integration.py` 中集成，形成一个"KG 增强 + 持久化 + LLM 推理"的混合经验库。

---

> 分析基于以下文件：`docs/00_ROADMAP.md`, `docs/01_DIAGNOSIS.md`, `docs/02_USER_ANALYSIS.md`, `docs/03_LLM_AGENT_PLAN.md`, `src/memory_manager.py`, `src/kg_integration.py`, `src/memory_components.py`, `src/persistent_memory_store.py`, `src/manifold_memory.py`, `src/manifold_forecasting_trainer.py`, `advanced_experience_memory.py`, `experience_integration.py`, `experience_knowledge_base.py`
