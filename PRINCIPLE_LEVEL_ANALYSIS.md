# 融合方法的大模型底层原理分析

## 核心问题

> 基于 LLM 特征提取 → 随机投影 → 流形记忆库检索 → LLM 预测 的融合架构，能否在底层原理层面真正提升大模型的预测性能？如果能，机制是什么？如果不能，如何改进？

---

## 一、先把架构拆到最底层

当前融合系统的数据流用矩阵形式写出来：

```
Step 1: LLM 提取特征
    h_llm = LLM(patient_text)              # 自然语言 → JSON 特征

Step 2: 拼接特征向量
    x = concat([formation(16), sofa(6), vitals(10), intervention(5), static(56)])
       = 93-dim vector

Step 3: 投影到流形空间
    context = W₂ · GELU( W₁ · x )         # W₁: 93×128, W₂: 128×256
    q = normalize( W_q · context )         # W_q: 256×32
    k = normalize( W_k · context )         # W_k: 256×32
    v = W_v · context                      # W_v: 256×48

Step 4: 记忆检索
    scores = softmax( q · K_bank^T / τ )   # K_bank: N×32, 存储的历史病例 keys
    residual = Σ scores_i · V_bank_i       # 注意力加权聚合

Step 5: LLM 预测 (将检索结果作为 prompt context)
    prediction = LLM(patient_text | memory_context)
```

**关键问题藏在哪里？** 看 Step 3 的 W 矩阵——它们是**随机初始化的，从未被训练过**。

---

## 二、为什么这很可能不 work（底层分析）

### 2.1 投影矩阵是随机的 → 检索是随机的 → 记忆增强是噪声

先说清楚 Transformer 内部是怎么做"相似性检索"的：

Transformer 的注意力机制之所以有效，是因为 Q、K、V 的投影矩阵经历了**数万亿 token 的梯度更新**，使得：
- 语义相近的 token 经过 K 投影后，在 32-128 维空间中距离近
- Q 投影学会了"问正确的问题"（query 应该关注哪些 key）

当前融合系统的投影网络（LLMFeatureProjector）从未被训练。W_q、W_k、W_v 是随机矩阵。这意味着：

$$q_{patient\_A} = W_q^{random} \cdot context_{patient\_A}$$
$$k_{patient\_B} = W_k^{random} \cdot context_{patient\_B}$$

即使 LLM 正确提取了临床特征，经过随机投影后：
- **两个临床相似的患者的 q 和 k 的余弦相似度 ≈ 0**（高维随机向量的内积期望为 0）
- **记忆库的检索结果本质上是在做随机选择**

这就像你有一个完美的图书分类系统（LLM 提取的特征），但图书馆管理员（投影矩阵）是瞎子，把每本书随机塞到架子上。你问他"找一本和这本类似的书"，他只能乱指。

### 2.2 LLM 自身已经在做"记忆检索"——但方式完全不同

GPT 类模型内部有两层"记忆"：

**参数化记忆（Weights）**：
$$P(token_{t+1} | token_{1:t}) = \text{softmax}(W_{lm\_head} \cdot \text{Transformer}(token_{1:t}))$$

模型权重 W 中编码了训练数据中的统计规律。对 deepseek-r1:32b 来说，它的 32B 参数本身就是一种巨大的经验库——它在预训练时见过的所有 ICU 病历文本都以压缩形式存储在这些参数中。

**上下文内记忆（Attention KV Cache）**：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

自回归生成时，每个新 token 可以 attend 到之前的所有 token。prompt 中的信息通过 attention 机制直接参与预测。

这意味着：**当你把记忆检索结果放进 prompt 时，LLM 自己的 attention 机制已经在做"这个历史病例和当前患者的相关性判断"**——它用的是经过预训练和 RLHF 优化的 Q/K/V 投影，而不是随机矩阵。

换句话说：**LLM 内建的 attention 比外部随机投影的记忆库检索质量高几个数量级。**

### 2.3 关键瓶颈：梯度无法回传到 LLM

```
LLM → 特征提取 → 投影网络 → 记忆库 → prompt → LLM 预测
 ↑                                                    |
 └──────────────── 梯度在此断裂 ──────────────────────┘
```

整个流程中，**没有任何梯度信号可以流回 LLM 或投影网络**。这意味着：

- LLM 不知道自己的特征提取对这个任务好不好
- 投影网络不知道自己的投影对检索质量好不好
- 记忆库写入的新经验是基于"未经验证"的投影的

对比端到端训练的系统：
$$\mathcal{L} = \text{MSE}(y_{true}, \hat{y})$$
$$\frac{\partial \mathcal{L}}{\partial W_{encoder}} \text{ 和 } \frac{\partial \mathcal{L}}{\partial W_{projector}} \text{ 可以回传}$$

梯度信号驱动 encoder 学习"什么样的编码能让检索更准、预测更好"，而融合系统完全缺失这个信号。

### 2.4 Prompt 中的"记忆增强"可能是一种干扰

LLM 的 prompt 已经有了非常丰富的上下文（患者描述、临床状态、KG 特征）。加入记忆检索结果后，这些新增的 token 会：

1. **稀释有效信息密度**：attention 需要分散到更多 token 上
2. **引入潜在错误**：如果检索到的历史病例不相关或结果错误，LLM 可能被误导
3. **激活错误的先验**：LLM 看到"类似病例"的信息后，可能过度依赖这个先验而忽略当前患者的具体特征（anchoring bias）

RAG 领域的经典发现：**检索质量是 RAG 性能的上限**。如果检索准确率低于 60-70%，RAG 反而会损害性能——因为模型需要额外消耗能力来判断"这个检索结果到底能不能信"。

---

## 三、但是，有没有仍然 work 的可能？

有。以下是几种理论上的有效机制：

### 3.1 多模型 Ensemble 确实 work（但不是因为记忆库）

$$\hat{y}_{ensemble} = \frac{1}{3}(\hat{y}_{deepseek} + \hat{y}_{gemma} + \hat{y}_{qwen})$$

Ensemble 降低方差的原理是纯统计学的，不依赖记忆库：

$$\text{Var}(\hat{y}_{ensemble}) = \frac{1}{9}\sum_{m}\text{Var}(\hat{y}_m) + \frac{2}{9}\sum_{m<n}\text{Cov}(\hat{y}_m, \hat{y}_n)$$

如果三个模型是不同架构、不同训练数据，它们的预测误差部分独立，那么 ensemble 的方差一定 ≤ 单个模型的平均方差。**这是融合系统最容易获得收益的部分**，但它与记忆库无关。

### 3.2 LLM Clinical Supervisor 的 Gate 决策确实 work（但不依赖记忆检索质量）

Memory Gate 的逻辑是：
```
输入：severity_bin, trend_bin, trajectory_label（这些都是 LLM 从患者描述中直接提取的）
输出：apply_memory: bool
```

这个判断的质量**几乎完全取决于 LLM 从患者文本中提取的临床状态签名的准确性**，与记忆检索的质量无关。它本质上是一个基于规则的临床判断（"stable_regime 不需要修正"），恰好 LLM 擅长做这种判断。

**但如果检索本身是随机的，Gate 说"可以应用"时，应用的也是随机噪声。**

### 3.3 可能性：当记忆库累积了足够多的高质量经验后，检索开始变得有用

即使投影是随机的，如果记忆库中积累了足够多的高质量经验条目，**随机检索碰巧命中有用信息的概率也会增加**。就像在一个足够大的图书馆里，即使管理员乱放书，你随便抽一本也可能恰好有用。

但问题是：**写入记忆库的经验本身是基于不可靠的检索和未验证的预测的**。如果初始检索质量低，写入的经验质量也低 → 恶性循环。

---

## 四、如何从根本上改进？

### 4.1 核心洞察：投影网络必须被训练

当前架构最致命的缺陷是：**LLM 提取的丰富语义信息，在经过随机投影后丢失了**。解决方案分三个层次：

---

### 方案 A：用 Contrastive Learning 训练投影网络（推荐，改动最小）

**原理**：给投影网络一个训练目标——"临床结局相似的患者，在流形空间中距离近"。

**需要准备的数据**：

```
患者对 (A, B) + 标签：
  - 正例 (label=1)：A 和 B 的临床结局相似
    · SOFA 轨迹相似（8h 内同方向、同幅度变化）
    · 干预效果相似（同样干预产生类似效果）
    · 同一 trajectory label 且 pattern label 相同

  - 负例 (label=0)：A 和 B 的临床结局不同
    · SOFA 轨迹方向相反
    · 不同 trajectory label
    · 同一 SOFA 值但临床走向相反
```

**数据来源**：直接从 `ai_clinician_dataset.csv`（老系统）或 `eicu_sepsis3_labels.csv`（新系统）中构建。约 20,000 个 stay，可以构造数百万个患者对。

**训练过程**：

```python
# Contrastive Loss (InfoNCE)
L = -log( exp(sim(q_A, k_pos) / τ) / 
          Σ_{neg} exp(sim(q_A, k_neg) / τ) )

# 其中 sim 是余弦相似度，q_A 是患者 A 的 query 向量，
# k_pos 是正例患者 B 的 key 向量, k_neg 是负例患者的 key 向量

# 梯度回传：
∂L/∂W_q → 更新 W_q
∂L/∂W_k → 更新 W_k
∂L/∂W_projector → 更新整个投影网络
```

训练后，两个临床相似的患者在流形空间中的 q 和 k 会有高余弦相似度，检索变得有意义。

**需要准备的**：
- 数据：构造患者对的标注脚本（从现有数据自动构建，约 200 行 Python）
- 计算：GPU（单卡 A100 或消费级 4090，训练几小时）
- 不需要 GPU 也能跑（CPU 上 93×128 的 MLP 训练很快）

---

### 方案 B：用 LLM 内部 Embedding 替代随机投影（改动最小，效果可能最好）

**原理**：如果 Ollama 模型能输出 token-level 或 sequence-level 的 hidden states（即 Transformer 最后一层的输出向量），直接用这些 embedding 经过一个小型适配器（adapter）映射到流形空间。

```python
# 不用 LLM 提取 JSON 特征再随机投影
# 而是直接用 LLM 的 internal representation

llm_hidden = ollama.embed(patient_text)     # shape: [seq_len, hidden_dim]
                                            # deepseek-r1:32b: hidden_dim=5120
                                            # gemma3:12b: hidden_dim=3584

# 轻量 adapter (可训练)
context = Adapter(llm_hidden.mean(dim=0))    # [hidden_dim] → [256]
q = W_q · context                            # [256] → [32]
k = W_k · context
v = W_v · context
```

**为什么这比方案 A 好**：
- LLM 的 hidden states 已经编码了极其丰富的语义信息（5120 维 vs 当前 93 维）
- 预训练过程已经将这些 hidden states 组织成了高度结构化的语义空间
- Adapter 只需要做一个简单的映射（语义空间 → 流形空间），而不是从零学习编码

**需要准备的**：
- Ollama 的 embedding API 支持（`ollama embed` 命令或 `/api/embed` 端点）
- 如果 Ollama 不直接支持，可以用 text-embedding 模型（如 bge-large, stella）替代
- Adapter 训练（方案 A 的 contrastive loss，但输入从 93 维变成 embedding 维度）

---

### 方案 C：消融验证——先去掉记忆库，只保留 Ensemble + Supervisor（最有说服力）

**原理**：如果分析正确，去掉随机投影的记忆库后性能不降反升，这将是最有力的证据。

设置 4 个配置做对比实验：

```
Config 1 (Baseline):      纯 LLM 单模型预测（原始系统 1）
Config 2 (Ensemble only): 3 模型 ensemble，无记忆库，无 Supervisor
Config 3 (+Supervisor):   Ensemble + LLM Clinical Supervisor (Gate only)，无记忆库
Config 4 (Full Fusion):   当前完整融合系统（Ensemble + Memory + Supervisor）

预测的排序（按原理分析）:
Config 2 > Config 1        (Ensemble 降低方差，确定性改进)
Config 3 ≈ Config 2        (Gate 在没有可靠记忆信号时默认为"不应用"，等于没加)
Config 4 ? Config 2        (如果投影随机，记忆增强是噪声，Config 4 < Config 2)
```

**如果 Config 4 < Config 2，直接证明了随机投影是瓶颈。**
**如果 Config 4 > Config 2，说明有我没分析到的有效机制，值得深入研究。**

---

### 方案 D：用 LLM 本身作为检索器（最纯粹，但最慢）

**原理**：完全弃用投影网络和流形记忆库的向量检索，改用 LLM 本身做相似性判断。

```python
def llm_retrieve(patient_current, memory_bank):
    """让 LLM 从记忆库中选最相关的历史病例"""
    candidates = memory_bank.sample(20)  # 粗筛 20 个
    
    prompt = f"""
    当前患者: {patient_current}
    
    以下是一些历史病例，请选出与当前患者临床最相似的 3 个：
    {format_candidates(candidates)}
    
    输出 JSON: {{"relevant_indices": [0, 5, 12], "relevance_scores": [0.9, 0.7, 0.5]}}
    """
    
    return llm_call(prompt)
```

**优势**：LLM 的语义理解能力得到完整保留，不需要任何投影网络训练。
**劣势**：每次检索需要一次 LLM 推理（几秒到几十秒），不适合高频调用。

**适用场景**：当前的预测流程每次只需要检索一次，延迟增加 3-5 秒在批量处理场景下完全可接受。

---

## 五、你需要准备什么？

### 5.1 立即可以做的（本周）

**1. 消融实验脚本**（验证分析是否正确）

```python
# 运行 4 个配置的对比实验
python run_ablation.py --mode compare \
    --data icu_stays_descriptions88.json \
    --output ablation_results/
```

这会输出 Config 1-4 的 MAE、Helped Rate、Harmed Rate 对比表。

**2. 患者对标注脚本**（为方案 A 准备训练数据）

从现有数据中自动提取：
- 同一 stay 内不同时间窗 → 天然正例（同一患者的不同时间点）
- 同一 trajectory label + 同一 pattern label → 强正例
- 不同 trajectory label → 天然负例

### 5.2 短期需要准备的（1-2 周）

**1. GPU 环境**（方案 A/B 的 projector/adapter 训练）
- 最低：CPU only（93×128 MLP 很快）
- 推荐：单卡 GPU（A100 或消费级 4090，4090 足够）
- 如果方案 B，需要能跑 Ollama embedding

**2. Ollama Embedding API 调研**
- 验证 `ollama embed` 或 `/api/embed` 是否可用
- 测试是否能从 deepseek-r1, gemma3, qwen3 中提取 hidden states

**3. 评估指标体系确认**
- 与医生确认 SOFA MAE 的临床可接受范围（通常 < 1.5 分）
- Stable regime harmed rate 的临床容忍上限（建议 < 25%）

### 5.3 中期需要准备的（2-4 周）

**1. 训练基础设施**
- Contrastive learning 训练脚本
- 验证集构造（确保与训练 split 隔离）
- 早停和 checkpoint 机制

**2. 医生协作**
- 盲评实验的 20 个案例准备
- 评分表和评估标准设计
- 伦理审查（如果需要）

---

## 六、结论

**从底层原理看，当前融合架构最大的问题是随机投影破坏了 LLM 提取的语义信息。** LLM 将患者文本编码为丰富的临床特征，但这些特征经过未训练的随机矩阵投影后丧失了语义结构，导致记忆检索变成随机选择，记忆增强变成噪声注入。

**好消息是**，这个问题有三个明确的改进路径，按优先级排序：

| 优先级 | 方案 | 改动量 | 有效性 | 需要什么 |
|--------|------|--------|--------|---------|
| P0 | 消融实验验证假设 | 100 行脚本 | 确定问题 | 时间 |
| P1 | Contrastive 训练投影网络 | 300 行训练代码 | ★★★★ | 标注数据（自动构建）|
| P2 | LLM Embedding 替代随机投影 | 200 行适配代码 | ★★★★★ | Ollama embedding API |
| P3 | LLM 直接检索（完全弃用投影） | 200 行 prompt 模板 | ★★★★★ | 容忍 3-5s 额外延迟 |

**如果只做一件事**：先跑消融实验（Config 1-4），验证"去掉记忆库后性能不降反升"的假设。如果验证通过，就明确了改进方向——要么训练投影网络（P1），要么改用 embedding（P2），要么直接用 LLM 检索（P3）。

**如果消融实验显示记忆库确实有增益**：说明有我没分析到的有效机制（可能是多路径 planner 权重的隐式 ensemble 效应，或者 Semantic Prototype Store 的方向对齐机制），需要进一步分析 gain 的来源。
