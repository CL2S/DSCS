# LLM Clinical Supervisor — 智能体增强经验记忆系统方案

> 创建：2026-05-13 | 基于 Phase A 实验结果与 7 项硬伤诊断

---

## 一、背景与问题定位

### 当前瓶颈

经过 R0-S2 七轮修改 + Phase A 分支解耦/不确定性/诊断摘要实验，系统已触及当前纯数值架构的性能天花板：

| 硬伤 | 状态 | 纯数值方案能否解决 |
|---|---|---|
| #1 训练随机性主导性能 | 未解决 | 否 — SGD 优化本质问题 |
| #2 无"记忆沉默"机制（stable_regime 77% harmed） | 未解决 | **否** — embedding 无法表达"不需要修正" |
| #3 Transition utility 信号不可靠（符号随机翻转） | 未解决 | **否** — 启发式公式与编码空间脱节 |
| #4 持久化经验放大噪声（pre_strength 暴涨 696x） | 未解决 | 部分 — 可加规则筛选但语义判断弱 |
| #5 检索与预测价值脱节 | 未解决 | 部分 — 可改 loss 但方向对齐难 |
| #6 Harm control 只管幅度不管方向 | 未解决 | 否 — 方向判断需语义理解 |
| #7 中等严重度系统性受损 | 未解决 | 部分 — 分层处理但无法区分个体 |

### 为什么 LLM 适合这个场景

当前系统的根本矛盾：**所有决策由连续向量空间中的数值运算完成，而临床判断本质上是离散的、语义的、基于规则的推理。**

一个 embedding 无法判断"这个 SOFA=4 的稳定患者不需要 memory 修正"——embedding 里没有"稳定"这个概念，只有一个 400 维向量中某个维度偏大或偏小。

LLM 恰好补这个缺口：**把需要临床常识判断的决策从数值管道里抽出来，交给能做语义推理的模块。** LLM 不做数值预测，只做临床语义判断——这正是 LLM 的强项。

---

## 二、架构设计：LLM Clinical Supervisor

### 核心原则

1. **LLM 不做数值预测。** SOFA 时序预测仍由深度学习模型完成。
2. **LLM 不做高频调用。** 仅在关键决策点调用，而非每个 forward pass。
3. **LLM 的输入是结构化临床摘要，不是原始时序数据。** 使用已有的 Clinical State Signature。
4. **LLM 的输出必须是结构化 JSON，temperature=0。**
5. **LLM 定位为"辅助筛选/判断"，不替代医生决策。**

### 架构图

```
                         ┌──────────────────────────┐
                         │     LLM Supervisor        │
                         │  (GPT-4o-mini / 开源模型)  │
                         │  temperature=0            │
                         │  JSON 输出                 │
                         └──────┬───────────────────┘
                                │ 只做三件事：
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
   ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
   │ Gate: 该不该修  │  │ Filter: 检索对不│  │ Utility: 效用好│
   │ 记忆？         │  │ 对？           │  │ 不好？         │
   │                │  │                │  │                │
   │ 替代硬伤 #2    │  │ 缓解硬伤 #4    │  │ 替代硬伤 #3    │
   └───────┬────────┘  └───────┬────────┘  └───────┬────────┘
           │                   │                   │
           ▼                   ▼                   ▼
    Memory Residual      Prototype Pool      Transition Score
    是否应用？            哪些可信？           效用是正还是负？
```

---

## 三、三个 LLM 决策点

### 决策点 1：Memory Gate（替代硬伤 #2）

**目标：** 让 LLM 判断当前患者是否需要经验修正。

**输入：**
```json
{
  "patient": {
    "stay_id": "stay_143157",
    "clinical_state": {
      "severity": "moderate",
      "level": "near_baseline",
      "trend": "stable",
      "volatility": "quiet",
      "trajectory": "stable_regime",
      "kg_flags": ["state_sepsis"]
    },
    "sofa_current": 5.2,
    "sofa_trend_6h": -0.3,
    "baseline_mae_history": 1.85
  },
  "memory_context": {
    "prototype_count": 3,
    "top_prototype_future_direction": "flat",
    "template_confidence": 0.62,
    "memory_quality": 0.71
  }
}
```

**LLM 任务：** 判断该患者的经验修正是否应该应用。

**输出：**
```json
{
  "gate": 0.3,
  "reasoning": "患者SOFA稳定在5.2附近，6小时趋势为-0.3（轻度改善），处于stable_regime。检索到的prototype未来方向为flat且置信度中等(0.62)。当前状态不需要大幅修正，建议gate=0.3仅保留少量记忆信号。",
  "confidence": 0.85
}
```

**gate 取值语义：** `1.0`（全量应用）`0.7`（大部分应用）`0.3`（少量保留）`0.0`（关闭记忆）。

**预期收益：** stable_regime harmed rate 从 77% 显著下降。LLM 能识别"稳定患者不需要修正"这一临床常识。

---

### 决策点 2：Retrieval Filter（缓解硬伤 #4）

**目标：** 从 top-k prototype 中筛选临床真正相关的。

**输入：**
```json
{
  "query_patient": {
    "clinical_state": { "severity": "high", "trend": "worsening", "kg_flags": ["state_septic_shock", "state_high_lactate"] },
    "sofa_current": 11.5,
    "intervention_summary": "vasopressor_on, antibiotic_covered"
  },
  "retrieved_prototypes": [
    {
      "prototype_id": "proto_a1",
      "future_direction": "mostly_up",
      "outcome_entropy": 0.45,
      "similarity_score": 0.72,
      "member_sofa_range": [8.0, 14.0],
      "kg_overlap": ["state_sepsis"]
    },
    {
      "prototype_id": "proto_b2", 
      "future_direction": "mostly_down",
      "outcome_entropy": 0.22,
      "similarity_score": 0.68,
      "member_sofa_range": [9.0, 13.0],
      "kg_overlap": ["state_septic_shock", "state_high_lactate"]
    }
  ]
}
```

**LLM 任务：** 判断哪些 prototype 在临床上与当前患者真正相关。

**输出：**
```json
{
  "filtered_prototypes": ["proto_b2"],
  "excluded": ["proto_a1"],
  "reasoning": "proto_b2与当前患者共享septic_shock和high_lactate两个关键KG标志，且outcome_entropy更低(0.22 vs 0.45)，临床可比性更强。proto_a1虽然形态相似度更高，但KG重叠少且未来方向相反。",
  "confidence": 0.82
}
```

**预期收益：** 过滤掉临床不相关的 prototype，减少持久化经验注入的噪声，降低 `memory_harm_pre_strength`。

---

### 决策点 3：Transition Utility（替代硬伤 #3）

**目标：** 用 LLM 的临床常识替代启发式 utility 公式。

**输入：**
```json
{
  "state_t": {
    "sofa": 9.0,
    "lactate": 3.8,
    "map": 58,
    "vasopressor": true,
    "kg_flags": ["state_septic_shock", "state_high_lactate"]
  },
  "action": {
    "type": "antibiotic_escalation",
    "timing": "within_3h",
    "vasopressor_adjustment": "maintain"
  },
  "state_t_plus_1": {
    "sofa": 7.5,
    "lactate": 2.1,
    "map": 68,
    "vasopressor": true
  },
  "delta": {
    "sofa_change": -1.5,
    "lactate_change": -1.7,
    "map_change": +10
  }
}
```

**LLM 任务：** 判断这个状态转移是否有正向临床意义。

**输出：**
```json
{
  "utility": 0.72,
  "reasoning": "SOFA下降1.5分，乳酸从3.8降至2.1（清除率良好），MAP从58回升至68。虽然升压药未撤除，但整体器官功能趋势改善明确。utility=0.72表示中等偏正向的临床改善。",
  "limitations": "无法判断改善是否因果于抗生素升级，可能受其他治疗或自身恢复影响",
  "confidence": 0.78
}
```

**预期收益：** utility 符号不再受训练随机波动支配。LLM 基于临床常识判断，符号稳定。

---

## 四、实施路线

### Phase L1：Retrieval Filter（最低风险，1周）

**为什么先做：** 
- 不影响预测管道核心逻辑
- 只改变 memory bank 的输入
- 失败也仅是退回原始检索结果

**修改：**
1. 新增 `src/llm_supervisor.py`：LLM 调用封装
2. 在 `memory_manager._semantic_hits()` 之后插入 LLM filter
3. CLI `--enable-llm-filter`，`--llm-model`（默认 gpt-4o-mini）
4. 验证：filter 后 prototype 的 outcome_entropy 是否降低，memory quality 是否提升

### Phase L2：Memory Gate（最大收益，1-2周）

**为什么第二：** 
- 直接攻击硬伤 #2（stable_regime 77% harmed）
- 需要前面 Retrieval Filter 的 LLM 基础设施

**修改：**
1. 在 `_apply_memory_harm_control` 之前插入 LLM gate 判断
2. LLM gate 与 numeric harm control 做 min() 融合
3. 验证：stable_regime harmed rate 是否显著下降

### Phase L3：Transition Utility（依赖前两者，1周）

**修改：**
1. 替代 `_transition_utility_from_sample` 中的启发式公式
2. 缓存 LLM utility 判断以减少调用
3. 验证：trans_utility 符号是否不再随机翻转

### 实验设计

每阶段三组对比：
- **G1（baseline）：** 纯数值（当前最佳配置）
- **G2（LLM ON）：** 启用 LLM 决策点
- **G3（LLM OFF，等价规则）：** 用等价的手工规则替代 LLM 判断（消融，证明 LLM 贡献不是来自简单规则）

---

## 五、模型选择与部署

### 推荐模型

| 模型 | 适用场景 | 成本 |
|---|---|---|
| GPT-4o-mini | 主实验，成本可控，临床常识足够 | ~$0.15/1M input tokens |
| GPT-4o | 最终验证，最高质量 | ~$2.50/1M input tokens |
| 本地 Qwen/DeepSeek | 离线实验，零成本 | 需 GPU |

### 调用频率与成本估算

512 病例 × 3 决策点 = 1536 次 LLM 调用。每次输入约 500 tokens，输出约 200 tokens。

| 模型 | 总成本 |
|---|---|
| GPT-4o-mini | ~$0.20 |
| GPT-4o | ~$3.50 |
| 本地开源 | $0 |

### 延迟考量

- 每次调用约 0.5-2s（取决于 API 延迟）
- 512 × 3 × 1s ≈ 25 分钟额外实验时间
- 可通过批处理 + 缓存减少

---

## 六、风险与缓解

| 风险 | 缓解 |
|---|---|
| LLM 输出不一致 | temperature=0，要求 JSON 格式，加 retry |
| LLM 医学知识不足 | 在 prompt 中嵌入脓毒症 SOFA 评分标准、KG 知识 |
| Gate 判断过于保守/激进 | 通过 gate 取值的 helped/harmed 分析来校准 prompt |
| API 调用失败 | fallback 到当前数值 gate，不影响系统运行 |
| 论文中是否可发表 | 定位为"LLM 辅助临床语义门控"，不声称 LLM 做医学决策 |
| 开源可复现性 | 记录 prompt 模板 + temperature=0 + 固定模型版本 = 可复现 |

---

## 七、与现有架构的关系

LLM Supervisor 是**外挂模块**，不改变现有深度学习管道：

```
现有管道（不变）:
  Encoder → Base Predictor → Memory Residual → Harm Control → Transition Residual → Fusion

LLM Supervisor（新增，三个插入点）:
  └─→ Retrieval Filter (插入在 semantic retrieval 之后)
  └─→ Memory Gate (插入在 harm control 之前)  
  └─→ Transition Utility (替换启发式 utility 公式)
```

这样设计的好处：
- 可以随时通过 CLI flag 开关，做 AB 测试
- LLM 故障不影响基础预测（fallback 到数值管道）
- 论文中可以清晰分离"深度学习组件"和"LLM 辅助决策组件"的贡献

---

## 八、预期指标体系

### 验证 LLM 有效性的关键指标

| 指标 | 当前值 | Phase L2 目标 | 测量方式 |
|---|---|---|---|
| stable_regime harmed rate | 77.3% | < 40% | memory_gain_audit |
| overall harmed rate | 47.3% | < 35% | memory_gain_audit |
| trans_utility 符号跨 run 一致性 | 随机翻转 | 同符号 | 多次运行比较 |
| memory_harm_pre_strength | 40.68 | < 15 | memory_diagnostics |
| LLM gate=0 的病例中 harmed rate | N/A | < 20% | 新增 audit |

### 论文贡献表述

> "We propose an LLM-based Clinical Supervisor that augments a deep learning memory-augmented forecasting system. The LLM performs three semantically-grounded clinical judgments — whether to apply experience-based correction, which retrieved prototypes are clinically relevant, and how to assess state-transition utility — effectively addressing the representation gap between continuous embedding spaces and discrete clinical reasoning. Experiments show that LLM supervision reduces harmful memory interventions by X% while maintaining or improving overall forecasting accuracy."

---

## 九、相关文件

- 硬伤诊断：[01_DIAGNOSIS.md](01_DIAGNOSIS.md)
- 修改路线图：[00_ROADMAP.md](00_ROADMAP.md)
- 方法缺陷分析：[02_USER_ANALYSIS.md](02_USER_ANALYSIS.md)
- Phase A 结果：`output/formal/phase_a/`
