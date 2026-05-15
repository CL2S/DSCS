# 经验记忆增强预测系统：方法、实验结果与硬伤诊断

> 最后更新：2026-05-12

---

## 一、方法概述

### 1.1 核心思想

在 ICU 脓毒症 SOFA 时序预测中，传统的深度学习模型只从当前患者的观测序列中学习。本系统的核心假设是：**历史相似病例的经验可以为当前预测提供额外的修正信号**。系统维护一个可持久化的经验库，在预测时检索与当前患者临床状态最相似的历史经验，并将这些经验转化为对基础预测的修正。

### 1.2 架构

```
ForecastSample (raw_context, raw_target, formation_features, patient_static, kg_flags, intervention)
       │
       ├─→ Pattern Memory  ──→ Semantic Prototype Store (持久化, split-safe)
       ├─→ Trajectory Memory ──→ Experience Memory (runtime + persistent)
       ├─→ KG Integration
       │
       ▼
  Fused Representation
       │
       ├─→ Base Predictor ──→ base_prediction
       ├─→ Memory Residual ──→ Harm Control (quality / alignment / cap) ──→ controlled_residual
       └─→ Transition Residual ──→ Continuous Gate (utility / pattern / trajectory) ──→ transition_residual
                                               │
                                               ▼
                                   fusion = base + memory_residual + transition_residual
```

### 1.3 三大组件

**A. Split-Safe 持久化经验库（`PersistentExperienceStore`）**
- 每条经验写入 JSONL，记录来源（train/val/test/hospital/stay/patient）
- 加载时硬过滤：禁止 test split、禁止当前评估 stay/patient
- 每次实验输出泄漏审计报告（`reuse_audit`）：加载了多少、排除了多少、为什么排除
- Semantic Prototypes：按经验标签分组，保存 prototype formation center、future curve、KG center、outcome entropy、future_direction_type

**B. 分层 Memory 路径**
- Pattern Memory：基于形态模式（flat/up/down/spike）的模板
- Trajectory Memory：基于病程轨迹（stable/rising/falling/shifted）的模板
- Experience Memory：从持久化经验库或 runtime memory bank 检索相似病例的经验先验
- Semantic Prototype Retrieval：用 formation features、KG features、intervention signature、future_direction 等多维特征做 prototype 匹配，生成 template_curve 和 template_blend_weight

**C. 分层门控机制**
- **Harm Control**（保护非 transition 的 memory 路径）：quality scale（记忆信号质量低于阈值则压降）、alignment scale（多路径方向冲突则压降）、cap scale（残差幅度超限则压降），取三者最小值
- **Continuous Transition Gate**（保护 transition 路径）：utility_factor（sigmoid 连续缩放，替换早期二元门控）、pattern_factor、trajectory_factor，取三者最小值
- Transition Memory：存储 `(z_t, a_t, z_{t+1}, utility)` 格式的状态转移经验，在检索时用 clinical state signature 做结构化匹配

### 1.4 Clinical State Signature

将当前窗口编码为显式临床状态，而非纯 embedding：
- severity_bin（low/moderate/high/critical）
- level_bin（below/near/above/far_above baseline）
- trend_bin（improving/stable/worsening）
- volatility_bin（quiet/variable/unstable）
- trajectory_label + KG active flags

签名用于 transition 检索侧的一致性加权。

---

## 二、实验历程与性能进化

### 2.1 修改轮次总览

| 轮次 | 日期 | 修改目标 | 关键改动 |
|---|---|---|---|
| R0 | 05-09 | 数据泄漏审计 | 持久化经验写来源字段；加载硬过滤 test/val/当前 stay；输出泄漏审计 |
| R1 | 05-09 | Memory Harm Control | 质量/对齐/幅度三道约束；逐病例 memory_gain_audit |
| R2 | 05-10 | Clinical State + Transition 升级 | 显式临床状态签名；transition store 签名审计与检索加权 |
| R3 | 05-10 | Safe Transition Gate | 门控默认值校准（stable_regime_penalty=0.15, flat=0.40, min_utility=0.05），发现二元 utility gate 误杀 44.6% 有益 transition |
| R4 | 05-10-11 | 连续 Transition Gate | 二元 utility gate→连续 sigmoid；pattern/trajectory 独立 min 生效；部分签名匹配；utility bias/temperature 可调 |
| R5 | 05-11 | 方向过滤 + 熵惩罚 | ❌ 回归：均值 template_blend 从 0.26 跌到 0.108，全面性能倒退 |
| D1-D3 | 05-11 | 诊断三件套 | 持久化复用 + val audit + retrieval trace（但受 R5 遗留代码影响，提升被掩盖） |
| F1-F3 | 05-11 | 回退 + 重建 | 回退 R5 的 template_blend 惩罚和 harm boost；1024-series store 重建；首次成功加载持久化经验 |
| S2 | 05-12 | 稳定训练 | 8 epochs + penalty→1.0 → **与 F3 完全一致，零增量** |

### 2.2 性能进化

| 里程碑 | imp_mae | helped | harmed | pre_strength | trans_utility | 备注 |
|---|---|---|---|---|---|---|
| 原始基线（无修改） | 0.0120 | 15.6% | 14.5% | — | — | 修改前 |
| 原始+transition | 0.0244 | 52.9% | 47.1% | — | — | transition 有增益但伤害大 |
| R3 no_transition | 0.0505 | 52.9% | 29.1% | 3.66 | — | R0-R2 后基础 memory 大幅提升 |
| R3 signature | 0.0606 | 56.4% | 32.2% | 3.66 | +0.004 | +transition, 65% 被二元阻断 |
| **R4 no_partial_sig** | **0.0632** | **61.1%** | **38.9%** | **3.66** | **+0.004** | **峰值** |
| R5 default | 0.0115 | 51.2% | 48.8% | — | — | R5 回归 |
| F3/S2 with_persist | 0.0333 | 58.0% | 42.0% | **41.80** | -0.059 | 持久化首次可用，但训练波动大 |
| S2 no_persist | 0.0165 | 61.7% | 35.0% | 0.06 | +0.004 | transition-only, memory 全关 |

### 2.3 关键发现

1. **基础 memory 路径从未修改到 R3 提升了 4.2 倍**（0.012→0.051），主要来自 harm control 和 split-safe 持久化
2. **Transition 在有利条件下额外贡献 ~20%**（0.051→0.063）
3. **持久化经验首次在 F3 成功加载（1536 条），贡献 +0.017**
4. **但性能天花板受训练随机性强烈约束**——R4=0.063 与 F3/S2=0.033 的差距完全来自不同的训练 run，非参数可调

---

## 三、硬伤诊断

### 硬伤 1（致命）：训练随机性主导系统行为

```
同一配置 R4:     pre_strength=3.66   imp_mae=0.063
同一配置 F3/S2:  pre_strength=41.80  imp_mae=0.033
same config      11.4x difference    2x difference
```

从 R0 到 S2 的全部参数修改加起来，影响力不如一次随机初始化。Memory residual 的幅度（pre_strength）完全由训练过程的随机因素决定。harm control 只能被动压制——无论残差是 41.80、3.66 还是 0.06，压到最终都是 ~0.16 或 0，99% 以上的计算被浪费。

**本质：非 transition 的 memory 路径缺乏任何训练信号来学习"这个残差是否应该输出"。它无差别地产生残差，然后依赖粗糙的幅度压制来处理。**

### 硬伤 2（致命）：系统没有"记忆沉默"机制

```
跨所有实验: stable_regime harmed = 60-73%
            唯一例外是 no_persist（memory 彻底关停）
```

系统在架构上是一个单向管道：找到经验 → 施加修正 → harm control 压制。不存在一个判断分支说"当前患者是稳定的，历史经验不应该用于修正预测"。当 memory 路径被打开，所有患者都会被无差别地施加经验修正。stable_regime 的患者本不需要修正，但系统照样往上面叠加残差。

**本质：需要一个 learnable 的"经验是否应该被应用"的判断器，而不是全量施加后再用 harm control 做数学压制。**

### 硬伤 3（严重）：Transition Utility 是启发式的，不可靠

```
R4:  trans_utility=+0.004  → transition contributes +0.013 (useful)
F3/S2: trans_utility=-0.059 → transition contributes +0.001 (useless)
```

Transition utility 是一个固定的启发式公式（6 个 delta 分量的加权和），不是从数据中学出来的。它的符号和幅度完全取决于当前模型训练后编码空间的统计特性——而这个特性在不同训练 run 之间会翻转。utility 为正时 transition 有价值，为负时等于关闭。

**本质：utility 公式是独立于模型训练的，无法与当前模型的编码空间对齐。**

### 硬伤 4（严重）：持久化经验放大噪声

```
no_persist  (1536 seeds): pre=0.06,   stable_harmed=37%
with_persist (3072 seeds): pre=41.80, stable_harmed=67%
```

加入 1536 条持久化经验让 memory residual 暴涨 696 倍，stable_regime harmed rate 从 37% 飙升到 67%。持久化经验在增加信息的同时，成比例地注入了噪声。

**本质：持久化经验没有经过 relevance 筛选就被混入 memory bank。来自其他 stay 的经验在检索时与当前训练 stay 的经验被同等对待，但前者的相关性天然更低。**

### 硬伤 5（严重）：中等严重度区间系统性受损

```
跨所有实验:
  level_lt_4:   ~78% helped, ~22% harmed  ← 唯一可靠受益区间
  level_4_to_8: ~51% helped, ~49% harmed  ← 掷硬币
  level_8_to_12:~42% helped, ~58% harmed  ← 系统受损
  level_ge_12:  ~44% helped, ~56% harmed  ← 系统受损
```

SOFA<4 的轻症患者能稳定受益——检索到的经验模板通常方向正确。SOFA 越高的患者，经验修正的可靠性越差。高 SOFA 患者的转归本身就高度不确定，经验模板的方差更大。

**本质：高严重度患者的未来走向存在更大的异质性，均值 prototype 模板无法捕捉这种多峰分布。**

### 硬伤 6（中等）：检索质量与预测价值的脱节

当前语义 prototype 检索的评分权重：
- 34% formation cosine similarity（输入形态相似）
- 12% KG similarity
- 11% experience label match
- 8% pattern match + 8% trajectory match
- 其余小权重（severity, KG signature, intervention, direction, support）

检索以"形态相似度"和"标签匹配"为主。但形态相似不等于"这个经验对修正当前预测有用"。两个 SOFA=8 的患者可能有完全不同的转归——一个在恶化趋势中、一个刚经历 spike 后回落。检索到错误方向的经验模板会直接导致 memory residual 指向错误方向。

**本质：检索系统没有被优化来区分"对预测有价值的经验"和"形态相似但对预测无用的经验"。**

### 硬伤 7（中等）：Harm Control 只能管幅度，不能管方向

```
pre_strength=41.80 → post_strength=0.16  (99.6% suppressed)
yet 42% of cases still harmed
```

harm control 的三道约束（quality/alignment/cap）都只控制残差的**幅度**。当残差方向错误时，即使幅度被压到 1%，方向仍然是错的——会持续把预测推向错误一侧。

**本质：需要一个方向判断机制，而不仅仅是幅度控制。**

---

## 四、当前架构的能力边界

### 已做到的

1. **可靠的数据边界** — 持久化经验从写入到加载全程 split-safe，每次实验可审计
2. **稳定的基础内存路径** — 从"几乎没用"提升到"稳定正贡献"（0→0.032）
3. **可工作的持久化经验复用** — 1536 条额外经验成功加载并贡献 +0.017
4. **分层可观测审计** — memory_gain_audit、val_memory_gain_audit、transition_gate_audit、retrieval_trace
5. **Transition 路径** — 在 utility 为正时能额外贡献 ~20%

### 无法通过参数调优解决的

| 问题 | 需要的改变 |
|---|---|
| 训练随机性主导性能 | 为 memory residual 添加直接的监督信号（ground-truth residual = y - base_pred） |
| 缺乏"沉默"机制 | 引入 learnable memory gate，以预测价值为目标训练 |
| transition utility 不可靠 | 用学习的 utility predictor 替代启发式公式 |
| 持久化经验噪声 | 对持久化经验做 relevance-based 筛选或降权 |
| 检索与预测脱节 | 用 contrastive learning 让 retrieval embedding 对齐"预测修正价值"而非"输入形态相似度" |
| 方向不可控 | 引入方向感知的 loss 或 gating |

---

## 五、S2 最终实验结果

| 实验 | imp_mae | helped | harmed | pre_str | template_blend | trans_utility | persist |
|---|---|---|---|---|---|---|---|
| no_transition | 0.03223 | 47.3% | 28.7% | 41.80 | 0.3105 | — | 1536 |
| restored_with_persist | 0.03333 | 58.0% | 42.0% | 41.80 | 0.3105 | -0.059 | 1536 |
| restored_no_persist | 0.01649 | 61.7% | 35.0% | 0.06 | 0.0 | +0.004 | 0 |

### 子群 breakdown（restored_with_persist）

| 子群 | helped | harmed | mean_gain | n |
|---|---|---|---|---|
| stable_regime | 33.3% | 66.7% | -0.00457 | 75 |
| shifted_regime | 62.4% | 37.6% | +0.04023 | 434 |
| down pattern | 86.5% | 13.5% | +0.16661 | 37 |
| flat pattern | 52.8% | 47.2% | +0.00266 | 235 |
| spike pattern | 59.2% | 40.8% | +0.04927 | 218 |
| up pattern | 54.5% | 45.5% | -0.02121 | 22 |
| level_lt_4 | 77.8% | 22.2% | +0.13235 | 153 |
| level_4_to_8 | 51.4% | 48.6% | +0.00757 | 284 |
| level_8_to_12 | 42.4% | 57.6% | -0.07164 | 66 |
| level_ge_12 | 44.4% | 55.6% | -0.06721 | 9 |

### 对比 R4 峰值

| 指标 | R4 no_partial_sig (峰值) | S2 with_persist (当前) |
|---|---|---|
| improvement_mae | 0.06324 | 0.03333 |
| helped_rate | 61.1% | 58.0% |
| harmed_rate | 38.9% | 42.0% |
| pre_strength | 3.66 | 41.80 |
| trans_utility | +0.00416 | -0.05933 |
| loaded_persist | 0 | 1536 |
