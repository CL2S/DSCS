# 脓毒症SOFA评分预测与智能评估系统 - 项目文档

## 一、项目概述

本项目是一个**基于大语言模型（LLM）的脓毒症（Sepsis）SOFA 评分预测与智能评估系统**。系统利用 DSPy 框架编排多个大模型（通过 Ollama 本地部署），实现对 ICU 患者的 SOFA（Sequential Organ Failure Assessment）评分时序预测、感染性休克风险评估、干预方案分析，并通过多模型评估、经验知识库检索增强、置信度校准等机制提升预测的一致性与可解释性。

### 核心能力

| 能力 | 说明 |
|------|------|
| **SOFA 评分预测** | 基于患者临床数据预测未来 8 小时的 SOFA 各系统评分及总分 |
| **感染性休克风险评估** | 输出风险等级（low/medium/high）及详细推理 |
| **干预方案分析** | 分析当前干预措施的预期效果，生成临床建议 |
| **多模型评估** | 使用多个评估模型对预测结果进行交叉验证和置信度评估 |
| **经验知识库增强** | 基于历史病例的混合记忆检索，增强预测的一致性和稳健性 |
| **事实预测与信任度** | 将模型预测与实际干预方案对比，计算模型信任度评分 |
| **多维可视化** | SOFA 趋势图、雷达图、柱状图、模型对比图表 |
| **双模式交互** | GUI 桌面应用 + Web 前端界面 |

---

## 二、技术架构

### 2.1 技术栈

| 层次 | 技术 |
|------|------|
| **语言** | Python 3.10 |
| **LLM 编排** | DSPy 2.6+ |
| **模型推理** | Ollama（本地部署，端口 11434） |
| **数据库** | SQLite + SQLAlchemy ORM |
| **GUI** | Tkinter + Matplotlib |
| **Web 前端** | 原生 HTML/CSS/JS（SPA + PWA） |
| **Web 后端** | Python http.server（集成于 main.py） |
| **可视化** | Matplotlib（自定义 viz 模块） |
| **数据处理** | NumPy, Pandas |
| **环境管理** | Conda (environment.yml) |

### 2.2 整体架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                         用户交互层                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐    │
│  │  CLI 命令行   │  │  GUI 桌面端  │  │  Web 前端 (SPA/PWA)  │    │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘    │
├─────────┼─────────────────┼──────────────────────┼────────────────┤
│         ▼                 ▼                      ▼                │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                      main.py (统一入口)                    │    │
│  │  - CLI 模式: --mode single/auto/confidence/batch          │    │
│  │  - Web 模式: --web --web-port 8000                        │    │
│  │  - GUI 模式: --gui                                        │    │
│  └──────────────────────────┬───────────────────────────────┘    │
│                             │                                     │
├─────────────────────────────┼─────────────────────────────────────┤
│                    core_functions.py (核心编排)                    │
│  - run_prediction() / run_evaluation()                            │
│  - select_best_prediction() / save_best_prediction_result()       │
├─────────────────────────────┼─────────────────────────────────────┤
│           ┌─────────────────┼─────────────────┐                   │
│           ▼                 ▼                   ▼                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐    │
│  │ 预测引擎      │  │ 评估引擎      │  │ 经验知识库            │    │
│  │ experiment.py │  │ evaluator.py │  │ experience_*.py      │    │
│  │ (DSPy Agent)  │  │ (Ollama调用) │  │ (混合记忆检索)        │    │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘    │
│         │                 │                      │                 │
│         ▼                 ▼                      ▼                 │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              Ollama 本地推理服务 (localhost:11434)         │    │
│  │  deepseek-r1:32b | gemma3:12b | qwen3:30b | mistral:7b   │    │
│  └──────────────────────────────────────────────────────────┘    │
├──────────────────────────────────────────────────────────────────┤
│                         数据持久层                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐    │
│  │ SQLite DB    │  │ JSON 结果文件 │  │ CSV 数据集            │    │
│  │ (ORM模型)    │  │ (output/)    │  │ (ai_clinician.csv)   │    │
│  └──────────────┘  └──────────────┘  └──────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 三、项目文件结构

### 3.1 核心运行与界面

| 文件 | 功能 |
|------|------|
| [main.py](main.py) | **项目统一入口**。集成 CLI/Web/GUI 三种运行模式，提供 `/api/run_single`、`/api/run_auto`、`/api/confidence` 等 HTTP 接口 |
| [gui.py](gui.py) | **Tkinter GUI 桌面应用**。包含选项卡导航、模型选择、输入面板、SOFA 图表展示等功能 |
| [core_functions.py](core_functions.py) | **核心预测与评估编排**。作为 main.py 和 gui.py 之间的桥梁，避免循环导入，封装预测/评估/择优核心逻辑 |

### 3.2 实验与模型逻辑

| 文件 | 功能 |
|------|------|
| [experiment.py](experiment.py) | **核心实验框架**（~73KB）。配置 DSPy 与 Ollama 接口，定义 `AdaptiveExperimentAgent` 及多个 DSPy 签名（风险评估、干预分析、生命体征对比等），实现 SOFA 评分计算与特征序列生成 |
| [sofa_prediction_evaluator.py](sofa_prediction_evaluator.py) | **SOFA 预测评估器**。构造评估提示词调用 Ollama 模型，解析置信度评分，保存评估报告 |
| [fact_prediction.py](fact_prediction.py) | **事实预测模块**。从临床描述中提取实际干预方案，与模型预测对比，计算模型信任度评分 |
| [counterfactual_reasoning.py](counterfactual_reasoning.py) | 反事实推理模块（独立工具，非主流程调用） |

### 3.3 经验知识库系统

| 文件 | 功能 |
|------|------|
| [advanced_experience_memory.py](advanced_experience_memory.py) | **前沿混合记忆经验库核心**。实现三层记忆架构（情景/语义/程序）+ 混合检索（语义+符号+质量+时效）+ 在线巩固 |
| [experience_integration.py](experience_integration.py) | **经验库集成模块**。优先加载高级经验库，失败时回退到基础版，提供无缝接入 |
| [experience_knowledge_base.py](experience_knowledge_base.py) | **基础版经验知识库**。提供基本的相似病例检索功能，作为保底方案 |
| [integrate_memory.py](integrate_memory.py) | 记忆整合工具脚本 |
| [memory_representation.py](memory_representation.py) | 记忆表示层（独立工具） |
| [memory_evolution.py](memory_evolution.py) | 记忆演化模块（独立工具） |
| [update_memory_from_report.py](update_memory_from_report.py) | 从报告更新记忆的脚本 |

### 3.4 数据处理与生成

| 文件 | 功能 |
|------|------|
| [patient_text_generator2.py](patient_text_generator2.py) | **患者临床描述生成器**。加载 CSV 数据，补齐缺失特征，计算 SOFA 各项指标，转化为自然语言描述 |
| [database.py](database.py) | **数据库管理模块**。SQLAlchemy ORM 定义 Users/Patients/Predictions/PredictionResults/SystemConfig 表，提供完整 CRUD 操作 |
| [knowledge_base_generator.py](knowledge_base_generator.py) | 知识库生成器（独立工具） |

### 3.5 评估与分析工具

| 文件 | 功能 |
|------|------|
| [assess_inv.py](assess_inv.py) | 干预措施评估工具。根据预设规则（SOFA 变化）对模型建议进行数值化评估 |
| [assess_predict.py](assess_predict.py) | 预测准确性评估工具。将模型风险等级与基于 SOFA 的真实严重程度对比 |
| [generate_model_trust_chart.py](generate_model_trust_chart.py) | **模型信任度图表生成器**。可视化不同模型的准确率与临床一致性 |
| [generate_refined_summary.py](generate_refined_summary.py) | 精细化总结生成器。正则+映射表优化翻译质量，修复医学格式 |
| [extract_expert_interventions.py](extract_expert_interventions.py) | 专家干预方案提取工具 |

### 3.6 可视化模块 (viz/)

| 文件 | 功能 |
|------|------|
| [viz/__init__.py](viz/__init__.py) | 可视化模块入口 |
| [viz/chart_builder.py](viz/chart_builder.py) | 图表构建器，统一图表生成接口 |
| [viz/sofa_charts.py](viz/sofa_charts.py) | SOFA 时序趋势图、组件分解图 |
| [viz/model_charts.py](viz/model_charts.py) | 模型对比图、置信度雷达图 |
| [viz/exporters.py](viz/exporters.py) | 图表导出（PNG/SVG/PDF） |
| [viz/themes.py](viz/themes.py) | 可视化主题（配色方案、字体配置） |

### 3.7 Web 前端 (web_app/)

| 文件 | 功能 |
|------|------|
| [web_app/index.html](web_app/index.html) | **Web 主页面**。SPA 应用，展示最佳结果、评估摘要、SOFA 图表 |
| [web_app/app.js](web_app/app.js) | **前端核心逻辑**（~56KB）。API 调用、图表渲染（折线图/柱状图）、模型对比展示 |
| [web_app/styles.css](web_app/styles.css) | 前端样式表（暗色主题） |
| [web_app/manifest.json](web_app/manifest.json) | PWA 清单文件 |
| [web_app/service-worker.js](web_app/service-worker.js) | PWA Service Worker，支持离线缓存 |

### 3.8 输入/输出数据

| 文件 | 说明 |
|------|------|
| [ai_clinician_dataset.csv](ai_clinician_dataset.csv) | **主数据集**（~216MB），包含 ICU 患者临床时序数据 |
| [icu_stays_descriptions88.json](icu_stays_descriptions88.json) | 88 个 ICU 病例的自然语言描述样本 |
| [extracted_expert_interventions.json](extracted_expert_interventions.json) | 提取的专家干预方案 |
| [knowledge_base_v1.json](knowledge_base_v1.json) | 知识库版本1数据 |
| [sepsis_prediction.db](sepsis_prediction.db) | SQLite 数据库文件 |
| [output/](output/) | 预测与评估结果输出目录 |
| [fact/](fact/) | 事实预测结果（按模型分目录: gemma3/meditron/medllama2/qwen3） |

### 3.9 文档与配置

| 文件 | 说明 |
|------|------|
| [README.md](README.md) | 项目简介与使用说明 |
| [PROJECT_CONFIG.md](PROJECT_CONFIG.md) | 项目配置文件说明 |
| [CODE_STRUCTURE_GUIDE.md](CODE_STRUCTURE_GUIDE.md) | 代码结构指南 |
| [TECHNICAL_ROUTE_EXPERIENCE_MEMORY.md](TECHNICAL_ROUTE_EXPERIENCE_MEMORY.md) | 经验记忆技术路线详细文档 |
| [EXECUTABLE_REFACTOR_CHECKLIST.md](EXECUTABLE_REFACTOR_CHECKLIST.md) | 函数级重构改造清单 |
| [基础经验v1.md](基础经验v1.md) | 基础经验文档 |
| [environment.yml](environment.yml) | Conda 环境依赖配置 |

---

## 四、核心工作流程

### 4.1 预测流程

```
1. 输入患者描述（自然语言临床文本 + 当前干预方案）
2. 患者文本生成器（patient_text_generator2.py）解析 CSV 数据
   - 补齐缺失特征
   - 计算各系统 SOFA 子评分
   - 生成结构化临床描述
3. configure_dspy(model_name) 配置目标 LLM（通过 Ollama）
4. AdaptiveExperimentAgent 执行多阶段推理：
   a. SepsisShockRiskAssessment — 休克风险评估
   b. AnalyzeInterventionAndRisk — 干预分析与 SOFA 特征时序生成
   c. CompareVitalSigns — 预测值与实际值对比
5. 输出结构化结果：
   - hourly_sofa_totals（未来8小时每小时 SOFA 总分）
   - sofa_scores_series（各组件分数序列）
   - predicted_sofa_scores（最终时刻截面分数）
   - intervention_analysis（含 risk_level, reasoning）
```

### 4.2 评估流程

```
1. 从预测结果提取 sofa_related_features
2. 构造评估提示词（sofa_prediction_evaluator.py）
   - 包含数值范围、趋势一致性、单位精度校验指令
   - 要求首行输出置信度评分
3. 调用 Ollama 模型进行评估
4. 解析置信度分数
5. 保存评估报告到 output/{模型名}/evaluator_{患者ID}_{模型名}_{时间戳}.json
```

### 4.3 经验增强流程

```
1. 系统初始化时加载 AdvancedExperienceMemoryBank
2. 从历史结果（output/best_result/result_*.json）批量导入记忆
3. 新病例到来时执行混合检索：
   - 语义向量相似度（哈希向量）
   - 符号规则匹配（干预类型 + 风险标签）
   - 案例质量加权
   - 时间衰减
4. 将 Top-K 相似经验注入预测上下文
5. 新预测完成后写回经验库（闭环更新）
```

### 4.4 经验知识库架构（三层记忆）

```
┌─────────────────────────────────────────┐
│         AdvancedExperienceMemoryBank     │
├─────────────────────────────────────────┤
│  Episodic Memory（情景记忆）              │
│  - 病例级实例：patient_id, 干预, 风险,    │
│    置信度, 质量分, embedding, 标签       │
│  - 回答："历史上是否出现过相似病例"        │
├─────────────────────────────────────────┤
│  Semantic Memory（语义记忆）              │
│  - 群体分布：风险分布, 干预有效性统计      │
│  - 回答："哪些干预在何种背景下更可靠"      │
├─────────────────────────────────────────┤
│  Procedural Memory（程序记忆）            │
│  - 规则聚合：干预+风险→支持度+质量→规则   │
│  - 回答："应优先遵循哪些经验策略"          │
├─────────────────────────────────────────┤
│  混合检索评分公式：                       │
│  S = 0.45·S_dense + 0.35·S_symbolic     │
│    + 0.15·S_quality + 0.05·S_recency    │
└─────────────────────────────────────────┘
```

---

## 五、运行方式

### 5.1 环境准备

```bash
# 创建 Conda 环境
conda env create -f environment.yml
conda activate sepsis

# 确保 Ollama 服务运行（默认 localhost:11434）
ollama serve
```

### 5.2 Web 模式

```bash
python3 main.py --web --web-port 8000
# 访问 http://localhost:8000/ui_preview.html
```

### 5.3 GUI 桌面模式

```bash
python3 main.py --gui
```

### 5.4 CLI 命令行模式

```bash
# 单次预测
python3 main.py --mode single --model gemma3:12b \
  --input "患者描述..." --intervention "去甲肾上腺素0.1 μg/kg/min"

# 自动模式（三模型轮换择优）
python3 main.py --mode auto --models gemma3:12b mistral:7b qwen3:30b \
  --input "患者描述..." --intervention "..."

# 置信度评估模式
python3 main.py --mode confidence

# 批量处理模式
python3 main.py --mode batch --model gemma3:12b
```

---

## 六、支持的大模型

系统通过 Ollama 支持多种开源大模型，默认配置在 [main.py:69-73](main.py#L69-L73)：

| 模型 | 参数规模 | 用途 |
|------|---------|------|
| `deepseek-r1:32b` | 32B | 预测/评估（推理能力强） |
| `gemma3:12b` | 12B | 预测/评估 |
| `qwen3:30b` | 30B | 预测/评估 |
| `mistral:7b` | 7B | 轻量评估 |
| `meditron` | - | 医疗领域专用（fact/ 目录） |
| `medllama2` | - | 医疗领域专用（fact/ 目录） |

---

## 七、数据库模型

基于 SQLAlchemy ORM（[database.py](database.py)），包含 5 张核心表：

| 表名 | 说明 | 关键字段 |
|------|------|---------|
| `users` | 用户认证与授权 | username, email, role, hashed_password |
| `patients` | 患者基本信息 | stay_id, subject_id, age, gender, admission_time |
| `predictions` | 预测记录 | patient_id, model_name, input_text, intervention, risk_level |
| `prediction_results` | 预测结果详情 | prediction_id, sofa_scores, evaluator_model, confidence_score |
| `system_config` | 系统配置 | config_key, config_value, description |

---

## 八、DSPy 签名定义

系统通过 DSPy 框架定义了多个结构化签名（[experiment.py](experiment.py)），确保 LLM 输出格式稳定：

| 签名 | 作用 |
|------|------|
| `SepsisShockRiskAssessment` | 感染性休克风险评估，提取关键临床指标与状态摘要 |
| `AnalyzeInterventionAndRisk` | 输出未来 8 小时 SOFA 特征时间序列、风险等级与推理 |
| `CompareVitalSigns` | 将预测值与实际生命体征对比，计算各特征 MSE |

---

## 九、前端功能（Web UI）

Web 前端（[web_app/](web_app/)）是一个完整的 SPA + PWA 应用：

- **最佳结果展示**：自动模式的择优结果卡片
- **SOFA 折线图**：干预曲线 vs Baseline 曲线对比
- **SOFA 柱状图**：最后时刻各系统评分分解
- **评估摘要卡片**：多评估器的分数、权重与输出
- **模型对比**：不同模型的预测差异可视化
- **暗色主题**：适配医学监护场景
- **PWA 支持**：可安装为桌面应用，支持离线缓存

---

## 十、关键设计决策

1. **DSPy 编排而非直接 API 调用**：通过 DSPy 的签名机制保证 LLM 输出结构稳定，便于后续优化和模型切换
2. **Ollama 本地部署**：医疗数据敏感性要求模型本地运行，Ollama 提供统一的本地模型管理
3. **三层记忆架构**（情景+语义+程序）：在传统 RAG 基础上增加了符号约束和规则巩固，提升临床可解释性
4. **"前沿优先，旧版回退"兼容策略**：确保新模块的引入不影响现有系统稳定性
5. **多模型交叉评估**：用不同模型互相评估预测结果，提高置信度判断的客观性
6. **GUI + Web 双界面**：同时满足桌面端深度交互和 Web 端轻量访问需求

---

## 十一、技术路线亮点

- **持续学习闭环**：预测 → 评估 → 写入经验库 → 增强后续预测
- **混合检索评分**：语义相似度(0.45) + 符号匹配(0.35) + 质量加权(0.15) + 时间衰减(0.05)
- **临床风险门控**：基于 qSOFA 的高危场景检测，限制经验置信度过度上调
- **无外部向量数据库依赖**：哈希向量方案实现了轻量级语义检索，降低部署复杂度

---

## 十二、待改进方向

1. 引入医学领域专用 Embedding 模型替代哈希向量
2. 规则巩固引入因果推断校正
3. 构建"记忆写入门控网络"，学习何时写入/遗忘
4. 前端迁移到现代框架（React/Vue）提升可维护性
5. 后端从简易 HTTP 服务迁移到 FastAPI
6. 增加用户认证与权限管理的实际实现

---

> 文档生成日期：2026-05-14
> 项目仓库：git@github.com:CL2S/DSCS.git
