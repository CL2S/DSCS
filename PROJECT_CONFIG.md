# 脓毒症智能分析实验项目配置说明

本文件详细介绍项目的运行环境、DSPy 的使用方式、所用大模型及评估流程、数据结构与接口，以及如何在本机或服务器上启动图形界面与命令行模式。

## 环境
- Python 版本：`3.10`
- 推荐使用 Conda 环境并基于 `environment.yml` 安装所有依赖
- 关键依赖：`dspy==2.6.27`、`numpy`、`pandas`、`optuna`、`httpx`、`litellm`、`openai`、`streamlit`、`matplotlib` 等
- 文件：`environment.yml`（路径：`/data/wzx/environment.yml`）
- 重要系统组件：`ollama` 本地/远程服务（端口默认 `11434`）用于模型推理与评估

## 项目结构与关键文件
- `main.py`：统一入口（CLI 与 Web UI），提供 `/api/run_single`、`/api/run_auto`、`/api/confidence` 等接口，并在 Web 模式下服务 `ui_preview.html`
- `ui_preview.html`：前端界面，展示最佳结果、评估摘要、SOFA 折线图与柱状图、其他模型结果
- `core_functions.py`：核心预测与评估编排，包含 `run_prediction`/`run_evaluation`/`select_best_prediction`/`save_best_prediction_result`
- `experiment.py`：DSPy 工作流、签名定义、SOFA 特征/评分生成与序列化逻辑，`configure_dspy` 设置 LLM
- `sofa_prediction_evaluator.py`：基于 Ollama 的评估器，构造提示词，调用 `ollama run`，解析置信度，保存评估报告
- `fact_prediction.py`：事实预测与模型信任评分（如存在）
- `environment.yml`：完整依赖列表与 Python 版本约束

## DSPy 简介与在项目中的使用
DSPy 是一个用于“指令式定义-编排-优化”LLM 工作流的框架。本项目用它来：
- 定义预测与评估的签名（输入/输出结构），保证推理与输出格式稳定
- 管理多阶段链式任务：风险评估、干预分析、临床报告生成、实际数据比较等
- 统一配置大模型（LLM）并在不同阶段复用


### 主要签名（部分）
- `AnalyzeInterventionAndRisk`：输出未来 8 小时的 `sofa_related_features` 时间序列、风险等级与详细推理
  - 位置：`/data/wzx/experiment.py:183-233`
- `CompareVitalSigns`：将模型预测与实际生命体征数据对比，计算各特征 MSE 与平均 MSE
  - 位置：`/data/wzx/experiment.py:234-279`
- `SepsisShockRiskAssessment`：感染性休克风险评估，抽取关键临床指标与状态摘要
  - 位置：`/data/wzx/experiment.py:156-174`

### DSPy 工作流
- 在 `core_functions.py:169-200` 中，`configure_dspy(model_name)` 后构建 `AdaptiveExperimentAgent` 并执行多阶段：
  - `shock_risk_assessment`（风险评估）
  - `analyze_intervention_and_risk`（干预分析与 SOFA 特征时间序列生成）
  - 序列化 prediction（含 `_store` 等内部结构），并裁剪为 `prediction.intervention_analysis`、`sofa_scores_series`、`hourly_sofa_totals`、`sofa_scores` 等可视化所需字段

## 使用大模型
- 预测与风险分析：通过 `experiment.py` 内的 DSPy 配置使用指定的 `ollama/<model>` 模型，例如 `ollama/gemma3:12b`、`ollama/mistral:7b`、`ollama/qwen3:30b` 等
- 评估（打分/置信度）：通过 `sofa_prediction_evaluator.py` 的 `evaluate_with_ollama` 调用 `ollama run <model> <prompt>`，要求模型在输出中包含“置信度评分/Confidence Score”字段以便解析
- 模型列表：
  - 默认定义：`main.py` 顶部的 `MODEL_NAMES`（若 `fact_prediction.py` 不存在时）
  - 位置：`/data/wzx/main.py:67-75`

### Ollama 配置与调用
- 评估提示词中包含对数值范围、趋势一致性、单位与精度的检查，以及首行置信度格式约束
- 位置：`/data/wzx/sofa_prediction_evaluator.py:73-169`
- 运行依赖：本机或远程的 Ollama 服务，默认 `http://localhost:11434`

## 评估与报告
- `run_evaluation(model_name, prediction_file)`：从预测结果提取 `sofa_related_features`，调用 `evaluate_with_ollama`，解析置信度并保存评估报告
  - 位置：`/data/wzx/core_functions.py:444-511`
- 报告保存路径：`output/{模型名}/evaluator_{patient_id}_{模型名}_{timestamp}.json`
  - 位置：`/data/wzx/sofa_prediction_evaluator.py:354-409`
- 在自动模式选择最佳模型时，会查找并附加最近的评估报告摘要到 `eval_breakdown` 供前端展示
  - 位置：`/data/wzx/core_functions.py:604-620`

## 数据结构与前端展示
- 预测输出（裁剪后）：`prediction.intervention_analysis`（含 `risk_level`、`sofa_related_features`、`reasoning`）
- 时间序列：`hourly_sofa_totals`（字典，键为字符串索引，如 `"0".."7"`）、`sofa_scores_series`（各组件序列）
- 静态截面：`predicted_sofa_scores`（最后时刻各组件与总分）
- 前端映射：
  - 折线图：`ui_preview.html` 的 `drawSofaLine`，展示干预与 baseline 两条曲线
  - 柱状图：`drawSOFABar`，展示最后时刻各组件得分
  - 评估摘要：`renderEvaluator` 卡片化展示评估器名称、分数、权重与模型输出

## 运行方式
### Web UI
```bash
python3 main.py --web --web-port 8000
# 访问 http://0.0.0.0:8000/ui_preview.html
```

### 命令行
```bash
# 单次预测
python3 main.py --mode single --model gemma3:12b --input "患者描述..." --intervention "去甲肾上腺素0.1 μg/kg/min"

# 自动模式（三模型轮换并择优）
python3 main.py --mode auto --models gemma3:12b mistral:7b qwen3:30b --input "患者描述..." --intervention "..."

# 事实预测与信任评估（如存在）
python3 main.py --mode confidence
```
