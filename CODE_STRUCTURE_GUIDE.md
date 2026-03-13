# 代码库文件结构说明文档

本文档对 `/data/wzx` 目录下的核心非 JSON 代码文件进行了梳理，介绍了每个文件的主要功能和在项目中的作用。

## 从 main.py 入口永远不会调用的文件（标注）
以下文件在“从 [main.py](file:///data/wzx/main.py) 启动”的前提下，**无论选择哪种运行模式（GUI/Web/CLI single/auto/confidence/batch/visualize 等）都不会被 main.py 的调用链 import/执行**（基于静态导入关系分析）：  
- [assess_inv.py](file:///data/wzx/assess_inv.py)  
- [assess_predict.py](file:///data/wzx/assess_predict.py)  
- [counterfactual_reasoning.py](file:///data/wzx/counterfactual_reasoning.py)  
- [demo2.py](file:///data/wzx/demo2.py)  
- [dump_content.py](file:///data/wzx/dump_content.py)  
- [experiment_improved.py](file:///data/wzx/experiment_improved.py)  
- [experiment_input.py](file:///data/wzx/experiment_input.py)  
- [extract_expert_interventions.py](file:///data/wzx/extract_expert_interventions.py)  
- [generate_refined_summary.py](file:///data/wzx/generate_refined_summary.py)  
- [knowledge_base_generator.py](file:///data/wzx/knowledge_base_generator.py)  
- [memory_evolution.py](file:///data/wzx/memory_evolution.py)  
- [sofa_prediction_main.py](file:///data/wzx/sofa_prediction_main.py)  
- [tasks.py](file:///data/wzx/tasks.py)  
- [test_demo2_viz.py](file:///data/wzx/test_demo2_viz.py)  
- [test_gui_viz.py](file:///data/wzx/test_gui_viz.py)  
- [test_viz.py](file:///data/wzx/test_viz.py)  
- [translate_and_generate.py](file:///data/wzx/translate_and_generate.py)  
- [update_memory_from_report.py](file:///data/wzx/update_memory_from_report.py)  

## 1. 核心运行与界面 (Core & UI)
- **[main.py](file:///data/wzx/main.py)**: 项目主程序入口。集成预测、评估、自动化模式、选项卡导航及 SOFA 图表显示功能。
- **[gui.py](file:///data/wzx/gui.py)**: 脓毒症休克预测与评估的 GUI 模块。负责界面布局、交互逻辑以及通过 `matplotlib` 展示可视化数据。
- **[demo2.py](file:///data/wzx/demo2.py)**: 基于 Streamlit 的 Web 交互演示程序。利用 `dspy` 调用远程 Ollama 模型进行脓毒症休克风险评估。（从 main.py 入口不可达）
- **[ui_preview.html](file:///data/wzx/ui_preview.html)**: 预览用的 HTML 文件（通常用于展示生成的报告或 UI 原型）。

## 2. 实验与模型逻辑 (Experiment & Model Logic)
- **[experiment.py](file:///data/wzx/experiment.py)**: 核心实验框架。配置 `dspy` (Ollama 接口)，定义自适应实验代理（AdaptiveExperimentAgent），处理推理、评估和知识库更新逻辑。
- **[experiment_improved.py](file:///data/wzx/experiment_improved.py)**: `experiment.py` 的改进版，包含优化的模型配置和输出处理。
- **[core_functions.py](file:///data/wzx/core_functions.py)**: 核心预测与评估函数库。作为 `main.py` 和 `gui.py` 之间的桥梁，避免循环导入。
- **[fact_prediction.py](file:///data/wzx/fact_prediction.py)**: 事实预测模块。从临床描述中提取实际干预方案，并与模型预测进行对比分析，计算模型信任度。
- **[sofa_prediction_evaluator.py](file:///data/wzx/sofa_prediction_evaluator.py)**: SOFA 评分预测评估器。负责调用 Ollama 模型、提取置信度并保存评估报告。

## 3. 数据处理与生成 (Data Processing & Generation)
- **[patient_text_generator2.py](file:///data/wzx/patient_text_generator2.py)**: 患者临床描述生成器。负责加载 CSV 数据、补齐缺失特征并计算 SOFA 各项指标，将其转化为自然语言描述。
- **[dump_content.py](file:///data/wzx/dump_content.py)**: 内容提取脚本。用于从 JSON 数据中提取特定字段或进行格式转换。（从 main.py 入口不可达）
- **[translate_and_generate.py](file:///data/wzx/translate_and_generate.py)**: 翻译与生成工具。处理医学术语的汉化/英化及 LaTeX 格式生成。（从 main.py 入口不可达）
- **[generate_refined_summary.py](file:///data/wzx/generate_refined_summary.py)**: 精细化总结生成器。利用正则和映射表优化翻译质量，修复文本间距和医学格式。（从 main.py 入口不可达）

## 4. 评估与分析工具 (Evaluation & Analysis)
- **[assess_inv.py](file:///data/wzx/assess_inv.py)**: 干预措施评估工具。根据预设规则（如 SOFA 变化）对模型建议的干预方案进行数值化评估。
- **[assess_predict.py](file:///data/wzx/assess_predict.py)**: 预测准确性评估工具。将模型预测的风险等级（low/medium/high）与基于 SOFA 的真实严重程度进行对比。
- **[generate_model_trust_chart.py](file:///data/wzx/generate_model_trust_chart.py)**: 模型信任度图表生成器。可视化展示不同模型的预测准确率和临床一致性。

## 5. 数据库与配置 (Database & Config)
- **[database.py](file:///data/wzx/database.py)**: 数据库管理模块。使用 SQLAlchemy ORM 定义表结构（患者、预测、反馈等），处理 CRUD 操作。
- **[sepsis_prediction.db](file:///data/wzx/sepsis_prediction.db)**: SQLite 数据库文件，存储系统运行产生的持久化数据。
- **[environment.yml](file:///data/wzx/environment.yml)**: Conda 环境配置文件，记录了项目运行所需的依赖包及其版本。

## 6. 测试与可视化 (Testing & Viz)
- **[test_viz.py](file:///data/wzx/test_viz.py)** / **[test_gui_viz.py](file:///data/wzx/test_gui_viz.py)**: 可视化功能的单元测试脚本。
- **[viz.py](file:///data/wzx/viz.py)**: (未直接列出但被引用) 绘图后端，支持趋势图和雷达图生成。

## 7. 经验知识库 (Experience & Knowledge Base)
- **[advanced_experience_memory.py](file:///data/wzx/advanced_experience_memory.py)**: 前沿混合记忆经验库的核心实现。包含情景记忆、语义记忆和程序记忆的三层架构，支持混合检索（语义+符号+质量+时效）和在线巩固。
- **[experience_integration.py](file:///data/wzx/experience_integration.py)**: 经验库集成模块。负责将经验检索无缝接入预测流程，实现从 `AdvancedExperienceMemoryBank` 到旧版 `ExperienceKnowledgeBase` 的自动回退与兼容。
- **[experience_knowledge_base.py](file:///data/wzx/experience_knowledge_base.py)**: 基础版经验知识库实现。提供基本的相似病例检索功能，作为系统的保底方案。

## 8. 文档与说明 (Documentation)
- **[README.md](file:///data/wzx/README.md)**: 项目总体介绍，包含 GitHub 提交指南与最新功能说明。
- **[PROJECT_CONFIG.md](file:///data/wzx/PROJECT_CONFIG.md)**: 项目配置参数说明。
- **[usage_instructions.txt](file:///data/wzx/usage_instructions.txt)** / **[readme.txt](file:///data/wzx/readme.txt)**: 使用说明与简单备注。
- **[技术路线终稿.pdf](file:///data/wzx/技术路线终稿.pdf)**: 项目设计的核心技术文档。
- **[TECHNICAL_ROUTE_EXPERIENCE_MEMORY.md](file:///data/wzx/TECHNICAL_ROUTE_EXPERIENCE_MEMORY.md)**: 经验记忆系统的详细技术路线文档，阐述了混合记忆架构、检索评分公式及系统集成方式。
- **[EXECUTABLE_REFACTOR_CHECKLIST.md](file:///data/wzx/EXECUTABLE_REFACTOR_CHECKLIST.md)**: 函数级重构清单。列出了为支持新经验库所需的具体代码改造点，如阈值解析和高危门控逻辑。
