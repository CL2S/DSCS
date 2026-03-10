## 总体准则
- 保持功能与输入/输出兼容，不改动对外接口、文件结构和默认行为。
- 删除无效代码（未被调用、明显错误）与重复实现；整合工具函数。
- 保留容错与回退逻辑，避免影响现有运行路径。

## core_functions.py
- 删除未使用的导入：`sys` (core_functions.py:8)、`extract_key_reasons` (core_functions.py:27)、`input_extract_patient_id` (core_functions.py:31)。
- 抽取并统一 `safe_get(d, path)` 为模块级工具函数：替换在 `run_prediction` 内部的重复定义 (core_functions.py:118–126) 和 `run_evaluation` 的两处重复定义 (core_functions.py:353–360, 370–377)，以及 `save_best_prediction_result` 内部使用 (core_functions.py:528–536)。
- 精简 `run_evaluation`：移除重复的 `predicted_sofa_features` 提取块，保留一次“四路径”回退提取 (core_functions.py:382–395)。
- 保持 `run_prediction` 的保存与返回分支不变（当前仅在存在 `sofa_scores_series` 时写文件），避免行为变化。

## gui.py
- 删除未使用的导入：`time` (gui.py:12)。
- 删除未使用的从 main 导入：`process_case_auto, process_all_cases_auto, interactive_mode, cli_main, start_web_ui` (gui.py:76–78)。
- 删除废弃且未调用的方法：`run_simple_interactive_mode` (gui.py:367–370)。
- 其余逻辑保持不变；GUI 的三种模式与结果展示不受影响。

## experiment.py
- 删除无效方法：`AgentState.switch_learning_phase` 引用不存在属性 `self.performance` (experiment.py:1056–1062)。该方法未被调用，直接移除。
- 删除未使用的持久化方法：`LearningData.save_memory` 与 `LearningData.load_memory` (experiment.py:844–856)，当前工程未调用；减少体积与潜在序列化问题。
- 不改动核心 `forward`、SOFA评分计算、预处理、MSE计算与各类回退逻辑。

## sofa_prediction_evaluator.py
- 删除未使用的导入：`argparse` (sofa_prediction_evaluator.py:9)。
- 保留 `evaluate_with_ollama`、`extract_patient_id`、`extract_model_confidence`、`extract_key_reasons` 与报告保存函数；不改动子进程行为与目录结构。

## fact_prediction.py
- 保留现有功能；不删除模块内函数。该模块被 GUI 的“confidence”模式与 CLI 使用。
- 可选（不影响功能）：减少“=== 调试信息”类打印噪声，仅保留关键提示；若您同意再执行。

## main.py
- 删除未使用的导入：`extract_model_confidence, extract_key_reasons` (main.py:57–59)。
- 删除未使用且含未定义引用的 `interactive_mode` (main.py:141–170)，CLI 路径不调用此函数，移除可避免潜在误用。
- 保持 CLI、批量与 Web 启动逻辑不变。

## 统一与重复项
- `safe_get` 统一到 `core_functions.py` 并在本文件内复用；其它文件如需路径取值将保持各自实现，避免跨模块耦合。
- 保留 `MODEL_NAMES` 的“try-import + 本地回退”双路径以维持健壮性（core_functions.py、gui.py、main.py 当前用法），不做跨文件统一以免改变失败时的默认集。

## 验证方案
- 单例预测：运行 single 模式，确保预测文件生成与评估报告保存路径不变；观察日志与输出 JSON 字段一致。
- 自动轮换：运行 auto 模式，确认三模型轮换与最佳选择流程正常；`output` 目录结构与文件命名保持一致。
- GUI 模式：启动 GUI，运行三种模式；确认加载与绘图正常（时序折线与组件柱状图）。
- 文档检查：导入各模块不报错；`evaluate_with_ollama` 子进程异常时仍能保存报告并回退。

## 交付内容
- 一组补丁：逐文件的最小改动集（删除未用导入/函数、合并 `safe_get`、清理重复块）。
- 不添加新依赖，不改动外部接口；仅精简与修复无效/重复代码。