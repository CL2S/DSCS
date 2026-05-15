# 实验与运行指南

## 目标

本文件回答三个问题：

1. 这个仓库有哪些运行入口
2. 各条主线怎么执行
3. 输出结果应该如何解读


## 运行入口

### 表格分类

- [run_experiment.py](/run_experiment.py)

适用场景：

- tabular 分类实验
- 早期动态记忆分类验证

### ICU / Sepsis 时序实验

- [run_temporal_experiment.py](/run_temporal_experiment.py)

适用场景：

- patient 级或 window 级时序任务
- legacy 与 manifold 两条 temporal 路线

### Forecasting 与反事实实验

- [run_forecasting_experiment.py](/run_forecasting_experiment.py)

适用场景：

- 多步时序预测
- eICU Sepsis-3 forecasting
- counterfactual donor 评估
- persistent memory 与 KG 接入实验

## 常用运行命令

### 表格分类

```bash
python run_experiment.py --csv input/diabetes_comma.csv --label-col SepsisLabel --output-json output/tabular/result.json
```

### Temporal ICU / Sepsis

```bash
python run_temporal_experiment.py --json input/icu_stays_descriptions88.json --task-type window --feature-mode prototype --target-mode trajectory_shape_balanced --history-window 4 --forecast-horizon 3 --folds 6 --output-json output/temporal/temporal_cv_window.json
```

### eICU Sepsis-3 forecasting

```bash
python run_forecasting_experiment.py --dataset-format eicu_sepsis3 --history-length 4 --forecast-horizon 2 --eicu-target-field total_sofa --eicu-max-series 512 --epochs 1 --batch-size 256 --encoder-type transformer --output-json output/forecasting/eicu_sepsis3_run.json
```

### eICU Sepsis-3 + KG

```bash
python run_forecasting_experiment.py --dataset-format eicu_sepsis3 --history-length 4 --forecast-horizon 2 --eicu-target-field total_sofa --eicu-max-series 512 --enable-kg --kg-directory input/knowledge/13_processed_ready/sepsis_kg_guideline_enhanced --epochs 1 --batch-size 256 --encoder-type transformer --output-json output/forecasting/eicu_sepsis3_kg_run.json
```

## 输出目录

### `output/tabular`

主要保存：

- 分类结果 JSON
- 简单对照结果

### `output/temporal`

主要保存：

- patient/window 级实验结果
- 交叉验证结果
- temporal 任务导出文件

### `output/forecasting`

主要保存：

- forecasting 主实验结果
- eICU forecasting 结果
- counterfactual donor 评估结果
- persistent memory 相关结果

### `output/analysis`

主要保存：

- 指标对照表
- KG 预处理结果
- donor 逐例对照表
- 其他结构化分析文件

## 结果文件怎么读

以 forecasting 结果 JSON 为例，优先看这些字段：

- `dataset_summary`
  - 数据规模、history length、forecast horizon
- `trainer_config`
  - 本次训练配置
- `memory_config`
  - 记忆系统配置
- `test_metrics`
  - factual 主指标
- `base_only`
  - 不使用 memory 的基线指标
- `memory_effectiveness`
  - memory 相对基线的收益
- `counterfactual_evaluation`
  - donor 与反事实相关指标
- `persistent_memory`
  - 持久化经验与 neural cache 使用情况

## 如何判断一次 forecasting 实验是否有效

建议按下面顺序检查：

1. `dataset_summary`
   - 样本数、history length、forecast horizon 是否符合预期
2. `training_summary`
   - 是否正常完成训练
3. `test_metrics`
   - factual 误差是否在合理范围
4. `base_only`
   - 是否有基线可比较
5. `memory_effectiveness`
   - `improvement_mae` 是否为正
6. `counterfactual_evaluation`
   - donor 是否成功找到
   - donor 是否合理

## eICU 反事实结果怎么读

eICU 反事实结果最重要的字段在 `counterfactual_evaluation`：

- `donor_found_rate`
- `donor_similarity_mean`
- `donor_kg_similarity_mean`
- `donor_guideline_compatibility_mean`
- `donor_state_match_mean`
- `donor_missing_care_penalty_mean`
- `donor_exact_experience_match_rate`
- `mean_predicted_delta`
- `predicted_improvement_rate`

这些字段的详细定义见：

- [KG_COUNTERFACTUAL_METRICS_AND_DEFAULTS_UPDATE.md](/docs/03_knowledge_graph/KG_COUNTERFACTUAL_METRICS_AND_DEFAULTS_UPDATE.md)

## 与运行最相关的源码

如果你在定位运行问题，通常优先看：

- [tsf_data.py](/src/tsf_data.py)
- [manifold_forecasting_trainer.py](/src/manifold_forecasting_trainer.py)
- [memory_manager.py](/src/memory_manager.py)
- [persistent_memory_store.py](/src/persistent_memory_store.py)
- [kg_integration.py](/src/kg_integration.py)

## 实验组织建议

建议把实验分成三类：

- `smoke`
  - 验证链路是否通
- `mid-scale`
  - 验证设计是否成立
- `formal`
  - 输出最终对照结果

对 eICU forecasting 与 counterfactual，推荐先从中等规模开始，再跑正式规模。
