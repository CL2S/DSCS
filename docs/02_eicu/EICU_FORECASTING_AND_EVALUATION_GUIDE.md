# eICU Forecasting And Evaluation Guide

## 1. 文档目的

本文档把当前项目中使用的 eICU Sepsis-3 处理后数据集说明清楚，重点回答 4 个问题：

- 这套数据由哪些文件组成，规模多大
- 主要字段有哪些，分别表示什么
- 缺失值情况如何，代码里怎么处理
- 这些数据最终如何进入 forecasting 和 counterfactual 任务

当前说明基于这两个处理后文件：

- [eicu_sepsis3_labels.csv](/e:/worktable/日常/脓毒症记忆/记忆代码/DSCS/temp_repo/eicu数据库/processed/eicu_sepsis3_labels.csv)
- [eicu_sepsis3_sofa_6h_trajectory.csv](/e:/worktable/日常/脓毒症记忆/记忆代码/DSCS/temp_repo/eicu数据库/processed/eicu_sepsis3_sofa_6h_trajectory.csv)

## 2. 数据集组成

这套数据分为两层。

第一层是 `labels` 表，按 `patientunitstayid` 聚合，每个 ICU stay 一行，主要保存：

- 患者基本信息
- 医院/病区信息
- 感染与脓毒症标签
- 抗菌药、升压药、乳酸等关键干预或结果摘要

第二层是 `trajectory` 表，按 6 小时时间窗展开，每个 ICU stay 多行，主要保存：

- SOFA 与 qSOFA 轨迹
- 各器官评分轨迹
- 一部分生命体征和实验室指标轨迹
- 呼吸支持、升压药等干预轨迹

## 3. 数据规模

以下统计基于当前 `processed` 文件直接计算得出。

| 项目 | 数值 |
| --- | ---: |
| `labels` 行数 | 19,257 |
| `labels` 列数 | 40 |
| `trajectory` 行数 | 231,084 |
| `trajectory` 列数 | 28 |
| ICU stay 数量 | 19,257 |
| 平均每个 stay 的时间窗数 | 12 |
| `bin_index` 范围 | -8 到 3 |

从 `bin_index` 范围和平均窗数可以看出，当前数据基本上围绕感染锚点构造了一个 12 个 6 小时窗的观测范围，也就是大约 72 小时的相对时间窗口。

## 4. 关键标签分布

下面这些比例同样来自当前 `labels` 文件统计。

| 指标 | 比例 |
| --- | ---: |
| `adult_eligible = 1` | 99.97% |
| `sepsis3_label = 1` | 55.14% |
| `septic_shock_label_operational = 1` | 6.84% |
| `septic_shock_label_relaxed = 1` | 9.88% |
| `post_24h_vasopressor_any = 1` | 31.52% |
| `antibiotic_course_ge_72h = 1` | 14.66% |
| `antibiotic_offset_minutes` 在 0 到 180 分钟内 | 20.63% |

这说明当前样本集中：

- 大部分是成人 ICU stay
- `Sepsis-3` 阳性比例较高
- 休克样本占比明显低于 sepsis 样本
- 一部分样本存在较早期抗菌药和升压药暴露

## 5. 数值字段概览

选几个最关键字段做一个直观概览。

### `labels` 层摘要

| 字段 | 均值 | 最小值 | 最大值 |
| --- | ---: | ---: | ---: |
| `age_years` | 64.65 | 16 | 90 |
| `admissionweight_kg` | 83.84 | 18.1 | 606 |
| `acutephysiologyscore` | 55.62 | -1 | 200 |
| `apachescore` | 68.59 | -1 | 205 |
| `post_24h_lactate_max` | 3.03 | 0.1 | 33.5 |
| `post_24h_max_total_sofa` | 6.14 | 0 | 22 |

### `trajectory` 层摘要

| 字段 | 均值 | 最小值 | 最大值 |
| --- | ---: | ---: | ---: |
| `total_sofa` | 4.10 | 0 | 22 |
| `qsofa_score` | 0.50 | 0 | 3 |
| `urine_24h_ml` | 358.41 | -600 | 48,793 |

### 干预轨迹阳性率

| 字段 | 阳性比例 |
| --- | ---: |
| `resp_support > 0` | 12.71% |
| `vasopressor_any > 0` | 8.63% |

需要注意两点：

- `acutephysiologyscore`、`apachescore` 中出现 `-1`，说明部分字段可能使用了占位值或编码值，不能简单当成真实生理下界解释。
- `urine_24h_ml` 中出现负值，说明原始处理结果里可能仍保留了异常值或记录偏差，后续建模时应更多把它视作“输入信号”，而不是直接临床解释。

## 6. 缺失值情况

### `labels` 表缺失率最高的字段

| 字段 | 缺失率 |
| --- | ---: |
| `post_24h_map_min` | 76.27% |
| `post_24h_lactate_max` | 47.86% |
| `sepsis_diagnosis_examples` | 36.49% |
| `apachescore` | 13.63% |
| `acutephysiologyscore` | 13.63% |
| `admissionweight_kg` | 3.13% |
| `apacheadmissiondx` | 0.65% |
| `ethnicity` | 0.45% |

### `trajectory` 表缺失率最高的字段

| 字段 | 缺失率 |
| --- | ---: |
| `sbp_min` | 91.83% |
| `map_min` | 91.75% |
| `pf_ratio` | 91.12% |
| `pao2_min` | 89.35% |
| `bilirubin_max` | 88.80% |
| `lactate_max` | 88.21% |
| `platelets_min` | 81.37% |
| `fio2_max` | 79.41% |
| `creatinine_max` | 79.04% |
| `gcs_min` | 71.77% |
| `rr_max` | 65.83% |

这组缺失率说明当前数据的一个现实特征：

- SOFA/qSOFA 相关派生评分相对完整
- 原始生命体征和实验室轨迹明显更稀疏
- 这也是为什么当前模型更依赖“评分轨迹 + 聚合特征 + 标签层摘要”，而不是直接做高维连续监护建模

## 7. 原始字段分组与含义

### 7.1 `labels` 表字段

#### 标识字段

- `patientunitstayid`: ICU stay 主键
- `patienthealthsystemstayid`: 更高层级住院主键
- `uniquepid`: 患者标识

#### 人口学与住院背景

- `gender`: 性别
- `age_raw`, `age_years`: 年龄原始值与数值化年龄
- `ethnicity`: 种族/族群信息
- `admissionweight_kg`: 入院体重
- `hospitalid`, `wardid`, `unittype`: 医院、病区、单元类型
- `unitdischargeoffset_minutes`, `unitdischargestatus`: ICU 出院时间与结局状态

#### 感染与脓毒症标签

- `has_sepsis_diagnosis`: 是否存在 sepsis 相关诊断
- `suspected_infection_from`: 疑似感染锚点来源
- `suspected_infection_offset_minutes`: 疑似感染相对时间
- `infection_anchor_type`, `infection_anchor_value`, `infection_anchor_offset_minutes`: 感染锚点类型、内容和相对时间
- `sepsis_diagnosis_examples`: 诊断文本示例
- `sepsis3_label`: 主要 Sepsis-3 标签
- `sepsis3_label_baseline0`: 另一种 baseline 处理下的 Sepsis-3 标签
- `septic_shock_label_operational`: 更严格的休克标签
- `septic_shock_label_relaxed`: 更宽松的休克标签

#### 病情严重程度摘要

- `acutephysiologyscore`, `apachescore`: Apache/生理严重程度分数
- `pre_baseline_total_sofa`: 基线前 SOFA
- `post_24h_max_total_sofa`: 24 小时后 SOFA 峰值
- `max_total_sofa_m48_to_p24`: 从感染前 48h 到后 24h 的 SOFA 峰值
- `delta_sofa_post_vs_pre`: 感染前后 SOFA 变化
- `post_24h_qsofa_max`: 24 小时后 qSOFA 峰值
- `post_24h_max_cardio_score`: 24 小时后心血管 SOFA 峰值

#### 干预与关键结果摘要

- `antibiotic_name`: 抗菌药名称
- `antibiotic_offset_minutes`: 抗菌药相对时间
- `antibiotic_course_ge_72h`: 是否达到较长疗程
- `post_24h_vasopressor_any`: 24 小时后是否使用升压药
- `post_24h_lactate_max`: 24 小时后乳酸峰值
- `post_24h_map_min`: 24 小时后 MAP 最低值
- `adult_eligible`: 是否满足成人分析条件

### 7.2 `trajectory` 表字段

#### 时间轴

- `patientunitstayid`: ICU stay 主键
- `bin_index`: 6 小时时间窗索引
- `rel_start_hours`, `rel_end_hours`: 相对小时起止

#### 评分轨迹

- `resp_score`, `coag_score`, `liver_score`, `cardio_score`, `cns_score`, `renal_score`: SOFA 分器官评分
- `total_sofa`: 总 SOFA
- `qsofa_score`: qSOFA

#### 生理与实验室轨迹

- `urine_24h_ml`: 24 小时尿量
- `bilirubin_max`: 胆红素
- `creatinine_max`: 肌酐
- `platelets_min`: 血小板
- `gcs_min`: GCS 最低值
- `rr_max`: 呼吸频率最大值
- `sbp_min`: 收缩压最低值
- `map_min`: 平均动脉压最低值
- `pao2_min`: 动脉氧分压最低值
- `fio2_max`: 吸入氧浓度最大值
- `pf_ratio`: PaO2/FiO2 比值
- `lactate_max`: 乳酸最大值
- `suspected_infection_offset_minutes`: 当前时间窗相对感染的偏移信息

#### 干预轨迹

- `resp_support`: 呼吸支持状态
- `vasopressor_any`: 升压药是否使用
- `vasopressor_score`: 升压药强度或综合评分

## 8. 进入模型后的特征分组

当前项目不会把原始 CSV 原封不动喂进模型，而是构造成 [ForecastSample](/src/tsf_data.py)。

主要包括 6 组输入：

### 8.1 `sequence`

当前多变量时间序列主输入共 10 维：

- `suspected_infection_offset_minutes`
- `resp_score`
- `coag_score`
- `liver_score`
- `cardio_score`
- `cns_score`
- `renal_score`
- `qsofa_score`
- `total_sofa`
- `urine_24h_ml`

### 8.2 `patient_static`

由三部分组成：

- 当前窗口上下文统计特征
- `labels` 表中的患者级与严重程度摘要
- KG 映射后的知识特征

### 8.3 `intervention_static`

当前主要包含 16 个干预摘要特征，例如：

- `antibiotic_offset_minutes`
- `antibiotic_course_ge_72h`
- `post_24h_vasopressor_any`
- 抗菌药文本 hash 特征
- 当前窗口中呼吸支持/升压药聚合统计

### 8.4 `intervention_sequence`

当前干预序列 3 维：

- `resp_support`
- `vasopressor_any`
- `vasopressor_score`

### 8.5 `formation_features`

固定 16 维，用于描述当前窗口的形态与轨迹模式，例如：

- slope
- volatility
- seasonal gap
- curvature
- regime mix

### 8.6 `kg_features`

来自 eICU -> KG 映射层，当前 14 维，典型包括：

- `kg_state_sepsis`
- `kg_state_septic_shock`
- `kg_state_hypotension`
- `kg_state_high_lactate`
- `kg_exam_sofa`
- `kg_exam_lactate`
- `kg_treat_early_antimicrobial`
- `kg_treat_vasopressor`
- `kg_guideline_alignment`

更详细的 KG 接入见：

- [KG_01_OVERVIEW.md](/docs/03_knowledge_graph/KG_01_OVERVIEW.md)
- [KG_02_INTEGRATION.md](/docs/03_knowledge_graph/KG_02_INTEGRATION.md)

## 9. 缺失值在代码里如何处理

当前缺失值处理并不是统一插补，而是按用途分层处理。

### 9.1 数值/类别字段识别

在 [tsf_data.py](/src/tsf_data.py) 的 `_coerce_numeric_columns(...)` 中：

- 能较稳定转成数值的列作为 numeric
- 其余列作为 categorical

### 9.2 窗口聚合特征

在 `_aggregate_context_features(...)` 中：

- 先 `dropna()`
- 如果整列在窗口内没有值，则该列的 `last / mean / max` 直接记为 `0`

### 9.3 时序主输入

在 `_normalize_multivariate_window(...)` 中：

- 用 `pd.to_numeric(..., errors="coerce").fillna(0.0)` 处理
- 也就是进入序列编码器前，缺失会被填成 `0`
- 然后再按列做窗口内标准化

### 9.4 干预序列

在 `_window_matrix(...)` 中：

- 通过 `_safe_float(...)` 读值
- 缺失最终也会被转成 `0`

这意味着当前系统对缺失值的处理是“任务型处理”，不是临床统计意义上的严格多重插补。因此：

- 对模型训练是可运行的
- 对单个字段做严格临床解释时要更谨慎

## 10. forecasting 与 counterfactual 任务中怎么用

### 10.1 factual forecasting

当前 factual 任务是：

- 用窗口级病情状态和干预输入
- 预测未来 `forecast_horizon` 个时间步的 `total_sofa`

### 10.2 counterfactual prediction

当前 counterfactual 路径是：

1. 先得到当前样本的 factual 表示
2. 从 memory / intervention store 中检索 donor
3. 用 donor 的干预替换当前样本干预
4. 再做一次前向，得到 counterfactual prediction

所以当前并不是在“重建患者状态”，而是在“保持患者状态不变的前提下，替换干预输入”。

更详细见：

- [SYSTEM_ARCHITECTURE_DETAILED_GUIDE.md](/docs/01_core/SYSTEM_ARCHITECTURE_DETAILED_GUIDE.md)
- [KG_03_COUNTERFACTUAL_EVAL.md](/docs/03_knowledge_graph/KG_03_COUNTERFACTUAL_EVAL.md)

## 11. 当前数据集的优点与局限

### 优点

- 有明确的 sepsis / shock 标签层
- 有 6 小时窗的病情轨迹
- 有可迁移的干预相关字段
- 可以自然接入 SOFA/qSOFA 与 KG 约束

### 局限

- 原始生命体征和实验室轨迹缺失率较高
- 一部分字段存在占位值或异常值
- 当前更适合做 donor-based counterfactual，而不是高频 ICU 连续监护建模
- 现阶段模型对缺失值的处理偏工程可用，不等于临床统计最优处理

## 12. 相关源码

- [tsf_data.py](/src/tsf_data.py)
- [run_forecasting_experiment.py](/run_forecasting_experiment.py)
- [manifold_forecasting_trainer.py](/src/manifold_forecasting_trainer.py)
