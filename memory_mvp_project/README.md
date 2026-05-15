# Memory MVP Project（独立实验目录）

本目录是一个**完全独立**的小项目，用于实现“动态记忆 + 检索 + 写入 + 遗忘”的分类实验流程，不会影响仓库其他模块。

## 目标
- 适配任意“表格特征 + 单标签”数据集（不限脓毒症）。
- 支持十多个特征（10+）场景，特征更多也可运行。
- 提供训练、验证、测试拆分与核心分类评估。

## 目录结构
- `src/data_utils.py`：数据加载、类别编码、切分、标准化
- `src/memory_model.py`：动态记忆分类器（读取/写入/遗忘）
- `src/evaluate.py`：准确率、F1、二分类AUC
- `run_experiment.py`：命令行入口（从CSV训练并评估）
- `examples/generate_demo_data.py`：生成演示数据（12特征 + 1标签）
- `requirements.txt`：依赖说明（标准库运行，无第三方依赖）

## 快速开始
### 1) 生成演示数据（12个特征 + 1个标签）
```bash
python memory_mvp_project/examples/generate_demo_data.py --output memory_mvp_project/examples/demo_dataset.csv
```

### 2) 运行实验
```bash
python memory_mvp_project/run_experiment.py \
  --csv memory_mvp_project/examples/demo_dataset.csv \
  --label-col label \
  --test-size 0.2 \
  --val-size 0.1 \
  --top-k 16 \
  --sim-threshold 0.8 \
  --merge-alpha 0.2 \
  --decay 0.997 \
  --forget-threshold 0.1 \
  --max-memory 5000
```

## 可适配的数据要求
- 必须有且仅有一个标签列（通过 `--label-col` 指定）。
- 其余列默认作为特征（支持数值/类别特征；类别会自动编码）。
- 特征数要求至少10列（本项目会做检查）。

## 与主仓库隔离说明
- 本项目仅新增在 `memory_mvp_project/` 内。
- 不修改现有训练脚本和业务逻辑，不引入全局耦合。
- 使用时通过显式路径调用，不影响其他部分正常运行。
