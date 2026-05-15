# FRCDG Code Refactoring Plan v3 (Based on JBHI Review)

## 1. Feature Taxonomy & Strict Actionability Constraints (Feature Masking)
**Background**: The reviewer highlighted that current counterfactuals modify features like `CholCheck` (screening record) or `chest pain type` (symptom) to flip predictions. While mathematically valid, these are not clinically "actionable" treatment levers. Modifying them produces observational explanations, not actionable recourse.
**Task**:
- Create a strict feature taxonomy mapping for both Heart and Diabetes datasets. Categorize features into:
  - `immutable` (e.g., age, sex)
  - `non-actionable observational/symptom` (e.g., chest pain type, resting ecg, CholCheck)
  - `actionable clinical targets` (e.g., resting bp s, cholesterol - manageable via medication/lifestyle)
- Update `counterfactual_generate.py` and rule files to dynamically freeze both `immutable` and `non-actionable` features based on the taxonomy.
- Re-run the counterfactual generation pipeline to evaluate success rate, sparsity, and proximity when the optimizer is restricted **ONLY** to clinically actionable features.

## 2. Predictor Credibility & Calibration Evaluation
**Background**: The reviewer questioned the necessity and calibration of the 5x 6-layer Transformer ensemble on small tabular datasets, noting the lack of comparison to standard tabular models.
**Task**:
- Train and evaluate standard baseline predictive models on the datasets: Logistic Regression, Random Forest, and XGBoost/LightGBM.
- Implement evaluation scripts to compute comprehensive metrics: AUROC, AUPRC, Brier score, and Expected Calibration Error (ECE) for both the Transformer ensemble and the baselines.
- Generate Calibration Plots.
- Output these metrics to a comparative table to justify the use of the Transformer ensemble.

## 3. Metric Hygiene & Uncertainty Reporting
**Background**: The review pointed out inconsistencies in reported metrics (mixing L0 sparsity with iterative steps) and the absence of actual confidence intervals despite mentions in table captions.
**Task**:
- Ensure the evaluation pipeline distinctly separates and logs:
  - `L0 Sparsity` (Number of distinct features modified)
  - `Adjustment Steps` (Total iterations/steps taken by the optimizer)
- Update aggregation scripts (`baseline_multiseed_eval.py` / `feature_modification_stats.py`) to systematically calculate and report **95% Confidence Intervals** (or mean ± std/IQR) for all primary metrics.

## 4. Complete SOTA Baselines for Diabetes Dataset
**Background**: The manuscript previously compared the Diabetes dataset only against weak baselines (Random/CounteRGAN), omitting the stronger ones evaluated on the Heart dataset.
**Task**:
- Ensure all recent SOTA baselines (DiCE, FACE-like, GenRe, COLA, DCE) are formally processed and summarized for the Diabetes dataset (leveraging overnight runs).
- For baselines that hit resource limits (e.g., GenRe OOM on CPU), implement memory-constrained fallbacks or formally log the failure boundaries for the paper.
- Run `Alibi CounterfactualProto` consistently (potentially on a bounded subset) and format the results explicitly.