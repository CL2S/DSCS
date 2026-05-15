param(
    [int]$MaxRounds = 20,
    [double]$TargetImprovementRatio = 0.10
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$projectRoot = Join-Path $repoRoot "memory_mvp_project"
$outputRoot = Join-Path $projectRoot "output\auto_iteration"
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$commonArgs = @(
    "memory_mvp_project\run_forecasting_experiment.py",
    "--dataset-format", "eicu_sepsis3",
    "--enable-kg",
    "--eicu-max-series", "128",
    "--epochs", "4",
    "--batch-size", "16",
    "--max-train-windows-per-series", "16",
    "--counterfactual-donor-score-mode", "structured",
    "--counterfactual-candidate-policy", "generated_best",
    "--enable-transition-memory"
)

$roundConfigs = @(
    @{ name = "both_additive_default"; args = @("--transition-factual-residual-mode","additive") },
    @{ name = "factual_only_additive"; args = @("--disable-transition-donor-path","--transition-factual-residual-mode","additive") },
    @{ name = "both_topk_4"; args = @("--transition-factual-residual-mode","additive","--transition-top-k","4") },
    @{ name = "factual_only_topk_4"; args = @("--disable-transition-donor-path","--transition-factual-residual-mode","additive","--transition-top-k","4") },
    @{ name = "both_topk_4_blend_012"; args = @("--transition-factual-residual-mode","additive","--transition-top-k","4","--transition-template-blend-weight","0.12") },
    @{ name = "factual_only_topk_4_blend_012"; args = @("--disable-transition-donor-path","--transition-factual-residual-mode","additive","--transition-top-k","4","--transition-template-blend-weight","0.12") },
    @{ name = "both_topk_4_blend_014"; args = @("--transition-factual-residual-mode","additive","--transition-top-k","4","--transition-template-blend-weight","0.14") },
    @{ name = "factual_only_topk_4_blend_014"; args = @("--disable-transition-donor-path","--transition-factual-residual-mode","additive","--transition-top-k","4","--transition-template-blend-weight","0.14") },
    @{ name = "both_topk_3_blend_012"; args = @("--transition-factual-residual-mode","additive","--transition-top-k","3","--transition-template-blend-weight","0.12") },
    @{ name = "factual_only_topk_3_blend_012"; args = @("--disable-transition-donor-path","--transition-factual-residual-mode","additive","--transition-top-k","3","--transition-template-blend-weight","0.12") },
    @{ name = "both_topk_5_blend_012"; args = @("--transition-factual-residual-mode","additive","--transition-top-k","5","--transition-template-blend-weight","0.12") },
    @{ name = "factual_only_topk_5_blend_012"; args = @("--disable-transition-donor-path","--transition-factual-residual-mode","additive","--transition-top-k","5","--transition-template-blend-weight","0.12") },
    @{ name = "both_delta_to_fusion_topk_4"; args = @("--transition-factual-residual-mode","delta_to_fusion_base","--transition-top-k","4","--transition-template-blend-weight","0.12") },
    @{ name = "factual_only_delta_to_fusion_topk_4"; args = @("--disable-transition-donor-path","--transition-factual-residual-mode","delta_to_fusion_base","--transition-top-k","4","--transition-template-blend-weight","0.12") },
    @{ name = "both_state075_action025"; args = @("--transition-factual-residual-mode","additive","--transition-top-k","4","--transition-state-weight","0.75","--transition-action-weight","0.25","--transition-template-blend-weight","0.12") },
    @{ name = "factual_only_state075_action025"; args = @("--disable-transition-donor-path","--transition-factual-residual-mode","additive","--transition-top-k","4","--transition-state-weight","0.75","--transition-action-weight","0.25","--transition-template-blend-weight","0.12") },
    @{ name = "both_score_008_select_002"; args = @("--transition-factual-residual-mode","additive","--transition-top-k","4","--transition-template-blend-weight","0.12","--transition-score-weight","0.08","--transition-selection-weight","0.02") },
    @{ name = "factual_only_score_008_select_002"; args = @("--disable-transition-donor-path","--transition-factual-residual-mode","additive","--transition-top-k","4","--transition-template-blend-weight","0.12","--transition-score-weight","0.08","--transition-selection-weight","0.02") },
    @{ name = "both_no_action_change"; args = @("--transition-factual-residual-mode","additive","--transition-top-k","4","--transition-template-blend-weight","0.12","--transition-action-change-weight","0.0","--transition-candidate-action-change-weight","0.0") },
    @{ name = "factual_only_no_action_change"; args = @("--disable-transition-donor-path","--transition-factual-residual-mode","additive","--transition-top-k","4","--transition-template-blend-weight","0.12","--transition-action-change-weight","0.0","--transition-candidate-action-change-weight","0.0") }
)

$summaryRows = @()
$bestRow = $null

for ($index = 0; $index -lt [Math]::Min($MaxRounds, $roundConfigs.Count); $index++) {
    $roundNumber = $index + 1
    $config = $roundConfigs[$index]
    $outputPath = Join-Path $outputRoot ("round_{0:00}_{1}.json" -f $roundNumber, $config.name)
    $cmdArgs = @($commonArgs + $config.args + @("--output-json", $outputPath))

    Write-Host ("[Round {0}] {1}" -f $roundNumber, $config.name)
    & python @cmdArgs

    $metricJson = & python -c "import json,sys; p=sys.argv[1]; obj=json.load(open(p,'r',encoding='utf-8')); lines=obj.get('evaluation_lines',{}); factual=lines.get('factual_forecasting',{}); donor=lines.get('counterfactual_donor_ranking',{}); out={'improvement_mae':factual.get('improvement',{}).get('improvement_mae',obj['memory_effectiveness']['improvement_mae']),'base_mae':factual.get('metrics',{}).get('memory_disabled',{}).get('mae',obj['base_only']['mae']),'hybrid_mae':factual.get('metrics',{}).get('memory_enabled',{}).get('mae',obj['test_metrics']['mae']),'hybrid_rmse':factual.get('metrics',{}).get('memory_enabled',{}).get('rmse',obj['test_metrics']['rmse']),'predicted_improvement_rate':donor.get('selection_quality',{}).get('predicted_improvement_rate',obj['counterfactual_evaluation'].get('predicted_improvement_rate',0.0)),'mean_predicted_delta':donor.get('selection_quality',{}).get('mean_predicted_delta',obj['counterfactual_evaluation'].get('mean_predicted_delta',0.0)),'donor_action_change_score_mean':donor.get('transition_signal',{}).get('donor_action_change_score_mean',obj['counterfactual_evaluation'].get('donor_action_change_score_mean',0.0))}; print(json.dumps(out, ensure_ascii=False))" $outputPath
    $metrics = $metricJson | ConvertFrom-Json
    $improvementMae = [double]$metrics.improvement_mae
    $baseMae = [double]$metrics.base_mae
    $improvementRatio = if ($baseMae -gt 0.0) { $improvementMae / $baseMae } else { 0.0 }
    $row = [PSCustomObject]@{
        round = $roundNumber
        name = $config.name
        hybrid_mae = [double]$metrics.hybrid_mae
        hybrid_rmse = [double]$metrics.hybrid_rmse
        improvement_mae = $improvementMae
        improvement_ratio_mae = $improvementRatio
        predicted_improvement_rate = [double]$metrics.predicted_improvement_rate
        mean_predicted_delta = [double]$metrics.mean_predicted_delta
        donor_action_change_score_mean = [double]$metrics.donor_action_change_score_mean
        output_json = $outputPath
    }
    $summaryRows += $row
    if ($null -eq $bestRow -or $row.improvement_ratio_mae -gt $bestRow.improvement_ratio_mae) {
        $bestRow = $row
    }
    if ($row.improvement_ratio_mae -ge $TargetImprovementRatio) {
        Write-Host ("Reached target at round {0}: {1:P2}" -f $roundNumber, $row.improvement_ratio_mae)
        break
    }
}

$summary = [PSCustomObject]@{
    target_improvement_ratio_mae = $TargetImprovementRatio
    rounds_executed = $summaryRows.Count
    best_round = $bestRow
    rounds = $summaryRows
}

$summaryPath = Join-Path $outputRoot "summary.json"
$summary | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $summaryPath
Write-Host ("Summary written to {0}" -f $summaryPath)
