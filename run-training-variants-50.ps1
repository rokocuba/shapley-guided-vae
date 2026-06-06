Set-StrictMode -Version Latest
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    Write-Error "Python not found: $python"
    exit 1
}

$numIterations = 50
$baseOutDir = 'analysis/output/training_runs'
$common = @(
    '--epochs', '200',
    '--batch-size', '256',
    '--hidden-dims', '512,512',
    '--latent-dim', '8',
    '--kl-target', '5.00',
    '--aux-loss-weight', '0.8',
    '--lr-plateau-factor', '0.7',
    '--lr-scheduler-monitor', 'pix_recon',
    '--lr-plateau-patience', '4',
    '--lr', '1e-3',
    '--lr-min', '1e-7'
)
$shapley = @(
    '--shapley-warmup-epochs', '50',
    '--shapley-min-sampling-phases', '3',
    '--shapley-group-size', '32',
    '--shapley-sampling-batch-size', '512'
)

for ($i = 1; $i -le $numIterations; $i++) {
    $iterLabel = "iter_{0:D2}" -f $i
    $iterOutDir = Join-Path $baseOutDir $iterLabel
    $iterCommon = $common + @('--output-dir', $iterOutDir)

    Write-Host "===== Iteration $i / $numIterations : $iterLabel ====="

    Write-Host "  E0 baseline"
    & $python .\main.py --training-type baseline @iterCommon
    if ($LASTEXITCODE -ne 0) { Write-Warning "  E0 failed with exit code $LASTEXITCODE" }

    Write-Host "  E0_0 pix_only"
    & $python .\main.py --training-type pix_only @iterCommon --aux-loss-weight 0
    if ($LASTEXITCODE -ne 0) { Write-Warning "  E0_0 failed with exit code $LASTEXITCODE" }

    Write-Host "  E1 Shapley baseline tactic"
    & $python .\main.py --training-type shapley --shapley-tactic baseline @iterCommon @shapley
    if ($LASTEXITCODE -ne 0) { Write-Warning "  E1 failed with exit code $LASTEXITCODE" }

    Write-Host "  E2 Shapley marginal tactic"
    & $python .\main.py --training-type shapley --shapley-tactic marginal @iterCommon @shapley
    if ($LASTEXITCODE -ne 0) { Write-Warning "  E2 failed with exit code $LASTEXITCODE" }

    Write-Host "  E3 Shapley conditional tactic"
    & $python .\main.py --training-type shapley --shapley-tactic conditional @iterCommon @shapley
    if ($LASTEXITCODE -ne 0) { Write-Warning "  E3 failed with exit code $LASTEXITCODE" }

    Write-Host "  Plotting iteration $iterLabel"
    & $python .\analysis\plot_training_results.py --runs $iterOutDir --out $iterOutDir --all-runs
    if ($LASTEXITCODE -ne 0) { Write-Warning "  plot_training_results failed with exit code $LASTEXITCODE" }

    Write-Host "  Comparing simple pixel baselines for $iterLabel"
    & $python .\analysis\compare_mean_baselines.py --out-dir $iterOutDir
    if ($LASTEXITCODE -ne 0) { Write-Warning "  compare_mean_baselines failed with exit code $LASTEXITCODE" }
}

Write-Host "===== All $numIterations iterations complete ====="
Write-Host "Merging cross-iteration summaries..."

$summaryRows = @()
for ($i = 1; $i -le $numIterations; $i++) {
    $iterLabel = "iter_{0:D2}" -f $i
    $iterOutDir = Join-Path $baseOutDir $iterLabel
    $summaryPath = Join-Path $iterOutDir 'run_summary.csv'
    if (Test-Path -LiteralPath $summaryPath) {
        $rows = Import-Csv -LiteralPath $summaryPath
        foreach ($row in $rows) {
            $row | Add-Member -NotePropertyName 'iteration' -NotePropertyValue $iterLabel -Force
            $summaryRows += $row
        }
    }
}

if ($summaryRows.Count -gt 0) {
    $mergedSummaryPath = Join-Path $baseOutDir 'run_summary_all_iterations.csv'
    $summaryRows | Export-Csv -LiteralPath $mergedSummaryPath -NoTypeInformation
    Write-Host "Cross-iteration summary saved to $mergedSummaryPath"

    $mergedRunHistoryPath = Join-Path $baseOutDir 'run_history_all_iterations.csv'
    $historyRows = @()
    for ($i = 1; $i -le $numIterations; $i++) {
        $iterLabel = "iter_{0:D2}" -f $i
        $iterOutDir = Join-Path $baseOutDir $iterLabel
        $runHistoryPath = Join-Path $iterOutDir 'run_history.csv'
        if (Test-Path -LiteralPath $runHistoryPath) {
            $rows = Import-Csv -LiteralPath $runHistoryPath
            foreach ($row in $rows) {
                $row | Add-Member -NotePropertyName 'iteration' -NotePropertyValue $iterLabel -Force
                $historyRows += $row
            }
        }
    }
    if ($historyRows.Count -gt 0) {
        $historyRows | Export-Csv -LiteralPath $mergedRunHistoryPath -NoTypeInformation
        Write-Host "Cross-iteration run history saved to $mergedRunHistoryPath"
    }
}

Write-Host "Done."
