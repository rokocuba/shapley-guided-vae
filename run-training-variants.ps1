Set-StrictMode -Version Latest
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    Write-Error "Python not found: $python"
    exit 1
}

$outDir = 'analysis/output/training_runs'
$common = @(
    '--epochs', '200',
    '--batch-size', '256',
    '--hidden-dims', '1024,1024',
    '--latent-dim', '8',
    '--kl-target', '3.00',
    '--aux-loss-weight', '0.4',
    '--lr-plateau-factor', '0.7',
    '--lr-scheduler-monitor', 'val_pix_recon',
    '--output-dir', $outDir
)
$shapley = @(
    '--shapley-warmup-epochs', '50',
    '--shapley-min-sampling-phases', '3',
    '--shapley-group-size', '16',
    '--shapley-sampling-batch-size', '512'
)

Write-Host 'E0 baseline'
& $python .\main.py --training-type baseline @common
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host 'E0_0 pix_only'
& $python .\main.py --training-type pix_only --aux-loss-weight 0 @common
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host 'E1 Shapley baseline tactic'
& $python .\main.py --training-type shapley --shapley-tactic baseline @common @shapley
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host 'E2 Shapley marginal tactic'
& $python .\main.py --training-type shapley --shapley-tactic marginal @common @shapley
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host 'E3 Shapley conditional tactic'
& $python .\main.py --training-type shapley --shapley-tactic conditional @common @shapley
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host 'Plotting runs'
& $python .\analysis\plot_training_results.py --runs $outDir --out $outDir --all-runs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host 'Comparing simple pixel baselines'
& $python .\analysis\compare_mean_baselines.py --out-dir $outDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
