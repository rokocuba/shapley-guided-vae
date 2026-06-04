# Activate the local virtual environment and run baseline + Shapley variants
Set-StrictMode -Version Latest
Set-Location $PSScriptRoot

$activatePath = Join-Path $PSScriptRoot '.venv\Scripts\Activate.ps1'
if (-not (Test-Path $activatePath)) {
    Write-Error "Virtual environment activation script not found: $activatePath"
    exit 1
}

# Define settings as arrays for reuse with splatting
$settings = @('--lr', '1e-4', '--lr-min', '1e-6', '--epochs', '200', '--latent-dim', '5', '--hidden-dims', '64,16')
$shapleySettings = @('--shapley-warmup-epochs', '30')

Write-Host "Activating venv..."
& $activatePath

Write-Host "Running baseline training..."
python .\main.py @settings

Write-Host "Running Shapley training..."
python .\main.py --training-type shapley @settings @shapleySettings

Write-Host "Running Shapley marginal training..."
python .\main.py --training-type shapley --shapley-tactic marginal @settings @shapleySettings

Write-Host "Running Shapley conditional training..."
python .\main.py --training-type shapley --shapley-tactic conditional @settings @shapleySettings

Write-Host "Plotting training results..."
python .\analysis\plot_training_results.py --runs .\analysis\output\training_runs --out .\analysis\output\training_runs --all-runs

python .\analysis\compare_mean_baselines.py --out-dir .\analysis\output\training_runs
