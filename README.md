# Shapley-Guided Pixel-Auxiliary VAE

BSc thesis code for dynamic auxiliary reconstruction scaling in a Variational Autoencoder using online Shapley values.

## Method

Dataset: UCI Multiple Features (`mfeat`), 2000 digit samples, 649 features.

| Block | Meaning                     | Features | Role                            |
| ----- | --------------------------- | -------: | ------------------------------- |
| `fou` | Fourier coefficients        |       76 | encoder input, auxiliary target |
| `fac` | Profile correlations        |      216 | encoder input, auxiliary target |
| `kar` | Karhunen-Loeve coefficients |       64 | encoder input, auxiliary target |
| `pix` | Pixel averages              |      240 | encoder input, primary target   |
| `zer` | Zernike moments             |       47 | encoder input, auxiliary target |
| `mor` | Morphological features      |        6 | encoder input, auxiliary target |

The encoder consumes all 649 normalized features. The decoder has one shared trunk and six output heads, concatenated back to a flat 649-vector for compatibility.

Training objective:

```text
L = pix_recon
    + aux_loss_weight * sum(w_g * aux_recon_g for g in {fou,fac,kar,zer,mor})
    + beta * KL
```

Default `aux_loss_weight` is `0.2`. Static baseline uses `w_g = 1/5`. Shapley runs estimate five auxiliary-block Shapley values from a pixel-only payoff and distribute one auxiliary weight budget over the five auxiliary losses. Pixel is the target, not a Shapley player, and its loss stays fixed at coefficient `1`.

Optional lambda decay:

```powershell
.\.venv\Scripts\python.exe main.py --decay-aux-loss-weight-after-kl-warmup
```

This keeps `aux_loss_weight` fixed until the resolved KL target warm-up epoch, then decays it to `--aux-loss-weight-decay-final` over the remaining epochs. Use `--aux-loss-weight-decay-curve exponential` for a faster early drop. With the default KL-target controller, the decay starts at `round(6 * epochs)`.

## Defaults

| Parameter                       |                 Value |
| ------------------------------- | --------------------: |
| Epochs                          |                `2000` |
| Batch size                      |                 `256` |
| Hidden dims                     |           `1024,1024` |
| Latent dim                      |                   `5` |
| KL target                       |                 `3.0` |
| KL target warm-up               | `round(0.6 * epochs)` |
| Warm-up before Shapley sampling |          `100` epochs |
| Dynamic activation delay        |   `5` sampling phases |

## Run

```powershell
.\.venv\Scripts\python.exe main.py
```

Core experiment matrix:

```powershell
.\run-training-variants.ps1
```

Manual runs:

```powershell
.\.venv\Scripts\python.exe main.py --training-type baseline
.\.venv\Scripts\python.exe main.py --training-type shapley --shapley-tactic baseline
.\.venv\Scripts\python.exe main.py --training-type shapley --shapley-tactic marginal
.\.venv\Scripts\python.exe main.py --training-type shapley --shapley-tactic conditional
```

## Artifacts

Each run writes `metadata.json`, `history.csv`, `callback_timing.csv`, `model.pt`, and feature-difference diagnostics.

Shapley runs also write:

- `shapley_weights.csv`
- `shapley_node_stats.csv`
- `shapley_phase_timing.csv`

Primary comparison metric: `val_pix_recon` over elapsed wall-clock seconds.

Plot runs:

```powershell
.\.venv\Scripts\python.exe analysis\plot_training_results.py --runs analysis/output/training_runs --out analysis/output/training_runs --all-runs
```

The plot script also writes `pixel_threshold_times.csv` with epoch and elapsed seconds to reach `1.05 * min(val_pix_recon)` from the static baseline.
