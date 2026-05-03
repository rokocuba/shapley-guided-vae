from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader


def _predict_scaled(
    model: nn.Module,
    x_scaled: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
    use_mean_latent: bool = True,
) -> np.ndarray:
    loader = DataLoader(x_scaled, batch_size=batch_size, shuffle=False)
    model_was_training = model.training
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            if use_mean_latent:
                mu, _ = model.encoder(xb)
                x_hat = model.decoder(mu)
            else:
                x_hat, _, _ = model(xb)
            preds.append(x_hat.detach().cpu().numpy())
    if model_was_training:
        model.train()
    return np.concatenate(preds, axis=0)


def plot_feature_difference_distributions(
    *,
    model: nn.Module,
    x_scaled: torch.Tensor,
    feature_names: list[str],
    scaler: StandardScaler | None,
    out_png: str | Path,
    out_summary_csv: str | Path,
    device: str | torch.device,
    bins: int = 60,
    use_mean_latent: bool = True,
    title_suffix: str = "",
    max_plots: int = 24,
) -> None:
    x_np = x_scaled.detach().cpu().numpy()
    x_hat_np = _predict_scaled(
        model,
        x_scaled,
        torch.device(device),
        use_mean_latent=use_mean_latent,
    )
    if scaler is None:
        x_denorm = x_np
        x_hat_denorm = x_hat_np
    else:
        x_denorm = scaler.inverse_transform(x_np)
        x_hat_denorm = scaler.inverse_transform(x_hat_np)
    diff = x_hat_denorm - x_denorm

    summary_rows: list[dict[str, float | str]] = []
    for i, name in enumerate(feature_names):
        values = diff[:, i]
        summary_rows.append(
            {
                "feature": name,
                "mean_diff": float(values.mean()),
                "std_diff": float(values.std()),
                "mae": float(np.abs(values).mean()),
                "q05": float(np.quantile(values, 0.05)),
                "q50": float(np.quantile(values, 0.50)),
                "q95": float(np.quantile(values, 0.95)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    selected_indices = np.arange(len(feature_names))
    selected_suffix = ""
    if len(feature_names) > max_plots:
        selected_indices = summary_df["mae"].to_numpy().argsort()[::-1][:max_plots]
        selected_indices = np.sort(selected_indices)
        selected_suffix = f" - top {max_plots} features by MAE"

    n_plots = len(selected_indices)
    cols = min(4, max(1, n_plots))
    rows = int(np.ceil(n_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_flat = np.atleast_1d(axes).flatten()

    for plot_index, feature_index in enumerate(selected_indices):
        ax = axes_flat[plot_index]
        values = diff[:, feature_index]
        ax.hist(values, bins=bins)
        ax.axvline(0.0, linewidth=1.0)
        ax.set_title(feature_names[feature_index])
        ax.set_xlabel("output - input")
        ax.set_ylabel("count")

    for j in range(n_plots, rows * cols):
        axes_flat[j].axis("off")

    suffix = f" - {title_suffix}" if title_suffix else ""
    fig.suptitle(f"Feature Difference Distributions (output - input){selected_suffix}{suffix}")
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.97))
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    out_summary_csv = Path(out_summary_csv)
    out_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_summary_csv, index=False)
