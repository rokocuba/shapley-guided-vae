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
) -> np.ndarray:
    loader = DataLoader(x_scaled, batch_size=batch_size, shuffle=False)
    model_was_training = model.training
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
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
    scaler: StandardScaler,
    out_png: str | Path,
    out_summary_csv: str | Path,
    device: str | torch.device,
    bins: int = 60,
) -> None:
    x_np = x_scaled.detach().cpu().numpy()
    x_hat_np = _predict_scaled(model, x_scaled, torch.device(device))
    x_denorm = scaler.inverse_transform(x_np)
    x_hat_denorm = scaler.inverse_transform(x_hat_np)
    diff = x_hat_denorm - x_denorm

    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
    axes_flat = axes.flatten()

    summary_rows: list[dict[str, float | str]] = []
    for i, name in enumerate(feature_names):
        ax = axes_flat[i]
        values = diff[:, i]
        ax.hist(values, bins=bins)
        ax.axvline(0.0, linewidth=1.0)
        ax.set_title(name)
        ax.set_xlabel("output - input")
        ax.set_ylabel("count")
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

    for j in range(len(feature_names), rows * cols):
        axes_flat[j].axis("off")

    fig.suptitle("Feature Difference Distributions (output - input, denormalized)")
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.97))
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    out_summary_csv = Path(out_summary_csv)
    out_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out_summary_csv, index=False)
