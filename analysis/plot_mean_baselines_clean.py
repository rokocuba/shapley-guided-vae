"""Clean, minimalistic comparison: mean baselines vs VAE methods.
Single bar chart, Croatian labels, no titles, thesis-ready.
Uses means from all 50 iterations, plus pca_8, digit class mean, and global mean."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from shapley import FeatureBlockIndex
from utils import fit_feature_scaler, load_dataset_bundle, transform_features

# --- paths ---
RUNS = Path("analysis/output/training_runs")

# --- load VAE data ---
summary_csv = RUNS / "run_summary_all_iterations.csv"
if not summary_csv.exists():
    print(f"Missing {summary_csv}")
    exit(1)
df_vae = pd.read_csv(summary_csv)

# Only keep methods with val_pix_recon
vae_groups = {
    "pix_only": df_vae[df_vae["training_type"] == "pix_only"]["final_val_pix_recon"],
    "baseline\n(statički)": df_vae[
        (df_vae["training_type"] == "baseline") & (df_vae["shapley_tactic"] == "none")
    ]["final_val_pix_recon"],
    "Shapley\n(baseline)": df_vae[
        (df_vae["training_type"] == "shapley")
        & (df_vae["shapley_tactic"] == "baseline")
    ]["final_val_pix_recon"],
    "Shapley\n(marginal)": df_vae[
        (df_vae["training_type"] == "shapley")
        & (df_vae["shapley_tactic"] == "marginal")
    ]["final_val_pix_recon"],
    "Shapley\n(conditional)": df_vae[
        (df_vae["training_type"] == "shapley")
        & (df_vae["shapley_tactic"] == "conditional")
    ]["final_val_pix_recon"],
}

# --- compute simple baselines (global mean / digit class mean) ---
from shapley import FeatureBlockIndex
from utils import fit_feature_scaler, load_dataset_bundle, transform_features
import torch


def make_split_indices(n_rows: int, test_size: float, split_seed: int):
    n_test = int(n_rows * test_size)
    n_test = max(1, min(n_rows - 1, n_test))
    gen = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(n_rows, generator=gen).numpy()
    return np.asarray(perm[n_test:], dtype=int), np.asarray(perm[:n_test], dtype=int)


bundle = load_dataset_bundle(data_dir="data")
train_idx, test_idx = make_split_indices(len(bundle.x_raw), 0.2, 555)
scaler = fit_feature_scaler(bundle.x_raw[train_idx])
x_scaled = transform_features(bundle.x_raw, scaler)
x_train = x_scaled[train_idx]
x_test = x_scaled[test_idx]
labels_train = bundle.sample_labels[train_idx]
labels_test = bundle.sample_labels[test_idx]
block_index = FeatureBlockIndex.from_feature_groups(bundle.feature_groups)
pix_block = block_index.blocks[block_index.names.index("pix")]

# global mean predictor
pred_global = np.tile(x_train.mean(axis=0), (len(test_idx), 1))
mse_global = ((pred_global - x_test) ** 2).mean(axis=0)
pix_global = float(mse_global[pix_block.start : pix_block.stop].mean())

# digit-class-mean predictor
cls_names = sorted(set(labels_train))
class_means = {cls: x_train[labels_train == cls].mean(axis=0) for cls in cls_names}
pred_class = np.vstack([class_means[lbl] for lbl in labels_test])
mse_class = ((pred_class - x_test) ** 2).mean(axis=0)
pix_class = float(mse_class[pix_block.start : pix_block.stop].mean())

# pca_8 predictor
from sklearn.decomposition import PCA

pca = PCA(n_components=8, svd_solver="full")
pca.fit(x_train)
pred_pca = pca.inverse_transform(pca.transform(x_test))
mse_pca = ((pred_pca - x_test) ** 2).mean(axis=0)
pix_pca = float(mse_pca[pix_block.start : pix_block.stop].mean())

simple = {
    "globalna\nsredina": pix_global,
    "sredina\nklase znamenke": pix_class,
    "PCA n=8": pix_pca,
}

# --- combine and sort ascending by mean ---
color_map = {
    "globalna\nsredina": "#aaaaaa",
    "sredina\nklase znamenke": "#aaaaaa",
    "PCA n=8": "#aaaaaa",
    "pix_only": "#2ca02c",
    "baseline\n(statički)": "#ff7f0e",
    "Shapley\n(baseline)": "#1f77b4",
    "Shapley\n(marginal)": "#9467bd",
    "Shapley\n(conditional)": "#d62728",
}

all_items = []
for k, v in simple.items():
    all_items.append((k, v, 0.0, color_map[k]))
for lb, series in vae_groups.items():
    all_items.append((lb, series.mean(), series.std(), color_map[lb]))

all_items.sort(key=lambda t: t[1])

labels = [t[0] for t in all_items]
all_means = [t[1] for t in all_items]
all_stds = [t[2] for t in all_items]
colors = [t[3] for t in all_items]

# --- build bar chart ---
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 200,
    }
)

fig, ax = plt.subplots(figsize=(8.0, 5.0))
x = np.arange(len(labels))
bars = ax.bar(x, all_means, color=colors, edgecolor="black", linewidth=0.6, width=0.65)

# error bars only for items with std > 0 (VAE methods)
vae_mask = [s > 0.0 for s in all_stds]
ax.errorbar(
    x[vae_mask],
    np.array(all_means)[vae_mask],
    yerr=np.array(all_stds)[vae_mask],
    fmt="none",
    ecolor="black",
    capsize=4,
    linewidth=1.0,
)

# numeric labels above each bar
y_max = max(all_means)
y_offset = y_max * 0.015
for xi, yi in zip(x, all_means):
    ax.text(xi, yi + y_offset, f"{yi:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("gubitak rekonstrukcije pix")
ax.grid(axis="y", alpha=0.25)

# give headroom for value labels
ax.set_ylim(0, y_max * 1.12)

fig.tight_layout()

OUT = Path("analysis/pictures/mean_baseline_vs_vae.png")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved -> {OUT}")
