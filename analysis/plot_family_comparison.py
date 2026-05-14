from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_rows(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    keep_labels = {"global_mean", "pca_5", "pca_6"}
    baseline = df[
        (df["entry_type"] == "baseline") & (df["label"].isin(keep_labels))
    ].copy()
    vae = df[df["entry_type"] == "vae"].copy()
    return pd.concat([baseline, vae], ignore_index=True)


def _display_label(label: str) -> str:
    mapping = {
        "global_mean": "global mean",
        "pca_5": "PCA k=5",
        "pca_6": "PCA k=6",
        "beta000-h1624168-l5-e100": "beta=0 VAE, latent=5",
        "beta000-h1624168-l6-e100": "beta=0 VAE, latent=6",
        "beta000det-h1624168-l5-e100": "beta=0 det-latent VAE, latent=5",
        "beta000det-h1624168-l6-e100": "beta=0 det-latent VAE, latent=6",
        "beta040-h1624168-l5-e100-warm100": "beta=0.40 VAE, latent=5",
        "beta040-h1624168-l5-e100-warm100-cosine": "beta=0.40 VAE, latent=5, cosine",
        "beta040-h1624168-l6-e100-warm100": "beta=0.40 VAE, latent=6",
        "beta040-h1624168-l6-e100-warm100-cosine": "beta=0.40 VAE, latent=6, cosine",
    }
    return mapping.get(label, label)


def _family(label: str) -> str:
    if label == "global_mean":
        return "baseline"
    if label.startswith("pca_"):
        return "pca"
    if label.startswith("beta000det"):
        return "beta=0 det vae"
    if label.startswith("beta000"):
        return "beta=0 vae"
    return "beta>0 vae"


def build_plot_frame(csv_paths: list[Path]) -> pd.DataFrame:
    frames = [_load_rows(path) for path in csv_paths]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["label"], keep="last")
    combined["display_label"] = combined["label"].map(_display_label)
    combined["family"] = combined["label"].map(_family)
    family_order = {
        "pca": 0,
        "beta=0 det vae": 1,
        "beta=0 vae": 2,
        "beta>0 vae": 3,
        "baseline": 4,
    }
    combined["family_order"] = combined["family"].map(family_order)
    combined = combined.sort_values(
        ["family_order", "best_val_recon", "final_val_recon"]
    )
    return combined.reset_index(drop=True)


def plot_direct_comparison(df: pd.DataFrame, out_path: Path) -> None:
    colors = {
        "pca": "#08519c",
        "beta=0 det vae": "#756bb1",
        "beta=0 vae": "#2ca25f",
        "beta>0 vae": "#de2d26",
        "baseline": "#636363",
    }

    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(df))

    for family, group in df.groupby("family"):
        idx = group.index.to_numpy(dtype=int)
        ax.scatter(
            group["best_val_recon"],
            y[idx],
            s=90,
            color=colors[family],
            label=f"{family} best",
            zorder=3,
        )

    vae_mask = df["entry_type"] == "vae"
    ax.scatter(
        df.loc[vae_mask, "final_val_recon"],
        y[vae_mask.to_numpy()],
        s=90,
        color="#f28e2b",
        marker="s",
        label="VAE final",
        zorder=3,
    )

    for _, row in df[df["entry_type"] == "vae"].iterrows():
        row_idx = int(row.name)
        ax.hlines(
            y[row_idx],
            xmin=float(row["best_val_recon"]),
            xmax=float(row["final_val_recon"]),
            color="#f28e2b",
            linewidth=1.4,
            alpha=0.8,
            zorder=2,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(df["display_label"])
    ax.set_xlabel("reconstruction loss on held-out test split")
    ax.set_title(
        "Direct comparison: PCA, beta=0 det/stochastic VAE, beta>0 VAE, global mean"
    )
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--beta040-csv",
        type=Path,
        default=Path(
            "analysis/output/training_plots_beta040_warm100_beefy/mean_baseline_comparison.csv"
        ),
    )
    parser.add_argument(
        "--beta000-csv",
        type=Path,
        default=Path(
            "analysis/output/training_plots_beta000_beefy/mean_baseline_comparison.csv"
        ),
    )
    parser.add_argument(
        "--beta000det-csv",
        type=Path,
        default=Path(
            "analysis/output/training_plots_beta000det_beefy/mean_baseline_comparison.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis/output/model_family_comparison"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = [args.beta040_csv, args.beta000_csv]
    if args.beta000det_csv.exists():
        csv_paths.append(args.beta000det_csv)
    df = build_plot_frame(csv_paths)
    df.to_csv(args.out_dir / "family_comparison.csv", index=False)
    plot_direct_comparison(df, args.out_dir / "family_comparison.png")
    print(f"saved={args.out_dir}")


if __name__ == "__main__":
    main()
