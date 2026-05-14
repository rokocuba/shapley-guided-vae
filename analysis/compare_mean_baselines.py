from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, random_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import fit_feature_scaler, load_dataset_bundle, transform_features


def load_scaled_data_and_metadata(
    data_dir: Path,
    dataset_name: str,
    test_size: float,
    split_seed: int,
    normalize_features: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
    str | None,
    np.ndarray,
]:
    bundle = load_dataset_bundle(data_dir=data_dir, dataset_name=dataset_name)
    train_idx, test_idx = make_split_indices(len(bundle.x_raw), test_size, split_seed)
    scaler = fit_feature_scaler(bundle.x_raw[train_idx]) if normalize_features else None
    x_scaled = transform_features(bundle.x_raw, scaler)
    labels = (
        bundle.sample_labels.astype(object, copy=False)
        if bundle.sample_labels is not None
        else np.array([], dtype=object)
    )
    return (
        x_scaled,
        train_idx,
        test_idx,
        bundle.feature_names,
        bundle.feature_groups,
        bundle.label_name,
        labels,
    )


def make_split_indices(
    n_rows: int, test_size: float, split_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    base = TensorDataset(torch.arange(n_rows))
    n_test = int(n_rows * test_size)
    n_test = max(1, min(n_rows - 1, n_test))
    n_train = n_rows - n_test
    generator = torch.Generator().manual_seed(split_seed)
    train_set, test_set = random_split(base, [n_train, n_test], generator=generator)
    return np.asarray(train_set.indices, dtype=int), np.asarray(
        test_set.indices, dtype=int
    )


def evaluate_mean_predictors(
    x_scaled: np.ndarray,
    labels: np.ndarray,
    label_name: str | None,
    feature_names: list[str],
    feature_groups: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = x_scaled[train_idx]
    x_test = x_scaled[test_idx]

    predictors = {
        "global_mean": np.repeat(x_train.mean(axis=0)[None, :], len(test_idx), axis=0)
    }
    if labels.size > 0:
        labels_train = labels[train_idx]
        labels_test = labels[test_idx]
        class_means = {
            cls: x_train[labels_train == cls].mean(axis=0)
            for cls in np.unique(labels_train)
        }
        predictor_name = f"{label_name or 'class'}_mean"
        predictors[predictor_name] = np.vstack(
            [class_means[label] for label in labels_test]
        )

    summary_rows: list[dict[str, float | int | str]] = []
    feature_rows: list[dict[str, float | str]] = []
    for predictor_name, preds in predictors.items():
        sq_error = (preds - x_test) ** 2
        feature_mse = sq_error.mean(axis=0)
        recon = float(feature_mse.mean())
        mae = float(np.abs(preds - x_test).mean())
        rmse = float(np.sqrt(sq_error.mean()))
        summary_rows.append(
            {
                "entry_type": "baseline",
                "label": predictor_name,
                "final_val_recon": recon,
                "best_val_recon": recon,
                "val_mae": mae,
                "val_rmse": rmse,
                "n_test": int(len(test_idx)),
            }
        )
        for feature_name, group_name, mse_value in zip(
            feature_names, feature_groups, feature_mse
        ):
            feature_rows.append(
                {
                    "predictor": predictor_name,
                    "feature": feature_name,
                    "feature_group": group_name,
                    "feature_mse": float(mse_value),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(feature_rows)


def evaluate_pca_predictors(
    x_scaled: np.ndarray,
    feature_names: list[str],
    feature_groups: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    latent_dims: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = x_scaled[train_idx]
    x_test = x_scaled[test_idx]

    summary_rows: list[dict[str, float | int | str]] = []
    feature_rows: list[dict[str, float | str]] = []
    for latent_dim in latent_dims:
        pca = PCA(n_components=latent_dim, svd_solver="full")
        pca.fit(x_train)
        preds = pca.inverse_transform(pca.transform(x_test))
        sq_error = (preds - x_test) ** 2
        feature_mse = sq_error.mean(axis=0)
        recon = float(feature_mse.mean())
        mae = float(np.abs(preds - x_test).mean())
        rmse = float(np.sqrt(sq_error.mean()))
        summary_rows.append(
            {
                "entry_type": "baseline",
                "label": f"pca_{latent_dim}",
                "final_val_recon": recon,
                "best_val_recon": recon,
                "val_mae": mae,
                "val_rmse": rmse,
                "n_test": int(len(test_idx)),
            }
        )
        for feature_name, group_name, mse_value in zip(
            feature_names, feature_groups, feature_mse
        ):
            feature_rows.append(
                {
                    "predictor": f"pca_{latent_dim}",
                    "feature": feature_name,
                    "feature_group": group_name,
                    "feature_mse": float(mse_value),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(feature_rows)


def _parse_optional_dim_list(value: str) -> list[int]:
    raw_parts = [part.strip() for part in value.split(",") if part.strip()]
    dims = [int(part) for part in raw_parts]
    if any(dim <= 0 for dim in dims):
        raise ValueError("PCA dims must be positive integers.")
    return dims


def _resolve_pca_dims(run_summary_path: Path, extra_pca_dims: list[int]) -> list[int]:
    inferred_dims: list[int] = []
    if run_summary_path.exists():
        summary = pd.read_csv(run_summary_path)
        if "latent_dim" in summary.columns:
            latent_series = pd.to_numeric(summary["latent_dim"], errors="coerce")
            inferred_dims = sorted(
                {
                    int(value)
                    for value in latent_series.dropna().tolist()
                    if int(value) > 0
                }
            )

    base_dims = inferred_dims if inferred_dims else [5, 6]
    return sorted({*base_dims, *extra_pca_dims})


def _extract_pca_dim(label: str) -> int | None:
    if not label.startswith("pca_"):
        return None
    try:
        return int(label.split("_", maxsplit=1)[1])
    except ValueError:
        return None


def build_model_frame(out_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(out_dir / "run_summary.csv")
    history = pd.read_csv(out_dir / "run_history.csv")
    best = (
        history.groupby(["run_id", "training_type"], as_index=False)["val_recon"]
        .min()
        .rename({"val_recon": "best_val_recon"}, axis=1)
    )
    merged = summary.merge(best, on=["run_id", "training_type"], how="left")
    merged = merged.rename(columns={"training_type": "label"})
    merged["entry_type"] = "vae"
    return merged[
        [
            "entry_type",
            "run_id",
            "label",
            "final_val_recon",
            "best_val_recon",
            "final_val_kl",
            "duration_sec",
        ]
    ].sort_values("best_val_recon")


def plot_reconstruction_comparison(
    comparison: pd.DataFrame, out_path: Path, dataset_name: str
) -> None:
    plot_df = comparison.copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(plot_df))
    ax.scatter(
        plot_df["best_val_recon"], y, s=70, label="best val_recon", color="#1f77b4"
    )
    ax.scatter(
        plot_df["final_val_recon"],
        y,
        s=70,
        label="final val_recon",
        color="#ff7f0e",
        marker="s",
    )

    for _, row in plot_df[plot_df["entry_type"] == "baseline"].iterrows():
        ax.axvline(row["best_val_recon"], linewidth=1.2, linestyle="--", alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("validation reconstruction loss (scaled feature MSE)")
    ax.set_title(f"{dataset_name} runs vs simple baselines")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_feature_baselines(
    feature_df: pd.DataFrame, out_path: Path, dataset_name: str
) -> None:
    n_features = feature_df["feature"].nunique()
    if n_features <= 40:
        pivot = feature_df.pivot(
            index="feature", columns="predictor", values="feature_mse"
        )
        ylabel = "feature MSE on test split"
        title = f"{dataset_name} per-feature baseline error"
    else:
        pivot = (
            feature_df.groupby(["feature_group", "predictor"], as_index=False)[
                "feature_mse"
            ]
            .mean()
            .pivot(index="feature_group", columns="predictor", values="feature_mse")
        )
        ylabel = "mean feature-group MSE on test split"
        title = f"{dataset_name} feature-group baseline error"

    baseline_cols = [
        col
        for col in ["global_mean", "wine_type_mean", "digit_class_mean"]
        if col in pivot.columns
    ]
    pca_cols = sorted(
        [col for col in pivot.columns if _extract_pca_dim(col) is not None],
        key=lambda col: cast(int, _extract_pca_dim(col)),
    )
    remaining = [
        col for col in pivot.columns if col not in baseline_cols and col not in pca_cols
    ]
    columns = baseline_cols + pca_cols + remaining
    anchor_col = pca_cols[-1] if pca_cols else columns[0]
    pivot = pivot[columns].sort_values(anchor_col, ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(pivot.index))
    width = max(0.12, 0.8 / max(1, len(columns)))
    offsets = np.linspace(
        -(len(columns) - 1) * width / 2,
        (len(columns) - 1) * width / 2,
        num=len(columns),
    )
    color_map = {
        "global_mean": "#c6dbef",
        "wine_type_mean": "#6baed6",
        "digit_class_mean": "#74c476",
    }
    if pca_cols:
        pca_colors = plt.cm.Blues(np.linspace(0.55, 0.9, num=len(pca_cols)))
        for col, color in zip(pca_cols, pca_colors):
            color_map[col] = color
    for offset, col in zip(offsets, columns):
        ax.bar(
            x + offset,
            pivot[col],
            width=width,
            label=col,
            color=color_map.get(col, None),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dataset-name", type=str, default="mfeat")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis/output/training_plots_beta040_warm100_beefy"),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--no-normalize-features", action="store_true")
    parser.add_argument(
        "--extra-pca-dims",
        type=str,
        default="",
        help="Comma-separated extra PCA component counts to plot in addition to inferred latent dims.",
    )
    args = parser.parse_args()

    extra_pca_dims = _parse_optional_dim_list(args.extra_pca_dims)
    pca_dims = _resolve_pca_dims(args.out_dir / "run_summary.csv", extra_pca_dims)

    (
        x_scaled,
        train_idx,
        test_idx,
        feature_names,
        feature_groups,
        label_name,
        labels,
    ) = load_scaled_data_and_metadata(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        test_size=args.test_size,
        split_seed=args.split_seed,
        normalize_features=not args.no_normalize_features,
    )
    baseline_summary, feature_df = evaluate_mean_predictors(
        x_scaled,
        labels,
        label_name,
        feature_names,
        feature_groups,
        train_idx,
        test_idx,
    )
    pca_summary, pca_feature_df = evaluate_pca_predictors(
        x_scaled,
        feature_names,
        feature_groups,
        train_idx,
        test_idx,
        latent_dims=pca_dims,
    )
    model_df = build_model_frame(args.out_dir)

    baseline_summary = pd.concat([baseline_summary, pca_summary], ignore_index=True)
    feature_df = pd.concat([feature_df, pca_feature_df], ignore_index=True)

    class_baseline_candidates = [
        label
        for label in baseline_summary["label"]
        if label.endswith("_mean") and label != "global_mean"
    ]
    class_baseline_label = (
        class_baseline_candidates[0] if class_baseline_candidates else None
    )
    class_baseline = None
    if class_baseline_label is not None:
        class_values = cast(
            pd.Series,
            baseline_summary.loc[
                baseline_summary["label"] == class_baseline_label, "best_val_recon"
            ],
        )
        class_baseline = float(class_values.iloc[0])
    global_values = cast(
        pd.Series,
        baseline_summary.loc[
            baseline_summary["label"] == "global_mean", "best_val_recon"
        ],
    )
    global_baseline = float(global_values.iloc[0])

    if not model_df.empty:
        if class_baseline is not None and class_baseline_label is not None:
            model_df[f"gap_vs_{class_baseline_label}"] = (
                model_df["best_val_recon"] - class_baseline
            )
            model_df[f"relative_to_{class_baseline_label}_pct"] = (
                100.0 * model_df[f"gap_vs_{class_baseline_label}"] / class_baseline
            )
        model_df["gap_vs_global_mean"] = model_df["best_val_recon"] - global_baseline
        model_df["relative_to_global_mean_pct"] = (
            100.0 * model_df["gap_vs_global_mean"] / global_baseline
        )

    if class_baseline is not None and class_baseline_label is not None:
        baseline_summary[f"gap_vs_{class_baseline_label}"] = (
            baseline_summary["best_val_recon"] - class_baseline
        )
        baseline_summary[f"relative_to_{class_baseline_label}_pct"] = (
            100.0 * baseline_summary[f"gap_vs_{class_baseline_label}"] / class_baseline
        )
    baseline_summary["gap_vs_global_mean"] = (
        baseline_summary["best_val_recon"] - global_baseline
    )
    baseline_summary["relative_to_global_mean_pct"] = (
        100.0 * baseline_summary["gap_vs_global_mean"] / global_baseline
    )

    comparison = pd.concat([baseline_summary, model_df], ignore_index=True, sort=False)
    comparison = comparison.sort_values(
        ["best_val_recon", "final_val_recon"], na_position="last"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.out_dir / "mean_baseline_comparison.csv", index=False)
    feature_df.to_csv(args.out_dir / "mean_baseline_feature_mse.csv", index=False)
    plot_reconstruction_comparison(
        comparison, args.out_dir / "mean_baseline_vs_vae.png", args.dataset_name
    )
    plot_feature_baselines(
        feature_df, args.out_dir / "mean_baseline_feature_mse.png", args.dataset_name
    )
    print(f"saved={args.out_dir} pca_dims={pca_dims}")


if __name__ == "__main__":
    main()
