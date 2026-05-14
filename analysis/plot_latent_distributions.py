from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import VAE, VAEConfig
from utils import load_dataset_bundle


def _load_metadata(run_dir: Path) -> dict:
    return json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))


def _make_split_indices(
    n_total: int,
    test_size: float,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_test = int(n_total * test_size)
    n_test = max(1, min(n_total - 1, n_test))
    generator = torch.Generator().manual_seed(split_seed)
    permutation = torch.randperm(n_total, generator=generator)
    test_idx = permutation[:n_test].numpy()
    train_idx = permutation[n_test:].numpy()
    return train_idx, test_idx


def _scale_from_metadata(x_raw: np.ndarray, metadata: dict) -> np.ndarray:
    if not metadata["config"].get("normalize_features", True):
        return x_raw.astype("float32", copy=True)
    mean = np.asarray(metadata["data"]["scaler_mean"], dtype=np.float32)
    scale = np.asarray(metadata["data"]["scaler_scale"], dtype=np.float32)
    x_raw = x_raw.astype("float32", copy=False)
    return ((x_raw - mean) / scale).astype("float32", copy=False)


def _select_split_indices(metadata: dict, n_total: int, split: str) -> np.ndarray:
    if split == "all":
        return np.arange(n_total, dtype=int)
    train_idx, test_idx = _make_split_indices(
        n_total=n_total,
        test_size=float(metadata["config"].get("test_size", 0.2)),
        split_seed=int(metadata["config"].get("split_seed", 42)),
    )
    return train_idx if split == "train" else test_idx


def _build_model(run_dir: Path, metadata: dict, device: torch.device) -> VAE:
    cfg = metadata["config"]
    model = VAE(
        VAEConfig(
            input_dim=int(metadata["data"]["n_features"]),
            hidden_dims=tuple(int(v) for v in cfg.get("hidden_dims", (16, 8))),
            latent_dim=int(cfg["latent_dim"]),
            input_dropout=float(cfg.get("input_dropout", 0.0)),
            deterministic_latent=bool(cfg.get("deterministic_latent", False)),
        )
    )
    model.load_state_dict(
        torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    return model


def _collect_latent_arrays(
    model: VAE,
    x_scaled: np.ndarray,
    batch_size: int,
    device: torch.device,
    sample_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(torch.from_numpy(x_scaled), batch_size=batch_size, shuffle=False)
    mu_batches: list[np.ndarray] = []
    logvar_batches: list[np.ndarray] = []
    z_batches: list[np.ndarray] = []

    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sample_seed)

    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            mu, logvar = model.encoder(xb)
            if model.config.deterministic_latent:
                z = mu
            else:
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
            mu_batches.append(mu.detach().cpu().numpy())
            logvar_batches.append(logvar.detach().cpu().numpy())
            z_batches.append(z.detach().cpu().numpy())

    return (
        np.concatenate(mu_batches, axis=0),
        np.concatenate(logvar_batches, axis=0),
        np.concatenate(z_batches, axis=0),
    )


def _summarize_latent_array(values: np.ndarray, stat_name: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for dim_index in range(values.shape[1]):
        column = values[:, dim_index]
        rows.append(
            {
                "stat": stat_name,
                "latent_dim": dim_index,
                "mean": float(column.mean()),
                "std": float(column.std()),
                "min": float(column.min()),
                "q05": float(np.quantile(column, 0.05)),
                "q25": float(np.quantile(column, 0.25)),
                "q50": float(np.quantile(column, 0.50)),
                "q75": float(np.quantile(column, 0.75)),
                "q95": float(np.quantile(column, 0.95)),
                "max": float(column.max()),
            }
        )
    return pd.DataFrame(rows)


def _make_latent_frame(
    *,
    row_indices: np.ndarray,
    labels: np.ndarray | None,
    label_name: str | None,
    mu: np.ndarray,
    logvar: np.ndarray,
    z: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame({"row_index": row_indices})
    if labels is not None and label_name is not None:
        frame[label_name] = labels
    for dim_index in range(mu.shape[1]):
        frame[f"mu_{dim_index}"] = mu[:, dim_index]
        frame[f"logvar_{dim_index}"] = logvar[:, dim_index]
        frame[f"z_{dim_index}"] = z[:, dim_index]
    return frame


def _plot_histograms(
    values: np.ndarray,
    *,
    title: str,
    out_path: Path,
    bins: int,
    color: str,
) -> None:
    latent_dim = values.shape[1]
    cols = min(3, max(1, latent_dim))
    rows = int(np.ceil(latent_dim / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    axes_flat = np.atleast_1d(axes).flatten()

    for dim_index in range(latent_dim):
        ax = axes_flat[dim_index]
        column = values[:, dim_index]
        ax.hist(column, bins=bins, color=color, alpha=0.85)
        ax.axvline(0.0, color="black", linewidth=0.9, alpha=0.6)
        ax.axvline(column.mean(), color="crimson", linestyle="--", linewidth=1.0)
        ax.set_title(f"dim {dim_index} | mean={column.mean():.3f} std={column.std():.3f}")
        ax.set_xlabel("value")
        ax.set_ylabel("count")

    for dim_index in range(latent_dim, rows * cols):
        axes_flat[dim_index].axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "all"],
        default="test",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    metadata = _load_metadata(run_dir)
    data_dir = Path(metadata["data"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    bundle = load_dataset_bundle(
        data_dir=data_dir,
        dataset_name=metadata["data"]["dataset_name"],
    )
    x_scaled = _scale_from_metadata(bundle.x_raw, metadata)
    split_indices = _select_split_indices(metadata, x_scaled.shape[0], args.split)
    selected_labels = (
        None
        if bundle.sample_labels is None
        else bundle.sample_labels[split_indices].copy()
    )
    x_selected = x_scaled[split_indices]

    device = torch.device(args.device)
    model = _build_model(run_dir, metadata, device)
    mu, logvar, z = _collect_latent_arrays(
        model,
        x_selected,
        batch_size=args.batch_size,
        device=device,
        sample_seed=args.sample_seed,
    )

    out_dir = args.out if args.out is not None else run_dir / "latent_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{metadata['run_id']}__{args.split}"
    _plot_histograms(
        mu,
        title=f"Latent mu distributions ({metadata['run_id']}, split={args.split})",
        out_path=out_dir / f"{base_name}__latent_mu_distributions.png",
        bins=args.bins,
        color="#1f77b4",
    )
    _plot_histograms(
        logvar,
        title=f"Latent logvar distributions ({metadata['run_id']}, split={args.split})",
        out_path=out_dir / f"{base_name}__latent_logvar_distributions.png",
        bins=args.bins,
        color="#ff7f0e",
    )
    _plot_histograms(
        z,
        title=f"Latent z distributions ({metadata['run_id']}, split={args.split})",
        out_path=out_dir / f"{base_name}__latent_z_distributions.png",
        bins=args.bins,
        color="#2ca02c",
    )

    summary = pd.concat(
        [
            _summarize_latent_array(mu, "mu"),
            _summarize_latent_array(logvar, "logvar"),
            _summarize_latent_array(z, "z"),
        ],
        ignore_index=True,
    )
    summary.to_csv(out_dir / f"{base_name}__latent_summary.csv", index=False)

    latent_frame = _make_latent_frame(
        row_indices=split_indices,
        labels=selected_labels,
        label_name=bundle.label_name,
        mu=mu,
        logvar=logvar,
        z=z,
    )
    latent_frame.to_csv(out_dir / f"{base_name}__latents.csv", index=False)
    print(f"saved_latent_analysis={out_dir}")


if __name__ == "__main__":
    main()