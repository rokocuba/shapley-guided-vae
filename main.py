from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from models import VAE, VAEConfig
from training import (
    BetaWarmupCallback,
    DynamicWeightedVAELoss,
    Trainer,
    plot_feature_difference_distributions,
    save_training_run,
    update_training_run_artifacts,
)
from utils import fit_feature_scaler, load_dataset_bundle, transform_features


def _make_split_indices(
    n_total: int,
    test_size: float,
    split_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in the open interval (0, 1).")
    n_test = int(n_total * test_size)
    n_test = max(1, min(n_total - 1, n_test))
    generator = torch.Generator().manual_seed(split_seed)
    permutation = torch.randperm(n_total, generator=generator)
    test_idx = permutation[:n_test]
    train_idx = permutation[n_test:]
    return train_idx, test_idx


def _parse_hidden_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims:
        raise ValueError("hidden_dims must contain at least one integer.")
    return dims


def run_baseline(
    data_dir: str | Path = "data",
    dataset_name: str = "mfeat",
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    lr_scheduler: str | None = None,
    lr_min: float = 1e-5,
    latent_dim: int = 5,
    hidden_dims: tuple[int, ...] | list[int] = (16, 24, 16, 8),
    test_size: float = 0.2,
    split_seed: int = 42,
    input_dropout: float = 0.0,
    normalize_features: bool = True,
    deterministic_latent: bool = False,
    beta: float = 0.15,
    beta_warmup: str | None = None,
    beta_start: float = 0.0,
    beta_warmup_epochs: int = 30,
    device: str | None = None,
    training_type: str = "baseline",
    shapley_tactic: str | None = None,
    output_dir: str | Path = "analysis/output/training_runs",
) -> list[dict[str, float]]:
    bundle = load_dataset_bundle(data_dir=data_dir, dataset_name=dataset_name)
    train_idx, test_idx = _make_split_indices(
        n_total=bundle.x_raw.shape[0],
        test_size=test_size,
        split_seed=split_seed,
    )
    scaler = (
        fit_feature_scaler(bundle.x_raw[train_idx.numpy()])
        if normalize_features
        else None
    )
    x_scaled = torch.from_numpy(transform_features(bundle.x_raw, scaler))
    hidden_dims_tuple = tuple(int(v) for v in hidden_dims)
    if len(hidden_dims_tuple) == 0:
        raise ValueError("hidden_dims must contain at least one layer width.")
    model = VAE(
        VAEConfig(
            input_dim=x_scaled.shape[1],
            hidden_dims=hidden_dims_tuple,
            latent_dim=latent_dim,
            input_dropout=input_dropout,
            deterministic_latent=deterministic_latent,
        )
    )
    loss_fn = DynamicWeightedVAELoss(input_dim=x_scaled.shape[1], beta=beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if lr_scheduler is not None:
        scheduler_name = lr_scheduler.strip().lower()
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs),
                eta_min=float(lr_min),
            )
        else:
            raise ValueError(
                "Unknown lr_scheduler. Supported values: 'cosine' or None."
            )
    callbacks = []
    if beta_warmup is not None:
        callbacks.append(
            BetaWarmupCallback(
                target_beta=beta,
                warmup_epochs=beta_warmup_epochs,
                strategy=beta_warmup,
                start_beta=beta_start,
            )
        )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        callbacks=callbacks,
        scheduler=scheduler,
    )
    train_tensor = x_scaled[train_idx]
    test_tensor = x_scaled[test_idx]
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)
    x_test = test_tensor
    history = trainer.fit(train_loader, epochs=epochs, val_loader=test_loader)

    feature_group_names: list[str] = []
    feature_group_sizes: list[int] = []
    for group_name in bundle.feature_groups:
        if not feature_group_names or feature_group_names[-1] != group_name:
            feature_group_names.append(group_name)
            feature_group_sizes.append(1)
        else:
            feature_group_sizes[-1] += 1
    class_names = None
    if bundle.sample_labels is not None:
        class_names = [str(value) for value in pd.unique(bundle.sample_labels)]

    run_dir = save_training_run(
        model=model,
        history=history,
        state=trainer.state,
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "lr_scheduler": lr_scheduler,
            "lr_min": lr_min,
            "latent_dim": latent_dim,
            "dataset_name": bundle.dataset_name,
            "test_size": test_size,
            "split_seed": split_seed,
            "hidden_dims": list(model.config.hidden_dims),
            "input_dropout": input_dropout,
            "normalize_features": normalize_features,
            "deterministic_latent": deterministic_latent,
            "beta": beta,
            "beta_warmup": beta_warmup,
            "beta_start": beta_start,
            "beta_warmup_epochs": beta_warmup_epochs,
            "device": str(trainer.device),
        },
        data_info={
            "data_dir": str(data_dir),
            "dataset_name": bundle.dataset_name,
            "n_rows": int(x_scaled.shape[0]),
            "n_rows_train": int(train_tensor.shape[0]),
            "n_rows_test": int(test_tensor.shape[0]),
            "n_features": int(x_scaled.shape[1]),
            "feature_names": bundle.feature_names,
            "feature_group_names": feature_group_names,
            "feature_group_sizes": feature_group_sizes,
            "label_name": bundle.label_name,
            "class_names": class_names,
            "scaler_mean": None if scaler is None else [float(v) for v in scaler.mean_],
            "scaler_scale": None if scaler is None else [float(v) for v in scaler.scale_],
        },
        output_dir=output_dir,
        training_type=training_type,
        shapley_tactic=shapley_tactic,
    )

    run_dir = Path(run_dir)
    residual_png = run_dir / "feature_difference_distributions.png"
    residual_summary_csv = run_dir / "feature_difference_summary.csv"
    plot_feature_difference_distributions(
        model=model,
        x_scaled=x_test,
        feature_names=bundle.feature_names,
        scaler=scaler,
        out_png=residual_png,
        out_summary_csv=residual_summary_csv,
        device=trainer.device,
        use_mean_latent=True,
        title_suffix="test split, deterministic(mu)",
    )
    update_training_run_artifacts(
        run_dir,
        {
            "feature_difference_distributions_png": str(residual_png),
            "feature_difference_summary_csv": str(residual_summary_csv),
        },
    )
    print(f"saved_run={run_dir}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dataset-name", type=str, default="mfeat")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-scheduler", type=str, default=None)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--latent-dim", type=int, default=5)
    parser.add_argument("--hidden-dims", type=str, default="16,24,16,8")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--input-dropout", type=float, default=0.0)
    parser.add_argument("--no-normalize-features", action="store_true")
    parser.add_argument("--deterministic-latent", action="store_true")
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--beta-warmup", type=str, default=None)
    parser.add_argument("--beta-start", type=float, default=0.0)
    parser.add_argument("--beta-warmup-epochs", type=int, default=30)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--training-type", type=str, default="baseline")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/output/training_runs"),
    )
    args = parser.parse_args()

    history = run_baseline(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_scheduler=args.lr_scheduler,
        lr_min=args.lr_min,
        latent_dim=args.latent_dim,
        hidden_dims=_parse_hidden_dims(args.hidden_dims),
        test_size=args.test_size,
        split_seed=args.split_seed,
        input_dropout=args.input_dropout,
        normalize_features=not args.no_normalize_features,
        deterministic_latent=args.deterministic_latent,
        beta=args.beta,
        beta_warmup=args.beta_warmup,
        beta_start=args.beta_start,
        beta_warmup_epochs=args.beta_warmup_epochs,
        device=args.device,
        training_type=args.training_type,
        output_dir=args.output_dir,
    )
    print(history[-1])
    # TODO: add Shapley callback wiring and experiment configuration.


if __name__ == "__main__":
    main()
