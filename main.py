from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from models import VAE, VAEConfig
from training import (
    BetaWarmupCallback,
    DynamicWeightedVAELoss,
    Trainer,
    plot_feature_difference_distributions,
    save_training_run,
    update_training_run_artifacts,
)
from utils import load_wine_bundle


def run_baseline(
    data_dir: str | Path = "data",
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    latent_dim: int = 5,
    test_size: float = 0.2,
    split_seed: int = 42,
    input_dropout: float = 0.0,
    beta: float = 1.0,
    beta_warmup: str | None = None,
    beta_start: float = 0.0,
    beta_warmup_epochs: int = 30,
    device: str | None = None,
    training_type: str = "baseline",
    shapley_tactic: str | None = None,
    output_dir: str | Path = "analysis/output/training_runs",
) -> list[dict[str, float]]:
    bundle = load_wine_bundle(data_dir)
    x = bundle.x_scaled
    model = VAE(
        VAEConfig(
            input_dim=x.shape[1], latent_dim=latent_dim, input_dropout=input_dropout
        )
    )
    loss_fn = DynamicWeightedVAELoss(input_dim=x.shape[1], beta=beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
    )
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in the open interval (0, 1).")
    n_total = x.shape[0]
    n_test = int(n_total * test_size)
    n_test = max(1, min(n_total - 1, n_test))
    n_train = n_total - n_test
    split_generator = torch.Generator().manual_seed(split_seed)
    train_set, test_set = random_split(x, [n_train, n_test], generator=split_generator)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    history = trainer.fit(train_loader, epochs=epochs, val_loader=test_loader)
    scaler_mean = bundle.scaler.mean_
    scaler_scale = bundle.scaler.scale_
    if scaler_mean is None or scaler_scale is None:
        raise RuntimeError("Scaler statistics are missing.")
    run_dir = save_training_run(
        model=model,
        history=history,
        state=trainer.state,
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "latent_dim": latent_dim,
            "test_size": test_size,
            "split_seed": split_seed,
            "hidden_dims": list(model.config.hidden_dims),
            "input_dropout": input_dropout,
            "beta": beta,
            "beta_warmup": beta_warmup,
            "beta_start": beta_start,
            "beta_warmup_epochs": beta_warmup_epochs,
            "device": str(trainer.device),
        },
        data_info={
            "data_dir": str(data_dir),
            "n_rows": int(x.shape[0]),
            "n_rows_train": int(n_train),
            "n_rows_test": int(n_test),
            "n_features": int(x.shape[1]),
            "feature_names": bundle.feature_names,
            "scaler_mean": [float(v) for v in scaler_mean],
            "scaler_scale": [float(v) for v in scaler_scale],
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
        x_scaled=bundle.x_scaled,
        feature_names=bundle.feature_names,
        scaler=bundle.scaler,
        out_png=residual_png,
        out_summary_csv=residual_summary_csv,
        device=trainer.device,
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
    history = run_baseline()
    print(history[-1])
    # TODO: add Shapley callback wiring and experiment configuration.


if __name__ == "__main__":
    main()
