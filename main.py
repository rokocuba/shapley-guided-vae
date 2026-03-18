from pathlib import Path

import torch

from models import VAE, VAEConfig
from training import DynamicWeightedVAELoss, Trainer, save_training_run
from utils import load_wine_bundle, make_wine_dataloader


def run_baseline(
    data_dir: str | Path = "data",
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    latent_dim: int = 5,
    input_dropout: float = 0.0,
    beta: float = 1.0,
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
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        callbacks=[],
    )
    train_loader = make_wine_dataloader(data_dir, batch_size=batch_size)
    history = trainer.fit(train_loader, epochs=epochs)
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
            "hidden_dims": list(model.config.hidden_dims),
            "input_dropout": input_dropout,
            "beta": beta,
            "device": str(trainer.device),
        },
        data_info={
            "data_dir": str(data_dir),
            "n_rows": int(x.shape[0]),
            "n_features": int(x.shape[1]),
            "feature_names": bundle.feature_names,
            "scaler_mean": [float(v) for v in scaler_mean],
            "scaler_scale": [float(v) for v in scaler_scale],
        },
        output_dir=output_dir,
        training_type=training_type,
        shapley_tactic=shapley_tactic,
    )
    print(f"saved_run={run_dir}")
    return history


def main() -> None:
    history = run_baseline()
    print(history[-1])
    # TODO: add Shapley callback wiring and experiment configuration.


if __name__ == "__main__":
    main()
