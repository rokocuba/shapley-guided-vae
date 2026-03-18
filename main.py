from pathlib import Path

import torch

from models import VAE, VAEConfig
from training import DynamicWeightedVAELoss, Trainer
from utils import load_wine_features, make_wine_dataloader


def run_baseline(
    data_dir: str | Path = "data",
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    latent_dim: int = 5,
    input_dropout: float = 0.0,
    beta: float = 1.0,
    device: str | None = None,
) -> list[dict[str, float]]:
    x = load_wine_features(data_dir)
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
    return trainer.fit(train_loader, epochs=epochs)


def main() -> None:
    history = run_baseline()
    print(history[-1])
    # TODO: add Shapley callback wiring and experiment configuration.


if __name__ == "__main__":
    main()
