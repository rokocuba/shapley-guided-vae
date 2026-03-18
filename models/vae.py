from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class VAEConfig:
    input_dim: int = 11
    hidden_dims: tuple[int, ...] = (16, 8)
    latent_dim: int = 5
    input_dropout: float = 0.0


class Encoder(nn.Module):
    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        dims = (config.input_dim, *config.hidden_dims)
        layers: list[nn.Module] = (
            [nn.Dropout(p=config.input_dropout)] if config.input_dropout > 0 else []
        )
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend((nn.Linear(in_dim, out_dim), nn.ReLU()))
        self.body = nn.Sequential(*layers)
        self.mu = nn.Linear(dims[-1], config.latent_dim)
        self.logvar = nn.Linear(dims[-1], config.latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        dims = (config.latent_dim, *reversed(config.hidden_dims), config.input_dim)
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend((nn.Linear(in_dim, out_dim), nn.ReLU()))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.body = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.body(z)


class VAE(nn.Module):
    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
