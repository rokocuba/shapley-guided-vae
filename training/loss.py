from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class LossOutput:
    total: torch.Tensor
    recon: torch.Tensor
    kl: torch.Tensor
    feature_mse: torch.Tensor


class DynamicWeightedVAELoss(nn.Module):
    def __init__(
        self,
        input_dim: int,
        beta: float = 1.0,
        init_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        w = torch.ones(input_dim) if init_weights is None else init_weights
        self.register_buffer("feature_weights", w / w.sum())
        self.beta = beta

    @torch.no_grad()
    def set_feature_weights(self, w: torch.Tensor, normalize: bool = True) -> None:
        w = w.detach().to(self.feature_weights.device, self.feature_weights.dtype)
        if normalize:
            w = w / (w.sum() + 1e-12)
        self.feature_weights.copy_(w)

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> LossOutput:
        feature_mse = (x_hat - x).pow(2).mean(dim=0)
        recon = (feature_mse * self.feature_weights).sum()
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
        total = recon + self.beta * kl
        return LossOutput(total=total, recon=recon, kl=kl, feature_mse=feature_mse)
