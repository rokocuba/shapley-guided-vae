from dataclasses import dataclass

import torch
from torch import nn

from shapley.blocks import FeatureBlockIndex


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

    @torch.no_grad()
    def set_uniform_feature_weights(self) -> None:
        self.feature_weights.fill_(1.0 / self.feature_weights.numel())

    @torch.no_grad()
    def set_block_weights(
        self,
        block_weights: torch.Tensor,
        block_index: FeatureBlockIndex,
    ) -> None:
        if block_index.input_dim != self.feature_weights.numel():
            raise ValueError(
                f"Block index input_dim {block_index.input_dim} does not match "
                f"loss input_dim {self.feature_weights.numel()}."
            )
        feature_weights = block_index.expand_block_weights(
            block_weights,
            device=self.feature_weights.device,
        )
        self.set_feature_weights(feature_weights, normalize=True)

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
