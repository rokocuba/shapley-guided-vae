from dataclasses import dataclass

import torch
from torch import nn

from shapley.blocks import FeatureBlockIndex


@dataclass(slots=True)
class LossOutput:
    total: torch.Tensor
    recon: torch.Tensor
    recon_base: torch.Tensor
    recon_unweighted: torch.Tensor
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
        w = w / w.sum()
        self.register_buffer("feature_weights", w.clone())
        self.register_buffer("base_feature_weights", w.clone())
        self.beta = beta

    @torch.no_grad()
    def set_feature_weights(self, w: torch.Tensor, normalize: bool = True) -> None:
        w = w.detach().to(self.feature_weights.device, self.feature_weights.dtype)
        if normalize:
            w = w / (w.sum() + 1e-12)
        self.feature_weights.copy_(w)

    @torch.no_grad()
    def set_base_feature_weights(self, w: torch.Tensor | None = None) -> None:
        if w is not None:
            w = w.detach().to(
                self.base_feature_weights.device,
                self.base_feature_weights.dtype,
            )
            w = w / (w.sum() + 1e-12)
            self.base_feature_weights.copy_(w)
        self.feature_weights.copy_(self.base_feature_weights)

    @torch.no_grad()
    def set_uniform_feature_weights(self) -> None:
        w = torch.full_like(self.feature_weights, 1.0 / self.feature_weights.numel())
        self.set_base_feature_weights(w)

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
        recon_base = (feature_mse * self.base_feature_weights).sum()
        recon_unweighted = feature_mse.mean()
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
        total = recon + self.beta * kl
        return LossOutput(
            total=total,
            recon=recon,
            recon_base=recon_base,
            recon_unweighted=recon_unweighted,
            kl=kl,
            feature_mse=feature_mse,
        )
