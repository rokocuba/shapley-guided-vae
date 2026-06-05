from dataclasses import dataclass

import torch
from torch import nn

from shapley.blocks import FeatureBlockIndex, mfeat_block_index


@dataclass(slots=True)
class LossOutput:
    total: torch.Tensor
    recon: torch.Tensor
    recon_base: torch.Tensor
    pix_recon: torch.Tensor
    aux_recon_base: torch.Tensor
    kl: torch.Tensor
    feature_mse: torch.Tensor
    block_losses: torch.Tensor


class DynamicWeightedVAELoss(nn.Module):
    def __init__(
        self,
        block_index: FeatureBlockIndex | None = None,
        beta: float = 1.0,
        aux_loss_weight: float = 0.2,
        primary_block: str = "pix",
    ) -> None:
        super().__init__()
        self.block_index = block_index or mfeat_block_index()
        self.beta = beta
        self.aux_loss_weight = float(aux_loss_weight)
        self.primary_block = primary_block
        if primary_block not in self.block_index.names:
            raise ValueError(f"Unknown primary block: {primary_block}.")

        self.primary_block_idx = self.block_index.names.index(primary_block)
        aux_indices = [
            idx
            for idx, block in enumerate(self.block_index.blocks)
            if block.name != primary_block
        ]
        self.aux_block_names = tuple(self.block_index.blocks[idx].name for idx in aux_indices)
        self.register_buffer(
            "aux_block_indices",
            torch.tensor(aux_indices, dtype=torch.long),
        )
        self.register_buffer(
            "aux_weights",
            torch.full((len(aux_indices),), 1.0 / len(aux_indices), dtype=torch.float32),
        )

    @torch.no_grad()
    def set_aux_weights(self, weights: torch.Tensor) -> None:
        if weights.shape[-1] != self.aux_block_indices.numel():
            raise ValueError(
                f"Expected {self.aux_block_indices.numel()} auxiliary weights, "
                f"got {weights.shape[-1]}."
            )
        weights = weights.detach().to(
            device=self.aux_weights.device,
            dtype=self.aux_weights.dtype,
        )
        if not torch.isfinite(weights).all():
            raise ValueError("auxiliary weights must be finite.")
        if (weights < 0.0).any():
            raise ValueError("auxiliary weights must be non-negative.")
        total = weights.sum()
        if float(total.item()) <= 0.0:
            raise ValueError("auxiliary weights must have positive sum.")
        self.aux_weights.copy_(weights / total)

    @torch.no_grad()
    def reset_aux_weights(self) -> None:
        self.aux_weights.fill_(1.0 / self.aux_weights.numel())

    def _block_losses(self, feature_mse: torch.Tensor) -> torch.Tensor:
        losses = [
            feature_mse[block.start : block.stop].mean()
            for block in self.block_index.blocks
        ]
        return torch.stack(losses)

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> LossOutput:
        feature_mse = (x_hat - x).pow(2).mean(dim=0)
        block_losses = self._block_losses(feature_mse)
        pix_recon = block_losses[self.primary_block_idx]
        aux_losses = block_losses.index_select(0, self.aux_block_indices)
        aux_weights = self.aux_weights.to(
            device=block_losses.device,
            dtype=block_losses.dtype,
        )

        aux_recon_base = aux_losses.mean()
        aux_recon_weighted = (aux_weights * aux_losses).sum()
        recon = pix_recon + self.aux_loss_weight * aux_recon_weighted
        recon_base = pix_recon + self.aux_loss_weight * aux_recon_base
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
        total = recon + self.beta * kl
        return LossOutput(
            total=total,
            recon=recon,
            recon_base=recon_base,
            pix_recon=pix_recon,
            aux_recon_base=aux_recon_base,
            kl=kl,
            feature_mse=feature_mse,
            block_losses=block_losses,
        )
