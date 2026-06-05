from __future__ import annotations

import torch
from torch import nn

from .blocks import FeatureBlockIndex, mfeat_block_index
from .masking import apply_feature_mask


class ReconstructionGame:
    def __init__(
        self,
        model: nn.Module,
        baseline: torch.Tensor,
        block_index: FeatureBlockIndex | None = None,
        value_block_index: FeatureBlockIndex | None = None,
        primary_block: str = "pix",
        device: str | torch.device = "cpu",
        eps: float = 1e-12,
    ) -> None:
        self.model = model
        self.baseline = baseline
        self.block_index = block_index or mfeat_block_index()
        self.value_block_index = value_block_index or self.block_index
        if self.block_index.input_dim != self.value_block_index.input_dim:
            raise ValueError("player and value block indices must share input_dim.")
        if primary_block not in self.value_block_index.names:
            raise ValueError(f"Unknown primary block: {primary_block}.")
        self.primary_block_idx = self.value_block_index.names.index(primary_block)
        self.device = torch.device(device)
        self.eps = float(eps)

    @torch.no_grad()
    def _deterministic_reconstruction(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.model.encoder(x)
        return self.model.decoder(mu)

    def _pixel_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        block = self.value_block_index.blocks[self.primary_block_idx]
        return (x_hat[:, block.start : block.stop] - x[:, block.start : block.stop]).pow(
            2
        ).mean(dim=1)

    @torch.no_grad()
    def payoff(self, x: torch.Tensor, coalition_mask: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x_masked = apply_feature_mask(
            x, coalition_mask.to(self.device), self.baseline.to(self.device)
        )
        full_player_mask = self.block_index.expand_block_mask(
            torch.ones(
                (x.shape[0], self.block_index.n_blocks),
                dtype=torch.bool,
                device=self.device,
            ),
            device=self.device,
        )
        x_full_game = apply_feature_mask(
            x,
            full_player_mask,
            self.baseline.to(self.device),
        )
        loss_full = self._pixel_loss(
            x,
            self._deterministic_reconstruction(x_full_game),
        )
        loss_masked = self._pixel_loss(
            x, self._deterministic_reconstruction(x_masked)
        )
        return ((loss_full - loss_masked) / (loss_full.abs() + self.eps)).mean()
