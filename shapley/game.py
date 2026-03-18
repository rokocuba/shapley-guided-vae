from __future__ import annotations

import torch
from torch import nn

from .masking import apply_feature_mask


class ReconstructionGame:
    def __init__(
        self,
        model: nn.Module,
        baseline: torch.Tensor,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.baseline = baseline
        self.device = torch.device(device)

    @torch.no_grad()
    def payoff(self, x: torch.Tensor, coalition_mask: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x_masked = apply_feature_mask(
            x, coalition_mask.to(self.device), self.baseline.to(self.device)
        )
        x_hat, _, _ = self.model(x_masked)
        mse = (x_hat - x).pow(2).mean(dim=1)
        return -mse.mean()
