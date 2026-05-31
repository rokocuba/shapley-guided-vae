from __future__ import annotations

from dataclasses import dataclass

import torch

from .blocks import FeatureBlockIndex


@dataclass(slots=True)
class ShapleyWeightResult:
    raw_scores: torch.Tensor
    block_weights: torch.Tensor
    feature_weights: torch.Tensor


def positive_shapley_weights(
    shapley_values: torch.Tensor,
    block_index: FeatureBlockIndex,
    *,
    on_all_nonpositive: str = "error",
    eps: float = 1e-12,
) -> ShapleyWeightResult:
    if shapley_values.ndim != 1:
        raise ValueError("shapley_values must be a 1D tensor.")
    if shapley_values.shape[0] != block_index.n_blocks:
        raise ValueError(
            f"Expected {block_index.n_blocks} Shapley values, got {shapley_values.shape[0]}."
        )
    if not torch.isfinite(shapley_values).all():
        raise ValueError("shapley_values must be finite.")

    policy = on_all_nonpositive.strip().lower()
    positive = torch.clamp_min(shapley_values, 0.0)
    total = positive.sum()
    if float(total.item()) <= eps:
        if policy == "error":
            raise ValueError(
                "All Shapley scores are non-positive; refusing to silently map them "
                "to training weights."
            )
        if policy == "flip_negative":
            positive = torch.clamp_min(-shapley_values, 0.0)
            total = positive.sum()
        elif policy == "uniform":
            positive = torch.ones_like(shapley_values)
            total = positive.sum()
        else:
            raise ValueError(
                "on_all_nonpositive must be one of: error, flip_negative, uniform."
            )
    if float(total.item()) <= eps:
        positive = torch.ones_like(shapley_values)
        total = positive.sum()

    block_weights = positive / total
    feature_weights = block_index.expand_block_weights(block_weights)
    return ShapleyWeightResult(
        raw_scores=shapley_values.detach().clone(),
        block_weights=block_weights,
        feature_weights=feature_weights,
    )
