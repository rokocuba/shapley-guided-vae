from __future__ import annotations

from dataclasses import dataclass

import torch

from .blocks import FeatureBlockIndex


@dataclass(slots=True)
class ShapleyWeightResult:
    raw_scores: torch.Tensor
    weights: torch.Tensor
    policy: str


def auxiliary_shapley_weights(
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
    if policy not in {"error", "flip_negative", "uniform"}:
        raise ValueError(
            "on_all_nonpositive must be one of: error, flip_negative, uniform."
        )

    raw = shapley_values.detach().to(dtype=torch.float32).clone()

    all_nonpositive = bool((raw <= 0.0).all().item())
    if all_nonpositive:
        if policy == "error":
            raise ValueError(
                "All Shapley scores are non-positive; refusing to silently map them "
                "to auxiliary weights."
            )
        if policy == "uniform":
            weights = torch.full_like(raw, 1.0 / raw.numel())
            return ShapleyWeightResult(
                raw_scores=raw,
                weights=weights,
                policy="uniform_all_nonpositive",
            )

    if policy == "flip_negative" and float(torch.clamp_min(raw, 0.0).sum()) <= eps:
        signal = torch.clamp_min(-raw, 0.0)
        mapping_policy = "flip_negative"
    else:
        signal = torch.clamp_min(raw, 0.0)
        mapping_policy = "positive_share"

    total = signal.sum()
    if float(total.item()) <= eps:
        if policy == "uniform":
            weights = torch.full_like(raw, 1.0 / raw.numel())
            return ShapleyWeightResult(
                raw_scores=raw,
                weights=weights,
                policy="uniform_no_positive_phi",
            )
        raise ValueError(
            "No positive Shapley scores are available for auxiliary weight sharing."
        )

    weights = signal / total.clamp_min(eps)

    return ShapleyWeightResult(
        raw_scores=raw,
        weights=weights,
        policy=mapping_policy,
    )
