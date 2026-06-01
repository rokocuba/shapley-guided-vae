from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .baselines import BaselineProvider
from .blocks import FeatureBlockIndex
from .game import ReconstructionGame
from .masking import apply_feature_mask
from .node_sampler import AdaptiveNodeSampler, CoalitionNodeIndex


class MonteCarloShapleyEstimator:
    def __init__(self, game: ReconstructionGame, n_permutations: int = 64) -> None:
        self.game = game
        self.n_permutations = n_permutations

    def estimate(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement permutation sampling for online Shapley approximation.
        raise NotImplementedError


@dataclass(slots=True)
class SamplingEpochResult:
    n_rows: int
    n_groups: int
    bootstrap: bool
    global_progress: float
    shapley_values: torch.Tensor


class BlockShapleyEstimator:
    def __init__(
        self,
        *,
        model: nn.Module,
        block_index: FeatureBlockIndex,
        baseline_provider: BaselineProvider,
        sampler: AdaptiveNodeSampler | None = None,
        group_size: int = 16,
        eps: float = 1e-12,
    ) -> None:
        if group_size < 1:
            raise ValueError("group_size must be >= 1.")
        self.model = model
        self.block_index = block_index
        self.baseline_provider = baseline_provider
        self.node_index = CoalitionNodeIndex(block_index)
        self.sampler = sampler or AdaptiveNodeSampler(self.node_index)
        self.group_size = int(group_size)
        self.eps = float(eps)
        self.previous_reference_recon: float | None = None
        self.value_feature_weights = block_index.equal_block_feature_weights()

    def to(self, device: torch.device | str) -> "BlockShapleyEstimator":
        self.sampler.to(device)
        self.value_feature_weights = self.value_feature_weights.to(device)
        return self

    @torch.no_grad()
    def deterministic_reconstruction(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.model.encoder(x)
        return self.model.decoder(mu)

    def row_reconstruction_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
    ) -> torch.Tensor:
        weights = self.value_feature_weights.to(device=x.device, dtype=x.dtype)
        return ((x_hat - x).pow(2) * weights).sum(dim=1)

    @torch.no_grad()
    def reference_reconstruction_loss(
        self,
        x: torch.Tensor,
        *,
        batch_size: int = 512,
    ) -> float:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        model_was_training = self.model.training
        self.model.eval()
        total = 0.0
        n_rows = 0
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size]
            x_hat = self.deterministic_reconstruction(xb)
            losses = self.row_reconstruction_loss(xb, x_hat)
            total += float(losses.sum().item())
            n_rows += int(losses.numel())
        if model_was_training:
            self.model.train()
        return total / max(1, n_rows)

    def compute_global_progress(self, current_reference_recon: float) -> float:
        if self.previous_reference_recon is None:
            self.previous_reference_recon = float(current_reference_recon)
            return 0.0
        previous = float(self.previous_reference_recon)
        progress = (previous - float(current_reference_recon)) / (
            abs(previous) + self.eps
        )
        self.previous_reference_recon = float(current_reference_recon)
        return min(1.0, max(0.0, progress))

    @torch.no_grad()
    def run_sampling_epoch(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        batch_size: int = 512,
        bootstrap: bool = False,
        generator: torch.Generator | None = None,
        reference_x: torch.Tensor | None = None,
    ) -> SamplingEpochResult:
        if x.ndim != 2:
            raise ValueError("x must be a 2D tensor.")
        if x.shape[1] != self.block_index.input_dim:
            raise ValueError(
                f"Expected input_dim {self.block_index.input_dim}, got {x.shape[1]}."
            )
        if labels is not None and labels.shape[0] != x.shape[0]:
            raise ValueError("labels must match x row count.")
        if batch_size < self.group_size:
            raise ValueError("batch_size must be >= group_size.")
        if batch_size % self.group_size != 0:
            raise ValueError("batch_size must be divisible by group_size.")

        model_was_training = self.model.training
        self.model.eval()
        device = next(self.model.parameters()).device
        self.to(device)
        x = x.to(device)
        labels_device = None if labels is None else labels.to(device)
        reference_data = x if reference_x is None else reference_x.to(device)
        current_reference = self.reference_reconstruction_loss(
            reference_data, batch_size=batch_size
        )
        global_progress = self.compute_global_progress(current_reference)

        permutation = torch.randperm(x.shape[0], generator=generator, device=device)
        if bootstrap:
            n_groups = self.node_index.n_nodes
            required_rows = n_groups * self.group_size
            if x.shape[0] < required_rows:
                raise ValueError(
                    f"Bootstrap requires at least {required_rows} rows, got {x.shape[0]}."
                )
            row_indices = permutation[:required_rows]
            group_node_ids = self.sampler.bootstrap_node_ids(device=device)
        else:
            n_groups = x.shape[0] // self.group_size
            row_indices = permutation[: n_groups * self.group_size]
            group_node_ids = self.sampler.sample_node_ids(n_groups)

        rows_per_batch = (batch_size // self.group_size) * self.group_size
        all_node_ids: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []

        for start in range(0, row_indices.numel(), rows_per_batch):
            batch_rows = row_indices[start : start + rows_per_batch]
            batch_group_ids = group_node_ids[
                start // self.group_size : (start + batch_rows.numel()) // self.group_size
            ]
            x_original = x.index_select(0, batch_rows)
            labels_batch = (
                None
                if labels_device is None
                else labels_device.index_select(0, batch_rows)
            )
            row_node_ids = batch_group_ids.repeat_interleave(self.group_size)
            feature_masks = self.node_index.feature_masks.index_select(0, row_node_ids)
            baseline = self.baseline_provider.sample(
                x_original, labels=labels_batch, generator=generator
            )
            x_masked = apply_feature_mask(x_original, feature_masks, baseline)

            x_hat_masked = self.deterministic_reconstruction(x_masked)
            x_hat_full = self.deterministic_reconstruction(x_original)
            loss_masked = self.row_reconstruction_loss(x_original, x_hat_masked)
            loss_full = self.row_reconstruction_loss(x_original, x_hat_full)
            values = (loss_full - loss_masked) / (loss_full.abs() + self.eps)
            all_node_ids.append(row_node_ids.detach())
            all_values.append(values.detach())

        node_ids = torch.cat(all_node_ids)
        values = torch.cat(all_values)
        self.sampler.stats.update(
            node_ids,
            values,
            global_progress=global_progress,
        )
        if model_was_training:
            self.model.train()
        return SamplingEpochResult(
            n_rows=int(values.numel()),
            n_groups=int(group_node_ids.numel()),
            bootstrap=bool(bootstrap),
            global_progress=float(global_progress),
            shapley_values=self.shapley_values(),
        )

    def shapley_values(self) -> torch.Tensor:
        return self.sampler.stats.shapley_values()
