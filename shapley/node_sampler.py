from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .blocks import FeatureBlockIndex


@dataclass(slots=True)
class NodeStatsSnapshot:
    node_id: int
    mask: str
    coalition_size: int
    c_value: float
    mean: float
    second_moment: float
    variance: float
    effective_count: float
    effective_count_pre: float
    blend_weight: float
    probability: float
    n_new: int
    global_progress: float


class CoalitionNodeIndex:
    def __init__(self, block_index: FeatureBlockIndex) -> None:
        self.block_index = block_index
        self.n_blocks = block_index.n_blocks
        if self.n_blocks < 1:
            raise ValueError("At least one block is required.")
        self.n_nodes = 2**self.n_blocks

        node_ids = torch.arange(self.n_nodes, dtype=torch.long)
        bit_ids = torch.arange(self.n_blocks, dtype=torch.long)
        self.block_masks = ((node_ids[:, None] >> bit_ids[None, :]) & 1).to(torch.bool)
        self.feature_masks = block_index.expand_block_mask(self.block_masks)
        self.coalition_sizes = self.block_masks.sum(dim=1).to(torch.long)
        self.c_values = torch.tensor(
            [self._structural_importance(int(k)) for k in self.coalition_sizes],
            dtype=torch.float32,
        )

    def _structural_importance(self, k: int) -> float:
        d = self.n_blocks
        total = 0.0
        if k < d:
            total += (d - k) * (
                math.factorial(k) * math.factorial(d - k - 1) / math.factorial(d)
            )
        if k > 0:
            total += k * (
                math.factorial(k - 1) * math.factorial(d - k) / math.factorial(d)
            )
        return total

    def to(self, device: torch.device | str) -> "CoalitionNodeIndex":
        self.block_masks = self.block_masks.to(device)
        self.feature_masks = self.feature_masks.to(device)
        self.coalition_sizes = self.coalition_sizes.to(device)
        self.c_values = self.c_values.to(device)
        return self

    def mask_string(self, node_id: int) -> str:
        mask = self.block_masks[node_id].to(device="cpu")
        return "".join("1" if bool(v) else "0" for v in mask)


class RollingNodeStats:
    def __init__(
        self,
        node_index: CoalitionNodeIndex,
        *,
        n_max: float = 256.0,
        eps: float = 1e-12,
    ) -> None:
        if n_max <= 0.0:
            raise ValueError("n_max must be positive.")
        self.node_index = node_index
        self.n_max = float(n_max)
        self.eps = float(eps)
        n = node_index.n_nodes
        self.mean = torch.zeros(n, dtype=torch.float32)
        self.second_moment = torch.zeros(n, dtype=torch.float32)
        self.variance = torch.zeros(n, dtype=torch.float32)
        self.effective_count = torch.zeros(n, dtype=torch.float32)
        self.effective_count_pre = torch.zeros(n, dtype=torch.float32)
        self.blend_weight = torch.zeros(n, dtype=torch.float32)
        self.n_new = torch.zeros(n, dtype=torch.long)
        self.global_progress = 0.0
        self.initialized = torch.zeros(n, dtype=torch.bool)

    def to(self, device: torch.device | str) -> "RollingNodeStats":
        self.mean = self.mean.to(device)
        self.second_moment = self.second_moment.to(device)
        self.variance = self.variance.to(device)
        self.effective_count = self.effective_count.to(device)
        self.effective_count_pre = self.effective_count_pre.to(device)
        self.blend_weight = self.blend_weight.to(device)
        self.n_new = self.n_new.to(device)
        self.initialized = self.initialized.to(device)
        return self

    def update(
        self,
        node_ids: torch.Tensor,
        values: torch.Tensor,
        *,
        global_progress: float,
    ) -> None:
        if node_ids.ndim != 1 or values.ndim != 1:
            raise ValueError("node_ids and values must be 1D tensors.")
        if node_ids.shape[0] != values.shape[0]:
            raise ValueError("node_ids and values must have the same length.")
        if node_ids.numel() == 0:
            return

        device = self.mean.device
        node_ids = node_ids.to(device=device, dtype=torch.long)
        values = values.to(device=device, dtype=self.mean.dtype)
        progress = min(1.0, max(0.0, float(global_progress)))
        self.global_progress = progress
        self.n_new.zero_()

        for node_id in torch.unique(node_ids).tolist():
            node_id_int = int(node_id)
            node_values = values[node_ids == node_id_int]
            n_new = int(node_values.numel())
            if n_new == 0:
                continue

            batch_mean = node_values.mean()
            batch_second = node_values.pow(2).mean()
            previous_n = torch.minimum(
                self.effective_count[node_id_int],
                torch.tensor(self.n_max, dtype=self.mean.dtype, device=device),
            )
            n_pre = previous_n * (1.0 - progress)
            k = n_new / (float(n_new) + float(n_pre.item()) + self.eps)

            if not bool(self.initialized[node_id_int]):
                k = 1.0
                n_pre = torch.zeros((), dtype=self.mean.dtype, device=device)
                self.initialized[node_id_int] = True

            blend = torch.tensor(k, dtype=self.mean.dtype, device=device)
            self.mean[node_id_int] = (
                (1.0 - blend) * self.mean[node_id_int] + blend * batch_mean
            )
            self.second_moment[node_id_int] = (
                (1.0 - blend) * self.second_moment[node_id_int]
                + blend * batch_second
            )
            self.variance[node_id_int] = torch.clamp_min(
                self.second_moment[node_id_int] - self.mean[node_id_int].pow(2),
                0.0,
            )
            self.effective_count_pre[node_id_int] = n_pre
            self.blend_weight[node_id_int] = blend
            self.effective_count[node_id_int] = min(self.n_max, float(n_pre.item()) + n_new)
            self.n_new[node_id_int] = n_new

    def shapley_values(self) -> torch.Tensor:
        d = self.node_index.n_blocks
        values = self.mean
        phis = torch.zeros(d, dtype=values.dtype, device=values.device)
        for i in range(d):
            bit = 1 << i
            for node_id in range(self.node_index.n_nodes):
                if node_id & bit:
                    continue
                k = int(self.node_index.coalition_sizes[node_id].item())
                weight = (
                    math.factorial(k)
                    * math.factorial(d - k - 1)
                    / math.factorial(d)
                )
                phis[i] += weight * (values[node_id | bit] - values[node_id])
        return phis

    def snapshot_rows(self, probabilities: torch.Tensor | None = None) -> list[dict[str, object]]:
        if probabilities is None:
            probabilities = torch.zeros_like(self.mean)
        probabilities = probabilities.to(device=self.mean.device, dtype=self.mean.dtype)
        rows: list[dict[str, object]] = []
        for node_id in range(self.node_index.n_nodes):
            rows.append(
                {
                    "node_id": node_id,
                    "mask": self.node_index.mask_string(node_id),
                    "coalition_size": int(self.node_index.coalition_sizes[node_id].item()),
                    "c_value": float(self.node_index.c_values[node_id].item()),
                    "mean": float(self.mean[node_id].item()),
                    "second_moment": float(self.second_moment[node_id].item()),
                    "variance": float(self.variance[node_id].item()),
                    "effective_count": float(self.effective_count[node_id].item()),
                    "effective_count_pre": float(self.effective_count_pre[node_id].item()),
                    "blend_weight": float(self.blend_weight[node_id].item()),
                    "probability": float(probabilities[node_id].item()),
                    "n_new": int(self.n_new[node_id].item()),
                    "global_progress": float(self.global_progress),
                }
            )
        return rows


class AdaptiveNodeSampler:
    def __init__(
        self,
        node_index: CoalitionNodeIndex,
        *,
        exploration_floor: float = 0.001,
        n_max: float = 256.0,
    ) -> None:
        if exploration_floor < 0.0:
            raise ValueError("exploration_floor must be nonnegative.")
        if exploration_floor * node_index.n_nodes >= 1.0:
            raise ValueError("exploration_floor is too large for the node count.")
        self.node_index = node_index
        self.exploration_floor = float(exploration_floor)
        self.stats = RollingNodeStats(node_index, n_max=n_max)

    def to(self, device: torch.device | str) -> "AdaptiveNodeSampler":
        self.node_index.to(device)
        self.stats.to(device)
        return self

    def bootstrap_node_ids(
        self,
        *,
        shuffle: bool = False,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        target_device = device if device is not None else self.node_index.block_masks.device
        node_ids = torch.arange(self.node_index.n_nodes, dtype=torch.long, device=target_device)
        if shuffle:
            order = torch.randperm(
                self.node_index.n_nodes, generator=generator, device=target_device
            )
            node_ids = node_ids.index_select(0, order)
        return node_ids

    def probabilities(self) -> torch.Tensor:
        variance = self.stats.variance.to(self.node_index.c_values.device)
        scores = torch.sqrt(torch.clamp_min(variance, 0.0)) * self.node_index.c_values
        if not torch.isfinite(scores).all() or float(scores.sum().item()) <= 0.0:
            return torch.full_like(scores, 1.0 / self.node_index.n_nodes)
        raw = scores / scores.sum()
        if self.exploration_floor == 0.0:
            return raw
        return (
            (1.0 - self.exploration_floor * self.node_index.n_nodes) * raw
            + self.exploration_floor
        )

    def sample_node_ids(
        self,
        num_groups: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if num_groups < 0:
            raise ValueError("num_groups must be nonnegative.")
        probs = self.probabilities()
        return torch.multinomial(
            probs,
            num_samples=num_groups,
            replacement=True,
            generator=generator,
        )
