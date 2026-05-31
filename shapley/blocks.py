from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True, slots=True)
class FeatureBlock:
    name: str
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class FeatureBlockIndex:
    blocks: tuple[FeatureBlock, ...]
    input_dim: int

    @classmethod
    def from_feature_groups(cls, feature_groups: Sequence[str]) -> "FeatureBlockIndex":
        if len(feature_groups) == 0:
            raise ValueError("feature_groups must not be empty.")

        blocks: list[FeatureBlock] = []
        start = 0
        current = feature_groups[0]
        for idx, group_name in enumerate(feature_groups[1:], start=1):
            if group_name == current:
                continue
            blocks.append(FeatureBlock(name=str(current), start=start, stop=idx))
            current = group_name
            start = idx
        blocks.append(
            FeatureBlock(name=str(current), start=start, stop=len(feature_groups))
        )
        return cls(blocks=tuple(blocks), input_dim=len(feature_groups))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(block.name for block in self.blocks)

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(block.size for block in self.blocks)

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    def group_ids(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        ids = torch.empty(self.input_dim, dtype=torch.long, device=device)
        for block_idx, block in enumerate(self.blocks):
            ids[block.start : block.stop] = block_idx
        return ids

    def expand_block_mask(
        self,
        block_mask: torch.Tensor,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if block_mask.shape[-1] != self.n_blocks:
            raise ValueError(
                f"Expected last dimension {self.n_blocks}, got {block_mask.shape[-1]}."
            )
        target_device = device if device is not None else block_mask.device
        block_mask = block_mask.to(device=target_device, dtype=torch.bool)
        group_ids = self.group_ids(device=target_device)
        return block_mask.index_select(dim=-1, index=group_ids)

    def expand_block_weights(
        self,
        block_weights: torch.Tensor,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if block_weights.shape[-1] != self.n_blocks:
            raise ValueError(
                f"Expected last dimension {self.n_blocks}, got {block_weights.shape[-1]}."
            )
        target_device = device if device is not None else block_weights.device
        block_weights = block_weights.to(device=target_device, dtype=torch.float32)
        block_sizes = torch.tensor(self.sizes, dtype=block_weights.dtype, device=target_device)
        per_feature_block_weights = block_weights / block_sizes.clamp_min(1.0)
        group_ids = self.group_ids(device=target_device)
        return per_feature_block_weights.index_select(dim=-1, index=group_ids)


def mfeat_block_index() -> FeatureBlockIndex:
    return FeatureBlockIndex(
        blocks=(
            FeatureBlock("fou", 0, 76),
            FeatureBlock("fac", 76, 292),
            FeatureBlock("kar", 292, 356),
            FeatureBlock("pix", 356, 596),
            FeatureBlock("zer", 596, 643),
            FeatureBlock("mor", 643, 649),
        ),
        input_dim=649,
    )
