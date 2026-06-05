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

    def excluding(self, *names: str) -> "FeatureBlockIndex":
        excluded = set(names)
        return FeatureBlockIndex(
            blocks=tuple(block for block in self.blocks if block.name not in excluded),
            input_dim=self.input_dim,
        )

    def group_ids(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        ids = torch.full((self.input_dim,), -1, dtype=torch.long, device=device)
        for block_idx, block in enumerate(self.blocks):
            ids[block.start : block.stop] = block_idx
        if bool((ids < 0).any().item()):
            raise ValueError("FeatureBlockIndex does not cover every input feature.")
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
        feature_mask = torch.zeros(
            (*block_mask.shape[:-1], self.input_dim),
            dtype=torch.bool,
            device=target_device,
        )
        for block_idx, block in enumerate(self.blocks):
            feature_mask[..., block.start : block.stop] = block_mask[..., block_idx].unsqueeze(
                -1
            )
        return feature_mask

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
