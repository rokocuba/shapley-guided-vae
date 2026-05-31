from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaselineProvider(ABC):
    @abstractmethod
    def sample(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class GlobalMeanBaselineProvider(BaselineProvider):
    def __init__(self, x_train: torch.Tensor) -> None:
        if x_train.ndim != 2:
            raise ValueError("x_train must be a 2D tensor.")
        self.baseline = x_train.detach().mean(dim=0)

    def sample(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.baseline.to(device=x.device, dtype=x.dtype).expand_as(x)


class MarginalBaselineProvider(BaselineProvider):
    def __init__(self, x_train: torch.Tensor) -> None:
        if x_train.ndim != 2:
            raise ValueError("x_train must be a 2D tensor.")
        if x_train.shape[0] == 0:
            raise ValueError("x_train must contain at least one row.")
        self.x_train = x_train.detach()

    def sample(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        idx = torch.randint(
            low=0,
            high=self.x_train.shape[0],
            size=(x.shape[0],),
            generator=generator,
            device=self.x_train.device,
        )
        return self.x_train.index_select(0, idx).to(device=x.device, dtype=x.dtype)


class ConditionalDigitBaselineProvider(BaselineProvider):
    def __init__(self, x_train: torch.Tensor, labels_train: torch.Tensor) -> None:
        if x_train.ndim != 2:
            raise ValueError("x_train must be a 2D tensor.")
        if labels_train.ndim != 1:
            raise ValueError("labels_train must be a 1D tensor.")
        if x_train.shape[0] != labels_train.shape[0]:
            raise ValueError("x_train and labels_train must have the same row count.")
        if x_train.shape[0] == 0:
            raise ValueError("x_train must contain at least one row.")

        self.x_train = x_train.detach()
        labels_train = labels_train.detach().to(device="cpu")
        self.labels_train = labels_train
        self.indices_by_label = {
            int(label.item()): torch.nonzero(labels_train == label, as_tuple=False)
            .flatten()
            .to(dtype=torch.long)
            for label in torch.unique(labels_train)
        }

    def sample(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if labels is None:
            return MarginalBaselineProvider(self.x_train).sample(
                x, labels=None, generator=generator
            )
        labels_cpu = labels.detach().to(device="cpu")
        selected = torch.empty(x.shape[0], dtype=torch.long)
        all_count = self.x_train.shape[0]
        for row_idx, label in enumerate(labels_cpu):
            candidates = self.indices_by_label.get(int(label.item()))
            if candidates is None or candidates.numel() == 0:
                selected[row_idx] = torch.randint(
                    0, all_count, (1,), generator=generator
                ).item()
                continue
            candidate_idx = torch.randint(
                0, candidates.numel(), (1,), generator=generator
            ).item()
            selected[row_idx] = candidates[candidate_idx]
        return self.x_train.index_select(0, selected.to(self.x_train.device)).to(
            device=x.device, dtype=x.dtype
        )


def make_baseline_provider(
    tactic: str,
    x_train: torch.Tensor,
    labels_train: torch.Tensor | None = None,
) -> BaselineProvider:
    normalized = tactic.strip().lower()
    if normalized in {"baseline", "global", "global_mean"}:
        return GlobalMeanBaselineProvider(x_train)
    if normalized == "marginal":
        return MarginalBaselineProvider(x_train)
    if normalized in {"conditional", "conditional_digit"}:
        if labels_train is None:
            raise ValueError("labels_train is required for conditional baselines.")
        return ConditionalDigitBaselineProvider(x_train, labels_train)
    raise ValueError(
        "Unknown baseline tactic. Supported values: baseline, marginal, conditional."
    )
