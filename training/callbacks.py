from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any


class Callback:
    def on_train_begin(self, trainer: Any, logs: dict[str, Any] | None = None) -> None:
        pass

    def on_epoch_begin(
        self, trainer: Any, epoch: int, logs: dict[str, Any] | None = None
    ) -> None:
        pass

    def on_batch_begin(
        self,
        trainer: Any,
        epoch: int,
        batch_idx: int,
        logs: dict[str, Any] | None = None,
    ) -> None:
        pass

    def on_batch_end(
        self,
        trainer: Any,
        epoch: int,
        batch_idx: int,
        logs: dict[str, Any] | None = None,
    ) -> None:
        pass

    def on_epoch_end(
        self, trainer: Any, epoch: int, logs: dict[str, Any] | None = None
    ) -> None:
        pass

    def on_train_end(self, trainer: Any, logs: dict[str, Any] | None = None) -> None:
        pass


@dataclass(slots=True)
class CallbackList:
    items: Sequence[Callback] = ()

    def call(self, hook: str, *args: Any, **kwargs: Any) -> None:
        trainer = args[0] if args else None
        for cb in self.items:
            started = perf_counter()
            getattr(cb, hook)(*args, **kwargs)
            elapsed = perf_counter() - started
            if trainer is None or not hasattr(trainer, "state"):
                continue
            key = f"{cb.__class__.__name__}.{hook}"
            state = trainer.state
            timing = getattr(state, "callback_timing", None)
            calls = getattr(state, "callback_calls", None)
            if timing is not None:
                timing[key] = timing.get(key, 0.0) + elapsed
            if calls is not None:
                calls[key] = calls.get(key, 0) + 1


class BetaWarmupCallback(Callback):
    def __init__(
        self,
        target_beta: float,
        warmup_epochs: int,
        strategy: str = "linear",
        start_beta: float = 0.0,
    ) -> None:
        strategy = strategy.strip().lower()
        allowed = {"linear", "cosine", "sigmoid"}
        if strategy not in allowed:
            raise ValueError(
                f"Unknown beta warmup strategy '{strategy}'. Allowed: {sorted(allowed)}"
            )
        if warmup_epochs < 1:
            raise ValueError("warmup_epochs must be >= 1.")
        self.target_beta = float(target_beta)
        self.start_beta = float(start_beta)
        self.warmup_epochs = int(warmup_epochs)
        self.strategy = strategy

    def _shape(self, progress: float) -> float:
        if self.strategy == "linear":
            return progress
        if self.strategy == "cosine":
            return 0.5 - 0.5 * math.cos(math.pi * progress)
        low = 1.0 / (1.0 + math.exp(6.0))
        high = 1.0 / (1.0 + math.exp(-6.0))
        raw = 1.0 / (1.0 + math.exp(-12.0 * (progress - 0.5)))
        return (raw - low) / (high - low)

    def beta_for_epoch(self, epoch: int) -> float:
        if self.warmup_epochs == 1:
            return self.target_beta
        if epoch >= self.warmup_epochs:
            return self.target_beta
        progress = max(0.0, min(1.0, epoch / float(self.warmup_epochs - 1)))
        shaped = self._shape(progress)
        return self.start_beta + (self.target_beta - self.start_beta) * shaped

    def on_train_begin(self, trainer: Any, logs: dict[str, Any] | None = None) -> None:
        trainer.loss_fn.beta = float(self.beta_for_epoch(0))

    def on_epoch_begin(
        self, trainer: Any, epoch: int, logs: dict[str, Any] | None = None
    ) -> None:
        trainer.loss_fn.beta = float(self.beta_for_epoch(epoch))


class ShapleyCallback(Callback):
    # TODO: implement online Shapley estimation + dynamic weight updates.
    pass
