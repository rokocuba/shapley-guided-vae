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


class KLBetaSchedulerCallback(Callback):
    def __init__(
        self,
        target_kl: float,
        step_size: float = 0.01,
        min_beta: float = 0.0,
        max_beta: float = 10.0,
        target_start: float = 0.0,
        target_warmup_epochs: int = 0,
        target_curve: str = "reciprocal",
        warmup_step_limit: float | None = None,
        beta_zero_epochs: int = 0,
    ) -> None:
        allowed = {"linear", "cosine", "sigmoid", "reciprocal"}
        if target_kl <= 0.0:
            raise ValueError("target_kl must be > 0.")
        if step_size <= 0.0:
            raise ValueError("step_size must be > 0.")
        if min_beta < 0.0:
            raise ValueError("min_beta must be >= 0.")
        if max_beta < min_beta:
            raise ValueError("max_beta must be >= min_beta.")
        if target_warmup_epochs < 0:
            raise ValueError("target_warmup_epochs must be >= 0.")
        if beta_zero_epochs < 0:
            raise ValueError("beta_zero_epochs must be >= 0.")
        target_curve = target_curve.strip().lower()
        if target_curve not in allowed:
            raise ValueError(
                f"Unknown target warmup curve '{target_curve}'. Allowed: {sorted(allowed)}"
            )
        if warmup_step_limit is not None and warmup_step_limit <= 0.0:
            raise ValueError("warmup_step_limit must be > 0 when provided.")
        self.target_kl = float(target_kl)
        self.step_size = float(step_size)
        self.min_beta = float(min_beta)
        self.max_beta = float(max_beta)
        self.target_start = float(target_start)
        self.target_warmup_epochs = int(target_warmup_epochs)
        self.target_curve = target_curve
        self.warmup_step_limit = (
            None if warmup_step_limit is None else float(warmup_step_limit)
        )
        self.beta_zero_epochs = int(beta_zero_epochs)

    def _clamp_beta(self, beta: float) -> float:
        return min(self.max_beta, max(self.min_beta, beta))

    def _shape(self, progress: float) -> float:
        if self.target_curve == "linear":
            return progress
        if self.target_curve == "cosine":
            return 0.5 - 0.5 * math.cos(math.pi * progress)
        if self.target_curve == "reciprocal":
            raw = 1.0 / (1.0 + 9.0 * progress)
            raw_end = 0.1
            return 1.0 - ((raw - raw_end) / (1.0 - raw_end))
        low = 1.0 / (1.0 + math.exp(6.0))
        high = 1.0 / (1.0 + math.exp(-6.0))
        raw = 1.0 / (1.0 + math.exp(-12.0 * (progress - 0.5)))
        return (raw - low) / (high - low)

    def target_for_epoch(self, epoch: int) -> float:
        if epoch < self.beta_zero_epochs:
            return self.target_start
        if self.target_warmup_epochs <= self.beta_zero_epochs:
            return self.target_kl
        if epoch >= self.target_warmup_epochs:
            return self.target_kl
        effective_warmup_epochs = self.target_warmup_epochs - self.beta_zero_epochs
        if effective_warmup_epochs <= 1:
            return self.target_kl
        effective_epoch = epoch - self.beta_zero_epochs
        progress = max(
            0.0,
            min(1.0, effective_epoch / float(effective_warmup_epochs - 1)),
        )
        shaped = self._shape(progress)
        return self.target_start + (self.target_kl - self.target_start) * shaped

    def _select_kl(self, logs: dict[str, Any] | None) -> float:
        if logs is None:
            raise KeyError("KLBetaSchedulerCallback requires epoch logs.")
        for key in ("val_kl", "kl"):
            value = logs.get(key)
            if value is None:
                continue
            kl_value = float(value)
            if math.isfinite(kl_value):
                return kl_value
        raise KeyError(
            "KLBetaSchedulerCallback requires a finite 'val_kl' or 'kl' value."
        )

    def on_train_begin(self, trainer: Any, logs: dict[str, Any] | None = None) -> None:
        if self.beta_zero_epochs > 0:
            trainer.loss_fn.beta = 0.0
            return
        trainer.loss_fn.beta = float(self._clamp_beta(float(trainer.loss_fn.beta)))

    def on_epoch_begin(
        self, trainer: Any, epoch: int, logs: dict[str, Any] | None = None
    ) -> None:
        if epoch < self.beta_zero_epochs:
            trainer.loss_fn.beta = 0.0

    def on_epoch_end(
        self, trainer: Any, epoch: int, logs: dict[str, Any] | None = None
    ) -> None:
        observed_kl = self._select_kl(logs)
        target_kl = self.target_for_epoch(epoch)
        if epoch < self.beta_zero_epochs:
            trainer.loss_fn.beta = 0.0
            if logs is not None:
                logs["kl_target"] = float(target_kl)
                logs["kl_target_final"] = float(self.target_kl)
                logs["next_beta"] = 0.0
            return
        current_beta = float(trainer.loss_fn.beta)
        beta_delta = self.step_size * (observed_kl - target_kl)
        if (
            self.warmup_step_limit is not None
            and self.target_warmup_epochs > 1
            and epoch < self.target_warmup_epochs
        ):
            beta_delta = max(
                -self.warmup_step_limit,
                min(self.warmup_step_limit, beta_delta),
            )
        next_beta = self._clamp_beta(current_beta + beta_delta)
        trainer.loss_fn.beta = float(next_beta)
        if logs is not None:
            logs["kl_target"] = float(target_kl)
            logs["kl_target_final"] = float(self.target_kl)
            logs["next_beta"] = float(next_beta)


class ShapleyCallback(Callback):
    # TODO: implement online Shapley estimation + dynamic weight updates.
    pass
