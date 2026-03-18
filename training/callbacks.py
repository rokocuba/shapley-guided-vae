from __future__ import annotations

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


class ShapleyCallback(Callback):
    # TODO: implement online Shapley estimation + dynamic weight updates.
    pass
