from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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
        for cb in self.items:
            getattr(cb, hook)(*args, **kwargs)


class ShapleyCallback(Callback):
    # TODO: implement online Shapley estimation + dynamic weight updates.
    pass
