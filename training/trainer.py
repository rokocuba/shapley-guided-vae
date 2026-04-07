from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .callbacks import Callback, CallbackList
from .loss import DynamicWeightedVAELoss


@dataclass(slots=True)
class TrainerState:
    epoch: int = 0
    batch: int = 0
    step: int = 0
    history: list[dict[str, float]] = field(default_factory=list)
    train_started_at: str = ""
    train_ended_at: str = ""
    train_duration_sec: float = 0.0
    epoch_durations_sec: list[float] = field(default_factory=list)
    callback_timing: dict[str, float] = field(default_factory=dict)
    callback_calls: dict[str, int] = field(default_factory=dict)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: DynamicWeightedVAELoss,
        device: str | torch.device = "cpu",
        callbacks: list[Callback] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.callbacks = CallbackList(callbacks or [])
        self.state = TrainerState()
        self.model.to(self.device)
        self.loss_fn.to(self.device)

    def _run_batch(self, x: torch.Tensor) -> dict[str, float]:
        x = x.to(self.device)
        x_hat, mu, logvar = self.model(x)
        out = self.loss_fn(x, x_hat, mu, logvar)
        self.optimizer.zero_grad(set_to_none=True)
        out.total.backward()
        self.optimizer.step()
        return {
            "loss": out.total.item(),
            "recon": out.recon.item(),
            "kl": out.kl.item(),
        }

    @torch.no_grad()
    def _evaluate_loader(self, loader: DataLoader[torch.Tensor]) -> dict[str, float]:
        model_was_training = self.model.training
        self.model.eval()
        running: dict[str, float] = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        n_batches = 0
        for x in loader:
            x = x.to(self.device)
            x_hat, mu, logvar = self.model(x)
            out = self.loss_fn(x, x_hat, mu, logvar)
            running["loss"] += out.total.item()
            running["recon"] += out.recon.item()
            running["kl"] += out.kl.item()
            n_batches += 1
        if model_was_training:
            self.model.train()
        return {f"val_{k}": v / max(1, n_batches) for k, v in running.items()}

    def fit(
        self,
        train_loader: DataLoader[torch.Tensor],
        epochs: int,
        val_loader: DataLoader[torch.Tensor] | None = None,
    ) -> list[dict[str, float]]:
        train_started = perf_counter()
        self.state.train_started_at = datetime.now(timezone.utc).isoformat()
        self.callbacks.call("on_train_begin", self, logs={})
        for epoch in range(epochs):
            epoch_started = perf_counter()
            self.state.epoch = epoch
            self.model.train()
            self.callbacks.call("on_epoch_begin", self, epoch, logs={})
            running: dict[str, float] = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
            n_batches = 0
            for batch_idx, x in enumerate(train_loader):
                self.state.batch = batch_idx
                self.callbacks.call("on_batch_begin", self, epoch, batch_idx, logs={})
                logs = self._run_batch(x)
                self.callbacks.call("on_batch_end", self, epoch, batch_idx, logs=logs)
                for k, v in logs.items():
                    running[k] += v
                self.state.step += 1
                n_batches += 1
            epoch_logs = {k: v / max(1, n_batches) for k, v in running.items()}
            if val_loader is not None:
                epoch_logs.update(self._evaluate_loader(val_loader))
            epoch_logs["beta"] = float(self.loss_fn.beta)
            self.state.history.append(epoch_logs)
            self.state.epoch_durations_sec.append(perf_counter() - epoch_started)
            self.callbacks.call("on_epoch_end", self, epoch, logs=epoch_logs)
        self.callbacks.call("on_train_end", self, logs={"history": self.state.history})
        self.state.train_duration_sec = perf_counter() - train_started
        self.state.train_ended_at = datetime.now(timezone.utc).isoformat()
        return self.state.history
