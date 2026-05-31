from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader

from shapley import BlockShapleyEstimator, FeatureBlockIndex, positive_shapley_weights

from .trainer import Trainer


@dataclass(slots=True)
class ShapleyTrainingConfig:
    total_epochs: int
    warmup_epochs: int = 100
    min_sampling_phases_before_c: int = 5
    phase_c_epochs: int = 1
    early_interval: int = 10
    middle_interval: int = 20
    late_interval: int = 40
    middle_after_fraction: float = 0.25
    late_after_fraction: float = 0.60
    sampling_batch_size: int = 512
    all_nonpositive_policy: str = "error"


@dataclass(slots=True)
class ShapleyTrainingResult:
    history: list[dict[str, Any]]
    shapley_weight_rows: list[dict[str, Any]] = field(default_factory=list)
    node_stat_rows: list[dict[str, Any]] = field(default_factory=list)
    phase_timing_rows: list[dict[str, Any]] = field(default_factory=list)


@contextmanager
def freeze_module(module: nn.Module) -> Iterator[None]:
    params = list(module.parameters())
    previous = [p.requires_grad for p in params]
    try:
        for param in params:
            param.requires_grad = False
        yield
    finally:
        for param, requires_grad in zip(params, previous):
            param.requires_grad = requires_grad


def _sampling_interval(epoch: int, config: ShapleyTrainingConfig) -> int:
    if epoch >= int(config.total_epochs * config.late_after_fraction):
        return config.late_interval
    if epoch >= int(config.total_epochs * config.middle_after_fraction):
        return config.middle_interval
    return config.early_interval


def _should_sample(epoch: int, config: ShapleyTrainingConfig) -> bool:
    completed_epochs = epoch + 1
    if completed_epochs < config.warmup_epochs:
        return False
    interval = max(1, _sampling_interval(epoch, config))
    return (completed_epochs - config.warmup_epochs) % interval == 0


def _train_one_epoch(
    trainer: Trainer,
    loader: DataLoader[torch.Tensor],
    *,
    epoch: int,
    phase: str,
    elapsed_train_sec: float,
    val_loader: DataLoader[torch.Tensor] | None = None,
) -> tuple[dict[str, Any], float]:
    epoch_started = perf_counter()
    trainer.state.epoch = epoch
    trainer.model.train()
    trainer.callbacks.call("on_epoch_begin", trainer, epoch, logs={"phase": phase})
    running: dict[str, float] = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    n_batches = 0
    for batch_idx, x in enumerate(loader):
        trainer.state.batch = batch_idx
        trainer.callbacks.call(
            "on_batch_begin", trainer, epoch, batch_idx, logs={"phase": phase}
        )
        logs = trainer._run_batch(x)
        trainer.callbacks.call(
            "on_batch_end", trainer, epoch, batch_idx, logs={**logs, "phase": phase}
        )
        for key, value in logs.items():
            running[key] += value
        trainer.state.step += 1
        n_batches += 1

    epoch_logs: dict[str, Any] = {
        key: value / max(1, n_batches) for key, value in running.items()
    }
    if val_loader is not None:
        epoch_logs.update(trainer._evaluate_loader(val_loader))
    epoch_duration_sec = perf_counter() - epoch_started
    elapsed_train_sec += epoch_duration_sec
    epoch_logs.update(
        {
            "phase": phase,
            "beta": float(trainer.loss_fn.beta),
            "lr": float(trainer.optimizer.param_groups[0]["lr"]),
            "epoch_duration_sec": float(epoch_duration_sec),
            "elapsed_train_sec": float(elapsed_train_sec),
        }
    )
    trainer.state.history.append(epoch_logs)
    trainer.state.epoch_durations_sec.append(epoch_duration_sec)
    trainer.callbacks.call("on_epoch_end", trainer, epoch, logs=epoch_logs)
    if trainer.scheduler is not None:
        if trainer.scheduler.__class__.__name__ == "ReduceLROnPlateau":
            monitor_key = trainer.scheduler_monitor
            if monitor_key not in epoch_logs:
                raise KeyError(
                    f"Scheduler monitor '{monitor_key}' not found in epoch logs."
                )
            trainer.scheduler.step(float(epoch_logs[monitor_key]))
        else:
            trainer.scheduler.step()
    return epoch_logs, elapsed_train_sec


def run_shapley_training(
    *,
    trainer: Trainer,
    train_loader: DataLoader[torch.Tensor],
    train_tensor: torch.Tensor,
    estimator: BlockShapleyEstimator,
    block_index: FeatureBlockIndex,
    config: ShapleyTrainingConfig,
    val_loader: DataLoader[torch.Tensor] | None = None,
    train_labels: torch.Tensor | None = None,
) -> ShapleyTrainingResult:
    train_started = perf_counter()
    trainer.state.train_started_at = datetime.now(timezone.utc).isoformat()
    result = ShapleyTrainingResult(history=trainer.state.history)
    elapsed_train_sec = (
        float(trainer.state.history[-1].get("elapsed_train_sec", 0.0))
        if trainer.state.history
        else 0.0
    )
    sampling_phase = 0
    cycle = 0
    applied_block_weights: torch.Tensor | None = None
    trainer.callbacks.call("on_train_begin", trainer, logs={"training_type": "shapley"})
    for epoch in range(config.total_epochs):
        trainer.loss_fn.set_uniform_feature_weights()
        logs, elapsed_train_sec = _train_one_epoch(
            trainer,
            train_loader,
            epoch=epoch,
            phase="A",
            elapsed_train_sec=elapsed_train_sec,
            val_loader=val_loader,
        )
        logs["cycle"] = cycle
        logs["sampling_phase"] = sampling_phase

        if not _should_sample(epoch, config):
            continue

        phase_started = perf_counter()
        sampling = estimator.run_sampling_epoch(
            train_tensor,
            labels=train_labels,
            batch_size=config.sampling_batch_size,
            bootstrap=sampling_phase == 0,
        )
        sampling_phase += 1
        probabilities = estimator.sampler.probabilities()
        result.phase_timing_rows.append(
            {
                "epoch_start": epoch,
                "epoch_end": epoch,
                "phase": "B",
                "cycle": cycle,
                "duration_sec": perf_counter() - phase_started,
                "num_batches": None,
                "num_samples": sampling.n_rows,
                "num_node_groups": sampling.n_groups,
                "tactic": None,
            }
        )
        for row in estimator.sampler.stats.snapshot_rows(probabilities):
            row.update({"epoch": epoch, "cycle": cycle, "sampling_phase": sampling_phase})
            result.node_stat_rows.append(row)

        weights = positive_shapley_weights(
            sampling.shapley_values.detach().cpu(),
            block_index,
            on_all_nonpositive=config.all_nonpositive_policy,
        )
        applied_block_weights = weights.block_weights
        for block_idx, block in enumerate(block_index.blocks):
            result.shapley_weight_rows.append(
                {
                    "epoch": epoch,
                    "cycle": cycle,
                    "sampling_phase": sampling_phase,
                    "block": block.name,
                    "block_size": block.size,
                    "phi_raw": float(weights.raw_scores[block_idx].item()),
                    "weight_raw": float(weights.block_weights[block_idx].item()),
                    "weight_applied": float(weights.block_weights[block_idx].item()),
                    "feature_weight_sum": float(
                        weights.feature_weights[block.start : block.stop].sum().item()
                    ),
                }
            )

        if sampling_phase < config.min_sampling_phases_before_c:
            cycle += 1
            continue

        trainer.loss_fn.set_block_weights(applied_block_weights, block_index)
        phase_started = perf_counter()
        with freeze_module(trainer.model.decoder):
            for c_epoch in range(config.phase_c_epochs):
                c_logs, elapsed_train_sec = _train_one_epoch(
                    trainer,
                    train_loader,
                    epoch=epoch,
                    phase="C",
                    elapsed_train_sec=elapsed_train_sec,
                    val_loader=val_loader,
                )
                c_logs["cycle"] = cycle
                c_logs["sampling_phase"] = sampling_phase
                c_logs["phase_c_epoch"] = c_epoch
        trainer.loss_fn.set_uniform_feature_weights()
        result.phase_timing_rows.append(
            {
                "epoch_start": epoch,
                "epoch_end": epoch,
                "phase": "C",
                "cycle": cycle,
                "duration_sec": perf_counter() - phase_started,
                "num_batches": len(train_loader) * config.phase_c_epochs,
                "num_samples": len(train_loader.dataset) * config.phase_c_epochs,
                "num_node_groups": None,
                "tactic": None,
            }
        )
        cycle += 1
    trainer.callbacks.call("on_train_end", trainer, logs={"history": trainer.state.history})
    trainer.state.train_duration_sec = perf_counter() - train_started
    trainer.state.train_ended_at = datetime.now(timezone.utc).isoformat()
    return result
