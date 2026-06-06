from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

import torch
from torch.utils.data import DataLoader

from shapley import (
    BlockShapleyEstimator,
    FeatureBlockIndex,
    auxiliary_shapley_weights,
)

from .trainer import Trainer


@dataclass(slots=True)
class ShapleyTrainingConfig:
    total_epochs: int
    warmup_epochs: int = 100
    min_sampling_phases_before_dynamic: int = 5
    early_interval: int = 10
    middle_interval: int = 20
    late_interval: int = 40
    middle_after_fraction: float = 0.25
    late_after_fraction: float = 0.60
    sampling_batch_size: int = 512
    all_nonpositive_policy: str = "error"
    tactic: str = "baseline"


@dataclass(slots=True)
class ShapleyTrainingResult:
    history: list[dict[str, Any]]
    shapley_weight_rows: list[dict[str, Any]] = field(default_factory=list)
    node_stat_rows: list[dict[str, Any]] = field(default_factory=list)
    phase_timing_rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class StaticJointTrainingConfig:
    total_epochs: int
    aux_weights: torch.Tensor


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
    advance_epoch_controls: bool = True,
    aux_weights: torch.Tensor | None = None,
) -> tuple[dict[str, Any], float]:
    epoch_started = perf_counter()
    trainer.state.epoch = epoch
    trainer.model.train()
    control_logs = {
        "phase": phase,
        "advance_epoch_controls": advance_epoch_controls,
        "dynamic_aux_step": aux_weights is not None,
    }
    trainer.callbacks.call("on_epoch_begin", trainer, epoch, logs=control_logs)
    running: dict[str, float] = {
        "loss": 0.0,
        "recon": 0.0,
        "recon_base": 0.0,
        "pix_recon": 0.0,
        "aux_recon_base": 0.0,
        "kl": 0.0,
    }
    n_batches = 0
    for batch_idx, x in enumerate(loader):
        trainer.state.batch = batch_idx
        trainer.callbacks.call(
            "on_batch_begin", trainer, epoch, batch_idx, logs=control_logs
        )
        if aux_weights is None:
            logs = trainer._run_batch(x)
        else:
            logs = _run_aux_weighted_batch(
                trainer,
                x,
                aux_weights=aux_weights,
            )
        trainer.callbacks.call(
            "on_batch_end", trainer, epoch, batch_idx, logs={**logs, **control_logs}
        )
        for key, value in logs.items():
            running.setdefault(key, 0.0)
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
            "logical_epoch": epoch,
            "advance_epoch_controls": advance_epoch_controls,
            "dynamic_aux_step": aux_weights is not None,
            "beta": float(trainer.loss_fn.beta),
            "aux_loss_weight": float(trainer.loss_fn.aux_loss_weight),
            "lr": float(trainer.optimizer.param_groups[0]["lr"]),
            "epoch_duration_sec": float(epoch_duration_sec),
            "elapsed_train_sec": float(elapsed_train_sec),
        }
    )
    trainer.state.history.append(epoch_logs)
    trainer.state.epoch_durations_sec.append(epoch_duration_sec)
    trainer.callbacks.call("on_epoch_end", trainer, epoch, logs=epoch_logs)
    if (
        advance_epoch_controls
        and trainer.scheduler is not None
        and epoch >= trainer.scheduler_start_epoch
    ):
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


def _run_aux_weighted_batch(
    trainer: Trainer,
    x: torch.Tensor,
    *,
    aux_weights: torch.Tensor,
) -> dict[str, float]:
    trainer.loss_fn.set_aux_weights(aux_weights)
    try:
        return trainer._run_batch(x)
    finally:
        trainer.loss_fn.reset_aux_weights()


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
    reference_tensor: torch.Tensor | None = None,
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
    applied_aux_weights: torch.Tensor | None = None
    trainer.callbacks.call("on_train_begin", trainer, logs={"training_type": "shapley"})
    for epoch in range(config.total_epochs):
        trainer.loss_fn.reset_aux_weights()
        use_dynamic_aux = (
            applied_aux_weights is not None
            and sampling_phase >= config.min_sampling_phases_before_dynamic
        )
        logs, elapsed_train_sec = _train_one_epoch(
            trainer,
            train_loader,
            epoch=epoch,
            phase="A",
            elapsed_train_sec=elapsed_train_sec,
            val_loader=val_loader,
            aux_weights=applied_aux_weights if use_dynamic_aux else None,
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
            reference_x=reference_tensor,
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
                "tactic": config.tactic,
            }
        )
        for row in estimator.sampler.stats.snapshot_rows(probabilities):
            row.update(
                {"epoch": epoch, "cycle": cycle, "sampling_phase": sampling_phase}
            )
            result.node_stat_rows.append(row)

        weights = auxiliary_shapley_weights(
            sampling.shapley_values.detach().cpu(),
            block_index,
            on_all_nonpositive=config.all_nonpositive_policy,
        )
        applied_aux_weights = weights.weights
        for block_idx, block in enumerate(block_index.blocks):
            result.shapley_weight_rows.append(
                {
                    "epoch": epoch,
                    "cycle": cycle,
                    "sampling_phase": sampling_phase,
                    "block": block.name,
                    "block_size": block.size,
                    "phi_raw": float(weights.raw_scores[block_idx].item()),
                    "weight": float(weights.weights[block_idx].item()),
                    "aux_weight_policy": weights.policy,
                }
            )

        trainer.loss_fn.reset_aux_weights()
        cycle += 1
    trainer.callbacks.call(
        "on_train_end", trainer, logs={"history": trainer.state.history}
    )
    trainer.state.train_duration_sec = perf_counter() - train_started
    trainer.state.train_ended_at = datetime.now(timezone.utc).isoformat()
    return result


def run_static_joint_training(
    *,
    trainer: Trainer,
    train_loader: DataLoader[torch.Tensor],
    block_index: FeatureBlockIndex,
    config: StaticJointTrainingConfig,
    val_loader: DataLoader[torch.Tensor] | None = None,
) -> list[dict[str, Any]]:
    train_started = perf_counter()
    trainer.state.train_started_at = datetime.now(timezone.utc).isoformat()
    elapsed_train_sec = (
        float(trainer.state.history[-1].get("elapsed_train_sec", 0.0))
        if trainer.state.history
        else 0.0
    )
    aux_weights = config.aux_weights.detach().to(
        device=trainer.device,
        dtype=trainer.loss_fn.aux_weights.dtype,
    )
    trainer.callbacks.call(
        "on_train_begin",
        trainer,
        logs={"training_type": "static-joint"},
    )
    for epoch in range(config.total_epochs):
        trainer.loss_fn.reset_aux_weights()
        logs, elapsed_train_sec = _train_one_epoch(
            trainer,
            train_loader,
            epoch=epoch,
            phase="A",
            elapsed_train_sec=elapsed_train_sec,
            val_loader=val_loader,
            aux_weights=aux_weights,
        )
        logs["cycle"] = 0
        logs["sampling_phase"] = 0
        logs["static_joint_weights"] = True
    trainer.loss_fn.reset_aux_weights()
    trainer.callbacks.call(
        "on_train_end", trainer, logs={"history": trainer.state.history}
    )
    trainer.state.train_duration_sec = perf_counter() - train_started
    trainer.state.train_ended_at = datetime.now(timezone.utc).isoformat()
    return trainer.state.history
