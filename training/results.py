from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import torch
from torch import nn


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_run_id(training_type: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{training_type}-{stamp}-{uuid4().hex[:8]}"


@dataclass(slots=True)
class TrainingRunRecord:
    run_id: str
    training_type: str
    shapley_tactic: str | None
    started_at: str
    ended_at: str
    duration_sec: float
    epochs: int
    final_metrics: dict[str, float]
    config: dict[str, Any]
    data: dict[str, Any]
    timing: dict[str, Any]
    artifacts: dict[str, str]


def save_training_run(
    *,
    model: nn.Module,
    history: list[dict[str, float]],
    state: Any,
    config: dict[str, Any],
    data_info: dict[str, Any],
    output_dir: str | Path = "analysis/output/training_runs",
    training_type: str = "baseline",
    shapley_tactic: str | None = None,
) -> Path:
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    run_id = _make_run_id(training_type)
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    model_path = run_dir / "model.pt"
    history_path = run_dir / "history.csv"
    metadata_path = run_dir / "metadata.json"
    callback_timing_path = run_dir / "callback_timing.csv"

    torch.save(model.state_dict(), model_path)
    pd.DataFrame(history).assign(epoch=lambda df: df.index).to_csv(
        history_path, index=False
    )

    callback_timing = getattr(state, "callback_timing", {})
    callback_calls = getattr(state, "callback_calls", {})
    callback_rows = [
        {"hook": k, "seconds": float(v), "calls": int(callback_calls.get(k, 0))}
        for k, v in callback_timing.items()
    ]
    pd.DataFrame(callback_rows).to_csv(callback_timing_path, index=False)

    record = TrainingRunRecord(
        run_id=run_id,
        training_type=training_type,
        shapley_tactic=shapley_tactic,
        started_at=getattr(state, "train_started_at", _now_utc()),
        ended_at=getattr(state, "train_ended_at", _now_utc()),
        duration_sec=float(getattr(state, "train_duration_sec", 0.0)),
        epochs=len(history),
        final_metrics=history[-1] if history else {},
        config=config,
        data=data_info,
        timing={
            "epoch_durations_sec": getattr(state, "epoch_durations_sec", []),
            "callback_timing_sec": callback_timing,
            "callback_calls": callback_calls,
        },
        artifacts={
            "model": str(model_path),
            "history_csv": str(history_path),
            "callback_timing_csv": str(callback_timing_path),
        },
    )
    metadata_path.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
    return run_dir


def load_training_runs(
    output_dir: str | Path = "analysis/output/training_runs",
    training_type: str | None = None,
    shapley_tactic: str | None = None,
) -> list[TrainingRunRecord]:
    out_root = Path(output_dir)
    if not out_root.exists():
        return []

    runs: list[TrainingRunRecord] = []
    for metadata_path in sorted(out_root.glob("*/metadata.json")):
        item = json.loads(metadata_path.read_text(encoding="utf-8"))
        record = TrainingRunRecord(**item)
        if training_type is not None and record.training_type != training_type:
            continue
        if shapley_tactic is not None and record.shapley_tactic != shapley_tactic:
            continue
        runs.append(record)
    return runs
