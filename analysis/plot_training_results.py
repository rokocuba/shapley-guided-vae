from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import load_training_runs


def _latest_runs_per_training_type(run_dir: Path) -> set[str]:
    runs = load_training_runs(run_dir)
    latest: dict[str, object] = {}
    for r in runs:
        prev = latest.get(r.training_type)
        if prev is None or r.started_at > prev.started_at:
            latest[r.training_type] = r
    return {r.run_id for r in latest.values()}


def build_summary_frame(
    run_dir: Path, selected_run_ids: set[str] | None = None
) -> pd.DataFrame:
    runs = load_training_runs(run_dir)
    rows: list[dict[str, object]] = []
    for r in runs:
        if selected_run_ids is not None and r.run_id not in selected_run_ids:
            continue
        rows.append(
            {
                "run_id": r.run_id,
                "training_type": r.training_type,
                "shapley_tactic": r.shapley_tactic or "none",
                "latent_dim": r.config.get("latent_dim"),
                "started_at": r.started_at,
                "ended_at": r.ended_at,
                "duration_sec": r.duration_sec,
                "epochs": r.epochs,
                "final_loss": r.final_metrics.get("loss"),
                "final_recon": r.final_metrics.get("recon"),
                "final_kl": r.final_metrics.get("kl"),
                "final_val_loss": r.final_metrics.get("val_loss"),
                "final_val_recon": r.final_metrics.get("val_recon"),
                "final_val_kl": r.final_metrics.get("val_kl"),
            }
        )
    return pd.DataFrame(rows).sort_values("started_at") if rows else pd.DataFrame()


def build_history_frame(
    run_dir: Path, selected_run_ids: set[str] | None = None
) -> pd.DataFrame:
    runs = load_training_runs(run_dir)
    frames: list[pd.DataFrame] = []
    for r in runs:
        if selected_run_ids is not None and r.run_id not in selected_run_ids:
            continue
        path = Path(r.artifacts["history_csv"])
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["run_id"] = r.run_id
        df["training_type"] = r.training_type
        df["shapley_tactic"] = r.shapley_tactic or "none"
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _read_optional_artifact(record: object, key: str) -> pd.DataFrame:
    artifact = record.artifacts.get(key)
    if not artifact:
        return pd.DataFrame()
    path = Path(artifact)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_shapley_weights_frame(
    run_dir: Path, selected_run_ids: set[str] | None = None
) -> pd.DataFrame:
    runs = load_training_runs(run_dir)
    frames: list[pd.DataFrame] = []
    for r in runs:
        if selected_run_ids is not None and r.run_id not in selected_run_ids:
            continue
        df = _read_optional_artifact(r, "shapley_weights_csv")
        if df.empty:
            continue
        df["run_id"] = r.run_id
        df["training_type"] = r.training_type
        df["shapley_tactic"] = r.shapley_tactic or "none"
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_shapley_node_stats_frame(
    run_dir: Path, selected_run_ids: set[str] | None = None
) -> pd.DataFrame:
    runs = load_training_runs(run_dir)
    frames: list[pd.DataFrame] = []
    for r in runs:
        if selected_run_ids is not None and r.run_id not in selected_run_ids:
            continue
        df = _read_optional_artifact(r, "shapley_node_stats_csv")
        if df.empty:
            continue
        df["run_id"] = r.run_id
        df["training_type"] = r.training_type
        df["shapley_tactic"] = r.shapley_tactic or "none"
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_shapley_phase_timing_frame(
    run_dir: Path, selected_run_ids: set[str] | None = None
) -> pd.DataFrame:
    runs = load_training_runs(run_dir)
    frames: list[pd.DataFrame] = []
    for r in runs:
        if selected_run_ids is not None and r.run_id not in selected_run_ids:
            continue
        df = _read_optional_artifact(r, "shapley_phase_timing_csv")
        if df.empty:
            continue
        df["run_id"] = r.run_id
        df["training_type"] = r.training_type
        df["shapley_tactic"] = r.shapley_tactic or "none"
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_duration(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return
    g = summary.groupby(["training_type", "shapley_tactic"], as_index=False)[
        "duration_sec"
    ].mean()
    labels = [f"{t}|{s}" for t, s in zip(g["training_type"], g["shapley_tactic"])]
    plt.figure(figsize=(10, 4))
    plt.bar(labels, g["duration_sec"])
    plt.ylabel("seconds")
    plt.title("Mean Training Duration by Type/Tactic")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "duration_by_type_tactic.png", dpi=150)
    plt.close()


def plot_duration_timeline(summary_all: pd.DataFrame, out_dir: Path) -> None:
    if summary_all.empty:
        return
    df = summary_all.copy()
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["started_at"])
    plt.figure(figsize=(11, 5))
    for (training_type, tactic), g in df.groupby(["training_type", "shapley_tactic"]):
        g = g.sort_values("started_at")
        label = f"{training_type}|{tactic}"
        plt.plot(
            g["started_at"], g["duration_sec"], marker="o", linewidth=1.2, label=label
        )
    plt.xlabel("started_at (UTC)")
    plt.ylabel("duration (sec)")
    plt.title("Training Duration Timeline (All Runs)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "duration_timeline_all_runs.png", dpi=150)
    plt.close()


def plot_single_metric_curves(
    history: pd.DataFrame,
    out_dir: Path,
    metric_key: str,
    title: str,
    y_label: str,
    file_name: str,
    split_label: str,
    use_log_scale: bool = False,
) -> None:
    if history.empty:
        return

    use_elapsed_seconds = (
        "elapsed_train_sec" in history.columns
        and history["elapsed_train_sec"].notna().any()
    )
    x_key = "elapsed_train_sec" if use_elapsed_seconds else "epoch"
    max_x = 0.0
    plotted_any = False

    plt.figure(figsize=(10, 5))
    for run_id, g in history.groupby("run_id"):
        g = g.sort_values("epoch")
        if metric_key not in g.columns or not g[metric_key].notna().any():
            continue

        x_values = g[x_key]
        y_values = g[metric_key].astype(float)
        if use_log_scale:
            y_values = y_values.where(y_values > 0.0)
            if not y_values.notna().any():
                continue
        label = (
            f"{g['training_type'].iloc[0]}|{g['shapley_tactic'].iloc[0]}|{run_id[-6:]}"
        )
        plt.plot(
            x_values,
            y_values,
            linewidth=1.3,
            alpha=0.8,
            label=label,
        )
        plotted_any = True
        if len(x_values) > 0:
            max_x = max(max_x, float(x_values.iloc[-1]))

    if use_elapsed_seconds and max_x > 0.0:
        n_lines = int(math.floor(max_x / 15.0))
        for k in range(1, n_lines + 1):
            plt.axvline(
                15.0 * k,
                color="black",
                linewidth=0.6,
                alpha=0.2,
                zorder=0,
            )

    if use_log_scale and plotted_any:
        plt.yscale("log")

    plt.xlabel("elapsed training seconds" if use_elapsed_seconds else "epoch")
    plt.ylabel(f"{y_label} (log scale)" if use_log_scale else y_label)
    plt.title(f"{title} (log scale)" if use_log_scale else title)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / file_name, dpi=150)
    plt.close()


def plot_shapley_weights(weights: pd.DataFrame, out_dir: Path) -> None:
    if weights.empty or "weight_applied" not in weights.columns:
        return
    for run_id, g in weights.groupby("run_id"):
        g = g.sort_values(["sampling_phase", "block"])
        plt.figure(figsize=(10, 5))
        for block, block_g in g.groupby("block"):
            plt.plot(
                block_g["sampling_phase"],
                block_g["weight_applied"],
                marker="o",
                linewidth=1.4,
                label=str(block),
            )
        title = (
            f"Shapley Block Weights "
            f"({g['shapley_tactic'].iloc[0]}|{run_id[-6:]})"
        )
        plt.xlabel("sampling phase")
        plt.ylabel("applied block weight")
        plt.title(title)
        plt.ylim(bottom=0.0)
        plt.legend(fontsize=8, ncol=3)
        plt.tight_layout()
        plt.savefig(out_dir / f"shapley_weights_{run_id}.png", dpi=150)
        plt.close()


def plot_shapley_node_variance(node_stats: pd.DataFrame, out_dir: Path) -> None:
    if node_stats.empty or "variance" not in node_stats.columns:
        return
    summary = (
        node_stats.groupby(["run_id", "shapley_tactic", "sampling_phase"], as_index=False)
        .agg(
            mean_variance=("variance", "mean"),
            max_variance=("variance", "max"),
            mean_effective_count=("effective_count", "mean"),
        )
        .sort_values(["run_id", "sampling_phase"])
    )
    summary.to_csv(out_dir / "shapley_node_variance_summary.csv", index=False)
    plt.figure(figsize=(10, 5))
    plotted = False
    for run_id, g in summary.groupby("run_id"):
        label_base = f"{g['shapley_tactic'].iloc[0]}|{run_id[-6:]}"
        plt.plot(
            g["sampling_phase"],
            g["mean_variance"],
            marker="o",
            linewidth=1.3,
            label=f"{label_base} mean",
        )
        plt.plot(
            g["sampling_phase"],
            g["max_variance"],
            marker="x",
            linewidth=1.0,
            alpha=0.8,
            label=f"{label_base} max",
        )
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel("sampling phase")
    plt.ylabel("node value variance")
    plt.title("Shapley Node Variance by Sampling Phase")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "shapley_node_variance.png", dpi=150)
    plt.close()


def plot_shapley_phase_timing(phase_timing: pd.DataFrame, out_dir: Path) -> None:
    if phase_timing.empty or "duration_sec" not in phase_timing.columns:
        return
    g = (
        phase_timing.groupby(["shapley_tactic", "phase"], as_index=False)[
            "duration_sec"
        ]
        .sum()
        .sort_values(["shapley_tactic", "phase"])
    )
    g.to_csv(out_dir / "shapley_phase_timing_summary.csv", index=False)
    labels = [f"{t}|{p}" for t, p in zip(g["shapley_tactic"], g["phase"])]
    plt.figure(figsize=(10, 4))
    plt.bar(labels, g["duration_sec"])
    plt.ylabel("seconds")
    plt.title("Shapley Phase Timing")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "shapley_phase_timing.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", type=Path, default=Path("analysis/output/training_runs")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("analysis/output/training_plots")
    )
    parser.add_argument("--all-runs", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    stale_plot_files = [
        args.out / "loss_curves.png",
        args.out / "recon_curves.png",
        args.out / "recon_train_curves.png",
        args.out / "recon_test_curves.png",
        args.out / "kl_curves.png",
        args.out / "shapley_node_variance.png",
        args.out / "shapley_phase_timing.png",
    ]
    for path in stale_plot_files:
        if path.exists():
            path.unlink()

    summary_all = build_summary_frame(args.runs, selected_run_ids=None)
    selected_run_ids = (
        None if args.all_runs else _latest_runs_per_training_type(args.runs)
    )
    summary = build_summary_frame(args.runs, selected_run_ids=selected_run_ids)
    history = build_history_frame(args.runs, selected_run_ids=selected_run_ids)
    shapley_weights = build_shapley_weights_frame(
        args.runs, selected_run_ids=selected_run_ids
    )
    shapley_node_stats = build_shapley_node_stats_frame(
        args.runs, selected_run_ids=selected_run_ids
    )
    shapley_phase_timing = build_shapley_phase_timing_frame(
        args.runs, selected_run_ids=selected_run_ids
    )

    summary_all.to_csv(args.out / "run_summary_all.csv", index=False)
    summary.to_csv(args.out / "run_summary.csv", index=False)
    history.to_csv(args.out / "run_history.csv", index=False)
    shapley_weights.to_csv(args.out / "shapley_weights.csv", index=False)
    shapley_node_stats.to_csv(args.out / "shapley_node_stats.csv", index=False)
    shapley_phase_timing.to_csv(args.out / "shapley_phase_timing.csv", index=False)

    baseline_all = (
        summary_all[summary_all["training_type"] == "baseline"].copy()
        if not summary_all.empty
        else pd.DataFrame()
    )
    baseline_all.to_csv(args.out / "baseline_duration_history.csv", index=False)

    plot_duration(summary, args.out)
    plot_duration_timeline(summary_all, args.out)
    plot_single_metric_curves(
        history,
        args.out,
        metric_key="recon",
        title="Train Reconstruction Curves",
        y_label="reconstruction loss",
        file_name="recon_train_curves.png",
        split_label="train",
        use_log_scale=True,
    )
    plot_single_metric_curves(
        history,
        args.out,
        metric_key="val_recon",
        title="Test Reconstruction Curves",
        y_label="reconstruction loss",
        file_name="recon_test_curves.png",
        split_label="test",
        use_log_scale=True,
    )
    plot_single_metric_curves(
        history,
        args.out,
        metric_key="kl",
        title="Train KL Curves",
        y_label="KL loss",
        file_name="kl_curves.png",
        split_label="train",
        use_log_scale=True,
    )
    plot_shapley_weights(shapley_weights, args.out)
    plot_shapley_node_variance(shapley_node_stats, args.out)
    plot_shapley_phase_timing(shapley_phase_timing, args.out)
    print(f"saved_plots={args.out}")


if __name__ == "__main__":
    main()
