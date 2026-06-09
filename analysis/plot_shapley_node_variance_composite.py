"""Generate a clean Shapley node variance plot -- all 3 tactics on the same axes.
Reads node stats from individual shapley run directories in iter_07."""

from __future__ import annotations

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from pathlib import Path

BASE = Path("analysis/output/training_runs/iter_07")
OUT = Path("analysis/pictures/shapley_node_variance.png")
TACTICS = ["baseline", "marginal", "conditional"]
TACTIC_LABELS_HR = {
    "baseline": "baseline",
    "marginal": "marginal",
    "conditional": "conditional",
}
TACTIC_COLORS = {
    "baseline": "#1f77b4",
    "marginal": "#ff7f0e",
    "conditional": "#2ca02c",
}


def find_shapley_dirs() -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for d in sorted(BASE.iterdir()):
        if not d.is_dir() or not d.name.startswith("shapley-"):
            continue
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        tactic = meta.get("shapley_tactic", None)
        if tactic in TACTICS:
            mapping[tactic] = d
    return mapping


def load_node_stats(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "shapley_node_stats.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No shapley_node_stats.csv in {run_dir.name}")
    return pd.read_csv(csv_path)


def agg_node_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("sampling_phase", as_index=False)
        .agg(
            mean_variance=("variance", "mean"),
            max_variance=("variance", "max"),
            mean_eff_count=("effective_count", "mean"),
        )
        .sort_values("sampling_phase")
    )


dirs = find_shapley_dirs()
summaries = {t: agg_node_stats(load_node_stats(dirs[t])) for t in TACTICS}
all_phases = sorted(list(summaries.values())[0]["sampling_phase"])

# --- plot ---
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 200,
    }
)

fig, ax = plt.subplots(figsize=(8.0, 4.8))

for tactic in TACTICS:
    s = summaries[tactic]
    phases = s["sampling_phase"].values
    color = TACTIC_COLORS[tactic]
    label = TACTIC_LABELS_HR[tactic]

    ax.plot(
        phases,
        s["mean_variance"],
        marker="o",
        linewidth=1.8,
        markersize=5,
        color=color,
        label=f"{label}",
    )

ax.set_xlabel("Shapley faza uzorkovanja")
ax.set_ylabel("srednja varijanca vrijednosti \u010dvora")
ax.set_xticks(all_phases)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.3f}"))
ax.legend(loc="upper right", framealpha=0.7, fontsize=7.5)
ax.grid(True, alpha=0.25)
ax.set_ylim(bottom=0.0)

fig.tight_layout()

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved -> {OUT}")
