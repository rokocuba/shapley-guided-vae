"""Generate a clean composite Shapley weight evolution plot (3 tactics, vertical stack)
for the thesis. Reads weights from individual shapley run directories in iter_07."""

from __future__ import annotations

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from pathlib import Path

# --- config ---
BASE = Path("analysis/output/training_runs/iter_07")
OUT = Path("analysis/pictures/shapley_weights_comparison.png")
TACTICS = ["baseline", "marginal", "conditional"]
TACTIC_LABELS_HR = {
    "baseline": "- baseline",
    "marginal": "- marginal",
    "conditional": "- conditional",
}
BLOCK_COLORS = {
    "fou": "#1f77b4",
    "fac": "#ff7f0e",
    "kar": "#2ca02c",
    "zer": "#d62728",
    "mor": "#9467bd",
}


# --- read data from individual shapley run directories ---
def find_shapley_dirs() -> dict[str, Path]:
    """Map tactic -> run directory path using metadata.json."""
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


def load_tactic_weights(tactic: str, run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "shapley_weights.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No shapley_weights.csv for tactic {tactic}")
    df = pd.read_csv(csv_path)
    return df.sort_values(["sampling_phase", "block"])


dirs = find_shapley_dirs()
print(f"Found Shapley directories: {dict((t, d.name) for t, d in dirs.items())}")
frames = {t: load_tactic_weights(t, dirs[t]) for t in TACTICS}

# --- plot ---
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 200,
    }
)

fig, axes = plt.subplots(3, 1, figsize=(7.5, 9.0), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.35)

for ax, tactic in zip(axes, TACTICS):
    df = frames[tactic]
    phases_all = sorted(df["sampling_phase"].unique())
    act_phase = 4  # dynamic weights active from phase 4 onward

    for block in sorted(df["block"].unique()):
        blk = df[df["block"] == block]
        ax.plot(
            blk["sampling_phase"],
            blk["weight"],
            marker="o",
            linewidth=1.6,
            markersize=4.5,
            color=BLOCK_COLORS[block],
            label=block,
        )

    ax.set_title(f"Shapley {TACTIC_LABELS_HR[tactic]}", fontweight="bold")
    ax.set_ylabel("$w_g$")
    ax.set_ylim(0.0, 0.75)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.10))
    ax.legend(loc="upper right", ncol=3, framealpha=0.7, fontsize=7.5)
    ax.grid(True, alpha=0.25)

ax_last = axes[-1]
ax_last.set_xlabel("Shapley faza uzorkovanja (B-faza)")
ticks = range(1, len(phases_all) + 1)
ax_last.set_xticks(ticks)
ax_last.set_xticklabels([str(t) for t in ticks])

fig.tight_layout()

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {OUT}")
