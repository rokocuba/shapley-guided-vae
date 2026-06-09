"""Generate a clean, thesis-quality forest plot of β₂ coefficients.
Croatian labels, minimal clutter. Reads from stat_test_results.csv."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

BASE = Path("analysis/output/training_runs")
OUT = Path("analysis/pictures/stat_test_coefficient_forest.png")

# --- mapped Croatian tactic labels ---
TACTIC_HR = {
    "baseline": "baseline",
    "marginal": "marginal",
    "conditional": "conditional",
}

# --- read data ---
df = pd.read_csv(BASE / "stat_test_results.csv")

# --- plot ---
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 200,
    }
)

fig, ax = plt.subplots(figsize=(7.5, 3.2))

tactics = df["tactic"].tolist()
betas = df["beta2"].tolist()
cis_lo = (df["beta2"] - df["beta2_ci_95_lower"]).tolist()
cis_hi = (df["beta2_ci_95_upper"] - df["beta2"]).tolist()
p_vals = df["beta2_p_value_one_sided"].tolist()

y_positions = [2, 1, 0]  # top to bottom: baseline → marginal → conditional

for i, (tactic, beta, lo, hi, p) in enumerate(
    zip(tactics, betas, cis_lo, cis_hi, p_vals)
):
    color = "#2ca25f" if p < 0.05 else "#636363"
    y = y_positions[i]
    ax.errorbar(
        beta,
        y,
        xerr=[[lo], [hi]],
        fmt="o",
        capsize=6,
        markersize=9,
        color=color,
        linewidth=2.0,
        zorder=3,
    )

ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
ax.set_yticks(y_positions)
ax.set_yticklabels([TACTIC_HR.get(t, t) for t in tactics])
ax.set_xlabel("$\\beta_2$ (doprinos statičkog baselinea)")
ax.grid(True, alpha=0.25, axis="x")

# annotate with β₂ and significance
for i, (beta, p) in enumerate(zip(betas, p_vals)):
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
    y = y_positions[i]
    ax.annotate(
        f"$\\beta_2$ = {beta:.5f}  {sig}",
        xy=(beta, y),
        xytext=(5, 7),
        textcoords="offset points",
        fontsize=9,
        va="bottom",
    )

fig.tight_layout()

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved -> {OUT}")
