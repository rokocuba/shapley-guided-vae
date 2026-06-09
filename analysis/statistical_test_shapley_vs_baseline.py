"""
Statistical test: Is Shapley-guided dynamic auxiliary weighting better than the
static pix-aux baseline?

Three separate linear regressions (one per Shapley masking tactic), each
controlling for final val_kl to isolate the effect of the training strategy
on val_pix_recon.

Model:
    val_pix_recon = β0 + β1 * val_kl + β2 * is_baseline + ε

    is_baseline = 1 for baseline, 0 for Shapley variant.

Hypothesis (one-sided):
    H0: β2 ≤ 0   (baseline is NOT worse → Shapley is NOT better)
    HA: β2 > 0   (baseline IS worse → Shapley IS better)

A positive β2 means that at the same KL level, the baseline has higher pixel
reconstruction loss than the Shapley variant — i.e., Shapley is better.

Also reports a simple paired t-test for reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = ROOT / "analysis" / "output" / "training_runs"

SHAPLEY_TACTICS = ["baseline", "marginal", "conditional"]
TACTIC_LABELS = {
    "baseline": "Shapley (global-mean replacement)",
    "marginal": "Shapley (marginal replacement)",
    "conditional": "Shapley (same-digit replacement)",
}


class TestResult(NamedTuple):
    tactic: str
    n_pairs: int
    # Paired t-test
    mean_diff: float  # baseline - shapley (positive = baseline worse = shapley better)
    t_stat: float
    t_p_value_one_sided: float
    # OLS with KL control
    beta2: float  # coefficient on is_baseline
    beta2_se: float
    beta2_p_value_one_sided: float
    beta2_ci_95: tuple[float, float]
    r_squared: float
    # Descriptive
    baseline_mean_recon: float
    shapley_mean_recon: float
    baseline_mean_kl: float
    shapley_mean_kl: float


def load_paired_data(runs_dir: Path) -> pd.DataFrame:
    """Load all iteration run_summary.csv files and return one merged DataFrame."""
    frames: list[pd.DataFrame] = []
    for iter_dir in sorted(runs_dir.glob("iter_*")):
        summary_path = iter_dir / "run_summary.csv"
        if not summary_path.exists():
            continue
        df = pd.read_csv(summary_path)
        df["iteration"] = iter_dir.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No iter_*/run_summary.csv found under {runs_dir}")
    return pd.concat(frames, ignore_index=True)


def _run_single_test(
    df: pd.DataFrame,
    tactic: str,
) -> TestResult:
    """Run both paired t-test and KL-controlled OLS for one Shapley tactic vs baseline."""
    # Extract paired data
    baseline_df = df[df["training_type"] == "baseline"].copy()
    shapley_df = df[
        (df["training_type"] == "shapley") & (df["shapley_tactic"] == tactic)
    ].copy()

    # Merge on iteration to ensure pairing
    merged = baseline_df[["iteration", "final_val_pix_recon", "final_val_kl"]].merge(
        shapley_df[["iteration", "final_val_pix_recon", "final_val_kl"]],
        on="iteration",
        suffixes=("_baseline", "_shapley"),
    )
    n = len(merged)
    if n == 0:
        raise ValueError(f"No paired data found for tactic '{tactic}'.")

    # ---- Paired t-test ----
    diffs = (
        merged["final_val_pix_recon_baseline"].values
        - merged["final_val_pix_recon_shapley"].values
    )
    mean_diff = float(np.mean(diffs))
    t_stat, t_p_two_sided = scipy.stats.ttest_1samp(diffs, popmean=0.0)
    t_p_one_sided = t_p_two_sided / 2.0 if t_stat > 0 else 1.0 - t_p_two_sided / 2.0

    # ---- OLS: val_pix_recon ~ val_kl + is_baseline ----
    # Stack: 2 rows per iteration
    y = np.concatenate(
        [
            merged["final_val_pix_recon_baseline"].values,
            merged["final_val_pix_recon_shapley"].values,
        ]
    )
    kl = np.concatenate(
        [
            merged["final_val_kl_baseline"].values,
            merged["final_val_kl_shapley"].values,
        ]
    )
    is_baseline = np.concatenate([np.ones(n), np.zeros(n)])

    X = np.column_stack([np.ones(2 * n), kl, is_baseline])
    model = sm.OLS(y, X)
    results = model.fit()

    beta2 = float(results.params[2])
    beta2_se = float(results.bse[2])
    # One-sided p-value for HA: β2 > 0
    t_val = beta2 / beta2_se if beta2_se > 0 else 0.0
    beta2_p_one_sided = 1.0 - float(scipy.stats.t.cdf(t_val, df=results.df_resid))
    ci = results.conf_int(alpha=0.10)  # 90% CI → 95% one-sided lower bound
    beta2_ci_95 = (float(ci[2, 0]), float(ci[2, 1]))

    return TestResult(
        tactic=tactic,
        n_pairs=n,
        mean_diff=mean_diff,
        t_stat=float(t_stat),
        t_p_value_one_sided=float(t_p_one_sided),
        beta2=beta2,
        beta2_se=beta2_se,
        beta2_p_value_one_sided=float(beta2_p_one_sided),
        beta2_ci_95=beta2_ci_95,
        r_squared=float(results.rsquared),
        baseline_mean_recon=float(merged["final_val_pix_recon_baseline"].mean()),
        shapley_mean_recon=float(merged["final_val_pix_recon_shapley"].mean()),
        baseline_mean_kl=float(merged["final_val_kl_baseline"].mean()),
        shapley_mean_kl=float(merged["final_val_kl_shapley"].mean()),
    )


def run_all_tests(runs_dir: Path) -> list[TestResult]:
    """Run the three comparisons and return results."""
    df = load_paired_data(runs_dir)
    results: list[TestResult] = []
    for tactic in SHAPLEY_TACTICS:
        result = _run_single_test(df, tactic)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_scatter_with_regression(
    df: pd.DataFrame,
    results: list[TestResult],
    out_path: Path,
) -> None:
    """Scatter grid: baseline vs Shapley val_pix_recon, one panel per tactic."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    colors = {"baseline": "#de2d26", "shapley": "#2ca25f"}

    for ax, tactic in zip(axes, SHAPLEY_TACTICS):
        baseline_df = df[df["training_type"] == "baseline"]
        shapley_df = df[
            (df["training_type"] == "shapley") & (df["shapley_tactic"] == tactic)
        ]
        merged = baseline_df[
            ["iteration", "final_val_pix_recon", "final_val_kl"]
        ].merge(
            shapley_df[["iteration", "final_val_pix_recon", "final_val_kl"]],
            on="iteration",
            suffixes=("_baseline", "_shapley"),
        )

        x = merged["final_val_pix_recon_shapley"].values
        y = merged["final_val_pix_recon_baseline"].values

        # Identity line: y = x
        lims = [
            min(x.min(), y.min()) - 0.002,
            max(x.max(), y.max()) + 0.002,
        ]
        ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.4, label="y = x (equal)")

        # Scatter
        ax.scatter(x, y, c="#333333", s=45, alpha=0.7, zorder=3)

        # Mean point
        ax.scatter(
            x.mean(),
            y.mean(),
            c="#de2d26",
            s=120,
            marker="X",
            zorder=5,
            edgecolors="white",
            linewidths=1.0,
            label=f"mean (shapley={x.mean():.4f}, baseline={y.mean():.4f})",
        )

        above = (y > x).sum()
        below = (y < x).sum()
        ax.set_title(
            f"{TACTIC_LABELS.get(tactic, tactic)}\n"
            f"baseline worse in {above}/{len(x)} pairs",
            fontsize=11,
        )
        ax.set_xlabel("Shapley val_pix_recon")
        if ax is axes[0]:
            ax.set_ylabel("Baseline val_pix_recon")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Baseline vs Shapley: Final val_pix_recon (N=50 paired iterations)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Scatter plot saved to {out_path}")


def plot_kl_controlled_regression(
    df: pd.DataFrame,
    results: list[TestResult],
    out_path: Path,
) -> None:
    """Show val_pix_recon vs val_kl with regression lines for baseline and Shapley."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, tactic in zip(axes, SHAPLEY_TACTICS):
        baseline_df = df[df["training_type"] == "baseline"]
        shapley_df = df[
            (df["training_type"] == "shapley") & (df["shapley_tactic"] == tactic)
        ]
        merged = baseline_df[
            ["iteration", "final_val_pix_recon", "final_val_kl"]
        ].merge(
            shapley_df[["iteration", "final_val_pix_recon", "final_val_kl"]],
            on="iteration",
            suffixes=("_baseline", "_shapley"),
        )

        # Build OLS to plot regression lines
        n = len(merged)
        y = np.concatenate(
            [
                merged["final_val_pix_recon_baseline"].values,
                merged["final_val_pix_recon_shapley"].values,
            ]
        )
        kl = np.concatenate(
            [
                merged["final_val_kl_baseline"].values,
                merged["final_val_kl_shapley"].values,
            ]
        )
        is_baseline = np.concatenate([np.ones(n), np.zeros(n)])

        # Scatter
        ax.scatter(
            merged["final_val_kl_shapley"],
            merged["final_val_pix_recon_shapley"],
            c="#2ca25f",
            s=45,
            alpha=0.7,
            label="Shapley",
            zorder=3,
        )
        ax.scatter(
            merged["final_val_kl_baseline"],
            merged["final_val_pix_recon_baseline"],
            c="#de2d26",
            s=45,
            alpha=0.7,
            label="Baseline",
            zorder=3,
        )

        # Regression lines (simple separate fits for visual reference)
        kl_range = np.linspace(kl.min(), kl.max(), 50)
        for label, indicator, color in [
            ("Shapley", 0.0, "#2ca25f"),
            ("Baseline", 1.0, "#de2d26"),
        ]:
            X_line = np.column_stack(
                [np.ones_like(kl_range), kl_range, np.full_like(kl_range, indicator)]
            )
            X_full = np.column_stack([np.ones(2 * n), kl, is_baseline])
            ols_fit = sm.OLS(y, X_full).fit()
            y_line = ols_fit.predict(X_line)
            ax.plot(kl_range, y_line, color=color, linewidth=1.5, alpha=0.8)

        # Find result for this tactic
        r = next(r_ for r_ in results if r_.tactic == tactic)
        ax.set_title(
            f"{TACTIC_LABELS.get(tactic, tactic)}\n"
            f"β₂ = {r.beta2:.5f}  (SE={r.beta2_se:.5f})\n"
            f"one-sided p = {r.beta2_p_value_one_sided:.4f}",
            fontsize=10,
        )
        ax.set_xlabel("final val_kl")
        if ax is axes[0]:
            ax.set_ylabel("final val_pix_recon")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "KL-Controlled Comparison: val_pix_recon vs val_kl",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.93))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"KL-controlled plot saved to {out_path}")


def plot_coefficient_forest(
    results: list[TestResult],
    out_path: Path,
) -> None:
    """Forest plot of β₂ estimates with 95% confidence intervals."""
    fig, ax = plt.subplots(figsize=(10, 4.5))

    tactics = [r.tactic for r in results]
    betas = [r.beta2 for r in results]
    ses = [r.beta2_se for r in results]
    cis = [r.beta2_ci_95 for r in results]
    p_vals = [r.beta2_p_value_one_sided for r in results]

    y_positions = np.arange(len(tactics))

    for i, (tactic, beta, se, ci, p) in enumerate(
        zip(tactics, betas, ses, cis, p_vals)
    ):
        color = "#2ca25f" if p < 0.05 else "#636363"
        ax.errorbar(
            beta,
            i,
            xerr=[[beta - ci[0]], [ci[1] - beta]],
            fmt="o",
            capsize=6,
            markersize=10,
            color=color,
            linewidth=2.0,
            zorder=3,
        )

    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([TACTIC_LABELS.get(t, t) for t in tactics])
    ax.set_xlabel("β₂ (is_baseline coefficient)")
    ax.set_title(
        "Effect of baseline (vs Shapley) on val_pix_recon, controlling for val_kl\n"
        "β₂ > 0 ⇒ baseline has higher loss ⇒ Shapley is better",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3, axis="x")

    # Annotate with p-values
    for i, (beta, p) in enumerate(zip(betas, p_vals)):
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        ax.annotate(
            f"β₂={beta:.5f}, p={p:.4f} {sig}",
            xy=(beta, i),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Coefficient forest plot saved to {out_path}")


def plot_paired_differences(
    df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Histogram of paired differences (baseline - Shapley) for each tactic."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, tactic in zip(axes, SHAPLEY_TACTICS):
        baseline_df = df[df["training_type"] == "baseline"]
        shapley_df = df[
            (df["training_type"] == "shapley") & (df["shapley_tactic"] == tactic)
        ]
        merged = baseline_df[["iteration", "final_val_pix_recon"]].merge(
            shapley_df[["iteration", "final_val_pix_recon"]],
            on="iteration",
            suffixes=("_baseline", "_shapley"),
        )
        diffs = (
            merged["final_val_pix_recon_baseline"].values
            - merged["final_val_pix_recon_shapley"].values
        )

        ax.hist(diffs, bins=20, color="#555555", edgecolor="white", alpha=0.8)
        ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.axvline(
            np.mean(diffs),
            color="#de2d26",
            linewidth=1.5,
            label=f"mean={np.mean(diffs):.5f}",
        )

        above_zero = (diffs > 0).sum()
        ax.set_title(
            f"{TACTIC_LABELS.get(tactic, tactic)}\n"
            f"baseline - shapley > 0 in {above_zero}/{len(diffs)} pairs\n"
            f"mean diff = {np.mean(diffs):.5f}",
            fontsize=10,
        )
        ax.set_xlabel("baseline val_pix_recon − shapley val_pix_recon")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Paired Differences: Baseline − Shapley val_pix_recon\n(positive = baseline worse = Shapley better)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.93))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Paired differences plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(results: list[TestResult]) -> None:
    """Print a formatted statistical report."""
    header = (
        f"{'Tactic':<16} {'N':>4}  "
        f"{'Mean Diff':>10}  {'t-stat':>8}  {'t p(1s)':>9}  "
        f"{'β₂':>10}  {'SE(β₂)':>9}  {'β₂ p(1s)':>9}  "
        f"{'R²':>6}  {'B.mean':>9}  {'S.mean':>9}"
    )
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print("STATISTICAL REPORT: Shapley vs Baseline")
    print("=" * len(header))
    print()
    print("H0: β₂ ≤ 0  (baseline NOT worse, Shapley NOT better)")
    print("HA: β₂ > 0  (baseline IS worse, Shapley IS better)")
    print()
    print("Paired t-test tests: mean(baseline - shapley) > 0")
    print("OLS tests: β₂ > 0 controlling for val_kl")
    print()
    print(header)
    print(sep)

    for r in results:
        sig = (
            "***"
            if r.beta2_p_value_one_sided < 0.001
            else (
                "**"
                if r.beta2_p_value_one_sided < 0.01
                else ("*" if r.beta2_p_value_one_sided < 0.05 else "")
            )
        )
        print(
            f"{r.tactic:<16} {r.n_pairs:>4}  "
            f"{r.mean_diff:>10.6f}  {r.t_stat:>8.3f}  {r.t_p_value_one_sided:>9.4f}  "
            f"{r.beta2:>10.6f}  {r.beta2_se:>9.6f}  {r.beta2_p_value_one_sided:>9.4f}{sig:<3}  "
            f"{r.r_squared:>6.3f}  {r.baseline_mean_recon:>9.6f}  {r.shapley_mean_recon:>9.6f}"
        )

    print(sep)
    print()
    print("Significance codes: *** p<0.001  ** p<0.01  * p<0.05")
    print()
    print("Interpretation:")
    print("  mean_diff > 0 → baseline has higher recon loss → Shapley better (raw)")
    print(
        "  β₂ > 0 → baseline has higher recon loss at same KL → Shapley better (KL-controlled)"
    )
    print("  One-sided p-value < 0.05 → reject H0, Shapley is significantly better")

    # Also report OLS details for each tactic
    print("\n" + "-" * 70)
    print("DETAILED OLS RESULTS")
    print("-" * 70)
    for r in results:
        print(f"\n--- {TACTIC_LABELS.get(r.tactic, r.tactic)} ---")
        print(f"  N pairs:            {r.n_pairs}")
        print(f"  Mean diff (B−S):    {r.mean_diff:.6f}")
        print(
            f"  Baseline mean recon:{r.baseline_mean_recon:.6f}  (KL={r.baseline_mean_kl:.4f})"
        )
        print(
            f"  Shapley mean recon: {r.shapley_mean_recon:.6f}  (KL={r.shapley_mean_kl:.4f})"
        )
        print(f"  β₂ (is_baseline):   {r.beta2:.6f}  SE={r.beta2_se:.6f}")
        print(f"  90% CI for β₂:      [{r.beta2_ci_95[0]:.6f}, {r.beta2_ci_95[1]:.6f}]")
        print(f"  One-sided p (β₂>0): {r.beta2_p_value_one_sided:.6f}")
        print(
            f"  Paired t-stat:      {r.t_stat:.4f}  one-sided p={r.t_p_value_one_sided:.6f}"
        )
        print(f"  R²:                 {r.r_squared:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statistical test: Shapley-guided vs baseline VAE"
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Path to training_runs directory containing iter_* folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for output plots (default: runs_dir).",
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir
    if not runs_dir.exists():
        print(f"Error: runs directory not found: {runs_dir}")
        sys.exit(1)

    out_dir = args.out_dir or runs_dir

    print(f"Loading data from {runs_dir} ...")
    df = load_paired_data(runs_dir)

    print(f"Loaded {len(df)} runs across {df['iteration'].nunique()} iterations.")
    print(f"Training types: {df['training_type'].value_counts().to_dict()}")

    results = run_all_tests(runs_dir)

    # Print text report
    print_report(results)

    # Generate plots
    print("\nGenerating plots...")
    plot_paired_differences(df, out_dir / "stat_test_paired_differences.png")
    plot_scatter_with_regression(df, results, out_dir / "stat_test_scatter.png")
    plot_kl_controlled_regression(df, results, out_dir / "stat_test_kl_controlled.png")
    plot_coefficient_forest(results, out_dir / "stat_test_coefficient_forest.png")

    # Save results table
    rows = []
    for r in results:
        rows.append(
            {
                "tactic": r.tactic,
                "n_pairs": r.n_pairs,
                "mean_diff": r.mean_diff,
                "t_stat": r.t_stat,
                "t_p_value_one_sided": r.t_p_value_one_sided,
                "beta2": r.beta2,
                "beta2_se": r.beta2_se,
                "beta2_p_value_one_sided": r.beta2_p_value_one_sided,
                "beta2_ci_95_lower": r.beta2_ci_95[0],
                "beta2_ci_95_upper": r.beta2_ci_95[1],
                "r_squared": r.r_squared,
                "baseline_mean_recon": r.baseline_mean_recon,
                "shapley_mean_recon": r.shapley_mean_recon,
                "baseline_mean_kl": r.baseline_mean_kl,
                "shapley_mean_kl": r.shapley_mean_kl,
            }
        )
    results_df = pd.DataFrame(rows)
    results_csv = out_dir / "stat_test_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nResults table saved to {results_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
