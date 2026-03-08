from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "analysis" / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def load_datasets():
    red = pd.read_csv(DATA_DIR / "winequality-red.csv", sep=";")
    white = pd.read_csv(DATA_DIR / "winequality-white.csv", sep=";")
    both = pd.concat([red, white], ignore_index=True)
    return {"red": red, "white": white, "both": both}


def build_summary(df):
    summary = df[FEATURES].describe(percentiles=[0.25, 0.5, 0.75]).T
    summary = summary.rename(columns={"25%": "q1", "50%": "median", "75%": "q3"})
    summary["variance"] = df[FEATURES].var()
    summary["skewness"] = df[FEATURES].skew()
    summary["kurtosis"] = df[FEATURES].kurtosis()
    return summary[
        [
            "count",
            "mean",
            "std",
            "variance",
            "min",
            "q1",
            "median",
            "q3",
            "max",
            "skewness",
            "kurtosis",
        ]
    ].round(6)


def save_distribution_figure(name, df):
    fig, axes = plt.subplots(4, 3, figsize=(14, 14))
    axes = axes.flatten()

    for axis, feature in zip(axes, FEATURES):
        axis.hist(
            df[feature], bins=30, color="#355070", edgecolor="white", linewidth=0.6
        )
        axis.set_title(feature)
        axis.grid(alpha=0.2)

    for axis in axes[len(FEATURES) :]:
        axis.remove()

    fig.suptitle(f"{name.title()} wine distributions", fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_correlation_figure(name, matrix, method):
    fig, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(FEATURES)))
    axis.set_xticklabels(FEATURES, rotation=45, ha="right")
    axis.set_yticks(range(len(FEATURES)))
    axis.set_yticklabels(FEATURES)
    axis.set_title(f"{name.title()} wine {method} correlation")

    for row in range(len(FEATURES)):
        for col in range(len(FEATURES)):
            axis.text(
                col,
                row,
                f"{matrix.iloc[row, col]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / f"{name}_{method}_correlation.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


def run_pca(df):
    scaled = StandardScaler().fit_transform(df[FEATURES])
    pca = PCA()
    pca.fit(scaled)

    components = [f"PC{i}" for i in range(1, len(FEATURES) + 1)]
    summary = pd.DataFrame(
        {
            "component": components,
            "eigenvalue": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": pca.explained_variance_ratio_.cumsum(),
        }
    )
    loadings = pd.DataFrame(pca.components_.T, index=FEATURES, columns=components)
    return summary.round(6), loadings.round(6)


def save_pca_figure(name, pca_summary):
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.bar(pca_summary["component"], pca_summary["eigenvalue"], color="#6d597a")
    axis.set_title(f"{name.title()} wine PCA eigenvalues")
    axis.set_xlabel("Principal component")
    axis.set_ylabel("Eigenvalue")
    axis.grid(axis="y", alpha=0.2)

    for index, value in enumerate(pca_summary["eigenvalue"]):
        axis.text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / f"{name}_pca_eigenvalues.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for name, df in load_datasets().items():
        build_summary(df).to_csv(OUTPUT_DIR / f"{name}_summary.csv")

        pearson = df[FEATURES].corr(method="pearson").round(6)
        spearman = df[FEATURES].corr(method="spearman").round(6)
        pearson.to_csv(OUTPUT_DIR / f"{name}_pearson_correlation.csv")
        spearman.to_csv(OUTPUT_DIR / f"{name}_spearman_correlation.csv")

        pca_summary, pca_loadings = run_pca(df)
        pca_summary.to_csv(OUTPUT_DIR / f"{name}_pca_summary.csv", index=False)
        pca_loadings.to_csv(OUTPUT_DIR / f"{name}_pca_loadings.csv")

        save_distribution_figure(name, df)
        save_correlation_figure(name, pearson, "pearson")
        save_correlation_figure(name, spearman, "spearman")
        save_pca_figure(name, pca_summary)


if __name__ == "__main__":
    main()
