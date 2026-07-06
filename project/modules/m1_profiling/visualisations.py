"""
MODULE 1 · FILE 4 — PROFILING VISUALISATIONS
============================================
Generates backend (headless) image files that make the profiling numbers
from File 2 visible. No pop-up windows — it writes PNGs a report or dashboard
can embed.

Four visuals (from the brief):
  1. Missing heatmap        -> missing_heatmap()
  2. Correlation heatmap    -> correlation_heatmap()
  3. Distributions          -> distribution_plots()
  4. Outlier distribution   -> outlier_plots()

Design principle: DATASET-AGNOSTIC. Works on whatever numeric columns exist;
degrades gracefully (returns None) when a plot doesn't apply — e.g. no numeric
columns, so no correlation heatmap.

Public entry point:  generate_all(df, output_dir) -> dict[str, str | None]
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend — required for backend generation
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
import pandas as pd  # noqa: E402

sns.set_theme(style="whitegrid")

# cap how many numeric columns we grid-plot, so huge datasets stay readable
MAX_PLOT_COLS = 12


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric columns only. Mixed-type text columns (e.g. '£44.99') are
    excluded until cleaning converts them — profiling shows data as it is."""
    return df.select_dtypes(include="number")


def missing_heatmap(df: pd.DataFrame, out_dir: Path) -> str | None:
    """Heatmap of where values are missing (rows x columns)."""
    if df.isna().sum().sum() == 0:
        return None  # nothing missing, no heatmap needed
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.isna(), cbar=False, cmap="viridis", yticklabels=False, ax=ax)
    ax.set_title("Missing-value map (yellow = missing)")
    ax.set_xlabel("Columns")
    path = out_dir / "missing_heatmap.png"
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)
    return str(path)


def correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> str | None:
    """Correlation heatmap across numeric columns."""
    num = _numeric(df)
    if num.shape[1] < 2:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(num.corr(numeric_only=True), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title("Correlation matrix (numeric columns)")
    path = out_dir / "correlation_heatmap.png"
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)
    return str(path)


def distribution_plots(df: pd.DataFrame, out_dir: Path) -> str | None:
    """A grid of histograms, one per numeric column."""
    num = _numeric(df).iloc[:, :MAX_PLOT_COLS]
    if num.shape[1] == 0:
        return None
    cols = num.columns
    n = len(cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = [axes] if n == 1 else axes.flatten()
    for i, col in enumerate(cols):
        sns.histplot(num[col].dropna(), kde=True, ax=axes[i])
        axes[i].set_title(f"Distribution: {col}")
    for j in range(n, len(axes)):  # hide any empty subplots
        axes[j].set_visible(False)
    path = out_dir / "distributions.png"
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)
    return str(path)


def outlier_plots(df: pd.DataFrame, out_dir: Path) -> str | None:
    """A grid of boxplots, one per numeric column — makes outliers visible."""
    num = _numeric(df).iloc[:, :MAX_PLOT_COLS]
    if num.shape[1] == 0:
        return None
    cols = num.columns
    n = len(cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = [axes] if n == 1 else axes.flatten()
    for i, col in enumerate(cols):
        sns.boxplot(x=num[col].dropna(), ax=axes[i])
        axes[i].set_title(f"Outliers: {col}")
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    path = out_dir / "outliers.png"
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)
    return str(path)


def generate_all(df: pd.DataFrame, output_dir: str | Path = "outputs/figures") -> dict:
    """Generate every applicable visual. Returns {name: path or None}."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "missing_heatmap": missing_heatmap(df, out_dir),
        "correlation_heatmap": correlation_heatmap(df, out_dir),
        "distributions": distribution_plots(df, out_dir),
        "outliers": outlier_plots(df, out_dir),
    }
    made = [k for k, v in results.items() if v]
    print(f"  ✔ generated {len(made)} figures in {out_dir}: {', '.join(made)}")
    return results