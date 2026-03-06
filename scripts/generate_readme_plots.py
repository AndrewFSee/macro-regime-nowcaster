"""Generate publication-quality plots for the README.

Run:
    python scripts/generate_readme_plots.py

Produces PNG files in docs/images/ for embedding in README.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

from dotenv import load_dotenv
load_dotenv()

OUT = _ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

# NBER recession dates
NBER = [
    ("1980-01-01", "1980-07-31"),
    ("1981-07-01", "1982-11-30"),
    ("1990-07-01", "1991-03-31"),
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]

DARK_BG = "#0e1117"
DARK_CARD = "#1a1d23"
ACCENT_GREEN = "#2ecc71"
ACCENT_RED = "#e74c3c"
ACCENT_BLUE = "#3498db"
ACCENT_PURPLE = "#9b59b6"
ACCENT_ORANGE = "#e67e22"
ACCENT_CYAN = "#1abc9c"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2d35"


def _style_dark(fig, axes):
    """Apply dark theme to figure and axes."""
    fig.patch.set_facecolor(DARK_BG)
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(DARK_CARD)
        ax.tick_params(colors=TEXT_COLOR, which="both")
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(True, alpha=0.15, color=GRID_COLOR)


def _add_nber(ax, ymin=0, ymax=1):
    """Add NBER recession shading."""
    for start, end in NBER:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                    alpha=0.18, color="#888888", zorder=0)


def run_nowcast():
    """Run the full pipeline and return nowcaster + result."""
    import os
    from src.data.fred_client import FREDClient
    from src.data.data_pipeline import DataPipeline
    from src.models.nowcaster import Nowcaster

    api_key = os.getenv("FRED_API_KEY", "")
    client = FREDClient(api_key=api_key)
    pipeline = DataPipeline(fred_client=client, start_date="2000-01-01")
    nowcaster = Nowcaster(pipeline=pipeline, n_factors=4, n_regimes=2, use_ensemble=True)
    result = nowcaster.run()
    return nowcaster, result


def plot_regime_probabilities(nowcaster, result):
    """Ensemble recession probability with NBER shading — the hero chart."""
    regime_probs = nowcaster.get_ensemble_probabilities()
    rec = regime_probs["recession"]
    exp = regime_probs["expansion"]

    fig, ax = plt.subplots(figsize=(14, 5))
    _style_dark(fig, ax)

    # Stacked area fill
    ax.fill_between(rec.index, 0, rec.values, alpha=0.7, color=ACCENT_RED,
                    label="P(Recession)")
    ax.fill_between(exp.index, rec.values, 1.0, alpha=0.5, color=ACCENT_GREEN,
                    label="P(Expansion)")

    # NBER shading
    _add_nber(ax)

    # Threshold line
    ax.axhline(0.5, color="#ffffff", linewidth=0.8, linestyle="--", alpha=0.4)

    ax.set_ylabel("Probability", fontsize=12, color=TEXT_COLOR)
    ax.set_title("Ensemble Recession Probability vs NBER Recessions",
                 fontsize=15, fontweight="bold", color=TEXT_COLOR, pad=12)
    ax.set_ylim(0, 1)
    ax.set_xlim(rec.index.min(), rec.index.max())
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper right", framealpha=0.6, facecolor=DARK_CARD,
              edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    fig.tight_layout()
    fig.savefig(OUT / "regime_probabilities.png", dpi=180, facecolor=DARK_BG,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ regime_probabilities.png")


def plot_latent_factors(nowcaster, result):
    """4-panel latent factor chart."""
    factors = nowcaster._last_factors

    factor_colors = [ACCENT_CYAN, ACCENT_BLUE, ACCENT_ORANGE, ACCENT_PURPLE]
    titles = ["Real Activity", "Labor Market", "Inflation", "Financial Stress"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True)
    _style_dark(fig, axes.flat)

    for i, (ax, col) in enumerate(zip(axes.flat, factors.columns)):
        series = factors[col].copy()
        # Display clipping
        clipped = series.clip(-3.0, 3.0)
        # Trim leading warm-up
        valid = clipped[clipped.abs() > 0.05]
        if len(valid) == 0:
            continue

        color = factor_colors[i % len(factor_colors)]
        ax.plot(valid.index, valid.values, color=color, linewidth=0.9, alpha=0.9)
        ax.set_title(titles[i] if i < len(titles) else col,
                     fontsize=11, fontweight="bold", color=TEXT_COLOR)
        ax.axhline(0, color=TEXT_COLOR, linewidth=0.5, linestyle="--", alpha=0.3)
        ax.set_ylabel("z-score", fontsize=9, color=TEXT_COLOR)
        ax.set_ylim(-3.5, 3.5)
        ax.set_xlim(valid.index.min(), valid.index.max())
        _add_nber(ax)
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("Latent Factors (Dynamic Factor Model with Varimax Rotation)",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "latent_factors.png", dpi=180, facecolor=DARK_BG,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ latent_factors.png")


def plot_regime_timeline(nowcaster, result):
    """Historical regime classification heatmap strip."""
    regime_probs = nowcaster.get_ensemble_probabilities()
    rec = regime_probs["recession"]

    fig, ax = plt.subplots(figsize=(14, 2.2))
    _style_dark(fig, ax)

    # Create a continuous color strip
    dates = rec.index
    colors_arr = np.array([rec.values, np.zeros_like(rec.values)])

    # Custom colormap: green (expansion) → red (recession)
    cmap = LinearSegmentedColormap.from_list(
        "regime", [(0, ACCENT_GREEN), (0.5, "#f39c12"), (1.0, ACCENT_RED)]
    )

    for j in range(len(dates) - 1):
        ax.axvspan(dates[j], dates[j + 1], color=cmap(rec.values[j]), alpha=0.85)

    _add_nber(ax)

    ax.set_yticks([])
    ax.set_title("Historical Regime Classification",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=8)
    ax.set_xlim(dates.min(), dates.max())
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("P(Recession)", color=TEXT_COLOR, fontsize=10)
    cbar.ax.tick_params(colors=TEXT_COLOR)

    fig.tight_layout()
    fig.savefig(OUT / "regime_timeline.png", dpi=180, facecolor=DARK_BG,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ regime_timeline.png")


def plot_ensemble_breakdown(nowcaster, result):
    """Signal comparison: RSM vs Probit vs CFNAI over time."""
    # Re-extract the components from the ensemble
    factors = nowcaster._last_factors
    idx = factors.index

    # RSM component
    try:
        rsm_probs = nowcaster._rsm.get_recession_probability()
        if isinstance(rsm_probs, pd.Series):
            rsm_ts = rsm_probs.reindex(idx).ffill().bfill().fillna(0.5)
        else:
            rsm_ts = pd.Series(rsm_probs, index=idx).fillna(0.5)
    except Exception:
        rsm_ts = pd.Series(0.5, index=idx)

    # Probit component
    try:
        probit_features = nowcaster._build_probit_features(factors, nowcaster._last_panel)
        all_proba = nowcaster._probit.predict_proba(probit_features)
        probit_ts = pd.Series(all_proba, index=probit_features.index).reindex(idx).ffill().bfill().fillna(0.5)
    except Exception:
        probit_ts = pd.Series(0.5, index=idx)

    # Ensemble
    ensemble_ts = nowcaster._ensemble_recession_ts

    fig, ax = plt.subplots(figsize=(14, 5))
    _style_dark(fig, ax)

    ax.plot(rsm_ts.index, rsm_ts.values, color=ACCENT_PURPLE, linewidth=1.2,
            alpha=0.8, label="RSM (Markov-Switching)")
    ax.plot(probit_ts.index, probit_ts.values, color=ACCENT_BLUE, linewidth=1.2,
            alpha=0.8, label="Probit (Supervised)")
    ax.plot(ensemble_ts.index, ensemble_ts.values, color="#ffffff", linewidth=2.0,
            alpha=0.95, label="Ensemble (Weighted)")

    _add_nber(ax)
    ax.axhline(0.5, color="#ffffff", linewidth=0.6, linestyle="--", alpha=0.3)

    ax.set_ylabel("P(Recession)", fontsize=12, color=TEXT_COLOR)
    ax.set_title("Ensemble Signal Decomposition: RSM vs Probit vs Ensemble",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=12)
    ax.set_ylim(0, 1)
    ax.set_xlim(ensemble_ts.index.min(), ensemble_ts.index.max())
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper right", framealpha=0.6, facecolor=DARK_CARD,
              edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    fig.tight_layout()
    fig.savefig(OUT / "ensemble_breakdown.png", dpi=180, facecolor=DARK_BG,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ ensemble_breakdown.png")


def plot_dashboard_hero(result):
    """A metrics card simulating the dashboard banner."""
    fig, ax = plt.subplots(figsize=(14, 3))
    _style_dark(fig, ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    regime = result.current_regime
    color = ACCENT_GREEN if regime == "expansion" else ACCENT_RED

    # Banner background
    banner = Rectangle((0.1, 0.3), 9.8, 2.4, linewidth=0, facecolor=color,
                        alpha=0.85, zorder=2, transform=ax.transData)
    banner.set_clip_on(False)
    ax.add_patch(banner)

    ax.text(5, 2.15, regime.upper(), fontsize=28, fontweight="bold",
            color="white", ha="center", va="center", zorder=3)
    ax.text(5, 1.3,
            f"GDP Nowcast: {result.gdp_nowcast:.2f}% ann.  |  "
            f"Recession Probability: {result.recession_probability:.1%}  |  "
            f"Ensemble: RSM + Probit + CFNAI",
            fontsize=11, color=(1, 1, 1, 0.92), ha="center", va="center",
            zorder=3)

    # Metrics strip
    metrics = [
        ("GDP Nowcast", f"{result.gdp_nowcast:.2f}%"),
        ("Recession Prob", f"{result.recession_probability:.1%}"),
        ("Current Regime", regime.title()),
        ("Model Signals", "4 active"),
    ]
    for i, (label, val) in enumerate(metrics):
        x = 1.2 + i * 2.5
        ax.text(x, 0.12, label, fontsize=8, color="#888888", ha="center",
                va="center", zorder=3)
        ax.text(x, -0.2, val, fontsize=14, fontweight="bold",
                color=TEXT_COLOR, ha="center", va="center", zorder=3)

    fig.tight_layout()
    fig.savefig(OUT / "dashboard_banner.png", dpi=180, facecolor=DARK_BG,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ dashboard_banner.png")


def plot_architecture():
    """System architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    _style_dark(fig, ax)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def _box(x, y, w, h, text, color=ACCENT_BLUE, fontsize=10):
        rect = Rectangle((x, y), w, h, linewidth=1.5, edgecolor=color,
                         facecolor=color, alpha=0.2, zorder=2)
        ax.add_patch(rect)
        rect_border = Rectangle((x, y), w, h, linewidth=1.5, edgecolor=color,
                                facecolor="none", zorder=3)
        ax.add_patch(rect_border)
        ax.text(x + w/2, y + h/2, text, fontsize=fontsize, color=TEXT_COLOR,
                ha="center", va="center", fontweight="bold", zorder=4)

    def _arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=TEXT_COLOR,
                                    lw=1.5, alpha=0.7))

    # Title
    ax.text(6, 9.5, "System Architecture", fontsize=18, fontweight="bold",
            color=TEXT_COLOR, ha="center", va="center")

    # Data layer
    _box(3.5, 8.2, 5, 0.8, "FRED API  ·  60+ Macro Series", ACCENT_CYAN, 11)
    _arrow(6, 8.2, 6, 7.6)

    # Pipeline
    _box(3.5, 6.8, 5, 0.8, "DataPipeline\nTransform · Align · Standardize", ACCENT_BLUE, 9)
    _arrow(6, 6.8, 6, 6.2)

    # DFM
    _box(3.5, 5.4, 5, 0.8, "Dynamic Factor Model (EM + Kalman)\n4 Latent Factors · Varimax Rotation", ACCENT_PURPLE, 9)
    _arrow(6, 5.4, 6, 4.8)

    # Ensemble
    _box(1, 3.5, 3, 0.8, "Markov-Switching\nRSM (wt: 0.25)", ACCENT_PURPLE, 8)
    _box(4.5, 3.5, 3, 0.8, "Recession Probit\n(wt: 0.50)", ACCENT_BLUE, 8)
    _box(8, 3.5, 3, 0.8, "CFNAI Signal\n(wt: 0.25)", ACCENT_ORANGE, 8)

    _arrow(4.5, 5.4, 2.5, 4.3)
    _arrow(6, 5.4, 6, 4.3)
    _arrow(7.5, 5.4, 9.5, 4.3)

    # Ensemble merge
    _arrow(2.5, 3.5, 5, 2.8)
    _arrow(6, 3.5, 6, 2.8)
    _arrow(9.5, 3.5, 7, 2.8)

    _box(4, 2, 4, 0.8, "Ensemble P(Recession)\nGDP Nowcast (OLS-calibrated)", ACCENT_GREEN, 9)

    # Output layer
    _arrow(4, 2, 2, 1.2)
    _arrow(6, 2, 6, 1.2)
    _arrow(8, 2, 10, 1.2)

    _box(0.5, 0.4, 3, 0.8, "Regime-Conditional\nAsset Allocation", ACCENT_GREEN, 8)
    _box(4.2, 0.4, 3.5, 0.8, "Streamlit\nDashboard", ACCENT_CYAN, 8)
    _box(8.5, 0.4, 3, 0.8, "LLM Narrative\n+ Fed Scraper", ACCENT_ORANGE, 8)

    fig.tight_layout()
    fig.savefig(OUT / "architecture.png", dpi=180, facecolor=DARK_BG,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ architecture.png")


if __name__ == "__main__":
    print("Generating README plots...")
    print()

    # Architecture (no data needed)
    plot_architecture()

    # Data-driven plots
    print("  Running nowcast pipeline...")
    nowcaster, result = run_nowcast()
    print(f"  GDP: {result.gdp_nowcast:.2f}%, Regime: {result.current_regime}")
    print()

    plot_regime_probabilities(nowcaster, result)
    plot_latent_factors(nowcaster, result)
    plot_regime_timeline(nowcaster, result)
    plot_ensemble_breakdown(nowcaster, result)
    plot_dashboard_hero(result)

    print()
    print(f"All plots saved to {OUT}/")
