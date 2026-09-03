"""
Generate report-quality charts from experiment results.

Outputs PNG + SVG to results/charts/:
  1. f1_comparison.png         - per-batch F1 for A vs B vs C
  2. drift_timeline.png        - detection timeline (Config A)
  3. detector_metrics.png      - precision / recall / F1 bar
  4. promotion_decisions.png   - promotions vs rejections
  5. avg_f1_under_drift.png    - avg F1 on drifted batches bar
  6. sensitivity_analysis.png  - F1/recall vs drift magnitude
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR = Path("results")
CHARTS_DIR  = RESULTS_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

COLORS = {"A": "#4f72ea", "B": "#ef4444", "C": "#f59e0b"}
LABELS = {
    "A": "Config A - Full System",
    "B": "Config B - Naive Retrain",
    "C": "Config C - Static",
}
DRIFT_START = 5   # matches run_experiments.py DRIFT_START


def _save(fig, name: str):
    fig.savefig(str(CHARTS_DIR / f"{name}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(CHARTS_DIR / f"{name}.svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: results/charts/{name}.png")


# -- Chart 1: per-batch F1 comparison -----------------------------------------

def plot_f1_comparison(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for config, grp in df.groupby("config"):
        grp = grp.sort_values("batch")
        ax.plot(grp["batch"], grp["f1"],
                label=LABELS.get(config, config),
                color=COLORS.get(config, "gray"),
                linewidth=2, marker="o", markersize=5)

    ax.axvline(x=DRIFT_START - 0.5, color="red", linestyle="--",
               linewidth=1.5, label="Drift onset")
    ax.axvspan(DRIFT_START - 0.5, df["batch"].max() + 0.5,
               alpha=0.06, color="red")
    ax.set_xlabel("Batch", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Per-Batch F1: Full System vs Naive Retrain vs Static",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    _save(fig, "f1_comparison")


# -- Chart 2: drift detection timeline ----------------------------------------

def plot_drift_timeline(df: pd.DataFrame):
    a = df[df["config"] == "A"].sort_values("batch")
    if a.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.scatter(a["batch"], a["detected_drift"].astype(int),
               color="#4f72ea", s=80, zorder=3, label="Drift Detected")
    ax.axvspan(DRIFT_START - 0.5, a["batch"].max() + 0.5,
               alpha=0.1, color="red", label="Ground-truth drift region")
    ax.axvline(x=DRIFT_START - 0.5, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Batch", fontsize=12)
    ax.set_ylabel("Drift Signal", fontsize=12)
    ax.set_title("Drift Detection Timeline (Config A - Full System)",
                 fontsize=13, fontweight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No drift", "Drift detected"])
    ax.set_ylim(-0.3, 1.5)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.grid(axis="x", alpha=0.3)
    _save(fig, "drift_timeline")


# -- Chart 3: detector metrics bar --------------------------------------------

def plot_detector_metrics(summary: list):
    a = next(s for s in summary if s["config"] == "A")
    metrics = ["Precision", "Recall", "F1"]
    values  = [a["detector_precision"] or 0,
               a["detector_recall"]    or 0,
               a["detector_f1"]        or 0]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(metrics, values,
                  color=["#4f72ea", "#22c55e", "#f59e0b"],
                  width=0.5, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Drift Detector Performance\n(Config A vs ground-truth labels)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, "detector_metrics")


# -- Chart 4: promotions vs rejections ----------------------------------------

def plot_promotion_decisions(summary: list):
    configs    = [s["config"] for s in summary]
    promotions = [s["total_promotions"] for s in summary]
    rejections = [s["total_rejections"] for s in summary]
    x = np.arange(len(configs))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, promotions, w, label="Promotions",  color="#22c55e", edgecolor="white")
    ax.bar(x + w/2, rejections, w, label="Rejections",  color="#ef4444", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(c, c) for c in configs], fontsize=9)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Model Promotions vs Validation Gate Rejections",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, "promotion_decisions")


# -- Chart 5: avg F1 under drift bar ------------------------------------------

def plot_avg_f1_bar(summary: list):
    configs = [s["config"] for s in summary]
    f1s     = [s["avg_f1_post_drift"] or 0 for s in summary]
    colors  = [COLORS.get(c, "gray") for c in configs]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([LABELS.get(c, c) for c in configs], f1s,
                  color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    top = max(f1s) * 1.4 + 0.02 if max(f1s) > 0 else 0.3
    ax.set_ylim(0, top)
    ax.set_ylabel("Average F1 (drifted batches)", fontsize=12)
    ax.set_title("Model Quality Under Drift\n(Higher = better)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, "avg_f1_under_drift")


# -- Chart 6: sensitivity analysis --------------------------------------------

def plot_sensitivity(sens_df: pd.DataFrame):
    """
    Shows how detector precision/recall/F1 changes with drift magnitude.
    A graceful degradation curve is more defensible than a flat-1.0 line.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    for metric, color, marker in [
        ("detector_precision", "#4f72ea", "o"),
        ("detector_recall",    "#22c55e", "s"),
        ("detector_f1",        "#f59e0b", "^"),
    ]:
        if metric in sens_df.columns:
            vals = sens_df[metric].fillna(0)
            ax.plot(sens_df["drift_level"], vals,
                    label=metric.replace("detector_", "").capitalize(),
                    color=color, linewidth=2,
                    marker=marker, markersize=8)
            for x, y in zip(sens_df["drift_level"], vals):
                ax.annotate(f"{y:.2f}", (x, y),
                            textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=9)

    ax.set_xlabel("Drift Magnitude", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Detector Sensitivity: Performance vs Drift Magnitude",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.25)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, "sensitivity_analysis")


# -- Main ---------------------------------------------------------------------

def main():
    csv_path = RESULTS_DIR / "experiment_results.csv"
    if not csv_path.exists():
        print("ERROR: Run scripts/run_experiments.py first.")
        return

    df = pd.read_csv(csv_path)
    with open(RESULTS_DIR / "experiment_summary.json") as f:
        summary = json.load(f)

    sens_path = RESULTS_DIR / "sensitivity_analysis.csv"
    sens_df = pd.read_csv(sens_path) if sens_path.exists() else None

    print(f"Generating charts -> {CHARTS_DIR}/")
    plot_f1_comparison(df)
    plot_drift_timeline(df)
    plot_detector_metrics(summary)
    plot_promotion_decisions(summary)
    plot_avg_f1_bar(summary)
    if sens_df is not None:
        plot_sensitivity(sens_df)

    print("\nAll charts saved:")
    for f in sorted(CHARTS_DIR.glob("*.png")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
