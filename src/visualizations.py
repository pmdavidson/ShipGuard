"""
All plotting routines for ShipGuard.  Each function saves a PNG and
returns the Figure so callers can optionally display it inline.

Plots produced:
  1. sensor_overview: multi-panel time-series per vessel with fault shading
  2. anomaly_timeline:   anomaly scores vs ground-truth fault labels
  3. confusion_matrix:   heat-map for the fault classifier
  4. feature_importance: top-N feature importance bar chart
  5. rul_scatter:        actual vs predicted RUL (scatter + residuals)
  6. eda_distributions:  KDE + box plots for each sensor by fault type
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# aesthetics
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.05)
FAULT_COLOURS = {0: "#2ecc71", 1: "#e74c3c", 2: "#e67e22", 3: "#9b59b6"}
FAULT_NAMES   = {0: "Normal", 1: "Cooling Failure",
                 2: "Fuel Degradation", 3: "Bearing Wear"}


def _save(fig: plt.Figure, path: Path, name: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    fpath = path / name
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    print(f"[viz] Saved → {fpath}")
    return fpath


"""
Apply a tick locator/formatter that auto-sizes to the visible date
range and never overlaps. AutoDateLocator caps the number of ticks;
ConciseDateFormatter drops redundant year/month repetition (e.g. shows
'31', '01' instead of '2025-03-31', '2025-04-01' colliding at a
month boundary).
"""
def _format_date_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


"""
Multi-panel time-series plot for one vessel.
Fault windows are shaded by fault type.
"""
def sensor_overview(df: pd.DataFrame,
                    vessel: str,
                    sensors: list[str] | None = None,
                    output_dir: Path = Path("reports/figures")) -> plt.Figure:
    if sensors is None:
        sensors = ["rpm", "coolant_temp", "egt", "fuel_lph", "vibration"]

    vdf = df[df["vessel_id"] == vessel].copy()
    vdf["timestamp"] = pd.to_datetime(vdf["timestamp"])
    vdf = vdf.set_index("timestamp").sort_index()

    n     = len(sensors)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    label_map  = {"rpm": "RPM", "coolant_temp": "Coolant Temp (°C)",
                  "egt": "Exhaust Gas Temp (°C)", "fuel_lph": "Fuel (L/h)",
                  "vibration": "Vibration (mm/s RMS)",
                  "oil_pressure": "Oil Pressure (bar)", "speed_kn": "Speed (kn)"}

    for ax, col in zip(axes, sensors):
        ax.plot(vdf.index, vdf[col], lw=0.8, color="#2c3e50", alpha=0.9)

        # shade fault windows
        fault_series = vdf["fault_label"]
        in_fault     = False
        start_idx    = None
        prev_type    = 0
        for ts, ftype in fault_series.items():
            if not in_fault and ftype != 0:
                in_fault   = True
                start_idx  = ts
                prev_type  = ftype
            elif in_fault and (ftype == 0 or ftype != prev_type):
                ax.axvspan(start_idx, ts,
                           color=FAULT_COLOURS[prev_type], alpha=0.25)
                in_fault  = ftype != 0
                start_idx = ts if in_fault else None
                prev_type = ftype
        if in_fault:
            ax.axvspan(start_idx, vdf.index[-1],
                       color=FAULT_COLOURS[prev_type], alpha=0.25)

        ax.set_ylabel(label_map.get(col, col), fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        if ax is not axes[-1]:
            ax.tick_params(axis="x", labelbottom=False)   # dates only on bottom panel

    # legend
    patches = [mpatches.Patch(color=FAULT_COLOURS[k], alpha=0.5, label=v)
               for k, v in FAULT_NAMES.items()]
    axes[0].legend(handles=patches, loc="upper right", fontsize=8, ncol=4)
    axes[0].set_title(f"Sensor Overview – {vessel}", fontsize=13, fontweight="bold")
    axes[-1].set_xlabel("Date", fontsize=9)
    _format_date_axis(axes[-1])
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()

    _save(fig, output_dir, f"sensor_overview_{vessel}.png")
    return fig


"""
Anomaly score (Isolation Forest) vs ground-truth fault events.
"""
def anomaly_timeline(timestamps: pd.Series,
                     anomaly_scores: np.ndarray,
                     fault_labels: np.ndarray,
                     vessel: str,
                     output_dir: Path = Path("reports/figures")) -> plt.Figure:
    ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
    labels = np.asarray(fault_labels)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6), sharex=True)

    # top: anomaly score (inverted so spikes = anomalies)
    ax1.plot(ts, -anomaly_scores, lw=0.7, color="#c0392b", alpha=0.85,
             label="Anomaly score (↑ = more anomalous)")
    ax1.set_ylabel("Anomaly Score", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.set_title(f"Isolation Forest Anomaly Detection – {vessel}",
                  fontsize=12, fontweight="bold")
    ax1.tick_params(axis="x", labelbottom=False)   # dates only on bottom panel

    # bottom: ground-truth fault label, drawn as contiguous coloured blocks.
    # NOTE: do NOT use ax.bar() with one bar per timestep here — with
    # thousands of 10-minute bars spread across weeks/months, each bar is
    # sub-pixel wide and the rasterizer drops most of them regardless of
    # height or color. axvspan over contiguous runs renders one solid
    # rectangle per run instead, which is resolution-independent.
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_ylabel("Fault Type", fontsize=9)
    ax2.set_xlabel("Date", fontsize=9)

    n = len(labels)
    i = 0
    while i < n:
        j = i
        while j < n and labels[j] == labels[i]:
            j += 1
        end_ts = ts.iloc[j] if j < n else ts.iloc[-1] + (ts.iloc[-1] - ts.iloc[-2])
        ax2.axvspan(ts.iloc[i], end_ts, color=FAULT_COLOURS[int(labels[i])],
                   alpha=0.85 if labels[i] != 0 else 0.3)
        i = j

    patches = [mpatches.Patch(color=FAULT_COLOURS[k], label=v)
               for k, v in FAULT_NAMES.items()]
    ax2.legend(handles=patches, fontsize=8, loc="upper right", ncol=4)

    _format_date_axis(ax2)
    fig.autofmt_xdate(rotation=0, ha="center")

    fig.tight_layout()
    _save(fig, output_dir, f"anomaly_timeline_{vessel}.png")
    return fig


def confusion_matrix_plot(cm: np.ndarray,
                          output_dir: Path = Path("reports/figures")) -> plt.Figure:
    labels = [FAULT_NAMES[i] for i in range(4)]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title("Fault Classifier – Confusion Matrix", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, output_dir, "confusion_matrix.png")
    return fig


def feature_importance_plot(importance: pd.Series,
                            top_n: int = 25,
                            title: str = "Feature Importance",
                            filename: str = "feature_importance.png",
                            output_dir: Path = Path("reports/figures")) -> plt.Figure:
    top = importance.head(top_n)
    fig, ax = plt.subplots(figsize=(9, top_n * 0.35 + 1.5))
    colours = sns.color_palette("viridis_r", len(top))
    ax.barh(top.index[::-1], top.values[::-1], color=colours[::-1])
    ax.set_xlabel("Importance Score", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    _save(fig, output_dir, filename)
    return fig


def rul_scatter(actuals: np.ndarray,
                predictions: np.ndarray,
                output_dir: Path = Path("reports/figures")) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # scatter: actual vs predicted
    lim = max(actuals.max(), predictions.max()) * 1.05
    ax1.scatter(actuals, predictions, alpha=0.15, s=8, color="#2980b9")
    ax1.plot([0, lim], [0, lim], "--", color="#e74c3c", lw=1.5, label="Perfect")
    ax1.set_xlim(0, lim); ax1.set_ylim(0, lim)
    ax1.set_xlabel("Actual RUL (timesteps)", fontsize=10)
    ax1.set_ylabel("Predicted RUL (timesteps)", fontsize=10)
    ax1.set_title("RUL Predictor – Actual vs Predicted", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)

    # residuals histogram
    residuals = predictions - actuals
    ax2.hist(residuals, bins=60, color="#8e44ad", edgecolor="white", linewidth=0.3)
    ax2.axvline(0, color="#e74c3c", lw=1.5, linestyle="--")
    ax2.set_xlabel("Residual (predicted − actual)", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.set_title("Residual Distribution", fontsize=11, fontweight="bold")

    fig.tight_layout()
    _save(fig, output_dir, "rul_scatter.png")
    return fig


"""KDE distributions of each sensor coloured by fault type."""
def eda_distributions(df: pd.DataFrame,
                      sensors: list[str] | None = None,
                      output_dir: Path = Path("reports/figures")) -> plt.Figure:
    if sensors is None:
        sensors = ["rpm", "coolant_temp", "egt", "fuel_lph", "vibration", "speed_kn"]

    label_map = {"rpm": "RPM", "coolant_temp": "Coolant Temp (°C)",
                 "egt": "Exhaust Gas Temp (°C)", "fuel_lph": "Fuel (L/h)",
                 "vibration": "Vibration (mm/s RMS)", "speed_kn": "Speed (kn)"}

    ncols = 3
    nrows = int(np.ceil(len(sensors) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4 * nrows))
    axes = axes.flatten()

    palette = {k: FAULT_COLOURS[k] for k in FAULT_NAMES}

    for ax, col in zip(axes, sensors):
        for ftype, grp in df.groupby("fault_label"):
            sns.kdeplot(grp[col].dropna(), ax=ax,
                        label=FAULT_NAMES[int(ftype)],
                        color=palette[int(ftype)], lw=1.8, alpha=0.85)
        ax.set_xlabel(label_map.get(col, col), fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(label_map.get(col, col), fontsize=10)
        ax.legend(fontsize=7)

    for ax in axes[len(sensors):]:
        ax.set_visible(False)

    fig.suptitle("Sensor Distributions by Fault Type", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, output_dir, "eda_distributions.png")
    return fig