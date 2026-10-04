"""Grid fixation task: paper Figures 4-9 and Table 1, plus the numbers quoted with them.

Reads Data/1_accuracy/. Run from anywhere:  python Code/analyse_accuracy.py [--show]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

from utils import (data_path, inverse_weights, pooled_median, save_figure, ue_percentiles,
                   unit_to_az_el)

CELL = 3.0  # FOV cell size [deg]; cells are centred on 0
MIN_IN_CELL = 5  # small-stratum safeguard of the inverse-density weighting
CENTRAL = (15, 15)  # central window [deg], tiles exactly at CELL
PCTS = [10, 25, 50, 75, 90]
METRICS = {"accuracy_centroid_deg": ("Accuracy", 1.0), "precision_rms_s2s_deg": ("Precision (RMS-S2S)", 0.2)}
X, Y = "stimulus_head_x_deg", "stimulus_head_y_deg"

FIG4 = ("p038_grid_1_grid1", "left", range(11, 25))  # recording, eye, fixations shown
FIG5 = "p042_grid_1_grid2"


def mad_mean(arr, threshold=3.0):
    """Mean after dropping values beyond threshold * scaled MAD from the median."""
    arr = np.asarray(arr)
    median = np.median(arr)
    scaled = np.abs(arr - median) / (np.median(np.abs(arr - median)) * 1.4826)
    return np.mean(arr[~(scaled > threshold)])


def cells(df, col):
    return np.round(df[col] / CELL) * CELL


# ---------------------------------------------------------------------------- numbers


def print_counts(df):
    # grid labels are not layouts (one 5x5 recording is 'grid1'); the 7x11 grid reaches +-15 deg
    per = df.groupby("participant").stimulus_monitor_x_deg.agg(
        lambda x: "7x11" if x.abs().max() > 10 else "5x5")
    print(f"{len(df)} fixations (both eyes) from {df.participant.nunique()} participants: "
          f"{(per == '5x5').sum()} on the 5x5 grid, {(per == '7x11').sum()} on the 7x11 grid; "
          f"{df.groupby('participant').participant_in_training_set.first().sum()} participants "
          f"had other sessions in the model's training set")
    print(f"{df.is_fixation_outlier.sum()} outlier fixations")
    for col, unit in (("head_movement_mm", "mm"), ("head_movement_deg", "deg")):
        print(f"Head {'motion' if unit == 'mm' else 'rotation'} per fixation: median {df[col].median():.2f} {unit}, "
              f"95th percentile {df[col].quantile(.95):.2f} {unit}")


def table1(edata, n_all):
    """Table 1: pooled medians, unweighted and inverse-density-weighted. Also returns the U|E
    tables behind Figs 8 and 9."""
    rows, ue = {}, {}
    for metric, (name, _) in METRICS.items():
        whole = inverse_weights(edata, metric, CELL, MIN_IN_CELL)
        central = inverse_weights(edata, metric, CELL, MIN_IN_CELL, CENTRAL)
        rows[name] = [pooled_median(edata, metric), pooled_median(whole, metric, "weight"),
                      pooled_median(central, metric, "weight")]
        print(f"\n{name}: {len(whole)} fixations in the whole-FOV weighting "
              f"({n_all - len(whole)} of {n_all} removed: outliers and cells under {MIN_IN_CELL}), "
              f"{len(central)} in the central {CENTRAL[0]}x{CENTRAL[1]} deg")
        ue[metric] = {
            "unweighted": ue_percentiles(edata, "participant", metric, PCTS),
            "whole": ue_percentiles(whole, "participant", metric, PCTS, "weight"),
            "central": ue_percentiles(central, "participant", metric, PCTS, "weight"),
        }
        for label, t in ue[metric].items():
            print(f"  U|E, {label}:\n{t.round(3).to_string()}")
    t1 = pd.DataFrame(rows, index=["Unweighted, all fixations", "Weighted, all FOV",
                                   "Weighted, central FOV"]).T
    print(f"\nTable 1: median accuracy and precision [deg]\n{t1.round(4).to_string()}")
    print(t1.round(2).to_string())
    return ue


# ---------------------------------------------------------------------------- figures


def fig4():
    """Timeline of gaze (red) and stimulus (black) in the head frame, one eye, whole fixations
    with blinks left in. Angles of the negated vectors, as in the original figure."""
    rec, eye, fixations = FIG4
    d = pd.read_csv(data_path("1_accuracy", f"{rec}_{eye}.csv"))
    d = d[d.fixation.isin(fixations)]
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    lims = [[np.inf, -np.inf], [np.inf, -np.inf]]
    for num, f in d.groupby("fixation"):
        gaze = unit_to_az_el(-f[["visual_axis_x", "visual_axis_y", "visual_axis_z"]].to_numpy())
        stim = unit_to_az_el(-f[["stimulus_dir_x", "stimulus_dir_y", "stimulus_dir_z"]].to_numpy())
        for i, ax in enumerate(axs):
            ax.plot(f.timestamp_sec, gaze[i], "r-")
            ax.plot(f.timestamp_sec, stim[i], "k-")
            ax.text(f.timestamp_sec.iloc[0], stim[i][0], num, ha="right", va="center")
            lims[i] = [min(lims[i][0], stim[i].min()), max(lims[i][1], stim[i].max())]
    for ax, (lo, hi) in zip(axs, lims):
        ax.set_ylim(lo - 2, hi + 2)  # blink artefacts run off the scale, as in the paper
        ax.grid(True)
    axs[0].set_ylabel("Azimuth (deg)")
    axs[1].set_ylabel("Elevation (deg)")
    axs[1].set_xlabel("Time (sec)")
    save_figure(fig, "fig04_timeline")


def fig5(df):
    """Optical (top) and visual (bottom) axis per frame around each grid target, both eyes;
    analysis-window frames without blinks, as the accuracy numbers use. Outlier targets: x."""
    pid, rec, grid = FIG5.split("_", 1)[0], "_".join(FIG5.split("_")[1:3]), FIG5.split("_")[-1]
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    for col, eye in enumerate(["left", "right"]):
        d = pd.read_csv(data_path("1_accuracy", f"{FIG5}_{eye}.csv"))
        d = d[d.in_analysis_window & ~d.blink]
        fx = df[(df.participant == pid) & (df.recording == rec) & (df.grid == grid) & (df.eye == eye)]
        outliers = set(fx.fixation[fx.is_fixation_outlier == 1])
        for num, f in d.groupby("fixation"):
            sx, sy = f.stimulus_monitor_x_deg.iloc[0], f.stimulus_monitor_y_deg.iloc[0]
            for ax in axs[:, col]:
                ax.plot(sx, sy, "xk" if num in outliers else "ok")
                ax.annotate(num, xy=(sx, sy), xytext=(sx + 0.2, sy + 0.2))
            axs[0, col].plot(sx + f.optical_deviation_x_deg, sy + f.optical_deviation_y_deg, ".",
                             color="#ff4d00", alpha=0.3, markersize=2)
            axs[1, col].plot(sx + mad_mean(f.visual_deviation_x_deg), sy + mad_mean(f.visual_deviation_y_deg),
                             "o", color="#1f77b4", markersize=2)
            axs[1, col].plot(sx + f.visual_deviation_x_deg, sy + f.visual_deviation_y_deg, ".",
                             color="#2ca02c", alpha=0.3, markersize=2)
        axs[0, col].set_title(f"{eye} eye, kappa {d.kappa_x_deg.iloc[0]:.2f} by {d.kappa_y_deg.iloc[0]:.2f} deg")
        print(f"Fig 5 {eye} eye kappa: {d.kappa_x_deg.iloc[0]:.2f} by {d.kappa_y_deg.iloc[0]:.2f} deg")
    ticks = np.arange(-21, 22, 3)
    for ax in axs.flat:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlim(-21, 21)
        ax.set_ylim(-11, 11)
        ax.grid()
    for ax in axs[:, 0]:
        ax.set_ylabel("Eccentricity vertical (deg)")
    for ax in axs[1]:
        ax.set_xlabel("Eccentricity horizontal (deg)")
    fig.tight_layout()
    save_figure(fig, "fig05_grid")


def fig6(edata):
    """Fixation positions in the head frame, and fixation counts per cell."""
    fig, axs = plt.subplots(1, 2, figsize=(18, 7))
    bx, by = cells(edata, X), cells(edata, Y)
    for cx, cy in set(zip(bx, by)):
        axs[0].add_patch(Rectangle((cx - CELL / 2, cy - CELL / 2), CELL, CELL,
                                   edgecolor="gray", facecolor="none", linewidth=0.8))
    axs[0].plot(edata[X], edata[Y], "kx", markersize=3, alpha=0.3)
    axs[0].set_xlim(bx.min() - CELL, bx.max() + CELL)
    axs[0].set_ylim(by.min() - CELL, by.max() + CELL)
    axs[0].set_aspect("equal")

    counts = pd.DataFrame({X: bx, Y: by}).groupby([Y, X]).size().unstack(fill_value=np.nan)
    sns.heatmap(counts, annot=True, fmt=".0f", square=True, cmap="Blues", vmin=0, vmax=200, ax=axs[1])
    axs[1].invert_yaxis()
    for ax in axs:
        ax.set_xlabel("X (deg)")
        ax.set_ylabel("Y (deg)")
    fig.tight_layout()
    save_figure(fig, "fig06_fixation_distribution")


def fig7(edata):
    """Median accuracy (top) and precision (bottom) per cell."""
    fig, axs = plt.subplots(2, 1, figsize=(11, 14))
    for ax, (metric, (name, vmax)) in zip(axs, METRICS.items()):
        grid = pd.DataFrame({X: cells(edata, X), Y: cells(edata, Y), metric: edata[metric]}).pivot_table(
            index=Y, columns=X, values=metric, aggfunc="median")
        sns.heatmap(grid, annot=True, fmt=".2f", square=True, cmap="Blues", vmin=0, vmax=vmax, ax=ax)
        ax.invert_yaxis()
        ax.set_title(name)
        ax.set_xlabel("X (deg)")
        ax.set_ylabel("Y (deg)")
    fig.tight_layout()
    save_figure(fig, "fig07_quality_by_cell")


def fig_ue(ue, region, name):
    """U|E heatmaps for one region: accuracy left, precision right."""
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.2))
    for ax, (metric, (_, vmax)) in zip(axs, METRICS.items()):
        sns.heatmap(ue[metric][region], annot=True, fmt=".2f", square=True, cmap="Blues",
                    vmin=0, vmax=vmax, ax=ax)
    fig.tight_layout()
    save_figure(fig, name)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--show", action="store_true", help="Show the figures as well as saving them")
    args = p.parse_args()

    df = pd.read_csv(data_path("1_accuracy", "fixations.csv"))
    edata = df[df.is_fixation_outlier == 0]

    print_counts(df)
    ue = table1(edata, len(df))

    fig4()
    fig5(df)
    fig6(edata)
    fig7(edata)
    fig_ue(ue, "whole", "fig08_ue_whole_fov")
    fig_ue(ue, "central", "fig09_ue_central_fov")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
