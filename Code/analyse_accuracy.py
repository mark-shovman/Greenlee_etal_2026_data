"""Grid fixation task: paper Figures 7-12 and Table 1, plus the numbers quoted with them.

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
MIN_SAMPLE_FRAC = 0.3  # the validity threshold the data were exported with (is_fixation_valid)
CENTRAL = (15, 15)  # central window [deg], tiles exactly at CELL
PCTS = [10, 25, 50, 75, 90]
METRICS = {"accuracy_centroid_deg": ("Accuracy", 1.0), "precision_rms_s2s_deg": ("Precision (RMS-S2S)", 0.2)}
TABLE1 = {**METRICS, "precision_std_deg": ("Precision (SD)", None)}  # Table 1 only; figures use METRICS
X, Y = "stimulus_head_x_deg", "stimulus_head_y_deg"

FIG7 = ("p038_grid_1_grid1", range(11, 25))  # recording, fixations shown
FIG8 = "p042_grid_1_grid2"


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
    low = {c: (df[c] < MIN_SAMPLE_FRAC).sum() for c in ("frac_tracked", "frac_head", "frac_unannotated")}
    print(f"{(df.is_fixation_valid == 0).sum()} invalid fixations, under {MIN_SAMPLE_FRAC:.0%} of the expected "
          f"samples: {low['frac_tracked']} tracked, {low['frac_head']} with head pose, "
          f"{low['frac_unannotated']} left after the blink annotation")
    for col, unit in (("head_movement_mm", "mm"), ("head_movement_deg", "deg")):
        print(f"Head {'motion' if unit == 'mm' else 'rotation'} per fixation: median {df[col].median():.2f} {unit}, "
              f"95th percentile {df[col].quantile(.95):.2f} {unit}")


def table1(edata, n_all):
    """Table 1: pooled medians, unweighted and inverse-density-weighted. Also returns the U|E
    tables behind Figs 11 and 12."""
    rows, ue = {}, {}
    for metric, (name, _) in TABLE1.items():
        whole = inverse_weights(edata, metric, CELL, MIN_IN_CELL)
        central = inverse_weights(edata, metric, CELL, MIN_IN_CELL, CENTRAL)
        rows[name] = [pooled_median(edata, metric), pooled_median(whole, metric, "weight"),
                      pooled_median(central, metric, "weight")]
        print(f"\n{name}: {len(whole)} fixations in the whole-FOV weighting "
              f"({n_all - len(whole)} of {n_all} removed: invalid and cells under {MIN_IN_CELL}), "
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


def table1_by_training(edata):
    """Table 1 with each cell as `all (trained/untrained)`: participants with and without other
    sessions in the model's training set. The groups keep the whole cohort's weights."""
    groups = {"all": None, "trained": True, "untrained": False}
    for label, flag in groups.items():
        g = edata if flag is None else edata[edata.participant_in_training_set == flag]
        print(f"{label}: {len(g)} fixations from {g.participant.nunique()} participants")
    rows = {}
    for metric, (name, _) in TABLE1.items():
        whole = inverse_weights(edata, metric, CELL, MIN_IN_CELL)
        central = inverse_weights(edata, metric, CELL, MIN_IN_CELL, CENTRAL)
        cols = []
        for flag in groups.values():
            sel = [d if flag is None else d[d.participant_in_training_set == flag]
                   for d in (edata, whole, central)]
            cols.append([pooled_median(sel[0], metric), pooled_median(sel[1], metric, "weight"),
                         pooled_median(sel[2], metric, "weight")])
        rows[name] = [f"{a:.3f} ({t:.3f}/{u:.3f})" for a, t, u in zip(*cols)]
    t1 = pd.DataFrame(rows, index=["Unweighted, all fixations", "Weighted, all FOV",
                                   "Weighted, central FOV"]).T
    print(f"\nTable 1 by training set, all (trained/untrained) [deg]\n{t1.to_string()}")


# ---------------------------------------------------------------------------- figures


def fig7():
    """Timeline of gaze (left eye blue, right eye red) and of the stimulus as seen from each eye
    (black: solid left, dashed right) in the head frame, whole fixations with blinks left in.
    Angles of the negated vectors, as in the original figure."""
    rec, fixations = FIG7
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    lims = [[np.inf, -np.inf], [np.inf, -np.inf]]
    for eye, color, style in (("left", "tab:blue", "-"), ("right", "tab:red", "--")):
        d = pd.read_csv(data_path("1_accuracy", f"{rec}_{eye}.csv"))
        d = d[d.fixation.isin(fixations)]
        for num, f in d.groupby("fixation"):
            gaze = unit_to_az_el(-f[["visual_axis_x", "visual_axis_y", "visual_axis_z"]].to_numpy())
            stim = unit_to_az_el(-f[["stimulus_dir_x", "stimulus_dir_y", "stimulus_dir_z"]].to_numpy())
            first = num == fixations[0]
            for i, ax in enumerate(axs):
                ax.plot(f.timestamp_sec, gaze[i], "-", color=color, label=f"{eye} eye gaze" if first else None)
                ax.plot(f.timestamp_sec, stim[i], "k" + style, linewidth=1,
                        label=f"stimulus, {eye} eye" if first else None)
                if eye == "left":
                    ax.text(f.timestamp_sec.iloc[0], stim[i][0], num, ha="right", va="center")
                lims[i] = [min(lims[i][0], stim[i].min()), max(lims[i][1], stim[i].max())]
    for ax, (lo, hi) in zip(axs, lims):
        ax.set_ylim(lo - 2, hi + 2)  # blink artefacts run off the scale, as in the paper
        ax.grid(True)
    axs[0].legend(loc="upper right", fontsize="small", ncol=2)
    axs[0].set_ylabel("Azimuth (deg)")
    axs[1].set_ylabel("Elevation (deg)")
    axs[1].set_xlabel("Time (sec)")
    save_figure(fig, "fig07_timeline")


def fig8(df):
    """Optical (top) and visual (bottom) axis per frame around each grid target, both eyes;
    analysis-window frames without blinks, as the accuracy numbers use. Targets: filled and open
    circles are the two checkerboard halves, each corrected with the other's kappa; x is invalid."""
    pid, rec, grid = FIG8.split("_", 1)[0], "_".join(FIG8.split("_")[1:3]), FIG8.split("_")[-1]
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    for col, eye in enumerate(["left", "right"]):
        d = pd.read_csv(data_path("1_accuracy", f"{FIG8}_{eye}.csv"))
        d = d[d.in_analysis_window & ~d.blink]
        fx = df[(df.participant == pid) & (df.recording == rec) & (df.grid == grid) & (df.eye == eye)]
        fx = fx.set_index("fixation")
        for num, f in d.groupby("fixation"):
            sx, sy = f.stimulus_monitor_x_deg.iloc[0], f.stimulus_monitor_y_deg.iloc[0]
            for ax in axs[:, col]:
                if not fx.is_fixation_valid[num]:
                    ax.plot(sx, sy, "xk")
                else:
                    ax.plot(sx, sy, "ok", markerfacecolor="k" if fx.kappa_half[num] == 0 else "none")
                ax.annotate(num, xy=(sx, sy), xytext=(sx + 0.2, sy + 0.2))
            axs[0, col].plot(sx + f.optical_deviation_x_deg, sy + f.optical_deviation_y_deg, ".",
                             color="#ff4d00", alpha=0.3, markersize=2)
            axs[1, col].plot(sx + f.visual_deviation_x_deg.mean(), sy + f.visual_deviation_y_deg.mean(),
                             "o", color="#1f77b4", markersize=2)
            axs[1, col].plot(sx + f.visual_deviation_x_deg, sy + f.visual_deviation_y_deg, ".",
                             color="#2ca02c", alpha=0.3, markersize=2)
        kx, ky = fx.kappa_grid_x_deg.iloc[0], fx.kappa_grid_y_deg.iloc[0]
        halves = fx.groupby("kappa_half")[["kappa_x_deg", "kappa_y_deg"]].first()
        axs[0, col].set_title(f"{eye} eye, kappa {kx:.2f} by {ky:.2f} deg")
        print(f"Fig 8 {eye} eye kappa: {kx:.2f} by {ky:.2f} deg; applied per half: "
              + ", ".join(f"{h}: {r.kappa_x_deg:.2f} by {r.kappa_y_deg:.2f}" for h, r in halves.iterrows()))
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
    save_figure(fig, "fig08_grid")


def fig9(edata):
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
    save_figure(fig, "fig09_fixation_distribution")


def fig10(edata):
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
    save_figure(fig, "fig10_quality_by_cell")


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
    edata = df[df.is_fixation_valid == 1]

    print_counts(df)
    ue = table1(edata, len(df))
    table1_by_training(edata)

    fig7()
    fig8(df)
    fig9(edata)
    fig10(edata)
    fig_ue(ue, "whole", "fig11_ue_whole_fov")
    fig_ue(ue, "central", "fig12_ue_central_fov")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
