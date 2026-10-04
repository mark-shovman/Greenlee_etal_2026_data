"""Background luminance change task: paper Figures 10-12 and the numbers quoted with them.

Reads Data/2_luminance/. Run from anywhere:  python Code/analyse_luminance.py [--show]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from statsmodels.formula.api import ols

from utils import (EYE_COLORS, EYES, IG, angle_deviation, angle_unit, blank, data_path, mean_trace,
                   read_series, save_figure, unit_mean)

MS = 1 / 1000  # [s] per ms; windows are given in ms relative to the luminance change
# Analysis windows (Methods): baseline, constriction, recovery
PARTS = {"before": (-500, 0), "during": (220, 720), "after": (1000, 1500)}
# The windows of the constriction figure quoted in the Results (see the README errata)
CONSTRICTION_WINDOWS = {"before": (-1000, 0), "after": (1000, 2000)}
EVENT_LOOKBACK = 1000  # [ms] saccades/blinks are looked for from here before the change
FIG10 = "p011"


def load():
    protocol = pd.read_csv(data_path("2_luminance", "protocol.csv"))
    events = pd.read_csv(data_path("2_luminance", "events.csv"))
    series = {(pid, eye): blank(read_series(data_path("2_luminance", f"{pid}_{eye}.csv")))
              for pid in protocol.participant.unique() for eye in EYES}
    return protocol, events, series


def episodes(protocol):
    """One row per episode: a dark stimulus, then the same one on a bright background.
    t_start: onset of the dark stimulus; t_change: the switch to bright; t_end: its end."""
    rows = []
    for pid, p in protocol.groupby("participant", sort=False):
        p = p.reset_index(drop=True)
        bright = [i for i in p.index[p.luminance == "bright"] if i > 0]
        for n, i in enumerate(bright, 1):
            rows.append((pid, n, p.stimulus_x_deg[i], p.stimulus_y_deg[i],
                         p.t_start_sec[i - 1], p.t_start_sec[i], p.t_end_sec[i]))
    return pd.DataFrame(rows, columns=["participant", "episode", "x", "y", "t_start", "t_change", "t_end"])


def window(r, part, parts=PARTS):
    t0, t1 = parts[part]
    return slice(r.t_change + t0 * MS, r.t_change + t1 * MS)


def episode_metrics(ep, events, series):
    """Per episode and eye: gaze spread and median pupil per window, apparent gaze shift, and
    the saccades and blinks overlapping the episode."""
    rows = []
    for _, r in ep.iterrows():
        for eye in EYES:
            d = series[(r.participant, eye)]
            m = {"participant": r.participant, "episode": r.episode, "eye": eye}
            means = {}
            for part in PARTS:
                gd = d.loc[window(r, part), IG]
                m[f"gaze_spread_{part}"] = angle_deviation(gd) if len(gd) else np.nan
                means[part] = unit_mean(gd) if len(gd) else None
                m[f"pupil_{part}"] = d.pupil_diameter_mm.loc[window(r, part)].median()
            for part in CONSTRICTION_WINDOWS:
                m[f"pupil_{part}_long"] = d.pupil_diameter_mm.loc[window(r, part, CONSTRICTION_WINDOWS)].median()
            m["shift"] = (angle_unit(means["before"], means["after"])
                          if means["before"] is not None and means["after"] is not None else np.nan)
            ev = events[(events.participant == r.participant) & (events.eye == eye)]
            t_from = r.t_change - EVENT_LOOKBACK * MS
            overlap = (ev.t_end_sec > t_from) & (ev.t_start_sec < r.t_end)
            m["n_saccades"] = int((overlap & (ev.event == "saccade")).sum())
            m["n_blinks"] = int((overlap & (ev.event == "blink")).sum())
            rows.append(m)
    return pd.DataFrame(rows)


def clean_episodes(metrics):
    """Episodes with no saccade and no blink in either eye; excluded in full otherwise."""
    per_ep = metrics.groupby(["participant", "episode"])[["n_saccades", "n_blinks"]].sum()
    ok = per_ep[(per_ep.n_saccades < 1) & (per_ep.n_blinks < 1)].index
    keep = metrics.set_index(["participant", "episode"]).index.isin(ok)
    n_trials, n_excl = len(per_ep), len(per_ep) - len(ok)
    print(f"{metrics.participant.nunique()} participants, {n_trials} trials; "
          f"{n_excl} ({100 * n_excl / n_trials:.1f}%) excluded for a saccade or blink in either eye; "
          f"{keep.sum()}/{len(metrics)} episode-eyes kept, from "
          f"{metrics[keep].participant.nunique()} participants")
    return metrics[keep].copy()


def print_stats(clean):
    long = (clean.pupil_before_long - clean.pupil_after_long).mean()
    short = (clean.pupil_before - clean.pupil_after).mean()
    print(f"Mean constriction, {CONSTRICTION_WINDOWS['before']} vs {CONSTRICTION_WINDOWS['after']} ms: "
          f"{long:.4f} mm (Results); {PARTS['before']} vs {PARTS['after']} ms: {short:.4f} mm (Methods windows)")
    print(f"Apparent gaze shift: {clean['shift'].mean():.3f} +/- {clean['shift'].std():.3f} deg")

    clean["pupil_change"] = clean.pupil_before - clean.pupil_after
    fit = ols("shift ~ pupil_change", data=clean).fit()
    print(f"OLS shift ~ constriction: b = {fit.params.pupil_change:.3f} deg/mm, "
          f"F({fit.df_model:.0f}, {fit.df_resid:.0f}) = {fit.fvalue:.2f}, p = {fit.f_pvalue:.2e}, "
          f"R2 = {fit.rsquared:.3f}")
    return clean


# ---------------------------------------------------------------------------- figures


def fig10(protocol, ep, series):
    """One participant end to end: stimulus, gaze relative to its first sample, pupil."""
    p = protocol[protocol.participant == FIG10]
    t0 = p.t_start_sec.iloc[0]
    fig, axs = plt.subplots(3, 1, sharex=True)
    # the protocol's Y runs the opposite way to gaze elevation, hence the sign
    for sign, col, ax in zip((1, -1), ["stimulus_x_deg", "stimulus_y_deg"], axs):
        ax.step((p.t_start_sec - t0) / MS, sign * p[col], where="post", color="k")
    for eye in EYES:
        d = series[(FIG10, eye)]
        gd = d[["azimuth_deg", "elevation_deg"]]
        gd = gd - gd.iloc[0]
        t = (d.index - t0) / MS
        axs[0].plot(t, gd.azimuth_deg, color=EYE_COLORS[eye])
        axs[1].plot(t, gd.elevation_deg, color=EYE_COLORS[eye])
        pupil = d.pupil_diameter_mm.sort_index()
        axs[2].plot((pupil.index - t0) / MS, pupil, color=EYE_COLORS[eye])
    for ax in axs:
        ax.set_facecolor("gray")
        ax.set_xlim(0, 32000)
        ax.grid(True)
        for _, r in ep[ep.participant == FIG10].iterrows():
            ax.axvline((r.t_start - t0) / MS, linestyle="--", color="k")
            ax.axvspan((r.t_change - t0) / MS, (r.t_end - t0) / MS, fc="white", alpha=1)
    axs[0].set_ylabel("Azimuth (deg)")
    axs[1].set_ylabel("Elevation (deg)")
    axs[2].set_ylabel("Pupil diameter (mm)")
    axs[2].set_xlabel("Time (msec)")
    axs[2].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    save_figure(fig, "fig10_example_participant")


def fig11(ep, clean, series, t_pre=1000, t_post=1500):
    """Pupil diameter, its change, and gaze shift around the change, every clean episode-eye;
    black: the mean trace. Baselines: the second before the change."""
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(9, 6.5))
    for ax in axs:
        ax.axvspan(-t_pre, 0, fc="k", alpha=0.5)
    pupil_traces, gaze_traces = [], []
    times = ep.set_index(["participant", "episode"]).t_change
    for _, r in clean.iterrows():
        t0 = times[(r.participant, r.episode)]
        d = series[(r.participant, r.eye)].loc[t0 - t_pre * MS:t0 + t_post * MS]
        c = EYE_COLORS[r.eye]

        pupil = d.pupil_diameter_mm.copy()
        pupil.index = (pupil.index - t0) / MS
        axs[0].plot(pupil.index, pupil, color=c, alpha=0.5, linewidth=0.3)
        change = (pupil - pupil.loc[-1000:0].mean()).rolling(5, center=True).mean()
        axs[1].plot(change.index, change, color=c, alpha=0.5, linewidth=0.3)
        pupil_traces.append(change)

        gd = d[IG].copy()
        gd.index = (gd.index - t0) / MS
        base = unit_mean(gd.loc[-1000:0])
        shift = gd.apply(lambda v: angle_unit(v, base), axis=1).rolling(11, center=True).mean()
        axs[2].plot(shift.index, shift, color=c, alpha=0.5, linewidth=0.3)
        gaze_traces.append(shift)

    for ax, traces in ((axs[1], pupil_traces), (axs[2], gaze_traces)):
        grid, mean, _ = mean_trace(traces, t_pre, t_post)
        ax.plot(grid, mean, "k-", linewidth=1.5)
    axs[0].set_ylabel("Pupil diameter (mm)")
    axs[1].set_ylabel("Pupil size change (mm)")
    axs[2].set_ylabel("Gaze shift (deg)")
    axs[2].set_xlabel("Time (msec)")
    axs[2].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    for ax in axs:
        ax.grid(True)
        ax.set_xlim(-t_pre, t_post)
    save_figure(fig, "fig11_luminance_change")


def fig12(clean):
    """Apparent gaze shift against pupil constriction, with the OLS fit over all episode-eyes.
    The x axis starts at 0, as in the paper, so the few dilating episode-eyes (x < 0) are fitted
    but not shown."""
    g = sns.lmplot(data=clean, x="pupil_change", y="shift")
    g.set_axis_labels("Pupil Size Change (mm)", "Apparent Gaze Shift (deg)")
    hidden = clean[clean.pupil_change < 0]
    print(f"Fig 12: {len(hidden)} episode-eyes with dilation (x < 0) are in the fit but outside the plot")
    plt.xlim(0, 2.30)
    plt.ylim(bottom=-0.02)
    plt.grid()
    save_figure(g.figure, "fig12_shift_vs_constriction")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--show", action="store_true", help="Show the figures as well as saving them")
    args = p.parse_args()

    protocol, events, series = load()
    ep = episodes(protocol)
    clean = print_stats(clean_episodes(episode_metrics(ep, events, series)))

    fig10(protocol, ep, series)
    fig11(ep, clean, series)
    fig12(clean)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
