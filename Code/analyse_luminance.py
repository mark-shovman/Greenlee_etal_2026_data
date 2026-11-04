"""Background luminance change task: paper Figures 13-15 and the numbers quoted with them.

Reads Data/2_luminance/. Run from anywhere:  python Code/analyse_luminance.py [--show]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from scipy import stats
from statsmodels.formula.api import mixedlm, ols

from utils import (EYE_COLORS, EYES, IG, angle_deviation, angle_unit, blank, data_path, finish_timeline,
                   kappa_blocks, mean_trace, read_series, save_figure, timeline_figure, unit_mean)

MS = 1 / 1000  # [s] per ms; windows are given in ms relative to the luminance change
# Analysis windows (Methods): baseline, constriction, recovery
PARTS = {"before": (-500, 0), "during": (220, 720), "after": (1000, 1500)}
# No-change control: the same shift between two windows before the change, against the baseline
CONTROL_WINDOW = {"early": (-1300, -800)}  # starts 700 ms after the dark target appears
# [ms] a saccade or blink here can move gaze between the windows of either shift (control start
# to recovery end); events outside it cannot, and do not exclude the episode
EVENT_SPAN = (CONTROL_WINDOW["early"][0], PARTS["after"][1])
FIG13 = "p011"  # also did the grid task, whose kappa corrects the figure's gaze
N_BOOT = 10000  # participant-level bootstrap resamples
SEED = 1


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
            m["shift"] = (angle_unit(means["before"], means["after"])
                          if means["before"] is not None and means["after"] is not None else np.nan)
            early = d.loc[window(r, "early", CONTROL_WINDOW), IG]
            m["shift_control"] = (angle_unit(unit_mean(early), means["before"])
                                  if len(early) and means["before"] is not None else np.nan)
            m["pupil_early"] = d.pupil_diameter_mm.loc[window(r, "early", CONTROL_WINDOW)].median()
            m["pupil_change"] = m["pupil_before"] - m["pupil_after"]  # constriction > 0
            m["pupil_change_control"] = m["pupil_early"] - m["pupil_before"]
            ev = events[(events.participant == r.participant) & (events.eye == eye)]
            span = window(r, "span", {"span": EVENT_SPAN})
            overlap = (ev.t_end_sec > span.start) & (ev.t_start_sec < span.stop)
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
    print(f"Constriction: {clean.pupil_change.mean():.3f} +/- {clean.pupil_change.std():.3f} mm; "
          f"apparent gaze shift: {clean['shift'].mean():.3f} +/- {clean['shift'].std():.3f} deg; "
          f"control shift: {clean.shift_control.mean():.3f} +/- {clean.shift_control.std():.3f} deg")
    fit = ols("shift ~ pupil_change", data=clean).fit()
    print(f"OLS shift ~ constriction: b = {fit.params.pupil_change:.3f} deg/mm, "
          f"F({fit.df_model:.0f}, {fit.df_resid:.0f}) = {fit.fvalue:.2f}, p = {fit.f_pvalue:.2e}, "
          f"R2 = {fit.rsquared:.3f}")
    return clean


def print_mixed_model(clean):
    """shift ~ constriction with participant and episode-within-participant random intercepts, so
    the two eyes of an episode and the episodes of a participant are not counted as independent.
    ML fits; the slope is tested by LRT against the model without it."""
    kw = dict(groups="participant", re_formula="1", vc_formula={"episode": "0 + C(episode)"})
    full = mixedlm("shift ~ pupil_change + eye", clean, **kw).fit(reml=False)
    null = mixedlm("shift ~ eye", clean, **kw).fit(reml=False)
    lr = 2 * (full.llf - null.llf)
    lo, hi = full.conf_int().loc["pupil_change"]
    print(f"Mixed model shift ~ constriction + eye + (1|participant) + (1|episode): "
          f"b = {full.params.pupil_change:.3f} deg/mm [95% CI {lo:.3f}, {hi:.3f}], "
          f"intercept = {full.params.Intercept:.3f} deg, "
          f"LRT chi2(1) = {lr:.2f}, p = {stats.chi2.sf(lr, 1):.2e}")
    print(f"  variances: participant {full.cov_re.iloc[0, 0]:.4f}, episode {full.vcomp[0]:.4f}, "
          f"residual {full.scale:.4f} deg^2")
    fit = full

    # check: the two eyes averaged, one row per episode
    ep = clean.groupby(["participant", "episode"])[["shift", "pupil_change"]].mean().reset_index()
    full = mixedlm("shift ~ pupil_change", ep, groups="participant").fit(reml=False)
    null = mixedlm("shift ~ 1", ep, groups="participant").fit(reml=False)
    lr = 2 * (full.llf - null.llf)
    lo, hi = full.conf_int().loc["pupil_change"]
    print(f"  per-episode means ({len(ep)} episodes): b = {full.params.pupil_change:.3f} deg/mm "
          f"[95% CI {lo:.3f}, {hi:.3f}], LRT chi2(1) = {lr:.2f}, p = {stats.chi2.sf(lr, 1):.2e}")
    return fit


def print_control(clean):
    """Is the apparent shift across the luminance change larger than across an interval of the
    same episode without one? Paired per episode-eye, tested on participant means (Wilcoxon
    signed-rank); the CI is in the bootstrap."""
    diff = (clean["shift"] - clean.shift_control).groupby(clean.participant).mean()
    w = stats.wilcoxon(diff)
    print(f"Shift - control, participant means: median {diff.median():.3f} deg, "
          f"{(diff > 0).sum()}/{len(diff)} participants > 0, Wilcoxon W = {w.statistic:.0f}, p = {w.pvalue:.2e}")


def print_bootstrap(clean):
    """Gaze shift, constriction and the no-change control with 95% CIs from resampling
    participants with replacement, each keeping all its episode-eyes."""
    by_pid = {pid: g for pid, g in clean.groupby("participant")}
    pids = list(by_pid)
    rng = np.random.default_rng(SEED)
    stats_of = {"mean shift": lambda d: d["shift"].mean(), "median shift": lambda d: d["shift"].median(),
                "mean constriction": lambda d: d.pupil_change.mean(),
                "mean control shift": lambda d: d.shift_control.mean(),
                "mean control constriction": lambda d: d.pupil_change_control.mean(),
                "mean shift - control": lambda d: (d["shift"] - d.shift_control).mean()}
    boot = {k: np.empty(N_BOOT) for k in stats_of}
    for i in range(N_BOOT):
        d = pd.concat([by_pid[p] for p in rng.choice(pids, len(pids))])
        for k, f in stats_of.items():
            boot[k][i] = f(d)
    print(f"Participant bootstrap ({len(pids)} participants, {N_BOOT} resamples), 95% CI:")
    for k, f in stats_of.items():
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        unit = "mm" if "constriction" in k else "deg"
        print(f"  {k}: {f(clean):.3f} [{lo:.3f}, {hi:.3f}] {unit}")


# ---------------------------------------------------------------------------- figures


def fig13(ep, series):
    """One participant end to end: gaze (visual axis) and the stimulus as seen from each eye
    (black: solid left, dashed right) in the head frame, and pupil diameter; blinks removed. The
    gaze is corrected with the participant's kappa from the grid task, the median over its grid
    blocks (Tables/kappa_blocks); the luminance analysis itself uses the uncorrected optical axis."""
    blocks = kappa_blocks(pd.read_csv(data_path("1_accuracy", "fixations.csv")))
    kappa = blocks[blocks.participant == FIG13].groupby("eye")[["kappa_x_deg", "kappa_y_deg"]].median()
    print(f"Fig 13: {FIG13} grid kappa " + ", ".join(
        f"{eye} {r.kappa_x_deg:.2f} by {r.kappa_y_deg:.2f} deg" for eye, r in kappa.iterrows()))
    fig, grid = timeline_figure(3)
    axs = grid[:, 0]
    for eye, style in zip(EYES, ("-", "--")):
        d = series[(FIG13, eye)]
        for ax, angle, k in zip(axs, ["azimuth_deg", "elevation_deg"], ["kappa_x_deg", "kappa_y_deg"]):
            # visual = optical - kappa, per tangent angle (as the grid's visual_axis)
            ax.plot(d.index, d[angle] - kappa.at[eye, k], color=EYE_COLORS[eye])
            ax.plot(d.index, d[f"stimulus_{angle}"], "k" + style, linewidth=1)
        pupil = d.pupil_diameter_mm.sort_index()
        axs[2].plot(pupil.index, pupil, color=EYE_COLORS[eye])
    for ax in axs:
        ax.set_facecolor("gray")
        ax.set_xlim(0, 32)
        for _, r in ep[ep.participant == FIG13].iterrows():
            ax.axvline(r.t_start, linestyle="--", color="k")
            ax.axvspan(r.t_change, r.t_end, fc="white", alpha=1)
    axs[0].set_ylabel("Azimuth (deg)")
    axs[1].set_ylabel("Elevation (deg)")
    axs[2].set_ylabel("Pupil diameter (mm)")
    finish_timeline(grid)
    save_figure(fig, "fig13_example_participant")


def fig14(ep, clean, series, t_pre=-EVENT_SPAN[0], t_post=EVENT_SPAN[1]):
    """Pupil diameter, its change, and gaze shift around the change, every clean episode-eye;
    black: the mean trace. Baselines: the baseline window. Dark background shaded; the control,
    baseline and recovery windows between dashed lines."""
    b0, b1 = PARTS["before"]
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(9, 6.5))
    for ax in axs:
        ax.axvspan(-t_pre, 0, fc="k", alpha=0.5)
        for a, b in (CONTROL_WINDOW["early"], PARTS["before"], PARTS["after"]):
            ax.axvline(a, color="k", linestyle="--", linewidth=0.8)
            ax.axvline(b, color="k", linestyle="--", linewidth=0.8)
    pupil_traces, gaze_traces = [], []
    times = ep.set_index(["participant", "episode"]).t_change
    for _, r in clean.iterrows():
        t0 = times[(r.participant, r.episode)]
        d = series[(r.participant, r.eye)].loc[t0 - t_pre * MS:t0 + t_post * MS]
        c = EYE_COLORS[r.eye]

        pupil = d.pupil_diameter_mm.copy()
        pupil.index = (pupil.index - t0) / MS
        axs[0].plot(pupil.index, pupil, color=c, alpha=0.5, linewidth=0.3)
        change = (pupil - pupil.loc[b0:b1].median()).rolling(5, center=True).mean()
        axs[1].plot(change.index, change, color=c, alpha=0.5, linewidth=0.3)
        pupil_traces.append(change)

        gd = d[IG].copy()
        gd.index = (gd.index - t0) / MS
        base = unit_mean(gd.loc[b0:b1])
        shift = gd.apply(lambda v: angle_unit(v, base), axis=1).rolling(11, center=True).mean()
        axs[2].plot(shift.index, shift, color=c, alpha=0.5, linewidth=0.3)
        gaze_traces.append(shift)

    for ax, traces in ((axs[1], pupil_traces), (axs[2], gaze_traces)):
        grid, mean, _ = mean_trace(traces, t_pre, t_post)
        ax.plot(grid, mean, "k-", linewidth=1.5)
    axs[0].set_ylabel("Pupil diameter (mm)")
    axs[1].set_ylabel("Pupil size change (mm)")
    axs[2].set_ylabel("Gaze shift (deg)")
    axs[2].set_xlabel("Time (ms)")
    axs[2].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    for ax in axs:
        ax.grid(True)
        ax.set_xlim(-t_pre, t_post)
    save_figure(fig, "fig14_luminance_change")


def fig15(clean, fit):
    """Apparent gaze shift against pupil constriction, every clean episode-eye, with the mixed
    model's fixed-effect line (the two eyes averaged) and its 95% CI. The x axis starts at 0, as in
    the paper, so the few dilating episode-eyes (x < 0) are fitted but not shown."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(clean.pupil_change, clean["shift"], s=12, alpha=0.6)
    x = np.linspace(0, 2.30, 50)
    cols = {"Intercept": np.ones_like(x), "eye[T.right]": np.full_like(x, 0.5), "pupil_change": x}
    names = list(fit.fe_params.index)
    X = np.column_stack([cols[n] for n in names])
    y = X @ fit.fe_params.to_numpy()
    se = np.sqrt(np.einsum("ij,jk,ik->i", X, fit.cov_params().loc[names, names].to_numpy(), X))
    ax.plot(x, y, "k-")
    ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, color="k", alpha=0.15, linewidth=0)
    hidden = clean[clean.pupil_change < 0]
    print(f"Fig 15: {len(hidden)} episode-eyes with dilation (x < 0) are in the fit but outside the plot")
    ax.set_xlabel("Pupil Size Change (mm)")
    ax.set_ylabel("Apparent Gaze Shift (deg)")
    ax.set_xlim(0, 2.30)
    ax.set_ylim(bottom=-0.02)
    ax.grid()
    save_figure(fig, "fig15_shift_vs_constriction")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--show", action="store_true", help="Show the figures as well as saving them")
    args = p.parse_args()

    protocol, events, series = load()
    ep = episodes(protocol)
    metrics = episode_metrics(ep, events, series)
    clean = print_stats(clean_episodes(metrics))
    fit = print_mixed_model(clean)
    print_control(clean)
    print_bootstrap(clean)

    print(f"\nAll episodes, no saccade/blink exclusion ({len(metrics)} episode-eyes):")
    print_stats(metrics)
    print_mixed_model(metrics)
    print_control(metrics)

    fig13(ep, series)
    fig14(ep, clean, series)
    fig15(clean, fit)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
