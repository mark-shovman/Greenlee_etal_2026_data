"""Background luminance change task: paper Figures 13-15 and the numbers quoted with them.

Reads Data/2_luminance/. Run from anywhere:  python Code/analyse_luminance.py [--show]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.patches import Rectangle
from scipy import stats
from statsmodels.formula.api import mixedlm

from utils import (EYE_COLORS, EYES, IG, align_traces, angle_unit, blank, data_path, finish_timeline,
                   kappa_blocks, pooled_median, read_series, save_figure, timeline_figure, unit_mean)

MS = 1 / 1000  # [s] per ms; windows are given in ms relative to the luminance change
# Analysis windows (Methods). The shift is baseline to recovery; the no-change control shift is
# control to baseline, the control window starting 700 ms after the dark target appears.
WINDOWS = {"control": (-1300, -800), "baseline": (-500, 0), "recovery": (1000, 1500)}
# [ms] a saccade or blink here can move gaze between the windows of either shift (control start
# to recovery end); events outside it cannot, and do not exclude the episode
EVENT_SPAN = (WINDOWS["control"][0], WINDOWS["recovery"][1])
SAMPLE_RATE = 120  # [Hz] eye cameras
# Missing-data gate, as the grid's: in each window, the share of the expected samples with gaze and
# with pupil must be at least this, in both eyes
MIN_COVERAGE = 0.3
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


def draw_windows(ax, t0=0, unit=1):
    """The control, baseline and recovery windows as thin open boxes over 5-95% of the panel
    height; t0: the luminance change, unit: x-axis units per ms."""
    for a, b in WINDOWS.values():
        ax.add_patch(Rectangle((t0 + a * unit, 0.05), (b - a) * unit, 0.9, transform=ax.get_xaxis_transform(),
                               fill=False, edgecolor="k", linewidth=0.8, zorder=3))


def window(r, span):
    """The time slice [s] of span [ms] around episode r's luminance change."""
    t0, t1 = span
    return slice(r.t_change + t0 * MS, r.t_change + t1 * MS)


def episode_metrics(ep, events, series):
    """Per episode and eye: median pupil per window, apparent gaze shift, the saccades and blinks
    overlapping the episode, and the lowest share of the expected samples with gaze or with pupil
    over the three windows (coverage)."""
    angle = lambda u, v: angle_unit(u, v) if u is not None and v is not None else np.nan
    rows = []
    for _, r in ep.iterrows():
        for eye in EYES:
            d = series[(r.participant, eye)]
            m = {"participant": r.participant, "episode": r.episode, "eye": eye}
            gaze, coverage = {}, []
            for name, (t0, t1) in WINDOWS.items():
                w = d.loc[window(r, (t0, t1))]
                gaze[name] = unit_mean(w[IG]) if len(w) else None
                m[f"pupil_{name}"] = w.pupil_diameter_mm.median()
                expected = (t1 - t0) * MS * SAMPLE_RATE
                coverage.append(w[["igX", "pupil_diameter_mm"]].notna().sum().min() / expected)
            m["shift"] = angle(gaze["baseline"], gaze["recovery"])
            m["shift_control"] = angle(gaze["control"], gaze["baseline"])
            m["pupil_change"] = m["pupil_baseline"] - m["pupil_recovery"]  # constriction > 0
            m["pupil_change_control"] = m["pupil_control"] - m["pupil_baseline"]
            ev = events[(events.participant == r.participant) & (events.eye == eye)]
            span = window(r, EVENT_SPAN)
            overlap = (ev.t_end_sec > span.start) & (ev.t_start_sec < span.stop)
            m["n_saccades"] = int((overlap & (ev.event == "saccade")).sum())
            m["n_blinks"] = int((overlap & (ev.event == "blink")).sum())
            m["coverage"] = min(coverage)
            rows.append(m)
    return pd.DataFrame(rows)


def clean_episodes(metrics):
    """Episodes with no saccade and no blink in either eye, and with both eyes passing the
    missing-data gate; excluded in full otherwise."""
    per_ep = metrics.groupby(["participant", "episode"]).agg(
        n_saccades=("n_saccades", "sum"), n_blinks=("n_blinks", "sum"), coverage=("coverage", "min"))
    no_event = (per_ep.n_saccades < 1) & (per_ep.n_blinks < 1)
    covered = per_ep.coverage >= MIN_COVERAGE
    ok = per_ep[no_event & covered].index
    keep = metrics.set_index(["participant", "episode"]).index.isin(ok)
    n_trials = len(per_ep)
    n_event, n_missing = (~no_event).sum(), (no_event & ~covered).sum()
    print(f"{metrics.participant.nunique()} participants, {n_trials} trials; "
          f"{n_event} ({100 * n_event / n_trials:.1f}%) excluded for a saccade or blink in either eye, "
          f"{n_missing} more for missing data (under {MIN_COVERAGE:.0%} of the expected samples in a window); "
          f"{keep.sum()}/{len(metrics)} episode-eyes kept, from "
          f"{metrics[keep].participant.nunique()} participants")
    return metrics[keep].copy()


def print_stats(clean):
    print(f"Constriction: {clean.pupil_change.mean():.3f} +/- {clean.pupil_change.std():.3f} mm; "
          f"apparent gaze shift: {clean['shift'].mean():.3f} +/- {clean['shift'].std():.3f} deg; "
          f"control shift: {clean.shift_control.mean():.3f} +/- {clean.shift_control.std():.3f} deg")


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


def mean_band(aligned, pids):
    """Mean over the episode-eyes at each time point (columns of aligned), and its 95% band from
    resampling participants with replacement, each keeping all its episode-eyes (as
    print_bootstrap). Not drawn where under half of the episode-eyes have data (the span edges)."""
    codes, uniq = pd.factorize(pids)
    aligned = np.where(np.isfinite(aligned).mean(1, keepdims=True) >= 0.5, aligned, np.nan)
    finite = np.isfinite(aligned)
    s = np.stack([np.where(finite, aligned, 0)[:, codes == k].sum(1) for k in range(len(uniq))])
    n = np.stack([finite[:, codes == k].sum(1) for k in range(len(uniq))])
    w = np.random.default_rng(SEED).multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=N_BOOT)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = (w @ s) / (w @ n)
        mean = s.sum(0) / n.sum(0)
    lo, hi = np.full_like(mean, np.nan), np.full_like(mean, np.nan)
    ok = np.isfinite(mean)
    lo[ok], hi[ok] = np.percentile(boot[:, ok], [2.5, 97.5], axis=0)
    return mean, lo, hi


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
            draw_windows(ax, r.t_change, MS)
    axs[0].set_ylabel("Azimuth (deg)")
    axs[1].set_ylabel("Elevation (deg)")
    axs[2].set_ylabel("Pupil diameter (mm)")
    finish_timeline(grid)
    save_figure(fig, "fig13_example_participant")


def fig14(ep, clean, series):
    """Pupil diameter, its change, and gaze shift around the change. Black: the mean over the
    clean episode-eyes, with its 95% band from the participant bootstrap; grey: each episode-eye
    (change and shift only). Change and shift are from the baseline window. Dark background
    shaded; the control, baseline and recovery windows as open boxes."""
    b0, b1 = WINDOWS["baseline"]
    t_pre, t_post = -EVENT_SPAN[0], EVENT_SPAN[1]  # [ms] the span shown, as the event check
    margin = 50  # [ms] the outer windows end at the data span
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(9, 6.5), layout="constrained")
    traces = {"diameter": [], "change": [], "shift": []}
    times = ep.set_index(["participant", "episode"]).t_change
    for _, r in clean.iterrows():
        t0 = times[(r.participant, r.episode)]
        d = series[(r.participant, r.eye)].loc[t0 - t_pre * MS:t0 + t_post * MS]
        pupil = d.pupil_diameter_mm.copy()
        pupil.index = (pupil.index - t0) / MS
        traces["diameter"].append(pupil)
        traces["change"].append((pupil - pupil.loc[b0:b1].median()).rolling(5, center=True).mean())
        gd = d[IG].copy()
        gd.index = (gd.index - t0) / MS
        base = unit_mean(gd.loc[b0:b1])
        traces["shift"].append(gd.apply(lambda v: angle_unit(v, base), axis=1).rolling(11, center=True).mean())

    for ax, (name, tr) in zip(axs, traces.items()):
        ax.axvspan(-t_pre - margin, 0, fc="gray")
        if name != "diameter":
            for t in tr:
                ax.plot(t.index, t, color="k", alpha=0.08, linewidth=0.4)
        grid, aligned = align_traces(tr, t_pre, t_post)
        mean, lo, hi = mean_band(aligned, clean.participant.to_numpy())
        ax.fill_between(grid, lo, hi, color="k", alpha=0.3, linewidth=0)
        ax.plot(grid, mean, "k-", linewidth=1.5)
        draw_windows(ax)
    axs[0].set_ylabel("Pupil diameter (mm)")
    axs[1].set_ylabel("Pupil size change (mm)")
    axs[2].set_ylabel("Gaze shift (deg)")
    axs[2].set_xlabel("Time (ms)")
    axs[2].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    for ax in axs:
        ax.grid(True)
        ax.set_xlim(-t_pre - margin, t_post + margin)
    save_figure(fig, "fig14_luminance_change")


def fig15(clean, fit):
    """Left (H1): per participant, the mean apparent gaze shift across the luminance change against
    the no-change control, with the identity line. Right (H2): the shift against pupil
    constriction, every clean episode-eye, with the mixed model's fixed-effect line (the two eyes
    averaged) and its 95% CI; dashed: the median accuracy of the grid task (Table 1, unweighted)."""
    fx = pd.read_csv(data_path("1_accuracy", "fixations.csv"))
    accuracy = pooled_median(fx[fx.is_fixation_valid == 1], "accuracy_centroid_deg")
    pm = clean.groupby("participant")[["shift", "shift_control"]].mean()
    print(f"Fig 15: {(pm['shift'] > pm.shift_control).sum()}/{len(pm)} participants above the identity line; "
          f"median grid accuracy {accuracy:.3f} deg")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")

    a1.scatter(pm.shift_control, pm["shift"], s=16, color="k", alpha=0.6)
    lim = 1.05 * pm.max().max()
    a1.plot([0, lim], [0, lim], "k--", linewidth=1)
    a1.set_xlim(0, lim)
    a1.set_ylim(0, lim)
    a1.set_aspect("equal")
    a1.set_xlabel("Control gaze shift (deg)")
    a1.set_ylabel("Apparent gaze shift (deg)")

    a2.scatter(clean.pupil_change_control, clean.shift_control, s=12, facecolors="none", edgecolors="k",
               linewidths=0.6, alpha=0.6)
    a2.scatter(clean.pupil_change, clean["shift"], s=12, color="k", alpha=0.5)
    names = list(fit.fe_params.index)

    def model_line(x):
        cols = {"Intercept": np.ones_like(x), "eye[T.right]": np.full_like(x, 0.5), "pupil_change": x}
        X = np.column_stack([cols[n] for n in names])
        se = np.sqrt(np.einsum("ij,jk,ik->i", X, fit.cov_params().loc[names, names].to_numpy(), X))
        return X @ fit.fe_params.to_numpy(), se

    x = np.linspace(clean.pupil_change.min(), clean.pupil_change.max(), 50)
    y, se = model_line(x)
    a2.plot(x, y, "k-")
    a2.fill_between(x, y - 1.96 * se, y + 1.96 * se, color="k", alpha=0.15, linewidth=0)
    # extrapolated to the control's pupil change, which is not in the fit
    x = np.linspace(clean.pupil_change_control.min(), clean.pupil_change.min(), 50)
    a2.plot(x, model_line(x)[0], "k:")
    a2.axhline(accuracy, color="k", linestyle="--", linewidth=1)
    a2.set_xlabel("Pupil constriction (mm)")
    a2.set_ylabel("Apparent gaze shift (deg)")
    a2.set_ylim(bottom=-0.02)
    for ax in (a1, a2):
        ax.grid(True)
    save_figure(fig, "fig15_shift_vs_constriction")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--show", action="store_true", help="Show the figures as well as saving them")
    args = p.parse_args()

    protocol, events, series = load()
    ep = episodes(protocol)
    metrics = episode_metrics(ep, events, series)
    clean = clean_episodes(metrics)
    print_stats(clean)
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
