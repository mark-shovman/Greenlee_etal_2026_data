"""Visual search task: paper Figures 16-17 and the numbers quoted with them. In the marker frame, where the symbols
stand still: the repeated-measures ANOVA on the participants' mean vergence point distance in front of the marker per
depth order, with Holm-corrected comparisons of adjacent depths. On distance from the head (Figs 16-17): the error of
the vergence point against the symbol.

Reads Data/3_vergence/. Run from anywhere:  python Code/analyse_vergence.py [--show]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from utils import (EYE_COLORS, EYES, IC, IG, blank, data_path, finish_timeline, read_series, save_figure,
                   timeline_figure, vergence_point)

N_DEPTHS = 6
MIN_FIXATION_DURATION = 1.0  # [s] a reference fixation must last this long
FIXATION_START_OFFSET = 0.3  # [s] skipped at the start of the reference fixation
# Symbol positions per rig (stimuli.csv), by order_from_participant 1-6: (x, y, z) [mm] in the marker frame (x right,
# y down, z away from the participant), the letter on the face towards the participant. From the scene-camera stereo
# pairs: median over the runs where all six were found (58 rig A, 15 rig B); runs scatter ~1 mm around them.
SYMBOLS = {
    "A": ((-8, 92, -339), (-8, 91, -285), (-9, 91, -229), (-8, 90, -175), (-9, 90, -119), (-10, 89, -63)),
    "B": ((-7, 96, -397), (-6, 92, -343), (-8, 92, -288), (-8, 94, -233), (-7, 93, -176), (-9, 90, -120)),
}
FIG16 = ("p057", "vergence_3")  # the example run: participant, recording
FFZ = [f"ffz{x}" for x in "XYZ"]  # forward axis of the eye's front camera, a unit vector


def load():
    stimuli = pd.read_csv(data_path("3_vergence", "stimuli.csv"))
    fixations = pd.read_csv(data_path("3_vergence", "fixations.csv"))
    raw = {}
    for (pid, rec), _ in stimuli.groupby(["participant", "recording"]):
        for eye in EYES:
            raw[(pid, rec, eye)] = read_series(data_path("3_vergence", f"{pid}_{rec}_{eye}.csv"))
    return stimuli, fixations, raw


def paired_eyes(clean, pid, rec):
    """Both eyes on the right eye's timeline (nearest sample), NaN samples and repeated
    timestamps dropped: the cameras free-run, so the eyes never share timestamps."""
    out = []
    for eye in EYES:
        df = clean[(pid, rec, eye)].dropna()
        out.append(df[~df.index.duplicated(keep="first")])
    left, right = out
    return left.reindex(right.index, method="nearest"), right


def reference_fixations(stimuli, fixations):
    """For each stimulus, the longest binocular fixation of at least MIN_FIXATION_DURATION
    that started while the stimulus was on screen."""
    s = stimuli.copy()
    s[["t_fix_start", "t_fix_end"]] = np.nan
    for (pid, rec), f in fixations.groupby(["participant", "recording"]):
        f = f[(f.t_end_sec - f.t_start_sec) > MIN_FIXATION_DURATION]
        for i in s.index[(s.participant == pid) & (s.recording == rec)]:
            during = f[(f.t_start_sec > s.at[i, "t_start_sec"]) & (f.t_start_sec < s.at[i, "t_end_sec"])]
            if len(during):
                j = (during.t_end_sec - during.t_start_sec).idxmax()
                s.loc[i, ["t_fix_start", "t_fix_end"]] = during.loc[j, ["t_start_sec", "t_end_sec"]].values
    s["t_fix_start_offset"] = s.t_fix_start + FIXATION_START_OFFSET
    return s


def head_distance(v, p):
    """Distance [mm] of the points p (N, 3) from the head, along its forward axis, per stimulus row of v."""
    head = v[["head_x", "head_y", "head_z"]].to_numpy()
    fwd = v[["fwd_x", "fwd_y", "fwd_z"]].to_numpy()
    return np.einsum("ij,ij->i", p - head, fwd)


def stimulus_vergence(s, clean):
    """Vergence per stimulus, from the mean limbus centres and gaze over its reference fixation (first 300 ms
    skipped); gaze points into the eye, so the rays run along -ig. The head over the same window: the midpoint of the
    limbus centres, and the mean of the two front cameras' forward axes. vergence_mm and symbol_mm: the vergence
    point's and the symbol's distance from the head, along its forward axis. marker_mm and symbol_marker_mm: their
    distance in front of the marker, towards the participant."""
    for i, r in s.iterrows():
        if np.isnan(r.t_fix_start):
            continue
        gd, ic, fz = {}, {}, {}
        for eye in EYES:
            w = clean[(r.participant, r.recording, eye)].loc[r.t_fix_start_offset:r.t_fix_end]
            g = w[IG].mean()
            gd[eye] = (g / np.linalg.norm(g)).values
            ic[eye] = w[IC].mean().values
            fz[eye] = w[FFZ].mean().values
        s.loc[i, ["vp_x", "vp_y", "vp_z"]] = vergence_point(ic["left"], -gd["left"], ic["right"], -gd["right"])[0]
        s.loc[i, ["head_x", "head_y", "head_z"]] = (ic["left"] + ic["right"]) / 2
        f = fz["left"] + fz["right"]
        s.loc[i, ["fwd_x", "fwd_y", "fwd_z"]] = f / np.linalg.norm(f)
    symbols = np.array([SYMBOLS[r][k - 1] for r, k in zip(s.rig, s.order_from_participant)], float)
    s["vergence_mm"] = head_distance(s, s[["vp_x", "vp_y", "vp_z"]].to_numpy())
    s["symbol_mm"] = head_distance(s, symbols)
    s["marker_mm"] = -s.vp_z
    s["symbol_marker_mm"] = -symbols[:, 2]
    return s


def vergence_series(clean, pid, rec):
    """Per sample, the vergence point's distance from the head (mm), as in Fig 17: along the mean of the
    two front cameras' forward axes, from the midpoint of the limbus centres; on the right eye's
    timeline; for display (Fig 16) only."""
    left, right = paired_eyes(clean, pid, rec)
    p = vergence_point(left[IC].values, -left[IG].values, right[IC].values, -right[IG].values)
    head = (left[IC].values + right[IC].values) / 2
    fwd = left[FFZ].values + right[FFZ].values
    fwd /= np.linalg.norm(fwd, axis=1)[:, None]
    return pd.Series(np.einsum("ij,ij->i", p - head, fwd), index=right.index)


# ---------------------------------------------------------------------------- repeated-measures ANOVA


def participant_means(v, col="marker_mm"):
    """Mean of col per participant and depth order, for the participants with an estimate at every
    depth: participants x depths."""
    return v.groupby(["participant", "order_from_participant"])[col].mean().unstack().dropna()


def rm_anova(m):
    """One-way repeated-measures ANOVA on participants x levels, with Mauchly's test of sphericity
    and the Greenhouse-Geisser correction. Returns a dict of the statistics."""
    y = m.to_numpy()
    n, k = y.shape
    grand = y.mean()
    ss_level = n * ((y.mean(0) - grand) ** 2).sum()
    ss_subject = k * ((y.mean(1) - grand) ** 2).sum()
    ss_error = ((y - grand) ** 2).sum() - ss_level - ss_subject
    df1, df2 = k - 1, (k - 1) * (n - 1)
    f = (ss_level / df1) / (ss_error / df2)
    # sphericity, on the covariance of k - 1 orthonormal contrasts
    c = np.linalg.qr(np.vstack([np.ones(k), np.eye(k)[:-1]]).T)[0][:, 1:]
    s = np.cov(y @ c, rowvar=False)
    pp = k - 1
    w = np.linalg.det(s) / (np.trace(s) / pp) ** pp
    chi2 = -(n - 1 - (2 * pp ** 2 + pp + 2) / (6 * pp)) * np.log(w)
    eps = np.trace(s) ** 2 / (pp * np.trace(s @ s))
    return dict(n=n, k=k, F=f, df1=df1, df2=df2, p=stats.f.sf(f, df1, df2), eta_p2=ss_level / (ss_level + ss_error),
                mauchly_W=w, mauchly_p=stats.chi2.sf(chi2, pp * (pp + 1) / 2 - 1), gg_eps=eps,
                gg_p=stats.f.sf(f, eps * df1, eps * df2))


def print_anova(v):
    """Depth order on the participants' mean vergence point distance in front of the marker (averaged over runs; the
    symbols stand still in the marker frame): repeated-measures ANOVA; adjacent depths by paired t-tests,
    Holm-corrected, the step set against the symbols' own step."""
    m = participant_means(v)
    sym = participant_means(v, "symbol_marker_mm").loc[m.index]
    a = rm_anova(m)
    print(f"Repeated-measures ANOVA over {a['n']} participants' mean distance in front of the marker per depth: "
          f"F({a['df1']}, {a['df2']}) = {a['F']:.2f}, p = {a['p']:.1e}, partial eta^2 = {a['eta_p2']:.2f}; "
          f"Mauchly W = {a['mauchly_W']:.3f}, p = {a['mauchly_p']:.1e}; Greenhouse-Geisser epsilon "
          f"{a['gg_eps']:.3f}: F({a['gg_eps'] * a['df1']:.2f}, {a['gg_eps'] * a['df2']:.2f}) = {a['F']:.2f}, "
          f"p = {a['gg_p']:.1e}")
    rows = []
    for k in range(1, N_DEPTHS):
        d = m[k] - m[k + 1]  # nearer minus farther, in front of the marker: positive when ordered correctly
        t, p = stats.ttest_1samp(d, 0)
        rows.append((k, d.mean(), (sym[k] - sym[k + 1]).mean(), d.mean() / d.std(), t, p, (d > 0).sum()))
    holm = multipletests([r[5] for r in rows], method="holm")[1]
    print(f"Adjacent depths, paired t-tests over the {len(m)} participants' means, Holm-corrected:")
    for (k, step, true, dz, t, p, right), ph in zip(rows, holm):
        print(f"  depth {k} to {k + 1}: step {step:.1f} mm (symbols {true:.1f} mm), dz = {dz:.2f}, "
              f"t({len(m) - 1}) = {t:.2f}, p_Holm = {ph:.1e}; correct sign in {right}/{len(m)}")


def print_error(v):
    """Vergence point against the symbol, both as distance from the head: median and IQR of the difference."""
    e = v.vergence_mm - v.symbol_mm
    print("Vergence minus symbol distance from the head [mm], median (IQR): "
          + "; ".join(f"{name} {g.median():.0f} ({g.quantile(.25):.0f} to {g.quantile(.75):.0f}), n = {len(g)}"
                      for name, g in [("all", e)] + [(f"rig {r}", e[v.rig == r]) for r in sorted(v.rig.unique())]))


# ---------------------------------------------------------------------------- figures


def fig16(s, raw, clean):
    """One run end to end: gaze azimuth and elevation per eye (faint: before blink removal), and the
    vergence point's distance from the head as in Fig 17, per sample (grey) and per stimulus over its
    reference fixation (black; dashed: the symbol's distance, from front-camera stereo)."""
    pid, rec = FIG16
    fig, grid = timeline_figure(3)
    axs = grid[:, 0]
    g = s[(s.participant == pid) & (s.recording == rec)].sort_values("t_start_sec")
    t0, t1 = g.t_start_sec.iloc[0], g.t_end_sec.iloc[-1]
    starts = g.t_start_sec  # every stimulus, also those without a vergence point
    for eye in EYES:
        c = EYE_COLORS[eye]
        for d, kw in ((raw[(pid, rec, eye)], dict(linewidth=0.3, alpha=0.1)), (clean[(pid, rec, eye)], {})):
            axs[0].plot(d.index, d.azimuth_deg, color=c, **kw)
            axs[1].plot(d.index, d.elevation_deg, color=c, **kw)
    va = vergence_series(clean, pid, rec)
    axs[2].plot(va.index, va, c="k", alpha=0.3)
    for _, r in g.dropna(subset=["vergence_mm"]).iterrows():
        span = (r.t_fix_start_offset, r.t_fix_end)
        axs[2].plot(span, (r.symbol_mm, r.symbol_mm), color="k", linestyle="--", linewidth=1)
        axs[2].plot(span, (r.vergence_mm, r.vergence_mm), color="k")
        axs[2].text(r.t_fix_start_offset, r.vergence_mm, f"[{r.symbol}]\n{r.vergence_mm:.0f}", ha="right",
                    va="center", weight="bold")
    # scale to the blink-free data; the faint traces' blink artefacts run off the axes
    for ax, col in zip(axs[:2], ["azimuth_deg", "elevation_deg"]):
        v = pd.concat([clean[(pid, rec, eye)].loc[t0:t1, col] for eye in EYES])
        ax.set_ylim(v.min() - 1, v.max() + 1)
    axs[2].set_ylim(va.loc[t0:t1].quantile(.01) - 100, va.loc[t0:t1].quantile(.99) + 100)
    for ax in axs:
        for t in list(starts) + [t1]:
            ax.axvline(t, color="k", linestyle="--")
    axs[0].set_ylabel("Azimuth (deg)")
    axs[1].set_ylabel("Elevation (deg)")
    axs[2].set_ylabel("Vergence point distance\nfrom the head (mm)")
    axs[0].set_xlim(t0, t1)
    finish_timeline(grid)
    save_figure(fig, "fig16_example_run")


RIG_STYLE = {"A": dict(marker="o", color="0.55"), "B": dict(marker="^", color="0.15")}


def fig17(v):
    """Vergence point against the symbol, both as distance from the head along its forward axis (head: midpoint of
    the limbus centres; forward: mean of the two front cameras' axes; both over the reference fixation), one point
    per reference fixation. Large markers: median per symbol and rig, bars: interquartile range. Dashed: vergence
    point on the symbol."""
    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    lim = (min(v.symbol_mm.min(), v.vergence_mm.min()) - 30, max(v.symbol_mm.max(), v.vergence_mm.max()) + 30)
    ax.plot(lim, lim, "k--", linewidth=1, zorder=1)
    for rig, st in RIG_STYLE.items():
        d = v[v.rig == rig]
        ax.scatter(d.symbol_mm, d.vergence_mm, s=9, marker=st["marker"], facecolor="none", edgecolor=st["color"],
                   linewidth=0.6, alpha=0.7, zorder=2,
                   label=f"Rig {rig} ({d.participant.nunique()} participants, {len(d)} fixations)")
        q = d.groupby("order_from_participant")[["symbol_mm", "vergence_mm"]].quantile([0.25, 0.5, 0.75]).unstack()
        x, y = q[("symbol_mm", 0.5)], q[("vergence_mm", 0.5)]
        ax.errorbar(x, y, yerr=[y - q[("vergence_mm", 0.25)], q[("vergence_mm", 0.75)] - y], fmt=st["marker"],
                    color="k", markerfacecolor="white", markersize=7, capsize=4, linewidth=1.2, zorder=3)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.grid(alpha=0.4)
    ax.set_xlabel("Symbol distance from the head (mm)")
    ax.set_ylabel("Vergence point distance from the head (mm)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    save_figure(fig, "fig17_vergence_vs_symbol_depth")


def prepare():
    """The data and the per-stimulus vergence: (stimuli with vergence, fixations, raw, clean)."""
    stimuli, fixations, raw = load()
    clean = {k: blank(v) for k, v in raw.items()}
    return stimulus_vergence(reference_fixations(stimuli, fixations), clean), fixations, raw, clean


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--show", action="store_true", help="Show the figures as well as saving them")
    args = p.parse_args()

    s, fixations, raw, clean = prepare()

    runs = s.groupby("participant").recording.nunique().value_counts().sort_index()
    valid = s[s.t_fix_start.notna()]  # with a reference fixation
    v = valid[valid.vergence_mm.notna()]  # and a vergence point
    print(f"{s.participant.nunique()} participants; runs per participant -> participants: {runs.to_dict()}; "
          f"{len(s)} stimuli, {len(valid)} with a reference fixation ({len(s) - len(valid)} without), "
          f"{len(v)} with a vergence point; per rig, participants: {v.groupby('rig').participant.nunique().to_dict()}")
    print("By depth order [mm], mean +/- SD: distance in front of the marker, and from the head:")
    cols = ["marker_mm", "symbol_marker_mm", "vergence_mm", "symbol_mm"]
    print(v.groupby("order_from_participant")[cols].agg(["mean", "std"]).round(1).to_string())

    print_anova(v)
    print_error(v)

    fig16(s, raw, clean)
    fig17(v)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
