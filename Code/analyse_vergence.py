"""Visual search task: paper Figures 13-15, the pairwise discriminability tests and the
leave-one-out depth decoding, with the numbers quoted with them.

Reads Data/3_vergence/. Run from anywhere:  python Code/analyse_vergence.py [--show]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from scipy import stats

from utils import (EYE_COLORS, EYES, IC, IG, blank, calculate_vergence_point, data_path, read_series,
                   save_figure, vergence_distance)

N_DEPTHS = 6
MIN_FIXATION_DURATION = 1.0  # [s] a reference fixation must last this long
FIXATION_START_OFFSET = 0.3  # [s] skipped at the start of the reference fixation
ALPHA = 0.001  # adjusted p below which a pair of depths counts as discriminated
FIG13 = [("p057", "vergence_3"), ("p023", "vergence_3")]  # average and poor discriminability


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


def stimulus_vergence(s, clean):
    """Vergence per stimulus from the mean eye positions and gaze over its reference fixation
    (first 300 ms skipped): the isosceles-triangle distance, and the least-squares vergence
    point, whose z is measured from the marker."""
    for i, r in s.iterrows():
        if np.isnan(r.t_fix_start):
            continue
        gd, ic = {}, {}
        for eye in EYES:
            w = clean[(r.participant, r.recording, eye)].loc[r.t_fix_start_offset:r.t_fix_end]
            g = w[IG].mean()
            gd[eye] = (g / np.linalg.norm(g)).values
            ic[eye] = w[IC].mean().values
        s.loc[i, "vergence_distance"] = vergence_distance(ic["left"], gd["left"], ic["right"], gd["right"])[0]
        s.loc[i, ["vp_x", "vp_y", "vp_z"]] = calculate_vergence_point(ic["left"], -gd["left"],
                                                                      ic["right"], -gd["right"])
    s["vd_mm"] = -s.vp_z  # distance in front of the marker, towards the participant
    return s


def angle_vergence_series(clean, pid, rec):
    left, right = paired_eyes(clean, pid, rec)
    return pd.Series(vergence_distance(left[IC].values, left[IG].values, right[IC].values, right[IG].values),
                     index=right.index)


def discriminability(s, clean):
    """Per run, an independent t-test between the instantaneous vergence distances of each
    two adjacent depths, over their reference fixations; Bonferroni over all valid tests."""
    rows = []
    for (pid, rec), g in s.groupby(["participant", "recording"], sort=False):
        va = angle_vergence_series(clean, pid, rec)
        g = g.set_index("order_from_participant")
        for order in range(1, N_DEPTHS):
            samples = []
            for o in (order, order + 1):
                r = g.loc[o]
                samples.append(va[r.t_fix_start_offset:r.t_fix_end].dropna()
                               if not np.isnan(r.t_fix_start) else va.iloc[:0])
            g1, g2 = samples
            p = stats.ttest_ind(g1, g2).pvalue if len(g1) and len(g2) else np.nan
            rows.append((pid, rec, order, len(g1), len(g2), p))
    d = pd.DataFrame(rows, columns=["participant", "recording", "pair", "n1", "n2", "p"])
    n_valid = d.p.notna().sum()
    d["p_adj"] = np.minimum(d.p * n_valid, 1.0)
    return d


def print_discriminability(d):
    n_valid = d.p.notna().sum()
    print(f"{len(d)} adjacent pairs, {n_valid} testable (Bonferroni factor {n_valid}); "
          f"{(d.p_adj < ALPHA).sum()} ({100 * (d.p_adj < ALPHA).sum() / n_valid:.0f}%) discriminated at "
          f"p_adj < {ALPHA}")
    per_run = d.groupby(["participant", "recording"]).agg(valid=("p", "count"), worst=("p_adj", "max"))
    complete = per_run[per_run.valid == N_DEPTHS - 1]
    print(f"{len(per_run)} runs, {len(complete)} with all {N_DEPTHS - 1} pairs valid; in "
          f"{(complete.worst < ALPHA).sum()} ({100 * (complete.worst < ALPHA).sum() / len(per_run):.0f}% of all runs) "
          f"every adjacent pair is discriminated at p_adj < {ALPHA}")
    n = pd.concat([d.n1, d.n2])
    n = n[n > 0]
    print(f"Samples per reference fixation in the t-tests: {n.min()}-{n.max()}, median {n.median():.0f}")


def decode_depths(df, scope):
    """Nearest-centroid decoding of depth order from one vergence estimate (vd_mm), leave-one-out:
    'within' - centroids from the participant's remaining stimuli; 'cross' - from all other
    participants. Returns per-participant accuracy and the confusion counts [true, decoded]."""
    depths = list(range(1, N_DEPTHS + 1))
    rows, confusion = [], pd.DataFrame(0, index=depths, columns=depths)
    df = df[df.vd_mm.notna()]
    for pid, g in df.groupby("participant"):
        if g.order_from_participant.nunique() < N_DEPTHS:
            continue
        others = None
        if scope == "cross":
            others = df[df.participant != pid].groupby("order_from_participant").vd_mm.mean()
            if len(others) < N_DEPTHS:
                continue
        correct = total = 0
        for i in g.index:
            if scope == "within":
                centroids = g.drop(i).groupby("order_from_participant").vd_mm.mean()
                if len(centroids) < N_DEPTHS:
                    continue  # holding this one out left a depth with no estimate
            else:
                centroids = others
            true = int(g.at[i, "order_from_participant"])
            pred = int((centroids - g.at[i, "vd_mm"]).abs().idxmin())
            confusion.at[true, pred] += 1
            correct += pred == true
            total += 1
        if total:
            rows.append((pid, total, correct / total))
    return pd.DataFrame(rows, columns=["participant", "n", "accuracy"]).set_index("participant"), confusion


def print_decoding(results):
    chance = 1 / N_DEPTHS
    for scope, (acc, confusion) in results.items():
        n = int(confusion.to_numpy().sum())
        off = np.abs(np.subtract.outer(confusion.index, confusion.columns))
        near = confusion.to_numpy()[off <= 1].sum()
        t, p = stats.ttest_1samp(acc.accuracy, chance)
        print(f"Decoding, {scope}: {len(acc)} participants, {n} stimuli; accuracy "
              f"{acc.accuracy.mean():.3f} +/- {acc.accuracy.std():.3f} (mean +/- SD), median {acc.accuracy.median():.3f}; "
              f"{(acc.accuracy > chance).sum()}/{len(acc)} above chance {chance:.3f}; "
              f"t({len(acc) - 1}) = {t:.2f}, p = {p:.2e}; {100 * near / n:.1f}% correct or one depth off")
    (a, _), (b, _) = results.values()
    common = a.index.intersection(b.index)
    t, p = stats.ttest_rel(a.accuracy[common], b.accuracy[common])
    print(f"within vs cross, paired over {len(common)} participants: t({len(common) - 1}) = {t:.2f}, p = {p:.2e}")


# ---------------------------------------------------------------------------- figures


def fig13(s, fixations, raw, clean):
    """Two runs end to end: gaze azimuth and elevation per eye (faint: before blink removal),
    binocular fixations (cyan), instantaneous vergence distance (grey) and each stimulus's
    vergence over its reference fixation (black)."""
    fig, all_axs = plt.subplots(3, 2, figsize=(16, 8))
    for col, (pid, rec) in enumerate(FIG13):
        axs = all_axs[:, col]
        for eye in EYES:
            c = EYE_COLORS[eye]
            for d, kw in ((raw[(pid, rec, eye)], dict(linewidth=0.3, alpha=0.1)), (clean[(pid, rec, eye)], {})):
                axs[0].plot(d.index, d.azimuth_deg, color=c, **kw)
                axs[1].plot(d.index, d.elevation_deg, color=c, **kw)
        for _, f in fixations[(fixations.participant == pid) & (fixations.recording == rec)].iterrows():
            for ax in axs:
                ax.axvspan(f.t_start_sec, f.t_end_sec, color="cyan", alpha=0.2)
        va = angle_vergence_series(clean, pid, rec) / 1000
        axs[2].plot(va.index, va, c="k", alpha=0.3)
        g = s[(s.participant == pid) & (s.recording == rec)].sort_values("t_start_sec")
        for _, r in g.dropna(subset=["vergence_distance"]).iterrows():
            y = r.vergence_distance / 1000
            axs[2].plot((r.t_fix_start_offset, r.t_fix_end), (y, y), color="k")
            axs[2].text(r.t_fix_start_offset, y, f"[{r.symbol}]\n{y:.2f}", ha="right", va="center", weight="bold")
        # scale to the blink-free data; the faint traces' blink artefacts run off the axes
        t0, t1 = g.t_start_sec.iloc[0], g.t_end_sec.iloc[-1]
        for ax, col in zip(axs[:2], ["azimuth_deg", "elevation_deg"]):
            v = pd.concat([clean[(pid, rec, eye)].loc[t0:t1, col] for eye in EYES])
            ax.set_ylim(v.min() - 1, v.max() + 1)
        axs[2].set_ylim(va.loc[t0:t1].quantile(.01) - 0.1, va.loc[t0:t1].quantile(.99) + 0.1)
        for ax in axs:
            ax.set_xlim(t0, t1)
            ax.grid(True)
            for t in list(g.t_start_sec) + [g.t_end_sec.iloc[-1]]:
                ax.axvline(t, color="k", linestyle="--")
        axs[0].set_title(f"{pid} {rec}")
        axs[0].set_ylabel("Azimuth (deg)")
        axs[1].set_ylabel("Elevation (deg)")
        axs[2].set_ylabel("Vergence distance (m)")
        axs[2].set_xlabel("Time (s)")
        axs[2].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.1f}"))
    fig.tight_layout()
    save_figure(fig, "fig13_example_runs")


def fig14(valid):
    fig = plt.figure()
    sns.boxplot(y="vd_mm", x="order_from_participant", color="lightgreen", data=valid,
                fliersize=0, showcaps=False)
    sns.swarmplot(y="vd_mm", x="order_from_participant", color="green", data=valid, size=2)
    m = valid.groupby("order_from_participant").vd_mm.mean()
    sd = valid.groupby("order_from_participant").vd_mm.std()
    for t in m.index:
        plt.text(t - 1, m[t] + 0.1, f"{m[t]:.1f}±{sd[t]:.1f}", ha="center", va="bottom", fontsize=8)
    plt.plot(m.index - 1, m, "k^")
    plt.grid(axis="y")
    plt.ylabel("Vergence Point - Distance to Marker (mm)")
    plt.xlabel("Depth Order from Participant")
    plt.ylim(-200, 800)
    plt.tight_layout()
    save_figure(fig, "fig14_vergence_by_depth")


def fig15(results):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (scope, (acc, confusion)) in zip(axs, results.items()):
        sns.heatmap(confusion.div(confusion.sum(axis=1), axis=0), annot=True, fmt=".2f", cmap="Greens",
                    vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Proportion of true depth"})
        ax.set_title(f"{scope} participant (mean acc. {acc.accuracy.mean():.2f})")
        ax.set_xlabel("Decoded Depth Order from Participant")
        ax.set_ylabel("True Depth Order from Participant")
    fig.tight_layout()
    save_figure(fig, "fig15_decoding_confusion")


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
    valid = s[s.vergence_distance.notna()]
    print(f"{s.participant.nunique()} participants; runs per participant -> participants: {runs.to_dict()}; "
          f"{len(s)} stimuli, {len(valid)} with a reference fixation ({len(s) - len(valid)} without)")
    print("Vergence point in front of the marker by depth order [mm], mean +/- SD:")
    print(valid.groupby("order_from_participant").vd_mm.agg(["mean", "std"]).round(1).to_string())

    print_discriminability(discriminability(s, clean))
    results = {scope: decode_depths(valid, scope) for scope in ("within", "cross")}
    print_decoding(results)

    fig13(s, fixations, raw, clean)
    fig14(valid)
    fig15(results)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
