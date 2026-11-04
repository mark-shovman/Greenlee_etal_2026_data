"""Per-participant supplementary table: grid accuracy and precision, luminance change, vergence.

Writes Tables/participants.csv (full precision) and Tables/participants.md (rounded). One row per
participant; cells are empty where a participant did not do that experiment. Demographics come from
Data/participants.csv when it is present; refraction is empty where it was not recorded.

Run from anywhere:  python Code/participant_table.py
"""

import os

import numpy as np
import pandas as pd

import analyse_luminance as lum
import analyse_vergence as vs
from analyse_accuracy import CELL, CENTRAL, METRICS, MIN_IN_CELL
from utils import ROOT, data_path, inverse_weights, nan_safe_percentile, weighted_percentile

TABLES = os.path.join(ROOT, "Tables")
N_BOOT = 2000
SEED = 2026
DEMOGRAPHICS = ["age_group", "sex", "refraction_left_D", "refraction_right_D"]
SHORT = {"accuracy_centroid_deg": "accuracy", "precision_rms_s2s_deg": "precision"}
REGIONS = {"whole": None, "central": CENTRAL}


def boot_ci(values, weights, rng):
    """95% percentile CI of the weighted median, resampling fixations with replacement; each
    fixation keeps its cohort weight."""
    values, weights = np.asarray(values), np.asarray(weights)
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    meds = [weighted_percentile(values[i], weights[i], 50) for i in idx]
    return np.percentile(meds, [2.5, 97.5])


# ---------------------------------------------------------------------------- grid


def grid_columns():
    df = pd.read_csv(data_path("1_accuracy", "fixations.csv"))
    edata = df[df.is_fixation_valid == 1]
    out = pd.DataFrame(index=pd.Index(sorted(df.participant.unique()), name="participant"))
    out["in_training_set"] = df.groupby("participant").participant_in_training_set.first()

    blocks = kappa_blocks(df)
    kappa = blocks.groupby(["participant", "eye"])[["kappa_x_deg", "kappa_y_deg"]].median().unstack("eye")
    for eye in ("left", "right"):
        for ax in ("x", "y"):
            out[f"kappa_{eye}_{ax}_deg"] = kappa[(f"kappa_{ax}_deg", eye)]

    out["n_fixations"] = df.groupby("participant").size()
    out["n_fixations_valid"] = edata.groupby("participant").size()

    rng = np.random.default_rng(SEED)
    ue50 = {}
    for metric, short in SHORT.items():
        out[f"{short}_median_deg"] = edata.groupby("participant")[metric].median()
        for region, bounds in REGIONS.items():
            w = inverse_weights(edata, metric, CELL, MIN_IN_CELL, bounds)
            col = f"{short}_{region}"
            out[f"n_{col}"] = w.groupby("participant").size()
            rows = {}
            for pid, g in w.groupby("participant"):
                lo, hi = boot_ci(g[metric], g.weight, rng)
                rows[pid] = (weighted_percentile(g[metric], g.weight, 50), lo, hi)
            est = pd.DataFrame.from_dict(rows, orient="index", columns=[f"{col}_deg", f"{col}_ci_low", f"{col}_ci_high"])
            out = out.join(est)
            ue50[col] = nan_safe_percentile(out[f"{col}_deg"], 50)
    for col in ("n_fixations_valid", "n_accuracy_whole", "n_accuracy_central",
                "n_precision_whole", "n_precision_central"):
        out[col] = out[col].fillna(0).astype(int)
    return out, ue50, len(edata), blocks


def kappa_blocks(df):
    """Kappa per grid block: the median offset over the block's valid fixations, as reported. The
    accuracy numbers use the per-half kappa instead (kappa_x_deg), fitted on the other half."""
    g = df.groupby(["participant", "eye", "recording", "grid"])
    out = g[["kappa_grid_x_deg", "kappa_grid_y_deg"]].first().rename(
        columns={"kappa_grid_x_deg": "kappa_x_deg", "kappa_grid_y_deg": "kappa_y_deg"})
    out["n_fixations_valid"] = g.is_fixation_valid.sum().astype(int)
    return out.reset_index()


# ---------------------------------------------------------------------------- luminance


def luminance_columns():
    protocol, events, series = lum.load()
    ep = lum.episodes(protocol)
    clean = lum.clean_episodes(lum.episode_metrics(ep, events, series))
    clean["constriction"] = clean.pupil_before - clean.pupil_after
    g = clean.groupby("participant")
    out = pd.DataFrame({
        "lum_valid_episode_eyes": g.size(),
        "lum_constriction_mm": g.constriction.mean(),
        "lum_gaze_shift_deg": g["shift"].mean(),
    }).reindex(sorted(protocol.participant.unique()))
    out["lum_valid_episode_eyes"] = out.lum_valid_episode_eyes.fillna(0).astype(int)
    out.index.name = "participant"
    return out, len(clean)


# ---------------------------------------------------------------------------- vergence


def vergence_columns():
    s, _, _, clean = vs.prepare()
    valid = s[s.vergence_distance.notna()]
    d = vs.discriminability(s, clean)
    out = pd.DataFrame({
        "vs_runs": s.groupby("participant").recording.nunique(),
        "vs_stimuli_valid": valid.groupby("participant").size(),
        "vs_pairs_discriminated": d.groupby("participant").p_adj.agg(lambda p: int((p < vs.ALPHA).sum())),
        "vs_pairs_testable": d.groupby("participant").p.count(),
    })
    for scope in ("within", "cross"):
        acc, _ = vs.decode_depths(valid, scope)
        out[f"vs_decoding_{scope}"] = acc.accuracy
    out["vs_stimuli_valid"] = out.vs_stimuli_valid.fillna(0).astype(int)
    out.index.name = "participant"
    return out, d, valid


# ---------------------------------------------------------------------------- output


def demographics(index):
    path = data_path("participants.csv")
    if not os.path.exists(path):
        print(f"No {path}: demographic columns left empty")
        return pd.DataFrame(index=index, columns=DEMOGRAPHICS)
    return pd.read_csv(path, index_col="participant")[DEMOGRAPHICS].reindex(index)


def fmt(v, dp):
    return "" if pd.isna(v) else f"{v:.{dp}f}"


def ci(r, col, dp):
    if pd.isna(r[f"{col}_deg"]):
        return ""
    return f"{r[f'{col}_deg']:.{dp}f} [{r[f'{col}_ci_low']:.{dp}f}, {r[f'{col}_ci_high']:.{dp}f}]"


def as_int(v):
    return "" if pd.isna(v) else f"{int(v)}"


def markdown(t):
    """The rounded table: degrees of accuracy and kappa to 2 dp, precision to 3 dp, refraction
    to 2 dp (it comes in 0.25 D steps)."""
    cols = {
        "Participant": lambda r: r.name,
        "Training": lambda r: "yes" if r.in_training_set else "no",
        "Age": lambda r: "" if pd.isna(r.age_group) else r.age_group,
        "Sex": lambda r: "" if pd.isna(r.sex) else r.sex,
        "Refraction L / R (D)": lambda r: "" if pd.isna(r.refraction_left_D) else
        f"{r.refraction_left_D:+.2f} / {r.refraction_right_D:+.2f}",
        "Kappa L x, y (°)": lambda r: f"{r.kappa_left_x_deg:.2f}, {r.kappa_left_y_deg:.2f}",
        "Kappa R x, y (°)": lambda r: f"{r.kappa_right_x_deg:.2f}, {r.kappa_right_y_deg:.2f}",
        "Fixations valid / all": lambda r: f"{r.n_fixations_valid} / {r.n_fixations}",
        "Accuracy, all FOV (°)": lambda r: ci(r, "accuracy_whole", 2),
        "Accuracy, central (°)": lambda r: ci(r, "accuracy_central", 2),
        "Precision, all FOV (°)": lambda r: ci(r, "precision_whole", 3),
        "Precision, central (°)": lambda r: ci(r, "precision_central", 3),
        "Luminance episode-eyes": lambda r: as_int(r.lum_valid_episode_eyes),
        "Constriction (mm)": lambda r: fmt(r.lum_constriction_mm, 2),
        "Gaze shift (°)": lambda r: fmt(r.lum_gaze_shift_deg, 2),
        "Vergence runs": lambda r: as_int(r.vs_runs),
        "Decoding, within": lambda r: fmt(r.vs_decoding_within, 2),
        "Decoding, cross": lambda r: fmt(r.vs_decoding_cross, 2),
        "Pairs discriminated / tested": lambda r: "" if pd.isna(r.vs_runs) else
        f"{int(r.vs_pairs_discriminated)} / {int(r.vs_pairs_testable)}",
    }
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in t.iterrows():
        lines.append("| " + " | ".join(str(f(r)) for f in cols.values()) + " |")
    return "\n".join(lines) + "\n"


def blocks_markdown(b):
    """The per-block kappa table, kappa to 2 dp."""
    lines = ["| Participant | Eye | Recording | Grid | Kappa x (°) | Kappa y (°) | Fixations valid |",
             "|---|---|---|---|---|---|---|"]
    for _, r in b.iterrows():
        lines.append(f"| {r.participant} | {r.eye} | {r.recording} | {r.grid} | {fmt(r.kappa_x_deg, 2)} | "
                     f"{fmt(r.kappa_y_deg, 2)} | {r.n_fixations_valid} |")
    return "\n".join(lines) + "\n"


def main():
    grid, ue50, n_valid, blocks = grid_columns()
    luminance, n_episode_eyes = luminance_columns()
    vergence, pairs, valid = vergence_columns()

    t = demographics(grid.index).join(grid).join(luminance).join(vergence)
    t = t[["in_training_set"] + DEMOGRAPHICS + [c for c in t.columns if c not in DEMOGRAPHICS + ["in_training_set"]]]
    counts = [c for c in t.columns if c.startswith(("n_", "vs_runs", "vs_stimuli", "vs_pairs", "lum_valid"))]
    t[counts] = t[counts].astype("Int64")  # blank where the experiment was not done, not 0.0

    # the table must add up to the cohort analyses it was split from
    assert t.n_fixations_valid.sum() == n_valid
    assert t.lum_valid_episode_eyes.sum() == n_episode_eyes
    assert t.vs_stimuli_valid.sum() == len(valid)
    assert t.vs_pairs_discriminated.sum() == (pairs.p_adj < vs.ALPHA).sum()
    assert t.vs_pairs_testable.sum() == pairs.p.notna().sum()
    print(f"{len(t)} participants: {t.n_fixations.notna().sum()} grid, {t.lum_constriction_mm.notna().sum()} "
          f"with valid luminance episodes, {t.vs_runs.notna().sum()} vergence")
    print("U50 over the per-participant weighted medians (= U50|E50 in Figs 11-12): "
          + ", ".join(f"{k} {v:.3f}" for k, v in ue50.items()))
    print(f"Decoding mean: within {t.vs_decoding_within.mean():.3f}, cross {t.vs_decoding_cross.mean():.3f}; "
          f"pairs discriminated {t.vs_pairs_discriminated.sum():.0f}/{t.vs_pairs_testable.sum():.0f}")

    os.makedirs(TABLES, exist_ok=True)
    t.to_csv(os.path.join(TABLES, "participants.csv"))
    with open(os.path.join(TABLES, "participants.md"), "w", encoding="utf-8") as f:
        f.write(markdown(t))
    blocks.to_csv(os.path.join(TABLES, "kappa_blocks.csv"), index=False)
    with open(os.path.join(TABLES, "kappa_blocks.md"), "w", encoding="utf-8") as f:
        f.write(blocks_markdown(blocks))
    print(f"Wrote {TABLES}")


if __name__ == "__main__":
    main()
