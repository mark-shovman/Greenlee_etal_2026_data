"""Per-participant supplementary table: grid accuracy and precision, luminance change, vergence.

Writes Tables/participants.csv and Tables/kappa_blocks.csv, at full precision. One row per
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
from utils import ROOT, data_path, inverse_weights, kappa_blocks, nan_safe_percentile, weighted_percentile

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
    s, _, _, _ = vs.prepare()
    valid = s[s.t_fix_start.notna() & s.vergence_mm.notna()]  # with a reference fixation and a vergence point
    out = pd.DataFrame({
        "vs_rig": s.groupby("participant").rig.first(),
        "vs_runs": s.groupby("participant").recording.nunique(),
        "vs_stimuli_valid": valid.groupby("participant").size(),
        # least-squares line of the vergence point's distance in front of the marker on depth order
        "vs_step_mm": valid.groupby("participant").apply(
            lambda g: -np.polyfit(g.order_from_participant, g.marker_mm, 1)[0]),
        "vs_error_mm": (valid.vergence_mm - valid.symbol_mm).groupby(valid.participant).median(),
    })
    out["vs_stimuli_valid"] = out.vs_stimuli_valid.fillna(0).astype(int)
    out.index.name = "participant"
    return out, valid


# ---------------------------------------------------------------------------- output


def demographics(index):
    path = data_path("participants.csv")
    if not os.path.exists(path):
        print(f"No {path}: demographic columns left empty")
        return pd.DataFrame(index=index, columns=DEMOGRAPHICS)
    return pd.read_csv(path, index_col="participant")[DEMOGRAPHICS].reindex(index)


def main():
    grid, ue50, n_valid, blocks = grid_columns()
    luminance, n_episode_eyes = luminance_columns()
    vergence, valid = vergence_columns()

    t = demographics(grid.index).join(grid).join(luminance).join(vergence)
    t = t[["in_training_set"] + DEMOGRAPHICS + [c for c in t.columns if c not in DEMOGRAPHICS + ["in_training_set"]]]
    counts = [c for c in t.columns if c.startswith(("n_", "vs_runs", "vs_stimuli", "lum_valid"))]
    t[counts] = t[counts].astype("Int64")  # blank where the experiment was not done, not 0.0

    # the table must add up to the cohort analyses it was split from
    assert t.n_fixations_valid.sum() == n_valid
    assert t.lum_valid_episode_eyes.sum() == n_episode_eyes
    assert t.vs_stimuli_valid.sum() == len(valid)
    print(f"{len(t)} participants: {t.n_fixations.notna().sum()} grid, {t.lum_constriction_mm.notna().sum()} "
          f"with valid luminance episodes, {t.vs_runs.notna().sum()} vergence")
    print("U50 over the per-participant weighted medians (= U50|E50 in Figs 11-12): "
          + ", ".join(f"{k} {v:.3f}" for k, v in ue50.items()))
    print(f"Vergence step per depth median {t.vs_step_mm.median():.1f} mm "
          f"(range {t.vs_step_mm.min():.1f}-{t.vs_step_mm.max():.1f})")

    os.makedirs(TABLES, exist_ok=True)
    t.to_csv(os.path.join(TABLES, "participants.csv"))
    blocks.to_csv(os.path.join(TABLES, "kappa_blocks.csv"), index=False)
    print(f"Wrote {TABLES}")


if __name__ == "__main__":
    main()
