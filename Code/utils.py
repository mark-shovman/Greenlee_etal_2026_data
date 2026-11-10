"""Shared helpers for the analysis scripts: data paths, angles, vergence, U|E percentiles.

Self-contained ports of the functions the paper's analysis used, kept numerically identical so
the scripts reproduce the paper's numbers from the open data.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data")
FIGURES = os.path.join(ROOT, "Figures")

EYES = ["left", "right"]
EYE_COLORS = {"left": "tab:blue", "right": "tab:red"}

# The example timelines (Figs 7, 13, 16) share width, panel height and the time axis
TIMELINE_WIDTH = 12  # [in]
TIMELINE_PANEL_HEIGHT = 2.2  # [in]
IC = [f"ic{x}" for x in "XYZ"]  # limbus centre [mm]
IG = [f"ig{x}" for x in "XYZ"]  # gaze direction, a unit vector pointing *into* the eye


def data_path(*parts):
    return os.path.join(DATA, *parts)


def read_series(path):
    """A per-sample CSV, indexed by its timestamp (seconds on the recording's clock)."""
    return pd.read_csv(path, index_col="timestamp_sec")


def save_figure(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")


def timeline_figure(nrows, ncols=1):
    """Rows of timeline panels, all sharing one time axis; a 2D array of axes."""
    return plt.subplots(nrows, ncols, sharex=True, squeeze=False, layout="constrained",
                        figsize=(TIMELINE_WIDTH, nrows * TIMELINE_PANEL_HEIGHT))


def finish_timeline(axs):
    """Grid on every panel; time on the recording clock, in seconds, under each column."""
    for ax in axs.flat:
        ax.grid(True)
    for ax in axs[-1]:
        ax.set_xlabel("Time (s)")


# ---------------------------------------------------------------------------- angles


def kappa_blocks(df):
    """Kappa per grid block: the median offset over the block's valid fixations, as reported. The
    accuracy numbers use the per-half kappa instead (kappa_x_deg), fitted on the other half."""
    g = df.groupby(["participant", "eye", "recording", "grid"])
    out = g[["kappa_grid_x_deg", "kappa_grid_y_deg"]].first().rename(
        columns={"kappa_grid_x_deg": "kappa_x_deg", "kappa_grid_y_deg": "kappa_y_deg"})
    out["n_fixations_valid"] = g.is_fixation_valid.sum().astype(int)
    return out.reset_index()



def angle_unit(u, v):
    """Angle in degrees between two unit vectors."""
    return np.degrees(np.arccos(np.dot(u, v)))


def unit_mean(gd):
    """Mean of unit vectors (DataFrame rows), normalised to a unit vector."""
    m = gd.mean().values
    return m / np.linalg.norm(m)


def unit_to_az_el(v):
    """Tangent (flat-screen) azimuth and elevation [deg] of direction vectors, shape (n, 3)."""
    v = np.asarray(v)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    az = np.degrees(np.arctan2(x, z))
    el = np.degrees(np.arctan2(y, z))
    degenerate = (x == 0) & (y == 0) & (z == 0)
    return np.where(degenerate, np.nan, az), np.where(degenerate, np.nan, el)


# ---------------------------------------------------------------------------- vergence


def vergence_point(o1, d1, o2, d2, epsilon=1e-9):
    """Vergence point of pairs of gaze rays (origin o, unit direction d; single vectors or (n, 3)
    arrays, one pair per row): the midpoint of the closest points of the two rays, shape (n, 3).
    NaN where the rays are near-parallel or meet behind either origin."""
    o1, d1, o2, d2 = (np.atleast_2d(np.asarray(v, dtype=float)) for v in (o1, d1, o2, d2))
    w0 = o1 - o2
    a, b, c = (d1 * d1).sum(1), (d1 * d2).sum(1), (d2 * d2).sum(1)
    d, e = (d1 * w0).sum(1), (d2 * w0).sum(1)
    denominator = a * c - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (b * e - c * d) / denominator
        t2 = (a * e - b * d) / denominator
    point = 0.5 * (o1 + t1[:, None] * d1 + o2 + t2[:, None] * d2)
    point[(np.abs(denominator) < epsilon) | (t1 < 0) | (t2 < 0)] = np.nan
    return point


# ---------------------------------------------------------------------------- U|E percentiles


def nan_safe_percentile(values, p):
    """np.percentile without NaNs; NaN for an empty input."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return np.percentile(arr, p) if arr.size else np.nan


def weighted_percentile(values, weights, p):
    """Survey-weighted p-th percentile via the weighted empirical CDF with midpoint plotting
    positions (Lumley 2010). At equal weights this is Hyndman & Fan type 5."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[ok], weights[ok]
    if values.size == 0:
        return np.nan
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights)
    return np.interp(p / 100.0, cdf, values)


def ue_percentiles(df, uid_col, data_col, pcts, weights=None):
    """U|E percentiles (Raju et al. 2025): the E-th percentile of data_col within each unit,
    then the U-th percentile of those across units. With `weights` (a column), the E step is
    survey-weighted; the U step is always unweighted. Rows: U, columns: E."""
    rows = []
    for key, g in df.groupby(uid_col):
        for p in pcts:
            e = (nan_safe_percentile(g[data_col], p) if weights is None
                 else weighted_percentile(g[data_col], g[weights], p))
            rows.append((key, p, e))
    e_p = pd.DataFrame(rows, columns=["unit", "E_percentile", data_col])
    u_p = {(u, e): nan_safe_percentile(g[data_col], u)
           for e, g in e_p.groupby("E_percentile") for u in pcts}
    out = pd.Series(u_p).unstack()
    out.index.name, out.columns.name = "U_percentile", "E_percentile"
    return out


def pooled_median(df, metric, weights=None):
    """Median of metric over all fixations; survey-weighted by the `weights` column if given."""
    v = df[metric].to_numpy(dtype=float)
    if weights is not None:
        return weighted_percentile(v, df[weights].to_numpy(dtype=float), 50)
    v = v[np.isfinite(v)]
    return np.median(v) if v.size else np.nan


def inverse_weights(df, metric, cell, min_in_cell, region=None, x="stimulus_head_x_deg",
                    y="stimulus_head_y_deg"):
    """Inverse-density (Horvitz-Thompson post-stratification) weights over FOV cells.

    Keeps the fixations inside `region` (None: all; (w, h): a rectangle of that full extent
    centred on 0), bins them into `cell`-degree cells centred on 0, drops cells holding fewer
    than `min_in_cell` fixations, and weights each fixation by 1 / (fixations in its cell).
    Returns the retained rows with a 'weight' column.
    """
    d = df.dropna(subset=[metric, x, y]).copy()
    if region is not None:
        w, h = region
        d = d[(d[x].abs() <= w / 2) & (d[y].abs() <= h / 2)]
    cx = np.round(d[x] / cell) * cell
    cy = np.round(d[y] / cell) * cell
    counts = d.groupby([cx, cy])[x].transform("size")
    keep = counts >= min_in_cell
    d = d[keep].copy()
    d["weight"] = 1.0 / counts[keep]
    return d


# ---------------------------------------------------------------------------- time series


def align_traces(traces, t_pre, t_post):
    """Per-episode time series on a common grid, shape (grid, traces).

    Each trace is resampled onto the grid by nearest-sample lookup within one sample period, so
    gaps stay gaps (NaN). Returns (grid, aligned).
    """
    traces = [t[~t.index.duplicated()] for t in traces]
    step = float(np.median([np.median(np.diff(t.index)) for t in traces]))
    grid = np.arange(-t_pre, t_post + step, step)
    aligned = np.column_stack([t.reindex(grid, method="nearest", tolerance=step).to_numpy() for t in traces])
    return grid, aligned


def blank(df, mask_col="blink"):
    """The per-sample data with the flagged samples set to NaN (the flag column dropped)."""
    out = df.drop(columns=[mask_col])
    out.loc[df[mask_col].to_numpy(), :] = np.nan
    return out
