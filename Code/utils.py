"""Shared helpers for the analysis scripts: data paths, angles, vergence, U|E percentiles.

Self-contained ports of the functions the paper's analysis used, kept numerically identical so
the scripts reproduce the paper's numbers from the open data.
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data")
FIGURES = os.path.join(ROOT, "Figures")

EYES = ["left", "right"]
EYE_COLORS = {"left": "b", "right": "r"}
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


# ---------------------------------------------------------------------------- angles


def angle_unit(u, v):
    """Angle in degrees between two unit vectors."""
    return np.degrees(np.arccos(np.dot(u, v)))


def unit_mean(gd):
    """Mean of unit vectors (DataFrame rows), normalised to a unit vector."""
    m = gd.mean().values
    return m / np.linalg.norm(m)


def angle_deviation(gd):
    """Mean angular deviation [deg] of the rows of gd from their mean direction."""
    m = unit_mean(gd)
    return np.nanmean([angle_unit(gd.iloc[i].values, m) for i in range(len(gd))])


def unit_to_az_el(v):
    """Tangent (flat-screen) azimuth and elevation [deg] of direction vectors, shape (n, 3)."""
    v = np.asarray(v)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    az = np.degrees(np.arctan2(x, z))
    el = np.degrees(np.arctan2(y, z))
    degenerate = (x == 0) & (y == 0) & (z == 0)
    return np.where(degenerate, np.nan, az), np.where(degenerate, np.nan, el)


# ---------------------------------------------------------------------------- vergence


def calculate_vergence_point(o1, d1, o2, d2, epsilon=1e-9, bias=0.5):
    """Vergence point of two gaze rays (origin o, unit direction d): the midpoint of their
    closest points. NaN when the rays are near-parallel or meet behind either origin."""
    w0 = o1 - o2
    a, b, c = np.dot(d1, d1), np.dot(d1, d2), np.dot(d2, d2)
    d, e = np.dot(d1, w0), np.dot(d2, w0)
    denominator = a * c - b * b
    if abs(denominator) < epsilon:
        return np.array([np.nan, np.nan, np.nan])
    t1 = (b * e - c * d) / denominator
    t2 = (a * e - b * d) / denominator
    if t1 < 0 or t2 < 0:
        return np.array([np.nan, np.nan, np.nan])
    return (o1 + t1 * d1) * bias + (o2 + t2 * d2) * (1.0 - bias)


def vergence_distance(ic_left, ig_left, ic_right, ig_right):
    """Distance to the vergence point from the angle between the gaze rays and the
    inter-limbus baseline, treating the rays as an isosceles triangle. Works per row on
    (n, 3) arrays or on single vectors."""
    ig_left, ig_right = np.atleast_2d(ig_left), np.atleast_2d(ig_right)
    cos_a = np.clip((ig_left * ig_right).sum(axis=1)
                    / np.linalg.norm(ig_left, axis=1) / np.linalg.norm(ig_right, axis=1), -1.0, 1.0)
    base = np.linalg.norm(np.atleast_2d(ic_left) - np.atleast_2d(ic_right), axis=1)
    return base / 2 / np.sqrt((1 - cos_a) / (1 + cos_a))


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


def mean_trace(traces, t_pre, t_post):
    """NaN-aware mean of per-episode time series on a common grid.

    Each trace is resampled onto the grid by nearest-sample lookup within one sample period, so
    gaps stay gaps, then averaged column-wise skipping NaN. Returns (grid, mean, coverage).
    """
    traces = [t[~t.index.duplicated()] for t in traces]
    step = float(np.median([np.median(np.diff(t.index)) for t in traces]))
    grid = np.arange(-t_pre, t_post + step, step)
    aligned = pd.concat([t.reindex(grid, method="nearest", tolerance=step) for t in traces], axis=1)
    return grid, aligned.mean(axis=1), aligned.count(axis=1)


def blank(df, mask_col="blink"):
    """The per-sample data with the flagged samples set to NaN (the flag column dropped)."""
    out = df.drop(columns=[mask_col])
    out.loc[df[mask_col].to_numpy(), :] = np.nan
    return out
