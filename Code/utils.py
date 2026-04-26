import numpy as np
import pandas as pd

def calculate_eu_percentiles(
    df, uid_col, data_col, e_percentiles=None, u_percentiles=None
):
    """
    [...]“E” represents individual-level error percentiles, indicating the distribution of spatial accuracy for each individual.
    “U” represents user percentiles, which describe population-level spatial accuracy by aggregating spatial accuracy across different individual error percentiles.
    Specifically, E50 and E95 indicate the distribution of spatial accuracy for an individual user, where E50 represents
    the median error and E95 represents the worst-case error with the 95th percentile of error.
    On the other hand, U50, and U95 represent population-level insights aggregated across the respective E percentiles of individual users.
    Thus, U50—E50 reflects the median value of E50 spatial accuracy values observed across the population during the trigger event.
    Finally, U95—E95 describes the 95th percentile of E95 error observed across the population in their worst-case interaction scenarios.

    Raju, M. H., Aziz, S., Proulx, M. J., & Komogortsev, O. (2025, May).
    Evaluating Eye Tracking Signal Quality with Real-time Gaze Interaction Simulation: A Study Using an Offline Dataset.
    In Proceedings of the 2025 Symposium on Eye Tracking Research and Applications (pp. 1-11).
    """

    if e_percentiles is None:
        e_percentiles = [10, 25, 50, 75, 90]
    if u_percentiles is None:
        u_percentiles = [10, 25, 50, 75, 90]

    # Group by user and calculate metric percentiles for each user
    ep = (
        df.groupby(uid_col)[data_col]
        .apply(lambda x: pd.Series({p: np.percentile(x, p) for p in e_percentiles}))
        .copy()
        .reset_index()
    )
    ep.columns = [uid_col, "E_percentile", data_col]

    # Calculate U percentiles for each E percentile
    up = (
        ep.groupby("E_percentile")[data_col]
        .apply(
            lambda x: pd.Series(
                {p: np.percentile(x.dropna(), p) for p in u_percentiles}
            )
        )
        .reset_index()
    )
    up.rename(columns={"level_1": "U_percentile"}, inplace=True)
    return up.pivot(index="U_percentile", columns="E_percentile", values=data_col)

def collate_series(s, min_duration=0):
    """
    Collates a series of discrete values into a list of continuous intervals

    Parameters:
    - s: a series of discrete values to be collated
    - min_duration: the minimum duration for each interval

    Returns:
    - list of tuples (t_start, t_end, value);

    NB t_start and t_end are set mid-way between index values
        except for the beginning of the first interval and the end of the last one which are set on the index value

    """

    def values_equal(v1, v2):
        if pd.isna(v1) and pd.isna(v2):
            return True
        return v1 == v2

    intervals = []
    t_start = s.index[0]
    current_value = s.iloc[0]

    for i in range(1, len(s)):
        if not values_equal(s.iloc[i], current_value):
            t_end = (s.index[i - 1] + s.index[i]) / 2
            if t_end - t_start >= min_duration:
                intervals.append([t_start, t_end, current_value])
                t_start = t_end
                current_value = s.iloc[i]
            else:
                if intervals:
                    if values_equal(intervals[-1][2], s.iloc[i]):
                        t_start = intervals[-1][0]
                        current_value = intervals[-1][2]
                        del intervals[-1]
                    else:
                        intervals[-1][1] = t_end
                        t_start = t_end
                        current_value = s.iloc[i]

    # Add the last stretch
    if s.index[-1] - t_start >= min_duration:
        intervals.append([t_start, s.index[-1], current_value])
    elif intervals:
        # Extend the previous stretch to the end
        intervals[-1][1] = s.index[-1]

    return intervals


def calculate_vergence_point(o1, d1, o2, d2, epsilon=1e-9, bias=0.5):
    """
    Vectorized version of calculate_vergence_point.
    Inputs are (N, 3) arrays.
    """
    w0 = o1 - o2  # (N, 3)

    a = np.sum(d1 * d1, axis=1)  # (N,)
    b = np.sum(d1 * d2, axis=1)  # (N,)
    c = np.sum(d2 * d2, axis=1)  # (N,)
    d = np.sum(d1 * w0, axis=1)  # (N,)
    e = np.sum(d2 * w0, axis=1)  # (N,)

    denominator = a * c - b * b  # (N,)

    # Initialize results
    vergence_point = np.full((len(o1), 3), np.nan)
    divergence = np.full(len(o1), np.inf)

    # Mask for non-parallel rays
    valid_mask = np.abs(denominator) >= epsilon

    t1 = np.full(len(o1), np.nan)
    t2 = np.full(len(o1), np.nan)

    t1[valid_mask] = (b[valid_mask] * e[valid_mask] - c[valid_mask] * d[valid_mask]) / denominator[valid_mask]
    t2[valid_mask] = (a[valid_mask] * e[valid_mask] - b[valid_mask] * d[valid_mask]) / denominator[valid_mask]

    # Mask for rays pointing towards each other (t1 > 0 and t2 > 0)
    convergence_mask = valid_mask & (t1 >= 0) & (t2 >= 0)

    p1_closest = o1[convergence_mask] + t1[convergence_mask, np.newaxis] * d1[convergence_mask]
    p2_closest = o2[convergence_mask] + t2[convergence_mask, np.newaxis] * d2[convergence_mask]

    vergence_point[convergence_mask] = p1_closest * bias + p2_closest * (1.0 - bias)
    divergence[convergence_mask] = np.linalg.norm(p1_closest - p2_closest, axis=1)

    return vergence_point, divergence


def unit_to_az_el(v, convention='fick'):
    """
    Convert 3D cartesian gaze vectors to angular coordinates (azimuth/elevation).

    Parameters
    ----------
    v : array-like, shape (n, 3)
        Gaze vectors with columns [x, y, z].
    convention : str, default='fick'
        'fick' or 'spherical'

    Returns
    -------
    azimuth, elevation : tuple of arrays in degrees
    """
    x, y, z = v[:, 0], v[:, 1], v[:, 2]

    if convention == 'fick':
        az = np.degrees(np.arctan2(x, z))
        el = np.degrees(np.arctan2(y, z))
    elif convention == 'spherical':
        hyp = np.sqrt(x**2 + z**2)
        az = np.degrees(np.arctan2(x, z))
        el = np.degrees(np.arctan2(y, hyp))
    else:
        raise ValueError(f"Unknown convention '{convention}'. Use either 'fick' (default) or 'spherical'.")

    return az, el