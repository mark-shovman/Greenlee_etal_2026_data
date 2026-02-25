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
