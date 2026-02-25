import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os

from Code.utils import calculate_eu_percentiles

# %%
df_all = None
for fname in os.listdir("Data/1_accuracy"):
    print(f"  {fname}")
    pid, grid, eye = fname.split(".")[0].split("_")
    df = pd.read_csv(f"Data/1_accuracy/{fname}")
    df["experiment"] = fname.split(".")[0]
    df["participant"] = pid
    df["grid"] = grid
    df["eye"] = eye
    df_all = pd.concat((df_all,df)) if df_all is not None else df

df_all["x"] = df_all.stimulus_x_deg + df_all.deviation_from_stimulus_x_dva
df_all["y"] = df_all.stimulus_y_deg + df_all.deviation_from_stimulus_y_dva
# %% aggregate by fixations
group_all = df_all.groupby(
        [
            "experiment",
            "participant",
            "grid",
            "eye",
            "stimulus_x_deg",
            "stimulus_y_deg",
        ]
    )

fix = group_all.mean().reset_index()

# TODO
fix["accuracy"] = None
fix["precision"] = None

fix
# %% remove outliers
fix_clean = fix[fix["is_fixation_outlier"] < 0.5]
fix_clean

# %%
grid = (
    fix_clean.groupby(["stimulus_x_deg", "stimulus_y_deg"])[
        ["precision", "accuracy"]
    ]
    .median()
    .reset_index()
)
grid

# %%
for y in ["accuracy", "precision"]:
    plt.figure(figsize=(11, 7), num=y)
    ax = sns.heatmap(
        grid.pivot(columns="stimulus_x_deg", index="stimulus_y_deg", values=y),
        annot=True,
        fmt=".2f",
        square=True,
        cbar=False,
        cmap="viridis",
        vmin=0.25 if y == "accuracy" else 0.12,
        vmax=1.33 if y == "accuracy" else 0.29,
    )
    ax.set_xlabel("X (deg)")
    ax.set_ylabel("Y (deg)")
    ax.set_title(y.capitalize())
    ax.invert_yaxis()
    plt.tight_layout()

# %%
for y in ["accuracy", "precision"]:
    eu = calculate_eu_percentiles(
        fix_clean,
        "participant",
        data_col=y,
        e_percentiles=[10, 25, 50, 75, 90],
        u_percentiles=[10, 25, 50, 75, 90],
    )

    plt.figure(figsize=(3.5, 3.5), num=y + " E|U")
    ax = sns.heatmap(eu, annot=True, fmt=".2f", square=True, cbar=False, cmap="flare_r")
    ax.set_title(y.capitalize())
    ax.set_xlabel("E percentile")
    ax.set_ylabel("U percentile")
    plt.tight_layout()

plt.show()

