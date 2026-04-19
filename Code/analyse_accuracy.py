"""

Figures 3 to 7 in the paper
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os

from Code.utils import calculate_eu_percentiles

# %% load all recordings
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

#%% NB: grid1 and grid2 are 7x11 w/o corners (61) fixations; grid3 is 5x5
for i,g in df_all.groupby('grid'):
    print(i, len(g.loc[:, ["stimulus_x_deg", "stimulus_y_deg"]].drop_duplicates()))
#%% Figure 4
pid, eye = 'p038', 'left'
fixations = df_all[(df_all.participant == pid) & (df_all.eye==eye)].copy()
fixations['x_opt'] = fixations.deviation_from_stimulus_x_dva + fixations.stimulus_x_deg - fixations.kappa_x_deg
fixations['y_opt'] = fixations.deviation_from_stimulus_y_dva + fixations.stimulus_y_deg - fixations.kappa_y_deg
_, axs = plt.subplots(2, 1, figsize=(7, 9), sharex=True, sharey=True)

fixations.plot(x='x_opt', y='y_opt', ax=axs[0], marker='o', color='#ff4d00', linestyle='', alpha=0.3, markersize=2, label="Optical axis")
fixations.plot(x='x', y='y', ax=axs[1], marker='o', color='#2ca02c', linestyle='', alpha=0.3, markersize=2, label="Visual axis (Gaze)")


ticks = np.arange(-21, 22, 3)
for ax in axs:
    fixations.plot(x="stimulus_x_deg", y="stimulus_y_deg", ax=ax, marker='o', color='k', linestyle='', alpha=0.3, markersize=2, label="Stimulus")

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid()

    ax.set_xlim(-21, 21)
    ax.set_ylim(-11, 11)

    ax.set_ylabel("Eccentricity vertical (deg)")

axs[1].set_xlabel("Eccentricity horizontal (deg)")
plt.tight_layout()
plt.show()

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

fix = group_all.mean()
fix["accuracy"] = None
fix["precision"] = None
#%% calculate accuracy and precision for each fixation
for (name, pid, grid, eye, x, y), g in group_all:
    print(name, x, y)
    mx = g.deviation_from_stimulus_x_dva.mean()
    my = g.deviation_from_stimulus_y_dva.mean()

    fix.loc[[(name, pid, grid, eye, x, y)], 'accuracy'] = np.sqrt(mx*mx+my*my)
    fix.loc[[(name, pid, grid, eye, x, y)], 'precision'] = ((g.deviation_from_stimulus_x_dva - mx)**2 + (g.deviation_from_stimulus_y_dva - my)**2).apply(np.sqrt).mean()

# %% rearrange into a grid
fix_clean = fix[fix["is_fixation_outlier"] < 0.5]

grid = (
    fix_clean.reset_index().groupby(["stimulus_x_deg", "stimulus_y_deg"])[
        ["precision", "accuracy"]
    ]
    .median()
    .reset_index()
)

# %% Figure 6
for y in ["accuracy", "precision"]:
    plt.figure(figsize=(11, 7), num=y)
    ax = sns.heatmap(
        grid.pivot(columns="stimulus_x_deg", index="stimulus_y_deg", values=y).astype(np.float64),
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

# %% Figure 7
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

