"""
Figures 12 and 13 in the paper
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os

from Code.utils import calculate_vergence_point, unit_to_az_el, collate_series

# %%
df_list = []
for fname in os.listdir("Data/3_vergence"):
    print(f"  {fname}")
    pid, _, session = fname.split(".")[0].split("_")
    df = pd.read_csv(f"Data/3_vergence/{fname}")
    df["experiment"] = fname.split(".")[0]
    df["participant"] = pid
    df["session"] = session
    df_list.append(df)

df_all = pd.concat(df_list, ignore_index=True)


#%%

vp, div = calculate_vergence_point(df_all[[f'ic{x}_left' for x in 'XYZ']].values,
                                   -df_all[[f'ig{x}_left' for x in 'XYZ']].values,
                                   df_all[[f'ic{x}_right' for x in 'XYZ']].values,
                                   -df_all[[f'ig{x}_right' for x in 'XYZ']].values)

df_all[['vergence_point_x', 'vergence_point_y', 'vergence_point_z']] = vp
df_all["divergence"] = div

df_all['x_left'], df_all['y_left'] = unit_to_az_el(-df_all[[f'ig{x}_left' for x in 'XYZ']].values, convention='fick')
df_all['x_right'], df_all['y_right'] = unit_to_az_el(-df_all[[f'ig{x}_right' for x in 'XYZ']].values, convention='fick')

#%%
experiment = "p010_vergence_2"

df = df_all[df_all["experiment"] == experiment].copy().set_index("timestamp_sec", drop=True)

fixations = collate_series(df.reference_fixation_id)
#%% Figure 12
fig, axs = plt.subplots(3, 1, sharex=True, num=experiment)

for eye in ["left", "right"]:
    c = "b" if eye == "left" else "r"
    df[[f'x_{eye}', f'y_{eye}']].plot(ax=axs[:2], subplots=True, color=c, legend=False)

for t_start, t_end, s in collate_series(df.stimulus_order_from_viewers):
    if pd.isna(s):
        continue
    for ax in axs:
        ax.axvline(t_start, linestyle=":", color="k")
        ax.axvline(t_end, linestyle=":", color="k")

(-df["vergence_point_z"]).plot(ax=axs[-1], c="k")

for t_start, t_end, s in collate_series(df.reference_fixation_id):
    if pd.isna(s):
        continue
    for ax in axs:
        ax.axvspan(t_start, t_end, color="cyan")
        # ax.axvline(t_start, linestyle=":", color="k", alpha=0.5)
        # ax.axvline(t_end, linestyle=":", color="k", alpha=0.5)
    y = (-df.loc[t_start:t_end, "vergence_point_z"]).mean()
    axs[2].plot((t_start, t_end), (y, y), color="k")
    axs[2].text(
        t_start, y, f"[{s}]\n{y:.2f}", ha="right", va="center", weight="bold"
    )

axs[0].set_ylabel("Azimuth (deg)")
axs[1].set_ylabel("Elevation (deg)")
axs[2].set_ylabel("VP distance from head (mm)")

axs[-1].set_xlabel("Time (sec)")

for ax in axs:
    ax.grid()

#%% Figure 13
fix = pd.DataFrame(-(df_all[~df_all.reference_fixation_id.isna()].groupby(["stimulus_order_from_viewers", "experiment"]).vergence_point_z.mean()))

plt.figure()
sns.boxplot(
    y="vergence_point_z",
    x="stimulus_order_from_viewers",
    color="lightgreen",
    legend=False,
    data=fix,
    log_scale=False,
    fliersize=0,
    capwidths=0,
)
sns.swarmplot(
    y="vergence_point_z",
    x="stimulus_order_from_viewers",
    color="green",
    legend=False,
    data=fix,
    size=2,
)
m = fix.groupby("stimulus_order_from_viewers")["vergence_point_z"].mean()
sd = fix.groupby("stimulus_order_from_viewers")["vergence_point_z"].std()
txt = {t: f"{m[t]:.2f}±{sd[t]:.2f}" for t in m.index}
for t in m.index:
    plt.text(
        t - 1,
        m[t] + 0.1,
        f"{m[t]:.2f}±{sd[t]:.2f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
plt.plot(m.index - 1, m, "k^")

plt.grid(axis="y")
plt.xlabel("")
plt.ylabel("Vergence Point - Distance to Marker (mm)")
plt.xlabel("Depth Order from Participant")
plt.ylim(-200, 800)
# plt.title(title)
plt.tight_layout()

