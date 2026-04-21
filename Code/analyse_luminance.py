"""
Figures 8 and 9 in the paper
"""

import os
import pandas as pd

import matplotlib.pyplot as plt

import Code.utils as utl
# %%
df_all = None
for fname in os.listdir("Data/2_luminance"):
    print(f"  {fname}")
    pid, eye = fname.split(".")[0].split("_")
    df = pd.read_csv(f"Data/2_luminance/{fname}")
    df["experiment"] = fname.split(".")[0]
    df["participant"] = pid
    df["eye"] = eye
    df_all = pd.concat((df_all,df)) if df_all is not None else df

#%% Figure 8
pid = "p001"
d_l = df_all[(df_all.participant == pid) & (df_all.eye == "left")].copy().set_index('timestamp_sec')
d_r = df_all[(df_all.participant == pid) & (df_all.eye == "right")].copy().set_index('timestamp_sec')

title = f"Participant {pid}"
fig, axs = plt.subplots(3, 1, sharex=True, num=title)
d_l[['stimulus_x_deg', 'stimulus_y_deg']].plot(subplots=True, color="k", ax=axs[:2], label="Stimulus")
axs[0].set_title(title)

d_l[["x_dva", "y_dva"]].plot(ax=axs[:2], subplots=True, color="b", label="Left eye")
d_r[["x_dva", "y_dva"]].plot(ax=axs[:2], subplots=True, color="r", label="Right eye")

d_l["pupil_diameter_mm"].sort_index().plot(ax=axs[2], color="b", label="Left eye")
d_r["pupil_diameter_mm"].sort_index().plot(ax=axs[2], color="r", label="Right eye")

axs[0].set_ylabel("Azimuth (deg)")
axs[1].set_ylabel("Elevation (deg)")
axs[2].set_ylabel("Pupil diameter (mm)")
axs[-1].set_xlabel("Time (sec)")

for ax in axs:
    ax.grid()
    for bc in utl.collate_series(d_l.luminance):
        ax.axvspan(bc[0], bc[1], fc="k" if bc[2]=='dark' else 'w', alpha=0.5)

#%% Figure 9
fig, axs = plt.subplots(2, 1, sharex=True, num="Effect of BC on PD")
axs[0].set_title("Effect of BC on PD")
for ax in axs:
    ax.axvspan(-2, 0, fc="k", alpha=0.5)

x = None
acc = None
count = 0

t_pre = 2
t_post = 2

for (pid, eye), g in df_all.groupby(["participant", "eye"]):
    d = g[['timestamp_sec', 'pupil_diameter_mm', 'luminance']].copy().set_index('timestamp_sec', drop=True)
    for bc in utl.collate_series(d.luminance):
        if bc[2] == 'dark':
            continue
        t0 = bc[0]
        c = "r" if eye == "right" else "b"
        pupil = d.pupil_diameter_mm[t0 - t_pre : t0 + t_post].copy()
        pupil.index -= t0
        pupil.plot(ax=axs[0], color=c, legend=False, alpha=0.5, linewidth=0.3)
        baseline = pupil.loc[-1:0].mean()
        pupil_change = (pupil - baseline).rolling(5, center=True).mean()
        pupil_change.plot(
            ax=axs[1], color=c, legend=False, alpha=0.5, linewidth=0.3
        )
axs[0].set_ylabel("Pupil diameter (mm)")
axs[1].set_ylabel("Pupil size change (mm)")

axs[1].set_xlabel("Time (msec)")

for ax in axs:
    ax.grid(True)
    ax.set_xlim(-2, 2)