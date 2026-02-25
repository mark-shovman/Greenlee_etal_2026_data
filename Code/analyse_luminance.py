import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os

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

