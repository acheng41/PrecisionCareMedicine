from matplotlib.animation import FuncAnimation
import pandas as pd

import numpy as np

from pathlib import Path
import os

import scipy.io as sio

import matplotlib.pyplot as plt

import torch

from matplotlib.pyplot import figure

from torch.distributed.tensor import zeros

from torch.utils.data import Dataset, DataLoader

from getParameters import get_gait_parameters_insole

from getParameters import gait_aligned_jnt

from visualization import plot_cop_and_gait, plot_heatmap_with_COP

from visualization import plot_heatmap_spatial

from matplotlib.animation import FuncAnimation

import seaborn as sns

import matplotlib

matplotlib.use("TkAgg")


data = sio.loadmat("data/gait_recording_102324_walk7.mat")

insoleAll_l = data["insoleAll_l"].astype(np.float64)

insoleAll_r = data["insoleAll_r"].astype(np.float64)

t_insole_l = data["t_insole_l"].astype(np.float64)

t_insole_r = data["t_insole_r"].astype(np.float64)


t_trackers = data["t_trackers"].astype(np.float64)

jnt_angles_all_l = np.array(data["jnt_angles_all_l"])

jnt_angles_all_r = np.array(data["jnt_angles_all_r"])

jnt_pos_all_l = np.array(data["jnt_pos_all_l"])

jnt_pos_all_r = np.array(data["jnt_pos_all_r"])


# %% plot COP & heat map from 50 steps

gait = get_gait_parameters_insole(insoleAll_r, insoleAll_l, t_insole_r, t_insole_l)

p_r = np.array(gait["insole_r"][gait["strike_r"][20] : gait["strike_r"][70], :])

average_pressure_r = np.mean(p_r, axis=0)

heat_map_r = average_pressure_r.reshape([64, 16], order="F")


p_l = np.array(gait["insole_l"][gait["strike_l"][20] : gait["strike_l"][70], :])

average_pressure_l = np.mean(p_l, axis=0)

heat_map_l = average_pressure_l.reshape([64, 16], order="F")

heat_map_l[:32, :] = np.flipud(heat_map_l[:32, :])


fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))

cax_l = ax_l.imshow(heat_map_l, vmin=np.min(heat_map_l), vmax=np.max(heat_map_l))

cax_r = ax_r.imshow(heat_map_r, vmin=np.min(heat_map_r), vmax=np.max(heat_map_r))

ax_l.set_title("Left Heatmap")

ax_r.set_title("Right Heatmap")

cbar_l = fig.colorbar(cax_l, ax=ax_l, orientation="vertical")

cbar_r = fig.colorbar(cax_r, ax=ax_r, orientation="vertical")


# COP

for i in range(50):
    ax_l.plot(
        gait["cop_y_l"][gait["strike_l"][20 + i] : gait["off_l"][20 + i]],
        gait["cop_x_l"][gait["strike_l"][20 + i] : gait["off_l"][20 + i]],
    )

    ax_r.plot(
        gait["cop_y_r"][gait["strike_r"][20 + i] : gait["off_r"][20 + i]],
        gait["cop_x_r"][gait["strike_r"][20 + i] : gait["off_r"][20 + i]],
    )


plt.tight_layout()

plt.show()


# %% compare some feature

t_split = 15 * 100

FM = np.zeros((48, 6))

for n, i in enumerate([1, 2, 4, 7, 8, 9]):
    file_name = "data/gait_recording_102324_walk" + str(i) + ".mat"
    data = sio.loadmat(file_name)

    insoleAll_l = data["insoleAll_l"].astype(np.float64)

    insoleAll_r = data["insoleAll_r"].astype(np.float64)

    t_insole_l = data["t_insole_l"].astype(np.float64)

    t_insole_r = data["t_insole_r"].astype(np.float64)

    for j in range(8):
        gait = get_gait_parameters_insole(
            insoleAll_r[j * t_split : (j + 1) * t_split, :],
            insoleAll_l[j * t_split : t_split * (j + 1), :],
            t_insole_r[j * t_split : t_split * (j + 1), :],
            t_insole_l[j * t_split : t_split * (j + 1), :],
        )

        FM[(n - 1) * 8 + j, 0] = gait["cadence"]

        FM[(n - 1) * 8 + j, 1] = gait["cycle_var_r"]

        FM[(n - 1) * 8 + j, 2] = gait["cycle_var_l"]

        FM[(n - 1) * 8 + j, 3] = gait["asym"]

        if i == 1 or i == 7:
            FM[(n - 1) * 8 + j, 4] = 0.6

        elif i == 2 or i == 8:
            FM[(n - 1) * 8 + j, 4] = 1.2
        else:
            FM[(n - 1) * 8 + j, 4] = 1.8

        if i <= 4:
            FM[(n - 1) * 8 + j, 5] = 1
        else:
            FM[(n - 1) * 8 + j, 5] = 0


data = {
    "cadence": FM[:, 0],
    "cycle_var_r": FM[:, 1],
    "cycle_var_l": FM[:, 2],
    "asym": FM[:, 3],
    "Speed_Label": FM[:, 4],
    # 'Status_Label':  FM[:,5],
}

df = pd.DataFrame(data)

sns.pairplot(df, hue="Speed_Label", palette="Set1", diag_kind="kde", corner=True)

plt.suptitle("Pairplot of Features Colored by Status_Label", y=1.02)


data = {
    "cadence": FM[:, 0],
    "cycle_var_r": FM[:, 1],
    "cycle_var_l": FM[:, 2],
    "asym": FM[:, 3],
    # 'Speed_Label':  FM[:,4],
    "Status_Label": FM[:, 5],
}

df = pd.DataFrame(data)

sns.pairplot(df, hue="Statues_Label", palette="Set1", diag_kind="kde", corner=True)

plt.suptitle("Pairplot of Features Colored by Status_Label", y=1.02)

plt.show()


# %% joint angles
