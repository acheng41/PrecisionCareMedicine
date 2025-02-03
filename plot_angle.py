from matplotlib.animation import FuncAnimation
import pandas as pd

import numpy as np

from pathlib import Path
import os

import scipy.io as sio

import matplotlib.pyplot as plt

import torch

from matplotlib.pyplot import figure

from scipy.ndimage import label

from torch.distributed.tensor import zeros

from torch.utils.data import Dataset, DataLoader

from getParameters import get_gait_parameters_insole

from getParameters import gait_aligned_jnt

from visualization import plot_cop_and_gait, plot_heatmap_with_COP

from visualization import plot_heatmap_spatial

from matplotlib.animation import FuncAnimation
import matplotlib

matplotlib.use("TkAgg")

data = sio.loadmat("data/gait_recording_102324_walk1.mat")

insoleAll_l = data["insoleAll_l"].astype(np.float64)

insoleAll_r = data["insoleAll_r"].astype(np.float64)

t_insole_l = data["t_insole_l"].astype(np.float64)

t_insole_r = data["t_insole_r"].astype(np.float64)


t_trackers = data["t_trackers"].astype(np.float64)

jnt_angles_all_l = np.array(data["jnt_angles_all_l"])

jnt_angles_all_r = np.array(data["jnt_angles_all_r"])

jnt_pos_all_l = np.array(data["jnt_pos_all_l"])

jnt_pos_all_r = np.array(data["jnt_pos_all_r"])

# # extract parameters

gait0 = get_gait_parameters_insole(insoleAll_r, insoleAll_l, t_insole_r, t_insole_l)

joint0 = gait_aligned_jnt(
    gait0, jnt_angles_all_l, jnt_angles_all_r, jnt_pos_all_l, jnt_pos_all_r, t_trackers
)


data = sio.loadmat("data/gait_recording_102324_walk2.mat")

insoleAll_l = data["insoleAll_l"].astype(np.float64)

insoleAll_r = data["insoleAll_r"].astype(np.float64)

t_insole_l = data["t_insole_l"].astype(np.float64)

t_insole_r = data["t_insole_r"].astype(np.float64)


t_trackers = data["t_trackers"].astype(np.float64)

jnt_angles_all_l = np.array(data["jnt_angles_all_l"])

jnt_angles_all_r = np.array(data["jnt_angles_all_r"])

jnt_pos_all_l = np.array(data["jnt_pos_all_l"])

jnt_pos_all_r = np.array(data["jnt_pos_all_r"])

# # extract parameters

gait = get_gait_parameters_insole(insoleAll_r, insoleAll_l, t_insole_r, t_insole_l)

joint = gait_aligned_jnt(
    gait, jnt_angles_all_l, jnt_angles_all_r, jnt_pos_all_l, jnt_pos_all_r, t_trackers
)


data = sio.loadmat("data/gait_recording_102324_walk8.mat")

insoleAll_l = data["insoleAll_l"].astype(np.float64)

insoleAll_r = data["insoleAll_r"].astype(np.float64)

t_insole_l = data["t_insole_l"].astype(np.float64)

t_insole_r = data["t_insole_r"].astype(np.float64)


t_trackers = data["t_trackers"].astype(np.float64)

jnt_angles_all_l = np.array(data["jnt_angles_all_l"])

jnt_angles_all_r = np.array(data["jnt_angles_all_r"])

jnt_pos_all_l = np.array(data["jnt_pos_all_l"])

jnt_pos_all_r = np.array(data["jnt_pos_all_r"])


gait2 = get_gait_parameters_insole(insoleAll_r, insoleAll_l, t_insole_r, t_insole_l)

joint2 = gait_aligned_jnt(
    gait, jnt_angles_all_l, jnt_angles_all_r, jnt_pos_all_l, jnt_pos_all_r, t_trackers
)


# %%
start_step = 10

end_step = 61
steps = range(start_step, end_step)

num_points_per_step = 100  # 每步数据点数

ankle_angles_normal = np.array(
    [
        joint["resampled_angles_r"]["ankle"][
            num_points_per_step * (step - 1) : num_points_per_step * step
        ]
        for step in steps
    ]
)

ankle_angles_boot = np.array(
    [
        joint2["resampled_angles_r"]["ankle"][
            num_points_per_step * (step - 1) : num_points_per_step * step
        ]
        for step in steps
    ]
)


knee_angles_normal = np.array(
    [
        joint["resampled_angles_r"]["knee"][
            num_points_per_step * (step - 1) : num_points_per_step * step
        ]
        for step in steps
    ]
)

knee_angles_boot = np.array(
    [
        joint2["resampled_angles_r"]["knee"][
            num_points_per_step * (step - 1) : num_points_per_step * step
        ]
        for step in steps
    ]
)


hip_angles_normal = np.array(
    [
        joint["resampled_angles_r"]["hip"][
            num_points_per_step * (step - 1) : num_points_per_step * step
        ]
        for step in steps
    ]
)

hip_angles_boot = np.array(
    [
        joint2["resampled_angles_r"]["hip"][
            num_points_per_step * (step - 1) : num_points_per_step * step
        ]
        for step in steps
    ]
)

# 计算平均值和标准差

ankle_mean_normal = np.mean(ankle_angles_normal, axis=0)

ankle_std_normal = np.std(ankle_angles_normal, axis=0)

ankle_mean_boot = np.mean(ankle_angles_boot, axis=0)

ankle_std_boot = np.std(ankle_angles_boot, axis=0)


knee_mean_normal = np.mean(knee_angles_normal, axis=0)

knee_std_normal = np.std(knee_angles_normal, axis=0)

knee_mean_boot = np.mean(knee_angles_boot, axis=0)

knee_std_boot = np.std(knee_angles_boot, axis=0)


hip_mean_normal = np.mean(hip_angles_normal, axis=0)

hip_std_normal = np.std(hip_angles_normal, axis=0)

hip_mean_boot = np.mean(hip_angles_boot, axis=0)

hip_std_boot = np.std(hip_angles_boot, axis=0)


# 设置 x 轴

x = np.linspace(0, 100, 100)


# 绘图

fig, ax = plt.subplots(3, 3, figsize=(15, 10))


# 定义绘制函数，避免代码重复


def plot_with_error(ax, mean_normal, std_normal, mean_boot, std_boot, title, n):
    ax.plot(x, mean_normal[:, n], label="normal", color="blue")

    ax.fill_between(
        x,
        mean_normal[:, n] - std_normal[:, n],
        mean_normal[:, n] + std_normal[:, n],
        color="blue",
        alpha=0.3,
    )

    ax.plot(x, mean_boot[:, n], label="stimulate", color="orange")

    ax.fill_between(
        x,
        mean_boot[:, n] - std_boot[:, n],
        mean_boot[:, n] + std_boot[:, n],
        color="orange",
        alpha=0.3,
    )
    ax.set_title(title)

    ax.legend(loc="best")


# 绘制每个关节的角度和误差,0)

plot_with_error(
    ax[0, 0],
    ankle_mean_normal,
    ankle_std_normal,
    ankle_mean_boot,
    ankle_std_boot,
    "Ankle Inversion/Eversion",
    0,
)

plot_with_error(
    ax[0, 1],
    knee_mean_normal,
    knee_std_normal,
    knee_mean_boot,
    knee_std_boot,
    "Knee Adduction/Abduction",
    0,
)

plot_with_error(
    ax[0, 2],
    hip_mean_normal,
    hip_std_normal,
    hip_mean_boot,
    hip_std_boot,
    "Hip Adduction/Abduction",
    0,
)

plot_with_error(
    ax[1, 0],
    ankle_mean_normal,
    ankle_std_normal,
    ankle_mean_boot,
    ankle_std_boot,
    "Ankle Adduction/Abduction",
    1,
)

plot_with_error(
    ax[1, 1],
    knee_mean_normal,
    knee_std_normal,
    knee_mean_boot,
    knee_std_boot,
    "Knee Internal/External",
    1,
)

plot_with_error(
    ax[1, 2],
    hip_mean_normal,
    hip_std_normal,
    hip_mean_boot,
    hip_std_boot,
    "Hip Internal/External",
    1,
)

plot_with_error(
    ax[2, 0],
    ankle_mean_normal,
    ankle_std_normal,
    ankle_mean_boot,
    ankle_std_boot,
    "Ankle Dorsi/Plantar Flexion",
    2,
)

plot_with_error(
    ax[2, 1],
    knee_mean_normal,
    knee_std_normal,
    knee_mean_boot,
    knee_std_boot,
    "Knee Flexion/Extension",
    2,
)

plot_with_error(
    ax[2, 2],
    hip_mean_normal,
    hip_std_normal,
    hip_mean_boot,
    hip_std_boot,
    "Hip Flexion/Extension",
    2,
)

ax[2, 1].set_xlabel("cycle (%)")

ax[1, 0].set_ylabel("angle (degree)")

plt.tight_layout()

plt.show()


# %% plot angle in different status

start_step = 51

end_step = 56

x = np.linspace(0, 100 * (end_step - start_step), 100 * (end_step - start_step))

fig, ax = plt.subplots(3, 3)


ax[0, 0].set_title("left ankle angle")

ax[0, 0].plot(
    x,
    joint["resampled_angles_l"]["ankle"][
        100 * (start_step - 1) : 100 * (end_step - 1), 0
    ],
    label="normal",
)

ax[0, 0].plot(
    x,
    joint2["resampled_angles_l"]["ankle"][
        100 * (start_step - 1) : 100 * (end_step - 1), 0
    ],
    label="boot",
)


ax[1, 0].plot(
    x,
    joint["resampled_angles_l"]["ankle"][
        100 * (start_step - 1) : 100 * (end_step - 1), 1
    ],
    label="normal",
)

ax[1, 0].plot(
    x,
    joint2["resampled_angles_l"]["ankle"][
        100 * (start_step - 1) : 100 * (end_step - 1), 1
    ],
    label="boot",
)


ax[2, 0].plot(
    x,
    joint["resampled_angles_l"]["ankle"][
        100 * (start_step - 1) : 100 * (end_step - 1), 2
    ],
    label="normal",
)

ax[2, 0].plot(
    x,
    joint2["resampled_angles_l"]["ankle"][
        100 * (start_step - 1) : 100 * (end_step - 1), 2
    ],
    label="boot",
)


ax[0, 1].set_title("left knee angle")

ax[0, 1].plot(
    x,
    joint["resampled_angles_l"]["knee"][
        100 * (start_step - 1) : 100 * (end_step - 1), 0
    ],
    label="normal",
)

ax[0, 1].plot(
    x,
    joint2["resampled_angles_l"]["knee"][
        100 * (start_step - 1) : 100 * (end_step - 1), 0
    ],
    label="boot",
)


ax[1, 1].plot(
    x,
    joint["resampled_angles_l"]["knee"][
        100 * (start_step - 1) : 100 * (end_step - 1), 1
    ],
    label="normal",
)

ax[1, 1].plot(
    x,
    joint2["resampled_angles_l"]["knee"][
        100 * (start_step - 1) : 100 * (end_step - 1), 1
    ],
    label="boot",
)


ax[2, 1].plot(
    x,
    joint["resampled_angles_l"]["knee"][
        100 * (start_step - 1) : 100 * (end_step - 1), 2
    ],
    label="normal",
)

ax[2, 1].plot(
    x,
    joint2["resampled_angles_l"]["knee"][
        100 * (start_step - 1) : 100 * (end_step - 1), 2
    ],
    label="boot",
)


ax[0, 2].set_title("left hip angle")

ax[0, 2].plot(
    x,
    joint["resampled_angles_l"]["hip"][
        100 * (start_step - 1) : 100 * (end_step - 1), 0
    ],
    label="normal",
)

ax[0, 2].plot(
    x,
    joint2["resampled_angles_l"]["hip"][
        100 * (start_step - 1) : 100 * (end_step - 1), 0
    ],
    label="boot",
)


ax[1, 2].plot(
    x,
    joint["resampled_angles_l"]["hip"][
        100 * (start_step - 1) : 100 * (end_step - 1), 1
    ],
    label="normal",
)

ax[1, 2].plot(
    x,
    joint2["resampled_angles_l"]["hip"][
        100 * (start_step - 1) : 100 * (end_step - 1), 1
    ],
    label="boot",
)


ax[2, 2].plot(
    x,
    joint["resampled_angles_l"]["hip"][
        100 * (start_step - 1) : 100 * (end_step - 1), 2
    ],
    label="normal",
)

ax[2, 2].plot(
    x,
    joint2["resampled_angles_l"]["hip"][
        100 * (start_step - 1) : 100 * (end_step - 1), 2
    ],
    label="boot",
)

plt.legend(loc="best")


fig, ax = plt.subplots(3, 3)


ax[0, 0].set_title("right ankle angle")

ax[0, 0].plot(
    x,
    joint["resampled_angles_r"]["ankle"][
        100 * (start_step - 1) : 100 * (end_step - 1), 0
    ],
    label="normal",
)

ax[0, 0].plot(
    x,
    joint2["resampled_angles_r"]["ankle"][
        100 * (start_step - 1) : 100 * (end_step - 1), 0
    ],
    label="boot",
)


ax[1, 0].plot(
    x,
    joint["resampled_angles_r"]["ankle"][
        100 * (start_step - 1) : 100 * (end_step - 1), 1
    ],
    label="normal",
)

ax[1, 0].plot(
    x,
    joint2["resampled_angles_r"]["ankle"][
        100 * (start_step - 1) : 100 * (end_step - 1), 1
    ],
    label="boot",
)


ax[2, 0].plot(
    x,
    joint["resampled_angles_r"]["ankle"][
        100 * (start_step - 1) : 100 * (end_step - 1), 2
    ],
    label="normal",
)

ax[2, 0].plot(
    x,
    joint2["resampled_angles_r"]["ankle"][
        100 * (start_step - 1) : 100 * (end_step - 1), 2
    ],
    label="boot",
)


ax[0, 1].set_title("right knee angle")

ax[0, 1].plot(
    x,
    joint["resampled_angles_r"]["knee"][
        100 * (start_step - 1) : 100 * (end_step - 1), 0
    ],
    label="normal",
)

ax[0, 1].plot(
    x,
    joint2["resampled_angles_r"]["knee"][
        100 * (start_step - 1) : 100 * (end_step - 1), 0
    ],
    label="boot",
)


ax[1, 1].plot(
    x,
    joint["resampled_angles_r"]["knee"][
        100 * (start_step - 1) : 100 * (end_step - 1), 1
    ],
    label="normal",
)

ax[1, 1].plot(
    x,
    joint2["resampled_angles_r"]["knee"][
        100 * (start_step - 1) : 100 * (end_step - 1), 1
    ],
    label="boot",
)


ax[2, 1].plot(
    x,
    joint["resampled_angles_r"]["knee"][
        100 * (start_step - 1) : 100 * (end_step - 1), 2
    ],
    label="normal",
)

ax[2, 1].plot(
    x,
    joint2["resampled_angles_r"]["knee"][
        100 * (start_step - 1) : 100 * (end_step - 1), 2
    ],
    label="boot",
)


ax[0, 2].set_title("right hip angle")

ax[0, 2].plot(
    x,
    joint["resampled_angles_r"]["hip"][
        100 * (start_step - 1) : 100 * (end_step - 1), 0
    ],
    label="normal",
)

ax[0, 2].plot(
    x,
    joint2["resampled_angles_r"]["hip"][
        100 * (start_step - 1) : 100 * (end_step - 1), 0
    ],
    label="boot",
)


ax[1, 2].plot(
    x,
    joint["resampled_angles_r"]["hip"][
        100 * (start_step - 1) : 100 * (end_step - 1), 1
    ],
    label="normal",
)

ax[1, 2].plot(
    x,
    joint2["resampled_angles_r"]["hip"][
        100 * (start_step - 1) : 100 * (end_step - 1), 1
    ],
    label="boot",
)


ax[2, 2].plot(
    x,
    joint["resampled_angles_r"]["hip"][
        100 * (start_step - 1) : 100 * (end_step - 1), 2
    ],
    label="normal",
)

ax[2, 2].plot(
    x,
    joint2["resampled_angles_r"]["hip"][
        100 * (start_step - 1) : 100 * (end_step - 1), 2
    ],
    label="boot",
)

plt.legend(loc="best")

plt.show()


# %% plot angles in different speeds

start_step = 25

end_step = 30

x = np.linspace(0, 100 * (end_step - start_step), 100 * (end_step - start_step))


fig, ax = plt.subplots(3, 3, figsize=(12, 8))


ax[0, 0].set_title("Ankle Inversion/Eversion")

ax[0, 0].plot(
    x,
    joint0["resampled_angles_l"]["ankle"][
        100 * (start_step - 1) : 100 * (end_step - 1), 0
    ],
    label="0.6m/s",
)

ax[0, 0].plot(
    x,
    joint["resampled_angles_l"]["ankle"][
        100 * (start_step - 1) : 100 * (end_step - 1), 0
    ],
    label="1.2m/s",
)

ax[0, 0].plot(
    x,
    joint2["resampled_angles_l"]["ankle"][
        100 * (start_step - 1) : 100 * (end_step - 1), 0
    ],
    label="1.8m/s",
)

ax[1, 0].set_title("Ankle Adduction/Abduction")

ax[1, 0].plot(
    x,
    joint0["resampled_angles_l"]["ankle"][
        100 * (start_step - 1) : 100 * (end_step - 1), 1
    ],
    label="0.6m/s",
)

ax[1, 0].plot(
    x,
    joint["resampled_angles_l"]["ankle"][
        100 * (start_step - 1) : 100 * (end_step - 1), 1
    ],
    label="1.2m/s",
)

ax[1, 0].plot(
    x,
    joint2["resampled_angles_l"]["ankle"][
        100 * (start_step - 1) : 100 * (end_step - 1), 1
    ],
    label="1.8m/s",
)

ax[2, 0].set_title("Ankle Dorsi/Plantar Flexion")

ax[2, 0].plot(
    x,
    joint0["resampled_angles_l"]["ankle"][
        100 * (start_step - 1) : 100 * (end_step - 1), 2
    ],
    label="0.6m/s",
)

ax[2, 0].plot(
    x,
    joint["resampled_angles_l"]["ankle"][
        100 * (start_step - 1) : 100 * (end_step - 1), 2
    ],
    label="1.2m/s",
)

ax[2, 0].plot(
    x,
    joint2["resampled_angles_l"]["ankle"][
        100 * (start_step - 1) : 100 * (end_step - 1), 2
    ],
    label="1.8m/s",
)


ax[0, 1].set_title("Knee Adduction/Abduction")

ax[0, 1].plot(
    x,
    joint0["resampled_angles_l"]["knee"][
        100 * (start_step - 1) : 100 * (end_step - 1), 0
    ],
    label="0.6m/s",
)

ax[0, 1].plot(
    x,
    joint["resampled_angles_l"]["knee"][
        100 * (start_step - 1) : 100 * (end_step - 1), 0
    ],
    label="1.2m/s",
)

ax[0, 1].plot(
    x,
    joint2["resampled_angles_l"]["knee"][
        100 * (start_step - 1) : 100 * (end_step - 1), 0
    ],
    label="1.8m/s",
)

ax[1, 1].set_title("Knee Internal/External")

ax[1, 1].plot(
    x,
    joint0["resampled_angles_l"]["knee"][
        100 * (start_step - 1) : 100 * (end_step - 1), 1
    ],
    label="0.6m/s",
)

ax[1, 1].plot(
    x,
    joint["resampled_angles_l"]["knee"][
        100 * (start_step - 1) : 100 * (end_step - 1), 1
    ],
    label="1.2m/s",
)

ax[1, 1].plot(
    x,
    joint2["resampled_angles_l"]["knee"][
        100 * (start_step - 1) : 100 * (end_step - 1), 1
    ],
    label="1.8m/s",
)

ax[2, 1].set_title("Knee Flexion/Extension")

ax[2, 1].plot(
    x,
    joint0["resampled_angles_l"]["knee"][
        100 * (start_step - 1) : 100 * (end_step - 1), 2
    ],
    label="0.6m/s",
)

ax[2, 1].plot(
    x,
    joint["resampled_angles_l"]["knee"][
        100 * (start_step - 1) : 100 * (end_step - 1), 2
    ],
    label="1.2m/s",
)

ax[2, 1].plot(
    x,
    joint2["resampled_angles_l"]["knee"][
        100 * (start_step - 1) : 100 * (end_step - 1), 2
    ],
    label="1.8m/s",
)


ax[0, 2].set_title("Hip Adduction/Abduction")

ax[0, 2].plot(
    x,
    joint0["resampled_angles_l"]["hip"][
        100 * (start_step - 1) : 100 * (end_step - 1), 0
    ],
    label="0.6m/s",
)

ax[0, 2].plot(
    x,
    joint["resampled_angles_l"]["hip"][
        100 * (start_step - 1) : 100 * (end_step - 1), 0
    ],
    label="1.2m/s",
)

ax[0, 2].plot(
    x,
    joint2["resampled_angles_l"]["hip"][
        100 * (start_step - 1) : 100 * (end_step - 1), 0
    ],
    label="1.8m/s",
)

ax[1, 2].set_title("Hip Internal/External")

ax[1, 2].plot(
    x,
    joint0["resampled_angles_l"]["hip"][
        100 * (start_step - 1) : 100 * (end_step - 1), 1
    ],
    label="0.6m/s",
)

ax[1, 2].plot(
    x,
    joint["resampled_angles_l"]["hip"][
        100 * (start_step - 1) : 100 * (end_step - 1), 1
    ],
    label="1.2m/s",
)

ax[1, 2].plot(
    x,
    joint2["resampled_angles_l"]["hip"][
        100 * (start_step - 1) : 100 * (end_step - 1), 1
    ],
    label="1.8m/s",
)

ax[
    2,
    2,
].set_title("Hip Flexion/Extension")

ax[2, 2].plot(
    x,
    joint0["resampled_angles_l"]["hip"][
        100 * (start_step - 1) : 100 * (end_step - 1), 2
    ],
    label="0.6m/s",
)

ax[2, 2].plot(
    x,
    joint["resampled_angles_l"]["hip"][
        100 * (start_step - 1) : 100 * (end_step - 1), 2
    ],
    label="1.2m/s",
)

ax[2, 2].plot(
    x,
    joint2["resampled_angles_l"]["hip"][
        100 * (start_step - 1) : 100 * (end_step - 1), 2
    ],
    label="1.8m/s",
)


ax[1, 0].set_ylabel("angle(degree)")

ax[2, 1].set_xlabel("cycle (%)")

plt.legend(loc="best")

plt.show()


# %%

plot_cop_and_gait(gait)

plot_heatmap_spatial(insoleAll_r, insoleAll_l)

plot_heatmap_with_COP(gait)
