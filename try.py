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

from visualization import plot_cop_and_gait

from visualization import plot_heatmap_spatial


data = sio.loadmat("data/gait_recording_080624_walk.mat")

insoleAll_l = data["insoleAll_l"].astype(np.float64)

insoleAll_r = data["insoleAll_r"].astype(np.float64)

t_insole_l = data["t_insole_l"].astype(np.float64)

t_insole_r = data["t_insole_r"].astype(np.float64)


t_trackers = data["t_trackers"].astype(np.float64)

jnt_angles_all_l = np.array(data["jnt_angles_all_l"])

jnt_angles_all_r = np.array(data["jnt_angles_all_r"])

jnt_pos_all_l = np.array(data["jnt_pos_all_l"])

jnt_pos_all_r = np.array(data["jnt_pos_all_r"])


# extract parameters

gait = get_gait_parameters_insole(insoleAll_r, insoleAll_l, t_insole_r, t_insole_l)
joint = gait_aligned_jnt(
    gait, jnt_angles_all_l, jnt_angles_all_r, jnt_pos_all_l, jnt_pos_all_r, t_trackers
)


# figure()

# plt.title('Left ankle angle')

# plt.subplot(3, 1, 1)

# plt.plot(t_trackers[joint['strike_l'][0]:joint['strike_l'][-1]], joint['jnt_angles_l'][joint['strike_l'][0]:joint['strike_l'][-1],2,0])

# plt.scatter(t_trackers[joint['strike_l']], joint['jnt_angles_l'][joint['strike_l'],2,0],color='red',marker='o', label='strike')

# plt.scatter(t_trackers[joint['off_l']], joint['jnt_angles_l'][joint['off_l'],2,0],color='green',marker='o', label='off')

# plt.legend()

# plt.subplot(3, 1, 2)

# plt.plot(t_trackers[joint['strike_l'][0]:joint['strike_l'][-1]], joint['jnt_angles_l'][joint['strike_l'][0]:joint['strike_l'][-1],2,1])

# plt.scatter(t_trackers[joint['strike_l']], joint['jnt_angles_l'][joint['strike_l'],2,1],color='red',marker='o', label='strike')

# plt.scatter(t_trackers[joint['off_l']], joint['jnt_angles_l'][joint['off_l'],2,1],color='green',marker='o', label='off')

# plt.legend()

# plt.subplot(3, 1, 3)

# plt.plot(t_trackers[joint['strike_l'][0]:joint['strike_l'][-1]], joint['jnt_angles_l'][joint['strike_l'][0]:joint['strike_l'][-1],2,2])

# plt.scatter(t_trackers[joint['strike_l']], joint['jnt_angles_l'][joint['strike_l'],2,2],color='red',marker='o', label='strike')

# plt.scatter(t_trackers[joint['off_l']], joint['jnt_angles_l'][joint['off_l'],2,2],color='green',marker='o', label='off')

# plt.legend()


# plt.show()


# plot_cop_and_gait(gait)

# plot_heatmap(insoleAll_r, insoleAll_l)
