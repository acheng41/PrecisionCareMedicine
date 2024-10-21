from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
from pathlib import Path
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from getParameters import get_gait_parameters_insole
from visualization import plot_cop_and_gait
from visualization import plot_heatmap

data = sio.loadmat('data/gait_recording_080624_walk.mat')
insoleAll_l = data['insoleAll_l'].astype(np.float64)
insoleAll_r = data['insoleAll_r'].astype(np.float64)
t_insole_l = data['t_insole_l'].astype(np.float64)
t_insole_r = data['t_insole_r'].astype(np.float64)

# extract parameters
gait = get_gait_parameters_insole(insoleAll_r, insoleAll_l, t_insole_r, t_insole_l)

plot_cop_and_gait(gait)
plot_heatmap(insoleAll_r, insoleAll_l)
