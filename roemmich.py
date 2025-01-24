import pandas as pd
import numpy as np
from pathlib import Path
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

insole_data_path = Path("data/112024")
insole_data = sio.loadmat(insole_data_path / "gait_recording_112024_walk2.mat")

vicon_data_path = Path("data/Pilot_Data/Session_two")
vicon_events_data = sio.loadmat(
    vicon_data_path / "Faster_kinematics_events_converted.mat", squeeze_me=True
)
vicon_kinematic_data = sio.loadmat(
    vicon_data_path / "Faster_kinematics_kinematics_converted.mat", squeeze_me=True
)
vicon_spatiotemporals_data = sio.loadmat(
    vicon_data_path / "Faster_kinematics_spatiotemporals_converted.mat", squeeze_me=True
)

joint_angles_data = (
    vicon_kinematic_data["kinematics"]["joint_angles"].item()["sagittal"].item()
)


joint_angles_df_left = pd.DataFrame(joint_angles_data["left"].item())
joint_angles_df_right = pd.DataFrame(joint_angles_data["right"].item())

events_df = pd.DataFrame(
    np.column_stack(vicon_events_data["events"].item()),
    columns=pd.Series(tup[0] for tup in vicon_events_data["events"].dtype.descr),
)
# %%
y = joint_angles_df_left["tro"]
x = np.arange(y.size)
plt.plot(x, y)
marker_indices = events_df["lhs"]
plt.plot(
    x[marker_indices], y[marker_indices], "ro", label="Heel Strikes"
)  # 'ro' for red circles

marker_indices = events_df["lto"]
plt.plot(x[marker_indices], y[marker_indices], "gx", label="Toe Offs")

# Display the plot
plt.legend()
plt.show()
