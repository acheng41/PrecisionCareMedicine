import numpy as np
from pathlib import Path
import scipy.io as sio
import scipy
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.animation import FuncAnimation
import cv2 as cv

from visualization import plot_heatmap_spatial
from mpl_toolkits.mplot3d import Axes3D
import pickle

from clustering_utils import *


# insole_data_path = Path("data/112024")
# insole_data = sio.loadmat(insole_data_path / "gait_recording_112024_walk4.mat")

insole_data_path = Path("data/080624")
insole_data = sio.loadmat(insole_data_path / "gait_recording_080624_walk4.mat")

# insole_data_path = Path("data/021925")
# insole_data = sio.loadmat(insole_data_path / "gait_recording_021925_walk11.mat")

# insole_data_path = Path("data/102324")
# insole_data = sio.loadmat(insole_data_path / "gait_recording_102324_walk8.mat")


insole_spatial_l, insole_spatial_r = convert_insole_spatial(
    insole_data["insoleAll_l"], insole_data["insoleAll_r"]
)


def plot_heatmap_spatial(
    insole_spatial_l: np.ndarray,
    insole_spatial_r: np.ndarray,
) -> None:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))

    img_l = ax_l.imshow(
        np.zeros((64, 16)),
        vmin=np.min(insole_spatial_l),
        vmax=np.max(insole_spatial_l),
        interpolation="nearest",
    )
    img_r = ax_r.imshow(
        np.zeros((64, 16)),
        vmin=np.min(insole_spatial_r),
        vmax=np.max(insole_spatial_r),
        interpolation="nearest",
    )

    ax_l.set_title("Left Heatmap")
    ax_r.set_title("Right Heatmap")
    plt.colorbar(img_l, ax=ax_l)
    plt.colorbar(img_r, ax=ax_r)

    def update(frame):
        img_l.set_array(insole_spatial_l[frame])
        img_r.set_array(insole_spatial_r[frame])

        return img_l, img_r

    ani = FuncAnimation(
        fig,
        update,
        frames=500,
        blit=True,
        repeat=True,
        interval=10,
    )

    plt.show()

    writervideo = animation.FFMpegWriter(fps=30)
    ani.save("stroke_walking.gif", writer=writervideo)


plot_heatmap_spatial(insole_spatial_l, insole_spatial_r)

fig, (ax_l, ax_r) = plt.subplots(1, 2)
ax_l.imshow(insole_spatial_l.mean(0))
ax_l.set_title("Left Insole")
ax_l.axis("off")

ax_r.imshow(insole_spatial_r.mean(0))
ax_r.set_title("Right Insole")
ax_r.axis("off")

fig.show()

plt.imshow(insole_spatial_l.mean(0))
plt.title("left heatmap")
plt.show()

height_valleys_l = compute_height_seg_coords(
    insole_spatial_l, validate=True, show_plot=True
)

height_valleys_r = compute_height_seg_coords(
    insole_spatial_r, validate=True, show_plot=True
)


width_valleys_l = compute_width_seg_coords(
    insole_spatial_l, height_valleys_l, validate=True, show_plot=True
)
width_valleys_r = compute_width_seg_coords(
    insole_spatial_r, height_valleys_r, validate=True, show_plot=True
)


plt.imshow(insole_spatial_l.mean(0))

for height_valley_l in height_valleys_l:
    plt.axhline(height_valley_l, color="r", linestyle="--")

for i, (height_valley_l, width_valley_l) in enumerate(
    zip(height_valleys_l, width_valleys_l)
):
    plt.vlines(
        width_valley_l,
        height_valleys_l[i - 1] if i > 0 else 0,
        height_valley_l,
        color="r",
        linestyle="--",
    )

plt.title("Left Insole")
plt.show()

plt.imshow(insole_spatial_l.mean(0).mean(1, keepdims=True))
plt.title("Colwise Mean")
plt.axis("off")

plt.show()

plt.imshow(
    insole_spatial_l.mean(0)[0 : width_valleys_l[0].item() + 1, :].mean(
        0, keepdims=True
    ),
)
plt.title("Toe Rowwise Mean")
plt.axis("off")

plt.show()

plt.imshow(
    insole_spatial_l.mean(0)[
        width_valleys_l[0].item() : width_valleys_l[1].item() + 1, :
    ].mean(0, keepdims=True)
)
plt.title("Ball Rowwise Mean")
plt.axis("off")

plt.show()

plt.imshow(
    insole_spatial_l.mean(0)[width_valleys_l[1].item() :, :].mean(0, keepdims=True)
)
plt.title("Heel Rowwise Mean")
plt.axis("off")

plt.show()

seg_coords = []
for insole_data_file in insole_data_path.glob("*"):
    if insole_data_file.suffix != ".mat":
        continue

    print(insole_data_file)
    insole_data = sio.loadmat(insole_data_file)

    insole_spatial_l, insole_spatial_r = convert_insole_spatial(
        insole_data["insoleAll_l"], insole_data["insoleAll_r"]
    )

    height_valleys_l = compute_height_seg_coords(
        insole_spatial_l, validate=True, show_plot=False
    )

    height_valleys_r = compute_height_seg_coords(
        insole_spatial_r, validate=True, show_plot=False
    )

    width_valleys_l = compute_width_seg_coords(
        insole_spatial_l, height_valleys_l, validate=True, show_plot=False
    )
    width_valleys_r = compute_width_seg_coords(
        insole_spatial_r, height_valleys_r, validate=True, show_plot=False
    )

    fig, (ax_l, ax_r) = plt.subplots(1, 2)
    ax_l.imshow(insole_spatial_l.mean(0))

    for height_valley_l in height_valleys_l:
        ax_l.axhline(height_valley_l, color="r", linestyle="--")

    for i, (height_valley_l, width_valley_l) in enumerate(
        zip(height_valleys_l, width_valleys_l)
    ):
        ax_l.vlines(
            width_valley_l,
            height_valleys_l[i - 1] if i > 0 else 0,
            height_valley_l,
            color="r",
            linestyle="--",
        )

    ax_r.imshow(insole_spatial_r.mean(0))
    for height_valley_r in height_valleys_r:
        ax_r.axhline(height_valley_r, color="r", linestyle="--")

    for i, (height_valley_r, width_valley_r) in enumerate(
        zip(height_valleys_r, width_valleys_r)
    ):
        ax_r.vlines(
            width_valley_r,
            height_valleys_r[i - 1] if i > 0 else 0,
            height_valley_r,
            color="r",
            linestyle="--",
        )

    fig.show()
    break

    seg_coords.append(
        {
            "filename": insole_data_file.name,
            "height_valleys_l": height_valleys_l.tolist(),
            "width_valleys_l": width_valleys_l.ravel().tolist(),
            "height_valleys_r": height_valleys_r.tolist(),
            "width_valleys_r": width_valleys_r.ravel().tolist(),
        }
    )

import json

print(json.dumps(seg_coords, indent=2))

with open("080624_seg_coords.json", "w") as ofile:
    json.dump(seg_coords, ofile, indent=2)

with open("080624_seg_coords.pkl", "wb") as ofile:
    pickle.dump(seg_coords, ofile)

with open("080624_seg_coords.pkl", "rb") as ifile:
    pickle.load(ifile)


def plot_heatmap_spatial(
    insole_spatial_l: np.ndarray, insole_spatial_r: np.ndarray, save_anim: bool
) -> None:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))

    img_l = ax_l.imshow(
        np.zeros((64, 16)),
        vmin=np.min(insole_spatial_l),
        vmax=np.max(insole_spatial_l),
        interpolation="nearest",
    )
    img_r = ax_r.imshow(
        np.zeros((64, 16)),
        vmin=np.min(insole_spatial_r),
        vmax=np.max(insole_spatial_r),
        interpolation="nearest",
    )

    ax_l.set_title("Left Heatmap")
    ax_r.set_title("Right Heatmap")
    plt.colorbar(img_l, ax=ax_l)
    plt.colorbar(img_r, ax=ax_r)

    hlines_l = [
        ax_l.axhline(height_valley_l, color="r", linestyle="--")
        for height_valley_l in height_valleys_l
    ]

    hlines_r = [
        ax_r.axhline(height_valley_r, color="r", linestyle="--")
        for height_valley_r in height_valleys_r
    ]

    vlines_l = [
        ax_l.vlines(
            width_valley_l,
            height_valleys_l[i - 1] if i > 0 else 0,
            height_valley_l,
            color="r",
            linestyle="--",
        )
        for i, (height_valley_l, width_valley_l) in enumerate(
            zip(height_valleys_l, width_valleys_l)
        )
    ]

    vlines_r = [
        ax_r.vlines(
            width_valley_r,
            height_valleys_r[i - 1] if i > 0 else 0,
            height_valley_r,
            color="r",
            linestyle="--",
        )
        for i, (height_valley_r, width_valley_r) in enumerate(
            zip(height_valleys_r, width_valleys_r)
        )
    ]

    def update(frame):
        img_l.set_array(insole_spatial_l[frame])
        img_r.set_array(insole_spatial_r[frame])

        return img_l, img_r, *hlines_l, *vlines_l, *hlines_r, *vlines_r

    ani = FuncAnimation(
        fig,
        update,
        # frames=max(len(insole_spatial_l), len(insole_spatial_r)),
        frames=range(200, 1200),
        # blit=True,
        # repeat=True,
        interval=10,
    )
    plt.show()

    if save_anim:
        writervideo = animation.FFMpegWriter(fps=30)
        # ani.save("seg_walking.gif", writer=writervideo)


plot_heatmap_spatial(insole_spatial_l, insole_spatial_r, save_anim=False)


def segment_insole_data(
    insole_spatial,
    height_valleys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    toe_region = insole_spatial[:, : height_valleys[0], :]  # Toe
    ball_region = insole_spatial[
        :, height_valleys[0] : height_valleys[1], :
    ]  #  Midfoot
    heel_region = insole_spatial[:, height_valleys[1] :, :]  #  Heel
    return toe_region, ball_region, heel_region


toe_region_l, ball_region_l, heel_region_l = segment_insole_data(
    insole_spatial_l,
    height_valleys_l,
)

toe_region_r, ball_region_r, heel_region_r = segment_insole_data(
    insole_spatial_r,
    height_valleys_r,
)


def butter_lowpass_filter(data, cutoff=8, fs=100, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)


def gait_segmentation(
    toe_region, ball_region, heel_region, h_th, t_th, show_plot: bool
):
    p_toe = np.mean(toe_region, axis=(1, 2))
    p_ball = np.mean(ball_region, axis=(1, 2))
    p_heel = np.mean(heel_region, axis=(1, 2))

    p_heel_filtered = butter_lowpass_filter(p_heel, 8, 75)
    p_fore_filtered = butter_lowpass_filter(p_ball + p_toe, 8, 75)

    p_heel_derivative = np.gradient(p_heel_filtered)
    p_fore_derivative = np.gradient(p_fore_filtered)

    hc_indices, _ = find_peaks(p_heel_derivative)
    to_indices, _ = find_peaks(-p_fore_derivative)

    hc_indices, _ = find_peaks(p_heel_derivative, height=h_th, distance=10)
    to_indices, _ = find_peaks(-p_fore_derivative, height=t_th, distance=10)

    if show_plot:
        plt.plot(p_heel_derivative, label="heel derivative")
        plt.plot(
            hc_indices,
            p_heel_derivative[hc_indices],
            "o",
        )
        plt.show()

        plt.plot(-p_fore_derivative, label="fore derivative")
        plt.plot(
            to_indices,
            -p_fore_derivative[to_indices],
            "o",
        )
        plt.show()

    return hc_indices, to_indices


def get_gait_parameters_insole(insole_r, insole_l, t_r, t_l):
    """
    Get gait parameters from insole data
    """
    gait = {
        "t_r": t_r,
        "insole_r": insole_r,
        "t_l": t_l,
        "insole_l": insole_l,
        "area": 0.002**2,
        "dim": [
            int(np.sqrt(insole_r.shape[1]) * 2),
            int(np.sqrt(insole_r.shape[1]) / 2),
        ],
        "foot_trace_r": np.zeros(len(t_r)),
        "foot_trace_l": np.zeros(len(t_l)),
        "cop_x_r": np.zeros(len(t_r)),
        "cop_y_r": np.zeros(len(t_r)),
        "cop_x_l": np.zeros(len(t_l)),
        "cop_y_l": np.zeros(len(t_l)),
        "cont_area_r": np.zeros(len(t_r)),
        "cont_area_l": np.zeros(len(t_l)),
        "pp_r": np.max(insole_r, axis=1),
        "pp_l": np.max(insole_l, axis=1),
        "pp_x_r": np.zeros_like(t_r),
        "pp_y_r": np.zeros_like(t_r),
        "pp_x_l": np.zeros_like(t_l),
        "pp_y_l": np.zeros_like(t_l),
    }

    # Center of Pressure, Gait Trajectory, Contact Area and Trace
    for i in range(len(t_r)):
        frame = insole_r[i, :].reshape(gait["dim"][0], gait["dim"][1], order="F")
        # frame = np.fliplr(frame)
        frame[: gait["dim"][0] // 2, :] = np.flipud(frame[: gait["dim"][0] // 2, :])

        gait["foot_trace_r"][i] = np.mean(frame)
        x, y = np.where(frame > 0)

        # COP
        sum_frame_r = np.sum(frame[x, y])
        if sum_frame_r > 0:
            gait["cop_x_r"][i] = np.sum(x * frame[x, y]) / sum_frame_r
            gait["cop_y_r"][i] = np.sum(y * frame[x, y]) / sum_frame_r
        else:
            gait["cop_x_r"][i] = np.nan
            gait["cop_y_r"][i] = np.nan

        gait["cont_area_r"][i] = len(x)

        # Instance peak pressure
        x, y = np.nonzero(frame == gait["pp_r"][i])
        if len(x) > 1:
            gait["pp_x_r"][i] = np.mean(x)
            gait["pp_y_r"][i] = np.mean(y)
        else:
            gait["pp_x_r"][i] = x
            gait["pp_y_r"][i] = y

    for i in range(len(t_l)):
        frame = insole_l[i, :].reshape(gait["dim"][0], gait["dim"][1], order="F")
        # frame[:gait['dim'][0] // 2, :] = np.flipud(frame[:gait['dim'][0] // 2, :])
        gait["foot_trace_l"][i] = np.mean(frame)
        x, y = np.where(frame > 0)
        sum_frame_l = np.sum(frame[x, y])
        if sum_frame_l > 0:
            gait["cop_x_l"][i] = np.sum(x * frame[x, y]) / sum_frame_l
            gait["cop_y_l"][i] = np.sum(y * frame[x, y]) / sum_frame_l
        else:
            gait["cop_x_l"][i] = np.nan
            gait["cop_y_l"][i] = np.nan

        gait["cont_area_l"][i] = len(x)

        # Instance peak pressure
        x, y = np.nonzero(frame == gait["pp_l"][i])
        if len(x) > 1:
            gait["pp_x_l"][i] = np.mean(x)
            gait["pp_y_l"][i] = np.mean(y)
        else:
            gait["pp_x_l"][i] = x
            gait["pp_y_l"][i] = y

    ## Heel Strike Toe Off and Related Parameters
    thresh_r = np.min(gait["foot_trace_r"]) + 0.1 * np.ptp(gait["foot_trace_r"])
    thresh_l = np.min(gait["foot_trace_l"]) + 0.1 * np.ptp(gait["foot_trace_l"])
    gait["strike_r"] = []
    gait["off_r"] = []
    gait["strike_l"] = []
    gait["off_l"] = []

    # Right foot strikes and offs
    for i in range(1, len(t_r) - 1):
        if gait["foot_trace_r"][i] >= thresh_r > gait["foot_trace_r"][i - 1]:
            gait["strike_r"].append(i)
        if gait["foot_trace_r"][i] >= thresh_r > gait["foot_trace_r"][i + 1]:
            gait["off_r"].append(i + 1)

    # Left foot strikes and offs
    for i in range(1, len(t_l) - 1):
        if gait["foot_trace_l"][i] >= thresh_l > gait["foot_trace_l"][i - 1]:
            gait["strike_l"].append(i)
        if gait["foot_trace_l"][i] >= thresh_l > gait["foot_trace_l"][i + 1]:
            gait["off_l"].append(i + 1)

    # Isolate complete gait cycles
    gait["off_r"] = [
        off
        for off in gait["off_r"]
        if gait["strike_r"][0] < off <= gait["strike_r"][-1]
    ]
    gait["off_l"] = [
        off
        for off in gait["off_l"]
        if gait["strike_l"][0] < off <= gait["strike_l"][-1]
    ]

    # Cycle duration
    gait["cycle_dur_r"] = np.diff(t_r[gait["strike_r"]].flatten())
    gait["cycle_dur_l"] = np.diff(t_l[gait["strike_l"]].flatten())

    # Cycle duration variability
    gait["cycle_var_r"] = (
        np.std(gait["cycle_dur_r"]) / np.mean(gait["cycle_dur_r"]) * 100
    )
    gait["cycle_var_l"] = (
        np.std(gait["cycle_dur_l"]) / np.mean(gait["cycle_dur_l"]) * 100
    )

    # Cadence
    gait["cadence"] = (
        min(len(gait["cycle_dur_r"]), len(gait["cycle_dur_l"]))
        / min(np.sum(gait["cycle_dur_r"]), np.sum(gait["cycle_dur_l"]))
        * 60
    )
    # Stance and swing phases
    gait["stance_r"] = (
        (t_r[gait["off_r"]] - t_r[gait["strike_r"][:-1]]) / gait["cycle_dur_r"] * 100
    )
    gait["stance_l"] = (
        (t_l[gait["off_l"]] - t_l[gait["strike_l"][:-1]]) / gait["cycle_dur_l"] * 100
    )

    gait["swing_r"] = 100 - gait["stance_r"]
    gait["swing_l"] = 100 - gait["stance_l"]

    # Asymmetry (swing)
    gait["asym"] = (
        (np.mean(gait["swing_l"]) - np.mean(gait["swing_r"]))
        / (0.5 * (np.mean(gait["swing_l"]) + np.mean(gait["swing_r"])))
        * 100
    )

    return gait


insole_l = insole_data["insoleAll_l"]
insole_r = insole_data["insoleAll_r"]
t_l = insole_data["t_insole_l"]
t_r = insole_data["t_insole_r"]
gait = get_gait_parameters_insole(insole_r, insole_l, t_r, t_l)
gait.keys()

hc_indices_l, to_indices_l = gait["strike_l"], gait["off_l"]
hc_indices_r, to_indices_r = gait["strike_r"], gait["off_r"]


hc_indices_l, to_indices_l = gait_segmentation(
    toe_region_l, ball_region_l, heel_region_l, 20, 20, show_plot=True
)
hc_indices_r, to_indices_r = gait_segmentation(
    toe_region_r, ball_region_r, heel_region_r, 20, 20, show_plot=True
)

to_mask_l = np.isin(np.arange(insole_spatial_l.shape[0]), to_indices_l)
to_mask_r = np.isin(np.arange(insole_spatial_r.shape[0]), to_indices_r)

hc_mask_l = np.isin(np.arange(insole_spatial_l.shape[0]), hc_indices_l)
hc_mask_r = np.isin(np.arange(insole_spatial_r.shape[0]), hc_indices_r)


def plot_heatmap_spatial(
    insole_spatial_l: np.ndarray, insole_spatial_r: np.ndarray, save_anim: bool
) -> None:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))

    img_l = ax_l.imshow(
        np.zeros((64, 16)),
        vmin=np.min(insole_spatial_l),
        vmax=np.max(insole_spatial_l),
        interpolation="nearest",
    )
    img_r = ax_r.imshow(
        np.zeros((64, 16)),
        vmin=np.min(insole_spatial_r),
        vmax=np.max(insole_spatial_r),
        interpolation="nearest",
    )

    ax_l.set_title("Left Heatmap")
    ax_r.set_title("Right Heatmap")
    plt.colorbar(img_l, ax=ax_l)
    plt.colorbar(img_r, ax=ax_r)

    hlines_l = [
        ax_l.axhline(height_valley_l, color="r", linestyle="--")
        for height_valley_l in height_valleys_l
    ]

    hlines_r = [
        ax_r.axhline(height_valley_r, color="r", linestyle="--")
        for height_valley_r in height_valleys_r
    ]

    vlines_l = [
        ax_l.vlines(
            width_valley_l,
            height_valleys_l[i - 1] if i > 0 else 0,
            height_valley_l,
            color="r",
            linestyle="--",
        )
        for i, (height_valley_l, width_valley_l) in enumerate(
            zip(height_valleys_l, width_valleys_l)
        )
    ]

    vlines_r = [
        ax_r.vlines(
            width_valley_r,
            height_valleys_r[i - 1] if i > 0 else 0,
            height_valley_r,
            color="r",
            linestyle="--",
        )
        for i, (height_valley_r, width_valley_r) in enumerate(
            zip(height_valleys_r, width_valleys_r)
        )
    ]

    FRAME_OFFSET = 200

    state_l = ""
    state_r = ""

    def update(frame):
        frame += FRAME_OFFSET
        img_l.set_array(insole_spatial_l[frame])
        img_r.set_array(insole_spatial_r[frame])
        nonlocal state_l, state_r

        if to_mask_l[frame]:
            state_l = "Toe Off"
        elif hc_mask_l[frame]:
            state_l = "Heel Strike"

        if to_mask_r[frame]:
            state_r = "Toe Off"
        elif hc_mask_r[frame]:
            state_r = "Heel Strike"

        ax_l.set_ylabel(state_l)
        ax_r.set_ylabel(state_r)

        img_l.set_cmap("plasma" if state_l == "Heel Strike" else "viridis")
        img_r.set_cmap("plasma" if state_r == "Heel Strike" else "viridis")

        return (img_l, img_r, *hlines_l, *vlines_l, *hlines_r, *vlines_r, ax_l, ax_r)

    ani = FuncAnimation(
        fig,
        update,
        frames=range(
            FRAME_OFFSET,
            FRAME_OFFSET + 500,
        ),
        # blit=True,
        repeat=True,
        interval=10,
    )
    plt.show()

    if save_anim:
        writervideo = animation.FFMpegWriter(fps=30)
        ani.save("seg_walking.gif", writer=writervideo)


plot_heatmap_spatial(insole_spatial_l, insole_spatial_r, save_anim=True)


# def segment_insole_data(insole_data):
#     toe_region = insole_data[:, :13, :]  # Toe
#     forefoot_region = insole_data[:, 13:31, :]  # Forefoot
#     midfoot_region = insole_data[:, 32:42, :]  #  Midfoot
#     heel_region = insole_data[:, 42:, :]  #  Heel
#     return heel_region, midfoot_region, forefoot_region, toe_region


# def butter_lowpass_filter(data, cutoff=8, fs=100, order=4):
#     nyquist = 0.5 * fs
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype="low", analog=False)
#     return filtfilt(b, a, data)


# def reshape_insole_grid(data):
#     num_frames = data.shape[0]
#     reshaped_data = np.empty((num_frames, 64, 16), dtype=data.dtype)
#     for i in range(num_frames):
#         frame = data[i].reshape(16, 64).T.copy()
#         frame[:32, :] = np.flipud(frame[:32, :])
#         reshaped_data[i] = frame

#     return reshaped_data


# def gait_segmentation(insole, h_th, t_th, ms_th):
#     insole = reshape_insole_grid(insole)
#     heel_region, midfoot_region, forefoot_region, toe_region = segment_insole_data(
#         insole
#     )
#     p_heel = np.mean(heel_region, axis=(1, 2))
#     p_toe = np.mean(toe_region, axis=(1, 2))
#     p_forefoot = np.mean(forefoot_region, axis=(1, 2))
#     p_insole = (insole > 25).astype(int)

#     p_heel_filtered = butter_lowpass_filter(p_heel, 8, 75)
#     p_fore_filtered = butter_lowpass_filter(p_forefoot + p_toe, 8, 75)
#     p_foot_filtered = butter_lowpass_filter(np.sum(insole, axis=(1, 2)), 8, 75)
#     plt.figure()

#     plt.plot(p_foot_filtered, label="foot")
#     plt.plot(p_fore_filtered, label="fore")
#     plt.plot(p_heel_filtered, label="heel")
#     plt.xlim(300, 700)
#     plt.legend()

#     p_heel_derivative = np.gradient(p_heel_filtered)
#     p_fore_derivative = np.gradient(p_fore_filtered)

#     hc_indices, _ = find_peaks(p_heel_derivative, height=h_th, distance=10)
#     to_indices, _ = find_peaks(-p_fore_derivative, height=t_th, distance=10)
#     ms_indices, _ = find_peaks(p_foot_filtered, height=ms_th, distance=10)

#     return hc_indices, to_indices, ms_indices


# def resample_data(data):
#     n = data.shape[0]
#     index = np.linspace(0, 100, n)
#     target = np.linspace(0, 100, 100)
#     resampled_data = np.zeros((100, 3))
#     for i in range(3):
#         interp_func = interp1d(index, data[:, i])
#         resampled_data[:, i] = interp_func(target)

#     return resampled_data


# def get_gait_parameters_insole2(insole_r, insole_l, t_r, t_l, thresholds):
#     """
#     Get gait parameters from insole data
#     """

#     [h_th_r, t_th_r, h_th_l, t_th_l, ms_th_r, ms_th_l, strike_th_l, strike_th_r] = (
#         thresholds
#     )

#     gait = {
#         "foot_trace_r": np.zeros(len(t_r)),
#         "foot_trace_l": np.zeros(len(t_l)),
#         "dim": [
#             int(np.sqrt(insole_r.shape[1]) * 2),
#             int(np.sqrt(insole_r.shape[1]) / 2),
#         ],
#         "cop_x_r": np.zeros(len(t_r)),
#         "cop_y_r": np.zeros(len(t_r)),
#         "cont_area_r": np.zeros(len(t_r)),
#         "cop_x_l": np.zeros(len(t_l)),
#         "cop_y_l": np.zeros(len(t_l)),
#         "cont_area_l": np.zeros(len(t_l)),
#     }
#     insole_data = {"insole_r_flipped": [], "insole_l_flipped": []}

#     for i in range(len(t_r)):
#         frame = insole_r[i, :].reshape(gait["dim"][0], gait["dim"][1], order="F")
#         frame[: gait["dim"][0] // 2, :] = np.flipud(frame[: gait["dim"][0] // 2, :])
#         gait["foot_trace_r"][i] = np.mean(frame)
#         insole_data["insole_r_flipped"].append(frame)
#         x, y = np.where(frame > 0)

#         # COP
#         sum_frame_r = np.sum(frame[x, y])
#         if sum_frame_r > 0:
#             gait["cop_x_r"][i] = np.sum(x * frame[x, y]) / sum_frame_r
#             gait["cop_y_r"][i] = np.sum(y * frame[x, y]) / sum_frame_r
#         else:
#             gait["cop_x_r"][i] = np.nan
#             gait["cop_y_r"][i] = np.nan

#         gait["cont_area_r"][i] = len(x)

#     for i in range(len(t_l)):
#         frame = insole_l[i, :].reshape(gait["dim"][0], gait["dim"][1], order="F")
#         frame[: gait["dim"][0] // 2, :] = np.flipud(frame[: gait["dim"][0] // 2, :])
#         insole_data["insole_l_flipped"].append(frame)
#         gait["foot_trace_l"][i] = np.mean(insole_l[i, :])

#     # Gait events
#     hc_indices, to_indices, ms_indices = gait_segmentation(
#         insole_r, h_th_r, t_th_r, ms_th_r
#     )
#     strike_r = hc_indices
#     off_r = to_indices
#     ms_r = ms_indices

#     hc_indices, to_indices, ms_indices = gait_segmentation(
#         insole_l, h_th_l, t_th_l, ms_th_l
#     )
#     strike_l = hc_indices
#     off_l = to_indices
#     ms_l = ms_indices

#     print(strike_l[:10])

#     # Isolate complete gait cycles
#     gait["step_r"] = []
#     gait["step_l"] = []
#     for i in range(len(strike_r) - 1):
#         start, end = strike_r[i], strike_r[i + 1]
#         if t_r[end] - t_r[start] > strike_th_r:
#             continue
#         step_off = [o for o in off_r if start <= o and o <= end]
#         ms = [j for j in ms_r if start <= j and j <= end]
#         if len(step_off) == 1:
#             gait["step_r"].append({"strike": [start, end], "ms": ms, "off": step_off})

#     for i in range(len(strike_l) - 1):
#         start, end = strike_l[i], strike_l[i + 1]
#         if t_l[end] - t_l[start] > strike_th_l:
#             continue
#         step_off = [o for o in off_l if start <= o and o <= end]
#         ms = [j for j in ms_l if start <= j and j <= end]

#         if len(step_off) == 1:
#             gait["step_l"].append({"strike": [start, end], "ms": ms, "off": step_off})

#     # Cycle duration
#     gait["cycle_dur_l"] = np.zeros((len(gait["step_l"])))
#     gait["swing_dur_l"] = np.zeros((len(gait["step_l"])))
#     gait["stance_dur_l"] = np.zeros((len(gait["step_l"])))
#     gait["cadence_l"] = np.zeros(len(gait["cycle_dur_l"]))
#     gait["cycle_dur_r"] = np.zeros((len(gait["step_r"])))
#     gait["swing_dur_r"] = np.zeros((len(gait["step_r"])))
#     gait["stance_dur_r"] = np.zeros((len(gait["step_r"])))
#     gait["cadence_r"] = np.zeros(len(gait["cycle_dur_r"]))

#     for i in range((len(gait["step_l"]))):
#         start, end = gait["step_l"][i]["strike"]
#         off = gait["step_l"][i]["off"]
#         gait["cycle_dur_l"][i] = t_l[end] - t_l[start]
#         gait["stance_dur_l"][i] = t_l[off] - t_l[start]
#     gait["swing_dur_l"] = gait["cycle_dur_l"] - gait["stance_dur_l"]
#     gait["stance_phase_l"] = gait["stance_dur_l"] / gait["cycle_dur_l"]
#     gait["swing_phase_l"] = gait["swing_dur_l"] / gait["cycle_dur_l"]
#     gait["cadence_l"] = 60 / gait["cycle_dur_l"]

#     for i in range((len(gait["step_r"]))):
#         start, end = gait["step_r"][i]["strike"]
#         off = gait["step_r"][i]["off"]
#         gait["cycle_dur_r"][i] = t_r[end] - t_r[start]
#         gait["stance_dur_r"][i] = t_r[off] - t_r[start]
#     gait["swing_dur_r"] = gait["cycle_dur_r"] - gait["stance_dur_r"]
#     gait["stance_phase_r"] = gait["stance_dur_r"] / gait["cycle_dur_r"]
#     gait["swing_phase_r"] = gait["swing_dur_r"] / gait["cycle_dur_r"]
#     gait["cadence_r"] = 60 / gait["cycle_dur_r"]

#     gait["swing_asym"] = np.abs(
#         (np.mean(gait["swing_phase_l"]) - np.mean(gait["swing_phase_r"]))
#     ) / (0.5 * (np.mean(gait["swing_phase_l"]) + np.mean(gait["swing_phase_r"])))

#     return gait, insole_data


# insole_l = insole_data["insoleAll_l"]
# insole_r = insole_data["insoleAll_r"]
# t_l = insole_data["t_insole_l"]
# t_r = insole_data["t_insole_r"]
# thresholds = [
#     30,
#     30,
#     30,
#     30,
#     20000,
#     20000,
#     2,
#     2,
# ]  # [h_th_r, t_th_r, h_th_l, t_th_l, ms_th_r, ms_th_l, strike_th_l, strike_th_r]

# gait, insole_data = get_gait_parameters_insole2(
#     insole_r, insole_l, t_r, t_l, thresholds
# )

# # Extracting strike and off indices
# strike_indices = [step["strike"][0] for step in gait["step_l"]]
# off_indices = [idx for step in gait["step_l"] for idx in step["off"]]
# ms_indices = [idx for step in gait["step_l"] for idx in step["ms"]]

# plt.plot(gait["foot_trace_l"])
# # Plot 'strike' with 'x' markers
# plt.scatter(
#     strike_indices,
#     [gait["foot_trace_l"][idx] for idx in strike_indices],
#     color="red",
#     marker="x",
#     label="Strike",
# )

# # Plot 'off' with 'o' markers
# plt.scatter(
#     off_indices,
#     [gait["foot_trace_l"][idx] for idx in off_indices],
#     color="blue",
#     marker="o",
#     label="Off",
# )

# # Plot 'off' with 'o' markers
# plt.scatter(
#     ms_indices,
#     [gait["foot_trace_l"][idx] for idx in ms_indices],
#     color="green",
#     marker="o",
#     label="MidStep",
# )
# plt.legend()
# plt.xlim(300, 700)

# plt.ylabel("Mean Pressure")
# plt.xlabel("Frame indices")
# plt.show()


# def produce_slice(strike_indices, off_indices, ms_indices, insole_all):
#     # double support from heel strike to mid stance

#     ds_startend = list(zip(strike_indices, ms_indices))

#     step = list(zip(strike_indices, off_indices))

#     # single support from mid stance to toe off
#     ss_startend = list(zip(ms_indices, off_indices))

#     # swing from toe off to next heel strike
#     swing_startend = list(zip(off_indices[:-1], strike_indices[1:]))

#     ds_frames, ss_frames, swing_frames, step_frames = [], [], [], []

#     for s, e in ds_startend:
#         ds_frames.append(insole_all["insole_l_flipped"][s:e])
#     for s, e in ss_startend:
#         ss_frames.append(insole_all["insole_l_flipped"][s:e])

#     for s, e in swing_startend:
#         swing_frames.append(insole_all["insole_l_flipped"][s:e])

#     for s, e in step:
#         step_frames.append(np.array(insole_all["insole_l_flipped"][s:e]))

#     return {
#         "double": ds_frames,
#         "single": ss_frames,
#         "swing": swing_frames,
#         "step": step_frames,
#     }


# segments = produce_slice(strike_indices, off_indices, ms_indices, insole_data)

# steps = segments["step"]

# height_valleys_l = compute_height_seg_coords(steps[13], validate=True, show_plot=True)

# for i, step in enumerate(segments["step"]):
#     height_valleys_l = compute_height_seg_coords(step, validate=True, show_plot=False)

#     height_valleys_r = compute_height_seg_coords(step, validate=True, show_plot=False)

#     width_valleys_l = compute_width_seg_coords(
#         step, height_valleys_l, validate=True, show_plot=False
#     )
#     width_valleys_r = compute_width_seg_coords(
#         step, height_valleys_r, validate=True, show_plot=False
#     )
