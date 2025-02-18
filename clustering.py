import numpy as np
from pathlib import Path
import scipy.io as sio
import scipy
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

from matplotlib.animation import FuncAnimation
import cv2 as cv

import skimage
from skimage.measure import regionprops, label
from skimage.draw import ellipse
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu, threshold_li

from visualization import plot_heatmap_spatial


# insole_data_path = Path("data/112024")
# insole_data = sio.loadmat(insole_data_path / "gait_recording_112024_walk4.mat")

insole_data_path = Path("data/080624")
insole_data = sio.loadmat(insole_data_path / "gait_recording_080624_walk3.mat")


def convert_insole_spatial(
    insoleAll_l: np.ndarray, insoleAll_r: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # from Sam's MATLAB files
    # use FORTRAN row-major reordering
    insole_spatial_l = insoleAll_l.reshape((-1, 64, 16), order="F")
    insole_spatial_r = insoleAll_r.reshape((-1, 64, 16), order="F")

    # need to flip the top half of the heatmap up-down
    insole_spatial_l[:, :32, :] = np.flip(insole_spatial_l[:, :32, :], axis=1)
    insole_spatial_r[:, :32, :] = np.flip(insole_spatial_r[:, :32, :], axis=1)

    # also flip the right insole left-right
    # Sam doesn't do this IIRC but it looks right
    insole_spatial_r = np.flip(insole_spatial_r, axis=2)

    return insole_spatial_l, insole_spatial_r


insole_spatial_l, insole_spatial_r = convert_insole_spatial(
    insole_data["insoleAll_l"], insole_data["insoleAll_r"]
)

# plot_heatmap_spatial(insole_spatial_r, insole_spatial_l)


def compute_height_seg_coords(
    insole_spatial: np.ndarray, validate: bool = True, show_plot: bool = False
) -> np.ndarray:
    # mean across frames and columns
    colwise_mean = insole_spatial.mean(axis=(0, 2))

    # smoothing filter
    # polyorders 2 and 3 have same effect because they share a center
    # window_length > 5 causes noticable drift in extrema locations
    smoothed = scipy.signal.savgol_filter(colwise_mean, window_length=5, polyorder=2)

    # invert to get min instead of max
    valleys, valley_props = scipy.signal.find_peaks(-smoothed, prominence=0)

    # get the most prominent valleys in any order
    # invert to get max instead of min
    NVALLEYS = 2
    valleys_mask = np.argpartition(-valley_props["prominences"], NVALLEYS)[:NVALLEYS]

    peaks = None
    peak_props = None

    if validate:
        NPEAKS = 3
        peaks, peak_props = scipy.signal.find_peaks(smoothed, prominence=0)
        peaks_mask = np.argpartition(-peak_props["prominences"], NPEAKS)[:NPEAKS]

        peak_valley_sorted = np.sort(
            np.hstack((peaks[peaks_mask], valleys[valleys_mask]))
        )

        # we want the ordering to be peak, valley, peak, valley, peak
        if not np.all(np.diff(np.isin(peak_valley_sorted, peaks[peaks_mask]))):
            raise Exception("validation failed")

    if show_plot:
        plt.plot(colwise_mean, label="colwise mean")
        plt.plot(smoothed, label="smoothed")

        plt.plot(valleys, smoothed[valleys], "o", label="valleys")
        plt.vlines(
            x=valleys,
            ymin=smoothed[valleys],
            ymax=valley_props["prominences"] + smoothed[valleys],
            color="C2",
        )

        if peaks is not None and peak_props is not None:
            plt.plot(peaks, smoothed[peaks], "x", label="peaks")
            plt.vlines(
                x=peaks,
                ymin=smoothed[peaks] - peak_props["prominences"],
                ymax=smoothed[peaks],
                color="C3",
            )

        plt.legend()
        plt.show()

    return valleys[valleys_mask]


compute_height_seg_coords(insole_spatial_l, validate=False, show_plot=True)

#############################################
##### clean up to here

# %%

col_line = insole_spatial_l.mean(axis=(0, 2))

smoothed = scipy.signal.savgol_filter(col_line, 5, 2)

plt.plot(col_line)
plt.plot(smoothed)

plt.show()

valleys, _ = scipy.signal.find_peaks(
    -col_line,
)
peaks, props = scipy.signal.find_peaks(smoothed, prominence=0)
valleys = sorted(valleys, key=lambda idx: col_line[idx])[:2]

plt.plot(smoothed)
for peak in peaks:
    plt.axvline(peak)

plt.show()

plt.plot(col_line)
for valley in valleys:
    plt.axvline(valley)

plt.show()

heatmap = insole_spatial_l.mean(axis=0)

plt.imshow(heatmap)
for valley in valleys:
    plt.axhline(valley, color="red")
plt.show()


heatmap_sub = heatmap[: valleys[1]]

plt.imshow(heatmap_sub)
plt.show()

# %%
toe_spatial = insole_spatial_l[:, : valleys[1], :]
toe_line = toe_spatial.mean(axis=(0, 1))
valleys_toe, _ = scipy.signal.find_peaks(-toe_line)
valleys_toe = sorted(valleys_toe, key=lambda idx: toe_line[idx])[:2]

plt.plot(toe_line)
for valley in valleys_toe:
    plt.axvline(valley)
plt.show()

# %%
ball_spatial = insole_spatial_l[:, valleys[1] : valleys[0], :]
ball_line = ball_spatial.mean(axis=(0, 1))
valleys_ball, _ = scipy.signal.find_peaks(-ball_line)
valleys_ball = sorted(valleys_ball, key=lambda idx: ball_line[idx])[:2]

plt.plot(ball_line)
for valley in valleys_ball:
    plt.axvline(valley)
plt.show()

# %%
heel_spatial = insole_spatial_l[:, valleys[0] :, :]
heel_line = heel_spatial.mean(axis=(0, 1))
valleys_heel, _ = scipy.signal.find_peaks(-heel_line)
valleys_heel = sorted(valleys_heel, key=lambda idx: heel_line[idx])[:2]

plt.plot(heel_line)
for valley in valleys_heel:
    plt.axvline(valley)
plt.show()

# %%

heatmap = insole_spatial_l.mean(axis=0)

plt.imshow(heatmap)

plt.vlines(valleys_toe, 0, valleys[1], color="orange")
plt.vlines(valleys_ball, valleys[1], valleys[0], color="orange")
plt.vlines(valleys_heel, valleys[0], 64, color="orange")

for valley in valleys:
    plt.axhline(valley, color="red")
plt.show()
