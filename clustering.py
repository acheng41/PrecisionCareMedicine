import numpy as np
from pathlib import Path
import scipy.io as sio
import scipy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.animation import FuncAnimation
import cv2 as cv

from visualization import plot_heatmap_spatial


# insole_data_path = Path("data/112024")
# insole_data = sio.loadmat(insole_data_path / "gait_recording_112024_walk4.mat")

# insole_data_path = Path("data/080624")
# insole_data = sio.loadmat(insole_data_path / "gait_recording_080624_walk4.mat")

insole_data_path = Path("data/022425")
insole_data = sio.loadmat(insole_data_path / "gait_recording_022425_walk2.mat")


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


plot_heatmap_spatial(insole_spatial_l, insole_spatial_r)


def compute_height_seg_coords(
    insole_spatial: np.ndarray, validate: bool = True, show_plot: bool = False
) -> np.ndarray:
    NVALLEYS = 2
    NPEAKS = NVALLEYS + 1

    # mean across frames and columns
    colwise_mean = insole_spatial.mean(axis=(0, 2))

    # remove outliers i.e. a row of dead pixels
    # important the dead pixels don't create a large valley in the middle of the heel
    # which will blow out the toes
    medianed = scipy.signal.medfilt(colwise_mean, kernel_size=3)
    # smoothing filter
    # polyorders 2 and 3 have same effect because they share a center
    # window_length > 5 causes noticable drift in extrema locations
    smoothed = scipy.signal.savgol_filter(medianed, window_length=5, polyorder=2)

    # invert to get min instead of max
    valleys, valley_props = scipy.signal.find_peaks(-smoothed, prominence=0)

    valleys_mask = np.argsort(valley_props["prominences"])[-NVALLEYS:]

    peaks, peak_props = scipy.signal.find_peaks(smoothed, prominence=0)
    peaks_mask = np.argsort(peak_props["prominences"])[-NPEAKS:]

    if show_plot:
        plt.plot(colwise_mean, label="colwise mean")
        plt.plot(medianed, label="medianed")
        plt.plot(smoothed, label="smoothed")

        plt.plot(
            valleys[valleys_mask], smoothed[valleys[valleys_mask]], "o", label="valleys"
        )
        plt.vlines(
            x=valleys[valleys_mask],
            ymin=smoothed[valleys[valleys_mask]],
            ymax=valley_props["prominences"][valleys_mask]
            + smoothed[valleys[valleys_mask]],
            color="C3",
        )

        plt.plot(peaks[peaks_mask], smoothed[peaks[peaks_mask]], "x", label="peaks")
        plt.vlines(
            x=peaks[peaks_mask],
            ymin=smoothed[peaks[peaks_mask]] - peak_props["prominences"][peaks_mask],
            ymax=smoothed[peaks[peaks_mask]],
            color="C4",
        )

        plt.legend()
        plt.show()

    if validate:
        peak_valley_sorted = np.sort(
            np.hstack((peaks[peaks_mask], valleys[valleys_mask]))
        )

        # we want the ordering to be peak, valley, peak, valley, peak
        if not np.all(np.diff(np.isin(peak_valley_sorted, peaks[peaks_mask]))):
            raise Exception("validation failed")

    return valleys[valleys_mask]


height_valleys_l = compute_height_seg_coords(
    insole_spatial_l, validate=False, show_plot=True
)

height_valleys_r = compute_height_seg_coords(
    insole_spatial_r, validate=False, show_plot=True
)


def compute_width_seg_coords(
    insole_spatial: np.ndarray,
    height_valleys: np.ndarray,
    validate: bool = True,
    show_plot: bool = False,
) -> np.ndarray:
    NVALLEYS = np.array([1, 1, 0])
    NPEAKS = NVALLEYS + 1
    TITLES = ["Toe", "Ball", "Heel"]

    # add 0 and len as bounds for window slide
    valleys_pad = np.r_[0, height_valleys, insole_spatial.shape[1]]

    width_valleys = []

    for i, window in enumerate(
        np.lib.stride_tricks.sliding_window_view(valleys_pad, 2)
    ):
        # mean across frames and rows
        rowwise_mean = insole_spatial[:, window[0] : window[1], :].mean(axis=(0, 1))

        # remove outliers i.e. a row of dead pixels
        # important the dead pixels don't create a large valley in the middle of the heel
        # which will blow out the toes
        medianed = scipy.signal.medfilt(rowwise_mean, kernel_size=3)
        # smoothing filter
        # polyorders 2 and 3 have same effect because they share a center
        # window_length > 5 causes noticable drift in extrema locations
        smoothed = scipy.signal.savgol_filter(medianed, window_length=5, polyorder=2)

        # invert to get min instead of max
        valleys, valley_props = scipy.signal.find_peaks(-smoothed, prominence=0)

        valleys_mask = np.argsort(valley_props["prominences"])[-NVALLEYS[i] :]

        peaks, peak_props = scipy.signal.find_peaks(smoothed, prominence=0)
        peaks_mask = np.argsort(peak_props["prominences"])[-NPEAKS[i] :]

        if show_plot:
            plt.plot(rowwise_mean, label="rowwise mean")
            plt.plot(medianed, label="medianed")
            plt.plot(smoothed, label="smoothed")

            plt.plot(
                valleys[valleys_mask],
                smoothed[valleys[valleys_mask]],
                "o",
                label="valleys",
            )
            plt.vlines(
                x=valleys[valleys_mask],
                ymin=smoothed[valleys[valleys_mask]],
                ymax=valley_props["prominences"][valleys_mask]
                + smoothed[valleys[valleys_mask]],
                color="C3",
            )

            plt.plot(peaks[peaks_mask], smoothed[peaks[peaks_mask]], "x", label="peaks")
            plt.vlines(
                x=peaks[peaks_mask],
                ymin=smoothed[peaks[peaks_mask]]
                - peak_props["prominences"][peaks_mask],
                ymax=smoothed[peaks[peaks_mask]],
                color="C4",
            )

            plt.title(TITLES[i])

            plt.legend()
            plt.show()

        if validate:
            peak_valley_sorted = np.sort(
                np.hstack((peaks[peaks_mask], valleys[valleys_mask]))
            )

            # we want the ordering to be peak, valley, peak, etc.
            if not np.all(np.diff(np.isin(peak_valley_sorted, peaks[peaks_mask]))):
                raise Exception("validation failed")

        width_valleys.append(valleys)

    # don't keep empty heel valleys array
    return np.vstack(width_valleys[:-1])


width_valleys_l = compute_width_seg_coords(
    insole_spatial_l, height_valleys_l, validate=True, show_plot=True
)
width_valleys_r = compute_width_seg_coords(
    insole_spatial_r, height_valleys_r, validate=True, show_plot=True
)


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
