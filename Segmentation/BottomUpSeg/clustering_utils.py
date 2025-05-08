import numpy as np
import scipy
import matplotlib.pyplot as plt


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
            valleys[valleys_mask],
            smoothed[valleys[valleys_mask]],
            "o",
            label="valleys",
            color="tab:red",
        )
        plt.vlines(
            x=valleys[valleys_mask],
            ymin=smoothed[valleys[valleys_mask]],
            ymax=valley_props["prominences"][valleys_mask]
            + smoothed[valleys[valleys_mask]],
            color="tab:red",
        )

        plt.plot(
            peaks[peaks_mask],
            smoothed[peaks[peaks_mask]],
            "x",
            label="peaks",
            color="tab:purple",
        )
        plt.vlines(
            x=peaks[peaks_mask],
            ymin=smoothed[peaks[peaks_mask]] - peak_props["prominences"][peaks_mask],
            ymax=smoothed[peaks[peaks_mask]],
            color="tab:purple",
        )
        plt.title("Height Segmentation")
        plt.xlabel("Index")
        plt.ylabel("Pressure")

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
        # medianed = scipy.signal.medfilt(rowwise_mean, kernel_size=3)
        # smoothing filter
        # polyorders 2 and 3 have same effect because they share a center
        # window_length > 5 causes noticable drift in extrema locations
        # smoothed = scipy.signal.savgol_filter(medianed, window_length=5, polyorder=2)
        smoothed = scipy.signal.savgol_filter(
            rowwise_mean, window_length=5, polyorder=2
        )

        # invert to get min instead of max
        valleys, valley_props = scipy.signal.find_peaks(-smoothed, prominence=0)

        valleys_mask = np.argsort(valley_props["prominences"])[-NVALLEYS[i] :]

        peaks, peak_props = scipy.signal.find_peaks(smoothed, prominence=0)
        peaks_mask = np.argsort(peak_props["prominences"])[-NPEAKS[i] :]

        if show_plot:
            plt.plot(rowwise_mean, label="rowwise mean")
            # plt.plot(medianed, label="medianed")
            plt.plot(smoothed, label="smoothed")

            plt.plot(
                valleys[valleys_mask],
                smoothed[valleys[valleys_mask]],
                "o",
                label="valleys",
                color="tab:red",
            )
            plt.vlines(
                x=valleys[valleys_mask],
                ymin=smoothed[valleys[valleys_mask]],
                ymax=valley_props["prominences"][valleys_mask]
                + smoothed[valleys[valleys_mask]],
                color="tab:red",
            )

            plt.plot(
                peaks[peaks_mask],
                smoothed[peaks[peaks_mask]],
                "x",
                label="peaks",
                color="tab:purple",
            )
            plt.vlines(
                x=peaks[peaks_mask],
                ymin=smoothed[peaks[peaks_mask]]
                - peak_props["prominences"][peaks_mask],
                ymax=smoothed[peaks[peaks_mask]],
                color="tab:purple",
            )

            plt.title(f"{TITLES[i]} Segmentation")
            plt.xlabel("Index")
            plt.ylabel("Pressure")

            plt.legend()
            plt.show()

        if validate:
            peak_valley_sorted = np.sort(
                np.hstack((peaks[peaks_mask], valleys[valleys_mask]))
            )

            # we want the ordering to be peak, valley, peak, etc.
            if not np.all(np.diff(np.isin(peak_valley_sorted, peaks[peaks_mask]))):
                raise Exception("validation failed")

        width_valleys.append(valleys[valleys_mask])

    # don't keep empty heel valleys array
    return np.vstack(width_valleys[:-1])


def compute_height_seg_coords_stroke(
    insole_spatial: np.ndarray, validate: bool = True, show_plot: bool = False
) -> np.ndarray:
    NVALLEYS = 1
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
            valleys[valleys_mask],
            smoothed[valleys[valleys_mask]],
            "o",
            label="valleys",
            color="tab:red",
        )
        plt.vlines(
            x=valleys[valleys_mask],
            ymin=smoothed[valleys[valleys_mask]],
            ymax=valley_props["prominences"][valleys_mask]
            + smoothed[valleys[valleys_mask]],
            color="tab:red",
        )

        plt.plot(
            peaks[peaks_mask],
            smoothed[peaks[peaks_mask]],
            "x",
            label="peaks",
            color="tab:purple",
        )
        plt.vlines(
            x=peaks[peaks_mask],
            ymin=smoothed[peaks[peaks_mask]] - peak_props["prominences"][peaks_mask],
            ymax=smoothed[peaks[peaks_mask]],
            color="tab:purple",
        )
        plt.title("Height Segmentation")
        plt.xlabel("Index")
        plt.ylabel("Pressure")

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


def compute_width_seg_coords_stroke(
    insole_spatial: np.ndarray,
    height_valleys: np.ndarray,
    validate: bool = True,
    show_plot: bool = False,
) -> np.ndarray:
    NVALLEYS = np.array([1, 0])
    NPEAKS = NVALLEYS + 1
    TITLES = ["Ball", "Heel"]

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
        # medianed = scipy.signal.medfilt(rowwise_mean, kernel_size=3)
        # smoothing filter
        # polyorders 2 and 3 have same effect because they share a center
        # window_length > 5 causes noticable drift in extrema locations
        # smoothed = scipy.signal.savgol_filter(medianed, window_length=5, polyorder=2)
        smoothed = scipy.signal.savgol_filter(
            rowwise_mean, window_length=5, polyorder=2
        )

        # invert to get min instead of max
        valleys, valley_props = scipy.signal.find_peaks(-smoothed, prominence=0)

        valleys_mask = np.argsort(valley_props["prominences"])[-NVALLEYS[i] :]

        peaks, peak_props = scipy.signal.find_peaks(smoothed, prominence=0)
        peaks_mask = np.argsort(peak_props["prominences"])[-NPEAKS[i] :]
        print(peaks_mask)

        if show_plot:
            plt.plot(rowwise_mean, label="rowwise mean")
            # plt.plot(medianed, label="medianed")
            plt.plot(smoothed, label="smoothed")

            plt.plot(
                valleys[valleys_mask],
                smoothed[valleys[valleys_mask]],
                "o",
                label="valleys",
                color="tab:red",
            )
            plt.vlines(
                x=valleys[valleys_mask],
                ymin=smoothed[valleys[valleys_mask]],
                ymax=valley_props["prominences"][valleys_mask]
                + smoothed[valleys[valleys_mask]],
                color="tab:red",
            )

            plt.plot(
                peaks[peaks_mask],
                smoothed[peaks[peaks_mask]],
                "x",
                label="peaks",
                color="tab:purple",
            )
            plt.vlines(
                x=peaks[peaks_mask],
                ymin=smoothed[peaks[peaks_mask]]
                - peak_props["prominences"][peaks_mask],
                ymax=smoothed[peaks[peaks_mask]],
                color="tab:purple",
            )

            plt.title(f"{TITLES[i]} Segmentation")
            plt.xlabel("Index")
            plt.ylabel("Pressure")

            plt.legend()
            plt.show()

        if validate:
            peak_valley_sorted = np.sort(
                np.hstack((peaks[peaks_mask], valleys[valleys_mask]))
            )

            # we want the ordering to be peak, valley, peak, etc.
            if not np.all(np.diff(np.isin(peak_valley_sorted, peaks[peaks_mask]))):
                raise Exception("validation failed")

        width_valleys.append(valleys[valleys_mask])

    # don't keep empty heel valleys array
    return np.vstack(width_valleys[:-1])
