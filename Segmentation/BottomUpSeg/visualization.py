import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d


def plot_cop_and_gait(gait):
    # Visualize COP and phase division
    gait["cop_x_r"] = gait["cop_x_r"][np.isfinite(gait["cop_x_r"])]
    gait["cop_y_r"] = gait["cop_y_r"][np.isfinite(gait["cop_y_r"])]
    gait["cop_x_l"] = gait["cop_x_l"][np.isfinite(gait["cop_x_l"])]
    gait["cop_y_l"] = gait["cop_y_l"][np.isfinite(gait["cop_y_l"])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("COP Trajectory - Left and Right Foot", fontsize=16)

    ax1.set_title("Right Foot COP")
    ax1.set_ylabel("COP X Position")
    ax1.set_xlabel("COP Y Position")
    ax1.set_ylim(np.max(gait["cop_x_r"]) + 0.1, np.min(gait["cop_x_r"]) - 0.1)
    ax1.set_xlim(np.min(gait["cop_y_r"]) - 0.1, np.max(gait["cop_y_r"]) + 0.1)

    ax2.set_title("Left Foot COP")
    ax2.set_ylabel("COP X Position")
    ax2.set_xlabel("COP Y Position")
    ax2.set_ylim(np.max(gait["cop_x_l"]) + 0.1, np.min(gait["cop_x_l"]) - 0.1)
    ax2.set_xlim(np.min(gait["cop_y_l"]) - 0.1, np.max(gait["cop_y_l"]) + 0.1)

    scatter_r = ax1.scatter([], [], color="blue", label="COP")
    scatter_l = ax2.scatter([], [], color="blue", label="COP")
    strike_r = ax1.scatter([], [], color="red", label="Strike", marker="x")
    off_r = ax1.scatter([], [], color="green", label="Off", marker="x")
    strike_l = ax2.scatter([], [], color="red", label="Strike", marker="x")
    off_l = ax2.scatter([], [], color="green", label="Off", marker="x")

    (line_r,) = ax1.plot([], [], color="blue", alpha=0.5, label="stance")
    (line_l,) = ax2.plot([], [], color="blue", alpha=0.5, label="stance")

    ax1.legend()
    ax2.legend()

    def update(frame):
        if frame < len(gait["strike_r"]) - 1:
            start_r = gait["strike_r"][frame]
            end_r = gait["strike_r"][frame + 1]
            stance_r = gait["off_r"][frame]

            line_r.set_data(
                gait["cop_y_r"][start_r:stance_r], gait["cop_x_r"][start_r:stance_r]
            )

            scatter_r.set_offsets(
                np.c_[gait["cop_y_r"][start_r:end_r], gait["cop_x_r"][start_r:end_r]]
            )
            strike_r.set_offsets(
                np.c_[
                    gait["cop_y_r"][gait["strike_r"][frame]],
                    gait["cop_x_r"][gait["strike_r"][frame]],
                ]
            )
            off_r.set_offsets(
                np.c_[
                    gait["cop_y_r"][gait["off_r"][frame] - 1],
                    gait["cop_x_r"][gait["off_r"][frame] - 1],
                ]
            )

        if frame < len(gait["strike_l"]) - 1:
            start_l = gait["strike_l"][frame]
            end_l = gait["strike_l"][frame + 1]
            stance_l = gait["off_l"][frame]

            line_l.set_data(
                gait["cop_y_l"][start_l:stance_l], gait["cop_x_l"][start_l:stance_l]
            )

            scatter_l.set_offsets(
                np.c_[gait["cop_y_l"][start_l:end_l], gait["cop_x_l"][start_l:end_l]]
            )
            strike_l.set_offsets(
                np.c_[
                    gait["cop_y_l"][gait["strike_l"][frame]],
                    gait["cop_x_l"][gait["strike_l"][frame]],
                ]
            )
            off_l.set_offsets(
                np.c_[
                    gait["cop_y_l"][gait["off_l"][frame] - 1],
                    gait["cop_x_l"][gait["off_l"][frame] - 1],
                ]
            )

        return scatter_r, scatter_l, strike_r, off_r, strike_l, off_l, line_r, line_l

    ani = FuncAnimation(
        fig,
        update,
        frames=max(len(gait["strike_r"]), len(gait["strike_l"])),
        blit=True,
        repeat=True,
        interval=1000,
    )

    plt.tight_layout()
    plt.show()


def plot_heatmap(insoleAll_r, insoleAll_l):
    ## based on the correspondence between the sensor and the actual position
    dim = [
        int(np.sqrt(insoleAll_r.shape[1]) * 2),
        int(np.sqrt(insoleAll_r.shape[1]) / 2),
    ]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))

    img_l = ax_l.imshow(
        np.zeros(dim),
        vmin=np.min(insoleAll_l),
        vmax=np.max(insoleAll_l),
        interpolation="nearest",
    )
    img_r = ax_r.imshow(
        np.zeros(dim),
        vmin=np.min(insoleAll_r),
        vmax=np.max(insoleAll_r),
        interpolation="nearest",
    )

    ax_l.set_title("Left Heatmap")
    ax_r.set_title("Right Heatmap")
    plt.colorbar(img_l, ax=ax_l)
    plt.colorbar(img_r, ax=ax_r)

    def update(frame):
        ## based on the correspondence between the sensor and the actual position
        insole_l = insoleAll_l[frame, :].reshape(dim, order="F")
        insole_l[:32, :] = np.flipud(insole_l[:32, :])
        insole_r = insoleAll_r[frame, :].reshape(dim, order="F")
        insole_r[:32, :] = np.flipud(insole_r[:32, :])

        img_l.set_array(insole_l)
        img_r.set_array(insole_r)

        return img_l, img_r

    ani = FuncAnimation(
        fig,
        update,
        frames=max(len(insoleAll_l), len(insoleAll_r)),
        blit=True,
        repeat=True,
        interval=10,
    )
    plt.show()


def plot_heatmap_with_COP(gait):
    insoleAll_r = gait["insole_r"]
    insoleAll_l = gait["insole_l"]

    dim = [
        int(np.sqrt(insoleAll_r.shape[1]) * 2),
        int(np.sqrt(insoleAll_r.shape[1]) / 2),
    ]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))

    img_l = ax_l.imshow(
        np.zeros(dim),
        vmin=np.min(insoleAll_l),
        vmax=np.max(insoleAll_l),
        interpolation="nearest",
    )
    img_r = ax_r.imshow(
        np.zeros(dim),
        vmin=np.min(insoleAll_r),
        vmax=np.max(insoleAll_r),
        interpolation="nearest",
    )

    ax_l.set_title("Left Heatmap")
    ax_r.set_title("Right Heatmap")
    plt.colorbar(img_l, ax=ax_l)
    plt.colorbar(img_r, ax=ax_r)

    cop_marker_l = ax_l.scatter([], [], color="red", label="COP")
    cop_marker_r = ax_r.scatter([], [], color="red", label="COP")

    def update(frame):
        ## based on the correspondence between the sensor and the actual position
        insole_l = insoleAll_l[frame, :].reshape(dim, order="F")
        insole_l[:32, :] = np.flipud(insole_l[:32, :])
        insole_r = insoleAll_r[frame, :].reshape(dim, order="F")
        # insole_r = np.fliplr(insole_r)
        # insole_r[:32, :] = np.flipud(insole_r[:32, :])

        img_l.set_array(insole_l)
        img_r.set_array(insole_r)

        cop_marker_l.set_offsets(np.c_[gait["cop_y_l"][frame], gait["cop_x_l"][frame]])
        cop_marker_r.set_offsets(np.c_[gait["cop_y_r"][frame], gait["cop_x_r"][frame]])
        return img_l, img_r, cop_marker_l, cop_marker_r

    ani = FuncAnimation(
        fig, update, frames=len(insoleAll_l), blit=True, repeat=True, interval=50
    )
    plt.show()


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
        frames=max(len(insole_spatial_l), len(insole_spatial_r)),
        blit=True,
        repeat=True,
        interval=10,
    )
    plt.show()
