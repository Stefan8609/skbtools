import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import colors

from data import gps_data_path, gps_output_path


def running_median(data, window=50):
    half = window // 2
    n = len(data)
    result = np.empty(n)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        result[i] = np.median(data[start:end])
    return result


def running_abs_dev(data, window=50):
    half = window // 2
    n = len(data)
    result = np.empty(n)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window_data = data[start:end]
        med = np.median(window_data)
        result[i] = np.median(np.abs(window_data - med))
    return result


def load_and_process_data(path, GNSS_start, GNSS_end):
    data = sio.loadmat(path)
    days = data["days"].flatten() - 59015
    times = data["times"].flatten()
    datetimes = (days * 24 * 3600) + times

    condition = (datetimes / 3600 >= GNSS_start) & (datetimes / 3600 <= GNSS_end)

    datetimes = datetimes[condition]
    x = data["x"].flatten()[condition]
    y = data["y"].flatten()[condition]
    z = data["z"].flatten()[condition]
    elev = data["elev"].flatten()[condition]

    return datetimes, x, y, z, elev


def load_common_filtered_data(GNSS_start=25, GNSS_end=39):
    paths = [
        gps_data_path("GPS_Data/Unit1-camp_bis.mat"),
        gps_data_path("GPS_Data/Unit2-camp_bis.mat"),
        gps_data_path("GPS_Data/Unit3-camp_bis.mat"),
        gps_data_path("GPS_Data/Unit4-camp_bis.mat"),
    ]

    all_data = [load_and_process_data(path, GNSS_start, GNSS_end) for path in paths]

    common_datetimes = set(all_data[0][0])
    for data in all_data[1:]:
        common_datetimes.intersection_update(data[0])
    common_datetimes = sorted(common_datetimes)

    filtered_data = []
    for datetimes, x, y, z, elev in all_data:
        mask = np.isin(datetimes, common_datetimes)
        filtered_data.append(
            [
                np.array(datetimes)[mask],
                np.array(x)[mask],
                np.array(y)[mask],
                np.array(z)[mask],
                np.array(elev)[mask],
            ]
        )

    return np.array(filtered_data, dtype=float)


def save_elevation_plot_cache(
    GNSS_start=25,
    GNSS_end=39,
    window=5000,
    elev_upper=-35,
    elev_lower=-38,
    save_path="bermuda_elevation_plot_cache.npz",
):
    filtered_data = load_common_filtered_data(GNSS_start, GNSS_end)

    coarse_mask = np.array(
        [
            (filtered_data[0, 4, :] < elev_upper)
            & (filtered_data[0, 4, :] > elev_lower)
            & (filtered_data[1, 4, :] < elev_upper)
            & (filtered_data[1, 4, :] > elev_lower)
            & (filtered_data[2, 4, :] < elev_upper)
            & (filtered_data[2, 4, :] > elev_lower)
            & (filtered_data[3, 4, :] < elev_upper)
            & (filtered_data[3, 4, :] > elev_lower)
        ]
    )[0]

    filtered_data = filtered_data[:, :, np.where(coarse_mask)[0]]

    save_dict = {
        "GNSS_start": np.array(GNSS_start),
        "GNSS_end": np.array(GNSS_end),
        "window": np.array(window),
        "elev_upper": np.array(elev_upper),
        "elev_lower": np.array(elev_lower),
    }

    for i in range(4):
        time_vals = filtered_data[i, 0, :]
        elevation = filtered_data[i, 4, :]

        median_elev = running_median(elevation, window=window)
        abs_dev = running_abs_dev(elevation, window=window)

        upper_band = median_elev + 2 * abs_dev
        lower_band = median_elev - 2 * abs_dev
        keep_mask = (elevation >= lower_band) & (elevation <= upper_band)
        reject_mask = ~keep_mask

        prefix = f"unit{i + 1}_"
        save_dict[f"{prefix}time_vals"] = time_vals
        save_dict[f"{prefix}elevation"] = elevation
        save_dict[f"{prefix}median_elev"] = median_elev
        save_dict[f"{prefix}abs_dev"] = abs_dev
        save_dict[f"{prefix}upper_band"] = upper_band
        save_dict[f"{prefix}lower_band"] = lower_band
        save_dict[f"{prefix}keep_mask"] = keep_mask
        save_dict[f"{prefix}reject_mask"] = reject_mask

    np.savez_compressed(save_path, **save_dict)
    print(f"Saved cache to {save_path}")


def plot_elevation_density_from_cache(
    cache_path="bermuda_elevation_plot_cache.npz",
    gridsize=180,
    figsize=(14, 8),
    save_path=None,
):
    cache = np.load(cache_path)

    fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    axs = axs.ravel()
    alpha = {0: "A", 1: "B", 2: "C", 3: "D"}

    median_color = "#2A6F97"
    cutoff_color = "orange"
    # median_color = "#2A6F97"
    # cutoff_color = "#D98F00"
    hard_cut_color = "#7A8B5B"
    rejected_color = "#C8553D"

    elev_upper = float(cache["elev_upper"])
    elev_lower = float(cache["elev_lower"])

    hb = None

    for i in range(4):
        ax = axs[i]
        prefix = f"unit{i + 1}_"

        time_vals = cache[f"{prefix}time_vals"]
        time_hours = time_vals / 3600.0
        elevation = cache[f"{prefix}elevation"]
        median_elev = cache[f"{prefix}median_elev"]
        upper_band = cache[f"{prefix}upper_band"]
        lower_band = cache[f"{prefix}lower_band"]
        keep_mask = cache[f"{prefix}keep_mask"]
        reject_mask = cache[f"{prefix}reject_mask"]

        hb = ax.hexbin(
            time_hours[keep_mask],
            elevation[keep_mask],
            gridsize=gridsize,
            mincnt=1,
            cmap="viridis",
            norm=colors.PowerNorm(gamma=0.5),
            linewidths=0,
        )

        ax.scatter(
            time_hours[reject_mask],
            elevation[reject_mask],
            s=3,
            c=rejected_color,
            alpha=0.10,
            edgecolors="none",
            label="Rejected points" if i == 0 else None,
            zorder=3,
        )

        ax.fill_between(
            time_hours,
            lower_band,
            upper_band,
            color=cutoff_color,
            alpha=0.10,
            zorder=2,
        )

        ax.plot(
            time_hours,
            median_elev,
            color=median_color,
            linewidth=2.2,
            label="Running median" if i == 0 else None,
            zorder=4,
        )

        ax.plot(
            time_hours,
            upper_band,
            color=cutoff_color,
            linewidth=1.7,
            linestyle="--",
            label=r"Median $\pm 2$ MAD" if i == 0 else None,
            zorder=4,
        )
        ax.plot(
            time_hours,
            lower_band,
            color=cutoff_color,
            linewidth=1.7,
            linestyle="--",
            zorder=4,
        )

        ax.axhline(
            elev_upper,
            color=hard_cut_color,
            linestyle=":",
            linewidth=1.5,
            label="Initial hard cutoffs" if i == 0 else None,
        )
        ax.axhline(
            elev_lower,
            color=hard_cut_color,
            linestyle=":",
            linewidth=1.5,
        )

        removed = np.count_nonzero(reject_mask)
        total = elevation.size

        ax.set_title(f"GPS Unit {i + 1} Elevation", fontsize=18, pad=10)
        ax.set_ylim(-39, -34)
        ax.grid(alpha=0.10, linestyle="--")

        # Only outer labels
        if i in [2, 3]:
            ax.set_xlabel("Time (hours)", fontsize=18)
        if i in [0, 2]:
            ax.set_ylabel("Elevation (m)", fontsize=18)

        ax.tick_params(labelsize=14)

        ax.text(
            0.02,
            0.95,
            alpha[i],
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                alpha=0.65,
                edgecolor="none",
            ),
        )

        ax.text(
            0.98,
            0.04,
            f"Removed: {removed} / {total}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=18,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.70,
                edgecolor="none",
            ),
        )

    # Leave room for legend and colorbar
    fig.subplots_adjust(
        left=0.07,
        right=0.90,
        bottom=0.08,
        top=0.88,
        wspace=0.10,
        hspace=0.18,
    )

    # Dedicated colorbar axis on the right
    cax = fig.add_axes([0.92, 0.15, 0.025, 0.70])
    cbar = fig.colorbar(hb, cax=cax)
    cbar.set_label("Point Density", fontsize=18)
    cbar.ax.tick_params(labelsize=11)

    # Figure-level legend above subplots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=False,
        fontsize=18,
    )
    plt.show()


if __name__ == "__main__":
    data_path = gps_output_path("Plot_Data/bermuda_elevation_plot_cache.npz")

    # save_elevation_plot_cache(
    #     GNSS_start=25,
    #     GNSS_end=39,
    #     window=5000,
    #     elev_upper=-35,
    #     elev_lower=-38,
    #     save_path=data_path,
    # )

    plot_elevation_density_from_cache(
        cache_path=data_path,
        gridsize=50,
    )
