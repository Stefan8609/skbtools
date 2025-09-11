import numpy as np
import matplotlib.pyplot as plt
from data import gps_data_path
from plotting.save import save_plot
from geometry.ECEF_Geodetic import ECEF_Geodetic

from Inversion_Workflow.Bermuda.Initialize_Bermuda_Data import (
    running_median,
    running_abs_dev,
    load_and_process_data,
)


def filtered_elevation_plots(window=5000, save=False, paper=False):
    if save and paper:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern"],
                "font.size": 14,
                "mathtext.fontset": "cm",
                "text.latex.preamble": r"\usepackage[utf8]{inputenc}"
                "\n"
                r"\usepackage{textcomp}",
            }
        )

    paths = [
        gps_data_path("GPS_Data/Unit1-camp_bis.mat"),
        gps_data_path("GPS_Data/Unit2-camp_bis.mat"),
        gps_data_path("GPS_Data/Unit3-camp_bis.mat"),
        gps_data_path("GPS_Data/Unit4-camp_bis.mat"),
    ]

    GNSS_start = 0
    GNSS_end = 100
    print(f"Loading GPS Data between hours {GNSS_start} and {GNSS_end}")

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
    filtered_data = np.array(filtered_data)

    print("Data Loaded")

    # Filter data based on elevation
    print("Starting Filtering Data")
    elev_upper = -35
    elev_lower = -38
    mask = np.array(
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
    )
    indices = np.where(mask[0])[0]
    filtered_data = filtered_data[:, :, indices]

    mask = np.ones(filtered_data.shape[2], dtype=bool)
    for i in range(4):
        elev = filtered_data[i, 4, :]
        median_elev = running_median(elev, window)
        abs_dev = running_abs_dev(elev, window)
        mask &= (elev >= median_elev - 2 * abs_dev) & (
            elev <= median_elev + 2 * abs_dev
        )
    indices = np.where(mask)[0]
    filtered_data = filtered_data[:, :, indices]

    print("End of Filtering Data")
    # Initialize Coordinates in form of Geiger's Method
    GPS_Coordinates = np.zeros((len(filtered_data[0, 0]), 4, 3))
    for i in range(len(filtered_data[0, 0])):
        for j in range(4):
            GPS_Coordinates[i, j, 0] = filtered_data[j, 1, i]
            GPS_Coordinates[i, j, 1] = filtered_data[j, 2, i]
            GPS_Coordinates[i, j, 2] = filtered_data[j, 3, i]

    # Initialize time-tagged data for GPS and CDOG
    GPS_data = filtered_data[0, 0, :]

    print("Starting Plotting Data")

    alpha = {0: "A", 1: "B", 2: "C", 3: "D"}
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Track per-unit masks and an overall cumulative mask
    global_mask = np.ones(filtered_data.shape[2], dtype=bool)

    for i in range(4):
        row, col = divmod(i, 2)
        elevation = filtered_data[i, 4, :]

        # Scatter raw elevation data
        axs[row, col].scatter(
            GPS_data, elevation, s=1, color="blue", label=r"Elevation Data"
        )

        # Rolling statistics and per-unit mask
        median_elev = running_median(elevation, window=window)
        abs_dev = running_abs_dev(elevation, window=window)
        mask_i = (elevation >= median_elev - 2 * abs_dev) & (
            elevation <= median_elev + 2 * abs_dev
        )
        global_mask &= mask_i

        # Bands and median
        upper_band = median_elev + 2 * abs_dev
        lower_band = median_elev - 2 * abs_dev
        axs[row, col].plot(
            GPS_data, median_elev, color="red", linewidth=2, label=r"Running Median"
        )
        axs[row, col].plot(
            GPS_data, upper_band, color="orange", label=r"2 Absolute Deviations"
        )
        axs[row, col].plot(GPS_data, lower_band, color="orange")

        # Titles and labels
        axs[row, col].set_title(f"GPS Unit {i + 1} Elevation")
        axs[row, col].set_xlabel(r"Time (s)")
        axs[row, col].set_ylabel(r"Elevation (m)")
        axs[row, col].set_ylim(-39, -34)

        # Subplot tag (A, B, C, D)
        axs[row, col].text(
            0.02,
            0.93,
            f"{alpha[i]}",
            transform=axs[row, col].transAxes,
        )
        if i == 0:
            axs[row, col].legend(loc="lower right")

    total_global = int(filtered_data.shape[2])
    kept_global = int(global_mask.sum())
    removed_global = total_global - kept_global

    axs[0, 1].text(
        0.95,
        0.05,
        f"Removed: {removed_global} / {total_global} points\n"
        f"Window: {window} samples\n"
        f"Overlap: {window - 1} samples",
        transform=axs[0, 1].transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
    )
    plt.tight_layout()
    plt.show()

    "End of Plotting Data"
    if save:
        if paper:
            save_plot(
                fig, func_name="Filtered_Elevation_Paper", subdir="Figs", ext="pdf"
            )
        else:
            save_plot(fig, func_name="Filtered_Elevation", subdir="Figs", ext="pdf")

    """ECEF vs normal elevation plot for check"""

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i in range(4):
        row, col = divmod(i, 2)
        lat, lon, converted_elevation = ECEF_Geodetic(GPS_Coordinates[:, i, :])
        elevation = filtered_data[i, 4, :]
        axs[row, col].scatter(
            GPS_data,
            converted_elevation - elevation,
            s=1,
            color="blue",
            label=r"Elevation Difference",
        )
    plt.tight_layout()
    plt.show()

    if save:
        if paper:
            save_plot(
                fig, func_name="Filtered_Elevation_Paper", subdir="Figs", ext="pdf"
            )
        else:
            save_plot(fig, func_name="Filtered_Elevation", subdir="Figs", ext="pdf")


def barycenter_elevation_plot(save=False, paper=False):
    data = np.load(gps_data_path("GPS_Data/Processed_GPS_Receivers_full.npz"))
    GPS_Coordinates = data["GPS_Coordinates"]
    GPS_data = data["GPS_data"]

    GPS_barycenter = np.mean(GPS_Coordinates, axis=1)

    lat, lon, elev = ECEF_Geodetic(GPS_barycenter)
    print(lat, lon)
    print(elev)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(GPS_data, elev, s=1, color="blue", label=r"Barycenter Elevation")
    ax.set_title("Barycenter Elevation over Time")
    ax.set_xlabel(r"Time (s)")
    ax.set_ylabel(r"Elevation (m)")
    # ax.set_ylim(-39, -34)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


"""
Difference between converted elevation and matlab file
elevation is +/- 1 mm
"""
if __name__ == "__main__":
    # filtered_elevation_plots(window=5000, save=False, paper=False)
    barycenter_elevation_plot(save=False)
