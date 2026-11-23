"""Candidate GNSS figure 1: antenna heights over the survey.

Question: is the kinematic GNSS solution of the vessel continuous over the survey,
and does it resolve the motion of the vessel?
Message: the ellipsoidal height of the vessel is continuous over 14 h (no steps at
the hourly file boundaries); its 10-min median follows the tide, and the four
antennas follow the same swell second by second.

Inputs: Processed_GPS_Receivers_DOG_1.npz (PRIDE PPP-AR positions of the four
antennas) and PRIDE_final/Antenna1-camp.mat (PDOP and number of satellites).
"""

import numpy as np
from pymap3d import ecef2geodetic
from scipy.io import loadmat

import _figure_common as fc
import _style as st

NAME = "figG1_gnss_height"
PROCESSED_FILE = fc.REPO / "Bermuda_Data/GPS_Data/Processed_GPS_Receivers_DOG_1.npz"
ANTENNA1_FILE = fc.REPO / "Bermuda_Data/GPS_Data/PRIDE_final/Antenna1-camp.mat"
ANTENNA_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
ZOOM_START_H = 6.0
ZOOM_LENGTH_S = 90.0
MEDIAN_WINDOW_S = 600.0
TIME_LIMITS_H = (0.0, 14.0)


def main():
    st.apply()
    processed = np.load(PROCESSED_FILE)
    hours = st.hours(processed["GPS_data"])
    coordinates = processed["GPS_Coordinates"]
    height_m = np.column_stack(
        [ecef2geodetic(*coordinates[:, a, :].T)[2] for a in range(4)]
    )
    quality = loadmat(ANTENNA1_FILE)
    quality_s = (
        (quality["days"].ravel() - fc.GNSS_DAY0_MJD) * 86400.0
        + quality["times"].ravel()
        - fc.GNSS_CLOCK_ADJUSTMENT_S
    )
    quality_h = st.hours(quality_s)

    fig = st.plt.figure(figsize=(st.DOUBLE_MM / st.MM_PER_INCH, 105.0 / st.MM_PER_INCH))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.6], height_ratios=[1.0, 1.0])
    ax_all = fig.add_subplot(grid[0, :])
    ax_zoom = fig.add_subplot(grid[1, 0])
    ax_quality = fig.add_subplot(grid[1, 1])

    zoom = (hours >= ZOOM_START_H) & (hours <= ZOOM_START_H + ZOOM_LENGTH_S / 3600.0)
    mean_height = height_m.mean(axis=1)
    ax_all.scatter(
        hours, mean_height, s=0.3, color=st.DATA_BLUE, linewidths=0, rasterized=True
    )
    ax_all.plot(
        [],
        [],
        "o",
        ms=3.5,
        color=st.DATA_BLUE,
        label="Mean of the four antennas, every epoch",
    )
    edges_s = np.arange(
        processed["GPS_data"][0],
        processed["GPS_data"][-1] + MEDIAN_WINDOW_S,
        MEDIAN_WINDOW_S,
    )
    centres_s, median, _ = fc.binned_percentiles(
        processed["GPS_data"], mean_height, edges_s, percentiles=(50.0,), min_count=60
    )
    ax_all.plot(st.hours(centres_s), median[:, 0], "k", lw=1.2, label="10-min median")
    print(f"10-min median height: {np.nanmin(median):.2f} to {np.nanmax(median):.2f} m")
    for a, color in enumerate(ANTENNA_COLORS):
        ax_zoom.plot(
            (hours[zoom] - ZOOM_START_H) * 3600.0,
            height_m[zoom, a],
            ".-",
            color=color,
            lw=0.8,
            ms=3.0,
            label=f"Antenna {a + 1}",
        )
        print(
            f"antenna {a + 1}: median h {np.median(height_m[:, a]):.3f} m, sd "
            f"{height_m[:, a].std():.3f} m"
        )
    ax_all.axvspan(
        ZOOM_START_H, ZOOM_START_H + ZOOM_LENGTH_S / 3600.0, color="0.5", lw=1.0
    )
    ax_all.set_ylabel("Ellipsoidal height $h$ (m)")
    ax_all.legend(loc="upper center", ncols=2, columnspacing=1.5)
    ax_all.set_ylim(top=ax_all.get_ylim()[1] + 0.5)
    ax_all.set_xlim(TIME_LIMITS_H)
    ax_all.set_xticks(np.arange(0, 15, 2))
    ax_all.set_xlabel(st.TIME_LABEL)
    ax_zoom.set_xlabel(f"Time after {ZOOM_START_H:.0f} h (s)")
    ax_zoom.set_ylabel("Ellipsoidal height $h$ (m)")
    ax_zoom.legend(loc="upper center", ncols=2, columnspacing=1.0)
    ax_zoom.set_ylim(top=ax_zoom.get_ylim()[1] + 0.55)

    inside = (quality_h >= TIME_LIMITS_H[0]) & (quality_h <= TIME_LIMITS_H[1])
    satellites = quality["nsat_bundle"][:, 0]
    ax_quality.plot(quality_h[inside], satellites[inside], color="black", lw=0.6)
    ax_quality.set_ylabel("Number of satellites")
    ax_quality.set_xlabel(st.TIME_LABEL)
    ax_quality.set_xlim(TIME_LIMITS_H)
    ax_quality.set_xticks(np.arange(0, 15, 2))
    pdop_axis = ax_quality.twinx()
    pdop_axis.plot(
        quality_h[inside], quality["pdop"].ravel()[inside], color="#D55E00", lw=0.6
    )
    pdop_axis.set_ylabel("PDOP", color="#D55E00")
    pdop_axis.grid(False)
    print(
        f"satellites {satellites[inside].min():.0f}–{satellites[inside].max():.0f}, "
        f"PDOP {quality['pdop'].ravel()[inside].min():.2f}–"
        f"{quality['pdop'].ravel()[inside].max():.2f}; epochs kept {hours.size}"
    )
    for ax, letter in zip((ax_all, ax_zoom, ax_quality), "abc", strict=True):
        st.panel_letter(ax, letter, x_pt=-34.0)
    st.save(fig, NAME)


if __name__ == "__main__":
    main()
