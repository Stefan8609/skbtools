"""Surface GNSS: vessel height and satellite geometry over the survey.

Question: is the kinematic GNSS solution of the vessel continuous over the survey,
and how good was the satellite geometry?
Message: the ellipsoidal height of the vessel is continuous over 17 h (no steps at
the hourly file boundaries) and its 10-min median follows the tide; 9 to 22
satellites were tracked with a PDOP between 1.0 and 2.2.

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
MEDIAN_WINDOW_S = 600.0
PDOP_COLOR = "#D55E00"
# Radii within which neighbours are counted for the density colour.
DENSITY_RADIUS_H = 0.1
DENSITY_RADIUS_M = 0.05


def main():
    st.apply()
    processed = np.load(PROCESSED_FILE)
    gps_s = processed["GPS_data"]
    coordinates = processed["GPS_Coordinates"]
    height_m = np.column_stack(
        [ecef2geodetic(*coordinates[:, a, :].T)[2] for a in range(4)]
    )
    mean_height = height_m.mean(axis=1)
    quality = loadmat(ANTENNA1_FILE)
    quality_h = st.hours(
        (quality["days"].ravel() - fc.GNSS_DAY0_MJD) * 86400.0
        + quality["times"].ravel()
        - fc.GNSS_CLOCK_ADJUSTMENT_S
    )
    inside = (quality_h >= st.TIME_LIMITS_H[0]) & (quality_h <= st.TIME_LIMITS_H[1])
    satellites = quality["nsat_bundle"][:, 0][inside]
    pdop = quality["pdop"].ravel()[inside]

    fig, (ax_height, ax_quality) = st.figure(
        st.DOUBLE_MM,
        105.0,
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1]},
    )
    points = st.density_scatter(
        ax_height,
        st.hours(gps_s),
        mean_height,
        DENSITY_RADIUS_H,
        DENSITY_RADIUS_M,
        s=0.6,
        label="Mean of the four antennas, every epoch",
    )
    edges_s = np.arange(gps_s[0], gps_s[-1] + MEDIAN_WINDOW_S, MEDIAN_WINDOW_S)
    centres_s, median, _ = fc.binned_percentiles(
        gps_s, mean_height, edges_s, percentiles=(50.0,), min_count=60
    )
    ax_height.plot(
        st.hours(centres_s), median[:, 0], "k", lw=1.2, label="10-min median"
    )
    ax_height.set_ylabel("Ellipsoidal height $h$ (m)")
    ax_height.legend(loc="upper left", ncols=2, columnspacing=1.5, markerscale=3)
    bar_axes = ax_height.inset_axes([0.70, 0.90, 0.27, 0.045])
    colourbar = fig.colorbar(points, cax=bar_axes, orientation="horizontal")
    colourbar.set_label(st.DENSITY_LABEL, labelpad=1)
    ax_height.set_ylim(top=ax_height.get_ylim()[1] + 0.75)

    ax_quality.plot(quality_h[inside], satellites, color="black", lw=0.6)
    ax_quality.set_ylabel("Number of satellites")
    ax_quality.set_xlabel(st.TIME_LABEL)
    ax_quality.set_xlim(st.TIME_LIMITS_H)
    ax_quality.set_xticks(st.TIME_TICKS_H)
    ax_quality.tick_params(right=False)
    pdop_axis = ax_quality.twinx()
    pdop_axis.plot(quality_h[inside], pdop, color=PDOP_COLOR, lw=0.6)
    pdop_axis.set_ylabel("PDOP", color=PDOP_COLOR)
    pdop_axis.tick_params(axis="y", colors=PDOP_COLOR)
    pdop_axis.spines["right"].set_color(PDOP_COLOR)
    pdop_axis.grid(False)

    for a in range(4):
        print(f"antenna {a + 1}: median h {np.median(height_m[:, a]):.3f} m")
    print(
        f"10-min median height {np.nanmin(median):.2f} to {np.nanmax(median):.2f} m; "
        f"satellites {satellites.min():.0f}–{satellites.max():.0f}; PDOP "
        f"{pdop.min():.2f}–{pdop.max():.2f}; epochs kept {gps_s.size}"
    )
    for ax, letter in zip((ax_height, ax_quality), "ab", strict=True):
        st.panel_letter(ax, letter, x_pt=-34.0)
    st.save(fig, NAME)


if __name__ == "__main__":
    main()
