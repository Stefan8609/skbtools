"""Lengths of the baselines between the GNSS antennas.

Message: the six antenna distances stay within 1-2 cm (robust sd) of their medians,
the precision of the kinematic positions.

Input: Processed_GPS_Receivers_DOG_1.npz (PRIDE PPP-AR positions, ECEF m).
"""

from itertools import combinations

import numpy as np
from scipy.stats import median_abs_deviation, norm

import _figure_common as fc
import _style as st

NAME = "figG2_gnss_baselines"
PROCESSED_FILE = fc.REPO / "Bermuda_Data/GPS_Data/Processed_GPS_Receivers_DOG_1.npz"
LIMIT_CM = 10.0
BIN_EDGES_CM = np.linspace(-LIMIT_CM, LIMIT_CM, 81)
# Radii within which neighbours are counted for the density colour.
DENSITY_RADIUS_H = 0.1
DENSITY_RADIUS_CM = 0.25


def main():
    """Draw and save the figure."""
    st.apply()
    processed = np.load(PROCESSED_FILE)
    hours = st.hours(processed["GPS_data"])
    coordinates = processed["GPS_Coordinates"]
    pairs = list(combinations(range(4), 2))
    fig, axes = st.figure(
        st.DOUBLE_MM,
        150.0,
        nrows=len(pairs),
        ncols=2,
        sharex="col",
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 5.0], "wspace": 0.03},
    )
    scatters = []
    for (ax_hist, ax_time), (first, second) in zip(axes, pairs, strict=True):
        length_m = np.linalg.norm(
            coordinates[:, first] - coordinates[:, second], axis=1
        )
        deviation_cm = (length_m - np.median(length_m)) * 100.0
        robust_sd_cm = median_abs_deviation(deviation_cm, scale="normal")
        outside = np.mean(np.abs(deviation_cm) > LIMIT_CM)
        ax_hist.hist(
            deviation_cm,
            bins=BIN_EDGES_CM,
            density=True,
            orientation="horizontal",
            color=st.DATA_BLUE,
            alpha=0.5,
        )
        grid_cm = np.linspace(-LIMIT_CM, LIMIT_CM, 300)
        ax_hist.plot(norm.pdf(grid_cm, 0.0, robust_sd_cm), grid_cm, "k", lw=1.0)
        ax_hist.invert_xaxis()
        scatters.append(
            st.density_scatter(
                ax_time, hours, deviation_cm, DENSITY_RADIUS_H, DENSITY_RADIUS_CM, s=0.5
            )
        )
        ax_time.set_title(
            f"Antennas {first + 1} and {second + 1}:  median length "
            f"{np.median(length_m):.3f} m,  robust sd {robust_sd_cm:.1f} cm",
            loc="left",
            pad=2,
        )
        print(
            f"antennas {first + 1}-{second + 1}: median {np.median(length_m):.4f} m, "
            f"robust sd {robust_sd_cm:.2f} cm, sd {deviation_cm.std():.2f} cm, "
            f"{100 * outside:.2f} % outside ±{LIMIT_CM:.0f} cm"
        )
    highest = max(scatter.get_array().max() for scatter in scatters)
    for scatter in scatters:  # one colour scale for the six baselines
        scatter.set_clim(1, highest)
    colourbar = fig.colorbar(scatters[0], ax=axes[:, 1].tolist(), aspect=45, pad=0.015)
    colourbar.set_label(st.DENSITY_LABEL)
    axes[0, 0].set_ylim(-LIMIT_CM, LIMIT_CM)
    axes[0, 1].set_xlim(st.TIME_LIMITS_H)
    axes[-1, 1].set_xticks(st.TIME_TICKS_H)
    axes[-1, 1].set_xlabel(st.TIME_LABEL)
    axes[-1, 0].set_xlabel("Density (1/cm)")
    fig.supylabel("Baseline length minus median (cm)", fontsize=st.FONT_PT)
    st.save(fig, NAME)


if __name__ == "__main__":
    main()
