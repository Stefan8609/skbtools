"""Candidate GNSS figure 3: the four antennas as a rigid body on the vessel.

Question: how well do the four antenna positions fit a rigid vessel, in which
direction do they scatter, and where does the transducer sit relative to them?
Message: at every epoch the four positions match one rigid grid to about 2 cm; the
scatter of each antenna about its grid point is a compact, nearly isotropic
horizontal cloud, and is somewhat larger vertically.

Each epoch's four positions are mapped onto the vessel frame by a rigid-body fit
(fc.receivers_in_body_frame); the scatter about each grid point is enlarged by
MAGNIFICATION in panel a so it can be seen at the scale of the vessel.

Inputs: Processed_GPS_Receivers_DOG_1.npz, bermuda_data.GPS1_TO_OTHERS and the
lever arm of the final MCMC run (transducer position).
"""

import numpy as np

import _figure_common as fc
import _style as st
from bermuda_data import GPS1_TO_OTHERS

NAME = "figG3_gnss_rigid_body"
PROCESSED_FILE = fc.REPO / "Bermuda_Data/GPS_Data/Processed_GPS_Receivers_DOG_1.npz"
MAGNIFICATION = 60.0
SCALE_BAR_CM = 5.0
PROBABILITY = 0.95
N_SHOWN = 800  # epochs drawn per antenna (all are used for the statistics)
SEED = 20261121
VERTICAL_LIMIT_CM = 10.0
ANTENNA_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")


def main():
    st.apply()
    coordinates = np.load(PROCESSED_FILE)["GPS_Coordinates"]
    body = fc.receivers_in_body_frame(coordinates, GPS1_TO_OTHERS)
    misfit_cm = (body - GPS1_TO_OTHERS) * 100.0  # (epoch, antenna, xyz)
    lever = fc.pooled_run(fc.FINAL_RUN)["lever"].mean(axis=0)
    shown = np.random.default_rng(SEED).choice(body.shape[0], N_SHOWN, replace=False)

    fig, (ax_plan, ax_up) = st.figure(
        st.DOUBLE_MM, 82.0, ncols=2, gridspec_kw={"width_ratios": [1.6, 1.0]}
    )
    edges_cm = np.linspace(-VERTICAL_LIMIT_CM, VERTICAL_LIMIT_CM, 61)
    for a, color in enumerate(ANTENNA_COLORS):
        centre = GPS1_TO_OTHERS[a, :2]
        scaled = MAGNIFICATION * misfit_cm[shown, a, :2] / 100.0
        ax_plan.scatter(
            *(centre + scaled).T,
            s=1.2,
            color=color,
            alpha=0.5,
            linewidths=0,
            rasterized=True,
        )
        covariance = np.cov(misfit_cm[:, a, :2].T)
        outline = fc.ellipse_points((0.0, 0.0), covariance, PROBABILITY)
        ax_plan.plot(*(centre + MAGNIFICATION * outline / 100.0).T, "k", lw=0.9)
        ax_plan.annotate(
            f"Antenna {a + 1}", centre, (0, 17), textcoords="offset points", ha="center"
        )
        ax_up.hist(
            misfit_cm[:, a, 2],
            bins=edges_cm,
            density=True,
            histtype="step",
            color=color,
            lw=1.2,
            label=f"Antenna {a + 1}",
        )
        sd = misfit_cm[:, a].std(axis=0)
        print(
            f"antenna {a + 1}: misfit sd x, y, z = {sd.round(2)} cm; rms 3-D "
            f"{np.sqrt(np.mean(np.sum(misfit_cm[:, a] ** 2, axis=1))):.2f} cm"
        )
    ax_plan.plot(*lever[:2], "*", ms=10, mfc="#ffd23f", mec="black", mew=0.8)
    ax_plan.annotate(
        "Transducer", lever[:2], (0, 9), textcoords="offset points", ha="center"
    )
    bar_m = MAGNIFICATION * SCALE_BAR_CM / 100.0
    ax_plan.plot([-1.0 - bar_m, -1.0], [10.9, 10.9], "k", lw=2.0)
    ax_plan.annotate(
        f"{SCALE_BAR_CM:.0f} cm of scatter\n(enlarged {MAGNIFICATION:.0f} times)",
        (-1.0 - bar_m / 2, 10.9),
        (0, 4),
        textcoords="offset points",
        ha="center",
    )
    ax_plan.set_aspect("equal")
    ax_plan.set_xlim(-16.0, 3.0)
    ax_plan.set_ylim(-11.5, 12.5)
    ax_plan.invert_yaxis()
    ax_plan.set_xlabel("Vessel frame $x$ (m)")
    ax_plan.set_ylabel("Vessel frame $y$ (m)")
    ax_plan.plot([], [], "k", lw=0.9, label="95 % ellipse of the scatter")
    ax_plan.legend(loc="upper left")
    ax_up.set_xlabel("Vertical misfit to the rigid grid (cm)")
    ax_up.set_ylabel("Density (1/cm)")
    ax_up.legend(loc="upper right")
    print(f"transducer lever arm (final run mean): {lever.round(3)} m")
    for ax, letter in zip((ax_plan, ax_up), "ab", strict=True):
        st.panel_letter(ax, letter)
    st.save(fig, NAME)


if __name__ == "__main__":
    main()
