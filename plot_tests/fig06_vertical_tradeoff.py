"""Why the height of a DOG is poorly determined and its horizontal position is not.

Question: why is the vertical position so much weaker than the horizontal one?
Message: moving a DOG north lengthens the ranges on one side of it and shortens
them on the other; no timing bias or ESV bias can copy that, so the horizontal
position is well determined (a). Raising a DOG shortens every range, which a
timing bias plus an ESV bias copy to within millimetres, so the height is poorly
determined (b).

Each point is one range of the survey. Moving the DOG by a vector d changes the
range by −d·u, u being the unit vector from the DOG to the vessel. A timing bias
changes every range by the same amount and an ESV bias dc changes a range R by
−R·dc/c; the pair that best copies each DOG movement is found by least squares.

Input: geometry of the paired epochs (final MCMC run and the GNSS file).
"""

import numpy as np

import _figure_common as fc
import _style as st

NAME = "fig06_vertical_tradeoff"
DOG = 3  # the other DOGs behave the same (numbers are printed)
MOVE_CM = 10.0
SOUND_SPEED_MPS = 1515.0
MOVE_COLOR = "black"
COPY_COLOR = "#D55E00"
PANELS = (
    (1, f"DOG moved {MOVE_CM:.0f} cm north: cannot be copied"),
    (2, f"DOG raised {MOVE_CM:.0f} cm: copied almost exactly"),
)


def geometry(data):
    """Unit vectors DOG → vessel (ENU), slant ranges (m), north offsets (km)."""
    to_enu = fc.enu_rotation_at(data["dog_ecef_m"])
    enu = (data["transducer_ecef_m"] - data["dog_ecef_m"]) @ to_enu.T
    slant_range = np.linalg.norm(enu, axis=1)
    return enu / slant_range[:, None], slant_range, enu[:, 1] / 1000.0


def best_copy(signature_cm, slant_range_m):
    """Best fit of a timing bias + ESV bias to a signature; fit, δt (µs), δc (mm/s)."""
    design = np.column_stack([np.ones_like(slant_range_m), slant_range_m])
    (constant_cm, slope_cm_per_m), *_ = np.linalg.lstsq(
        design, signature_cm, rcond=None
    )
    timing_us = constant_cm / 100.0 / SOUND_SPEED_MPS * 1e6
    esv_mmps = -slope_cm_per_m / 100.0 * SOUND_SPEED_MPS * 1e3
    return design @ (constant_cm, slope_cm_per_m), timing_us, esv_mmps


def main():
    st.apply()
    residuals, _ = fc.final_residuals(fc.pooled_run(fc.FINAL_RUN))
    fig, axes = st.figure(st.DOUBLE_MM, 72.0, ncols=2, sharey=True)
    for dog in st.DOGS:
        unit, slant_range, north_km = geometry(residuals[dog])
        for ax, (component, title) in zip(axes, PANELS, strict=True):
            signature = -MOVE_CM * unit[:, component]
            copy, timing_us, esv_mmps = best_copy(signature, slant_range)
            misfit_cm = np.sqrt(np.mean((signature - copy) ** 2))
            print(
                f"DOG {dog}, {'ENU'[component]} +{MOVE_CM:.0f} cm: best copy δt "
                f"{timing_us:+.1f} µs, δc {esv_mmps:+.1f} mm/s; misfit "
                f"{misfit_cm:.2f} cm rms of a {np.sqrt(np.mean(signature**2)):.2f} cm "
                f"rms signature"
            )
            if dog != DOG:
                continue
            points = {"s": 1.2, "linewidths": 0, "rasterized": True}
            ax.scatter(north_km, signature, color=MOVE_COLOR, **points)
            ax.scatter(north_km, copy, color=COPY_COLOR, **points)
            ax.plot([], [], "o", ms=3.5, color=MOVE_COLOR, label="DOG moved")
            ax.plot(
                [],
                [],
                "o",
                ms=3.5,
                color=COPY_COLOR,
                label=f"Best copy: $\\delta t$ = {fc.signed(timing_us, '+.0f')} µs, "
                f"$\\delta c$ = {fc.signed(esv_mmps, '+.1f')} mm/s",
            )
            ax.set_title(title, loc="left", pad=3)
            ax.annotate(
                f"Difference: {misfit_cm:.1f} cm rms",
                (0.98, 0.05),
                xycoords="axes fraction",
                ha="right",
                bbox={"facecolor": "white", "edgecolor": "0.7", "pad": 3},
            )
            ax.axhline(0.0, color="0.5", lw=0.6)
            ax.legend(loc="upper center")
            ax.set_xticks([-5, 0, 5])
            ax.set_xlabel(f"North offset of the vessel from DOG {DOG}, $N$ (km)")
    axes[0].set_ylim(-12.5, 16.5)
    axes[0].set_ylabel("Change of the range (cm)")
    for ax, letter in zip(axes, "ab", strict=True):
        st.panel_letter(ax, letter, y_pt=14.0)
    st.save(fig, NAME)


if __name__ == "__main__":
    main()
