"""FX-4 (remake of Fig. 4): observed and modelled one-way travel times, all DOGs.

Message: the modelled one-way travel times follow the observed ones through every
pass over each DOG, and none is shorter than the vertical travel time.

Answers R4 m3 (travel times in Fig. 4 shorter than the ~3.4 s vertical travel time
and inconsistent with Fig. 5 for DOG 3: values before applying δt), R1 4e/3f (time
axis unified with the other time series) and R3 #10 (DOG naming and markers).

Observed travel times follow Eq. A21, T_obs = t_DOG − δt − t_GPS, on the fixed pairs
of the final MCMC run with δt at its posterior mean; modelled travel times are the
forward model at the posterior mean. These are exactly the pairs whose residuals
FX-5 shows; the script asserts it (same pair count, same emission epochs, same
residuals as ``fc.final_residuals``) and asserts that no observed travel time is
shorter than the vertical travel time to the DOG at the fastest effective sound
speed of the ESV table. It also prints a cross-check against the independent pairing
saved by Bermuda_Data_time_bias.py.

Inputs: the chains of ``fc.FINAL_RUN`` in Bermuda_Data/Output/MCMC_Chains/ (via the
run's saved configuration: CDOG_Data/DOG{n}-camp.mat, GPS_Data/
Processed_GPS_Receivers_DOG_1.npz, ESV_Tables/global_table_esv_extended.mat) and
Bermuda_Data/Output/Plot_Data/integer_pick_metrics_DOG{1,3,4}.npz (cross-check only).
"""

import numpy as np
from matplotlib.lines import Line2D

import _figure_common as fc
from Inversion_Workflow.MCMC_Functions import MCMC_sampler as ms
from plotting import paper_style

# --- configuration ---------------------------------------------------------------
NAME = "fig_FX4_travel_times"
RUN = fc.FINAL_RUN
RUN_DIR = fc.CHAINS_DIR / RUN
PICK_METRICS = str(
    fc.REPO / "Bermuda_Data/Output/Plot_Data/integer_pick_metrics_DOG{}.npz"
)
GAP_S = 60.0  # the modelled line breaks where pairs are further apart
TRAVEL_TIME_LIMITS_S = (3.0, 9.0)
FIGURE_HEIGHT_MM = 100.0


def travel_times(pooled):
    """Observed (Eq. A21) and modelled travel times per DOG at the posterior mean.

    Returns {dog: dict(gps_time_s, observed_s, modelled_s, vertical_s)} and the
    sound speed (m/s) used for the vertical travel time.
    """
    model, problem = fc.run_model_and_problem(pooled)
    theta = pooled["theta_samples"].mean(axis=0)
    params = ms.unpack(theta, model)
    if np.any(params["drift_rate"] != 0):
        raise ValueError("T_obs below assumes no clock drift; this run has drift")
    trans = (problem["R_stacked"] @ params["lever"]).reshape(-1, 3) + problem["d"]
    gps = np.load(fc.REPO / "Bermuda_Data" / fc.run_config(pooled)["gps_file"])
    gps_time = gps["GPS_data"]
    # Fastest effective speed the forward model can produce: no travel time can be
    # shorter than the vertical distance at this speed.
    fastest_mps = problem["esv_matrix"].max() + max(0.0, params["esv_bias"].max())
    out = {}
    for j, dog in enumerate(model["config"]["dogs"]):
        sel = problem["pair_slices"][j]
        index = problem["pair_gps_index"][sel]
        dog_ecef = problem["CDOG_reference"] + params["CDOG_aug"][j]
        modelled = ms.dog_travel_times(
            dog_ecef,
            trans,
            params["esv_bias"][j],
            problem["dz_array"],
            problem["angle_array"],
            problem["esv_matrix"],
            problem.get("epoch_window"),
        )[index]
        # pair_time_base is t_DOG - offset - t_GPS; offset - time_bias = δt (Eq. 6).
        observed = problem["pair_time_base"][sel] + params["time_bias"][j]
        out[dog] = {
            "gps_time_s": gps_time[index],
            "observed_s": observed,
            "modelled_s": modelled,
            "vertical_s": fc.vertical_travel_time_s(
                dog_ecef, trans[index], fastest_mps
            ),
        }
    return out, fastest_mps


def check_against_fx5(series, fx5):
    """Assert the pairs and residuals equal those FX-5 plots (fc.final_residuals)."""
    for dog, s in series.items():
        if s["gps_time_s"].shape != fx5[dog]["gps_time_s"].shape:
            raise AssertionError(
                f"DOG {dog}: {s['gps_time_s'].size} pairs here, "
                f"{fx5[dog]['gps_time_s'].size} in FX-5"
            )
        if not np.array_equal(s["gps_time_s"], fx5[dog]["gps_time_s"]):
            raise AssertionError(f"DOG {dog}: emission epochs differ from FX-5")
        residual_cm = (s["observed_s"] - s["modelled_s"]) * ms.RANGE_CM_PER_S
        if not np.allclose(residual_cm, fx5[dog]["residual_cm"], atol=1e-6):
            raise AssertionError(f"DOG {dog}: residuals differ from FX-5")


def check_physical(series):
    """Assert no observed travel time is shorter than the vertical travel time."""
    for dog, s in series.items():
        short = s["observed_s"] < s["vertical_s"]
        if short.any():
            raise AssertionError(
                f"DOG {dog}: {short.sum()} travel times below the vertical travel "
                f"time (minimum {s['observed_s'].min():.4f} s)"
            )


def cross_check_driver(series):
    """Compare with the independent pairing of Bermuda_Data_time_bias.py (print)."""
    for dog, s in series.items():
        saved = np.load(PICK_METRICS.format(dog))
        emission = saved["GPS_clock"] - saved["GPS_full"] + saved["time_bias"]
        common, here, there = np.intersect1d(
            np.round(s["gps_time_s"], 3), np.round(emission, 3), return_indices=True
        )
        difference_ms = (s["observed_s"][here] - saved["CDOG_full"][there]) * 1e3
        print(
            f"DOG {dog}: {s['gps_time_s'].size} pairs here, "
            f"{saved['CDOG_full'].size} in the driver, {common.size} common; "
            f"median |ΔT_obs| = {np.median(np.abs(difference_ms)):.3f} ms"
        )


def draw(series):
    paper_style.apply()
    fig, axes = paper_style.figure(
        "double", FIGURE_HEIGHT_MM, nrows=3, sharex=True, sharey=True
    )
    fc.despine(*axes)
    for ax, (dog, s), letter in zip(axes, series.items(), "abc", strict=True):
        style = paper_style.DOG_STYLE[dog]
        hours = paper_style.hours_since_start(s["gps_time_s"])
        dots = ax.plot(
            hours,
            s["observed_s"],
            style["marker"],
            ms=2.5,
            mew=0,
            color=style["color"],
        )[0]
        paper_style.rasterize_dense(dots, hours.size)
        gap = np.flatnonzero(np.diff(s["gps_time_s"]) > GAP_S) + 1
        ax.plot(
            np.insert(hours, gap, np.nan),
            np.insert(s["modelled_s"], gap, np.nan),
            color="black",
            lw=0.5,
            label="Modelled (posterior mean)",
        )
        ax.axhline(
            s["vertical_s"].min(),
            color=fc.CONTEXT,
            ls="--",
            lw=0.6,
            label="Vertical travel time",
        )
        # Direct label in the corner that the early, near-vertical pass leaves empty.
        ax.text(0.008, 0.95, style["label"], transform=ax.transAxes, va="top")
        paper_style.panel_label(ax, letter)
    fig.supylabel("One-way travel time (s)", fontsize=paper_style.FONT_SIZE_PT)
    observed = Line2D(
        [],
        [],
        ls="none",
        marker="o",
        ms=3,
        color=fc.CONTEXT,
        mew=0,
        label="Observed (DOG colour)",
    )
    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles=[observed, *handles],
        loc="outside lower center",
        ncol=3,
        handlelength=1.6,
    )
    axes[-1].set_xlabel(paper_style.TIME_LABEL)
    axes[-1].set_xlim(*paper_style.TIME_LIMITS_H)
    axes[-1].set_xticks(np.arange(0, 15, 2))
    axes[-1].set_ylim(*TRAVEL_TIME_LIMITS_S)
    axes[-1].set_yticks([4, 6, 8])
    return fig


if __name__ == "__main__":
    fc.require(RUN_DIR, *(PICK_METRICS.format(dog) for dog in (1, 3, 4)))
    pooled = fc.pooled_run(RUN)
    series, fastest_mps = travel_times(pooled)
    fx5, _ = fc.final_residuals(pooled)
    check_against_fx5(series, fx5)
    check_physical(series)
    print(
        f"checks passed: pairs = FX-5 pairs; T_obs ≥ vertical at {fastest_mps:.2f} m/s"
    )
    for dog, s in series.items():
        print(
            f"DOG {dog}: {s['observed_s'].size} pairs; "
            f"T_obs {s['observed_s'].min():.4f}-{s['observed_s'].max():.4f}"
            f" s, vertical ≥ {s['vertical_s'].min():.4f} s "
            f"({s['vertical_s'].min() * fastest_mps / 1515.0:.4f} s at 1515 m/s)"
        )
    cross_check_driver(series)
    fc.save(draw(series), NAME)
