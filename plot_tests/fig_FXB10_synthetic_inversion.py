"""FX-B10 (remake of Fig. B10): one synthetic inversion on the Bermuda trajectory.

One realisation of ``Modular_Synthetic.modular_synthetic`` (cheap, about 5 s), run
here with a fixed seed: synthetic travel times on the real DOG 3 GNSS epochs and
track, σd = 2 cm per receiver, σt = 20 µs, generated with the reference ESV table
and inverted with the strongly perturbed one, unknown clock offset, lever arm and
DOG position solved by the Gauss–Newton offset search. (a) Observed and modelled
travel times; (b) range residual (observed − modelled, 1515 m/s), its median in
15-min bins and 16–84 % band; (c) residual histogram with the normal curve of the
same mean and sd. The errors of the recovered parameters and the residual sd are
printed for the caption.

Message: one synthetic inversion on the real track recovers the travel times to a
few centimetres of range, with residuals close to normal.

Reviewer comment: R1 4n (Fig. B10 crowded and overlapping): three large panels
replace the submitted eight-panel layout and its random zoom window.

Inputs: none on disk beyond the ESV tables and the GNSS file the synthetic reads.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import norm

import _figure_common as fc
from acoustics.synthetic_esv_tables import ESV_GENERATE, ESV_STRONG
from Inversion_Workflow.Synthetic.Modular_Synthetic import TRUE_LEVER, modular_synthetic
from plotting import paper_style

# --- configuration -----------------------------------------------------------------
SEED = 20261118
DOG_NUM = 3
TIME_NOISE_S = 20e-6
POSITION_NOISE_M = 0.02
GENERATE_ESV = ESV_GENERATE
INVERSION_ESV = ESV_STRONG  # submitted B10: the strongly perturbed table
DOWNSAMPLE = 5  # GNSS epochs kept: every 5th
OFFSET_RANGE_S = (1000.0, 10000.0)  # true clock offset drawn as in synthetic_trials
MEDIAN_BIN_H = 0.25
HEIGHT_MM = 110.0
OBSERVED_COLOR = paper_style.DOG_STYLE[DOG_NUM]["color"]
MODELLED_COLOR = "k"
NAME = "fig_FXB10_synthetic_inversion"
# -----------------------------------------------------------------------------------


def run_synthetic():
    """One seeded realisation; returns the series and parameter errors."""
    np.random.seed(SEED)
    true_offset = np.random.uniform(*OFFSET_RANGE_S)
    out = modular_synthetic(
        TIME_NOISE_S,
        POSITION_NOISE_M,
        0.0,
        0.0,
        esv1=GENERATE_ESV,
        esv2=INVERSION_ESV,
        generate_type=1,
        inversion_type=1,
        plot=False,
        DOG_num=DOG_NUM,
        downsample=DOWNSAMPLE,
        true_offset=true_offset,
    )
    inversion_result, observed_s, modelled_s = out[0], out[2], out[4]
    cdog_clock, true_dog, offset, lever = out[5], out[7], out[10], out[11]
    rotation = fc.enu_rotation_at(true_dog)
    return {
        "hours": paper_style.hours_since_start(cdog_clock),
        "observed_s": observed_s,
        "modelled_s": modelled_s,
        "residual_cm": (observed_s - modelled_s)
        * paper_style.RANGE_SOUND_SPEED_MPS
        * 100.0,
        "dog_error_enu_cm": rotation @ (inversion_result[:3] - true_dog) * 100.0,
        "lever_error_cm": (lever - TRUE_LEVER[1]) * 100.0,
        "dt_error_us": (offset - inversion_result[3] - true_offset) * 1e6,
    }


def main():
    result = run_synthetic()
    hours, residual = result["hours"], result["residual_cm"]
    sigma_cm = residual.std(ddof=1)
    dog_error, lever_error = result["dog_error_enu_cm"], result["lever_error_cm"]
    print(f"{len(residual)} pairs; residual sd {sigma_cm:.2f} cm")
    print("DOG error ENU (cm)", np.round(dog_error, 2), np.linalg.norm(dog_error))
    print("lever error (cm)", np.round(lever_error, 2), np.linalg.norm(lever_error))
    print(f"dt error {result['dt_error_us']:.1f} us")

    paper_style.apply()  # after the synthetic imports, which turn LaTeX text on
    fig = plt.figure(
        figsize=(
            paper_style.DOUBLE_COLUMN_MM / paper_style.MM_PER_INCH,
            HEIGHT_MM / paper_style.MM_PER_INCH,
        )
    )
    grid = fig.add_gridspec(2, 2, width_ratios=(4.0, 1.25), height_ratios=(1.2, 1.0))
    ax_tt = fig.add_subplot(grid[0, 0])
    ax_key = fig.add_subplot(grid[0, 1])
    ax_res = fig.add_subplot(grid[1, 0], sharex=ax_tt)
    ax_hist = fig.add_subplot(grid[1, 1], sharey=ax_res)

    observed = ax_tt.scatter(
        hours, result["observed_s"], s=1.5, color=OBSERVED_COLOR, lw=0, label="Observed"
    )
    paper_style.rasterize_dense(observed, len(hours))
    modelled = ax_tt.scatter(
        hours, result["modelled_s"], s=0.2, color=MODELLED_COLOR, lw=0, label="Modelled"
    )
    paper_style.rasterize_dense(modelled, len(hours))
    ax_tt.set_ylabel("One-way travel time (s)")
    ax_tt.tick_params(labelbottom=False)
    ax_tt.set_xlim(*paper_style.TIME_LIMITS_H)

    points = ax_res.scatter(hours, residual, s=0.8, color="0.6", lw=0)
    paper_style.rasterize_dense(points, len(hours))
    edges = np.arange(0.0, paper_style.TIME_LIMITS_H[1] + MEDIAN_BIN_H, MEDIAN_BIN_H)
    centres, bands, _ = fc.binned_percentiles(hours, residual, edges, min_count=10)
    ax_res.fill_between(
        centres, bands[:, 0], bands[:, 2], color=OBSERVED_COLOR, alpha=0.35, lw=0
    )
    ax_res.plot(centres, bands[:, 1], color="k", lw=1.0)
    ax_res.axhline(0.0, color=fc.CONTEXT, lw=0.5, zorder=0)
    ax_res.set_xticks(np.arange(0, 15, 2))
    ax_res.set_xlabel(paper_style.TIME_LABEL)
    ax_res.set_ylabel("Range residual (cm)")

    limit = np.ceil(1.05 * np.max(np.abs(residual)))  # every residual inside
    bins = np.linspace(-limit, limit, 61)
    ax_hist.hist(
        residual, bins=bins, orientation="horizontal", color="0.6", density=True
    )
    grid_cm = np.linspace(-limit, limit, 200)
    ax_hist.plot(
        norm.pdf(grid_cm, residual.mean(), sigma_cm), grid_cm, color="k", lw=1.0
    )
    ax_hist.set_xlabel("Density (1/cm)")
    ax_hist.tick_params(labelleft=False)
    ax_res.set_ylim(-limit, limit)
    assert np.all(np.abs(residual) < limit)
    fc.despine(ax_tt, ax_res, ax_hist)

    # One key for the three panels, in the empty cell beside (a).
    ax_key.axis("off")
    ax_key.legend(
        handles=[
            Line2D([], [], ls="none", marker="o", ms=3, color=OBSERVED_COLOR),
            Line2D([], [], ls="none", marker="o", ms=1.5, color=MODELLED_COLOR),
            Line2D([], [], ls="none", marker="o", ms=2, color="0.6"),
            Line2D([], [], color="k", lw=1.0),
            Patch(color=OBSERVED_COLOR, alpha=0.35, lw=0),
        ],
        labels=[
            "Observed",
            "Modelled",
            "Residual",
            f"Median, {60 * MEDIAN_BIN_H:.0f}-min bins;\nnormal curve in (c)",
            "16–84 % band",
        ],
        loc="center left",
        borderaxespad=0.0,
    )

    paper_style.panel_label(ax_tt, "a")
    paper_style.panel_label(ax_res, "b")
    paper_style.panel_label(ax_hist, "c")
    fc.save(fig, NAME)


if __name__ == "__main__":
    main()
