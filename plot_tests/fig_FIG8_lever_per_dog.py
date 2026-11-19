"""FIG-8: lever-arm posteriors of the per-DOG MCMC runs against the prior and Table 1.

One panel per body-frame component (lₓ, l_y, l_z). Each row is one per-DOG run
(``ar1_dog1``, ``ar1_dog3``, ``ar1_dog4``) plus the joint final run: posterior mean
(filled marker) with its central 95 % interval, the submitted Table 1 lever of that
DOG (open marker), and the drawing prior (grey band, ±1 sd and ±2 sd). The panel
title gives the spread (max − min) of the per-DOG posterior means.

Reviewer comments: R4 5 (show the per-DOG lever posteriors; Table 1 levers differ
by 12–93 cm horizontally, far more than the posterior widths), R1 4h ("similar lever
arms" needs specific discussion).

Inputs: chains of the per-DOG runs (variants ``ar1_dog{1,3,4}`` of MCMC_sampler,
not yet run) and of the final joint run.
"""

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import _figure_common as fc
from plotting import paper_style

# --- configuration -----------------------------------------------------------------
PER_DOG_RUNS = {1: "ar1_dog1", 3: "ar1_dog3", 4: "ar1_dog4"}
JOINT_RUN = fc.FINAL_RUN
# Submitted Table 1 (Paper_I.pdf, p. 17): lever arm p (m, body frame) per DOG and
# joint, from the Gauss-Newton inversion. Published values, shown for comparison.
SUBMITTED_TABLE1_LEVER_M = {
    1: (-12.77, 9.58, -12.07),
    3: (-12.89, 9.45, -12.28),
    4: (-13.36, 8.65, -12.95),
    "joint": (-12.87, 9.44, -13.36),
}
INTERVAL_PROBABILITY = 0.95
HEIGHT_MM = 70.0
JOINT_COLOR = "0.1"
PRIOR_COLOR = "0.6"
NAME = "fig_FIG8_lever_per_dog"
# -----------------------------------------------------------------------------------


def lever_figure(levers_by_dog, joint_lever, prior_center, prior_sd):
    """Dot-whisker figure of lever posteriors; returns it and the DOG spread (cm).

    levers_by_dog maps DOG number to (n, 3) lever draws (m); joint_lever is (n, 3);
    prior_center and prior_sd are (3,) m. The spread is max − min of the per-DOG
    posterior means for each component.
    """
    dogs = list(levers_by_dog)
    means = np.array([levers_by_dog[dog].mean(axis=0) for dog in dogs])
    spread_cm = np.ptp(means, axis=0) * 100.0
    rows = dogs + ["joint"]
    draws = [levers_by_dog[dog] for dog in dogs] + [joint_lever]

    paper_style.apply()
    fig, axes = paper_style.figure("double", HEIGHT_MM, ncols=3, sharey=True)
    for k, ax in enumerate(axes):
        for sd_factor, alpha in ((2.0, 0.08), (1.0, 0.12)):
            ax.axvspan(
                prior_center[k] - sd_factor * prior_sd[k],
                prior_center[k] + sd_factor * prior_sd[k],
                color=PRIOR_COLOR,
                alpha=alpha,
                lw=0,
                zorder=0,
            )
        for y, (row, samples) in enumerate(zip(rows, draws, strict=True)):
            if row == "joint":
                color, marker = JOINT_COLOR, "D"
            else:
                color = paper_style.DOG_STYLE[row]["color"]
                marker = paper_style.DOG_STYLE[row]["marker"]
            mean, low, high = fc.central_interval(samples[:, k], INTERVAL_PROBABILITY)
            ax.plot([low, high], [y, y], color=color, lw=1.6, zorder=2)
            ax.plot(mean, y, marker=marker, color=color, mec="k", mew=0.4, zorder=3)
            ax.plot(
                SUBMITTED_TABLE1_LEVER_M[row][k],
                y + 0.25,
                marker=marker,
                mfc="none",
                mec=color,
                mew=1.0,
                ls="none",
                zorder=3,
            )
        # Zoom on the estimates; the prior band runs off the axes (caption says so).
        shown = [fc.central_interval(d[:, k], INTERVAL_PROBABILITY)[1:] for d in draws]
        shown = np.concatenate(
            [np.ravel(shown), [SUBMITTED_TABLE1_LEVER_M[row][k] for row in rows]]
        )
        pad = 0.15 * np.ptp(shown)
        ax.set_xlim(shown.min() - pad, shown.max() + pad)
        ax.set_xlabel(f"{fc.LEVER_LABELS[k]} (m), body frame")
        ax.set_title(f"spread of DOG means {spread_cm[k]:.0f} cm")
        ax.grid(axis="x", lw=0.3, color="0.85")
        paper_style.panel_label(ax, "abc"[k])
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(
        [paper_style.DOG_STYLE[row]["label"] for row in dogs] + ["Joint"]
    )
    axes[0].set_ylim(len(rows) - 0.4, -0.6)
    fig.legend(
        handles=[
            Line2D([], [], color="0.3", marker="o", mec="k", lw=1.6),
            Line2D([], [], color="0.3", marker="o", mfc="none", ls="none"),
            Patch(color=PRIOR_COLOR, alpha=0.2, lw=0),
            Patch(color=PRIOR_COLOR, alpha=0.08, lw=0),
        ],
        labels=[
            f"Posterior mean, {100 * INTERVAL_PROBABILITY:.0f} % interval",
            "Submitted Table 1",
            "Prior ±1 sd",
            "Prior ±2 sd",
        ],
        loc="outside lower center",
        ncols=4,
    )
    return fig, spread_cm


def main():
    pooled = {dog: fc.pooled_run(run) for dog, run in PER_DOG_RUNS.items()}
    names = list(pooled[1]["param_names"])
    index = [names.index(f"lever_{axis}") for axis in "xyz"]
    prior_center = pooled[1]["prior_center"][index]
    prior_sd = pooled[1]["prior_sd"][index]
    for dog, run in pooled.items():
        if not np.allclose(run["prior_sd"][index], prior_sd) or not np.allclose(
            run["prior_center"][index], prior_center
        ):
            raise ValueError(f"lever prior of {PER_DOG_RUNS[dog]} differs from DOG 1's")
    fig, spread_cm = lever_figure(
        {dog: run["lever"] for dog, run in pooled.items()},
        fc.pooled_run(JOINT_RUN)["lever"],
        prior_center,
        prior_sd,
    )
    print("spread of per-DOG posterior means (cm)", np.round(spread_cm, 1))
    table1 = np.array([SUBMITTED_TABLE1_LEVER_M[dog] for dog in PER_DOG_RUNS])
    print("spread of Table 1 levers (cm)", np.round(np.ptp(table1, axis=0) * 100, 1))
    fc.save(fig, NAME)


if __name__ == "__main__":
    fc.require(*(fc.CHAINS_DIR / run for run in PER_DOG_RUNS.values()))
    main()
