"""FX-5: range residuals of the final model against time, all three DOGs.

Message: the residuals of the final model stay within a few tens of centimetres
of zero all survey long, with the largest excursions next to antenna-4 height steps.

Replaces the submitted Figs. 5, 9, C12 and C13. One row per DOG on the common time
axis (R1 2c, R1 4e): every residual as a light, rasterized point, with the median
and 16-84 % band of 5-min bins on top. Residuals are observed minus modelled travel
time at the posterior mean of the final run, as range at 1515 m/s (R1 4i). No
local-window sigma boxes (R3 #5); the overall sd of each DOG is in its legend.
Dashed lines bound the zoom window of the submitted Figs. 5 and 9, labelled
"9.8-10.0 h" there on the old axis (GPS_data / 3600), so R1 4l can be answered.
Triangles mark the antenna-4 height steps of FIG-4 (same detection, fc.ANTENNA4_*).

Inputs: the chains of FINAL_RUN (Bermuda_Data/Output/MCMC_Chains/ar1_common30/), the
data they were sampled on (rebuilt by _figure_common.final_residuals) and the run's
GNSS file for the antenna-4 steps.
"""

import numpy as np

import _figure_common as fc
from plotting import paper_style

# --- configuration -----------------------------------------------------------------
RUN = fc.FINAL_RUN
CHAIN_DIR = fc.CHAINS_DIR / RUN
NAME = "fig_FX5_residuals_time"
DOGS = (1, 3, 4)
BIN_MINUTES = 5.0  # median and 16-84 % band
MIN_PER_BIN = 20  # fewer pairs in a bin: no median drawn (gaps stay gaps)
Y_LIMITS_CM = (-75.0, 100.0)  # holds every residual of the final run (no clipping)
# Zoom window of the submitted Figs. 5 and 9: zoom_start = 35000 s of GPS_data and a
# 1200 s length (src/plotting/Misc/Plot_from_saved_data.py, MCMC_best_sample.py).
SUBMITTED_ZOOM_GPS_S = (35000.0, 36200.0)
HEIGHT_MM = 140.0
# -----------------------------------------------------------------------------------


def draw(residuals, steps_h):
    """Three rows (DOG 1, 3, 4) of residual against time; returns the figure."""
    fig, axes = paper_style.figure("double", HEIGHT_MM, nrows=3, sharex=True)
    edges_h = np.arange(
        paper_style.TIME_LIMITS_H[0],
        paper_style.TIME_LIMITS_H[1] + 1e-9,
        BIN_MINUTES / 60.0,
    )
    zoom_h = paper_style.hours_since_start(np.asarray(SUBMITTED_ZOOM_GPS_S))
    for k, (ax, dog) in enumerate(zip(axes, DOGS, strict=True)):
        style = paper_style.DOG_STYLE[dog]
        hours = paper_style.hours_since_start(residuals[dog]["gps_time_s"])
        res_cm = residuals[dog]["residual_cm"]
        if res_cm.min() < Y_LIMITS_CM[0] or res_cm.max() > Y_LIMITS_CM[1]:
            raise ValueError(f"DOG {dog} residuals exceed {Y_LIMITS_CM} cm")
        centres, pct, _ = fc.binned_percentiles(
            hours, res_cm, edges_h, min_count=MIN_PER_BIN
        )
        ax.axhline(0.0, color="0.5", lw=0.5, zorder=1)
        for edge in zoom_h:
            ax.axvline(edge, color="0.3", lw=0.7, ls="--", zorder=1)
        points = ax.scatter(
            hours,
            res_cm,
            s=0.8,
            color=style["color"],
            alpha=0.25,
            lw=0,
            zorder=2,
            label=f"{style['label']}, σ = {res_cm.std():.1f} cm",
        )
        paper_style.rasterize_dense(points, res_cm.size)
        ax.fill_between(
            centres,
            pct[:, 0],
            pct[:, 2],
            color="0.2",
            alpha=0.25,
            lw=0,
            zorder=3,
            label=f"16–84 % of residuals per {BIN_MINUTES:.0f}-min bin",
        )
        ax.plot(
            centres,
            pct[:, 1],
            color="black",
            lw=0.7,
            zorder=4,
            label=f"Median per {BIN_MINUTES:.0f}-min bin",
        )
        # Short marks hanging from the top edge, the accent lines of FIG-4.
        ax.vlines(
            steps_h,
            0.93,
            1.0,
            transform=ax.get_xaxis_transform(),
            color=fc.ACCENT,
            lw=1.2,
            zorder=5,
            label="Antenna-4 height step",
        )
        ax.set_ylim(*Y_LIMITS_CM)
        ax.set_yticks([-50, 0, 50])
        ax.set_ylabel("Range residual (cm)")
        paper_style.panel_label(ax, "abc"[k])
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(
            handles[:1],
            labels[:1],
            loc="upper right",
            bbox_to_anchor=(1.0, 0.94),
            markerscale=5,
            handletextpad=0.2,
        )
        legend.legend_handles[0].set_alpha(1.0)
    fc.despine(*axes)
    zoom = axes[0].plot([], [], color="0.3", lw=0.7, ls="--")[0]
    fig.legend(
        handles[1:] + [zoom],
        labels[1:] + ["Submitted Figs. 5, 9 zoom"],
        loc="outside upper center",
        ncols=2,
        columnspacing=1.0,
        handlelength=1.6,
    )
    axes[-1].set_xlim(*paper_style.TIME_LIMITS_H)
    axes[-1].set_xticks(np.arange(0, 15, 2))
    axes[-1].set_xlabel(paper_style.TIME_LABEL)
    return fig


if __name__ == "__main__":
    fc.require(CHAIN_DIR / "chain_00.npz")
    paper_style.apply()
    pooled = fc.pooled_run(RUN)
    residuals, _ = fc.final_residuals(pooled)
    _, _, step_s, _ = fc.antenna4_steps(fc.run_config(pooled))
    steps_h = paper_style.hours_since_start(step_s)
    zoom_h = paper_style.hours_since_start(np.asarray(SUBMITTED_ZOOM_GPS_S))
    print(f"submitted zoom window = {zoom_h[0]:.3f}-{zoom_h[1]:.3f} h since start")
    print("antenna-4 steps (h):", np.round(steps_h, 3))
    for dog in DOGS:
        hours = paper_style.hours_since_start(residuals[dog]["gps_time_s"])
        res_cm = residuals[dog]["residual_cm"]
        inside = (hours >= zoom_h[0]) & (hours <= zoom_h[1])
        print(
            f"DOG {dog}: n = {res_cm.size}, mean = {res_cm.mean():+.2f} cm, "
            f"sd = {res_cm.std():.2f} cm; zoom window n = {inside.sum()}, "
            f"sd = {res_cm[inside].std():.2f} cm"
        )
    fig = draw(residuals, steps_h)
    fc.save(fig, NAME)
