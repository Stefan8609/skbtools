"""GNSS height filter of the Bermuda survey (remade Fig. A3), per antenna and combined.

Message: the height filter rejects an epoch when any one antenna leaves its
running median ± 2 MAD band, which removes about a third of the epochs, spread
over the whole survey.

Panels (a)-(d): WGS84 ellipsoidal height of each antenna against time since survey
start, with the 5000-epoch running median and median ± 2 MAD band. Kept epochs
are grey; epochs where this antenna failed are in the accent colour; epochs
dropped because another antenna failed are light grey. The strip (e) gives the
share of rejected epochs per 10-min bin under the combined rule ("an epoch is
rejected if any antenna fails"). The counts that were printed on the figure are
printed to the terminal for the caption. Every count is computed here by
re-running the filter of Inversion_Workflow/Bermuda/Initialize_Bermuda_Data.py on
the raw antenna files, and the kept epochs are checked against the processed file
the inversions read.

Reviewer comments: R3 M7 (per-receiver vs combined counts, the "any receiver fails"
rule and the hard cut-offs), R3 #11 (legible type).
Inputs: Bermuda_Data/GPS_Data/Bermuda_Testing/Antenna{1..4}-camp.mat (PRIDE-PPPAR
positions) and Bermuda_Data/GPS_Data/Processed_GPS_Receivers_DOG_1.npz.
Run from the repo root with PYTHONPATH=src.
"""

import matplotlib.pyplot as plt
import numpy as np

import _figure_common as fc
from plotting import paper_style

# --- configuration -----------------------------------------------------------------
GNSS_FILES = [
    fc.REPO / f"Bermuda_Data/GPS_Data/Bermuda_Testing/Antenna{k}-camp.mat"
    for k in (1, 2, 3, 4)
]
PROCESSED_FILE = fc.REPO / "Bermuda_Data/GPS_Data/Processed_GPS_Receivers_DOG_1.npz"
NAME = "fig_FXA3_gnss_rejection"
HEIGHT_MM = 168.0
Y_LIMITS_M = (-38.1, -34.9)  # just beyond the hard cut-offs: every point is shown
BIN_H = 10 / 60  # time bin of panel (e)
KEPT_COLOR = "0.55"
FAILED_COLOR = fc.ACCENT  # this antenna failed
OTHER_COLOR = "0.85"  # rejected because another antenna failed
# -----------------------------------------------------------------------------------


def thousands(n):
    """Integer with a thin space as thousands separator (journal style)."""
    return f"{n:,}".replace(",", " ")


def main():
    fc.require(*GNSS_FILES, PROCESSED_FILE)
    paper_style.apply()
    # Manual layout: with matplotlib 3.7, savefig switches constrained layout back
    # on (rcParams) for a figure created without it, and moves the axes.
    plt.rcParams["figure.constrained_layout.use"] = False
    epochs = fc.load_gnss_epochs(GNSS_FILES)
    masks = fc.gnss_rejection_masks(epochs["height_m"])
    processed_gps_s = np.load(PROCESSED_FILE)["GPS_data"]
    if not np.array_equal(epochs["gps_s"][masks["kept"]], processed_gps_s):
        raise ValueError(f"kept epochs differ from GPS_data in {PROCESSED_FILE}")

    hours = paper_style.hours_since_start(epochs["gps_s"])
    heights = epochs["height_m"]
    n_epochs = hours.size
    hard_any = masks["hard_fail"].any(axis=0)
    n_stage2 = int((~hard_any).sum())
    band_per_antenna, band_any = fc.rejection_counts(masks["band_fail"])
    hard_per_antenna, n_hard = fc.rejection_counts(masks["hard_fail"])
    n_kept = int(masks["kept"].sum())
    left_of_axis = int((hours < paper_style.TIME_LIMITS_H[0]).sum())
    print(
        f"common epochs {n_epochs}; hard cut-offs {n_hard} {hard_per_antenna}; "
        f"band per antenna {band_per_antenna.tolist()} (sum "
        f"{band_per_antenna.sum()}), any {band_any}; kept {n_kept}; "
        f"rejected {n_epochs - n_kept}; epochs before 0 h: {left_of_axis}"
    )

    print(
        f"step 1 (hard cut-offs) rejects {n_hard}; step 2 on {n_stage2} epochs "
        f"rejects {band_any}; rejected {100 * (n_epochs - n_kept) / n_epochs:.1f} %; "
        f"sum of per-antenna failures "
        f"{int((band_per_antenna + hard_per_antenna).sum())}"
    )

    fig = plt.figure(
        figsize=(
            paper_style.DOUBLE_COLUMN_MM / paper_style.MM_PER_INCH,
            HEIGHT_MM / paper_style.MM_PER_INCH,
        ),
    )
    grid = fig.add_gridspec(
        5,
        1,
        height_ratios=[1, 1, 1, 1, 0.22],
        left=0.085,
        right=0.905,
        top=0.975,
        bottom=0.14,
        hspace=0.22,
    )
    axes = [fig.add_subplot(grid[k]) for k in range(4)]
    for k, ax in enumerate(axes):
        failed = masks["band_fail"][k] | masks["hard_fail"][k]
        other = ~failed & ~masks["kept"]
        for select, color, label in (
            (other, OTHER_COLOR, "Rejected: another antenna failed"),
            (masks["kept"], KEPT_COLOR, "Kept"),
            (failed, FAILED_COLOR, "Rejected: this antenna failed"),
        ):
            points = ax.scatter(
                hours[select], heights[k, select], s=0.5, color=color, lw=0, label=label
            )
            paper_style.rasterize_dense(points, int(select.sum()))
        median, mad = masks["median_m"][k], masks["mad_m"][k]
        stage2 = ~hard_any
        ax.plot(
            hours[stage2], median[stage2], color="k", lw=0.8, label="Running median"
        )
        for sign in (-1, 1):
            ax.plot(
                hours[stage2],
                median[stage2] + sign * fc.GNSS_MAD_FACTOR * mad[stage2],
                color="k",
                lw=0.6,
                ls="--",
                label=f"Median ± {fc.GNSS_MAD_FACTOR:.0f} MAD" if sign > 0 else None,
            )
        for j, limit in enumerate(fc.GNSS_HEIGHT_LIMITS_M):
            label = "Hard cut-offs" if j == 0 else None
            ax.axhline(limit, color=fc.CONTEXT, lw=0.6, ls=":", label=label)
        ax.set_ylim(Y_LIMITS_M)
        ax.set_xlim(paper_style.TIME_LIMITS_H)
        ax.set_yticks([-37, -36])
        ax.text(0.008, 0.9, f"Antenna {k + 1}", transform=ax.transAxes, va="top")
        ax.tick_params(labelbottom=False)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: fc.signed(v, ".0f"))
        )
        paper_style.panel_label(ax, "abcd"[k])
    fc.despine(*axes)
    fig.supylabel("Ellipsoidal height (m)", x=0.012, fontsize=paper_style.FONT_SIZE_PT)

    handles, labels = axes[0].get_legend_handles_labels()
    order = [1, 2, 0, 3, 4, 5]  # kept, this antenna, another antenna, lines
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        markerscale=7,
        handlelength=2.2,
        columnspacing=1.5,
    )

    # (e): share of rejected epochs per time bin under the combined rule.
    strip = fig.add_subplot(grid[4], sharex=axes[0])
    edges = np.arange(
        paper_style.TIME_LIMITS_H[0], paper_style.TIME_LIMITS_H[1] + 1e-9, BIN_H
    )
    counts = np.histogram(hours, edges)[0]
    share = 100 * np.histogram(hours[~masks["kept"]], edges)[0] / np.maximum(counts, 1)
    share[counts == 0] = np.nan
    image = strip.pcolormesh(
        edges, [0.0, 1.0], share[None, :], cmap="Greys", vmin=0, vmax=100
    )
    strip.set_yticks([])
    strip.set_xticks(np.arange(0, 15, 2))
    strip.set_xlabel(paper_style.TIME_LABEL)
    paper_style.panel_label(strip, "e")
    box = strip.get_position()
    colorbar_axes = fig.add_axes([box.x1 + 0.012, box.y0, 0.012, 3.5 * box.height])
    colorbar = fig.colorbar(image, cax=colorbar_axes)
    colorbar.set_ticks([0, 50, 100])
    colorbar.set_label("Rejected (%)")
    colorbar.outline.set_linewidth(0.6)
    fc.save(fig, NAME)


if __name__ == "__main__":
    main()
