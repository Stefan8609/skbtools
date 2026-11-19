"""FX-B8/B9 (remake of Figs. B8 and B9 as one figure): synthetic DOG and lever errors.

Monte Carlo realisations of the Bermuda synthetic (Sample_Ellipse.error_ellipse:
Bermuda DOG 1 trajectory, σd = 2 cm per receiver, σt = 20 µs, free clock offset,
lever and DOG solved by the Gauss-Newton offset search). (a–c) DOG position error in
local ENU at the true DOG; (d–f) lever-arm error, body frame. Grey points are the
realisations, the black curve their 95 % covariance ellipse, the red marker the mean
error (bias) with its 95 % confidence interval, the cross the truth. (g) Table of
the mean bias ± 95 % CI for the inversion with the identical ESV table (control) and
with the moderate and strong sound-speed perturbations, so the part of the bias that
comes from the sound speed can be told from the part that comes from GNSS noise and
geometry (K2).

Reviewer comments: R4 4 (are the biases in B8/B9 significant, and why?), R1 3a (the
∼70 cm vertical bias deserves prominence), K2 (new ESV tables, free offset, 1000
realisations).

Inputs: ``Plot_Data/synthetic_error_ellipse_<table>.npz`` per inversion table
(``estimate_converted`` m ENU, ``lever_diff_array`` m, ``RMSE_array`` cm, failed
realisations already removed by the producer).
"""

import matplotlib.pyplot as plt
import numpy as np

import _figure_common as fc
from acoustics.synthetic_esv_tables import ESV_GENERATE, ESV_MODERATE, ESV_STRONG
from plotting import paper_style

# --- configuration -----------------------------------------------------------------
PLOT_DATA = fc.REPO / "Bermuda_Data/Output/Plot_Data"
MAIN_TABLE = ESV_MODERATE  # the table of the scatter panels (submitted B8/B9)
TABLES = {  # inversion ESV table -> row label of the bias table
    ESV_GENERATE: "Identical (control)",
    ESV_MODERATE: "Moderate",
    ESV_STRONG: "Strong",
}
INPUTS = {name: PLOT_DATA / f"synthetic_error_ellipse_{name}.npz" for name in TABLES}
N_ATTEMPTED = 1000  # num_points of Sample_Ellipse.__main__ (failures are not saved)
CI_PROBABILITY = 0.95
ELLIPSE_PROBABILITY = 0.95
HEIGHT_MM = 150.0
POINT_COLOR = "0.55"
BIAS_COLOR = fc.POSITIVE_COLOR
NAME = "fig_FXB8_synthetic_bias"
# -----------------------------------------------------------------------------------

DOG_LABELS = ("ΔE (cm)", "ΔN (cm)", "ΔU (cm)")
LEVER_LABELS = tuple(f"Δ{label} (cm)" for label in fc.LEVER_LABELS)
PAIRS = ((0, 1), (0, 2), (1, 2))


def error_panel(ax, errors_cm, i, j, labels):
    """Realisations, 95 % ellipse, mean bias with CI and the truth for one pair."""
    xy = errors_cm[:, [i, j]]
    layer = ax.scatter(*xy.T, s=1.5, color=POINT_COLOR, alpha=0.5, lw=0)
    paper_style.rasterize_dense(layer, len(xy))
    boundary = fc.ellipse_points(xy.mean(axis=0), np.cov(xy.T), ELLIPSE_PROBABILITY)
    ax.plot(*boundary.T, color="k", lw=0.8)
    mean, half = fc.mean_with_ci(xy, CI_PROBABILITY)
    ax.errorbar(
        mean[0],
        mean[1],
        xerr=half[0],
        yerr=half[1],
        marker="o",
        ms=3.5,
        color=BIAS_COLOR,
        mec="k",
        mew=0.4,
        elinewidth=1.0,
        capsize=1.5,
        zorder=4,
    )
    ax.plot(0, 0, marker="+", ms=8, mew=1.2, color="k", zorder=5)
    ax.set_xlabel(labels[i])
    ax.set_ylabel(labels[j])
    ax.grid(lw=0.3, color="0.88")


def bias_table(ax, data):
    """Mean bias ± 95 % CI (cm) and realisation counts for every inversion table."""
    rows = []
    for name in TABLES:
        errors = np.hstack([data[name]["dog_cm"], data[name]["lever_cm"]])
        mean, half = fc.mean_with_ci(errors, CI_PROBABILITY)
        rows.append(
            [f"{fc.signed(m, '+.2f')}±{h:.2f}" for m, h in zip(mean, half, strict=True)]
            + [f"{len(errors)} / {N_ATTEMPTED}"]
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        rowLabels=list(TABLES.values()),
        colLabels=[label.replace(" (cm)", "") for label in DOG_LABELS + LEVER_LABELS]
        + ["n kept"],
        bbox=[0.17, 0.0, 0.83, 0.85],  # row labels sit left of the bbox
        cellLoc="center",
        edges="horizontal",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(paper_style.FONT_SIZE_PT)
    table.set_in_layout(False)
    ax.set_title(
        f"Mean error ± {100 * CI_PROBABILITY:.0f} % CI (cm) by inversion ESV table",
        loc="left",
    )
    return rows


def bias_figure(data):
    """Combined B8/B9 figure; data maps table name to dict(dog_cm, lever_cm)."""
    paper_style.apply()
    fig = plt.figure(
        figsize=(
            paper_style.DOUBLE_COLUMN_MM / paper_style.MM_PER_INCH,
            HEIGHT_MM / paper_style.MM_PER_INCH,
        )
    )
    grid = fig.add_gridspec(3, 3, height_ratios=(1.0, 1.0, 0.55))
    main = data[MAIN_TABLE]
    letters = iter("abcdefg")
    for row, (errors, labels) in enumerate(
        ((main["dog_cm"], DOG_LABELS), (main["lever_cm"], LEVER_LABELS))
    ):
        for col, (i, j) in enumerate(PAIRS):
            ax = fig.add_subplot(grid[row, col])
            error_panel(ax, errors, i, j, labels)
            paper_style.panel_label(ax, next(letters))
    ax_table = fig.add_subplot(grid[2, :])
    rows = bias_table(ax_table, data)
    paper_style.panel_label(ax_table, next(letters))
    return fig, rows


def load(path):
    with np.load(path) as npz:
        return {
            "dog_cm": npz["estimate_converted"] * 100.0,
            "lever_cm": npz["lever_diff_array"] * 100.0,
        }


def main():
    data = {name: load(path) for name, path in INPUTS.items()}
    fig, rows = bias_figure(data)
    for label, row in zip(TABLES.values(), rows, strict=True):
        print(label, row)
    fc.save(fig, NAME)


if __name__ == "__main__":
    fc.require(*INPUTS.values())
    main()
