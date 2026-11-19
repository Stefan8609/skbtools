"""FIG-1: processing flowchart from GNSS positions to the reported DOG position.

Answers R4 3 (a flowchart), R1 2b and R2 (foreground the one-way specific steps:
clock-offset self-calibration), R3 M8 (which numbers are *the* position: the box at
the end names ``fc.FINAL_SOLUTION``) and R3 M1 (stage roles: the circular variance
fixes the integer second, the RMSE the fraction; code ``find_int_offset`` and
``find_subint_offset``, Appendix A).

Boxes run left to right in the top row, then right to left in the bottom row; every
arrow is a sequential hand-over, each box names its output after "→". The counts in
the quality-control box are recomputed from the raw antenna files, as in FX-A3.

Inputs: Bermuda_Data/GPS_Data/Bermuda_Testing/Antenna{1..4}-camp.mat; the gate and
the run name come from the final run's saved configuration.
Run from the repo root with PYTHONPATH=src.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import _figure_common as fc
from plotting import paper_style

# --- configuration -----------------------------------------------------------------
NAME = "fig_FIG1_flowchart"
RUN = fc.FINAL_RUN
GNSS_FILES = [
    fc.REPO / f"Bermuda_Data/GPS_Data/Bermuda_Testing/Antenna{k}-camp.mat"
    for k in (1, 2, 3, 4)
]
HEIGHT_MM = 112.0
BOX_W_MM, BOX_H_MM, GAP_MM = 39.0, 40.0, 5.0
LEFT_MM = 1.5
ROW_BOTTOM_MM = (62.0, 8.0)  # top row, bottom row
ONE_WAY_COLOR = paper_style.CATEGORY_COLORS[1]  # one-way specific steps
FINAL_COLOR = "black"
# -----------------------------------------------------------------------------------


def thousands(n):
    return f"{n:,}".replace(",", " ")


def box(ax, x_mm, y_mm, title, lines, output, color="0.35", lw=0.8, shade="white"):
    """Rounded box with a bold title, body lines and an output line after →."""
    ax.add_patch(
        FancyBboxPatch(
            (x_mm, y_mm),
            BOX_W_MM,
            BOX_H_MM,
            boxstyle="round,pad=0,rounding_size=2",
            fc=shade,
            ec=color,
            lw=lw,
        )
    )
    ax.text(
        x_mm + BOX_W_MM / 2,
        y_mm + BOX_H_MM - 2.5,
        title,
        ha="center",
        va="top",
        fontweight="bold",
        color=color if color != "0.35" else "black",
    )
    ax.text(
        x_mm + BOX_W_MM / 2,
        y_mm + BOX_H_MM - 12.0,
        "\n".join(lines),
        ha="center",
        va="top",
        linespacing=1.25,
    )
    ax.text(
        x_mm + BOX_W_MM / 2,
        y_mm + 2.2,
        output,
        ha="center",
        va="bottom",
        fontstyle="italic",
        linespacing=1.25,
    )


def arrow(ax, start, end, **kwargs):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            lw=1.0,
            color="0.2",
            **kwargs,
        )
    )


if __name__ == "__main__":
    fc.require(*GNSS_FILES)
    config = fc.run_config(fc.pooled_run(RUN))
    epochs = fc.load_gnss_epochs(GNSS_FILES)
    masks = fc.gnss_rejection_masks(epochs["height_m"])
    n_epochs, n_kept = epochs["gps_s"].size, int(masks["kept"].sum())
    gate_s = config["pair_threshold_s"]
    window_min = config["esv_window_s"] / 60.0

    paper_style.apply()
    plt.rcParams["figure.constrained_layout.use"] = False
    width_mm = paper_style.DOUBLE_COLUMN_MM
    fig = plt.figure(
        figsize=(
            width_mm / paper_style.MM_PER_INCH,
            HEIGHT_MM / paper_style.MM_PER_INCH,
        ),
        layout="none",
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_mm)
    ax.set_ylim(0, HEIGHT_MM)
    ax.axis("off")

    top, bottom = ROW_BOTTOM_MM
    x_top = [LEFT_MM + i * (BOX_W_MM + GAP_MM) for i in range(4)]
    x_bottom = x_top[::-1]  # bottom row runs right to left
    box(
        ax,
        x_top[0],
        top,
        "1  GNSS positions",
        ["4 antennas", "PRIDE-PPPAR", "1 Hz, WGS84"],
        "→ antenna positions",
    )
    box(
        ax,
        x_top[1],
        top,
        "2  Quality control",
        [
            "epoch rejected if any",
            "antenna fails the",
            f"cut-offs or band: {thousands(n_kept)}",
            f"of {thousands(n_epochs)} kept",
        ],
        "→ kept epochs",
    )
    box(
        ax,
        x_top[2],
        top,
        "3  Transducer position",
        [
            "rigid-body fit of the",
            "4 antennas, attitude,",
            f"lever arm {fc.LEVER_LABELS[0]}, "
            f"{fc.LEVER_LABELS[1]}, {fc.LEVER_LABELS[2]}",
        ],
        "→ transducer track",
    )
    box(
        ax,
        x_top[3],
        top,
        "4  Pairing",
        ["arrival ↔ emission", "epoch, gate", f"γ = {gate_s:g} s", "outlier rejection"],
        "→ travel-time pairs",
    )
    box(
        ax,
        x_bottom[0],
        bottom,
        "5  Clock offset",
        ["1: circular variance", "→ integer second", "2: RMSE profile", "→ fraction"],
        f"→ {fc.DT_SYMBOL}\n(one-way specific)",
        color=ONE_WAY_COLOR,
        lw=2.0,
        shade="#FBEBDD",
    )
    box(
        ax,
        x_bottom[1],
        bottom,
        "6  Gauss–Newton",
        [
            "least squares for",
            f"x_R, {fc.DC_SYMBOL}, {fc.DT_SYMBOL}",
            "lever-arm search",
        ],
        "→ start values",
    )
    box(
        ax,
        x_bottom[2],
        bottom,
        "7  MCMC",
        ["priors, AR(1) noise,", f"{window_min:.0f}-min {fc.DC_SYMBOL}(t)", "8 chains"],
        "→ posterior of all\nparameters",
    )
    box(
        ax,
        x_bottom[3],
        bottom,
        "8  Reported",
        [fc.FINAL_SOLUTION, "of the DOG position", f"(run {RUN})"],
        "→ ECEF XYZ, E, N, U",
        color=FINAL_COLOR,
        lw=2.2,
    )

    mid_top, mid_bottom = top + BOX_H_MM / 2, bottom + BOX_H_MM / 2
    for i in range(3):
        arrow(ax, (x_top[i] + BOX_W_MM, mid_top), (x_top[i + 1], mid_top))
        arrow(ax, (x_bottom[i], mid_bottom), (x_bottom[i + 1] + BOX_W_MM, mid_bottom))
    centre_4 = x_top[3] + BOX_W_MM / 2
    arrow(ax, (centre_4, top), (centre_4, bottom + BOX_H_MM))
    ax.text(
        LEFT_MM + 1.0,
        (top + bottom + BOX_H_MM) / 2,
        "Orange outline: one-way specific step. Every arrow is a sequential hand-over.",
        ha="left",
        va="center",
    )
    fc.save(fig, NAME)
