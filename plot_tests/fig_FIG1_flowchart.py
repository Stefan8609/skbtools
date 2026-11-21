"""FIG-1: processing flowchart from GNSS positions to the reported DOG position.

Message: eight sequential steps lead from the GNSS positions to the reported DOG
position, and only the clock-offset step is specific to one-way ranging.

Answers R4 3 (a flowchart), R1 2b and R2 (foreground the one-way specific steps:
clock-offset self-calibration), R3 M8 (which numbers are *the* position: the box at
the end names ``fc.FINAL_SOLUTION``) and R3 M1 (stage roles: the circular variance
fixes the integer second, the RMSE the fraction; code ``find_int_offset`` and
``find_subint_offset``, Appendix A).

Boxes run left to right in the top row, then right to left in the bottom row; every
arrow is a sequential hand-over. Each box has a verb-phrase title and one line of
detail; what each step hands on is in the caption. The counts in the
quality-control box are recomputed from the raw antenna files, as in FX-A3.

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
HEIGHT_MM = 52.0
BOX_W_MM, BOX_H_MM, GAP_MM = 39.0, 17.0, 5.0
LEFT_MM = 1.5
ROW_BOTTOM_MM = (31.0, 4.0)  # top row, bottom row
# -----------------------------------------------------------------------------------


def thousands(n):
    return f"{n:,}".replace(",", " ")


def box(ax, x_mm, y_mm, title, detail, color="0.45", lw=0.8):
    """Rounded box with a bold title and one line of detail."""
    ax.add_patch(
        FancyBboxPatch(
            (x_mm, y_mm),
            BOX_W_MM,
            BOX_H_MM,
            boxstyle="round,pad=0,rounding_size=1.5",
            fc="white",
            ec=color,
            lw=lw,
        )
    )
    centre = x_mm + BOX_W_MM / 2
    ax.text(centre, y_mm + 11.0, title, ha="center", va="center", fontweight="bold")
    ax.text(centre, y_mm + 5.5, detail, ha="center", va="center", color="0.25")


def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            lw=0.8,
            color="0.3",
            shrinkA=0,
            shrinkB=0,
        )
    )


if __name__ == "__main__":
    fc.require(*GNSS_FILES)
    config = fc.run_config(fc.pooled_run(RUN))
    epochs = fc.load_gnss_epochs(GNSS_FILES)
    masks = fc.gnss_rejection_masks(epochs["height_m"])
    n_epochs, n_kept = epochs["gps_s"].size, int(masks["kept"].sum())
    gate_s = config["pair_threshold_s"]

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
    box(ax, x_top[0], top, "1  Position antennas", "PRIDE-PPPAR, 1 Hz")
    box(
        ax,
        x_top[1],
        top,
        "2  Reject bad epochs",
        f"{thousands(n_kept)} of {thousands(n_epochs)} kept",
    )
    box(ax, x_top[2], top, "3  Locate transducer", "rigid-body fit, lever arm")
    box(ax, x_top[3], top, "4  Pair arrivals", f"with emissions, gate {gate_s:g} s")
    box(
        ax,
        x_bottom[0],
        bottom,
        "5  Align DOG clock",
        "integer second, then fraction",
        color=fc.ACCENT,
        lw=1.8,
    )
    box(ax, x_bottom[1], bottom, "6  Invert", "Gauss–Newton start values")
    box(
        ax,
        x_bottom[2],
        bottom,
        "7  Sample posterior",
        "MCMC, AR(1) noise",
    )
    box(
        ax,
        x_bottom[3],
        bottom,
        "8  Report position",
        fc.FINAL_SOLUTION,
        color="black",
        lw=1.8,
    )

    mid_top, mid_bottom = top + BOX_H_MM / 2, bottom + BOX_H_MM / 2
    for i in range(3):
        arrow(ax, (x_top[i] + BOX_W_MM, mid_top), (x_top[i + 1], mid_top))
        arrow(ax, (x_bottom[i], mid_bottom), (x_bottom[i + 1] + BOX_W_MM, mid_bottom))
    centre_4 = x_top[3] + BOX_W_MM / 2
    arrow(ax, (centre_4, top), (centre_4, bottom + BOX_H_MM))
    fc.save(fig, NAME)
