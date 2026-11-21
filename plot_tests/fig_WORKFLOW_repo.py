"""Repository workflow: every input, processing step, validation and outcome.

Message: the repository takes PRIDE-PPPAR antenna positions and DOG arrival times
through eight named steps to the DOG positions, and the synthetic and
model-sensitivity tests feed the same inversion and sampler.

Broader than FIG-1 (the paper's processing chain): for the response letter, the
supplement and readers who want to reproduce the results. Each box names the
module or script in ``src/`` that does the work (paths relative to ``src/``;
``Inversion_Workflow/`` is left out of the names of its sub-packages).

- Inputs (left): GNSS processing outside the repository (dashed), the four-antenna
  positions, the configuration, the DOG arrival times, the sound-speed profile and
  ESV tables, and the GEBCO grid.
- Steps (middle), in order, top to bottom.
- Validation (right): the integer-second test, the synthetic generators and the
  ensemble tests that invert them with the same Gauss-Newton code, and the MCMC
  model variants.
- Outcomes (bottom).
Arrows are horizontal or vertical and cross no box or text.

Inputs: none (the diagram is drawn from the names in the configuration block;
``check_modules`` fails if a named module no longer exists).
Run from the repo root with PYTHONPATH=src.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import _figure_common as fc
from plotting import paper_style

# --- configuration -----------------------------------------------------------------
NAME = "fig_WORKFLOW_repo"
HEIGHT_MM = 216.0
BOX_H_MM = 14.5
ROW_PITCH_MM = 19.0
FIRST_ROW_TOP_MM = HEIGHT_MM - 9.0  # top of row 0; the column headers sit above
# Columns: (left edge, width) in mm.
COLUMNS = {"input": (2.0, 46.0), "step": (57.0, 54.0), "validation": (120.0, 52.0)}
HEADERS = {
    "input": "Inputs",
    "step": "Processing steps",
    "validation": "Validation",
}
# One light fill per role; inputs are white, the external input is dashed.
FILL = {
    "input": "white",
    "external": "white",
    "step": "0.90",
    "validation": "#DDEBF7",
    "outcome": "#FCEBD5",
}
IW = "Inversion_Workflow"
# (key, role, column, row, title, module lines, files that must exist under src/).
BOXES = (
    (
        "pride",
        "external",
        "input",
        0,
        "GNSS observations",
        ("PRIDE-PPPAR", "(outside the repository)"),
        (),
    ),
    (
        "antennas",
        "input",
        "input",
        1,
        "4-antenna positions",
        ("GNSS_Processing/ppp2mat.py",),
        ("GNSS_Processing/ppp2mat.py",),
    ),
    (
        "config",
        "input",
        "input",
        2,
        "Configuration",
        ("bermuda_data.py", "run_inversion.py settings"),
        ("bermuda_data.py", "run_inversion.py"),
    ),
    (
        "arrivals",
        "input",
        "input",
        3,
        "DOG arrival times",
        ("CDOG_Data/DOG{n}-camp.mat",),
        (),
    ),
    (
        "profile",
        "input",
        "input",
        4,
        "Sound-speed profile",
        ("acoustics/svp.py",),
        ("acoustics/svp.py",),
    ),
    (
        "esv_tables",
        "input",
        "input",
        5,
        "ESV tables",
        ("acoustics/ray_tracing.py", "acoustics/ESV_table_Bermuda.py"),
        ("acoustics/ray_tracing.py", "acoustics/ESV_table_Bermuda.py"),
    ),
    (
        "gebco",
        "input",
        "input",
        8,
        "GEBCO bathymetry",
        ("gebco_2025_bermuda_*.asc",),
        (),
    ),
    (
        "qc",
        "step",
        "step",
        1,
        "1  GNSS quality control",
        ("Bermuda/Initialize_Bermuda_Data.py",),
        (f"{IW}/Bermuda/Initialize_Bermuda_Data.py",),
    ),
    (
        "transducer",
        "step",
        "step",
        2,
        "2  Transducer position",
        ("geometry/rigid_body.py", "Forward_Model/Find_Transponder.py"),
        ("geometry/rigid_body.py", f"{IW}/Forward_Model/Find_Transponder.py"),
    ),
    (
        "pairing",
        "step",
        "step",
        3,
        "3  Pairing arrivals to emissions",
        ("Inversion/Numba_xAline.py:", "two_pointer_index"),
        (f"{IW}/Inversion/Numba_xAline.py",),
    ),
    (
        "clock",
        "step",
        "step",
        4,
        "4  Clock alignment",
        ("integer second, then fraction:", "Numba_xAline.py: align_offset"),
        (),
    ),
    (
        "esv",
        "step",
        "step",
        5,
        "5  ESV correction",
        ("Forward_Model/Find_ESV.py", "Forward_Model/Calculate_Times_Bias.py"),
        (
            f"{IW}/Forward_Model/Find_ESV.py",
            f"{IW}/Forward_Model/Calculate_Times_Bias.py",
        ),
    ),
    (
        "gauss_newton",
        "step",
        "step",
        6,
        "6  Gauss–Newton inversion",
        ("run_inversion.py", "Inversion/Numba_xAline_Geiger_bias.py"),
        (f"{IW}/Inversion/Numba_xAline_Geiger_bias.py",),
    ),
    (
        "mcmc",
        "step",
        "step",
        7,
        "7  MCMC sampling",
        ("MCMC_Functions/MCMC_sampler.py",),
        (f"{IW}/MCMC_Functions/MCMC_sampler.py",),
    ),
    (
        "posterior",
        "step",
        "step",
        8,
        "8  Posterior analysis",
        ("MCMC_Functions/MCMC_diagnostics.py", "MCMC_posterior_summary.py"),
        (
            f"{IW}/MCMC_Functions/MCMC_diagnostics.py",
            f"{IW}/MCMC_Functions/MCMC_posterior_summary.py",
        ),
    ),
    (
        "integer",
        "validation",
        "validation",
        4,
        "Integer-second test",
        ("Bermuda/Integer_Ambiguity.py",),
        (f"{IW}/Bermuda/Integer_Ambiguity.py",),
    ),
    (
        "generators",
        "validation",
        "validation",
        5,
        "Synthetic generators",
        ("Synthetic/Modular_Synthetic.py", "Synthetic/Realistic_Synthetic.py"),
        (
            f"{IW}/Synthetic/Modular_Synthetic.py",
            f"{IW}/Synthetic/Realistic_Synthetic.py",
        ),
    ),
    (
        "ensembles",
        "validation",
        "validation",
        6,
        "Coverage, budget, geometry tests",
        ("Realistic_Synthetic_Ensemble.py",),
        (f"{IW}/Synthetic/Realistic_Synthetic_Ensemble.py",),
    ),
    (
        "variants",
        "validation",
        "validation",
        7,
        "Model-sensitivity runs",
        ("MCMC_sampler.py: RUN_VARIANTS", "Realistic_Synthetic_MCMC.py"),
        (f"{IW}/Synthetic/Realistic_Synthetic_MCMC.py",),
    ),
)
# Arrows between boxes: (from, to, direction of travel).
ARROWS = (
    ("pride", "antennas", "down"),
    ("antennas", "qc", "right"),
    ("config", "transducer", "right"),
    ("arrivals", "pairing", "right"),
    ("profile", "esv_tables", "down"),
    ("esv_tables", "esv", "right"),
    ("qc", "transducer", "down"),
    ("transducer", "pairing", "down"),
    ("pairing", "clock", "down"),
    ("clock", "esv", "down"),
    ("esv", "gauss_newton", "down"),
    ("gauss_newton", "mcmc", "down"),
    ("mcmc", "posterior", "down"),
    ("integer", "clock", "left"),
    ("generators", "ensembles", "down"),
    ("ensembles", "gauss_newton", "left"),
    ("variants", "mcmc", "left"),
)
# Outcomes below the columns: the figures under the inputs, the results in a frame
# under the steps and the validation column. (title, detail line).
FIGURES_OUTCOME = ("Paper figures and tables", "plot_tests/, README.md")
RESULT_OUTCOMES = (
    ("DOG positions", "ECEF and E, N, U ± posterior sd"),
    ("Lever arm, clock offsets", "posterior mean ± sd"),
    ("ESV correction time series", "30-min windows, per-DOG offset"),
    ("Diagnostics", "R̂, ESS, residuals"),
)
OUTCOME_H_MM = 12.5
OUTCOME_GAP_MM = 3.0
RESULTS_TOP_MM = 34.5  # top of the frame around the result outcomes
FRAME_PAD_MM = 2.5
LINE_STEP_MM = 3.7  # line spacing of the text in a box
# -----------------------------------------------------------------------------------


def check_modules():
    """Fail if a module named in a box no longer exists under src/."""
    missing = [
        path
        for *_, files in BOXES
        for path in files
        if not (fc.REPO / "src" / path).exists()
    ]
    if missing:
        raise FileNotFoundError(f"modules named in the diagram are missing: {missing}")


def draw_box(ax, x, y, width, height, role, title, details):
    """Rounded box (lower-left corner x, y) with a title and grey detail lines."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0,rounding_size=1.2",
            fc=FILL[role],
            ec="0.3",
            lw=0.7,
            ls="--" if role == "external" else "-",
        )
    )
    lines = 1 + len(details)
    top = y + height / 2 + 0.5 * (lines - 1) * LINE_STEP_MM
    ax.text(x + width / 2, top, title, ha="center", va="center")
    for k, detail in enumerate(details, start=1):
        ax.text(
            x + width / 2,
            top - k * LINE_STEP_MM,
            detail,
            ha="center",
            va="center",
            color="0.35",
        )


def draw_arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7,
            lw=0.8,
            color="0.25",
            shrinkA=0,
            shrinkB=0,
        )
    )


def box_rectangle(column, row):
    """Lower-left corner, width and height (mm) of the box in a column and row."""
    x, width = COLUMNS[column]
    y = FIRST_ROW_TOP_MM - row * ROW_PITCH_MM - BOX_H_MM
    return x, y, width, BOX_H_MM


def connect(ax, rectangles, start, end, direction):
    """Straight arrow between facing edges of two boxes in the same row or column."""
    x0, y0, w0, h0 = rectangles[start]
    x1, y1, w1, h1 = rectangles[end]
    if direction == "down":
        draw_arrow(ax, (x0 + w0 / 2, y0), (x0 + w0 / 2, y1 + h1))
    elif direction == "right":
        draw_arrow(ax, (x0 + w0, y0 + h0 / 2), (x1, y0 + h0 / 2))
    else:
        draw_arrow(ax, (x0, y0 + h0 / 2), (x1 + w1, y0 + h0 / 2))


def draw():
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

    for column, header in HEADERS.items():
        x, width = COLUMNS[column]
        # The validation column starts lower, at its first box.
        row = 4 if column == "validation" else 0
        y = FIRST_ROW_TOP_MM - row * ROW_PITCH_MM + 2.0
        ax.text(
            x + width / 2,
            y,
            header,
            ha="center",
            va="bottom",
            fontsize=paper_style.PANEL_LABEL_SIZE_PT,
        )

    rectangles = {}
    for key, role, column, row, title, details, _ in BOXES:
        rectangles[key] = box_rectangle(column, row)
        draw_box(ax, *rectangles[key], role, title, details)
    for start, end, direction in ARROWS:
        connect(ax, rectangles, start, end, direction)

    # Outcomes: figures under the inputs; results in a frame under the other columns.
    frame_left = COLUMNS["step"][0] - FRAME_PAD_MM
    frame_right = sum(COLUMNS["validation"]) + FRAME_PAD_MM - 2.5
    frame_bottom = RESULTS_TOP_MM - 2 * OUTCOME_H_MM - OUTCOME_GAP_MM - 2 * FRAME_PAD_MM
    ax.add_patch(
        Rectangle(
            (frame_left, frame_bottom),
            frame_right - frame_left,
            RESULTS_TOP_MM - frame_bottom,
            fc="none",
            ec="0.5",
            lw=0.6,
        )
    )
    ax.text(
        frame_right - 1.5,
        RESULTS_TOP_MM + 1.5,
        "Outcomes",
        ha="right",
        va="bottom",
        fontsize=paper_style.PANEL_LABEL_SIZE_PT,
    )
    inner = frame_right - frame_left - 2 * FRAME_PAD_MM
    outcome_w = (inner - OUTCOME_GAP_MM) / 2
    for k, (title, detail) in enumerate(RESULT_OUTCOMES):
        x = frame_left + FRAME_PAD_MM + (k % 2) * (outcome_w + OUTCOME_GAP_MM)
        y = (
            RESULTS_TOP_MM
            - FRAME_PAD_MM
            - OUTCOME_H_MM
            - (k // 2) * (OUTCOME_H_MM + OUTCOME_GAP_MM)
        )
        draw_box(ax, x, y, outcome_w, OUTCOME_H_MM, "outcome", title, (detail,))
    # Posterior analysis feeds the results; the results and GEBCO feed the figures.
    x, y, width, _ = rectangles["posterior"]
    draw_arrow(ax, (x + width / 2, y), (x + width / 2, RESULTS_TOP_MM))
    x, width = COLUMNS["input"]
    figures_y = RESULTS_TOP_MM - FRAME_PAD_MM - OUTCOME_H_MM
    draw_box(
        ax,
        x,
        figures_y,
        width,
        OUTCOME_H_MM,
        "outcome",
        *FIGURES_OUTCOME[:1],
        FIGURES_OUTCOME[1:],
    )
    draw_arrow(
        ax,
        (x + width / 2, rectangles["gebco"][1]),
        (x + width / 2, figures_y + OUTCOME_H_MM),
    )
    draw_arrow(
        ax,
        (frame_left, figures_y + OUTCOME_H_MM / 2),
        (x + width, figures_y + OUTCOME_H_MM / 2),
    )
    return fig


if __name__ == "__main__":
    check_modules()
    fc.save(draw(), NAME)
