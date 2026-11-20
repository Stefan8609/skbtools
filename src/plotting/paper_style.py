"""One figure style for Paper I (reviewer comments R1 4e, R3 #10, R3 #11, R4 m2).

- Figures are drawn at their printed width (single or double column) and saved as
  vector PDF, with a PNG preview for slides. Only dense scatter layers are
  rasterized (``rasterize_dense``), so text and lines stay vector.
- Every text element is at least 8 pt.
- One colour and marker per DOG (``DOG_STYLE``, Okabe-Ito, colour-blind safe), and
  the label "DOG n" everywhere.
- One time axis: hours since the first GPS epoch of the processed Bermuda survey
  (``hours_since_start``, ``TIME_LABEL``, ``TIME_LIMITS_H``).
- Times convert to range at ``RANGE_SOUND_SPEED_MPS`` everywhere (R1 4i).

Use: ``apply()`` once, ``fig, ax = figure("single", 60)``, plot, ``save(fig, name)``.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    NOMINAL_SOUND_SPEED_MPS,
)

# Springer column widths; check against the current Satellite Navigation guidelines.
SINGLE_COLUMN_MM = 84.0
DOUBLE_COLUMN_MM = 174.0
MM_PER_INCH = 25.4
FONT_SIZE_PT = 8.0
PANEL_LABEL_SIZE_PT = 9.0
LINE_WIDTH_PT = 0.8
RASTERIZE_ABOVE_POINTS = 5000

SURVEY_START_GPS_S = 20056.0  # first epoch of Processed_GPS_Receivers_DOG_*.npz
TIME_LABEL = "Time since survey start (h)"
TIME_LIMITS_H = (0.0, 14.0)
RANGE_SOUND_SPEED_MPS = NOMINAL_SOUND_SPEED_MPS

# Okabe-Ito colours with distinct markers.
DOG_STYLE = {
    1: {"color": "#E69F00", "marker": "o", "label": "DOG 1"},
    3: {"color": "#56B4E9", "marker": "s", "label": "DOG 3"},
    4: {"color": "#009E73", "marker": "^", "label": "DOG 4"},
}
# Further categories (models, tracks) in the same palette, after the DOG colours.
CATEGORY_COLORS = ["#0072B2", "#D55E00", "#CC79A7", "#000000", "#F0E442", "#999999"]

FIGURE_DIR = Path(__file__).resolve().parents[2] / "Bermuda_Data" / "Figs" / "paper"


def apply():
    """Set the matplotlib rcParams of the paper style."""
    mpl.rcParams.update(
        {
            # Some plotting modules turn LaTeX on at import; the paper figures use
            # matplotlib text (Unicode, editable in the PDF).
            "text.usetex": False,
            "font.family": "sans-serif",
            # Arial/Helvetica as the journal's sans-serif; DejaVu Sans if absent.
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "sans",
            "mathtext.it": "sans:italic",
            "mathtext.bf": "sans:bold",
            "mathtext.default": "regular",
            "font.size": FONT_SIZE_PT,
            "axes.titlesize": FONT_SIZE_PT,
            "axes.labelsize": FONT_SIZE_PT,
            "xtick.labelsize": FONT_SIZE_PT,
            "ytick.labelsize": FONT_SIZE_PT,
            "legend.fontsize": FONT_SIZE_PT,
            "figure.titlesize": FONT_SIZE_PT,
            "lines.linewidth": LINE_WIDTH_PT,
            "lines.markersize": 3.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,  # editable TrueType text in the PDF
            "ps.fonttype": 42,
            "image.cmap": "viridis",
        }
    )


def figure(width="single", height_mm=60.0, **subplots_kwargs):
    """``plt.subplots`` at the printed size: width "single", "double" or mm."""
    if width == "single":
        width_mm = SINGLE_COLUMN_MM
    elif width == "double":
        width_mm = DOUBLE_COLUMN_MM
    else:
        width_mm = float(width)
    size = (width_mm / MM_PER_INCH, height_mm / MM_PER_INCH)
    return plt.subplots(figsize=size, **subplots_kwargs)


def hours_since_start(gps_seconds):
    """Hours since the first GPS epoch of the survey (the common time axis)."""
    return (gps_seconds - SURVEY_START_GPS_S) / 3600.0


def panel_label(ax, letter):
    """Bold "(a)" in the top-left corner, outside the data area."""
    ax.text(
        -0.02,
        1.02,
        f"({letter})",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=PANEL_LABEL_SIZE_PT,
        fontweight="bold",
    )


def rasterize_dense(artist, n_points):
    """Rasterize a scatter layer with many points; keeps the PDF small."""
    if n_points > RASTERIZE_ABOVE_POINTS:
        artist.set_rasterized(True)
    return artist


def save(fig, name, directory=None):
    """Save ``name``.pdf (vector) and ``name``.png (preview); return the PDF path."""
    directory = Path(directory) if directory is not None else FIGURE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    pdf = directory / f"{name}.pdf"
    fig.savefig(pdf)
    fig.savefig(directory / f"{name}.png")
    return pdf
