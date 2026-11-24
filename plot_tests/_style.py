"""Style of the revision figures, sized for Satellite Navigation (SpringerOpen).

Journal rules (submission guidelines, read 2026-11-21): figures print 85 mm (half
page) or 170 mm (full page) wide and at most 225 mm high including the legend; lines
wider than 0.25 pt; fonts embedded; keys inside the graphic, titles in the caption.

Use: ``apply()`` once, ``fig, axes = figure(170, 70, ncols=2)``, plot,
``save(fig, name)``.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import _figure_common as fc

SINGLE_MM = 85.0
DOUBLE_MM = 170.0
MM_PER_INCH = 25.4
FONT_PT = 8.0  # every text on a figure (R3 #11 asks for at least 8 pt)
PANEL_LETTER_PT = 10.0

# Grey scale: text and results are INK, comparisons CONTEXT, guides FAINT.
INK = "black"
CONTEXT = "0.50"
FAINT = "0.80"

# One colour and marker per DOG where DOGs share a panel (Okabe-Ito, colour-blind
# safe). Text that names a DOG is always black, in the standard font.
DOG = {
    1: {"color": "#0072B2", "marker": "o", "label": "DOG 1"},
    3: {"color": "#D55E00", "marker": "s", "label": "DOG 3"},
    4: {"color": "#009E73", "marker": "^", "label": "DOG 4"},
}
DOGS = (1, 3, 4)

OUTPUT_DIR = fc.OUTPUT_DIR  # Bermuda_Data/Figs/plot_tests (gitignored)
CACHE_DIR = OUTPUT_DIR / "cache"


def apply():
    """Set the matplotlib rcParams of the figure style."""
    mpl.rcParams.update(
        {
            "text.usetex": False,  # MCMC_sampler turns LaTeX on at import
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "sans",
            "mathtext.it": "sans:italic",
            "mathtext.bf": "sans:bold",
            "font.size": FONT_PT,
            "axes.titlesize": FONT_PT,
            "axes.labelsize": FONT_PT,
            "xtick.labelsize": FONT_PT,
            "ytick.labelsize": FONT_PT,
            "legend.fontsize": FONT_PT,
            "axes.linewidth": 0.6,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.grid": True,
            "grid.color": "0.88",
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "lines.linewidth": 1.0,
            "lines.markersize": 4.0,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.7",
            "legend.fancybox": False,
            "legend.handlelength": 1.4,
            "figure.constrained_layout.use": True,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,  # embedded, editable TrueType
            "ps.fonttype": 42,
        }
    )


def figure(width_mm, height_mm, **subplots_kwargs):
    """``plt.subplots`` at the printed size (mm)."""
    if width_mm not in (SINGLE_MM, DOUBLE_MM):
        raise ValueError(f"width must be {SINGLE_MM} or {DOUBLE_MM} mm; got {width_mm}")
    size = (width_mm / MM_PER_INCH, height_mm / MM_PER_INCH)
    return plt.subplots(figsize=size, **subplots_kwargs)


def panel_letter(ax, letter, x_pt=-28.0, y_pt=4.0):
    """Bold lower-case panel letter above the top-left corner (Springer style)."""
    ax.annotate(
        letter,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(x_pt, y_pt),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=PANEL_LETTER_PT,
        fontweight="bold",
    )


def hours(gps_seconds):
    """Hours since the first GNSS epoch of the survey, the time axis of every figure."""
    from plotting.paper_style import hours_since_start

    return hours_since_start(gps_seconds)


TIME_LABEL = "Time $t$ since survey start (h)"
TIME_LIMITS_H = (0.0, 17.0)  # as plotting.paper_style.TIME_LIMITS_H
TIME_TICKS_H = (0, 2, 4, 6, 8, 10, 12, 14, 16)
DATA_BLUE = "#1f77b4"  # observations, as in the residual plots of Plot_Modular
SIGMA_RED = "#d62728"  # ±1 sd lines, as in Plot_Modular


def save(fig, name):
    """Report size, font and text-overlap problems, then write name.pdf and name.png."""
    for problem in fc.figure_problems(fig, (SINGLE_MM, DOUBLE_MM)):
        print(f"CHECK {name}: {problem}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}.pdf")
    fig.savefig(OUTPUT_DIR / f"{name}.png")
    print(f"saved {OUTPUT_DIR / name}.pdf")


DENSITY_CMAP = "viridis"
DENSITY_LABEL = "Nearby observations"


def density_scatter(ax, x, y, x_radius, y_radius, **scatter_kwargs):
    """Scatter coloured by the number of points near each point (densest on top).

    Same encoding as the Puerto Rico elevation plots: the colour is the count of
    points within x_radius and y_radius of each point (fc.nearby_counts).
    """
    counts = fc.nearby_counts(x, y, x_radius, y_radius)
    order = np.argsort(counts)
    return ax.scatter(
        np.asarray(x)[order],
        np.asarray(y)[order],
        c=counts[order],
        cmap=DENSITY_CMAP,
        linewidths=0,
        rasterized=True,
        **scatter_kwargs,
    )
