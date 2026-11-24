"""Posterior of three parameter triples of the final model as 3-D clouds.

Question: how tightly do the data constrain a DOG position and the lever arm, and
which parameters trade off against each other?
Message: (a) the position of a DOG is a vertical needle, about 1 cm wide
horizontally and 4 cm vertically; (b) the lever arm is determined to millimetres
horizontally and about 1 cm vertically; (c) the ESV-bias offset and timing bias of
a DOG and the height of the transducer above it, l_z − U, are almost perfectly
correlated: the data fix one combination of the three far better than any one.

Every point is one posterior draw, about the posterior mean; the same draws are
projected on the three walls. Colour is the log posterior density of the three
plotted parameters relative to its maximum, ln P − ln P_max (linear scale), from a
3-D kernel density estimate of the draws; draws below the colour range take its
end colour. The line on each wall is the 95 % ellipse of that projection (Gaussian
with the posterior covariance). In panel a the three axes share one scale; in b
and c each axis spans a fixed number of posterior standard deviations.

Input: the final MCMC run.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import _figure_common as fc
import _style as st

NAME = "fig05_posterior_3d"
DOG = 3  # DOG of panels a and c
ENU_HALF_WIDTH_CM = 20.0  # panel a: cubic box, same scale on E, N and U
LEVER_BOX_SD = 5.0  # panel b: half-width of each axis in posterior sd
# Panel c: a wide box and a view turned to the side keep the tilted cloud clear of
# its projections on the walls.
TRADEOFF_BOX_SD = 6.0
TRADEOFF_VIEW = {"elev": 18.0, "azim": -38.0}
VIEW = {"elev": 16.0, "azim": -58.0}
N_DRAWS = 2500  # drawn per panel (thinned from the pooled chains)
SEED = 20261121
CMAP = "viridis"  # bright = near the maximum of the posterior
DELTA_MIN = -5.0  # lower end of the colour scale
POINT_SIZE = 2.2
POINT_ALPHA = 0.85
SHADOW_SIZE = 1.3
SHADOW_ALPHA = 0.5
ELLIPSE_PROBABILITY = 0.95
ELLIPSE_STYLE = {"color": "0.25", "lw": 0.7}
PANEL_WIDTH = 0.31  # of the figure width
PANEL_LEFT = (0.005, 0.33, 0.645)


def round_tick(value):
    """Largest of 1, 2 or 5 times a power of ten that does not exceed value."""
    base = 10.0 ** np.floor(np.log10(value))
    return max(m * base for m in (1.0, 2.0, 5.0) if m * base <= value)


def draw_cloud(ax, draws, half_widths, labels, rng, view, zlabel_pad=-8):
    """Draws (n, 3) about their mean as a 3-D cloud with coloured wall projections.

    half_widths (3,) are the half-lengths of the box along the three parameters,
    in their units; labels are the three axis labels.
    """
    centred = draws - draws.mean(axis=0)
    keep = rng.choice(centred.shape[0], N_DRAWS, replace=False)
    shown = centred[keep]
    # Density in box units, so that parameters of different units weigh the same.
    scaled = shown / half_widths
    density = gaussian_kde(scaled.T)(scaled.T)
    delta = np.log(density / density.max())
    order = np.argsort(delta)  # the highest posterior is drawn last, on top
    shown, delta = shown[order], delta[order]
    x, y, z = shown.T
    colours = {"c": delta, "cmap": CMAP, "vmin": DELTA_MIN, "vmax": 0.0}
    shadow = {"s": SHADOW_SIZE, "alpha": SHADOW_ALPHA, "linewidths": 0, **colours}
    ax.scatter(x, y, zs=-half_widths[2], zdir="z", rasterized=True, **shadow)
    ax.scatter(y, z, zs=-half_widths[0], zdir="x", rasterized=True, **shadow)
    ax.scatter(x, z, zs=half_widths[1], zdir="y", rasterized=True, **shadow)
    covariance = np.cov(centred.T)
    for (i, j), zdir, position in (
        ((0, 1), "z", -half_widths[2]),
        ((1, 2), "x", -half_widths[0]),
        ((0, 2), "y", half_widths[1]),
    ):
        outline = fc.ellipse_points(
            (0.0, 0.0), covariance[np.ix_((i, j), (i, j))], ELLIPSE_PROBABILITY
        )
        ax.plot(*outline.T, zs=position, zdir=zdir, **ELLIPSE_STYLE)
    cloud = ax.scatter(
        x,
        y,
        z,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        linewidths=0,
        depthshade=False,
        rasterized=True,
        **colours,
    )
    for axis, setter, half_width in zip(
        (ax.xaxis, ax.yaxis, ax.zaxis),
        (ax.set_xlim, ax.set_ylim, ax.set_zlim),
        half_widths,
        strict=True,
    ):
        setter(-half_width, half_width)
        tick = round_tick(0.8 * half_width)
        axis.set_ticks((-tick, 0.0, tick))
        axis.set_pane_color("white")
        axis.pane.set_edgecolor("0.6")
        axis._axinfo["grid"].update(color="0.88", linewidth=0.5)
    ax.computed_zorder = False  # keep the plotting order: walls, then the cloud
    ax.set_box_aspect((1, 1, 1), zoom=0.95)
    ax.patch.set_alpha(0.0)  # neighbouring axes overlap; keep their labels visible
    ax.view_init(**view)
    ax.set_xlabel(labels[0], labelpad=-6)
    ax.set_ylabel(labels[1], labelpad=-6)
    ax.set_zlabel(labels[2], labelpad=zlabel_pad)
    ax.tick_params(pad=-3)
    return cloud


def sd_header(draws, symbols, unit):
    """Text with the posterior standard deviation of each plotted parameter."""
    sd = draws.std(axis=0)
    parts = [
        f"$\\sigma_{s}$ = {value:.1f}" for s, value in zip(symbols, sd, strict=True)
    ]
    return ",  ".join(parts) + f" {unit}"


def correlation_header(draws, symbols):
    """Text with the three pairwise correlations of the plotted parameters."""
    rho = np.corrcoef(draws.T)
    return ",  ".join(
        f"$\\rho_{{{symbols[i]}{symbols[j]}}}$ = {fc.signed(rho[i, j], '.2f')}"
        for i, j in ((0, 1), (0, 2), (1, 2))
    )


def main():
    st.apply()
    plt.rcParams["figure.constrained_layout.use"] = False  # 3-D axes placed by hand
    pooled = fc.pooled_run(fc.FINAL_RUN)
    j = fc.run_config(pooled)["dogs"].index(DOG)
    enu_cm, _ = fc.dog_enu_cm(pooled, DOG)
    lever_cm = pooled["lever"] * 100.0
    tradeoff = np.column_stack(
        [
            # The DOG's ESV bias minus the common window series.
            (pooled["esv_bias"][:, j, 0] - pooled["esv_common"][:, 0]) * 1e3,
            pooled["dt"][:, j] * 1e6,
            lever_cm[:, 2] - enu_cm[:, 2],
        ]
    )
    panels = (
        {
            "title": f"DOG {DOG} position",
            "header": sd_header(enu_cm, "ENU", "cm"),
            "draws": enu_cm,
            "half_widths": np.full(3, ENU_HALF_WIDTH_CM),
            "labels": ("East $E$ (cm)", "North $N$ (cm)", "Up $U$ (cm)"),
            "view": VIEW,
        },
        {
            "title": "Lever arm",
            "header": sd_header(lever_cm, "xyz", "cm"),
            "draws": lever_cm,
            "half_widths": LEVER_BOX_SD * lever_cm.std(axis=0),
            "labels": ("$l_x$ (cm)", "$l_y$ (cm)", "$l_z$ (cm)"),
            "view": VIEW,
        },
        {
            "title": f"DOG {DOG} biases and height",
            "header": correlation_header(tradeoff, ("c", "t", "z")),
            "draws": tradeoff,
            "half_widths": TRADEOFF_BOX_SD * tradeoff.std(axis=0),
            "labels": (
                "ESV bias $\\delta c$ (mm/s)",
                "Timing bias $\\delta t$ (µs)",
                "$l_z - U$ (cm)",
            ),
            "view": TRADEOFF_VIEW,
            "zlabel_pad": -6,  # the side view puts the tick labels further out
        },
    )
    rng = np.random.default_rng(SEED)
    fig = plt.figure(figsize=(st.DOUBLE_MM / st.MM_PER_INCH, 80.0 / st.MM_PER_INCH))
    for panel, left, letter in zip(panels, PANEL_LEFT, "abc", strict=True):
        ax = fig.add_axes([left, 0.13, PANEL_WIDTH, 0.80], projection="3d")
        cloud = draw_cloud(
            ax,
            panel["draws"],
            panel["half_widths"],
            panel["labels"],
            rng,
            panel["view"],
            panel.get("zlabel_pad", -8),
        )
        fig.text(
            left + 0.015, 0.93, letter, fontsize=st.PANEL_LETTER_PT, fontweight="bold"
        )
        fig.text(left + 0.045, 0.93, panel["title"])
        fig.text(left + 0.045, 0.875, panel["header"])
        print(
            f"{panel['title']}: sd {panel['draws'].std(axis=0).round(3)}; correlations "
            f"{np.corrcoef(panel['draws'].T)[np.triu_indices(3, 1)].round(3)}"
        )
    bar_axes = fig.add_axes([0.34, 0.13, 0.32, 0.024])
    colourbar = fig.colorbar(
        cloud, cax=bar_axes, orientation="horizontal", extend="min"
    )
    colourbar.solids.set_alpha(1.0)
    colourbar.set_label(
        "Log posterior relative to its maximum, $\\ln P - \\ln P_{max}$", labelpad=2
    )
    st.save(fig, NAME)


if __name__ == "__main__":
    main()
