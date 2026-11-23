"""Posterior of the three DOG positions as 3-D clouds with wall projections.

Question: how tightly do the data constrain each DOG position, and in which
direction is it weakest?
Message: each posterior is a vertical needle: about 1 cm wide horizontally and 4 to
6 cm vertically, with no tilt (no horizontal-vertical correlation).

Every point is one posterior draw; the same draws are projected on the three
walls. The three axes share one scale, so the shape of the cloud is the shape of
the uncertainty. Colour is the log posterior density of the position relative to
its maximum, ln P − ln P_max (linear scale), from a 3-D kernel density estimate of
the draws; draws below the colour range take its end colour. The line on each wall
is the 95 % ellipse of that projection (Gaussian with the posterior covariance).

Input: the final MCMC run.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import _figure_common as fc
import _style as st

NAME = "fig05_posterior_3d"
HALF_WIDTH_CM = 20.0  # cubic box, same scale on E, N and U
TICKS_CM = (-15, 0, 15)
N_DRAWS = 2500  # drawn per DOG (thinned from the pooled chains)
SEED = 20261121
VIEW = {"elev": 16.0, "azim": -58.0}
CMAP = "viridis"  # bright = near the maximum of the posterior
DELTA_MIN = -5.0  # lower end of the colour scale
POINT_SIZE = 2.2
POINT_ALPHA = 0.85
ELLIPSE_PROBABILITY = 0.95
ELLIPSE_STYLE = {"color": "0.25", "lw": 0.7}
SHADOW_SIZE = 1.3
SHADOW_ALPHA = 0.5


def draw_dog(ax, enu_cm, rng):
    wall = HALF_WIDTH_CM
    keep = rng.choice(enu_cm.shape[0], N_DRAWS, replace=False)
    draws = enu_cm[keep]
    density = gaussian_kde(draws.T)(draws.T)
    delta = np.log(density / density.max())
    order = np.argsort(delta)
    draws, delta = draws[order], delta[order]
    east, north, up = draws.T
    colours = {"c": delta, "cmap": CMAP, "vmin": DELTA_MIN, "vmax": 0.0}
    shadow = {"s": SHADOW_SIZE, "alpha": SHADOW_ALPHA, "linewidths": 0, **colours}
    ax.scatter(east, north, zs=-wall, zdir="z", rasterized=True, **shadow)
    ax.scatter(north, up, zs=-wall, zdir="x", rasterized=True, **shadow)
    ax.scatter(east, up, zs=wall, zdir="y", rasterized=True, **shadow)
    covariance = np.cov(enu_cm.T)
    for (i, j), zdir, position in (
        ((0, 1), "z", -wall),
        ((1, 2), "x", -wall),
        ((0, 2), "y", wall),
    ):
        outline = fc.ellipse_points(
            (0.0, 0.0), covariance[np.ix_((i, j), (i, j))], ELLIPSE_PROBABILITY
        )
        ax.plot(*outline.T, zs=position, zdir=zdir, **ELLIPSE_STYLE)
    cloud = ax.scatter(
        east,
        north,
        up,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        linewidths=0,
        depthshade=False,
        rasterized=True,
        **colours,
    )
    for axis, setter in zip(
        (ax.xaxis, ax.yaxis, ax.zaxis),
        (ax.set_xlim, ax.set_ylim, ax.set_zlim),
        strict=True,
    ):
        setter(-wall, wall)
        axis.set_ticks(TICKS_CM)
        axis.set_pane_color("white")
        axis.pane.set_edgecolor("0.6")
        axis._axinfo["grid"].update(color="0.88", linewidth=0.5)
    ax.computed_zorder = False  # keep the plotting order: walls, then the cloud
    ax.set_box_aspect((1, 1, 1), zoom=0.95)
    ax.patch.set_alpha(0.0)  # neighbouring axes overlap; keep their labels visible
    ax.view_init(**VIEW)
    ax.set_xlabel("East $E$ (cm)", labelpad=-6)
    ax.set_ylabel("North $N$ (cm)", labelpad=-6)
    ax.set_zlabel("Up $U$ (cm)", labelpad=-8)
    ax.tick_params(pad=-3)
    return cloud


def main():
    st.apply()
    plt.rcParams["figure.constrained_layout.use"] = False  # 3-D axes placed by hand
    pooled = fc.pooled_run(fc.FINAL_RUN)
    rng = np.random.default_rng(SEED)
    fig = plt.figure(figsize=(st.DOUBLE_MM / st.MM_PER_INCH, 80.0 / st.MM_PER_INCH))
    for k, (dog, letter) in enumerate(zip(st.DOGS, "abc", strict=True)):
        enu_cm, _ = fc.dog_enu_cm(pooled, dog)
        left = 0.005 + 0.325 * k
        ax = fig.add_axes([left, 0.13, 0.33, 0.83], projection="3d")
        cloud = draw_dog(ax, enu_cm, rng)
        sd = enu_cm.std(axis=0)
        rho = np.corrcoef(enu_cm.T)
        fig.text(
            left + 0.02, 0.93, letter, fontsize=st.PANEL_LETTER_PT, fontweight="bold"
        )
        fig.text(left + 0.055, 0.93, st.DOG[dog]["label"])
        fig.text(
            left + 0.055,
            0.875,
            f"$\\sigma_E$ = {sd[0]:.1f},  $\\sigma_N$ = {sd[1]:.1f},  "
            f"$\\sigma_U$ = {sd[2]:.1f} cm",
        )
        print(
            f"DOG {dog}: sd E, N, U = {sd.round(2)} cm; correlations "
            f"EN {rho[0, 1]:+.2f}, EU {rho[0, 2]:+.2f}, NU {rho[1, 2]:+.2f}"
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
