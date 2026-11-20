"""3-D posterior cloud of the DOG positions in local ENU, with shadows on the walls.

The MCMC draws of each DOG position are rotated into local ENU at that DOG's
posterior mean and shown relative to it (cm). The cloud is coloured by the 3-D
marginal density of (E, N, U) (Gaussian KDE, ln of density over its peak) or, for
comparison, by the joint log posterior of all parameters. Each wall carries the
same draws projected onto it as grey points (darker where denser), with the 68 %
(solid) and 95 % (dashed) Gaussian covariance ellipses and the correlation ρ,
coloured by sign (vermillion positive, blue negative).

Outputs: the three-DOG figure (main, 174 mm) and its equal-scale variant
(supplement), one 120 mm figure per DOG, the encoding comparison (KDE vs joint log
posterior) and the palette comparison (PNG only).

Reviewer links: R3 #11 and R4 m2 (units and reference on every axis), R4 2, 2b and
2c (correlations, data vs prior). Run from the repo root with PYTHONPATH=src.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, Normalize
from mpl_toolkits.mplot3d import proj3d
from scipy.stats import gaussian_kde

import _figure_common as fc
from plotting import paper_style

# --- configuration -----------------------------------------------------------------
RUN = fc.FINAL_RUN
DOGS = (1, 3, 4)
SINGLE_DOG_WIDTH_MM = 120.0  # approved: per-DOG figures for the supplement/letter
N_CLOUD_POINTS = 5000  # draws shown in each 3-D cloud
N_SHADOW_POINTS = 6000  # draws projected onto each wall
N_KDE_FIT = 20000  # draws the 3-D KDE is fitted to
SHADOW_ALPHA = 0.55
SHADOW_CMAP = "GnBu"  # shadows coloured by 2-D density, darker = denser
N_SHADOW_KDE_FIT = 3000  # draws each 2-D shadow KDE is fitted to
# One cubic box (same cm scale on E, N and U) for every DOG, so the panels compare
# directly: half-width = this many posterior sd of the widest axis of any DOG,
# rounded up to BOX_ROUND_CM.
BOX_HALF_WIDTH_SD = 3.5
BOX_ROUND_CM = 10.0
RHO_FULL_COLOUR = 0.2  # |ρ| at which the correlation colour is fully saturated
RHO_NEGATIVE_RGB = (0.78, 0.08, 0.10)  # red
RHO_POSITIVE_RGB = (0.00, 0.28, 0.78)  # blue
RHO_ZERO_RGB = (0.25, 0.25, 0.25)
VIEW_ELEVATION_DEG = 30
VIEW_AZIMUTH_DEG = -55
ELLIPSE_PROBABILITY = 0.95
CLIP_PERCENTILE = 1.0  # colour values below this percentile take the lowest colour
CLOUD_CMAP = "magma"  # chosen from the palette comparison (README)
PALETTES = ("viridis", "cividis", "magma")
PANE_COLOR = (0.94, 0.955, 0.975, 1.0)
GRID_COLOR = (0.80, 0.82, 0.86, 1.0)
ELLIPSE_COLOR = "0.12"
SEED = 20261118
# -----------------------------------------------------------------------------------

AXIS_NAMES = "ENU"
WALLS = ((0, 1, 2), (0, 2, 1), (1, 2, 0))  # (i, j, wall axis): floor, back, side


def shadow_cmap():
    """Sequential map for the wall shadows, without its palest end."""
    return ListedColormap(colormaps[SHADOW_CMAP](np.linspace(0.4, 1.0, 256)))


def cloud_cmap(name):
    """Dark (low density) to bright (peak), without the near-black end."""
    return ListedColormap(colormaps[name](np.linspace(0.12, 1.0, 256)), name=name)


def kde_log_density_ratio(enu, points, rng):
    """ln(p / p_max) of the 3-D marginal KDE of ``enu`` at ``points``."""
    fit = enu[rng.choice(len(enu), min(N_KDE_FIT, len(enu)), replace=False)]
    kde = gaussian_kde(fit.T)
    log_p = kde.logpdf(points.T)
    peak = max(log_p.max(), kde.logpdf(np.zeros((3, 1)))[0])
    return log_p - peak


def prior_sd_cm(pooled, dog):
    """Prior sd (cm) of a DOG position; the prior is isotropic in ECEF."""
    names = list(pooled["param_names"])
    sds = [pooled["prior_sd"][names.index(f"CDOG_aug_DOG{dog}_{a}")] for a in "xyz"]
    if not np.allclose(sds, sds[0]):
        raise ValueError(f"DOG {dog} position prior is not isotropic: {sds} m")
    return 100.0 * sds[0]


def box_limits(sds):
    """Common cubic axis limits (cm) for all DOGs; sds: posterior sd arrays (cm)."""
    half = BOX_HALF_WIDTH_SD * max(sd.max() for sd in sds)
    half = np.ceil(half / BOX_ROUND_CM) * BOX_ROUND_CM
    return [(-half, half)] * 3


def wall_values(limits):
    """Floor (low U), back wall (high N) and side wall (low E) positions."""
    return {2: limits[2][0], 1: limits[1][1], 0: limits[0][0]}


def draw_panel(ax, all_enu, enu, colour, norm, cmap, limits):
    """Cloud, shadows, ellipses and ρ labels of one DOG; returns the cloud artist.

    Shadows, ellipses and ρ use all draws (all_enu); the cloud shows enu, coloured
    by colour.
    """
    rng = np.random.default_rng(SEED)
    walls = wall_values(limits)
    pick = rng.choice(len(all_enu), min(N_SHADOW_POINTS, len(all_enu)), replace=False)
    shadow = all_enu[pick]
    rhos = {}
    for i, j, wall in WALLS:
        xyz = [None, None, None]
        xyz[i], xyz[j] = shadow[:, i], shadow[:, j]
        xyz[wall] = np.full(len(shadow), walls[wall])
        fit = all_enu[rng.choice(len(all_enu), N_SHADOW_KDE_FIT, replace=False)]
        density = gaussian_kde(fit[:, [i, j]].T)(shadow[:, [i, j]].T)
        order = np.argsort(density)  # densest drawn last
        xyz = [None if v is None else v[order] for v in xyz]
        ax.scatter(
            *xyz,
            s=1.6,
            c=density[order],
            cmap=shadow_cmap(),
            alpha=SHADOW_ALPHA,
            linewidths=0,
            depthshade=False,
            rasterized=True,
            zorder=1,
        )
        pair = all_enu[:, [i, j]]
        cov = np.cov(pair, rowvar=False)
        boundary = fc.ellipse_points(pair.mean(axis=0), cov, ELLIPSE_PROBABILITY)
        line = [None, None, None]
        line[i], line[j] = boundary[:, 0], boundary[:, 1]
        line[wall] = np.full(len(boundary), walls[wall])
        ax.plot(*line, color=ELLIPSE_COLOR, lw=0.8, zorder=3)
        rhos[(i, j)] = fc.correlation(all_enu[:, i], all_enu[:, j])

    order = np.argsort(colour)  # high values (dense core) drawn last, on top
    cloud = ax.scatter(
        *enu[order].T,
        c=colour[order],
        cmap=cmap,
        norm=norm,
        s=1.2,
        alpha=0.85,
        linewidths=0,
        depthshade=False,
        rasterized=True,
        zorder=5,
    )
    for axis, (low, high) in zip("xyz", limits, strict=True):
        getattr(ax, f"set_{axis}lim")(low, high)
    style_axes(ax, limits)
    place_rho_labels(ax, rhos, limits)
    return cloud


def style_axes(ax, limits):
    ax.view_init(elev=VIEW_ELEVATION_DEG, azim=VIEW_AZIMUTH_DEG)
    ax.set_xlabel("ΔE (cm)", labelpad=-3)
    ax.set_ylabel("ΔN (cm)", labelpad=0)
    ax.set_zlabel("ΔU (cm)", labelpad=-4)
    ax.tick_params(pad=-3.5, labelsize=paper_style.FONT_SIZE_PT)
    tick = np.floor(0.75 * limits[0][1] / 10.0) * 10.0  # clear of the box corners
    ticks = [-tick, 0.0, tick]
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(PANE_COLOR)
        axis._axinfo["grid"].update(color=GRID_COLOR, linewidth=0.4)
        axis.set_ticks(ticks)
        axis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:g}".replace("-", fc.MINUS))
        )


def rho_colour(rho):
    """Grey at zero, increasingly red for negative and blue for positive ρ."""
    weight = min(abs(rho) / RHO_FULL_COLOUR, 1.0)
    target = RHO_POSITIVE_RGB if rho > 0 else RHO_NEGATIVE_RGB
    return tuple(
        (1 - weight) * zero + weight * end
        for zero, end in zip(RHO_ZERO_RGB, target, strict=True)
    )


def place_rho_labels(ax, rhos, limits):
    """Upright ρ text at each wall's corner farthest from the cloud (projected)."""
    walls = wall_values(limits)
    # Corner of each wall away from the ellipses: (i, j) fractions of the half-width.
    # Floor: in its open front part (high E, low N), between ellipse and edge.
    # Back and side walls: their upper outer corners, text above the box edge.
    corners = {(0, 1): (0.40, -0.64), (0, 2): (1.0, 1.0), (1, 2): (-1.0, 1.0)}
    alignment = {
        (0, 1): ("center", "center"),
        (0, 2): ("right", "bottom"),
        (1, 2): ("left", "bottom"),
    }
    projection = ax.get_proj()
    for (i, j), rho in rhos.items():
        wall = 3 - i - j
        point = [0.0, 0.0, 0.0]
        point[i] = corners[(i, j)][0] * limits[i][1]
        point[j] = corners[(i, j)][1] * limits[j][1]
        point[wall] = walls[wall]
        x2, y2, _ = proj3d.proj_transform(*point, projection)
        ha, va = alignment[(i, j)]
        rounded = round(rho, 2) + 0.0  # no negative zero
        colour = rho_colour(rounded)
        value = fc.signed(rounded) if rounded else "0.00"
        ax.annotate(
            f"ρ$_{{{AXIS_NAMES[i]}{AXIS_NAMES[j]}}}$ = {value}",
            xy=(x2, y2),
            xycoords="data",
            xytext=(
                {"right": -3, "left": 3, "center": 0}[ha],
                2 if va == "bottom" else 0,
            ),
            textcoords="offset points",
            ha=ha,
            va=va,
            color=colour,
            fontsize=paper_style.FONT_SIZE_PT,
            # Bold only for a correlation that matters (|ρ| ≥ 0.3).
            fontweight="bold" if abs(rounded) >= 0.3 else "normal",
            zorder=10,
        )


def sd_text(enu):
    """Posterior sd per axis, cm."""
    posterior = ", ".join(
        f"σ$_{n}$ {s:.1f}" for n, s in zip(AXIS_NAMES, enu.std(axis=0), strict=True)
    )
    return f"{posterior} cm"


def shared_norm(values):
    """Colour range from the CLIP_PERCENTILE-th percentile up to 0 (the peak)."""
    low = np.percentile(np.concatenate(values), CLIP_PERCENTILE)
    return Normalize(vmin=low, vmax=0.0, clip=True)


ENCODING_LABEL = {
    "kde": "ln(density / peak) of the 3-D marginal posterior",
    "kde_short": "ln(density / peak)",
    "logpost": "Joint log posterior − max",
}


def minus_formatter():
    return plt.FuncFormatter(lambda v, _: f"{v:g}".replace("-", fc.MINUS))


def row_figure(panels, width_mm, height_mm, bar_label, shared_bar=True, prior_cm=None):
    """One row of 3-D panels in cubic boxes.

    Each panel is a dict with enu (shown draws), shadow_enu (all draws), values,
    norm, cmap, title, subtitle and limits. With shared_bar one colourbar sits under
    the row, with a legend for the ellipses on its left and, if prior_cm is given,
    the position prior (cm, the same for every DOG) on its right; otherwise each panel
    gets its own colourbar, labelled by bar_label (a string or a list per panel).
    """
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    n = len(panels)
    step = 1.0 / n
    stacked = n == 1  # one panel: legend and prior go below the colourbar
    bar_y = 0.145 if stacked else 0.135
    bottom = 0.25 if stacked else (0.22 if shared_bar else 0.26)
    top = 0.985 if stacked else 0.95  # top of the 3-D axes
    title_y, subtitle_y = (0.995, 0.945) if stacked else (0.985, 0.93)
    for k, panel in enumerate(panels):
        ax = fig.add_axes(
            [k * step - 0.06, bottom, step + 0.04, top - bottom]
            if stacked
            else [k * step - 0.005, bottom, step - 0.03, top - bottom],
            projection="3d",
            computed_zorder=False,
        )
        ax.patch.set_alpha(0.0)  # neighbouring panels must not clip each other
        ax.set_box_aspect((1, 1, 1), zoom=0.95)
        cloud = draw_panel(
            ax,
            panel["shadow_enu"],
            panel["enu"],
            panel["values"],
            panel["norm"],
            panel["cmap"],
            panel["limits"],
        )
        centre = (k + 0.5) * step
        fig.text(centre, title_y, panel["title"], ha="center", va="top", weight="bold")
        fig.text(centre, subtitle_y, panel["subtitle"], ha="center", va="top")
        if not shared_bar:
            cax = fig.add_axes([centre - 0.3 * step, 0.12, 0.6 * step, 0.026])
            bar = fig.colorbar(cloud, cax=cax, orientation="horizontal", extend="min")
            bar.set_label(bar_label[k])
            bar.ax.xaxis.set_major_formatter(minus_formatter())
            bar.ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    if shared_bar:
        cax = (
            fig.add_axes([0.2, bar_y, 0.6, 0.024])
            if stacked
            else fig.add_axes([0.36, bar_y, 0.34, 0.026])
        )
        bar = fig.colorbar(cloud, cax=cax, orientation="horizontal", extend="min")
        bar.set_label(bar_label)
        bar.ax.xaxis.set_major_formatter(minus_formatter())
        handles = [plt.Line2D([], [], color=ELLIPSE_COLOR, lw=0.8)]
        fig.legend(
            handles,
            ["95 % ellipse"],
            loc="lower left",
            bbox_to_anchor=(0.0, 0.0 if stacked else 0.03),
            frameon=False,
            handlelength=2.2,
            borderaxespad=0.2,
        )
        if prior_cm is not None:
            fig.text(
                0.995,
                0.03 if stacked else 0.06,
                f"Prior on each position:{' ' if stacked else chr(10)}"
                f"σ = {prior_cm / 100:g} m ({prior_cm:g} cm) per axis",
                ha="right",
                va="center",
            )
    return fig


def dog_panel(dog, enu, shown, values, norm, cmap_name, limits):
    return {
        "enu": enu[shown],
        "shadow_enu": enu,
        "values": values,
        "norm": norm,
        "cmap": cloud_cmap(cmap_name),
        "title": paper_style.DOG_STYLE[dog]["label"],
        "subtitle": sd_text(enu),
        "limits": limits,
    }


if __name__ == "__main__":
    paper_style.apply()
    plt.rcParams["figure.constrained_layout.use"] = False
    pooled = fc.pooled_run(RUN)
    enus = {dog: fc.dog_enu_cm(pooled, dog)[0] for dog in DOGS}
    prior_cm = {prior_sd_cm(pooled, dog) for dog in DOGS}
    if len(prior_cm) != 1:
        raise ValueError(f"DOG position priors differ (cm): {sorted(prior_cm)}")
    prior_cm = prior_cm.pop()
    limits = box_limits([enu.std(axis=0) for enu in enus.values()])
    rng = np.random.default_rng(SEED)
    shown = {
        dog: rng.choice(len(enu), N_CLOUD_POINTS, replace=False)
        for dog, enu in enus.items()
    }
    kde = {
        dog: kde_log_density_ratio(enus[dog], enus[dog][shown[dog]], rng)
        for dog in DOGS
    }
    logpost = pooled["logpost"] - pooled["logpost"].max()
    joint = {dog: logpost[shown[dog]] for dog in DOGS}
    kde_norm = shared_norm(list(kde.values()))
    double = paper_style.DOUBLE_COLUMN_MM

    # Main figure: all DOGs in the same cubic box.
    panels = [
        dog_panel(dog, enus[dog], shown[dog], kde[dog], kde_norm, CLOUD_CMAP, limits)
        for dog in DOGS
    ]
    fig = row_figure(panels, double, 82.0, ENCODING_LABEL["kde"], prior_cm=prior_cm)
    fc.save(fig, "fig_3d_posterior_three_dogs")
    plt.close(fig)

    # Encoding comparison (DOG 3): (a) joint log posterior, (b) marginal density.
    joint_norm = shared_norm([joint[3]])
    panels = [
        dog_panel(3, enus[3], shown[3], joint[3], joint_norm, CLOUD_CMAP, limits),
        dog_panel(3, enus[3], shown[3], kde[3], kde_norm, CLOUD_CMAP, limits),
    ]
    panels[0]["title"] = "DOG 3: (a) joint log posterior"
    panels[1]["title"] = "DOG 3: (b) marginal density"
    labels = [ENCODING_LABEL["logpost"], ENCODING_LABEL["kde_short"]]
    fig = row_figure(panels, double, 90.0, labels, shared_bar=False)
    fc.save(fig, "fig_3d_posterior_encodings")
    plt.close(fig)

    # Palette comparison (DOG 3, PNG only).
    panels = [
        dog_panel(3, enus[3], shown[3], kde[3], kde_norm, name, limits)
        for name in PALETTES
    ]
    for panel, name in zip(panels, PALETTES, strict=True):
        panel["title"] = f"DOG 3: {name}"
    fig = row_figure(
        panels, double, 90.0, [ENCODING_LABEL["kde_short"]] * 3, shared_bar=False
    )
    fc.save(fig, "fig_3d_posterior_palettes")
    plt.close(fig)

    # One figure per DOG (120 mm, supplement and response letter).
    for dog in DOGS:
        panel = dog_panel(
            dog, enus[dog], shown[dog], kde[dog], kde_norm, CLOUD_CMAP, limits
        )
        fig = row_figure(
            [panel],
            SINGLE_DOG_WIDTH_MM,
            98.0,
            ENCODING_LABEL["kde"],
            prior_cm=prior_cm,
        )
        fc.save(fig, f"fig_3d_posterior_DOG{dog}", (SINGLE_DOG_WIDTH_MM,))
        plt.close(fig)
