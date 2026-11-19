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
N_SHADOW_POINTS = 20000  # draws projected onto each wall
N_KDE_FIT = 20000  # draws the 3-D KDE is fitted to
SHADOW_ALPHA = 0.022
# Box half-width in posterior sd: wider horizontally so the floor has room for its
# label clear of the 95 % ellipse.
BOX_HALF_WIDTH_SD = (5.5, 5.5, 4.5)
VIEW_ELEVATION_DEG = 30
VIEW_AZIMUTH_DEG = -55
ELLIPSE_PROBABILITIES = (0.68, 0.95)
CLIP_PERCENTILE = 1.0  # colour values below this percentile take the lowest colour
DRAW_PRIOR = True  # the prior's 95 % ellipse on each wall, where it falls in the box
CLOUD_CMAP = "magma_r"  # chosen from the palette comparison (README)
PALETTES = ("viridis_r", "cividis_r", "magma_r")
PANE_COLOR = (0.94, 0.955, 0.975, 1.0)
GRID_COLOR = (0.80, 0.82, 0.86, 1.0)
ELLIPSE_COLOR = "0.12"
SEED = 20261118
# -----------------------------------------------------------------------------------

AXIS_NAMES = "ENU"
WALLS = ((0, 1, 2), (0, 2, 1), (1, 2, 0))  # (i, j, wall axis): floor, back, side


def cloud_cmap(name):
    """Sequential map without its palest end, so low values show on the panes."""
    return ListedColormap(colormaps[name](np.linspace(0.0, 0.88, 256)), name=name)


def kde_log_density_ratio(enu, points, rng):
    """ln(p / p_max) of the 3-D marginal KDE of ``enu`` at ``points``."""
    fit = enu[rng.choice(len(enu), min(N_KDE_FIT, len(enu)), replace=False)]
    kde = gaussian_kde(fit.T)
    log_p = kde.logpdf(points.T)
    peak = max(log_p.max(), kde.logpdf(np.zeros((3, 1)))[0])
    return log_p - peak


def prior_enu(pooled, dog):
    """Prior mean of a DOG (ENU cm from its posterior mean) and its ENU covariance."""
    names = list(pooled["param_names"])
    j = [f"CDOG_aug_DOG{dog}_{axis}" for axis in "xyz"]
    index = [names.index(name) for name in j]
    centre = pooled["prior_center"][index]
    sd = pooled["prior_sd"][index]
    _, mean_ecef = fc.dog_enu_cm(pooled, dog)
    rotation = fc.enu_rotation_at(mean_ecef)
    offset = (pooled["CDOG_reference"] + centre - mean_ecef) @ rotation.T * 100.0
    cov = rotation @ np.diag((sd * 100.0) ** 2) @ rotation.T
    return offset, cov


def prior_sd_cm(pooled, dog):
    """Prior sd (cm) of a DOG position; the prior is isotropic in ECEF."""
    names = list(pooled["param_names"])
    sds = [pooled["prior_sd"][names.index(f"CDOG_aug_DOG{dog}_{a}")] for a in "xyz"]
    if not np.allclose(sds, sds[0]):
        raise ValueError(f"DOG {dog} position prior is not isotropic: {sds} m")
    return 100.0 * sds[0]


def box_limits(sd, equal_scale):
    """Axis limits (cm); with equal_scale one cm scale on all three axes."""
    half = np.array(BOX_HALF_WIDTH_SD) * sd
    if equal_scale:
        half = np.full(3, half.max())
    return [(-h, h) for h in half]


def wall_values(limits):
    """Floor (low U), back wall (high N) and side wall (low E) positions."""
    return {2: limits[2][0], 1: limits[1][1], 0: limits[0][0]}


def draw_panel(ax, all_enu, enu, colour, norm, cmap, limits, prior=None):
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
        ax.scatter(
            *xyz,
            s=0.8,
            c="0.35",
            alpha=SHADOW_ALPHA,
            linewidths=0,
            depthshade=False,
            rasterized=True,
            zorder=1,
        )
        pair = all_enu[:, [i, j]]
        cov = np.cov(pair, rowvar=False)
        for probability, style in zip(ELLIPSE_PROBABILITIES, ("-", "--"), strict=True):
            boundary = fc.ellipse_points(pair.mean(axis=0), cov, probability)
            line = [None, None, None]
            line[i], line[j] = boundary[:, 0], boundary[:, 1]
            line[wall] = np.full(len(boundary), walls[wall])
            ax.plot(*line, color=ELLIPSE_COLOR, lw=0.7, ls=style, zorder=3)
        if prior is not None:
            draw_prior_ellipse(ax, prior, i, j, wall, walls[wall], limits)
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
    style_axes(ax)
    place_rho_labels(ax, rhos, limits)
    return cloud


def draw_prior_ellipse(ax, prior, i, j, wall, wall_value, limits):
    """Dotted 95 % prior ellipse on a wall, only the part inside the box."""
    offset, cov = prior
    boundary = fc.ellipse_points(offset[[i, j]], cov[np.ix_([i, j], [i, j])], 0.95, 720)
    inside = (
        (boundary[:, 0] >= limits[i][0])
        & (boundary[:, 0] <= limits[i][1])
        & (boundary[:, 1] >= limits[j][0])
        & (boundary[:, 1] <= limits[j][1])
    )
    if not inside.any():
        return
    line = [None, None, None]
    line[i] = np.where(inside, boundary[:, 0], np.nan)
    line[j] = np.where(inside, boundary[:, 1], np.nan)
    line[wall] = np.full(len(boundary), wall_value)
    ax.plot(*line, color=ELLIPSE_COLOR, lw=0.7, ls=":", zorder=3)


def style_axes(ax):
    ax.view_init(elev=VIEW_ELEVATION_DEG, azim=VIEW_AZIMUTH_DEG)
    ax.set_xlabel("ΔE (cm)", labelpad=-3)
    ax.set_ylabel("ΔN (cm)", labelpad=0)
    ax.set_zlabel("ΔU (cm)", labelpad=-4)
    ax.tick_params(pad=-3.5, labelsize=paper_style.FONT_SIZE_PT)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(PANE_COLOR)
        axis._axinfo["grid"].update(color=GRID_COLOR, linewidth=0.4)
        axis.set_major_locator(plt.MaxNLocator(4, symmetric=True))
        axis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:g}".replace("-", fc.MINUS))
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
        rounded = round(rho, 2)
        if rounded > 0:
            colour = fc.POSITIVE_COLOR
        elif rounded < 0:
            colour = fc.NEGATIVE_COLOR
        else:
            colour, rounded = "0.25", 0.0
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


def sd_text(enu, prior_sd_cm):
    """Posterior sd per axis and the prior sd (isotropic) of the position, cm."""
    posterior = ", ".join(
        f"σ$_{n}$ {s:.1f}" for n, s in zip(AXIS_NAMES, enu.std(axis=0), strict=True)
    )
    return f"{posterior} cm (prior {prior_sd_cm:.0f})"


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


def row_figure(panels, width_mm, height_mm, bar_label, shared_bar=True):
    """One row of 3-D panels.

    Each panel is a dict with enu (shown draws), shadow_enu (all draws), values,
    norm, cmap, title, limits, prior and equal_scale. With shared_bar one colourbar
    sits under the row; otherwise each panel gets its own, labelled by bar_label
    (a string or a list per panel).
    """
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    n = len(panels)
    step = 1.0 / n
    bottom = 0.22 if shared_bar else 0.26
    for k, panel in enumerate(panels):
        ax = fig.add_axes(
            [k * step, bottom, step - 0.03, 0.89 - bottom],
            projection="3d",
            computed_zorder=False,
        )
        limits = panel["limits"]
        aspect = [h for _, h in limits] if panel["equal_scale"] else (1, 1, 1)
        ax.set_box_aspect(aspect, zoom=0.95)
        cloud = draw_panel(
            ax,
            panel["shadow_enu"],
            panel["enu"],
            panel["values"],
            panel["norm"],
            panel["cmap"],
            limits,
            panel["prior"],
        )
        centre = (k + 0.5) * step
        title = fig.text(
            centre, 0.985, panel["title"], ha="center", va="top", weight="bold"
        )
        if panel.get("dog") is not None:
            # DOG marker in its colour, left of the title (one DOG style everywhere).
            style = paper_style.DOG_STYLE[panel["dog"]]
            fig.canvas.draw()
            box = title.get_window_extent().transformed(fig.transFigure.inverted())
            fig.add_artist(
                plt.Line2D(
                    [box.x0 - 0.012],
                    [0.5 * (box.y0 + box.y1)],
                    marker=style["marker"],
                    color=style["color"],
                    markeredgecolor="0.1",
                    markeredgewidth=0.5,
                    markersize=5,
                    transform=fig.transFigure,
                )
            )
        fig.text(centre, 0.93, panel["subtitle"], ha="center", va="top")
        if not shared_bar:
            cax = fig.add_axes([centre - 0.3 * step, 0.12, 0.6 * step, 0.026])
            bar = fig.colorbar(cloud, cax=cax, orientation="horizontal", extend="min")
            bar.set_label(bar_label[k])
            bar.ax.xaxis.set_major_formatter(minus_formatter())
            bar.ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    if shared_bar:
        cax = fig.add_axes([0.30, 0.135, 0.40, 0.026])
        bar = fig.colorbar(cloud, cax=cax, orientation="horizontal", extend="min")
        bar.set_label(bar_label)
        bar.ax.xaxis.set_major_formatter(minus_formatter())
    return fig


def dog_panel(pooled, dog, enu, shown, values, norm, cmap_name, equal_scale=False):
    sd = enu.std(axis=0)
    return {
        "enu": enu[shown],
        "shadow_enu": enu,
        "values": values,
        "norm": norm,
        "cmap": cloud_cmap(cmap_name),
        "title": paper_style.DOG_STYLE[dog]["label"],
        "dog": dog,
        "subtitle": sd_text(enu, prior_sd_cm(pooled, dog)),
        "limits": box_limits(sd, equal_scale),
        "prior": prior_enu(pooled, dog) if DRAW_PRIOR else None,
        "equal_scale": equal_scale,
    }


def report_prior(pooled, enus):
    """Print where each DOG's prior 95 % region lies relative to the box."""
    for dog, enu in enus.items():
        offset, cov = prior_enu(pooled, dog)
        radius = np.sqrt(fc.chi2_radius_squared(0.95, 2) * np.diag(cov))
        print(
            f"DOG {dog}: prior centre {np.round(offset, 1)} cm from posterior mean, "
            f"95 % half-widths {np.round(radius, 0)} cm, posterior sd "
            f"{np.round(enu.std(axis=0), 1)} cm"
        )


if __name__ == "__main__":
    paper_style.apply()
    plt.rcParams["figure.constrained_layout.use"] = False
    pooled = fc.pooled_run(RUN)
    enus = {dog: fc.dog_enu_cm(pooled, dog)[0] for dog in DOGS}
    report_prior(pooled, enus)
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

    # Main figure and its equal-scale supplement variant.
    for equal_scale, name in ((False, ""), (True, "_equal_scale")):
        panels = [
            dog_panel(
                pooled,
                dog,
                enus[dog],
                shown[dog],
                kde[dog],
                kde_norm,
                CLOUD_CMAP,
                equal_scale,
            )
            for dog in DOGS
        ]
        fig = row_figure(panels, double, 88.0, ENCODING_LABEL["kde"])
        fc.save(fig, f"fig_3d_posterior_three_dogs{name}")
        plt.close(fig)

    # Encoding comparison (DOG 3): (a) joint log posterior, (b) marginal density.
    joint_norm = shared_norm([joint[3]])
    panels = [
        dog_panel(pooled, 3, enus[3], shown[3], joint[3], joint_norm, CLOUD_CMAP),
        dog_panel(pooled, 3, enus[3], shown[3], kde[3], kde_norm, CLOUD_CMAP),
    ]
    panels[0]["title"] = "DOG 3: (a) joint log posterior"
    panels[1]["title"] = "DOG 3: (b) marginal density"
    labels = [ENCODING_LABEL["logpost"], ENCODING_LABEL["kde_short"]]
    fig = row_figure(panels, double, 90.0, labels, shared_bar=False)
    fc.save(fig, "fig_3d_posterior_encodings")
    plt.close(fig)

    # Palette comparison (DOG 3, PNG only).
    panels = [
        dog_panel(pooled, 3, enus[3], shown[3], kde[3], kde_norm, name)
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
            pooled, dog, enus[dog], shown[dog], kde[dog], kde_norm, CLOUD_CMAP
        )
        fig = row_figure([panel], SINGLE_DOG_WIDTH_MM, 105.0, ENCODING_LABEL["kde"])
        fc.save(fig, f"fig_3d_posterior_DOG{dog}", (SINGLE_DOG_WIDTH_MM,))
        plt.close(fig)
