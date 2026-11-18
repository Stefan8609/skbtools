"""3-D posterior cloud of one DOG's ENU position with its 2-D marginals as shadows.

Prototype for the revision (plot_tests, not yet a paper figure). The MCMC draws of
the DOG position are rotated into local ENU at the DOG's posterior mean and shown
relative to that mean (cm). Each pair of coordinates is projected onto a wall of the
box: E-N on the floor, E-U on the back wall, N-U on the side wall. Every shadow
shows the smoothed 2-D sample density, the 68 % and 95 % covariance ellipses
(Gaussian, 2 degrees of freedom) and the correlation coefficient.

Reviewer links: R3 #11 and R4 m2 (units and reference on every axis), R4 2 and
R4 2c (correlations between the position components).
"""

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from plotting import paper_style

# --- configuration -----------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPO / "Bermuda_Data" / "Output" / "MCMC_Chains" / "ar1"
DOG = 3
DOGS_IN_RUN = [1, 3, 4]  # order of the DOG axis of CDOG_aug in the chain files
N_CLOUD_POINTS = 6000  # draws shown in the 3-D cloud (all draws feed the shadows)
SHADOW_BINS = 90
SHADOW_SMOOTH_BINS = 1.6
BOX_HALF_WIDTH_SD = 4.2  # box half-width in posterior sd, per axis
EQUAL_SCALE = False  # True: one cm scale on all three axes (U then dominates)
VIEW_ELEVATION_DEG = 22
VIEW_AZIMUTH_DEG = -52
ELLIPSE_PROBABILITIES = (0.68, 0.95)
SEED = 20261118
OUTPUT_DIR = paper_style.FIGURE_DIR.parent / "plot_tests"  # gitignored
# -----------------------------------------------------------------------------

WGS84_A_M = 6378137.0
WGS84_E2 = 6.69437999014e-3
CHI2_2DOF = {p: -2.0 * np.log(1.0 - p) for p in ELLIPSE_PROBABILITIES}


def ecef_to_lat_lon_deg(xyz_m):
    """Geodetic latitude and longitude (WGS84) of one ECEF point."""
    x, y, z = xyz_m
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(6):
        n = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
        height = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + height)))
    return np.rad2deg(lat), np.rad2deg(lon)


def enu_rotation(lat_deg, lon_deg):
    """Rows E, N, U (3, 3) rotating ECEF differences into local ENU."""
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    return np.array(
        [
            [-np.sin(lon), np.cos(lon), 0.0],
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        ]
    )


def load_enu_draws_cm(run_dir, dog):
    """Post-burn-in draws of one DOG, ENU relative to the posterior mean (n, 3) cm."""
    paths = sorted(glob.glob(str(run_dir / "chain_*.npz")))
    if not paths:
        raise FileNotFoundError(f"No chain_*.npz files in {run_dir}")
    dog_index = DOGS_IN_RUN.index(dog)
    offsets, reference = [], None
    for path in paths:
        with np.load(path, allow_pickle=True) as chain:
            offsets.append(chain["CDOG_aug"][:, dog_index, :])
            reference = chain["CDOG_reference"]
    offsets = np.concatenate(offsets)
    mean_ecef = reference + offsets.mean(axis=0)
    rotation = enu_rotation(*ecef_to_lat_lon_deg(mean_ecef))
    enu_cm = (offsets - offsets.mean(axis=0)) @ rotation.T * 100.0
    return enu_cm, len(paths)


def ellipse_points(mean, cov, probability, n=200):
    """Boundary (n, 2) of the Gaussian region holding ``probability`` in 2-D."""
    values, vectors = np.linalg.eigh(cov)
    angle = np.linspace(0.0, 2.0 * np.pi, n)
    circle = np.stack([np.cos(angle), np.sin(angle)])
    scale = np.sqrt(CHI2_2DOF[probability] * values)
    return (vectors @ (scale[:, None] * circle)).T + mean


def smoothed_density(a, b, a_limits, b_limits):
    """Histogram density of (a, b) on a grid, smoothed and scaled to peak 1."""
    counts, a_edges, b_edges = np.histogram2d(
        a, b, bins=SHADOW_BINS, range=[a_limits, b_limits]
    )
    density = gaussian_filter(counts, SHADOW_SMOOTH_BINS)
    a_centres = 0.5 * (a_edges[1:] + a_edges[:-1])
    b_centres = 0.5 * (b_edges[1:] + b_edges[:-1])
    grid_a, grid_b = np.meshgrid(a_centres, b_centres, indexing="ij")
    return grid_a, grid_b, density / density.max()


def draw_shadow(ax, draws, pair, wall_axis, wall_value, limits, shadow_cmap, color):
    """Density, ellipses and correlation of draws[:, pair] on one wall of the box."""
    i, j = pair
    grid_i, grid_j, density = smoothed_density(
        draws[:, i], draws[:, j], limits[i], limits[j]
    )
    levels = np.linspace(0.03, 1.0, 16)
    # contourf with zdir/offset takes its arguments in the wall's own (X, Y, Z) slots.
    if wall_axis == 2:
        ax.contourf(
            grid_i,
            grid_j,
            density,
            levels=levels,
            zdir="z",
            offset=wall_value,
            cmap=shadow_cmap,
            antialiased=True,
        )
    elif wall_axis == 1:
        ax.contourf(
            grid_i,
            density,
            grid_j,
            levels=levels,
            zdir="y",
            offset=wall_value,
            cmap=shadow_cmap,
            antialiased=True,
        )
    else:
        ax.contourf(
            density,
            grid_i,
            grid_j,
            levels=levels,
            zdir="x",
            offset=wall_value,
            cmap=shadow_cmap,
            antialiased=True,
        )

    pair_draws = draws[:, [i, j]]
    mean = pair_draws.mean(axis=0)
    cov = np.cov(pair_draws, rowvar=False)
    for probability, style in zip(ELLIPSE_PROBABILITIES, ["-", "--"], strict=True):
        boundary = ellipse_points(mean, cov, probability)
        line = [None, None, None]
        line[i], line[j] = boundary[:, 0], boundary[:, 1]
        line[wall_axis] = np.full(len(boundary), wall_value)
        ax.plot(*line, color=color, lw=0.9, ls=style, zorder=10)

    rho = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    names = "ENU"
    label = f"ρ$_{{{names[i]}{names[j]}}}$ = {rho:+.2f}".replace("-", "\u2212")
    anchor = [0.0, 0.0, 0.0]
    # Upper-left corner of each wall; on the floor, the front-left corner.
    anchor[i] = limits[i][0] + 0.05 * np.ptp(limits[i])
    if wall_axis == 2:
        anchor[j] = limits[j][0] + 0.06 * np.ptp(limits[j])
    else:
        anchor[j] = limits[j][1] - 0.07 * np.ptp(limits[j])
    anchor[wall_axis] = wall_value
    text_direction = {0: "y", 1: "x", 2: "x"}[wall_axis]
    ax.text(
        *anchor,
        label,
        zdir=text_direction,
        color="0.15",
        fontsize=paper_style.FONT_SIZE_PT,
        ha="left",
        va="center",
    )
    return rho


def make_figure(enu_cm, n_chains):
    """Build the 3-D shadow figure; return the figure."""
    rng = np.random.default_rng(SEED)
    sd = enu_cm.std(axis=0)
    if EQUAL_SCALE:
        half = np.full(3, BOX_HALF_WIDTH_SD * sd.max())
    else:
        half = BOX_HALF_WIDTH_SD * sd
    limits = [(-h, h) for h in half]
    floor, back_wall, side_wall = limits[2][0], limits[1][1], limits[0][0]

    fig = plt.figure(
        figsize=(
            paper_style.DOUBLE_COLUMN_MM / 25.4 * 0.62,
            paper_style.DOUBLE_COLUMN_MM / 25.4 * 0.52,
        )
    )
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    ax.set_box_aspect(half / half.max() if EQUAL_SCALE else (1, 1, 1), zoom=0.86)

    color = paper_style.DOG_STYLE[DOG]["color"]
    shadow_cmap = LinearSegmentedColormap.from_list(
        "shadow", ["#ffffff00", "#9aa3ad", "#3b4550"]
    )
    draw_shadow(ax, enu_cm, (0, 1), 2, floor, limits, shadow_cmap, "0.1")
    draw_shadow(ax, enu_cm, (0, 2), 1, back_wall, limits, shadow_cmap, "0.1")
    draw_shadow(ax, enu_cm, (1, 2), 0, side_wall, limits, shadow_cmap, "0.1")

    shown = enu_cm[rng.choice(len(enu_cm), N_CLOUD_POINTS, replace=False)]
    whitened = (shown - shown.mean(axis=0)) / sd
    radius = np.sqrt((whitened**2).sum(axis=1))
    order = np.argsort(-radius)  # dense core drawn last, on top
    cloud_cmap = LinearSegmentedColormap.from_list(
        "cloud", ["#d6ecf8", color, "#0072B2", "#08306b"]
    )
    ax.scatter(
        *shown[order].T,
        c=-radius[order],
        cmap=cloud_cmap,
        s=1.6,
        vmin=-3.5,
        vmax=0.0,
        alpha=0.7,
        linewidths=0,
        depthshade=False,
        rasterized=True,
        zorder=20,
    )

    for axis, (low, high) in zip("xyz", limits, strict=True):
        getattr(ax, f"set_{axis}lim")(low, high)
    ax.set_xlabel("ΔE (cm)", labelpad=-6)
    ax.set_ylabel("ΔN (cm)", labelpad=-6)
    ax.set_zlabel("ΔU (cm)", labelpad=-7)
    ax.tick_params(pad=-3, labelsize=paper_style.FONT_SIZE_PT)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.97, 0.97, 0.975, 1.0))
        axis._axinfo["grid"].update(color=(0.82, 0.82, 0.84, 1.0), linewidth=0.4)
        axis.set_major_locator(plt.MaxNLocator(5, integer=True, symmetric=True))
    ax.view_init(elev=VIEW_ELEVATION_DEG, azim=VIEW_AZIMUTH_DEG)

    # Which DOG, run and sample size go in the caption; the panel keeps the sd.
    sd_text = "\n".join(f"σ$_{n}$ = {s:.1f} cm" for n, s in zip("ENU", sd, strict=True))
    fig.text(
        0.03,
        0.96,
        f"{paper_style.DOG_STYLE[DOG]['label']}\n{sd_text}",
        ha="left",
        va="top",
        fontsize=paper_style.FONT_SIZE_PT,
        linespacing=1.3,
    )
    print(f"{len(enu_cm):,} draws from {n_chains} chains")
    return fig


if __name__ == "__main__":
    paper_style.apply()
    plt.rcParams["figure.constrained_layout.use"] = False
    enu_cm, n_chains = load_enu_draws_cm(RUN_DIR, DOG)
    figure = make_figure(enu_cm, n_chains)
    figure.subplots_adjust(left=0.0, right=0.97, bottom=-0.03, top=0.99)
    print(paper_style.save(figure, f"fig_mcmc_enu_3d_shadows_DOG{DOG}", OUTPUT_DIR))
