from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from plotting.Ellipses.Error_Ellipse import compute_error_ellipse
from plotting.Ellipses.Prior_Ellipse import plot_prior_ellipse
from plotting.MCMC_plots import get_init_params_and_prior
from plotting.Misc.KDE_MCMC_plot import plot_kde_mcmc
from plotting.save import save_plot

from geometry.rigid_body import findRotationAndDisplacement
from data import gps_output_path, gps_data_path

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "font.size": 16,  # change this freely
        "mathtext.fontset": "cm",
        "text.latex.preamble": r"\usepackage[utf8]{inputenc}"
        "\n"
        r"\usepackage{textcomp, amsmath}",
    }
)


def default_rv_plot_data_path():
    """Return the default NPZ cache path for this figure."""
    plot_data_dir = gps_output_path("Plot_Data")
    return f"{plot_data_dir}/Atlantic_RV_Plot_Data.npz"


def _compute_contour_kde2d(pts2d, levels=20, gridsize=200):
    """Evaluate the expensive 2D KDE contour grid once.

    The returned arrays are all plain NumPy arrays, so they can be stored in
    the NPZ cache and redrawn later with ``ax.contourf`` without re-running
    ``scipy.stats.gaussian_kde``.
    """
    pts2d = np.asarray(pts2d, dtype=float)
    kde = gaussian_kde(pts2d.T)

    xlin = np.linspace(pts2d[:, 0].min(), pts2d[:, 0].max(), gridsize)
    ylin = np.linspace(pts2d[:, 1].min(), pts2d[:, 1].max(), gridsize)
    xgrid, ygrid = np.meshgrid(xlin, ylin)
    xy_coords = np.vstack([xgrid.ravel(), ygrid.ravel()])

    density = kde(xy_coords).reshape(xgrid.shape)

    # Convert density to expected counts per cell (match KDE_MCMC_plot style)
    N = pts2d.shape[0]
    dx = xlin[1] - xlin[0]
    dy = ylin[1] - ylin[0]
    cell_area = dx * dy
    counts = density * N * cell_area

    if isinstance(levels, int):
        level_values = np.linspace(0, float(counts.max()), levels)
    else:
        level_values = np.asarray(levels, dtype=float)

    return xgrid, ygrid, counts, level_values


def _draw_contour_kde2d(
    ax,
    xgrid,
    ygrid,
    counts,
    level_values,
    cmap=None,
    alpha=0.9,
):
    """Draw an already-evaluated 2D KDE contour grid."""
    if cmap is not None:
        try:
            ax.set_facecolor(cmap(0))
        except Exception:
            pass

    return ax.contourf(
        xgrid,
        ygrid,
        counts,
        levels=level_values,
        cmap=cmap,
        alpha=alpha,
        antialiased=True,
        zorder=1,
    )


def _contour_kde2d(ax, pts2d, levels=20, gridsize=200, cmap=None, alpha=0.9):
    """Plot a 2D Gaussian KDE as filled contours.

    This is the original compute-and-draw path. Use
    ``_contour_kde2d_cached`` when a precomputed contour grid may exist in the
    NPZ cache.
    """
    xgrid, ygrid, counts, level_values = _compute_contour_kde2d(
        pts2d, levels=levels, gridsize=gridsize
    )
    return _draw_contour_kde2d(
        ax, xgrid, ygrid, counts, level_values, cmap=cmap, alpha=alpha
    )


def _save_contour_to_dict(out, key, pts2d, levels=20, gridsize=200):
    """Evaluate one contour grid and store it in a flat NPZ-friendly dict."""
    xgrid, ygrid, counts, level_values = _compute_contour_kde2d(
        pts2d, levels=levels, gridsize=gridsize
    )
    out[f"contour_{key}_xgrid"] = xgrid
    out[f"contour_{key}_ygrid"] = ygrid
    out[f"contour_{key}_counts"] = counts
    out[f"contour_{key}_levels"] = level_values


def _contour_kde2d_cached(
    ax,
    contour_cache,
    key,
    pts2d,
    levels=20,
    gridsize=200,
    cmap=None,
    alpha=0.9,
):
    """Draw a cached KDE contour when available; otherwise compute it."""
    if contour_cache is not None:
        xkey = f"contour_{key}_xgrid"
        ykey = f"contour_{key}_ygrid"
        ckey = f"contour_{key}_counts"
        lkey = f"contour_{key}_levels"
        if all(k in contour_cache for k in (xkey, ykey, ckey, lkey)):
            return _draw_contour_kde2d(
                ax,
                contour_cache[xkey],
                contour_cache[ykey],
                contour_cache[ckey],
                contour_cache[lkey],
                cmap=cmap,
                alpha=alpha,
            )

    return _contour_kde2d(
        ax, pts2d, levels=levels, gridsize=gridsize, cmap=cmap, alpha=alpha
    )


def _precompute_rv_plot_contours(
    *,
    GPS_Vessel_rot,
    GPS1,
    levers_rot,
    downsample,
    contour_levers,
):
    """Precompute every expensive KDE contour used in the RV figure.

    This includes the main top/side panels and all zoomed GPS/XDCR insets.
    Later figure edits can redraw the contours directly from these arrays.
    """
    contours = {}

    lever_xy = levers_rot[:, :2] + GPS1[:2]
    lever_xy_ds = lever_xy[::downsample]
    for i in range(4):
        gxy_full = GPS_Vessel_rot[:, i, :2] + GPS1[:2]
        gps_xy_ds = gxy_full[::downsample]
        _save_contour_to_dict(
            contours,
            f"top_gps_main_{i}",
            gps_xy_ds,
            levels=contour_levers,
            gridsize=200,
        )
        _save_contour_to_dict(
            contours,
            f"top_gps_inset_{i}",
            gps_xy_ds,
            levels=contour_levers,
            gridsize=120,
        )

    _save_contour_to_dict(
        contours,
        "top_lever_main",
        lever_xy_ds,
        levels=contour_levers,
        gridsize=200,
    )
    _save_contour_to_dict(
        contours,
        "top_lever_inset",
        lever_xy_ds,
        levels=contour_levers,
        gridsize=120,
    )

    lever_xz = np.column_stack((levers_rot[:, 0] + GPS1[0], levers_rot[:, 2] + GPS1[2]))
    lever_xz_ds = lever_xz[::downsample]
    for i in range(4):
        gps_xz_full = np.column_stack(
            (
                GPS_Vessel_rot[:, i, 0] + GPS1[0],
                GPS_Vessel_rot[:, i, 2] + GPS1[2],
            )
        )
        gps_xz_ds = gps_xz_full[::downsample]
        _save_contour_to_dict(
            contours,
            f"side_gps_main_{i}",
            gps_xz_ds,
            levels=contour_levers,
            gridsize=160,
        )
        _save_contour_to_dict(
            contours,
            f"side_gps_inset_{i}",
            gps_xz_ds,
            levels=contour_levers,
            gridsize=120,
        )

    _save_contour_to_dict(
        contours,
        "side_lever_main",
        lever_xz_ds,
        levels=contour_levers,
        gridsize=160,
    )
    _save_contour_to_dict(
        contours,
        "side_lever_inset",
        lever_xz_ds,
        levels=contour_levers,
        gridsize=120,
    )

    return contours


def px_to_world_segments(
    segments_px,
    scale,
    x_shift_px=0,
    y_shift_px=0,
    gps1_offset=(0, 0, 0),
    view="side",
    flip_y=False,
):
    """Convert pixel coordinates to ship-fixed meter coordinates.

    Parameters
    ----------
    segments_px : sequence of tuple
        Iterable of pixel-space segments [((x1, y1), (x2, y2)), ...].
    scale : float
        Meters per pixel.
    x_shift_px, y_shift_px : float, optional
        Pixel offsets to subtract before scaling.
    gps1_offset : tuple of float, optional
        (x, y, z) offset in meters to translate results.
    view : {"side", "top"}, optional
        Mapping of pixel axes to world axes. Default "side".
    flip_y : bool, optional
        If True, multiply the pixel y-coordinates by -1 before scaling.

    Returns
    -------
    list of tuple
        World-coordinate segments [((x1, y1), (x2, y2)), ...].
    """
    ox, oy, oz = gps1_offset
    world = []
    for (x1, y1), (x2, y2) in segments_px:
        xp1, yp1 = x1 - x_shift_px, y1 - y_shift_px
        xp2, yp2 = x2 - x_shift_px, y2 - y_shift_px

        if flip_y:
            yp1 *= -1
            yp2 *= -1
        xm1, ym1 = xp1 * scale, yp1 * scale
        xm2, ym2 = xp2 * scale, yp2 * scale

        if view == "side":
            p1 = (xm1 + ox, ym1 + oz)
            p2 = (xm2 + ox, ym2 + oz)
        else:
            p1 = (xm1 + ox, ym1 + oy)
            p2 = (xm2 + ox, ym2 + oy)
        world.append((p1, p2))
    return world


# Helper: add a schematic inset showing the full boat outline on the left
def _add_schematic_inset(
    ax,
    segments,
    box=(0.02, 0.10, 0.40, 0.80),
    line_color="0.4",
    line_width=0.8,
    scale=1.0,
    box_coord="axes",
):
    """Add a boat-outline inset to the given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to attach the inset to.
    segments : iterable
        World-coordinate segments [((x1, y1), (x2, y2)), ...] for the outline.
    box : tuple of float, optional
        (x0, y0, width, height) for the inset box. Interpreted in the
        coordinate system given by *box_coord*. Default (0.02, 0.10, 0.40, 0.80).
    line_color : color, optional
        Segment line color. Default "0.35".
    line_width : float, optional
        Segment line width. Default 0.8.
    title : str, optional
        Inset title. Default "Schematic".
    scale : float, optional
        Multiplier applied to width and height of *box*. Default 1.0.
    box_coord : {"axes", "data", "figure"}, optional
        Coordinate system for *box*: axes-fraction [0–1], data units of *ax*,
        or figure-fraction. Default "axes".

    Returns
    -------
    matplotlib.axes.Axes
        The inset axes.
    """
    # Create inset inside the axes, positioned along the left edge
    x0, y0, w, h = box
    w *= scale
    h *= scale
    if box_coord == "data":
        transform = ax.transData
    elif box_coord == "figure":
        transform = ax.figure.transFigure
    else:
        transform = ax.transAxes
    ax_in = ax.inset_axes([x0, y0, w, h], transform=transform, zorder=1)

    xs, ys = [], []
    for (x1, y1), (x2, y2) in segments:
        ax_in.plot([x1, x2], [y1, y2], color=line_color, linewidth=line_width, zorder=2)
        xs.extend([x1, x2])
        ys.extend([y1, y2])

    # Fit view to all segments with a small padding
    x_min, x_max = (min(xs), max(xs)) if xs else (0.0, 1.0)
    y_min, y_max = (min(ys), max(ys)) if ys else (0.0, 1.0)
    pad_x = (x_max - x_min) * 0.05 or 1.0
    pad_y = (y_max - y_min) * 0.05 or 1.0
    ax_in.set_xlim(x_min - pad_x, x_max + pad_x)
    ax_in.set_ylim(y_min - pad_y, y_max + pad_y)

    # Styling: equal aspect, no ticks
    ax_in.set_aspect("equal", "box")
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    for spine in ax_in.spines.values():
        spine.set_visible(False)

    return ax_in


def _save_rv_plot_cache(
    cache_path,
    *,
    segments,
    side_segments,
    GPS_Vessel_rot,
    GPS1,
    levers_rot,
    lever_init_rot,
    lever_prior,
    CDOG_augs,
    aug_prior,
    aug_init,
    rotation_deg,
    gps1_offset,
    downsample,
    zoom_half_range,
    tick_step_cm,
    inset_label_fontsize,
    schematic_box,
    schematic_scale,
    figsize,
    height_ratios,
    contour_levers,
):
    """Save the fully preprocessed data needed to redraw ``RV_Plot``.

    This cache intentionally stores arrays after the expensive GPS/MCMC
    preprocessing steps. The replay script can then redraw the figure without
    loading the MCMC chain, loading the raw GPS file, or recomputing the rigid
    body rotations.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # These are the expensive scipy.stats.gaussian_kde evaluations. Store the
    # resulting contour grids/counts in the same NPZ so figure replays only call
    # ax.contourf on precomputed arrays.
    contour_cache = _precompute_rv_plot_contours(
        GPS_Vessel_rot=np.asarray(GPS_Vessel_rot, dtype=float),
        GPS1=np.asarray(GPS1, dtype=float),
        levers_rot=np.asarray(levers_rot, dtype=float),
        downsample=int(downsample),
        contour_levers=int(contour_levers),
    )

    metadata_and_samples = dict(
        segments=np.asarray(segments, dtype=float),
        side_segments=np.asarray(side_segments, dtype=float),
        GPS_Vessel_rot=np.asarray(GPS_Vessel_rot, dtype=float),
        GPS1=np.asarray(GPS1, dtype=float),
        levers_rot=np.asarray(levers_rot, dtype=float),
        lever_init_rot=np.asarray(lever_init_rot, dtype=float),
        lever_prior=np.asarray(lever_prior, dtype=float),
        CDOG_augs=np.asarray(CDOG_augs, dtype=float),
        aug_prior=np.asarray(aug_prior, dtype=float),
        aug_init=np.asarray(aug_init, dtype=float),
        rotation_deg=np.asarray(rotation_deg, dtype=float),
        gps1_offset=np.asarray(gps1_offset, dtype=float),
        downsample=np.asarray(downsample, dtype=int),
        zoom_half_range=np.asarray(zoom_half_range, dtype=float),
        tick_step_cm=np.asarray(tick_step_cm, dtype=int),
        inset_label_fontsize=np.asarray(inset_label_fontsize, dtype=float),
        schematic_box=np.asarray(schematic_box, dtype=float),
        schematic_scale=np.asarray(schematic_scale, dtype=float),
        figsize=np.asarray(figsize, dtype=float),
        height_ratios=np.asarray(height_ratios, dtype=float),
        contour_levers=np.asarray(contour_levers, dtype=int),
    )

    np.savez_compressed(cache_path, **metadata_and_samples, **contour_cache)
    print(f"Saved RV_Plot cache, including KDE contour grids, to {cache_path}")
    return contour_cache


def _load_rv_plot_cache(cache_path):
    """Load cached plotting arrays saved by ``_save_rv_plot_cache``."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"No RV_Plot cache found at {cache_path}")

    with np.load(cache_path, allow_pickle=False) as cache:
        return {key: cache[key] for key in cache.files}


def RV_Plot_from_cache(cache_path=None, save=True):
    """Redraw ``RV_Plot`` from a saved NPZ cache.

    Use this after running the full script once with ``save_npz=True``. This
    avoids reloading the MCMC chain, reloading the raw GPS file, and recomputing
    the GPS rigid-body rotations.
    """
    return RV_Plot(cache_path=cache_path, load_npz=True, save=save)


def RV_Plot(
    segments=None,
    side_segments=None,
    lever_arms=None,
    lever_prior=None,
    lever_init=None,
    CDOG_augs=None,
    aug_prior=None,
    aug_init=None,
    cache_path=None,
    save_npz=False,
    load_npz=False,
    save=True,
):
    """Top-down (row 1) spans full width; side-view (row 2) spans full width;
    rows 3–4 are split into 3 columns each and returned for external plotting.

    Simplified: many parameters are fixed for consistency and to reduce
    complexity.
    """

    # ---- Fixed constants (tune here, not via function args) ----
    ROTATION_DEG = 29.5
    GPS1_OFFSET = (39.5, 2.2, 14.2)
    DOWNSAMPLE = 5
    ZOOM_HALF_RANGE = 0.10  # meters
    TICK_STEP_CM = 5
    INSET_LABEL_FONTSIZE = 7
    SCHEMATIC_BOX = (0.02, 0.16, 0.17, 0.34)
    SCHEMATIC_SCALE = 2.0
    SEGMENT_COLOR = "0.5"
    FIGSIZE = (10, 14)
    HEIGHT_RATIOS = (3.0, 2.0, 1.5, 1.5)
    MARGINS = dict(left=0.08, right=0.98, top=0.95, bottom=0.07, hspace=0.15)
    CONTOUR_LEVERS = 15

    if cache_path is None:
        cache_path = default_rv_plot_data_path()

    contour_cache = None

    # ---- Data load / cache load ----
    if load_npz:
        cache = _load_rv_plot_cache(cache_path)
        contour_cache = cache

        segments = cache["segments"]
        side_segments = cache["side_segments"]
        GPS_Vessel_rot = cache["GPS_Vessel_rot"]
        GPS1 = cache["GPS1"]
        levers_rot = cache["levers_rot"]
        lever_init_rot = cache["lever_init_rot"]
        lever_prior = cache["lever_prior"]
        CDOG_augs = cache["CDOG_augs"]
        aug_prior = cache["aug_prior"]
        aug_init = cache["aug_init"]

        # Keep the cached values available for exact replay if you changed
        # these constants after writing the cache.
        ROTATION_DEG = float(cache.get("rotation_deg", ROTATION_DEG))
        GPS1_OFFSET = tuple(np.asarray(cache.get("gps1_offset", GPS1_OFFSET)).tolist())
        DOWNSAMPLE = int(cache.get("downsample", DOWNSAMPLE))
        ZOOM_HALF_RANGE = float(cache.get("zoom_half_range", ZOOM_HALF_RANGE))
        TICK_STEP_CM = int(cache.get("tick_step_cm", TICK_STEP_CM))
        INSET_LABEL_FONTSIZE = float(
            cache.get("inset_label_fontsize", INSET_LABEL_FONTSIZE)
        )
        SCHEMATIC_BOX = tuple(
            np.asarray(cache.get("schematic_box", SCHEMATIC_BOX)).tolist()
        )
        SCHEMATIC_SCALE = float(cache.get("schematic_scale", SCHEMATIC_SCALE))
        FIGSIZE = tuple(np.asarray(cache.get("figsize", FIGSIZE)).tolist())
        HEIGHT_RATIOS = tuple(
            np.asarray(cache.get("height_ratios", HEIGHT_RATIOS)).tolist()
        )
        CONTOUR_LEVERS = int(cache.get("contour_levers", CONTOUR_LEVERS))
    else:
        required = {
            "segments": segments,
            "side_segments": side_segments,
            "lever_arms": lever_arms,
            "lever_prior": lever_prior,
            "lever_init": lever_init,
            "CDOG_augs": CDOG_augs,
            "aug_prior": aug_prior,
            "aug_init": aug_init,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "Missing required inputs for full RV_Plot run: " + ", ".join(missing)
            )

        data = np.load(gps_data_path("GPS_Data/Processed_GPS_Receivers_DOG_1.npz"))
        GPS_grid = np.array(
            [
                [0.0, 0.0, 0.0],
                [-2.39341409, -4.22350344, 0.02941493],
                [-12.09568416, -0.94568462, 0.0043972],
                [-8.68674054, 5.16918806, 0.02499322],
            ]
        )
        GPS_Coordinates = data["GPS_Coordinates"]

        GPS_Vessel = np.zeros_like(GPS_Coordinates)
        for i in range(GPS_Coordinates.shape[0]):
            R_mtrx, d = findRotationAndDisplacement(GPS_Coordinates[i].T, GPS_grid.T)
            GPS_Vessel[i] = (R_mtrx @ GPS_Coordinates[i].T + d[:, None]).T

        GPS1 = np.array(GPS1_OFFSET)

        theta = np.deg2rad(ROTATION_DEG)
        Rz = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )

        lever_init_rot = np.asarray(lever_init) @ Rz.T
        levers_rot = np.asarray(lever_arms) @ Rz.T
        GPS_Vessel_rot = GPS_Vessel @ Rz.T

        if save_npz:
            contour_cache = _save_rv_plot_cache(
                cache_path,
                segments=segments,
                side_segments=side_segments,
                GPS_Vessel_rot=GPS_Vessel_rot,
                GPS1=GPS1,
                levers_rot=levers_rot,
                lever_init_rot=lever_init_rot,
                lever_prior=lever_prior,
                CDOG_augs=CDOG_augs,
                aug_prior=aug_prior,
                aug_init=aug_init,
                rotation_deg=ROTATION_DEG,
                gps1_offset=GPS1_OFFSET,
                downsample=DOWNSAMPLE,
                zoom_half_range=ZOOM_HALF_RANGE,
                tick_step_cm=TICK_STEP_CM,
                inset_label_fontsize=INSET_LABEL_FONTSIZE,
                schematic_box=SCHEMATIC_BOX,
                schematic_scale=SCHEMATIC_SCALE,
                figsize=FIGSIZE,
                height_ratios=HEIGHT_RATIOS,
                contour_levers=CONTOUR_LEVERS,
            )

    fig = plt.figure(figsize=FIGSIZE, constrained_layout=False)
    gs = fig.add_gridspec(4, 1, height_ratios=list(HEIGHT_RATIOS))
    fig.subplots_adjust(**MARGINS)

    # Rows 1 and 2: single full-width axes guaranteed to span the figure
    ax = fig.add_subplot(gs[0])
    ax_side = fig.add_subplot(gs[1])

    # Row 3 and Row 4: split into 3 columns each using sub-gridspecs
    gs_row3 = gs[2].subgridspec(1, 3)

    ax3_1 = fig.add_subplot(gs_row3[0, 0])
    ax3_2 = fig.add_subplot(gs_row3[0, 1])
    ax3_3 = fig.add_subplot(gs_row3[0, 2])
    ax3 = [ax3_1, ax3_2, ax3_3]

    gs_row4 = gs[3].subgridspec(1, 3)
    ax4_1 = fig.add_subplot(gs_row4[0, 0])
    ax4_2 = fig.add_subplot(gs_row4[0, 1])
    ax4_3 = fig.add_subplot(gs_row4[0, 2])
    ax4 = [ax4_1, ax4_2, ax4_3]

    # Top-down (X–Y) lever KDE
    lever_xy = levers_rot[:, :2] + GPS1[:2]

    gps_xy_list = []
    centroids = []
    for i in range(4):
        gxy_full = GPS_Vessel_rot[:, i, :2] + GPS1[:2]
        gps_xy_list.append(gxy_full[::DOWNSAMPLE])
        centroids.append(gxy_full.mean(axis=0))

    white_green_cmap = LinearSegmentedColormap.from_list(
        "white_green", ["white", "green"]
    )

    white_red_cmap = LinearSegmentedColormap.from_list("white_red", ["white", "red"])

    for idx, pts in enumerate(gps_xy_list):
        _contour_kde2d_cached(
            ax,
            contour_cache,
            f"top_gps_main_{idx}",
            pts,
            levels=CONTOUR_LEVERS,
            gridsize=200,
            cmap=white_green_cmap,
            alpha=0.9,
        )

    lever_xy_ds = lever_xy[::DOWNSAMPLE]
    _contour_kde2d_cached(
        ax,
        contour_cache,
        "top_lever_main",
        lever_xy_ds,
        levels=CONTOUR_LEVERS,
        gridsize=200,
        cmap=white_red_cmap,
        alpha=0.9,
    )

    # 68% error ellipses (GPS and lever) + prior
    for i in range(4):
        gps_i = GPS_Vessel_rot[:, i, :2] + GPS1[:2]
        ellipse, _ = compute_error_ellipse(gps_i, confidence=0.68, zorder=3)
        ax.add_patch(ellipse)

    ellipse, _ = compute_error_ellipse(lever_xy, confidence=0.68, zorder=3)
    prior_ellipse = plot_prior_ellipse(
        mean=lever_init_rot[:2] + GPS1[:2],
        cov=np.diag(lever_prior[:2] ** 2),
        confidence=0.68,
        zorder=3,
    )
    ax.add_patch(ellipse)
    ax.add_patch(prior_ellipse)

    # Lever inset in top view (same size/style as GPS insets)
    lever_c_xy = lever_xy.mean(axis=0)
    cx, cy = float(lever_c_xy[0]), float(lever_c_xy[1])
    w, h = 2.0, 2.5  # match GPS inset size in top view
    # Offset the inset slightly from the centroid to avoid overlap
    dx, dy = 2.5, 0.27
    ix, iy = cx + dx, cy + dy
    axins_lev_xy = ax.inset_axes(
        [ix - w / 2.0, iy - h / 2.0, w, h], transform=ax.transData
    )
    _contour_kde2d_cached(
        axins_lev_xy,
        contour_cache,
        "top_lever_inset",
        lever_xy_ds,
        levels=CONTOUR_LEVERS,
        gridsize=120,
        cmap=white_red_cmap,
        alpha=0.9,
    )
    ellipse_lev_xy_ins, _ = compute_error_ellipse(lever_xy, confidence=0.68, zorder=4)
    ellipse_lev_xy_ins.set_fill(False)
    ellipse_lev_xy_ins.set_linewidth(1.0)
    axins_lev_xy.add_patch(ellipse_lev_xy_ins)

    axins_lev_xy.set_xlim(cx - 0.4, cx + 0.4)
    axins_lev_xy.set_ylim(cy - 0.4, cy + 0.4)
    rng_m = 0.4
    cm_max = int(np.floor(rng_m * 100.0))
    step = max(1, int(20))
    tick_cm = np.arange(-cm_max, cm_max + 1, step, dtype=int)
    xticks = cx + (tick_cm / 100.0)
    yticks = cy + (tick_cm / 100.0)
    axins_lev_xy.set_xticks(xticks)
    axins_lev_xy.set_yticks(yticks)
    axins_lev_xy.set_xticklabels([f"{v}" for v in tick_cm])
    axins_lev_xy.set_yticklabels([f"{v}" for v in tick_cm])
    axins_lev_xy.tick_params(axis="both", labelsize=INSET_LABEL_FONTSIZE)
    axins_lev_xy.set_xlabel("cm", fontsize=INSET_LABEL_FONTSIZE)
    axins_lev_xy.set_ylabel("cm", fontsize=INSET_LABEL_FONTSIZE)
    for spine in axins_lev_xy.spines.values():
        spine.set_linewidth(0.8)
        spine.set_alpha(0.8)

    ax.annotate(
        "",
        xy=(ix, iy),
        xytext=(cx, cy),
        arrowprops=dict(arrowstyle="->", lw=1.0, alpha=0.9),
        zorder=4,
    )

    # Redraw outline on top of KDE so it's visible
    for (x1, y1), (x2, y2) in segments:
        ax.plot([x1, x2], [y1, y2], color=SEGMENT_COLOR, linewidth=1.2, zorder=10)

    # Top view insets (positions fixed as before)
    GPS_labels = {
        1: "$\mathbf{x}^\mathrm{S}_1$",
        2: "$\mathbf{x}^\mathrm{S}_2$",
        3: "$\mathbf{x}^\mathrm{S}_3$",
        4: "$\mathbf{x}^\mathrm{S}_4$",
    }
    for idx, (pts, c) in enumerate(zip(gps_xy_list, centroids), start=1):
        w = 2.0
        h = 2.5
        cx, cy = float(c[0]), float(c[1])
        if idx == 1:
            dx, dy = 2.0, 2.24
        elif idx == 2:
            dx, dy = 2.0, -0.9
        elif idx == 3:
            dx, dy = -2.0, 1.05
        else:
            dx, dy = 4.0, 2.0
        ix, iy = cx + dx, cy + dy
        axins = ax.inset_axes(
            [ix - w / 2.0, iy - h / 2.0, w, h], transform=ax.transData
        )
        _contour_kde2d_cached(
            axins,
            contour_cache,
            f"top_gps_inset_{idx - 1}",
            pts,
            levels=CONTOUR_LEVERS,
            gridsize=120,
            cmap=white_green_cmap,
            alpha=0.9,
        )

        ellipse_ins, _ = compute_error_ellipse(pts, confidence=0.68, zorder=4)
        ellipse_ins.set_fill(False)
        ellipse_ins.set_linewidth(1.0)
        axins.add_patch(ellipse_ins)

        axins.set_xlim(cx - ZOOM_HALF_RANGE, cx + ZOOM_HALF_RANGE)
        axins.set_ylim(cy - ZOOM_HALF_RANGE, cy + ZOOM_HALF_RANGE)
        rng_m = ZOOM_HALF_RANGE
        cm_max = int(np.floor(rng_m * 100.0))
        step = max(1, int(TICK_STEP_CM))
        tick_cm = np.arange(-cm_max, cm_max + 1, step, dtype=int)
        xticks = cx + (tick_cm / 100.0)
        yticks = cy + (tick_cm / 100.0)
        axins.set_xticks(xticks)
        axins.set_yticks(yticks)
        axins.set_xticklabels([f"{v}" for v in tick_cm])
        axins.set_yticklabels([f"{v}" for v in tick_cm])
        axins.tick_params(axis="both", labelsize=INSET_LABEL_FONTSIZE)
        axins.set_xlabel("cm", fontsize=INSET_LABEL_FONTSIZE)
        axins.set_ylabel("cm", fontsize=INSET_LABEL_FONTSIZE)
        for spine in axins.spines.values():
            spine.set_linewidth(0.8)
            spine.set_alpha(0.8)
        ax.annotate(
            "",
            xy=(ix, iy),
            xytext=(cx, cy),
            arrowprops=dict(arrowstyle="->", lw=1.0, alpha=0.9),
            zorder=4,
        )
        if idx == 1:
            text_offset = [-0.6, -1]
        if idx == 2:
            text_offset = [-0.6, 0.0]
        if idx == 3:
            text_offset = [0.6, 0.0]
        if idx == 4:
            text_offset = [0.6, -1.4]
        ax.text(
            cx + text_offset[0],
            cy + text_offset[1],
            GPS_labels[idx],
            fontsize=18,
            ha="center",
            va="bottom",
        )

    # Always add schematic inset with fixed placement/scale
    _add_schematic_inset(
        ax,
        segments,
        box=SCHEMATIC_BOX,
        scale=SCHEMATIC_SCALE,
        box_coord="axes",
    )

    # Axes cosmetics
    ax.set_xlim(20, 44)
    ax.set_ylim(-7.5, 7.5)
    ax.set_xlabel("")
    ax.set_ylabel("Y (m)")
    ax.set_title("Top-Down View")

    # ---- Bottom panel: side-view schematic from segments (always drawn) ----
    # Build GPS X–Z lists
    gps_xz_list = []
    centroids_xz = []
    for i in range(4):
        g_xz_full = np.column_stack(
            (
                GPS_Vessel_rot[:, i, 0] + GPS1[0],
                GPS_Vessel_rot[:, i, 2] + GPS1[2],
            )
        )
        gps_xz_list.append(g_xz_full[::DOWNSAMPLE])
        centroids_xz.append(g_xz_full.mean(axis=0))

    # KDE contours for each GPS in side view
    for idx, pts in enumerate(gps_xz_list):
        _contour_kde2d_cached(
            ax_side,
            contour_cache,
            f"side_gps_main_{idx}",
            pts,
            levels=CONTOUR_LEVERS,
            gridsize=160,
            cmap=white_green_cmap,
            alpha=0.9,
        )

    # 68% error ellipses for each GPS in side view
    for pts in gps_xz_list:
        ellipse_xz, _ = compute_error_ellipse(pts, confidence=0.68, zorder=4)
        ax_side.add_patch(ellipse_xz)

    # Lever distribution (X–Z)
    lever_xz = np.column_stack((levers_rot[:, 0] + GPS1[0], levers_rot[:, 2] + GPS1[2]))
    lever_xz_ds = lever_xz[::DOWNSAMPLE]
    _contour_kde2d_cached(
        ax_side,
        contour_cache,
        "side_lever_main",
        lever_xz_ds,
        levels=CONTOUR_LEVERS,
        gridsize=160,
        cmap=white_red_cmap,
        alpha=0.9,
    )

    ellipse_lev_xz, _ = compute_error_ellipse(lever_xz, confidence=0.68, zorder=5)
    ax_side.add_patch(ellipse_lev_xz)

    # Prior ellipse for lever (X–Z)
    prior_mean_xz = np.array([lever_init_rot[0] + GPS1[0], lever_init_rot[2] + GPS1[2]])
    prior_cov_xz = np.diag([lever_prior[0] ** 2, lever_prior[2] ** 2])
    prior_ellipse_xz = plot_prior_ellipse(
        mean=prior_mean_xz, cov=prior_cov_xz, confidence=0.68, zorder=5
    )
    ax_side.add_patch(prior_ellipse_xz)

    # Lever inset in side view
    lever_c_xz = lever_xz.mean(axis=0)
    cx, cz = float(lever_c_xz[0]), float(lever_c_xz[1])
    w, h = 2.0, 5.0
    dx, dz = 6.0, 3.0
    ix, iz = cx + dx, cz + dz
    axins_lev_xz = ax_side.inset_axes(
        [ix - w / 2.0, iz - h / 2.0, w, h], transform=ax_side.transData
    )
    _contour_kde2d_cached(
        axins_lev_xz,
        contour_cache,
        "side_lever_inset",
        lever_xz_ds,
        levels=CONTOUR_LEVERS,
        gridsize=120,
        cmap=white_red_cmap,
        alpha=0.9,
    )
    ellipse_lev_xz_ins, _ = compute_error_ellipse(lever_xz, confidence=0.68, zorder=5)
    ellipse_lev_xz_ins.set_fill(False)
    ellipse_lev_xz_ins.set_linewidth(1.0)
    axins_lev_xz.add_patch(ellipse_lev_xz_ins)

    axins_lev_xz.set_xlim(cx - 0.4, cx + 0.4)
    axins_lev_xz.set_ylim(cz - 0.4, cz + 0.4)
    rng_m = 0.4
    cm_max = int(np.floor(rng_m * 100.0))
    step = max(1, int(20))
    tick_cm = np.arange(-cm_max, cm_max + 1, step, dtype=int)
    xticks = (np.array(tick_cm) / 100.0) + cx
    zticks = (np.array(tick_cm) / 100.0) + cz
    axins_lev_xz.set_xticks(xticks)
    axins_lev_xz.set_yticks(zticks)
    axins_lev_xz.set_xticklabels([f"{v}" for v in tick_cm])
    axins_lev_xz.set_yticklabels([f"{v}" for v in tick_cm])
    axins_lev_xz.tick_params(axis="both", labelsize=INSET_LABEL_FONTSIZE)
    axins_lev_xz.set_xlabel("cm", fontsize=INSET_LABEL_FONTSIZE)
    axins_lev_xz.set_ylabel("cm", fontsize=INSET_LABEL_FONTSIZE)
    for spine in axins_lev_xz.spines.values():
        spine.set_linewidth(0.8)
        spine.set_alpha(0.8)

    ax_side.annotate(
        "",
        xy=(ix, iz),
        xytext=(cx, cz),
        arrowprops=dict(arrowstyle="->", lw=1.0, alpha=0.9),
        zorder=4,
    )

    # Create square insets near each GPS centroid in X–Z, positioned below the
    # bridge line
    for idx, (pts, c) in enumerate(zip(gps_xz_list, centroids_xz), start=1):
        w = 2.0
        h = 5.0
        cx, cz = float(c[0]), float(c[1])
        # offsets (keep below)
        if idx in (1, 3):
            dx, dz = -2.0, -1.0
        else:
            dx, dz = 2.0, -1.0
        ix, iz = cx + dx, cz + dz
        axins = ax_side.inset_axes(
            [ix - w / 2.0, iz - h / 2.0, w, h], transform=ax_side.transData
        )
        _contour_kde2d_cached(
            axins,
            contour_cache,
            f"side_gps_inset_{idx - 1}",
            pts,
            levels=CONTOUR_LEVERS,
            gridsize=120,
            cmap=white_green_cmap,
            alpha=0.9,
        )
        ellipse_ins, _ = compute_error_ellipse(pts, confidence=0.68, zorder=5)
        ellipse_ins.set_fill(False)
        ellipse_ins.set_linewidth(1.0)
        axins.add_patch(ellipse_ins)

        axins.set_xlim(cx - ZOOM_HALF_RANGE, cx + ZOOM_HALF_RANGE)
        axins.set_ylim(cz - ZOOM_HALF_RANGE, cz + ZOOM_HALF_RANGE)
        rng_m = ZOOM_HALF_RANGE
        cm_max = int(np.floor(rng_m * 100.0))
        step = max(1, int(TICK_STEP_CM))
        tick_cm = np.arange(-cm_max, cm_max + 1, step, dtype=int)
        xticks = (np.array(tick_cm) / 100.0) + cx
        zticks = (np.array(tick_cm) / 100.0) + cz
        axins.set_xticks(xticks)
        axins.set_yticks(zticks)
        axins.set_xticklabels([f"{v}" for v in tick_cm])
        axins.set_yticklabels([f"{v}" for v in tick_cm])
        axins.tick_params(axis="both", labelsize=INSET_LABEL_FONTSIZE)
        axins.set_xlabel("cm", fontsize=INSET_LABEL_FONTSIZE)
        axins.set_ylabel("cm", fontsize=INSET_LABEL_FONTSIZE)
        for spine in axins.spines.values():
            spine.set_linewidth(0.8)
            spine.set_alpha(0.8)

        ax_side.annotate(
            "",
            xy=(ix, iz),
            xytext=(cx, cz),
            arrowprops=dict(arrowstyle="->", lw=1.0, alpha=0.9),
            zorder=4,
        )

    _add_schematic_inset(
        ax_side,
        side_segments,
        box=(0.02, 0.12, 0.15, 0.30),
        scale=2.0,
        box_coord="axes",
    )

    # Side schematic segments
    xs, zs = [], []
    for (x1, z1), (x2, z2) in side_segments:
        ax_side.plot([x1, x2], [z1, z2], color=SEGMENT_COLOR, linewidth=1.0, zorder=2)
        xs.extend([x1, x2])
        zs.extend([z1, z2])

    ax_side.set_xlim(20, 44)
    ax_side.set_ylim(-1.5, 16.5)
    ax_side.set_title("Side View")
    ax_side.set_xlabel("X (m)")
    ax_side.set_ylabel("Z (m)")

    # Standard deviation table (x, y, z) in centimeters
    gps_std_rows = []
    for i in range(4):
        pts_3d = GPS_Vessel_rot[:, i, :]
        sx_cm, sy_cm, sz_cm = (np.std(pts_3d, axis=0) * 100.0).tolist()
        gps_std_rows.append([sx_cm, sy_cm, sz_cm])

    lever_sx_cm, lever_sy_cm, lever_sz_cm = (
        np.std(levers_rot, axis=0) * 100.0
    ).tolist()

    # Prepare table contents (rounded to 0.1 cm)
    row_labels = [
        "$\mathbf{x}^\mathrm{S}_1$",
        "$\mathbf{x}^\mathrm{S}_2$",
        "$\mathbf{x}^\mathrm{S}_3$",
        "$\mathbf{x}^\mathrm{S}_4$",
        "$\mathbf{x}^\mathrm{T}$",
    ]
    cell_text = [
        [
            f"{gps_std_rows[0][0]:.1f}",
            f"{gps_std_rows[0][1]:.1f}",
            f"{gps_std_rows[0][2]:.1f}",
        ],
        [
            f"{gps_std_rows[1][0]:.1f}",
            f"{gps_std_rows[1][1]:.1f}",
            f"{gps_std_rows[1][2]:.1f}",
        ],
        [
            f"{gps_std_rows[2][0]:.1f}",
            f"{gps_std_rows[2][1]:.1f}",
            f"{gps_std_rows[2][2]:.1f}",
        ],
        [
            f"{gps_std_rows[3][0]:.1f}",
            f"{gps_std_rows[3][1]:.1f}",
            f"{gps_std_rows[3][2]:.1f}",
        ],
        [f"{lever_sx_cm:.1f}", f"{lever_sy_cm:.1f}", f"{lever_sz_cm:.1f}"],
    ]

    col_labels = [r"$\sigma_x$ (cm)", r"$\sigma_y$ (cm)", r"$\sigma_z$ (cm)"]

    tbl = ax_side.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        bbox=[0.63, 0.02, 0.32, 0.55],
        colLoc="center",
        rowLoc="left",
        cellLoc="center",
        edges="closed",
    )
    # Style the table to be compact and readable
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.15, 1.5)

    tbl.set_zorder(20)
    for _key, cell in tbl.get_celld().items():
        cell.set_facecolor("white")
        cell.set_alpha(1.0)
        cell.set_zorder(20)

    # Auto-size columns to content and minimize padding so columns use only needed space
    try:
        # Must draw once before auto-sizing to get a valid renderer
        ax_side.figure.canvas.draw_idle()
        cols = [-1] + list(range(len(col_labels)))  # include row-label column (-1)
        # Auto width based on rendered text extents
        if hasattr(tbl, "auto_set_column_width"):
            tbl.auto_set_column_width(cols)
    except Exception:
        # Fallback: keep going even if backend doesn't support autosize here
        pass

    # Reduce per-cell padding to tighten layout
    for _key, cell in tbl.get_celld().items():
        # Smaller padding yields tighter columns while preserving readability
        if hasattr(cell, "PAD"):
            cell.PAD = 0.12

    CDOG_label = {0: "DOG 1", 1: "DOG 3", 2: "DOG 4"}
    for i in range(3):
        CDOG_aug = CDOG_augs[:, i]
        init = aug_init[i]
        _, _, _ = plot_kde_mcmc(
            CDOG_aug,
            nbins=100,
            prior_mean=init,
            prior_sd=aug_prior,
            ellipses=1,
            return_axes=True,
            ax1=ax3[i],
            ax2=ax4[i],
            fig=fig,
        )
        ax3[i].legend().remove()
        ax4[i].legend().remove()
        ax3[i].set_title(f"{CDOG_label[i]} in ENU")
        ax4[i].set_title("")

    # Collect current limits
    row3_xmins, row3_xmaxs, row3_ymins, row3_ymaxs = [], [], [], []
    row4_xmins, row4_xmaxs, row4_ymins, row4_ymaxs = [], [], [], []
    for a in ax3:
        xlim = a.get_xlim()
        ylim = a.get_ylim()
        row3_xmins.append(xlim[0])
        row3_xmaxs.append(xlim[1])
        row3_ymins.append(ylim[0])
        row3_ymaxs.append(ylim[1])
    for a in ax4:
        xlim = a.get_xlim()
        ylim = a.get_ylim()
        row4_xmins.append(xlim[0])
        row4_xmaxs.append(xlim[1])
        row4_ymins.append(ylim[0])
        row4_ymaxs.append(ylim[1])

    # Compute common limits per row
    row3_xlim = (min(row3_xmins), max(row3_xmaxs))
    row3_ylim = (min(row3_ymins), max(row3_ymaxs))
    row4_xlim = (min(row4_xmins), max(row4_xmaxs))
    row4_ylim = (min(row4_ymins), max(row4_ymaxs))

    # Apply common limits and lock equal aspect for metric fidelity
    for a in ax3:
        a.set_xlim(row3_xlim)
        a.set_ylim(row3_ylim)
        a.set_aspect("equal", "box")
    for a in ax4:
        a.set_xlim(row4_xlim)
        a.set_ylim(row4_ylim)
        a.set_aspect("equal", "box")

    # Remove colorbar
    fig = ax.figure
    for cax in fig.axes:
        if cax.get_label() == "<colorbar>":
            cax.remove()

    # Add text to plots
    ax.text(
        0.6,
        0.435,
        "bridge",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=14,
    )
    ax.text(
        0.155,
        0.43,
        "bridge",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=14,
        color="0.2",
    )
    ax.text(
        0.058,
        0.755,
        "moonpool",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=14,
    )
    ax.text(
        0.097,
        0.518,
        "moonpool",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=14,
        color="0.2",
    )

    ax_side.text(
        0.6,
        0.88,
        "bridge",
        transform=ax_side.transAxes,
        ha="center",
        va="bottom",
        fontsize=14,
    )
    ax_side.text(
        0.137,
        0.451,
        "bridge",
        transform=ax_side.transAxes,
        ha="center",
        va="bottom",
        fontsize=14,
        color="0.2",
    )
    ax_side.text(
        0.07,
        0.012,
        "moonpool",
        transform=ax_side.transAxes,
        ha="center",
        va="bottom",
        fontsize=14,
    )
    ax_side.text(
        0.1387,
        0.32,
        "moonpool",
        transform=ax_side.transAxes,
        ha="center",
        va="bottom",
        fontsize=14,
        color="0.2",
    )

    # Add plot letters
    ax.text(
        0.005,
        1.07,
        r"\textbf{A}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=20,
    )
    ax_side.text(
        0.005,
        1.095,
        r"\textbf{B}",
        transform=ax_side.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=20,
    )
    ax3[0].text(
        -0.05,
        1.3,
        r"\textbf{C}",
        transform=ax3[0].transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=20,
    )
    if save:
        save_plot(fig, func_name="RV_Plot", subdir="Figs", ext="pdf")

    return fig, (ax, ax_side, ax3, ax4)


if __name__ == "__main__":
    """Top View"""
    bridge_segments = [
        ((361.68853695, 255.13511531), (361.68853695, 118.19570136)),
        ((361.68853695, 118.19570136), (412.0361991, 118.19570136)),
        ((412.0361991, 118.19570136), (412.0361991, 134.16699219)),
        ((412.0361991, 134.16699219), (453.99414063, 134.16699219)),
        ((453.99414063, 134.16699219), (453.99414063, 118.30517578)),
        ((453.99414063, 118.30517578), (550.13084465, 118.30517578)),
        ((550.13084465, 118.30517578), (550.13084465, 217.21435547)),
        ((550.13084465, 217.21435547), (425.11539656, 217.21435547)),
        ((425.11539656, 217.21435547), (425.11539656, 255.13511531)),
        ((425.11539656, 255.13511531), (361.68853695, 255.13511531)),
    ]

    ship_segments = [
        ((47.18438914, 243.84615385), (47.18438914, 67.09863281)),
        ((47.18438914, 67.09863281), (172.00781250, 51.22558594)),
        ((172.00781250, 51.22558594), (815.26562500, 51.41308594)),
        ((815.26562500, 51.41308594), (1000.39257813, 158.67382812)),
        ((1000.39257813, 158.67382812), (815.26562500, 259.94409180)),
        ((815.26562500, 259.94409180), (175.97265625, 259.94409180)),
        ((175.97265625, 259.94409180), (47.18438914, 243.84615385)),
    ]

    moonpool_segments = [
        ((413.96794872, 366.93099548), (432.05656109, 366.93099548)),
        ((432.05656109, 366.93099548), (432.05656109, 347.07126697)),
        ((432.05656109, 347.07126697), (413.96794872, 347.07126697)),
        ((413.96794872, 347.07126697), (413.96794872, 366.93099548)),
    ]

    bridge_center = 166.0
    ship_center = 158.03
    moonpool_y_shift = 276 + 158.03
    bridge_x_shift = -175
    moonpool_x_shift = -3

    top_down_scale = 0.054715

    chain = np.load(gps_output_path("mcmc_chain_1_22_new_MCMC_long.npz"))
    levers = chain["lever"][::5000]
    CDOG_augs = chain["CDOG_aug"][::5000]

    init_params, prior_scales, _ = get_init_params_and_prior(chain)

    lever_init = init_params["lever"]
    lever_prior = prior_scales["lever"]
    aug_init = init_params["CDOG_aug"]
    aug_prior = prior_scales["CDOG_aug"]

    segments_bridge = px_to_world_segments(
        bridge_segments,
        scale=top_down_scale,
        x_shift_px=bridge_x_shift,
        y_shift_px=bridge_center,
        gps1_offset=(0, 0, 0),
        view="top",
        flip_y=True,
    )
    segments_ship = px_to_world_segments(
        ship_segments,
        scale=top_down_scale,
        x_shift_px=0,
        y_shift_px=ship_center,
        gps1_offset=(0, 0, 0),
        view="top",
        flip_y=True,
    )
    segments_moonpool = px_to_world_segments(
        moonpool_segments,
        scale=top_down_scale,
        x_shift_px=moonpool_x_shift,
        y_shift_px=moonpool_y_shift,
        gps1_offset=(0, 0, 0),
        view="top",
        flip_y=True,
    )
    segments = np.concatenate([segments_bridge, segments_ship, segments_moonpool])

    # Make side segments manually (give height manually for
    # top and hull and moonpool)
    side_ship_segments = [
        ((10, 8.5), (29.366, 8.5)),
        ((29.366, 8.5), (29.366, 14.2)),
        ((29.366, 14.2), (39.676, 14.2)),
        ((39.676, 14.2), (39.676, 10.0)),
        ((39.676, 10.0), (45, 10.0)),
        ((45, 10.0), (54.74, 10.0)),
        ((54.74, 10.0), (44.74, 0.0)),
        ((44.74, 0.0), (8.0, 0.0)),
        ((8.0, 0.0), (2.58, 8.5)),
        ((2.58, 8.5), (10, 8.5)),
    ]
    side_moonpool_segments = [
        ((23.804, 0.790), (22.814, 0.790)),
        ((22.814, 0.790), (22.814, -0.200)),
        ((22.814, -0.200), (23.804, -0.200)),
        ((23.804, -0.200), (23.804, 0.790)),
    ]

    side_segments = side_ship_segments + side_moonpool_segments

    plot_data_dir = gps_output_path("Plot_Data")
    plot_data_path = f"{plot_data_dir}/Atlantic_RV_Plot_Data.npz"
    load = True

    if load:
        fig, axes = RV_Plot_from_cache(
            cache_path=default_rv_plot_data_path(),
            save=True,
        )
        plt.show()
    else:
        fig, (ax, ax_side, ax3, ax4) = RV_Plot(
            segments,
            side_segments,
            levers,
            lever_prior,
            lever_init,
            CDOG_augs,
            aug_prior,
            aug_init,
            cache_path=plot_data_path,
            save_npz=True,
        )

        plt.show()
