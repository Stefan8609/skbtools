import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from plotting.Ellipses.Error_Ellipse import compute_error_ellipse
from plotting.Ellipses.Prior_Ellipse import plot_prior_ellipse
from plotting.MCMC_plots import get_init_params_and_prior
from plotting.KDE_MCMC_plot import plot_kde_mcmc
from plotting.save import save_plot

from geometry.rigid_body import findRotationAndDisplacement
from data import gps_output_path, gps_data_path

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "mathtext.fontset": "cm",
        "text.latex.preamble": r"\usepackage[utf8]{inputenc}"
        "\n"
        r"\usepackage{textcomp}",
    }
)


def _contour_kde2d(ax, pts2d, levels=50, gridsize=200, cmap=None, alpha=0.9):
    """Plot a 2D Gaussian KDE as filled contours.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    pts2d : (N, 2) ndarray
        Input points as columns [x, y].
    levels : int, optional
        Number of contour levels. Default 50.
    gridsize : int, optional
        Grid resolution per axis. Default 200.
    cmap : matplotlib.colors.Colormap, optional
        Colormap for the density.
    alpha : float, optional
        Transparency for the filled contours. Default 0.9.

    Returns
    -------
    matplotlib.contour.QuadContourSet
        The contourf artist.
    """
    kde = gaussian_kde(pts2d.T)
    xlin = np.linspace(pts2d[:, 0].min(), pts2d[:, 0].max(), gridsize)
    ylin = np.linspace(pts2d[:, 1].min(), pts2d[:, 1].max(), gridsize)
    xgrid, ygrid = np.meshgrid(xlin, ylin)
    xy_coords = np.vstack([xgrid.ravel(), ygrid.ravel()])
    density = kde(xy_coords).reshape(xgrid.shape)
    return ax.contourf(
        xgrid, ygrid, density, levels=levels, cmap=cmap, alpha=alpha, zorder=1
    )


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
    line_color="0.7",
    line_width=0.8,
    title="Schematic",
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
    ax_in = ax.inset_axes([x0, y0, w, h], transform=transform)

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

    # Styling: equal aspect, no ticks, subtle frame
    ax_in.set_aspect("equal", "box")
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    for spine in ax_in.spines.values():
        spine.set_visible(False)

    ax_in.set_title(title, fontsize=8, pad=2)
    return ax_in


def RV_Plot(
    segments,
    side_segments,
    lever_arms,
    lever_prior,
    lever_init,
    CDOG_augs,
    aug_prior,
    aug_init,
):
    """Top-down (row 1) spans full width; side-view (row 2) spans full width;
    rows 3–4 are split into 3 columns each and returned for external plotting.

    Simplified: many parameters are fixed for consistency and to reduce
    complexity.
    """

    # ---- Fixed constants (tune here, not via function args) ----
    ROTATION_DEG = 29.5
    GPS1_OFFSET = (39.5, 2.2, 15.5)
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

    # ---- Data load ----
    data = np.load(gps_data_path("GPS_Data/Processed_GPS_Receivers_DOG_1.npz"))
    GPS_grid = data["gps1_to_others"]
    GPS_Coordinates = data["GPS_Coordinates"]

    GPS_Vessel = np.zeros_like(GPS_Coordinates)
    for i in range(GPS_Coordinates.shape[0]):
        R_mtrx, d = findRotationAndDisplacement(GPS_Coordinates[i].T, GPS_grid.T)
        GPS_Vessel[i] = (R_mtrx @ GPS_Coordinates[i].T + d[:, None]).T

    GPS1 = np.array(GPS1_OFFSET)

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

    # ---- Rotate to ship frame ----
    theta = np.deg2rad(ROTATION_DEG)
    Rz = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    lever_init_rot = lever_init @ Rz.T
    levers_rot = np.asarray(lever_arms) @ Rz.T

    # Top-down (X–Y) lever KDE
    lever_xy = levers_rot[:, :2] + GPS1[:2]

    GPS_Vessel_rot = GPS_Vessel @ Rz.T

    gps_xy_list = []
    centroids = []
    for i in range(4):
        gxy_full = GPS_Vessel_rot[:, i, :2] + GPS1[:2]
        gps_xy_list.append(gxy_full[::DOWNSAMPLE])
        centroids.append(gxy_full.mean(axis=0))

    white_blue_cmap = LinearSegmentedColormap.from_list("white_blue", ["white", "blue"])
    for pts in gps_xy_list:
        _contour_kde2d(
            ax, pts, levels=50, gridsize=200, cmap=white_blue_cmap, alpha=0.9
        )

    white_red_cmap = LinearSegmentedColormap.from_list("white_red", ["white", "red"])
    lever_xy_ds = lever_xy[::DOWNSAMPLE]
    _contour_kde2d(
        ax, lever_xy_ds, levels=50, gridsize=200, cmap=white_red_cmap, alpha=0.9
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

    # Redraw outline on top of KDE so it's visible
    for (x1, y1), (x2, y2) in segments:
        ax.plot([x1, x2], [y1, y2], color=SEGMENT_COLOR, linewidth=1.2, zorder=10)

    # Top view insets (positions fixed as before)
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
            dx, dy = -2.0, 2.0
        ix, iy = cx + dx, cy + dy
        axins = ax.inset_axes(
            [ix - w / 2.0, iy - h / 2.0, w, h], transform=ax.transData
        )
        _contour_kde2d(
            axins, pts, levels=30, gridsize=120, cmap=white_blue_cmap, alpha=0.9
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
    for pts in gps_xz_list:
        _contour_kde2d(
            ax_side, pts, levels=30, gridsize=160, cmap=white_blue_cmap, alpha=0.9
        )

    # 68% error ellipses for each GPS in side view
    for pts in gps_xz_list:
        ellipse_xz, _ = compute_error_ellipse(pts, confidence=0.68, zorder=4)
        ax_side.add_patch(ellipse_xz)

    # Lever distribution (X–Z)
    lever_xz = np.column_stack((levers_rot[:, 0] + GPS1[0], levers_rot[:, 2] + GPS1[2]))
    lever_xz_ds = lever_xz[::DOWNSAMPLE]
    _contour_kde2d(
        ax_side, lever_xz_ds, levels=30, gridsize=160, cmap=white_red_cmap, alpha=0.9
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

    # Create square insets near each GPS centroid in X–Z, positioned below the
    # bridge line
    for idx, (pts, c) in enumerate(zip(gps_xz_list, centroids_xz), start=1):
        w = 2.0
        h = 5.0
        cx, cz = float(c[0]), float(c[1])
        # offsets (keep below)
        if idx in (1, 3):
            dx, dz = -2.0, -3.5
        else:
            dx, dz = 2.0, -3.5
        ix, iz = cx + dx, cz + dz
        axins = ax_side.inset_axes(
            [ix - w / 2.0, iz - h / 2.0, w, h], transform=ax_side.transData
        )
        _contour_kde2d(
            axins, pts, levels=30, gridsize=120, cmap=white_blue_cmap, alpha=0.9
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
            zorder=6,
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

    CDOG_label = {0: "CDOG 1", 1: "CDOG 3", 2: "CDOG 4"}
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

    # Remove colorbar
    fig = ax.figure
    for cax in fig.axes:
        if cax.get_label() == "<colorbar>":
            cax.remove()

    # Add plot letters
    ax.text(
        0.005, 1.04, "A", transform=ax.transAxes, ha="left", va="top", fontweight="bold"
    )
    ax_side.text(
        0.005,
        1.055,
        "B",
        transform=ax_side.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )
    ax3[0].text(
        0.02,
        1.075,
        "C",
        transform=ax3[0].transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )
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

    chain = np.load(gps_output_path("mcmc_chain_9_16_new_table.npz"))
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
    side_bridge_segments = [
        ((10, 8.0), (29.366, 8.0)),
        ((29.366, 8.0), (29.366, 15.5)),
        ((29.366, 15.5), (39.676, 15.5)),
        ((39.676, 15.5), (39.676, 11.0)),
        ((39.676, 11.0), (45, 11.0)),
    ]
    side_hull_segments = [((20, 0.0), (44, 0.0))]
    side_moonpool_segments = [
        ((23.804, 0.2), (22.814, 0.2)),
        ((22.814, 0.2), (22.814, -0.790)),
        ((22.814, -0.790), (23.804, -0.790)),
        ((23.804, -0.790), (23.804, 0.2)),
    ]

    side_segments = side_bridge_segments + side_hull_segments + side_moonpool_segments

    fig, (ax, ax_side, ax3, ax4) = RV_Plot(
        segments,
        side_segments,
        levers,
        lever_prior,
        lever_init,
        CDOG_augs,
        aug_prior,
        aug_init,
    )

    plt.show()

# ADD LABELS IN THE PLOT
# CHANGE AXES NAMES TO OFFICIAL TERMS (RESEARCH INTO THESE)
# MAKE ARROWS IN SIDEVIEW GO TO INSET NOT CENTER
# INSET THE LEVER IN THE SIDEPLOT AND GIVE SCALE
# MAKE THE ENU AXES MATCH FOR EACH OF THE PLOTS
# MAKE THE ERROR ELLIPSES IN THE ENU PLOTS BLACK AND PRIOR THE SAME BLUE
# MAKE THE CONTOURING IN THE INSETS THE SAME AS THE KDE PLOTS
# MAKE A DIAGRAM OF THE PRINCIPAL COMPONENT (WHAT IS IT?)
# PUT A XI LABEL RIGHT NEXT TO THE RED DOTTED LINE
# ADD ARROW TO DOTTED LINE AND MAKE THE ARROW THE LENGTH OF THE X-AXES
# MAKE DOTTED LINE SOLID
# ROUND THE STANDARD DEVIATION TO ONE DECIMAL PLACE
# AND ALIGN THE THE TEXT
# PUT STANDARD DEVIATIONS OF GPS IN THE PLOTS
#   GET THE NUMBERS, BUT NOT RENDER THEM LATER
#   OR FIND FREE SPACE IN THE PLOTS TO MAKE A TABLE
# MAKE CDOG AND GPS COLORS DIFFERENT
#   MAKE GPS LAND COLORED (GREEN)
