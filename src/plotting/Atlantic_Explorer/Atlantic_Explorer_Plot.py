import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from plotting.Ellipses.Error_Ellipse import compute_error_ellipse
from plotting.Ellipses.Prior_Ellipse import plot_prior_ellipse

from geometry.rigid_body import findRotationAndDisplacement
from data import gps_output_path, gps_data_path


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
    line_color="0.35",
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
        spine.set_alpha(0.6)
        spine.set_linewidth(0.8)

    ax_in.set_title(title, fontsize=8, pad=2)
    return ax_in


def RV_Plot(
    segments,
    lever_arms,
    lever_prior,
    lever_init,
    rotation_deg=29.5,
    gps1_offset=(39.5, 2.2, 15.0),
    downsample=1,
    inset_size=2.0,
    zoom_half_range=0.25,
    tick_step_cm=25,
    inset_label_fontsize=7,
    schematic_inset=True,
    schematic_box=(0.02, 0.16, 0.17, 0.34),
    schematic_scale=2.0,
    schematic_box_coord="axes",
    side_segments=None,
):
    """Top-down + side-view figure for the R/V: top plot retains size;
    bottom is a half-height side-view schematic built from segments.

    Parameters
    ----------
    segments : sequence
        World-coordinate segments [((x1, y1), (x2, y2)), ...] for the hull.
    lever_arms : (N, 3) array-like
        Lever-arm samples in ship-fixed coordinates (meters).
    lever_prior : (3,) array-like
        Prior standard deviations (meters) for lever components.
    lever_init : (3,) array-like
        Initial lever-arm estimate (meters).
    rotation_deg : float, optional
        Z-rotation (degrees) applied to lever arms and GPS grid. Default 29.5.
    gps1_offset : (3,) tuple, optional
        (x, y, z) location of GPS1 in ship-fixed coordinates. Default (39.5, 2.2, 15.0).
    downsample : int, optional
        Subsampling stride for plotting. Default 1.
    inset_size : float, optional
        Width and height (in data units) of each zoom-inset. Default 2.0.
    zoom_half_range : float, optional
        Half-size (meters) of the inset view window. Default 0.25.
    tick_step_cm : int, optional
        Tick spacing for inset axes in centimeters. Default 25.
    inset_label_fontsize : int, optional
        Font size for inset axis labels. Default 7.
    schematic_inset : bool, optional
        If True, draw a schematic inset of the full hull. Default True.
    schematic_box : tuple, optional
        Position/size of the schematic inset. Interpreted per *schematic_box_coord*.
    schematic_scale : float, optional
        Multiplier applied to schematic_box width/height. Default 2.0.
    schematic_box_coord : {"axes", "data", "figure"}, optional
        Coordinate system for *schematic_box*. Default "axes".
    side_segments : sequence of tuple, optional
        World-coordinate segments for the side-view schematic:
        [((x1, z1), (x2, z2)), ...]. If None, the panel is left empty with
        axes shown.

    Returns
    -------
    (fig, (ax_top, ax_side)) : tuple
        Matplotlib figure, top axes (top-down plot), and bottom axes (side-view image).
    """

    # Load GPS
    data = np.load(gps_data_path("GPS_Data/Processed_GPS_Receivers_DOG_1.npz"))
    GPS_grid = data["gps1_to_others"]
    GPS_Coordinates = data["GPS_Coordinates"]

    GPS_Vessel = np.zeros_like(GPS_Coordinates)
    for i in range(GPS_Coordinates.shape[0]):
        R_mtrx, d = findRotationAndDisplacement(GPS_Coordinates[i].T, GPS_grid.T)
        GPS_Vessel[i] = (R_mtrx @ GPS_Coordinates[i].T + d[:, None]).T

    GPS1 = np.array(gps1_offset)

    fig = plt.figure(figsize=(8, 9))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2, 1], hspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax_side = fig.add_subplot(gs[1, 0])

    theta = np.deg2rad(rotation_deg)
    Rz = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    lever_init_rot = lever_init @ Rz.T
    levers_rot = np.asarray(lever_arms) @ Rz.T
    lever_xy = levers_rot[:, :2] + GPS1[:2]

    GPS_Vessel_rot = GPS_Vessel @ Rz.T

    gps_xy_list = []
    centroids = []
    for i in range(4):
        gxy_full = GPS_Vessel_rot[:, i, :2] + GPS1[:2]
        gps_xy_list.append(gxy_full[::downsample])
        centroids.append(gxy_full.mean(axis=0))

    white_blue_cmap = LinearSegmentedColormap.from_list("white_blue", ["white", "blue"])
    for pts in gps_xy_list:
        _contour_kde2d(
            ax, pts, levels=50, gridsize=200, cmap=white_blue_cmap, alpha=0.9
        )

    white_red_cmap = LinearSegmentedColormap.from_list("white_red", ["white", "red"])
    lever_xy = lever_xy[::downsample]
    _contour_kde2d(
        ax, lever_xy, levels=50, gridsize=200, cmap=white_red_cmap, alpha=0.9
    )

    for i in range(4):
        gps_i = GPS_Vessel_rot[:, i, :2] + GPS1[:2]  # (n_samples, 2)
        ellipse, pct = compute_error_ellipse(gps_i, confidence=0.68, zorder=3)
        ax.add_patch(ellipse)

    lever_xy = np.column_stack((levers_rot[:, 0] + GPS1[0], levers_rot[:, 1] + GPS1[1]))
    ellipse, pct = compute_error_ellipse(lever_xy, confidence=0.68, zorder=3)
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
        ax.plot([x1, x2], [y1, y2], color="0.3", linewidth=1.2, zorder=10)

    for idx, (pts, c) in enumerate(zip(gps_xy_list, centroids), start=1):
        w = inset_size
        h = inset_size
        cx, cy = float(c[0]), float(c[1])
        if idx == 1:
            dx, dy = 2.0, 2.24
        elif idx == 2:
            dx, dy = 2.0, -0.75
        elif idx == 3:
            dx, dy = -2.0, 1.2
        else:
            dx, dy = -2.0, 2.0
        ix, iy = cx + dx, cy + dy
        axins = ax.inset_axes(
            [ix - w / 2.0, iy - h / 2.0, w, h], transform=ax.transData
        )
        _contour_kde2d(
            axins, pts, levels=30, gridsize=120, cmap=white_blue_cmap, alpha=0.9
        )
        axins.plot([cx], [cy], marker="o", markersize=2, zorder=3)
        axins.set_xlim(cx - zoom_half_range, cx + zoom_half_range)
        axins.set_ylim(cy - zoom_half_range, cy + zoom_half_range)
        rng_m = zoom_half_range
        cm_max = int(np.floor(rng_m * 100.0))
        step = max(1, int(tick_step_cm))
        tick_cm = np.arange(-cm_max, cm_max + 1, step, dtype=int)
        xticks = cx + (tick_cm / 100.0)
        yticks = cy + (tick_cm / 100.0)
        axins.set_xticks(xticks)
        axins.set_yticks(yticks)
        axins.set_xticklabels([f"{v}" for v in tick_cm])
        axins.set_yticklabels([f"{v}" for v in tick_cm])
        axins.tick_params(axis="both", labelsize=inset_label_fontsize)
        axins.set_xlabel("cm", fontsize=inset_label_fontsize)
        axins.set_ylabel("cm", fontsize=inset_label_fontsize)
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

    if schematic_inset:
        _add_schematic_inset(
            ax,
            segments,
            box=schematic_box,
            scale=schematic_scale,
            box_coord=schematic_box_coord,
        )

    ax.set_xlim(20, 44)
    ax.set_ylim(-7.5, 7.5)
    ax.set_xlabel("X (m) – forward")
    ax.set_ylabel("Y (m) – beam")
    ax.set_title("Top-Down View: Lever Arms vs. Ship Hull")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="upper right")
    ax.set_aspect("equal", "box")
    # Bottom panel: side-view schematic from segments
    if side_segments is not None:
        xs, zs = [], []
        for (x1, z1), (x2, z2) in side_segments:
            ax_side.plot([x1, x2], [z1, z2], color="0.5", linewidth=1.0, zorder=2)
            xs.extend([x1, x2])
            zs.extend([z1, z2])
        ax_side.set_xlim(20, 44)
        ax_side.set_ylim(-1.5, 17.5)
    else:
        ax_side.text(
            0.5,
            0.5,
            "No side-view segments provided",
            ha="center",
            va="center",
            transform=ax_side.transAxes,
        )

    ax_side.set_xlabel("X (m) – forward")
    ax_side.set_ylabel("Z (m) – depth")
    return fig, (ax, ax_side)


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

    downsample = 5

    chain = np.load(gps_output_path("mcmc_chain_8-7.npz"))
    levers = chain["lever"][::5000]
    try:
        lever_init = chain["init_lever"]
        lever_prior = chain["prior_lever"]
    except Exception:
        print("Chain has no initial/prior info, using defaults")
        lever_init = np.array([-13.12, 9.7, -15.9])
        lever_prior = np.array([0.3, 0.3, 0.3])

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

    """Side View"""
    side_view_segments = [
        ((94.17609, 442.05618), (89.08673, 397.63989)),
        ((89.08673, 397.63989), (162.03431, 287.52451)),
        ((162.03431, 287.52451), (174.98906, 304.64329)),
        ((174.98906, 304.64329), (116.23002, 383.14291)),
        ((116.23002, 383.14291), (174.52640, 410.13198)),
        ((174.52640, 410.13198), (253.31561, 410.24887)),
        ((253.31561, 410.24887), (253.50490, 349.00075)),
        ((253.50490, 349.00075), (234.32089, 343.28130)),
        ((234.32089, 343.28130), (313.08258, 344.71116)),
        ((313.08258, 344.71116), (294.73265, 348.40498)),
        ((294.73265, 348.40498), (293.89857, 393.20739)),
        ((293.89857, 393.20739), (386.83974, 393.32655)),
        ((386.83974, 393.32655), (410.67081, 370.21041)),
        ((410.67081, 370.21041), (391.12934, 362.46531)),
        ((391.12934, 362.46531), (417.31348, 362.63330)),
        ((417.31348, 362.63330), (419.77881, 303.03027)),
        ((419.77881, 303.03027), (542.03027, 303.03027)),
        ((542.03027, 303.03027), (537.38965, 332.90430)),
        ((537.38965, 332.90430), (583.50586, 331.59912)),
        ((583.50586, 331.59912), (583.21582, 359.44287)),
        ((583.21582, 359.44287), (660.36621, 350.30664)),
        ((660.36621, 350.30664), (658.13348, 376.15309)),
        ((658.13348, 376.15309), (749.01735, 369.05807)),
        ((749.01735, 369.05807), (672.99925, 478.96362)),
        ((672.99925, 478.96362), (268.87378, 478.96362)),
        ((268.87378, 478.96362), (93.10156, 437.16406)),
    ]

    moonpool_segments = [
        ((338.1, 477.1), (350.3, 477.1)),
        ((350.3, 477.1), (350.3, 490)),
        ((350.3, 490), (338.1, 490)),
        ((338.1, 490), (338.1, 477.1)),
    ]

    side_view_scale = 0.0820725
    side_view_segments = px_to_world_segments(
        side_view_segments,
        scale=side_view_scale,
        x_shift_px=0,
        y_shift_px=39.31 / side_view_scale,
        gps1_offset=(0, 0, 0),
        view="side",
        flip_y=True,
    )
    moonpool_segments = px_to_world_segments(
        moonpool_segments,
        scale=side_view_scale,
        x_shift_px=0,
        y_shift_px=39.31 / side_view_scale,
        gps1_offset=(0, 0, 0),
        view="side",
        flip_y=True,
    )

    # Build side-view segments for the bottom panel (side schematic)
    side_segments = np.concatenate([side_view_segments, moonpool_segments])

    # Make side segments manually
    side_segments = [()]

    fig, (ax, ax_side) = RV_Plot(
        segments,
        levers,
        lever_prior,
        lever_init,
        downsample=downsample,
        side_segments=side_segments,
    )
    plt.show()
