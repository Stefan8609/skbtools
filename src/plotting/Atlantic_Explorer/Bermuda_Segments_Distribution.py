import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from plotting.Ellipses.Error_Ellipse import compute_error_ellipse
from plotting.Ellipses.Prior_Ellipse import plot_prior_ellipse

from geometry.rigid_body import findRotationAndDisplacement
from data import gps_output_path, gps_data_path


def px_to_world_segments(
    segments_px,
    scale,
    x_shift_px=0,
    y_shift_px=0,
    gps1_offset=(0, 0, 0),
    view="side",
    flip_y=False,
):
    """
    Convert pixel‐coordinates to the same meter‐coordinates as your lever arms.

    - scale: m/pixel
    - x_shift_px, y_shift_px: the same center‐subtractions you used before
    - gps1_offset: (x,y,z) in meters
    - view: 'side' or 'top' → decide which pixel‐axis maps to world‐axis
    - flip_y: if True, flips the segment vertically over the bisecting line
    """
    ox, oy, oz = gps1_offset
    world = []
    for (x1, y1), (x2, y2) in segments_px:
        # first subtract your pixel‐shifts:
        xp1, yp1 = x1 - x_shift_px, y1 - y_shift_px
        xp2, yp2 = x2 - x_shift_px, y2 - y_shift_px

        if flip_y:
            yp1 *= -1
            yp2 *= -1

        # scale into meters:
        xm1, ym1 = xp1 * scale, yp1 * scale
        xm2, ym2 = xp2 * scale, yp2 * scale

        if view == "side":
            # pixel‐Y → world‐Z, pixel‐X → world‐X
            p1 = (xm1 + ox, ym1 + oz)
            p2 = (xm2 + ox, ym2 + oz)
        else:
            # top‐down: pixel‐Y → world‐Y (beam)
            p1 = (xm1 + ox, ym1 + oy)
            p2 = (xm2 + ox, ym2 + oy)
        world.append((p1, p2))
    return world


def plot_2d_projection_topdown(
    segments,
    lever_arms,
    lever_prior,
    lever_init,
    rotation_deg=29.5,
    gps1_offset=(39.5, 2.2, 15.0),
    downsample=1,
):
    """
    Plot 2D top-down projection of lever‐arms and GPS distributions against an
    approximate box‐hull

    Parameters
    ----------
    lever_arms : (N,3) array‐like
        X, Y, Z lever‐arm positions (in meters) in ship‐fixed coordinates.
    rotation_deg : float, optional
        Rotation angle about the Z‐axis (in degrees) to match the 3D plot’s
        orientation. Default is 29.5.
    gps1_offset : tuple of 3 floats, optional
        (x, y, z) offset of the GPS1 reference point in ship‐fixed coords.
        Default is (37, 8.3, 15.0).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes3D objects
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

    # Set up figure
    fig, ax = plt.subplots(figsize=(8, 6))

    for (x1, y1), (x2, y2) in segments:
        # Draw the line segment
        plt.plot([x1, x2], [y1, y2], color="k", linewidth=2, zorder=2)

    # Build rotation matrix about Z
    theta = np.deg2rad(rotation_deg)
    Rz = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    # Rotate & translate lever arms
    lever_init_rot = lever_init @ Rz.T
    levers_rot = np.asarray(lever_arms) @ Rz.T
    lever_xy = levers_rot[:, :2] + GPS1[:2]

    # Rotate GPS grid
    GPS_Vessel_rot = GPS_Vessel @ Rz.T

    # 2D KDE of GPS distributions
    white_blue_cmap = LinearSegmentedColormap.from_list("white_blue", ["white", "blue"])
    for i in range(4):
        # Compute KDE for this GPS unit
        gps_xy = GPS_Vessel_rot[:, i, :2] + GPS1[:2]  # Apply GPS1 shift
        gps_xy = gps_xy[::downsample]  # Downsample for performance
        # KDE
        kde = gaussian_kde(gps_xy.T)
        xgrid, ygrid = np.meshgrid(
            np.linspace(gps_xy[:, 0].min(), gps_xy[:, 0].max(), 200),
            np.linspace(gps_xy[:, 1].min(), gps_xy[:, 1].max(), 200),
        )
        xy_coords = np.vstack([xgrid.ravel(), ygrid.ravel()])
        density = kde(xy_coords).reshape(xgrid.shape)
        # Plot density as filled contour
        ax.contourf(
            xgrid,
            ygrid,
            density,
            levels=50,
            cmap=white_blue_cmap,
            alpha=1,
            zorder=1,
        )

    # 2D KDE of lever arms
    white_red_cmap = LinearSegmentedColormap.from_list("white_red", ["white", "red"])
    lever_xy = lever_xy[::downsample]
    kde = gaussian_kde(lever_xy.T)
    xgrid, ygrid = np.meshgrid(
        np.linspace(lever_xy[:, 0].min(), lever_xy[:, 0].max(), 200),
        np.linspace(lever_xy[:, 1].min(), lever_xy[:, 1].max(), 200),
    )
    grid_coords = np.vstack([xgrid.ravel(), ygrid.ravel()])
    density = kde(grid_coords).reshape(xgrid.shape)

    # Plot the lever density as filled contours
    ax.contourf(
        xgrid,
        ygrid,
        density,
        levels=50,
        cmap=white_red_cmap,
        alpha=1,
        zorder=1,
    )

    # 68% error ellipse for GPS distributions
    for i in range(4):
        gps_i = GPS_Vessel_rot[:, i, :2] + GPS1[:2]  # (n_samples, 2)
        ellipse, pct = compute_error_ellipse(gps_i, confidence=0.68, zorder=3)
        ax.add_patch(ellipse)
        ax.text(
            6,
            20 - 2 * i,
            f"Points within GPS {i + 1} ellipse: {pct:.1f}%",
            color="black",
            fontsize=10,
            ha="center",
            va="center",
        )

    # 68% error ellipse for lever-arm cloud
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
    ax.text(
        9.13,
        12,
        f"Points within lever ellipse percentage: {pct:.1f}%",
        color="black",
        fontsize=10,
        ha="center",
        va="center",
    )

    # Labels & styling
    ax.set_xlabel("X (m) – forward")
    ax.set_ylabel("Y (m) – depth")
    ax.set_title("Top-Down View: Lever Arms vs. Ship Hull")
    ax.legend(loc="upper right")
    ax.set_aspect("equal", "box")

    plt.tight_layout()
    return fig, ax


def plot_2d_projection_side(
    segments,
    lever_arms,
    lever_prior,
    lever_init,
    rotation_deg=29.5,
    gps1_offset=(44.55, 2.2, 15.4),
    downsample=1,
):
    """
    Plot 2D top-down projection of lever‐arms and GPS distributions against an
    approximate box‐hull

    Parameters
    ----------
    lever_arms : (N,3) array‐like
        X, Y, Z lever‐arm positions (in meters) in ship‐fixed coordinates.
    rotation_deg : float, optional
        Rotation angle about the Z‐axis (in degrees) to match the 3D plot’s
        orientation. Default is 29.5.
    gps1_offset : tuple of 3 floats, optional
        (x, y, z) offset of the GPS1 reference point in ship‐fixed coords.
        Default is (37, 8.3, 15.0).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes3D objects
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

    # Set up figure
    fig, ax = plt.subplots(figsize=(8, 6))

    for (x1, y1), (x2, y2) in segments:
        # Draw the line segment
        plt.plot([x1, x2], [y1, y2], color="k", linewidth=2, zorder=2)

    # Build rotation matrix about Z
    theta = np.deg2rad(rotation_deg)
    Rz = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    # Rotate & translate lever arms
    lever_init_rot = lever_init @ Rz.T
    levers_rot = np.asarray(lever_arms) @ Rz.T
    levers_xz = levers_rot[:, [0, 2]] + GPS1[[0, 2]]

    # Rotate GPS grid
    GPS_Vessel_rot = GPS_Vessel @ Rz.T

    # 2D KDE of GPS distributions
    white_blue_cmap = LinearSegmentedColormap.from_list("white_blue", ["white", "blue"])
    for i in range(4):
        # Compute KDE for this GPS unit
        gps_xz = GPS_Vessel_rot[:, i, [0, 2]] + GPS1[[0, 2]]
        gps_xz = gps_xz[::downsample]  # Downsample for performance

        # KDE
        kde = gaussian_kde(gps_xz.T)
        xgrid, zgrid = np.meshgrid(
            np.linspace(gps_xz[:, 0].min(), gps_xz[:, 0].max(), 200),
            np.linspace(gps_xz[:, 1].min(), gps_xz[:, 1].max(), 200),
        )
        xz_coords = np.vstack([xgrid.ravel(), zgrid.ravel()])
        density = kde(xz_coords).reshape(xgrid.shape)
        # Plot density as filled contour
        ax.contourf(
            xgrid,
            zgrid,
            density,
            levels=50,
            cmap=white_blue_cmap,
            alpha=1,
            zorder=1,
        )

    # 2D KDE of lever arms
    white_red_cmap = LinearSegmentedColormap.from_list("white_red", ["white", "red"])
    kde = gaussian_kde(levers_xz.T)
    xgrid, zgrid = np.meshgrid(
        np.linspace(levers_xz[:, 0].min(), levers_xz[:, 0].max(), 200),
        np.linspace(levers_xz[:, 1].min(), levers_xz[:, 1].max(), 200),
    )
    grid_coords = np.vstack([xgrid.ravel(), zgrid.ravel()])
    density = kde(grid_coords).reshape(xgrid.shape)

    # Plot the lever density as filled contours
    ax.contourf(
        xgrid,
        zgrid,
        density,
        levels=50,
        cmap=white_red_cmap,
        alpha=1,
        zorder=1,
    )

    # 68% error ellipse for GPS distributions
    for i in range(4):
        gps_i = GPS_Vessel_rot[:, i, :]  # (n_samples, 3)
        gps_xz = gps_i[:, [0, 2]] + GPS1[[0, 2]]  # (n_samples, 2)
        ellipse, pct = compute_error_ellipse(gps_xz, confidence=0.68, zorder=3)
        ax.add_patch(ellipse)
        ax.text(
            6,
            30 - 2 * i,
            f"Points within GPS {i + 1} ellipse: {pct:.1f}%",
            color="black",
            fontsize=10,
            ha="center",
            va="center",
        )

    # 68% error ellipse for lever-arm cloud
    levers_xz = levers_rot[:, [0, 2]] + GPS1[[0, 2]]
    levers_xz = levers_xz[::downsample]

    ellipse, pct = compute_error_ellipse(levers_xz, confidence=0.68, zorder=3)
    prior_ellipse = plot_prior_ellipse(
        mean=lever_init_rot[[0, 2]] + GPS1[[0, 2]],
        cov=np.diag(lever_prior[[0, 2]] ** 2),
        confidence=0.68,
        zorder=3,
    )
    ax.add_patch(ellipse)
    ax.add_patch(prior_ellipse)
    # annotate percentage
    ax.text(
        9.13,
        22,
        f"Points within lever ellipse percentage: {pct:.1f}%",
        color="black",
        fontsize=10,
        ha="center",
        va="center",
    )

    # Labels & styling
    ax.set_xlabel("X (m) – forward")
    ax.set_ylabel("Y (m) – depth")
    ax.set_title("Top-Down View: Lever Arms vs. Ship Hull")
    ax.legend(loc="upper right")
    ax.set_aspect("equal", "box")

    plt.tight_layout()
    return fig, ax


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
        ((368.54864253, 251.66402715), (368.54864253, 123.49660633)),
        ((368.54864253, 123.49660633), (411.01757812, 123.49660633)),
        ((411.01757812, 123.49660633), (411.01282051, 136.22322775)),
        ((411.01282051, 136.22322775), (457.07006836, 136.09863281)),
        ((457.01809955, 136.14253394), (457.01809955, 123.02941176)),
        ((457.01293945, 123.24609375), (543.94775391, 123.05761719)),
        ((544.12707391, 123.10558069), (543.90422323, 208.97737557)),
        ((544.04931641, 208.38769531), (418.03613281, 208.11523437)),
        ((418.03959276, 207.95927602), (418.03959276, 251.19457014)),
        ((418.04736328, 251.07128906), (368.03431373, 251.04901961)),
    ]

    ship_segments = [
        ((47.18438914, 243.84615385), (47.18438914, 67.09863281)),
        ((53.02148437, 67.09863281), (172.00781250, 51.22558594)),
        ((172.00781250, 51.41308594), (815.26562500, 51.41308594)),
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

    downsample = 1

    chain = np.load(gps_output_path("mcmc_chain_moonpool_better.npz"))
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
    fig, ax = plot_2d_projection_topdown(
        segments, levers, lever_prior, lever_init, downsample=downsample
    )
    plt.show()

    fig, ax = plot_2d_projection_topdown(
        segments, levers, lever_prior, lever_init, downsample=downsample
    )
    ax.set_xlim(22, 41)
    ax.set_ylim(-5.5, 5.5)
    plt.show()

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

    railing_segments = [
        ((420.39501953, 302.03784180), (420.39501953, 290.79882812)),
        ((420.39501953, 290.79882812), (542.09057617, 290.79882812)),
        ((542.09057617, 290.79882812), (542.09057617, 302.03784180)),
        ((542.09057617, 302.03784180), (420.39501953, 302.03784180)),
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
    railing_segments = px_to_world_segments(
        railing_segments,
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
    segments = np.concatenate([side_view_segments, railing_segments, moonpool_segments])

    fig, ax = plot_2d_projection_side(
        segments, levers, lever_prior, lever_init, downsample=downsample
    )
    plt.show()


"""
Ratio between the posterior and the prior (resolution)
How much information we can add to our prior belief

Make the plot from the screenshot (x,y and chi,z) for combined
ESV and for each ESV segment individually (compare these plots)
    We can overlay the segment plots to see where the CDOG lies in each


FIX ELLIPSES IN KDE_MCMC

Transdimensional MCMC for the ESV bias (in each segment)
    Updates are dependent on depth (2 or 3 diff updates)
    Piecewise updates

Azimuthal view for the ESV bias segments...

Fix labeling of DOGS in split esv plot (and indexing of segments)
"""
