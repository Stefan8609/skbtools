import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import chi2, gaussian_kde

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
    lever_arms, rotation_deg=29.5, gps1_offset=(39.5, 2.2, 15.0)
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
        plt.plot([x1, x2], [y1, y2], color="k", linewidth=2)

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
    levers_rot = np.asarray(lever_arms) @ Rz.T
    lever_xy = levers_rot[:, :2] + GPS1[:2]

    # Rotate GPS grid
    GPS_Vessel_rot = GPS_Vessel @ Rz.T

    # 2D KDE of GPS distributions
    white_blue_cmap = LinearSegmentedColormap.from_list("white_blue", ["white", "blue"])
    for i in range(4):
        # Compute KDE for this GPS unit
        gps_xy = GPS_Vessel_rot[::50, i, :2] + GPS1[:2]  # Apply GPS1 shift

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
        )

    # 2D KDE of lever arms
    white_red_cmap = LinearSegmentedColormap.from_list("white_red", ["white", "red"])
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
    )

    # 68% error ellipse for GPS distributions
    for i in range(4):
        gps_i = GPS_Vessel_rot[:, i, :]  # (n_samples, 3)
        gps_xy = gps_i[:, :2] + GPS1[:2]  # (n_samples, 2)

        cov = np.cov(gps_xy, rowvar=False)
        mean = gps_xy.mean(axis=0)

        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2 * np.sqrt(vals * chi2.ppf(0.68, 2))

        ellipse = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            edgecolor="black",
            fc="none",
            lw=1,
        )
        ax.add_patch(ellipse)

        # Mahalanobis distance for coverage
        inv_cov = np.linalg.inv(cov)
        diffs = gps_xy - mean
        d2 = np.einsum("nj,jk,nk->n", diffs, inv_cov, diffs)
        pct = np.mean(d2 <= chi2.ppf(0.68, 2)) * 100

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
    cov_l = np.cov(lever_xy, rowvar=False)
    mean_l = lever_xy.mean(axis=0)
    vals_l, vecs_l = np.linalg.eigh(cov_l)
    order = vals_l.argsort()[::-1]
    vals_l, vecs_l = vals_l[order], vecs_l[:, order]
    angle_l = np.degrees(np.arctan2(vecs_l[1, 0], vecs_l[0, 0]))
    width_l, height_l = 2 * np.sqrt(vals_l * chi2.ppf(0.95, 2))
    ellipse_l = Ellipse(
        xy=mean_l,
        width=width_l,
        height=height_l,
        angle=angle_l,
        edgecolor="black",
        fc="none",
        lw=1,
    )
    ax.add_patch(ellipse_l)
    inv_cov_l = np.linalg.inv(cov_l)
    diffs = lever_xy - mean_l
    d2 = np.einsum("nj,jk,nk->n", diffs, inv_cov_l, diffs)
    pct = np.mean(d2 <= chi2.ppf(0.68, 2)) * 100
    # annotate percentage
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


if __name__ == "__main__":
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

    chain = np.load(gps_output_path("mcmc_chain_adroit_5_test_xy_lever.npz"))
    levers = chain["lever"][::5000]

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
    fig, ax = plot_2d_projection_topdown(levers)
    plt.show()

    fig, ax = plot_2d_projection_topdown(levers)
    ax.set_xlim(22, 41)
    ax.set_ylim(-5.5, 5.5)
    plt.show()
