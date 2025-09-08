import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from data import gps_data_path


def plot_2d_projection_side(
    lever_arms,
    hull_dims=(52.0, 12.0, 15.24),
    hull_origin=(0.0, 0.0, 0.0),
    rotation_deg=29.5,
    gps1_offset=(36.7, 8.1, 15.24),
    grid_barycenter=None,
    background_image_path="Figs/Images/Bermuda_side.png",
    bg_alpha=0.7,
):
    """
    Plot 2D side projection of lever‐arms and GPS distributions against an
    approximate box‐hull

    Parameters
    ----------
    lever_arms : (N,3) array‐like
        X, Y, Z lever‐arm positions (in meters) in ship‐fixed coordinates.
    hull_dims : tuple of 3 floats, optional
        (length, beam, height) of the hull box in the same units as lever_arms.
        Default is (52.0, 12.0, 15.24).
    hull_origin : tuple of 3 floats, optional
        (x0, y0, z0) coordinates of the hull‐box corner
        (e.g. the baseline & stern origin). Default is (0.0, 0.0, 0.0).
    rotation_deg : float, optional
        Rotation angle about the Z‐axis (in degrees) to match the 3D plot’s
        orientation. Default is 29.5.
    gps1_offset : tuple of 3 floats, optional
        (x, y, z) offset of the GPS1 reference point in ship‐fixed coords.
        Default is (37, 8.3, 15.0).
    grid_barycenter : (3,) array‐like, optional
        Barycenter shift applied to the GPS grid, loaded from file by default.
    background_image_path : str, optional
        Path to plan‐view image to use as background.
    bg_alpha : float, optional
        Transparency for background image (0.0–1.0). Default is 0.3.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes3D objects
    """

    if grid_barycenter is None:
        grid_barycenter = np.array([-5.79399176, 0.0, 0.01600236])

    data = np.load(gps_data_path("GPS_Data/GPS_grid_all_barycenter_focused.npz"))
    GPS_grid = data["GPS_grid"]

    # Unpack
    L, _, H = hull_dims
    x0, _, z0 = hull_origin
    GPS1 = np.array(gps1_offset)
    gb = grid_barycenter

    # Build the 4 bottom-face corners (no rotation)
    corners = np.array(
        [
            [0, 0, 0],
            [L, 0, 0],
            [L, 0, H],
            [0, 0, H],
        ]
    ) + np.array(hull_origin)

    # Set up figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # overlay background image if provided
    if background_image_path:
        img = plt.imread(gps_data_path(background_image_path))
        img = np.flipud(img)
        # map image to the hull‐box extent
        ax.imshow(
            img,
            extent=[x0, x0 + L + 1.2, z0, z0 + H + 6.4],
            origin="lower",
            aspect="equal",
            alpha=bg_alpha,
            zorder=0,
        )

    # Draw hull edges exactly as in 3D (axis-aligned)
    bottom_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for i, j in bottom_edges:
        xi = [corners[i][0], corners[j][0]]
        zi = [corners[i][2], corners[j][2]]
        ax.plot(xi, zi, color="black", linewidth=0.1)

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

    # Rotate GPS grid
    GPS_grid_rot = GPS_grid @ Rz.T
    gb = gb @ Rz.T

    GPS_Coords = GPS_grid_rot.reshape(-1, 3)
    ax.scatter(
        GPS_Coords[::10, 0] + GPS1[0] + gb[0],
        GPS_Coords[::10, 2] + GPS1[2] + gb[2],
        color="blue",
        s=10,
        alpha=0.05,
        label="GPS Distributions",
    )

    # Plot lever-arm points
    ax.scatter(
        levers_rot[:, 0] + GPS1[0],
        levers_rot[:, 2] + GPS1[2],
        color="red",
        s=10,
        alpha=0.05,
        label="Lever Arms",
    )

    # 95% error ellipse for GPS distributions
    for i in range(1, 4):
        gps_i = GPS_grid_rot[i]
        gps_xz = np.column_stack(
            (gps_i[:, 0] + GPS1[0] + gb[0], gps_i[:, 2] + GPS1[2] + gb[2])
        )
        cov = np.cov(gps_xz, rowvar=False)
        mean = gps_xz.mean(axis=0)
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
        # Mahalanobis distances
        inv_cov = np.linalg.inv(cov)
        diffs = gps_xz - mean
        d2 = np.einsum("nj,jk,nk->n", diffs, inv_cov, diffs)
        pct = np.mean(d2 <= chi2.ppf(0.68, 2)) * 100
        # annotate percentage
        ax.text(
            1,
            32 - 2 * i,
            f"Points within GPS {i + 1} ellipse percentage: {pct:.1f}%",
            color="black",
            fontsize=10,
            ha="center",
            va="center",
        )

    # 95% error ellipse for lever-arm cloud
    lever_xz = np.column_stack((levers_rot[:, 0] + GPS1[0], levers_rot[:, 2] + GPS1[2]))
    cov_l = np.cov(lever_xz, rowvar=False)
    mean_l = lever_xz.mean(axis=0)
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
    diffs = lever_xz - mean_l
    d2 = np.einsum("nj,jk,nk->n", diffs, inv_cov_l, diffs)
    pct = np.mean(d2 <= chi2.ppf(0.68, 2)) * 100
    # annotate percentage
    ax.text(
        1,
        24,
        f"Points within lever ellipse percentage: {pct:.1f}%",
        color="black",
        fontsize=10,
        ha="center",
        va="center",
    )

    # Labels & styling
    ax.set_xlabel("X (m) – forward")
    ax.set_ylabel("Y (m) – starboard")
    ax.set_title("Side View: Lever Arms vs. Ship Hull")
    ax.legend(loc="upper right")
    ax.set_aspect("equal", "box")

    # Auto-scale with a little padding
    pad = max(L, H) * 0.1
    ax.set_xlim(-pad, L + pad)
    ax.set_ylim(-pad, H + pad)

    plt.tight_layout()
    return fig, ax


def plot_2d_projection_topdown(
    lever_arms,
    hull_dims=(52.0, 12.0, 15.24),
    hull_origin=(0.0, 0.0, 0.0),
    rotation_deg=29.5,
    gps1_offset=(36.7, 8.1, 15.0),
    grid_barycenter=None,
    background_image_path="Figs/Images/Bermuda_top_down.png",
    bg_alpha=0.7,
):
    """
    Plot 2D top-down projection of lever‐arms and GPS distributions against an
    approximate box‐hull

    Parameters
    ----------
    lever_arms : (N,3) array‐like
        X, Y, Z lever‐arm positions (in meters) in ship‐fixed coordinates.
    hull_dims : tuple of 3 floats, optional
        (length, beam, height) of the hull box in the same units as lever_arms.
        Default is (52.0, 12.0, 15.24).
    hull_origin : tuple of 3 floats, optional
        (x0, y0, z0) coordinates of the hull‐box corner
        (e.g. the baseline & stern origin). Default is (0.0, 0.0, 0.0).
    rotation_deg : float, optional
        Rotation angle about the Z‐axis (in degrees) to match the 3D plot’s
        orientation. Default is 29.5.
    gps1_offset : tuple of 3 floats, optional
        (x, y, z) offset of the GPS1 reference point in ship‐fixed coords.
        Default is (37, 8.3, 15.0).
    grid_barycenter : (3,) array‐like, optional
        Barycenter shift applied to the GPS grid, loaded from file by default.
    background_image_path : str, optional
        Path to plan‐view image to use as background.
    bg_alpha : float, optional
        Transparency for background image (0.0–1.0). Default is 0.3.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes3D objects
    """

    if grid_barycenter is None:
        grid_barycenter = np.array([-5.79399176, 0.0, 0.01600236])

    data = np.load(gps_data_path("GPS_Data/GPS_grid_all_barycenter_focused.npz"))
    GPS_grid = data["GPS_grid"]

    # Unpack
    L, B, _ = hull_dims
    x0, y0, _ = hull_origin
    GPS1 = np.array(gps1_offset)
    gb = grid_barycenter

    # Build the 4 bottom-face corners (no rotation)
    corners = np.array(
        [
            [0, 0, 0],
            [L, 0, 0],
            [L, B, 0],
            [0, B, 0],
        ]
    ) + np.array(hull_origin)

    # Set up figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # overlay background image if provided
    if background_image_path:
        img = plt.imread(gps_data_path(background_image_path))
        img = np.flipud(img)
        # map image to the hull‐box extent
        ax.imshow(
            img,
            extent=[x0, x0 + L, y0, y0 + B],
            origin="lower",
            aspect="equal",
            alpha=bg_alpha,
            zorder=0,
        )

    # Draw hull edges exactly as in 3D (axis-aligned)
    bottom_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for i, j in bottom_edges:
        xi = [corners[i][0], corners[j][0]]
        yi = [corners[i][1], corners[j][1]]
        ax.plot(xi, yi, color="black", linewidth=0.1)

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
    GPS_grid_rot = GPS_grid @ Rz.T
    gb = gb @ Rz.T

    GPS_Coords = GPS_grid_rot.reshape(-1, 3)
    ax.scatter(
        GPS_Coords[::10, 0] + GPS1[0] + gb[0],
        GPS_Coords[::10, 1] + GPS1[1] + gb[1],
        color="blue",
        s=10,
        alpha=0.05,
        label="GPS Distributions",
    )

    # Plot lever-arm points
    ax.scatter(
        lever_xy[:, 0],
        lever_xy[:, 1],
        color="red",
        s=10,
        alpha=0.05,
        label="Lever Arms",
    )

    # 95% error ellipse for GPS distributions
    for i in range(1, 4):
        gps_i = GPS_grid_rot[i]
        gps_xy = np.column_stack(
            (gps_i[:, 0] + GPS1[0] + gb[0], gps_i[:, 1] + GPS1[1] + gb[1])
        )
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
        # Mahalanobis distances
        inv_cov = np.linalg.inv(cov)
        diffs = gps_xy - mean
        d2 = np.einsum("nj,jk,nk->n", diffs, inv_cov, diffs)
        pct = np.mean(d2 <= chi2.ppf(0.68, 2)) * 100
        # annotate percentage
        ax.text(
            1,
            32 - 2 * i,
            f"Points within GPS {i + 1} ellipse percentage: {pct:.1f}%",
            color="black",
            fontsize=10,
            ha="center",
            va="center",
        )

    # 95% error ellipse for lever-arm cloud
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
        1,
        24,
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

    # Auto-scale with a little padding
    all_x = np.concatenate([lever_xy[:, 0], corners[:, 0]])
    all_y = np.concatenate([lever_xy[:, 1], corners[:, 1]])
    pad = max(L, B) * 0.1
    ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax.set_ylim(all_y.min() - pad, all_y.max() + pad)

    plt.tight_layout()
    return fig, ax


def plot_lever_arms_3d(
    lever_arms,
    hull_dims=(52.0, 12.0, 15.24),
    hull_origin=(0.0, 0.0, 0.0),
    elev=20,
    azim=-60,
):
    """
    Plot 3D distribution of lever‐arms against an approximate box‐hull.

    Parameters
    ----------
    lever_arms : (N,3) array‐like
        X, Y, Z lever‐arm positions (in meters) in ship‐fixed coordinates.
    hull_dims : tuple of 3 floats, optional
        (length, beam, height) of the hull box in the same units as lever_arms.
        Default is (60 m, 12 m, 15 m).
    hull_origin : tuple of 3 floats, optional
        (x0, y0, z0) coordinates of the hull‐box corner
        (e.g. the baseline & stern origin).
    elev : float, optional
        Elevation angle (in degrees) for the 3D view.
    azim : float, optional
        Azimuth angle (in degrees) for the 3D view.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes3D objects
    """
    # Load GPS data
    data = np.load(gps_data_path("GPS_Data/GPS_grid_all_barycenter_focused.npz"))
    GPS_grid = data["GPS_grid"]
    grid_barycenter = np.array([-5.79399176, 0.0, 0.01600236])

    L, B, H = hull_dims
    x0, y0, z0 = hull_origin

    # Build the 8 corners of the box
    corners = np.array(
        [
            [0, 0, 0],
            [L, 0, 0],
            [L, B, 0],
            [0, B, 0],
            [0, 0, H],
            [L, 0, H],
            [L, B, H],
            [0, B, H],
        ]
    ) + np.array(hull_origin)

    # Define the 12 edges by pairs of corner‐indices
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # lower rectangle
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # upper rectangle
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # verticals
    ]

    # Set up figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    # Plot hull edges
    for i, j in edges:
        xs, ys, zs = zip(corners[i], corners[j])
        ax.plot(xs, ys, zs, color="black", linewidth=5)

    # Plot lever‐arm points
    lever_arms = np.asarray(lever_arms)

    # Rotate lever arms to match the hull orientation
    theta = np.deg2rad(29.5)
    Rz = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    # apply rotation to lever arms and GPS grid
    levers = lever_arms @ Rz.T
    GPS_grid = GPS_grid @ Rz.T
    grid_barycenter = grid_barycenter @ Rz.T
    print(levers)

    # Plot GPS distributions
    GPS1 = [36.7, 8.1, 15.0]
    GPS_Coords = GPS_grid.reshape(-1, 3)
    ax.scatter(
        GPS_Coords[::10, 0] + GPS1[0] + grid_barycenter[0],
        GPS_Coords[::10, 1] + GPS1[1] + grid_barycenter[1],
        GPS_Coords[::10, 2] + GPS1[2] + grid_barycenter[2],
        color="blue",
        s=10,
        alpha=0.05,
        label="GPS Distributions",
    )

    # Plot lever distribution
    ax.scatter(
        GPS1[0] + levers[:, 0],
        GPS1[1] + levers[:, 1],
        GPS1[2] + levers[:, 2],
        color="red",
        s=10,
        alpha=0.05,
        label="Lever Arms",
    )

    # Axes labels
    ax.set_xlabel("X (m) – forward")
    ax.set_ylabel("Y (m) – starboard")
    ax.set_zlabel("Z (m) – up")

    ax.set_box_aspect((L, B, H))

    ax.set_title("Lever‐Arms vs. Ship Hull")
    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # Example lever arms in ship-fixed coordinates
    from data import gps_output_path

    chain = np.load(gps_output_path("mcmc_chain_8-7.npz"))

    levers = chain["lever"][::5000]

    # Plot the lever arms
    # fig, ax = plot_lever_arms_3d(levers)
    # fig, ax = plot_2d_projection_topdown(levers)
    fig, ax = plot_2d_projection_side(levers)
    plt.show()
