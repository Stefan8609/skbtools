import numpy as np
import matplotlib.pyplot as plt


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

    # apply rotation
    levers = lever_arms @ Rz.T
    print(levers)

    GPS1 = [37, 8.3, 15.0]
    ax.scatter(
        GPS1[0] + levers[:, 0],
        GPS1[1] + levers[:, 1],
        GPS1[2] + levers[:, 2],
        color="red",
        s=10,
        alpha=0.5,
        label="Lever Arms",
    )
    ax.scatter(GPS1[0], GPS1[1], GPS1[2], color="blue", s=50, label="GPS1")

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

    chain = np.load(gps_output_path("mcmc_chain_adroit_1.npz"))

    levers = chain["lever"][::5000]

    # Plot the lever arms
    fig, ax = plot_lever_arms_3d(levers)
    plt.show()
