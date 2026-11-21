import numpy as np
import matplotlib.pyplot as plt
from geometry.fit_plane import fitPlane
from geometry.project_to_plane import projectToPlane
from bermuda_data import GPS1_TO_OTHERS, gps_data_path
from geometry.rigid_body import findRotationAndDisplacement

np.set_printoptions(suppress=True)


def GPS_Lever_arms(GPS_Coordinates):
    """Receiver positions (4, N, 3) relative to GPS1 in the plane of the grid.

    This is how ``gps1_to_others`` was derived (vessel frame, m). Prints and plots the
    mean grid and the angle of the GPS1-GPS2 normal.
    """
    all_arms = np.zeros((4, len(GPS_Coordinates), 3))
    xaxes = []
    for i in range(len(GPS_Coordinates)):
        xs, ys, zs = GPS_Coordinates[i].T
        points = np.array([xs, ys, zs]).T

        # Get barycenter and normal vector defining plane
        barycenter = np.mean(points, axis=0, keepdims=True)
        normVect = fitPlane(xs, ys, zs)

        # Check that normal vector is oriented on the
        # same side of the plane at each time and flip if not
        if np.dot(np.array([0, 0, 1]), normVect) < 0:
            normVect *= -1

        # Get x-axis defined by projection of vector
        # from barycenter to GPS1 onto the plane
        xaxis = points[0] - barycenter
        xaxis = projectToPlane(xaxis, normVect)[0]
        xaxis /= np.linalg.norm(xaxis)

        xaxes.append(xaxis)

        # Get y-axis defined by cross product
        # of x-axis and plane normal vector
        yaxis = np.cross(normVect, xaxis)
        yaxis /= np.linalg.norm(yaxis)

        # z-axis is defined by the normal vector of the plane
        zaxis = normVect / np.linalg.norm(normVect)

        # Project vector between GPS1 and other GPS onto the axis
        for j in range(1, len(GPS_Coordinates[i])):
            vect = points[j] - points[0]
            xdist = np.dot(vect, xaxis)
            ydist = np.dot(vect, yaxis)
            zdist = np.dot(vect, zaxis)

            lever_arm = np.array([xdist, ydist, zdist])
            all_arms[j, i] = lever_arm
    grid = np.zeros((4, 3))
    for i in range(4):
        grid[i] = np.mean(all_arms[i], axis=0, keepdims=True)[0]
        plt.scatter(grid[i, 0], grid[i, 1], label=f"GPS{i + 1}")
    print(np.mean(grid, axis=0))
    print(grid)

    v12 = grid[1] - grid[0]
    n12 = np.array([-v12[1], v12[0], 0.0])
    n12 /= np.linalg.norm(n12)

    # Print angle between [1,0] and normal vector between GPS1 and GPS2
    x_unit = np.array([1.0, 0.0, 0.0])
    cos_theta = np.clip(np.dot(x_unit, n12), -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    print(f"Angle between [1,0] and normal: {angle_deg:.2f}°")

    midpoint = (grid[0] + grid[1]) / 2
    plt.arrow(
        midpoint[0],
        midpoint[1],
        n12[0],
        n12[1],
        head_width=0.1,
        length_includes_head=True,
        color="k",
        label="Normal(1→2)",
    )
    plt.legend()
    plt.axis("equal")
    plt.show()
    save_path = gps_data_path("GPS_Data/GPS_grid_all.npz")
    if save_path.parent.exists():
        np.savez(save_path, GPS_grid=all_arms)


def fit_rigid_grid(GPS_Coordinates, frame_grid, n_iterations=10):
    """Rigid receiver grid (4, 3) best fitting GPS_Coordinates (N, 4, 3), in m.

    Every epoch is rotated onto the current grid and the grid replaced by the
    per-receiver median of the aligned epochs, until it stops changing. The result
    is then rotated onto ``frame_grid`` (so the vessel-frame axes, and with them the
    lever arm, keep their meaning) and shifted to put receiver 1 at the origin.
    Returns ``(grid, misfit)`` with misfit (N, 4) the distance (m) of each receiver
    from the fitted grid.
    """
    GPS_Coordinates = np.asarray(GPS_Coordinates, dtype=float)
    if GPS_Coordinates.ndim != 3 or GPS_Coordinates.shape[1:] != (4, 3):
        raise ValueError(f"expected (N, 4, 3), got {GPS_Coordinates.shape}")
    grid = np.asarray(frame_grid, dtype=float)
    aligned = np.empty_like(GPS_Coordinates)
    for _ in range(n_iterations):
        for i, epoch in enumerate(GPS_Coordinates):
            R, d = findRotationAndDisplacement(epoch.T, grid.T)
            aligned[i] = (R @ epoch.T + d[:, None]).T
        median = np.median(aligned, axis=0)
        R, d = findRotationAndDisplacement(median.T, np.asarray(frame_grid).T)
        new_grid = (R @ median.T + d[:, None]).T
        new_grid -= new_grid[0]
        converged = np.max(np.abs(new_grid - grid)) < 1e-6
        grid = new_grid
        if converged:
            break
    for i, epoch in enumerate(GPS_Coordinates):
        R, d = findRotationAndDisplacement(epoch.T, grid.T)
        aligned[i] = (R @ epoch.T + d[:, None]).T
    return grid, np.linalg.norm(aligned - grid, axis=2)


if __name__ == "__main__":
    data = np.load(gps_data_path("GPS_Data/Processed_GPS_Receivers_DOG_1.npz"))
    GPS_Coordinates = data["GPS_Coordinates"]
    # The grid in bermuda_data.GPS1_TO_OTHERS
    grid, misfit = fit_rigid_grid(GPS_Coordinates, GPS1_TO_OTHERS)
    print("Rigid grid (m):")
    print(np.round(grid, 6))
    print(
        "Misfit rms per receiver (cm):",
        np.round(100 * np.sqrt(np.mean(misfit**2, 0)), 2),
    )
    GPS_Lever_arms(GPS_Coordinates)
