import numpy as np
import matplotlib.pyplot as plt
from geometry.fit_plane import fitPlane
from geometry.project_to_plane import projectToPlane
from data import gps_data_path

np.set_printoptions(suppress=True)


def GPS_Lever_arms(GPS_Coordinates):
    """Compute lever arms between GPS receivers.

    Parameters
    ----------
    GPS_Coordinates : array-like, shape (N, 4, 3)
        Positions of the four GPS units for ``N`` time steps.

    Returns
    -------
    numpy.ndarray
        Array of lever arm vectors with shape ``(4, N, 3)``.
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


if __name__ == "__main__":
    data = np.load(gps_data_path("GPS_Data/Processed_GPS_Receivers_DOG_1.npz"))
    GPS_Coordinates = data["GPS_Coordinates"]
    GPS_Lever_arms(GPS_Coordinates)
