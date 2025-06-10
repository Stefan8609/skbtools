import numpy as np
from fitPlane import fitPlane
from projectToPlane import projectToPlane

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

    print(all_arms)
    for i in range(4):
        arm = np.mean(all_arms[i], axis=0, keepdims=True)[0]
        print(arm)
        print(np.linalg.norm(arm))


if __name__ == "__main__":
    data = np.load("../GPSData/Processed_GPS_Receivers_DOG_1.npz")
    GPS_Coordinates = data["GPS_Coordinates"]
    GPS_Lever_arms(GPS_Coordinates)
