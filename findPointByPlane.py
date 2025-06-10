import numpy as np
from fitPlane import fitPlane
from projectToPlane import projectToPlane
from rodriguesRotationMatrix import rotationMatrix


def findTheta(barycenter, xyzt, normVect):
    """Return the angle between ``xyzt`` and the plane normal.

    Parameters
    ----------
    barycenter : array-like of float, shape (3,)
        Centroid of the reference point cloud.
    xyzt : array-like of float, shape (3,)
        Point of interest.
    normVect : array-like of float, shape (3,)
        Normal vector of the fitted plane.

    Returns
    -------
    float
        Angle in radians between ``xyzt`` and ``normVect``.
    """
    disVect = np.array(xyzt - barycenter)
    dot = np.dot(disVect, normVect)
    disVect_Length = np.linalg.norm(disVect)
    normVect_Length = np.linalg.norm(normVect)
    theta = np.arccos(dot / (disVect_Length * normVect_Length))
    return theta


def findPhi(barycenter, xyzt, point, normVect):
    """Compute the in-plane angle from ``point`` to ``xyzt``.

    Parameters
    ----------
    barycenter : array-like of float, shape (3,)
        Cloud centroid.
    xyzt : array-like of float, shape (3,)
        Point of interest.
    point : array-like of float, shape (3,)
        Reference point lying in the plane.
    normVect : array-like of float, shape (3,)
        Plane normal vector.

    Returns
    -------
    float
        Angle ``phi`` in radians within the plane.
    """
    pointVect = np.array(point - barycenter)
    pointProjection = projectToPlane(pointVect, normVect)
    distanceVect = np.array(xyzt - barycenter)
    distanceProjection = projectToPlane(distanceVect, normVect)

    # Cannot use normal method of getting angle
    # because rotation could be out of arccos range of [0, 180]
    # Use plane embedded in 3D instead
    det = np.dot(normVect, np.cross(pointProjection, distanceProjection))
    dot = np.dot(pointProjection, distanceProjection)
    phi = np.arctan2(det, dot)
    return phi


def findLength(barycenter, xyzt):
    """Return the distance from ``barycenter`` to ``xyzt``.

    Parameters
    ----------
    barycenter : array-like of float, shape (3,)
        Reference point.
    xyzt : array-like of float, shape (3,)
        Point of interest.

    Returns
    -------
    float
        Euclidean distance between the two points.
    """
    length = np.linalg.norm(np.array(xyzt - barycenter))
    return length


def initializeFunction(xs, ys, zs, pointIdx, xyzt):
    """Initialize geometric parameters describing ``xyzt``.

    Parameters
    ----------
    xs, ys, zs : array-like of float
        Coordinates of the point cloud.
    pointIdx : int
        Index of the reference point within the cloud.
    xyzt : array-like of float, shape (3,)
        Point of interest.

    Returns
    -------
    list
        ``[theta, phi, length, orientation]`` describing the orientation and
        radial position of ``xyzt`` relative to the cloud.
    """
    points = np.array([xs, ys, zs])
    barycenter = np.mean(points, axis=1)
    normVect = fitPlane(xs, ys, zs)

    if np.dot(np.array(points[:, pointIdx] - barycenter), normVect) > 0:
        orientation = True
    else:
        orientation = False
    theta = findTheta(barycenter, xyzt, normVect)
    phi = findPhi(barycenter, xyzt, points[:, pointIdx], normVect)
    length = findLength(barycenter, xyzt)
    return [theta, phi, length, orientation]


def findXyzt(
    xs, ys, zs, pointIdx, length, theta, phi, orientation
):  # Main function finding the xyzt given initial conditions
    """Reconstruct ``xyzt`` from the updated point cloud.

    Parameters
    ----------
    xs, ys, zs : array-like of float
        Coordinates of the transformed point cloud.
    pointIdx : int
        Index of the reference point used during initialization.
    length : float
        Distance from the barycenter to ``xyzt``.
    theta, phi : float
        Rotation angles returned by :func:`initializeFunction`.
    orientation : bool
        True if the plane normal pointed towards ``xyzt`` at initialization.

    Returns
    -------
    list
        ``[xyztVector, barycenter, normVect]`` giving the location of ``xyzt``
        relative to the new point cloud.
    """
    barycenter = np.mean(np.array([xs, ys, zs]), axis=1)
    normVect = fitPlane(xs, ys, zs)
    point = np.array([xs[pointIdx], ys[pointIdx], zs[pointIdx]])

    # Confirm same orientation otherwise invert the direction of the normal vector
    if (np.dot(point - barycenter, normVect) > 0) != orientation:
        normVect = normVect * -1

    normVect_Length = np.linalg.norm(normVect)
    unitNorm = normVect / normVect_Length

    # Scale the normal vector to the length
    #  of the distance between barycenter and xyzt
    xyztVector = normVect * length / normVect_Length

    # Rotate the scaled vector theta degrees around
    #  vector between barycenter and chosen point
    rotationPoint = projectToPlane(point, normVect)
    rotationPoint_Vect = rotationPoint - barycenter
    rotationVector = np.cross(normVect, rotationPoint_Vect)
    rotationVector = rotationVector / np.linalg.norm(rotationVector)
    Theta_Matrix = rotationMatrix(theta, rotationVector)
    xyztVector = np.matmul(Theta_Matrix, xyztVector)

    # Rotate the scaled vector phi degrees around the normal vector
    Phi_Matrix = rotationMatrix(phi, unitNorm)
    xyztVector = np.matmul(Phi_Matrix, xyztVector)

    # The scaled and rotated vector should now lie on the position of the xyzt
    return [xyztVector, barycenter, normVect]


def demo(xs=None, ys=None, zs=None, xyzt=None, rot=None, translate=None):
    """Run a randomized demonstration of point reconstruction.

    Parameters
    ----------
    xs, ys, zs : array-like of float, optional
        Initial coordinates for the point cloud.
    xyzt : array-like of float, optional
        Location of the target point.
    rot, translate : array-like of float, optional
        Random rotation (radians) and translation applied to the cloud.
    """
    if xs is None:
        xs = np.random.rand(4) * 10 - 20
    if ys is None:
        ys = np.random.rand(4) * 10 - 20
    if zs is None:
        zs = np.random.rand(4) * 5 - 4
    if xyzt is None:
        xyzt = np.random.rand(3) * 30 - 25
    if rot is None:
        rot = np.random.rand(3) * np.pi / 2 - np.pi / 4
    if translate is None:
        translate = np.random.rand(3) * 10 - 5
    [theta0, phi0, length0, orientation0] = initializeFunction(xs, ys, zs, 0, xyzt)
    [theta1, phi1, length1, orientation1] = initializeFunction(xs, ys, zs, 1, xyzt)
    [theta2, phi2, length2, orientation2] = initializeFunction(xs, ys, zs, 2, xyzt)
    [theta3, phi3, length3, orientation3] = initializeFunction(xs, ys, zs, 3, xyzt)

    xRot = np.array(
        [
            [1, 0, 0],
            [0, np.cos(rot[0]), -np.sin(rot[0])],
            [0, np.sin(rot[0]), np.cos(rot[0])],
        ]
    )
    yRot = np.array(
        [
            [np.cos(rot[1]), 0, np.sin(rot[1])],
            [0, 1, 0],
            [-np.sin(rot[1]), 0, np.cos(rot[1])],
        ]
    )
    zRot = np.array(
        [
            [np.cos(rot[2]), -np.sin(rot[2]), 0],
            [np.sin(rot[2]), np.cos(rot[2]), 0],
            [0, 0, 1],
        ]
    )
    totalRot = np.matmul(xRot, np.matmul(yRot, zRot))

    # Rotate the point cloud and xyzt according to rotations chosen
    for i in range(len(xs)):
        xs[i], ys[i], zs[i] = np.matmul(totalRot, np.array([xs[i], ys[i], zs[i]]))
        xs[i] += translate[0]
        ys[i] += translate[1]
        zs[i] += translate[2]
    xyzt = np.matmul(totalRot, xyzt)
    xyzt = xyzt + translate

    # Apply perturbation to some points to investigate
    #   how error occurs when points are not
    #   In exact position after translation/rotation
    xs += np.random.normal(0, 0.02, 4)
    ys += np.random.normal(0, 0.02, 4)
    zs += np.random.normal(0, 0.02, 4)
    # Error due to perturbation scales fast with perturbation magnitude

    finalVect0, barycenter, normVect = findXyzt(
        xs, ys, zs, 0, length0, theta0, phi0, orientation0
    )
    finalVect1, barycenter, normVect = findXyzt(
        xs, ys, zs, 1, length1, theta1, phi1, orientation1
    )
    finalVect2, barycenter, normVect = findXyzt(
        xs, ys, zs, 2, length2, theta2, phi2, orientation2
    )
    finalVect3, barycenter, normVect = findXyzt(
        xs, ys, zs, 3, length3, theta3, phi3, orientation3
    )

    all_final_vect = np.array([finalVect0, finalVect1, finalVect2, finalVect3])
    average_final_vect = np.mean(all_final_vect, axis=0)
    print(all_final_vect)
    print(average_final_vect)

    # Plot the point cloud, plane of best fit, and vector to xyzt
    ax = plotPlane(barycenter, normVect, [min(xs), max(xs)], [min(ys), max(ys)])

    ax.scatter(xs, ys, zs, color="g")
    ax.scatter(xyzt[0], xyzt[1], xyzt[2], color="r")
    ax.quiver(
        barycenter[0],
        barycenter[1],
        barycenter[2],
        average_final_vect[0],
        average_final_vect[1],
        average_final_vect[2],
        color="k",
    )

    data = []
    for i in range(len(xs)):
        tup = (f"point {i}:", xs[i], ys[i], zs[i])
        data.append(tup)
    data.append(
        (
            "xyzt predicted:",
            barycenter[0] + average_final_vect[0],
            barycenter[1] + average_final_vect[1],
            barycenter[2] + average_final_vect[2],
        )
    )
    data.append(("xyzt actual:", xyzt[0], xyzt[1], xyzt[2]))
    data.append(
        (
            "Error (x,y,z):",
            barycenter[0] + average_final_vect[0] - xyzt[0],
            barycenter[1] + average_final_vect[1] - xyzt[1],
            barycenter[2] + average_final_vect[2] - xyzt[2],
        )
    )
    data.append(
        (
            "Error1 (x,y,z):",
            barycenter[0] + finalVect0[0] - xyzt[0],
            barycenter[1] + finalVect0[1] - xyzt[1],
            barycenter[2] + finalVect0[2] - xyzt[2],
        )
    )

    printTable(["Point", "X", "Y", "Z"], data)
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ECCO_and_plotting.plotPlane import plotPlane
    from ECCO_and_plotting.printTable import printTable

    # xs = np.array([0, -2.4054, -12.11, -8.7])
    # ys = np.array([0, -4.21, -0.956, 5.165])
    # zs = np.array([0, 0.060621, 0.00877, 0.0488])
    # xyzt = np.array([-6.4, 2.46, -15.24])
    # demo(xs, ys, zs, xyzt)

    demo()
