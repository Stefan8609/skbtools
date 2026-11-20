import numpy as np
from numba import njit


@njit(cache=True)
def findRotationAndDisplacement(xyzs_init, xyzs_final):
    """Rigid body transform between two point clouds.

    Parameters
    ----------
    xyzs_init : ndarray
        ``(3, N)`` initial coordinates.
    xyzs_final : ndarray
        ``(3, N)`` final coordinates.

    Returns
    -------
    tuple
        ``(R_mtrx, d)`` rotation matrix and translation vector.
    """
    # Compute centroid of the initial and final point clouds

    centroid_init = np.array(
        [[np.mean(xyzs_init[0]), np.mean(xyzs_init[1]), np.mean(xyzs_init[2])]]
    ).T
    centroid_final = np.array(
        [[np.mean(xyzs_final[0]), np.mean(xyzs_final[1]), np.mean(xyzs_final[2])]]
    ).T

    # Compute matrices for each point in
    # point cloud subtracted by its respective centroid
    A_mtrx = xyzs_init - centroid_init
    B_mtrx = xyzs_final - centroid_final

    # Get a matrix product of the the two matrices above and find its SVD
    C_mtrx = B_mtrx @ A_mtrx.T
    U, S, V_t = np.linalg.svd(C_mtrx)

    # Use the SVD to compute the rotation matrix and
    # displacement between the point clouds
    #   Following the instructions of the source above
    det = np.linalg.det(U @ V_t)
    R_mtrx = U @ np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, det]]) @ V_t
    d = centroid_final - R_mtrx @ centroid_init
    d = d.T[0]
    return R_mtrx, d
