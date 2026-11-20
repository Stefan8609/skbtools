import numpy as np


def findRotationAndDisplacement(xyzs_init, xyzs_final):
    """
    Inspiration from Inge Soderkvist "Using SVD for Some Fitting Problems"

    Estimate the rigid body rotation and translation between two point clouds.

    Parameters
    ----------
    xyzs_init : array-like, shape (3, N)
        Initial coordinates.
    xyzs_final : array-like, shape (3, N)
        Transformed coordinates.

    Returns
    -------
    tuple
        ``(R, d)`` where ``R`` is a ``3 x 3`` rotation matrix and ``d`` is the
        translation vector.
    """

    # Compute centroid of the initial and final point clouds
    centroid_init = np.mean(xyzs_init, axis=1, keepdims=True)
    centroid_final = np.mean(xyzs_final, axis=1, keepdims=True)

    # Compute matrices for each point in point cloud subtracted
    # by its respective centroid
    A_mtrx = xyzs_init - centroid_init
    B_mtrx = xyzs_final - centroid_final

    # Get a matrix product of the the two matrices above and find its SVD
    C_mtrx = np.tensordot(B_mtrx, A_mtrx, axes=(1, 1))
    U, S, V_t = np.linalg.svd(C_mtrx)

    # Use the SVD to compute the rotation matrix and displacement
    # between the point clouds
    #   Following the instructions of the source above
    det = np.linalg.det(U @ V_t)
    R_mtrx = U @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, det]]) @ V_t
    d = centroid_final - R_mtrx @ centroid_init
    d = d.T[0]
    return R_mtrx, d


"""
DEMO BELOW
"""
