import numpy as np
from numba import njit


@njit(cache=True)
def findRotationAndDisplacement(xyzs_init, xyzs_final):
    """Rotation R (3, 3) and translation d (3,) best mapping xyzs_init onto xyzs_final.

    Both are (3, N). Eqs. A10-A11: the SVD solution of Soderkvist, "Using SVD for
    some fitting problems"; the determinant term rules out a reflection.
    """
    # Centroids of the two point clouds
    centroid_init = np.array(
        [[np.mean(xyzs_init[0]), np.mean(xyzs_init[1]), np.mean(xyzs_init[2])]]
    ).T
    centroid_final = np.array(
        [[np.mean(xyzs_final[0]), np.mean(xyzs_final[1]), np.mean(xyzs_final[2])]]
    ).T

    # Centred point clouds
    A_mtrx = xyzs_init - centroid_init
    B_mtrx = xyzs_final - centroid_final

    # SVD of the cross-covariance
    C_mtrx = B_mtrx @ A_mtrx.T
    U, S, V_t = np.linalg.svd(C_mtrx)

    # Rotation without reflection, then the displacement
    det = np.linalg.det(U @ V_t)
    R_mtrx = U @ np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, det]]) @ V_t
    d = centroid_final - R_mtrx @ centroid_init
    d = d.T[0]
    return R_mtrx, d
