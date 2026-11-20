import numpy as np


def findRotationAndDisplacement(xyzs_init, xyzs_final):
    """NumPy version of the solver in ``Numba_RigidBodyMovementProblem``."""

    # Centroids of the two point clouds
    centroid_init = np.mean(xyzs_init, axis=1, keepdims=True)
    centroid_final = np.mean(xyzs_final, axis=1, keepdims=True)

    # Centred point clouds
    A_mtrx = xyzs_init - centroid_init
    B_mtrx = xyzs_final - centroid_final

    # SVD of the cross-covariance
    C_mtrx = np.tensordot(B_mtrx, A_mtrx, axes=(1, 1))
    U, S, V_t = np.linalg.svd(C_mtrx)

    # Rotation without reflection, then the displacement
    det = np.linalg.det(U @ V_t)
    R_mtrx = U @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, det]]) @ V_t
    d = centroid_final - R_mtrx @ centroid_init
    d = d.T[0]
    return R_mtrx, d
