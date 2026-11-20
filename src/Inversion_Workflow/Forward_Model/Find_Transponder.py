import numpy as np
from numba import njit

from geometry.Numba_RigidBodyMovementProblem import findRotationAndDisplacement


@njit(cache=True)
def findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder):
    """Transducer positions (N, 3) from the GPS positions (N, 4, 3) of each epoch.

    Eq. 1 with A7-A11: the rigid-body fit of the receiver grid ``gps1_to_others``
    (vessel frame, m from GPS1) to each epoch moves the lever arm
    ``gps1_to_transponder``. ECEF metres.
    """
    xs, ys, zs = gps1_to_others.T
    initial_transponder = gps1_to_transponder
    n = len(GPS_Coordinates)
    transponder_coordinates = np.zeros((n, 3))
    for i in range(n):
        new_xs, new_ys, new_zs = GPS_Coordinates[i].T
        xyzs_init = np.vstack((xs, ys, zs))
        xyzs_final = np.vstack((new_xs, new_ys, new_zs))

        R_mtrx, d = findRotationAndDisplacement(xyzs_init, xyzs_final)
        transponder_coordinates[i] = R_mtrx @ initial_transponder + d
    return transponder_coordinates
