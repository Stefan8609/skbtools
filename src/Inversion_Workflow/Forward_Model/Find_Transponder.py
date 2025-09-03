import numpy as np
from numba import njit

from geometry.Numba_RigidBodyMovementProblem import findRotationAndDisplacement


@njit(cache=True)
def findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder):
    """Estimate transponder positions for each time step.

    Parameters
    ----------
    GPS_Coordinates : ndarray
        ``(N, 4, 3)`` array of GPS positions for each time step.
    gps1_to_others : ndarray
        Relative positions of the additional GPS receivers to GPS1.
    gps1_to_transponder : ndarray
        Initial guess for the lever arm from GPS1 to the transponder.

    Returns
    -------
    ndarray
        ``(N, 3)`` array of transponder coordinates.
    """
    # Given initial relative GPS locations and transponder and GPS Coords
    # at each timestep
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
