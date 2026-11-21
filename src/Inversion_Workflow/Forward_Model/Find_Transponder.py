import numpy as np
from numba import njit

from geometry.Numba_RigidBodyMovementProblem import findRotationAndDisplacement


@njit(cache=True)
def vessel_rotations(GPS_Coordinates, gps1_to_others):
    """Rotations (N, 3, 3) and displacements (N, 3) fitted at each epoch (A10-A11).

    The rigid-body fit maps the receiver grid ``gps1_to_others`` (vessel frame, m
    from GPS1) onto the GPS positions (N, 4, 3) of each epoch. It does not depend on
    the lever arm, so loops over lever arms compute it once.
    """
    xs, ys, zs = gps1_to_others.T
    n = len(GPS_Coordinates)
    rotations = np.zeros((n, 3, 3))
    displacements = np.zeros((n, 3))
    for i in range(n):
        new_xs, new_ys, new_zs = GPS_Coordinates[i].T
        xyzs_init = np.vstack((xs, ys, zs))
        xyzs_final = np.vstack((new_xs, new_ys, new_zs))

        R_mtrx, d = findRotationAndDisplacement(xyzs_init, xyzs_final)
        rotations[i] = R_mtrx
        displacements[i] = d
    return rotations, displacements


@njit(cache=True)
def transponder_from_rotations(rotations, displacements, gps1_to_transponder):
    """Transducer positions (N, 3), ECEF m: Eq. 1, ``R @ lever + d`` at each epoch."""
    n = len(rotations)
    transponder_coordinates = np.zeros((n, 3))
    for i in range(n):
        transponder_coordinates[i] = (
            rotations[i] @ gps1_to_transponder + displacements[i]
        )
    return transponder_coordinates


@njit(cache=True)
def findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder):
    """Transducer positions (N, 3) from the GPS positions (N, 4, 3) of each epoch.

    Eq. 1 with A7-A11: the rigid-body fit of the receiver grid ``gps1_to_others``
    (vessel frame, m from GPS1) to each epoch moves the lever arm
    ``gps1_to_transponder``. ECEF metres.
    """
    rotations, displacements = vessel_rotations(GPS_Coordinates, gps1_to_others)
    return transponder_from_rotations(rotations, displacements, gps1_to_transponder)
