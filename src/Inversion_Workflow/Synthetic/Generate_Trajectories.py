import numpy as np
from numba import njit


# Numba-friendly helpers
@njit(cache=True, fastmath=True)
def _rot_matrix_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """Rotation matrix with intrinsic X (roll), Y (pitch), Z (yaw) order.

    Parameters
    ----------
    rx, ry, rz : float
        Rotations (radians) about x, y, z respectively.
    """
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    xRot = np.empty((3, 3))
    xRot[0, 0], xRot[0, 1], xRot[0, 2] = 1.0, 0.0, 0.0
    xRot[1, 0], xRot[1, 1], xRot[1, 2] = 0.0, cx, -sx
    xRot[2, 0], xRot[2, 1], xRot[2, 2] = 0.0, sx, cx

    yRot = np.empty((3, 3))
    yRot[0, 0], yRot[0, 1], yRot[0, 2] = cy, 0.0, sy
    yRot[1, 0], yRot[1, 1], yRot[1, 2] = 0.0, 1.0, 0.0
    yRot[2, 0], yRot[2, 1], yRot[2, 2] = -sy, 0.0, cy

    zRot = np.empty((3, 3))
    zRot[0, 0], zRot[0, 1], zRot[0, 2] = cz, -sz, 0.0
    zRot[1, 0], zRot[1, 1], zRot[1, 2] = sz, cz, 0.0
    zRot[2, 0], zRot[2, 1], zRot[2, 2] = 0.0, 0.0, 1.0

    # totalRot = xRot @ yRot @ zRot (explicit to avoid creating temporaries)
    tmp = np.empty((3, 3))
    totalRot = np.empty((3, 3))

    # tmp = xRot @ yRot
    for i in range(3):
        for j in range(3):
            s = 0.0
            for k in range(3):
                s += xRot[i, k] * yRot[k, j]
            tmp[i, j] = s

    # totalRot = tmp @ zRot
    for i in range(3):
        for j in range(3):
            s = 0.0
            for k in range(3):
                s += tmp[i, k] * zRot[k, j]
            totalRot[i, j] = s

    return totalRot


@njit(cache=True)
def _apply_rot(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Return mat @ vec without allocating Python temporaries."""
    out = np.empty(3)
    out[0] = mat[0, 0] * vec[0] + mat[0, 1] * vec[1] + mat[0, 2] * vec[2]
    out[1] = mat[1, 0] * vec[0] + mat[1, 1] * vec[1] + mat[1, 2] * vec[2]
    out[2] = mat[2, 0] * vec[0] + mat[2, 1] * vec[1] + mat[2, 2] * vec[2]
    return out


# Trajectory generators (Numba-compatible, no Python random)
@njit(cache=True, fastmath=True)
def generateRandomData(
    n: int, gps1_to_others: np.ndarray, gps1_to_transponder: np.ndarray
):
    """Generate a fully random synthetic trajectory.

    Returns
    -------
    tuple
        (CDog, GPS_Coordinates, transponder_coordinates, gps1_to_others,
        gps1_to_transponder)
    """
    # CDOG position
    CDog = np.empty(3, dtype=np.float64)
    CDog[0] = np.random.uniform(-1000.0, 1000.0)
    CDog[1] = np.random.uniform(-1000.0, 1000.0)
    CDog[2] = np.random.uniform(-5235.0, -5225.0)

    # Base GPS1 point
    xyz_point = np.empty(3, dtype=np.float64)
    xyz_point[0] = np.random.uniform(-1000.0, 1000.0)
    xyz_point[1] = np.random.uniform(-1000.0, 1000.0)
    xyz_point[2] = np.random.uniform(-10.0, 10.0)

    # Translations per timestep (scale Z by 1/100)
    translations = (np.random.random((n, 3)) * 15000.0) - 7500.0
    for i in range(n):
        translations[i, 2] *= 0.01

    # Random yaw/pitch/roll in [-pi/2, pi/2]
    rot = (np.random.random((n, 3)) * np.pi) - (0.5 * np.pi)

    GPS_Coordinates = np.zeros((n, 4, 3), dtype=np.float64)
    transponder_coordinates = np.zeros((n, 3), dtype=np.float64)

    for i in range(n):
        R = _rot_matrix_xyz(rot[i, 0], rot[i, 1], rot[i, 2])

        # GPS1 = base + translation
        GPS_Coordinates[i, 0, 0] = xyz_point[0] + translations[i, 0]
        GPS_Coordinates[i, 0, 1] = xyz_point[1] + translations[i, 1]
        GPS_Coordinates[i, 0, 2] = xyz_point[2] + translations[i, 2]

        # Other GPS antennas
        for j in range(1, 4):
            disp = _apply_rot(R, gps1_to_others[j])
            GPS_Coordinates[i, j, 0] = GPS_Coordinates[i, 0, 0] + disp[0]
            GPS_Coordinates[i, j, 1] = GPS_Coordinates[i, 0, 1] + disp[1]
            GPS_Coordinates[i, j, 2] = GPS_Coordinates[i, 0, 2] + disp[2]

        # Transponder
        tdisp = _apply_rot(R, gps1_to_transponder)
        transponder_coordinates[i, 0] = GPS_Coordinates[i, 0, 0] + tdisp[0]
        transponder_coordinates[i, 1] = GPS_Coordinates[i, 0, 1] + tdisp[1]
        transponder_coordinates[i, 2] = GPS_Coordinates[i, 0, 2] + tdisp[2]

    return (
        CDog,
        GPS_Coordinates,
        transponder_coordinates,
        gps1_to_others,
        gps1_to_transponder,
    )


@njit(cache=True, fastmath=True)
def generateLine(n: int, gps1_to_others: np.ndarray, gps1_to_transponder: np.ndarray):
    """Generate a roughly linear track with small lateral/vertical variation."""
    # CDOG
    CDog = np.empty(3, dtype=np.float64)
    CDog[0] = np.random.uniform(-1000.0, 1000.0)
    CDog[1] = np.random.uniform(-1000.0, 1000.0)
    CDog[2] = np.random.uniform(-5235.0, -5225.0)

    x = (np.random.random(n) * 15000.0) - 7500.0
    y = x + (np.random.random(n) * 50.0) - 25.0
    z = (np.random.random(n) * 5.0) - 10.0

    # Argsort by x (lexicographic sort in Numba is cumbersome; x is primary driver)
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    z = z[idx]

    GPS_Coordinates = np.zeros((n, 4, 3), dtype=np.float64)
    transponder_coordinates = np.zeros((n, 3), dtype=np.float64)

    # Set GPS1 positions
    for i in range(n):
        GPS_Coordinates[i, 0, 0] = x[i]
        GPS_Coordinates[i, 0, 1] = y[i]
        GPS_Coordinates[i, 0, 2] = z[i]

    # Random yaw/pitch/roll per step
    rot = (np.random.random((n, 3)) * np.pi) - (0.5 * np.pi)

    for i in range(n):
        R = _rot_matrix_xyz(rot[i, 0], rot[i, 1], rot[i, 2])
        for j in range(1, 4):
            disp = _apply_rot(R, gps1_to_others[j])
            GPS_Coordinates[i, j, 0] = GPS_Coordinates[i, 0, 0] + disp[0]
            GPS_Coordinates[i, j, 1] = GPS_Coordinates[i, 0, 1] + disp[1]
            GPS_Coordinates[i, j, 2] = GPS_Coordinates[i, 0, 2] + disp[2]
        tdisp = _apply_rot(R, gps1_to_transponder)
        transponder_coordinates[i, 0] = GPS_Coordinates[i, 0, 0] + tdisp[0]
        transponder_coordinates[i, 1] = GPS_Coordinates[i, 0, 1] + tdisp[1]
        transponder_coordinates[i, 2] = GPS_Coordinates[i, 0, 2] + tdisp[2]

    return (
        CDog,
        GPS_Coordinates,
        transponder_coordinates,
        gps1_to_others,
        gps1_to_transponder,
    )


@njit(cache=True, fastmath=True)
def generateCross(n: int, gps1_to_others: np.ndarray, gps1_to_transponder: np.ndarray):
    """Two perpendicular passes (a simple cross)."""
    # CDOG
    CDog = np.empty(3, dtype=np.float64)
    CDog[0] = np.random.uniform(-1000.0, 1000.0)
    CDog[1] = np.random.uniform(-1000.0, 1000.0)
    CDog[2] = np.random.uniform(-5235.0, -5225.0)

    half = n // 2
    x1 = (np.random.random(half) * 15000.0) - 7500.0
    x2 = (np.random.random(half) * 15000.0) - 7500.0
    x = np.empty(n)
    # Sort each half ascending to mimic line sweeps
    x1 = np.sort(x1)
    x2 = np.sort(x2)
    for i in range(half):
        x[i] = x1[i]
        x[half + i] = x2[i]

    y = x + (np.random.random(n) * 50.0) - 25.0
    # Flip the second half to create the cross
    for i in range(half, n):
        y[i] = -y[i]

    z = (np.random.random(n) * 5.0) - 10.0

    GPS_Coordinates = np.zeros((n, 4, 3), dtype=np.float64)
    transponder_coordinates = np.zeros((n, 3), dtype=np.float64)

    for i in range(n):
        GPS_Coordinates[i, 0, 0] = x[i]
        GPS_Coordinates[i, 0, 1] = y[i]
        GPS_Coordinates[i, 0, 2] = z[i]

    rot = (np.random.random((n, 3)) * np.pi) - (0.5 * np.pi)

    for i in range(n):
        R = _rot_matrix_xyz(rot[i, 0], rot[i, 1], rot[i, 2])
        for j in range(1, 4):
            disp = _apply_rot(R, gps1_to_others[j])
            GPS_Coordinates[i, j, 0] = GPS_Coordinates[i, 0, 0] + disp[0]
            GPS_Coordinates[i, j, 1] = GPS_Coordinates[i, 0, 1] + disp[1]
            GPS_Coordinates[i, j, 2] = GPS_Coordinates[i, 0, 2] + disp[2]
        tdisp = _apply_rot(R, gps1_to_transponder)
        transponder_coordinates[i, 0] = GPS_Coordinates[i, 0, 0] + tdisp[0]
        transponder_coordinates[i, 1] = GPS_Coordinates[i, 0, 1] + tdisp[1]
        transponder_coordinates[i, 2] = GPS_Coordinates[i, 0, 2] + tdisp[2]

    return (
        CDog,
        GPS_Coordinates,
        transponder_coordinates,
        gps1_to_others,
        gps1_to_transponder,
    )


@njit(cache=True, fastmath=True)
def generateRealistic(
    n: int, gps1_to_others: np.ndarray, gps1_to_transponder: np.ndarray
):
    """Create a synthetic survey with four legs and random boat motion.

    The path length is truncated to a multiple of 4 to form a box pattern.
    """
    # number of usable samples (multiple of 4)
    m = (n // 4) * 4

    # CDOG
    CDog = np.empty(3, dtype=np.float64)
    CDog[0] = np.random.uniform(-5000.0, 5000.0)
    CDog[1] = np.random.uniform(-5000.0, 5000.0)
    CDog[2] = np.random.uniform(-5235.0, -5225.0)

    # Build four legs, each of length m/4
    leg = m // 4

    x1 = np.sort((np.random.random(leg) * 15000.0) - 7500.0)
    x2 = -np.sort(-((np.random.random(leg) * 15000.0) - 7500.0))
    x3 = np.sort((np.random.random(leg) * 15000.0) - 7500.0)
    x4 = -np.sort(-((np.random.random(leg) * 15000.0) - 7500.0))

    y1 = x1 + (np.random.random(leg) * 50.0) - 25.0
    y2 = 7500.0 + (np.random.random(leg) * 50.0) - 25.0
    y3 = -x1 + (np.random.random(leg) * 50.0) - 25.0
    y4 = -7500.0 + (np.random.random(leg) * 50.0) - 25.0

    x = np.empty(m)
    y = np.empty(m)
    for i in range(leg):
        x[i] = x1[i]
        y[i] = y1[i]
        x[leg + i] = x2[i]
        y[leg + i] = y2[i]
        x[2 * leg + i] = x3[i]
        y[2 * leg + i] = y3[i]
        x[3 * leg + i] = x4[i]
        y[3 * leg + i] = y4[i]

    z = (np.random.random(m) * 5.0) - 10.0

    GPS_Coordinates = np.zeros((m, 4, 3), dtype=np.float64)
    transponder_coordinates = np.zeros((m, 3), dtype=np.float64)

    for i in range(m):
        GPS_Coordinates[i, 0, 0] = x[i]
        GPS_Coordinates[i, 0, 1] = y[i]
        GPS_Coordinates[i, 0, 2] = z[i]

    rot = (np.random.random((m, 3)) * np.pi) - (0.5 * np.pi)

    for i in range(m):
        R = _rot_matrix_xyz(rot[i, 0], rot[i, 1], rot[i, 2])
        for j in range(1, 4):
            disp = _apply_rot(R, gps1_to_others[j])
            GPS_Coordinates[i, j, 0] = GPS_Coordinates[i, 0, 0] + disp[0]
            GPS_Coordinates[i, j, 1] = GPS_Coordinates[i, 0, 1] + disp[1]
            GPS_Coordinates[i, j, 2] = GPS_Coordinates[i, 0, 2] + disp[2]
        tdisp = _apply_rot(R, gps1_to_transponder)
        transponder_coordinates[i, 0] = GPS_Coordinates[i, 0, 0] + tdisp[0]
        transponder_coordinates[i, 1] = GPS_Coordinates[i, 0, 1] + tdisp[1]
        transponder_coordinates[i, 2] = GPS_Coordinates[i, 0, 2] + tdisp[2]

    return (
        CDog,
        GPS_Coordinates,
        transponder_coordinates,
        gps1_to_others,
        gps1_to_transponder,
    )
