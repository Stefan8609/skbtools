import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _rot_matrix_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """Rotation matrix X (roll) @ Y (pitch) @ Z (yaw), angles in radians."""
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

    # totalRot = xRot @ yRot @ zRot, written out
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


@njit(cache=True, fastmath=True)
def generateRealistic(n, gps1_to_others=None, gps1_to_transponder=None):
    """Box-pattern survey of four legs with random vessel motion.

    ``n`` is truncated to a multiple of 4. Returns ``(CDog, GPS_Coordinates,
    transponder_coordinates, gps1_to_others, gps1_to_transponder)`` in m, with
    GPS_Coordinates (m, 4, 3) and transponder_coordinates (m, 3).
    """
    if gps1_to_others is None:
        gps1_to_others = np.array(
            [
                [0.0, 0.0, 0.0],
                [-2.39341409, -4.22350344, 0.02941493],
                [-12.09568416, -0.94568462, 0.0043972],
                [-8.68674054, 5.16918806, 0.02499322],
            ]
        )
    if gps1_to_transponder is None:
        gps1_to_transponder = np.array([-12.4659, 9.6021, -13.2993])

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
    y3 = -x3 + (np.random.random(leg) * 50.0) - 25.0
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
            disp = _apply_rot(R, gps1_to_others[j, :])
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
