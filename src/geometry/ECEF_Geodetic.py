# Adapted from
# https://gis.stackexchange.com/questions/265909/converting-from-ecef-to-geodetic-coordinates

import numpy as np
from numba import njit


@njit(cache=True)
def ECEF_Geodetic(coords):
    """Latitude (deg), longitude (deg) and WGS84 ellipsoidal height (m) of ECEF points.

    ``coords`` is (N, 3) in metres. The latitude is refined by two fixed-point
    iterations.
    """
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    # WGS84
    a = 6378137.0
    f = 1.0 / 298.257223563
    b = a - f * a
    e = np.sqrt(np.power(a, 2.0) - np.power(b, 2.0)) / a
    clambda = np.arctan2(y, x)
    p = np.sqrt(np.power(x, 2.0) + np.power(y, 2))
    theta = np.arctan2(z, p * (1.0 - np.power(e, 2.0)))
    cs = np.cos(theta)
    sn = np.sin(theta)
    N = np.power(a, 2.0) / np.sqrt(np.power(a * cs, 2.0) + np.power(b * sn, 2.0))
    h = p / cs - N
    k = 0
    while k < 2:
        theta = np.arctan2(z, p * (1.0 - np.power(e, 2.0) * N / (N + h)))
        cs = np.cos(theta)
        sn = np.sin(theta)
        N = np.power(a, 2.0) / np.sqrt(np.power(a * cs, 2.0) + np.power(b * sn, 2.0))
        h = p / cs - N
        k += 1
    clambda = np.degrees(clambda)
    theta = np.degrees(theta)
    return theta, clambda, h
