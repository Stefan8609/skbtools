"""
Function to convert from ECEF coordinates to Geodetic coordinates
    Uses the WGS84 ellipsoid model
    Adapted from https://gis.stackexchange.com/questions/265909/converting-from-ecef-to-geodetic-coordinates
"""

import numpy as np
from numba import njit


@njit(cache=True)
def ECEF_Geodetic(coords):
    """Convert ECEF coordinates to geodetic coordinates.

    Parameters
    ----------
    coords : ndarray, shape (N, 3)
        Array of ``(x, y, z)`` positions in metres.

    Returns
    -------
    tuple of ndarray
        ``(lat, lon, h)`` arrays in degrees/metres for each input point.
    """
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    # --- WGS84 constants
    a = 6378137.0
    f = 1.0 / 298.257223563
    # --- derived constants
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
