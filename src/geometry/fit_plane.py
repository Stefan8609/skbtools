"""
Fits plane to cloud of points by minimizing absolute distance
Uses singular value decomposition to get least squares regression for plane
Written by Stefan Kildal-Brandt

Inputs:
   xs (array[n]) = x coordinates of points in cloud
   ys (array[n]) = y coordinates of points in cloud
   zs (array[n]) = z coordinates of points in cloud
Output:
    normVect (vector len=3) = The vector normal to the plane
"""

import numpy as np


def fitPlane(xs, ys, zs):
    """Calculate the normal vector of the best-fit plane for a point cloud.

    Parameters
    ----------
    xs, ys, zs : array-like of float
        Coordinates of the points.

    Returns
    -------
    numpy.ndarray
        Unit normal vector of the fitted plane.
    """

    points = np.array([xs, ys, zs])
    svd = np.linalg.svd(points - np.mean(points, axis=1, keepdims=True))
    left = svd[0]
    normVect = np.array(left[:, -1], dtype=np.float64)
    return normVect

