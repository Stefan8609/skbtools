"""Least-squares plane through a point cloud. Written by Stefan Kildal-Brandt."""

import numpy as np


def fitPlane(xs, ys, zs):
    """Unit normal of the least-squares plane through the points (xs, ys, zs)."""

    points = np.array([xs, ys, zs])
    svd = np.linalg.svd(points - np.mean(points, axis=1, keepdims=True))
    left = svd[0]
    normVect = np.array(left[:, -1], dtype=np.float64)
    return normVect
