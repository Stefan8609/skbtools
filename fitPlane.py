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
    points = np.array([xs, ys, zs])
    svd = np.linalg.svd(points - np.mean(points, axis=1, keepdims=True))
    left = svd[0]
    normVect = np.array(left[:, -1], dtype=np.float64)
    return normVect


"""
Demo - Find and plot the plane for a point cloud
Inputs:
   xs (array[n]) = x coordinates of the points in the cloud
   ys (array[n]) = y coordinates of the points in the cloud
   zs (array[n]) = z coordinates of the points in the cloud
Default value for parameters are 10 random points with coordinates between -100 and 100
"""

import matplotlib.pyplot as plt
from ECCO_and_plotting.plotPlane import plotPlane


def demo(
    xs=np.random.rand(10) * 100,
    ys=np.random.rand(10) * 100,
    zs=np.random.rand(10) * 100,
):
    normVect = fitPlane(xs, ys, zs)
    points = np.array([xs, ys, zs])
    ax = plotPlane(
        np.mean(points, axis=1), normVect, [min(xs), max(xs)], [min(ys), max(ys)]
    )
    ax.scatter(xs, ys, zs, color="g")

    plt.show()


if __name__ == "__main__":
    demo()
