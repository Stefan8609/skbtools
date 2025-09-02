"""Utility routines for visualizing a plane in 3‑D space."""

import numpy as np
import matplotlib.pyplot as plt
from plotting.save import save_plot


def plotPlane(point, normVect, xrange, yrange, save=False, chain_name=None, path="Figs"):
    """Plot a plane defined by a point and its normal vector.

    Parameters
    ----------
    point : array-like of float, shape (3,)
        Point lying on the plane.
    normVect : array-like of float, shape (3,)
        Normal vector of the plane.
    xrange, yrange : sequence of float
        ``[min, max]`` bounds describing the extent of the mesh grid.
    save : bool, optional
        If True, save the figure to disk.
    chain_name : str, optional
        Identifier used in the saved filename.
    path : str, optional
        Subdirectory under the Data directory in which to save.

    Returns
    -------
    matplotlib.axes.Axes
        Axes object containing the 3‑D plot.
    """

    x_grid = np.linspace(xrange[0], xrange[1], 10)
    y_grid = np.linspace(yrange[0], yrange[1], 10)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Use equation ax+by+cz+d=0 for plane
    a, b, c = normVect
    d = -np.dot(normVect, point)
    Z = (-a * X - b * Y - d) / c

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, alpha=0.7, rstride=50, cstride=50, color="b")

    ax.set_title("Plot of Plane")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if save:
        save_plot(fig, chain_name, "plot_plane", subdir=path)

    return ax


"""
Demo - Plots a plane given a point and normal vector
Inputs:
    point (vector len=3) = xyz coordinates of a point in the plane
    normVect (vector len=3) = The normal vector which defines the plane
Default values are the point (10,-10,5) and the normal vector (-.5, 1, .2)
"""


def demo(point=None, normVect=None):
    """Visualize a plane using example input values.

    Parameters
    ----------
    point : array-like of float, optional
        Location on the plane, by default ``[10, -10, 5]``.
    normVect : array-like of float, optional
        Plane normal vector, by default ``[-0.5, 1, 0.2]``.
    """
    if point is None:
        point = [10, -10, 5]
    if normVect is None:
        normVect = [-0.5, 1, 0.2]
    ax = plotPlane(
        point, normVect, [point[0] - 10, point[0] + 10], [point[1] - 10, point[1] + 10]
    )
    ax.scatter(point[0], point[1], point[2], color="g")
    plt.show()


if __name__ == "__main__":
    demo()
