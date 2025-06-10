"""
Plots a plane given a normal vector and a point on the plane
Written by Stefan Kildal-Brandt

Inputs:
    (vector len=3) point = point on the plane
    (vector len=3) normVect = Vector normal to the plane
    (list[min, max]) xrange = The range over which the plane extends in the x-direction
    (list[min, max]) yrange = The range over which the plane extends in the y-direction
Outputs:
    (axes object) ax = Figure object containing axes of plot
"""

import numpy as np
import matplotlib.pyplot as plt

def plotPlane(point, normVect, xrange, yrange):
    x_grid = np.linspace(xrange[0], xrange[1], 10)
    y_grid = np.linspace(yrange[0], yrange[1], 10)
    X, Y = np.meshgrid(x_grid, y_grid)

    #Use equation ax+by+cz+d=0 for plane
    a, b, c = normVect
    d = -np.dot(normVect, point)
    Z = (-a*X-b*Y-d)/c

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X,Y,Z, alpha=0.7, rstride=50, cstride=50, color='b')

    ax.set_title('Plot of Plane')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    return ax


"""
Demo - Plots a plane given a point and normal vector
Inputs:
    point (vector len=3) = xyz coordinates of a point in the plane
    normVect (vector len=3) = The normal vector which defines the plane
Default values are the point (10,-10,5) and the normal vector (-.5, 1, .2)
"""

def demo(point = [10,-10,5], normVect = [-.5, 1, .2]):
    ax = plotPlane(point, normVect, [point[0]-10,point[0]+10], [point[1]-10,point[1]+10])
    ax.scatter(point[0], point[1], point[2], color='g')
    plt.show()

if __name__ == "__main__":
    demo()