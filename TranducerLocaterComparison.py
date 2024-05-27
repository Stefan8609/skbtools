from RigidBodyMovementProblem import findRotationAndDisplacement
from findPointByPlane import initializeFunction, findXyzt
from fitPlane import fitPlane

import numpy as np
import matplotlib.pyplot as plt

xs = np.random.rand(4) * 10 - 5
ys = np.random.rand(4) * 10 - 5
zs = np.random.rand(4) * 2 - 1
xyzt = np.random.rand(3) * 30 - 25
rot = np.random.rand(3) * np.pi - 2 * np.pi
translate = np.random.rand(3) * 10 - 5

xyzs_init = np.array([xs, ys, zs])
xyzt_init = xyzt

xRot = np.array([[1, 0, 0], [0, np.cos(rot[0]), -np.sin(rot[0])], [0, np.sin(rot[0]), np.cos(rot[0])]])
yRot = np.array([[np.cos(rot[1]), 0, np.sin(rot[1])], [0, 1, 0], [-np.sin(rot[1]), 0, np.cos(rot[1])]])
zRot = np.array([[np.cos(rot[2]), -np.sin(rot[2]), 0], [np.sin(rot[2]), np.cos(rot[2]), 0], [0, 0, 1]])
totalRot = np.matmul(xRot, np.matmul(yRot, zRot))

# Rotate the point cloud and xyzt according to rotations chosen
for i in range(len(xs)):
    xs[i], ys[i], zs[i] = np.matmul(totalRot, np.array([xs[i], ys[i], zs[i]]))
    xs[i] += translate[0]
    ys[i] += translate[1]
    zs[i] += translate[2]
xyzt = np.matmul(totalRot, xyzt)
xyzt = xyzt + translate

# Apply perturbation to some points to investigate how error occurs when points are not
#   In exact position after translation/rotation
xs += np.random.normal(0, .02, 4)
ys += np.random.normal(0, .02, 4)
zs += np.random.normal(0, .02, 4)
# Error due to perturbation scales fast with perturbation magnitude

xyzs_final = np.array([xs, ys, zs])
xyzt_final = xyzt

#Plot initial and final planes (start here)
# x_grid = np.linspace(min(xyzs_init[0]), xrange[1], 10)
# y_grid = np.linspace(yrange[0], yrange[1], 10)
# X, Y = np.meshgrid(x_grid, y_grid)
#
# # Use equation ax+by+cz+d=0 for plane
# a, b, c = normVect
# d = -np.dot(normVect, point)
# Z = (-a * X - b * Y - d) / c
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, alpha=0.7, rstride=50, cstride=50, color='b')
#
# ax.set_title('Plot of Plane')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')