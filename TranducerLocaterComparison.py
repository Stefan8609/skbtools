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

new_xs = np.zeros_like(xs)
new_ys = np.zeros_like(ys)
new_zs = np.zeros_like(zs)

# Rotate the point cloud and xyzt according to rotations chosen
for i in range(len(xs)):
    new_xs[i], new_ys[i], new_zs[i] = np.matmul(totalRot, np.array([xs[i], ys[i], zs[i]]))
    new_xs[i] += translate[0]
    new_ys[i] += translate[1]
    new_zs[i] += translate[2]
xyzt_final = np.matmul(totalRot, xyzt)
xyzt_final = xyzt_final + translate

# Apply perturbation to some points to investigate how error occurs when points are not
#   In exact position after translation/rotation
xs += np.random.normal(0, .05, 4)
ys += np.random.normal(0, .05, 4)
zs += np.random.normal(0, .05, 4)

new_xs += np.random.normal(0, .05, 4)
new_ys += np.random.normal(0, .05, 4)
new_zs += np.random.normal(0, .05, 4)

# Error due to perturbation scales fast with perturbation magnitude

xyzs_final = np.array([new_xs, new_ys, new_zs])

#Get norm vect for plane plotting
norm_vect_init = fitPlane(xs, ys, zs)
norm_vect_final = fitPlane(new_xs, new_ys, new_zs)

#Get barycenters of point clouds
barycenter_init = np.mean(np.array([xs, ys, zs]), axis=1)
barycenter_final = np.mean(np.array([new_xs, new_ys, new_zs]), axis=1)

#Find xyzt using both methods
lever_init = xyzt_init - barycenter_init

#   Orthogonal Procrustes Solution
R, d = findRotationAndDisplacement(xyzs_init, xyzs_final)
lever_final_pro = R @ lever_init

#   Plane Rotation Solution
[theta, phi, length, orientation] = initializeFunction(xs, ys, zs, 0, xyzt_init)
lever_final_rot, barycenter_rot, normVect_rot = findXyzt(new_xs, new_ys, new_zs, 0, length, theta, phi, orientation)

#Plot initial plane
x_grid = np.linspace(min(xs), max(xs), 10)
y_grid = np.linspace(min(ys), max(ys), 10)
X, Y = np.meshgrid(x_grid, y_grid)

# Use equation ax+by+cz+d=0 for plane
a, b, c = norm_vect_init
d = -np.dot(norm_vect_init, barycenter_init)
Z = (-a * X - b * Y - d) / c

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.7, rstride=50, cstride=50, color='b', label="Initial Plane")
ax.scatter(xyzt_init[0], xyzt_init[1], xyzt_init[2], color='b')
ax.scatter(xs, ys, zs)

ax.set_title('Xyzt location given rotated and translated plane')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


#Plot final plane
x_grid = np.linspace(min(new_xs), max(new_xs), 10)
y_grid = np.linspace(min(new_ys), max(new_ys), 10)
X, Y = np.meshgrid(x_grid, y_grid)

# Use equation ax+by+cz+d=0 for plane
a, b, c = norm_vect_final
d = -np.dot(norm_vect_final, barycenter_final)
Z = (-a * X - b * Y - d) / c

ax.plot_surface(X, Y, Z, alpha=0.7, rstride=50, cstride=50, color='r', label="Final Plane")
ax.scatter(xyzt_final[0], xyzt_final[1], xyzt_final[2], color='r')
ax.scatter(new_xs, new_ys, new_zs)

#Vector for the actual lever and the estimated solutions after rotation and translation
ax.quiver(barycenter_init[0], barycenter_init[1], barycenter_init[2],
          lever_init[0], lever_init[1], lever_init[2], color="k", label="Actual Lever")
ax.quiver(barycenter_final[0], barycenter_final[1], barycenter_final[2],
          lever_final_pro[0], lever_final_pro[1], lever_final_pro[2], color="g", label="Procrustes Solution")
ax.quiver(barycenter_final[0], barycenter_final[1], barycenter_final[2],
          lever_final_rot[0], lever_final_rot[1], lever_final_rot[2], color="y", label="Plane Rotation Solution")

ax.legend()

plt.show()