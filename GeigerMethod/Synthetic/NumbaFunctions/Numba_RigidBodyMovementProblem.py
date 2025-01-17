import numpy as np
from numba import njit

@njit(cache=True)
def findRotationAndDisplacement(xyzs_init, xyzs_final):
    #Compute centroid of the initial and final point clouds

    centroid_init = np.array([[np.mean(xyzs_init[0]), np.mean(xyzs_init[1]), np.mean(xyzs_init[2])]]).T
    centroid_final = np.array([[np.mean(xyzs_final[0]), np.mean(xyzs_final[1]), np.mean(xyzs_final[2])]]).T

    #Compute matrices for each point in point cloud subtracted by its respective centroid
    A_mtrx = xyzs_init - centroid_init
    B_mtrx = xyzs_final - centroid_final

    #Get a matrix product of the the two matrices above and find its SVD
    C_mtrx = B_mtrx @ A_mtrx.T
    U, S, V_t = np.linalg.svd(C_mtrx)

    #Use the SVD to compute the rotation matrix and displacement between the point clouds
    #   Following the instructions of the source above
    det = np.linalg.det(U@V_t)
    R_mtrx = U @ np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,det]]) @ V_t
    d = centroid_final - R_mtrx@centroid_init
    d = d.T[0]
    return R_mtrx, d

def demo():
    xs = np.random.rand(4) * 10 - 5
    ys = np.random.rand(4) * 10 - 5
    zs = np.random.rand(4) * 2 - 1
    xyzt = np.random.rand(3) * 30 - 25
    rot = np.random.rand(3) * np.pi - 2 * np.pi
    translate = np.random.rand(3) * 10 - 5

    xyzs_init = np.array([xs,ys,zs])
    xyzt_init = xyzt

    xRot = np.array([[1, 0, 0], [0, np.cos(rot[0]), -np.sin(rot[0])], [0, np.sin(rot[0]), np.cos(rot[0])]])
    yRot = np.array([[np.cos(rot[1]), 0, np.sin(rot[1])], [0, 1, 0], [-np.sin(rot[1]), 0, np.cos(rot[1])]])
    zRot = np.array([[np.cos(rot[2]), -np.sin(rot[2]), 0], [np.sin(rot[2]), np.cos(rot[2]), 0], [0, 0, 1]])
    totalRot = np.matmul(xRot, np.matmul(yRot, zRot))

    #Rotate the point cloud and xyzt according to rotations chosen
    for i in range(len(xs)):
        xs[i], ys[i], zs[i] = np.matmul(totalRot, np.array([xs[i], ys[i], zs[i]]))
        xs[i] += translate[0]
        ys[i] += translate[1]
        zs[i] += translate[2]
    xyzt = np.matmul(totalRot, xyzt)
    xyzt = xyzt + translate

    #Apply perturbation to some points to investigate how error occurs when points are not
    #   In exact position after translation/rotation
    # xs += np.random.normal(0, .02, 4)
    # ys += np.random.normal(0, .02, 4)
    # zs += np.random.normal(0, .02, 4)
    #Error due to perturbation scales fast with perturbation magnitude

    xyzs_final = np.array([xs,ys,zs])
    xyzt_final = xyzt

    print(xyzs_init, '\n', xyzs_final)

    R_mtrx, d = findRotationAndDisplacement(xyzs_init, xyzs_final)

    print(d)
    print(xyzt_init)
    print(xyzt_final)
    print(np.matmul(R_mtrx, xyzt_init) + d)
    print(xyzt_final - (np.matmul(R_mtrx, xyzt_init) + d))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from plotPlane import plotPlane
    from printTable import printTable

    demo()