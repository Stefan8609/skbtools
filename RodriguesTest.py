import numpy as np
from fitPlane import fitPlane
from projectToPlane import projectToPlane
from rodriguesRotationMatrix import rotationMatrix

def rodrigues_rotation(xs, ys, zs, new_xs, new_ys, new_zs):
    n1 = fitPlane(xs,ys,zs)
    n2 = fitPlane(new_xs, new_ys, new_zs)
    cross = np.cross(n1, n2)
    cross_mag = np.linalg.norm(cross)
    theta = np.arcsin(cross_mag)
    print(theta)
    k = cross/cross_mag
    R = rotationMatrix(theta, k)
    return R


def demo(xs=np.random.rand(4)*10-5,
         ys=np.random.rand(4)*10-5,
         zs=np.random.rand(4)*2-1,
         xyzt=np.random.rand(3)*30-25,
         rot=np.random.rand(3)*np.pi/2-np.pi/4,
         translate=np.random.rand(3)*10-5):


    xRot = np.array([[1, 0, 0], [0, np.cos(rot[0]), -np.sin(rot[0])], [0, np.sin(rot[0]), np.cos(rot[0])]])
    yRot = np.array([[np.cos(rot[1]), 0, np.sin(rot[1])], [0, 1, 0], [-np.sin(rot[1]), 0, np.cos(rot[1])]])
    zRot = np.array([[np.cos(rot[2]), -np.sin(rot[2]), 0], [np.sin(rot[2]), np.cos(rot[2]), 0], [0, 0, 1]])
    totalRot = np.matmul(xRot, np.matmul(yRot, zRot))

    #Rotate the point cloud and xyzt according to rotations chosen
    new_xs = np.copy(xs)
    new_ys = np.copy(ys)
    new_zs = np.copy(zs)
    new_xyzt = np.copy(xyzt)
    for i in range(len(new_xs)):
        new_xs[i], new_ys[i], new_zs[i] = np.matmul(totalRot, np.array([new_xs[i], new_ys[i], new_zs[i]]))
        new_xs[i] += translate[0]
        new_ys[i] += translate[1]
        new_zs[i] += translate[2]
    new_xyzt = np.matmul(totalRot, new_xyzt)
    new_xyzt = new_xyzt + translate

    #Apply perturbation to some points to investigate how error occurs when points are not
    #   In exact position after translation/rotation
    # new_xs += np.random.normal(0, .02, 4)
    # new_ys += np.random.normal(0, .02, 4)
    # New_zs += np.random.normal(0, .02, 4)
    #Error due to perturbation scales fast with perturbation magnitude

    R = rodrigues_rotation(xs, ys, zs, new_xs, new_ys, new_zs)

    points = np.array([xs,ys,zs]).T
    centroid = np.mean(points, axis=0)

    p = xyzt - centroid

    new_p = R @ p

    print(np.arccos(np.dot(p, new_p)/(np.linalg.norm(p) * np.linalg.norm(new_p))))

    new_points = np.array([new_xs, new_ys, new_zs]).T
    new_centroid = np.mean(new_points, axis=0)

    guess_xyzt = new_centroid + new_p

    print(np.arccos(np.dot(xyzt, new_xyzt)/(np.linalg.norm(xyzt)*np.linalg.norm(new_xyzt))))

    print(guess_xyzt, new_xyzt)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from plotPlane import plotPlane
    from printTable import printTable

    xs = np.array([0, -2.4054, -12.11, -8.7])
    ys = np.array([0, -4.21, -0.956, 5.165])
    zs = np.array([0, 0.060621, 0.00877, 0.0488])
    xyzt = np.array([-6.4, 2.46, -15.24])

    demo(xs, ys, zs, xyzt)