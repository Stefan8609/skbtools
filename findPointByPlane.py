"""
Given an initial cloud of points with a known point of interest, this algorithm finds the point of interest
    after the cloud of points has been translated/rotated

Given a cloud of n points and one additional independent point (POI), determines the relative position of the POI
with respect to that cloud by first characterizing the cloud by its plane of best fit (found using SVD) and

Written by Stefan Kildal-Brandt

2 functions: Initialization and findXyzt

Initialization - Given the cloud of points and known point of interest, this function finds the latitude angle, colatitude angle, and length
    from the barycenter of the point cloud to the point of interest,
Inputs:
    xs (list len=n) = x coordinates of points
    ys (list len=n) = y coordinates of points
    zs (list len=n) = z coordinates of points   (xs, ys, zs are initial states used for initialization purposes)
    pointIdx (int (0, n-1)) = Index of the point in cloud that will be used as a reference
    xyzt (list len=3) = The coordinates of the point of interest that we wish to find
Outputs:
    theta (float) = Latitude angle in radians from the plane of best fit from point cloud to xyzt
    phi (float) = Colatitude angle in radians along the plane of best fit from the chosen reference point to xyzt
    length (float) = Length of vector between barycenter of point cloud and xyzt
    orientation (boolean) = True/False depending on what side of the plane the normal vector points out of
    
findXyzt - Given the information found in the initialization function and a new set of xyz points in the point cloud, this function
    finds where the xyzt point is relative to the new plane of best fit
Inputs:
    xs (list len=n) = x coordinates of points
    ys (list len=n) = y coordinates of points
    zs (list len=n) = z coordinates of points   (xy, ys, zs in translated/rotated point cloud)
    pointIdx (int (0, n-1)) = Index of the point in cloud that will be used as a reference
    theta (float) = Latitude angle in radians from the plane of best fit from point cloud to xyzt
    phi (float) = Colatitude angle in radians along the plane of best fit from the chosen reference point to xyzt
    length (float) = Length of vector between barycenter of point cloud and xyzt
    orientation (boolean) = True/False depending on what side of the plane the normal vector points out of
Outputs:
    xyztVector: Vector between barycenter of the translated/rotated point cloud and the new xyzt point
    barycenter: Barycenter of the translated/rotated point cloud
    normVector: Normal vector of the plane of best fit for the translated/rotated point cloud
"""

import numpy as np
from fitPlane import fitPlane
from projectToPlane import projectToPlane
from rodriguesRotationMatrix import rotationMatrix

def findTheta(barycenter, xyzt,  normVect):
    disVect = np.array(xyzt-barycenter)
    dot = np.dot(disVect, normVect)
    disVect_Length = np.linalg.norm(disVect)
    normVect_Length = np.linalg.norm(normVect)
    theta = np.arccos(dot / (disVect_Length * normVect_Length))
    return theta

def findPhi(barycenter, xyzt, point, normVect):
    pointVect = np.array(point-barycenter)
    pointProjection = projectToPlane(pointVect, normVect)
    distanceVect = np.array(xyzt - barycenter)
    distanceProjection = projectToPlane(distanceVect, normVect)

    #Cannot use normal method of getting angle because rotation could be out of arccos range of [0, 180]
    #Use plane embedded in 3D instead - https://math.stackexchange.com/questions/878785/how-to-find-an-angle-in-range0-360-between-2-vectors
    det = np.dot(normVect, np.cross(pointProjection, distanceProjection))
    dot = np.dot(pointProjection, distanceProjection)
    phi = np.arctan2(det, dot)
    return phi

def findLength(barycenter, xyzt):
    length = np.linalg.norm(np.array(xyzt-barycenter))
    return length

def initializeFunction(xs, ys, zs, pointIdx, xyzt):
    points = np.array([xs, ys, zs])
    barycenter = np.mean(points, axis=1)
    normVect = fitPlane(xs, ys, zs)
    # check orientation of normVect versus point so can correct for normVect direction later
    # if np.dot(np.array([0,0,1]), normVect) > 0: #Testing this method to see if it tracks orientation better for non-rigid gps
    #     orientation=True
    # else:
    #     orientation=False

    if np.dot(np.array(points[:, pointIdx] - barycenter), normVect) > 0:
        orientation = True
    else:
        orientation = False
    theta = findTheta(barycenter, xyzt, normVect)
    phi = findPhi(barycenter, xyzt, points[:, pointIdx], normVect)
    length = findLength(barycenter, xyzt)
    return [theta, phi, length, orientation]

def findXyzt(xs, ys, zs, pointIdx, length, theta, phi, orientation): #Main function finding the xyzt given initial conditions
    barycenter = np.mean(np.array([xs, ys ,zs]), axis=1)
    normVect = fitPlane(xs, ys, zs)
    point = np.array([xs[pointIdx], ys[pointIdx], zs[pointIdx]])

    # Confirm same orientation otherwise invert the direction of the normal vector
    if (np.dot(point - barycenter, normVect) > 0) != orientation:
        normVect = normVect * -1

    # if (np.dot(np.array([0,0,1]), normVect) > 0) != orientation:
    #     normVect = normVect * -1

    normVect_Length = np.linalg.norm(normVect)
    unitNorm = normVect / normVect_Length

    #Scale the normal vector to the length of the distance between barycenter and xyzt
    xyztVector = normVect * length / normVect_Length

    #Rotate the scaled vector theta degrees around vector between barycenter and chosen point
    rotationPoint = projectToPlane(point, normVect)
    rotationPoint_Vect = rotationPoint - barycenter
    rotationVector = np.cross(normVect, rotationPoint_Vect)
    rotationVector = rotationVector / np.linalg.norm(rotationVector)
    Theta_Matrix = rotationMatrix(theta, rotationVector)
    xyztVector = np.matmul(Theta_Matrix, xyztVector)

    #Rotate the scaled vector phi degrees around the normal vector
    Phi_Matrix = rotationMatrix(phi, unitNorm)
    xyztVector = np.matmul(Phi_Matrix, xyztVector)

    #The scaled and rotated vector should now lie on the position of the xyzt
    return [xyztVector, barycenter, normVect]

"""
Demo - Demonstrate how a point is inversely found using this method after initialization
Input:
    xs (list len=n) = x coordinates of points   
    ys (list len=n) = y coordinates of points
    zs (list len=n) = z coordinates of points   (xs, ys, zs are initial states used for initialization purposes)
    xyzt (list len=3) = xyz of point that we want to find (xyzt is the initial state of this point used for initialization)
    rot (list len=3) = list containing the xyz rotations applied to points
    translate (list len=3) = list containing the xyz translations applied to points
Default:
    xs = 10 random coordinates between 0 and 10
    ys = 10 random coordinates between 0 and 10
    zs = 10 random coordinates between 0 and 25
    xyzt = random between 0 and 25
    rot = randomly determined between 0 and pi/2 for each axis (radians)
    translate = randomly determined -10 and 10 for each coordinate

"""
def demo(xs=np.random.rand(4)*10-20,
         ys=np.random.rand(4)*10-20,
         zs=np.random.rand(4)*5-4,
         xyzt=np.random.rand(3)*30-25,
         rot=np.random.rand(3)*np.pi/2-np.pi/4,
         translate=np.random.rand(3)*10-5):
    [theta0, phi0, length0, orientation0] = initializeFunction(xs, ys, zs, 0, xyzt)
    [theta1, phi1, length1, orientation1] = initializeFunction(xs, ys, zs, 1, xyzt)
    [theta2, phi2, length2, orientation2] = initializeFunction(xs, ys, zs, 2, xyzt)
    [theta3, phi3, length3, orientation3] = initializeFunction(xs, ys, zs, 3, xyzt)


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
    xs += np.random.normal(0, .02, 4)
    ys += np.random.normal(0, .02, 4)
    zs += np.random.normal(0, .02, 4)
    #Error due to perturbation scales fast with perturbation magnitude

    finalVect0, barycenter, normVect = findXyzt(xs, ys, zs, 0, length0, theta0, phi0, orientation0)
    finalVect1, barycenter, normVect = findXyzt(xs, ys, zs, 1, length1, theta1, phi1, orientation1)
    finalVect2, barycenter, normVect = findXyzt(xs, ys, zs, 2, length2, theta2, phi2, orientation2)
    finalVect3, barycenter, normVect = findXyzt(xs, ys, zs, 3, length3, theta3, phi3, orientation3)

    all_final_vect = np.array([finalVect0, finalVect1, finalVect2, finalVect3])
    average_final_vect = np.mean(all_final_vect, axis=0)
    print(all_final_vect)
    print(average_final_vect)

    #Plot the point cloud, plane of best fit, and vector to xyzt
    ax = plotPlane(barycenter, normVect, [min(xs), max(xs)], [min(ys), max(ys)])

    ax.scatter(xs, ys, zs, color='g')
    ax.scatter(xyzt[0], xyzt[1], xyzt[2], color='r')
    ax.quiver(barycenter[0], barycenter[1], barycenter[2], average_final_vect[0], average_final_vect[1], average_final_vect[2], color='k')

    data = []
    for i in range(len(xs)):
        tup = (f"point {i}:", xs[i], ys[i], zs[i])
        data.append(tup)
    data.append(("xyzt predicted:", barycenter[0]+average_final_vect[0], barycenter[1]+average_final_vect[1], barycenter[2]+average_final_vect[2]))
    data.append(("xyzt actual:", xyzt[0], xyzt[1], xyzt[2]))
    data.append(("Error (x,y,z):", barycenter[0]+average_final_vect[0]-xyzt[0],barycenter[1]+average_final_vect[1]-xyzt[1], barycenter[2]+average_final_vect[2]-xyzt[2]))
    data.append(("Error1 (x,y,z):", barycenter[0]+finalVect0[0]-xyzt[0],barycenter[1]+finalVect0[1]-xyzt[1], barycenter[2]+finalVect0[2]-xyzt[2]))

    printTable(["Point","X","Y", "Z"], data)
    plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from plotPlane import plotPlane
    from printTable import printTable

    # xs = np.array([0, -2.4054, -12.11, -8.7])
    # ys = np.array([0, -4.21, -0.956, 5.165])
    # zs = np.array([0, 0.060621, 0.00877, 0.0488])
    # xyzt = np.array([-6.4, 2.46, -15.24])
    # demo(xs, ys, zs, xyzt)

    demo()