# Written by Stefan Kildal-Brandt

import numpy as np
from fitPlane import fitPlane
from projectToPlane import projectToPlane
from rodriguesRotationMatrix import rotationMatrix

def findTheta(barycenter, xyzt,  normVect): #Finds the angle between the normal vector and the vector between xyzt and plane
    disVect = np.array(xyzt-barycenter)
    dot = np.dot(disVect, normVect)
    disVect_Length = np.linalg.norm(disVect)
    normVect_Length = np.linalg.norm(normVect)
    theta = np.arccos(dot / (disVect_Length * normVect_Length))
    return theta

def findPhi(barycenter, xyzt, point, normVect): #Finds between a point and xyzt when they are projected onto the plane
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

def findLength(barycenter, xyzt): #Find length of vector between barycenter and xyzt
    length = np.linalg.norm(np.array(xyzt-barycenter))
    return length

def initializeFunction(xs, ys, zs, pointIdx, xyzt): #Given initial conditions, find the variables required for later analysis
    points = np.array([xs, ys, zs])
    barycenter = np.mean(points, axis=1)
    normVect = fitPlane(xs, ys, zs)
    #check orientation of normVect versus point so can correct for normVect direction later
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

### findTheta and findPhi and findLength above are for initialization with the t=0 state of the ship

### Below are the functions used finding the xyzt given what we know from the initial state

def findXyzt(xs, ys, zs, pointIdx, length, theta, phi, orientation): #Main function finding the xyzt given initial conditions
    barycenter = np.mean(np.array([xs, ys ,zs]), axis=1)
    normVect = fitPlane(xs, ys, zs)
    point = np.array([xs[pointIdx], ys[pointIdx], zs[pointIdx]])

    #Confirm same orientation otherwise invert the direction of the normal vector
    if (np.dot(point - barycenter, normVect) > 0) != orientation:
        normVect = normVect * -1
    #
    # if (np.dot(np.array([0,0,1]), normVect) > 0) != orientation:
    #     normVect = normVect * -1

    normVect_Length = np.linalg.norm(normVect)
    unitNorm = normVect / normVect_Length

    #Scale the normal vector to the length of the distance between barycenter and xyzt
    finalVector = normVect * length / normVect_Length

    #Rotate the scaled vector theta degrees around vector between barycenter and chosen point
    rotationPoint = projectToPlane(point, normVect)
    rotationPoint_Vect = rotationPoint - barycenter
    rotationVector = np.cross(normVect, rotationPoint_Vect)
    rotationVector = rotationVector / np.linalg.norm(rotationVector)
    Theta_Matrix = rotationMatrix(theta, rotationVector)
    finalVector = np.matmul(Theta_Matrix, finalVector)

    #Rotate the scaled vector phi degrees around the normal vector
    Phi_Matrix = rotationMatrix(phi, unitNorm)
    finalVector = np.matmul(Phi_Matrix, finalVector)

    #The scaled and rotated vector should now lie on the position of the xyzt
    return [finalVector, barycenter, normVect]

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
    xs = 10 random coordinates between -10 and 10
    ys = 10 random coordinates between -10 and 10
    zs = 10 random coordinates between -10 and 10
    xyzt = random between -25 and 25
    rot = randomly determined between pi and -pi for each axis (radians)
    translate = randomly determined -10 and 10 for each coordinate
"""
import matplotlib.pyplot as plt
from plotPlane import plotPlane

def demo(xs=np.random.rand(10)*10,
         ys=np.random.rand(10)*10,
         zs=np.random.rand(10)*25,
         xyzt=np.array([15,10,-20]),
         rot=np.random.rand(3)*np.pi,
         translate=np.random.rand(3)*10):
    [theta, phi, length, orientation] = initializeFunction(xs, ys, zs, 3, xyzt)

    xRot = np.array([[1, 0, 0], [0, np.cos(rot[0]), -np.sin(rot[0])], [0, np.sin(rot[0]), np.cos(rot[0])]])
    yRot = np.array([[np.cos(rot[1]), 0, np.sin(rot[1])], [0, 1, 0], [-np.sin(rot[1]), 0, np.cos(rot[1])]])
    zRot = np.array([[np.cos(rot[2]), -np.sin(rot[2]), 0], [np.sin(rot[2]), np.cos(rot[2]), 0], [0, 0, 1]])
    totalRot = np.matmul(xRot, np.matmul(yRot, zRot))

    for i in range(len(xs)):
        xs[i], ys[i], zs[i] = np.matmul(totalRot, np.array([xs[i], ys[i], zs[i]]))
        xs[i] +=translate[0]
        ys[i] += translate[1]
        zs[i] += translate[2]
    xyzt = np.matmul(totalRot, xyzt)
    xyzt = xyzt + translate

    finalVect, barycenter, normVect = findXyzt(xs, ys, zs, 3, length, theta, phi, orientation)
    print(finalVect, xyzt - barycenter)

    ax = plotPlane(barycenter, normVect, [min(xs), max(xs)], [min(ys), max(ys)])
    ax.scatter(xs, ys, zs, color='g')
    ax.scatter(xyzt[0], xyzt[1], xyzt[2], color='r')
    ax.quiver(barycenter[0], barycenter[1], barycenter[2], finalVect[0], finalVect[1], finalVect[2], color='k')

    plt.show()

demo()

# Runs fitPlane to find the plane through xyz and determines angles to xyzt
# theta is the colatitude with respect to the normal vector of the plane
# phi is the angle between the projection of xyzt onto the plane and one of the xyzs projected onto the plane