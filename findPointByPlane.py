import numpy as np
import random

### General Functions
def getPlane(xs, ys, zs): #General function using singular value decomposition to get least squares regression for plane
    points = np.array([xs,ys,zs])
    svd = np.linalg.svd(points - np.mean(points, axis=1, keepdims=True))
    left = svd[0]
    normVect = np.array(left[:, -1], dtype=np.float64)
    return normVect

def projectToPlane(pointVect, normVect): #Projects a vector to a plane by subtracting out its component parallel to normal vector
    dot = np.dot(pointVect, normVect)
    normVect_Length = np.linalg.norm(normVect)
    projection = pointVect - (dot * normVect / normVect_Length**2)
    return projection

### Initialization functions

def findTheta(barycenter, transponder,  normVect): #Finds the angle between the normal vector and the vector between transponder and plane
    disVect = np.array(transponder-barycenter)
    dot = np.dot(disVect, normVect)
    disVect_Length = np.linalg.norm(disVect)
    normVect_Length = np.linalg.norm(normVect)
    theta = np.arccos(dot / (disVect_Length * normVect_Length))
    return theta

def findPhi(barycenter, transponder, point, normVect): #Finds between a point and transponder when they are projected onto the plane
    pointVect = np.array(point-barycenter)
    pointProjection = projectToPlane(pointVect, normVect)
    distanceVect = np.array(transponder - barycenter)
    distanceProjection = projectToPlane(distanceVect, normVect)

    #Cannot use normal method of getting angle because rotation could be out of arccos range of [0, 180]
    #Use plane embedded in 3D instead - https://math.stackexchange.com/questions/878785/how-to-find-an-angle-in-range0-360-between-2-vectors
    det = np.dot(normVect, np.cross(pointProjection, distanceProjection))
    dot = np.dot(pointProjection, distanceProjection)
    phi = np.arctan2(det, dot)
    return phi

def lengthDistVect(barycenter, transponder): #Find length of vector between barycenter and transponder
    length = np.linalg.norm(np.array(transponder-barycenter))
    return length

def initializeFunction(xs, ys, zs, pointIdx, transponder): #Given initial conditions, find the variables required for later analysis
    points = np.array([xs, ys, zs])
    barycenter = np.mean(points, axis=1)
    normVect = getPlane(xs, ys, zs)
    #check orientation of normVect versus point so can correct for normVect direction later
    if np.dot(np.array(points[:, pointIdx] - barycenter), normVect) > 0:
        orientation = True
    else:
        orientation = False
    theta = findTheta(barycenter, transponder, normVect)
    phi = findPhi(barycenter, transponder, points[:, pointIdx], normVect)
    length = lengthDistVect(barycenter, transponder)
    return [theta, phi, length, orientation]

### findTheta and findPhi and lengthDistVect above are for initialization with the t=0 state of the ship

### Below are the functions used finding the transponder given what we know from the initial state

def rotationMatrix(angle, vect): #Creates a Rodrigues rotation matrix for given angle and unit-vector
    A_Matrix = np.array([[0, -vect[2], vect[1]],[vect[2], 0, -vect[0]], [-vect[1], vect[0], 0]])
    Rotation_Matrix = np.identity(3)+A_Matrix*np.sin(angle)+np.matmul(A_Matrix,A_Matrix)*(1-np.cos(angle))
    return Rotation_Matrix

def findTransponder(xs, ys, zs, pointIdx, length, theta, phi, orientation): #Main function finding the transponder given initial conditions
    #Initialize
    barycenter = np.mean(np.array([xs, ys ,zs]), axis=1)
    normVect = getPlane(xs, ys, zs)
    point = np.array([xs[pointIdx], ys[pointIdx], zs[pointIdx]])

    #Confirm same orientation otherwise invert the direction of the normal vector
    if (np.dot(point - barycenter, normVect) > 0) != orientation:
        normVect = normVect * -1

    normVect_Length = np.linalg.norm(normVect)
    unitNorm = normVect / normVect_Length

    #Scale the normal vector to the length of the distance between barycenter and transponder
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

    #The scaled and rotated vector should now lie on the position of the transponder
    return finalVector, barycenter

###TESTING BELOW

#Initialize
# xs = [-10, -10, 10, -10]
# ys = [-10, 10, -10, 10]
# zs = [0, 0, 0, 0]
# transponder = np.array([15,10,-20])
#
# [theta, phi, length, orientation] = initializeFunction(xs, ys, zs, 3, transponder)
#
# # print(theta, phi, length, orientation)
#
# test, barycenter = findTransponder(xs,ys,zs,3,length,theta,phi, orientation)
#
# print(test+barycenter, transponder)
#
# ## Rotate and see if it matches for randomly generated rotations
# for i in range(5):
#     roll = random.choice((-1, 1)) * random.random()*100 * np.pi/180
#     pitch = random.choice((-1, 1)) * random.random()*100 * np.pi/180
#     yaw = random.choice((-1, 1)) * random.random()*100 * np.pi/180
#
#     print(roll * 180/np.pi, pitch * 180/np.pi, yaw * 180/np.pi)
#
#
#     pitchChange = np.array([[1,0,0],[0,np.cos(pitch),-np.sin(pitch)],[0,np.sin(pitch),np.cos(pitch)]])
#     rollChange = np.array([[np.cos(roll),0,np.sin(roll)],[0,1,0],[-np.sin(roll),0,np.cos(roll)]])
#     yawChange = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
#
#     testRot = np.matmul(np.matmul(pitchChange, rollChange),yawChange)
#     newXs = [0]*len(xs)
#     newYs = [0]*len(ys)
#     newZs = [0]*len(zs)
#
#     for i in range(len(xs)):
#         newXs[i], newYs[i], newZs[i] = np.matmul(testRot, np.array([xs[i], ys[i],zs[i]]))
#
#     new_barycenter = np.array([sum(newXs)/4, sum(newYs)/4, sum(newZs)/4], dtype=np.float64)
#     new_transponder = np.matmul(testRot, transponder)
#
#     test2 = findTransponder(newXs,newYs, newZs,3,length,theta,phi, orientation)
#
#     print(test2, new_transponder-new_barycenter)
#if needed put under line 54, think this is deprecated at this point though
# if theta > np.pi:
#     theta = -theta
#     phi = np.pi - phi  # Subtract 180 degrees for theta change, then rotate angle of direction because normvect in wrong direction

#For gps data call variables such as ['x'] like that
#Use chat gpt to save in .mat file before returning information