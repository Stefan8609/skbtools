"""
Projects a vector to a plane by subtracting out its component parallel to normal vector
Written by Stefan Kildal-Brandt

Inputs:
    pointVect (list len=3) = The vector that is going to be projected
    normVect (list len=3) = The vector normal to the desired plane
Output:
    projection (list len=3) = The projected vector
"""

import numpy as np

def projectToPlane(pointVect, normVect):
    dot = np.dot(pointVect, normVect)
    normVect_Length = np.linalg.norm(normVect)
    projection = pointVect - (dot * normVect / normVect_Length**2)
    return projection