"""Projection of a vector onto a plane. Written by Stefan Kildal-Brandt."""

import numpy as np


def projectToPlane(pointVect, normVect):
    """``pointVect`` minus its component along ``normVect``."""

    dot = np.dot(pointVect, normVect)
    normVect_Length = np.linalg.norm(normVect)
    projection = pointVect - (dot * normVect / normVect_Length**2)
    return projection
