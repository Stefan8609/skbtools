"""
Creates a Rodrigues rotation matrix for given angle and unit-vector
By Stefan Kildal-Brandt

Inputs:
    angle (int) = The angle which the rotation matrix rotates a vector
    vect (list len=3) = the unit vector to use as axis of rotation
Output:
    Rotation_Matrix (array 3x3) = The Rodrigues rotation matrix
"""

import numpy as np


def rotationMatrix(angle, vect):
    """Return the Rodrigues rotation matrix for a rotation about ``vect``.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.
    vect : array-like of float, shape (3,)
        Unit vector describing the axis of rotation.

    Returns
    -------
    numpy.ndarray
        ``3 x 3`` rotation matrix corresponding to the rotation.
    """

    A_Matrix = np.array(
        [[0, -vect[2], vect[1]], [vect[2], 0, -vect[0]], [-vect[1], vect[0], 0]]
    )
    Rotation_Matrix = (
        np.identity(3)
        + A_Matrix * np.sin(angle)
        + np.matmul(A_Matrix, A_Matrix) * (1 - np.cos(angle))
    )
    return Rotation_Matrix
