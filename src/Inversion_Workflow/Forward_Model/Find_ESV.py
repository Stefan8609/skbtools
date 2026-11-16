import numpy as np
from numba import njit


@njit
def find_esv(beta, dz, dz_array, angle_array, esv_matrix):
    """Bilinearly interpolate effective sound velocities from the ESV table.

    Interpolation keeps the modelled travel times continuous in the geometry. A
    stepwise lookup moves a pair to the next table row (about 2 cm of range in the
    Bermuda table) as soon as its vertical offset crosses a 1 m grid line, which
    makes the likelihood jagged at the scale of the posterior. Queries outside the
    table are clamped to its edges.

    Parameters
    ----------
    beta : ndarray
        Ray takeoff angles in degrees.
    dz : ndarray
        Vertical distance between the receiver and the source, in metres.
    dz_array, angle_array : ndarray
        Increasing grids of the ESV table (rows and columns of ``esv_matrix``).
    esv_matrix : ndarray
        Effective sound velocity table, shape ``(len(dz_array), len(angle_array))``.

    Returns
    -------
    ndarray
        Effective sound velocity for each input pair.
    """
    esv = np.empty(dz.shape[0], dtype=np.float64)
    last_row = dz_array.shape[0] - 2
    last_col = angle_array.shape[0] - 2

    for k in range(dz.shape[0]):
        z = min(max(dz[k], dz_array[0]), dz_array[-1])
        b = min(max(beta[k], angle_array[0]), angle_array[-1])

        i = min(max(np.searchsorted(dz_array, z) - 1, 0), last_row)
        j = min(max(np.searchsorted(angle_array, b) - 1, 0), last_col)
        fz = (z - dz_array[i]) / (dz_array[i + 1] - dz_array[i])
        fb = (b - angle_array[j]) / (angle_array[j + 1] - angle_array[j])

        esv[k] = (
            (1.0 - fz) * (1.0 - fb) * esv_matrix[i, j]
            + fz * (1.0 - fb) * esv_matrix[i + 1, j]
            + (1.0 - fz) * fb * esv_matrix[i, j + 1]
            + fz * fb * esv_matrix[i + 1, j + 1]
        )

    return esv
