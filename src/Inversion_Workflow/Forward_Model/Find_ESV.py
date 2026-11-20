import numpy as np
import scipy.io as sio
from numba import njit


def load_esv_table(path):
    """Read an ESV table .mat file as (dz_array, angle_array, esv_matrix).

    The matrix is returned as loadmat gives it (Fortran order).
    """
    table = sio.loadmat(path)
    return table["distance"].flatten(), table["angle"].flatten(), table["matrice"]


@njit
def find_esv(beta, dz, dz_array, angle_array, esv_matrix):
    """Effective sound velocity (m/s) at each (beta in deg, dz in m), bilinear.

    Eqs. A16-A17 take the nearest table node; here the table is interpolated, which
    keeps the modelled travel times continuous in the geometry. A stepwise lookup
    moves a pair to the next table row (about 2 cm of range in the Bermuda table) as
    soon as its vertical offset crosses a 1 m grid line, which makes the likelihood
    jagged at the scale of the posterior. Queries outside the table are clamped to
    its edges. ``esv_matrix`` is ``(len(dz_array), len(angle_array))``.
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


@njit
def find_esv_gradient(beta, dz, dz_array, angle_array, esv_matrix):
    """Partial derivatives of the ``find_esv`` interpolant, from the same cell.

    Returns ``(dc_ddz, dc_dbeta)`` in (m/s)/m and (m/s)/degree. Conventions where
    the interpolant is not differentiable: a query exactly on an interior grid line
    takes the derivative of the lower cell (the one ``find_esv`` interpolates in); a
    query strictly outside an axis is clamped by ``find_esv``, so the derivative
    along that axis is 0; a query exactly on an axis edge takes the inside cell's
    derivative (the one-sided derivative from the table side).
    """
    dc_ddz = np.zeros(dz.shape[0], dtype=np.float64)
    dc_dbeta = np.zeros(dz.shape[0], dtype=np.float64)
    last_row = dz_array.shape[0] - 2
    last_col = angle_array.shape[0] - 2

    for k in range(dz.shape[0]):
        z = min(max(dz[k], dz_array[0]), dz_array[-1])
        b = min(max(beta[k], angle_array[0]), angle_array[-1])

        i = min(max(np.searchsorted(dz_array, z) - 1, 0), last_row)
        j = min(max(np.searchsorted(angle_array, b) - 1, 0), last_col)
        width_z = dz_array[i + 1] - dz_array[i]
        width_b = angle_array[j + 1] - angle_array[j]
        fz = (z - dz_array[i]) / width_z
        fb = (b - angle_array[j]) / width_b

        if dz_array[0] <= dz[k] <= dz_array[-1]:
            dc_ddz[k] = (
                (1.0 - fb) * (esv_matrix[i + 1, j] - esv_matrix[i, j])
                + fb * (esv_matrix[i + 1, j + 1] - esv_matrix[i, j + 1])
            ) / width_z
        if angle_array[0] <= beta[k] <= angle_array[-1]:
            dc_dbeta[k] = (
                (1.0 - fz) * (esv_matrix[i, j + 1] - esv_matrix[i, j])
                + fz * (esv_matrix[i + 1, j + 1] - esv_matrix[i + 1, j])
            ) / width_b

    return dc_ddz, dc_dbeta
