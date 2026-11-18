import warnings

import numpy as np
from numba import njit

from geometry.ECEF_Geodetic import ECEF_Geodetic
from Inversion_Workflow.Forward_Model.Find_ESV import find_esv


@njit
def calculateTimesRayTracing_Bias(
    guess, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix
):
    """Compute travel times with an optional ESV bias term.

    Parameters
    ----------
    guess : array-like, shape (3,)
        Current estimate of the source location.
    transponder_coordinates : ndarray
        ``(N, 3)`` array of transponder positions.
    esv_bias : float or ndarray
        Bias added to the effective sound velocity.
    dz_array, angle_array, esv_matrix : ndarray
        Lookup table for computing the ESV.

    Returns
    -------
    tuple of ndarray
        ``(times, esv)`` travel times and sound velocities.
    """
    hori_dist = np.sqrt(
        (transponder_coordinates[:, 0] - guess[0]) ** 2
        + (transponder_coordinates[:, 1] - guess[1]) ** 2
    )
    abs_dist = np.sqrt(np.sum((transponder_coordinates - guess) ** 2, axis=1))
    beta = np.arccos(hori_dist / abs_dist) * 180 / np.pi
    dz = np.abs(guess[2] - transponder_coordinates[:, 2])
    esv = find_esv(beta, dz, dz_array, angle_array, esv_matrix) + esv_bias
    times = abs_dist / esv
    return times, esv


@njit
def calculateTimesRayTracing_Bias_Real(
    guess, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix
):
    """Calculate times for real data using geodetic depths."""
    abs_dist = np.sqrt(np.sum((transponder_coordinates - guess) ** 2, axis=1))
    depth_arr = ECEF_Geodetic(transponder_coordinates)[2]

    guess = guess[np.newaxis, :]
    lat, lon, depth = ECEF_Geodetic(guess)
    dz = depth_arr - depth
    beta = np.arcsin(dz / abs_dist) * 180 / np.pi
    esv = find_esv(beta, dz, dz_array, angle_array, esv_matrix) + esv_bias
    times = abs_dist / esv
    return times, esv


def check_esv_coverage(
    guess,
    transponder_coordinates,
    dz_array,
    angle_array,
    real_data,
    name="ESV table",
    on_outside="raise",
):
    """Fail or warn if the (angle, dz) queried by the travel-time model leave a table.

    ``find_esv`` clamps such queries to the table edge, which silently changes the
    geometry. The queries are those of ``calculateTimesRayTracing_Bias_Real`` when
    ``real_data`` (geodetic depths) and of ``calculateTimesRayTracing_Bias``
    otherwise. ``on_outside`` is ``"raise"`` (ValueError) or ``"warn"``.
    """
    if on_outside not in ("raise", "warn"):
        raise ValueError(f"on_outside must be 'raise' or 'warn', got {on_outside!r}")
    guess = np.asarray(guess, dtype=np.float64)
    transponder_coordinates = np.asarray(transponder_coordinates, dtype=np.float64)
    abs_dist = np.sqrt(np.sum((transponder_coordinates - guess) ** 2, axis=1))
    if real_data:
        dz = ECEF_Geodetic(transponder_coordinates)[2] - ECEF_Geodetic(guess[None])[2]
        beta = np.degrees(np.arcsin(dz / abs_dist))
    else:
        hori_dist = np.hypot(
            transponder_coordinates[:, 0] - guess[0],
            transponder_coordinates[:, 1] - guess[1],
        )
        beta = np.degrees(np.arccos(hori_dist / abs_dist))
        dz = np.abs(guess[2] - transponder_coordinates[:, 2])

    problems = []
    for label, values, axis, unit in (
        ("dz", dz, dz_array, "m"),
        ("angle", beta, angle_array, "deg"),
    ):
        if values.min() < axis[0] or values.max() > axis[-1]:
            problems.append(
                f"{label} queried {values.min():.2f}-{values.max():.2f} {unit}, "
                f"table covers {axis[0]:.2f}-{axis[-1]:.2f} {unit}"
            )
    if problems:
        message = f"{name} does not cover the queries (clamped): " + "; ".join(problems)
        if on_outside == "raise":
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)
