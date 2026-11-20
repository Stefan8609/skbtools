import warnings

import numpy as np
from numba import njit

from geometry.ECEF_Geodetic import ECEF_Geodetic
from Inversion_Workflow.Forward_Model.Find_ESV import find_esv, find_esv_gradient

# Nominal sound speed (m/s) that turns travel-time residuals into range, as in the
# paper's c*sigma. Reporting only; the forward model uses the ESV table.
NOMINAL_SOUND_SPEED_MPS = 1515.0


@njit
def calculateTimesRayTracing_Bias(
    guess, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix
):
    """Travel times (s) and ESVs (m/s) to each transducer, depths from ECEF z.

    Eq. A18: T = |x - s| / (c_eff(beta, dz) + esv_bias), with beta the elevation
    angle (deg) and dz the vertical separation (m) looked up in the ESV table. Used
    for the synthetics.
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
    """As ``calculateTimesRayTracing_Bias``, with dz and beta from geodetic heights."""
    abs_dist = np.sqrt(np.sum((transponder_coordinates - guess) ** 2, axis=1))
    depth_arr = ECEF_Geodetic(transponder_coordinates)[2]

    guess = guess[np.newaxis, :]
    lat, lon, depth = ECEF_Geodetic(guess)
    dz = depth_arr - depth
    beta = np.arcsin(dz / abs_dist) * 180 / np.pi
    esv = find_esv(beta, dz, dz_array, angle_array, esv_matrix) + esv_bias
    times = abs_dist / esv
    return times, esv


@njit
def modelled_times(
    guess,
    transponder_coordinates,
    esv_bias,
    dz_array,
    angle_array,
    esv_matrix,
    real_data,
):
    """Travel times and ESVs: geodetic depths for real data, ECEF z for synthetics."""
    if real_data:
        return calculateTimesRayTracing_Bias_Real(
            guess, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix
        )
    return calculateTimesRayTracing_Bias(
        guess, transponder_coordinates, esv_bias, dz_array, angle_array, esv_matrix
    )


@njit
def esv_gradient(
    guess, transponder_coordinates, dz_array, angle_array, esv_matrix, real_data
):
    """Gradients of the reference ESV c0 with respect to the DOG and transducer.

    Returns ``(grad_x, grad_s)``, each ``(N, 3)`` ECEF/Cartesian (m/s)/m, by the
    chain rule through the geometry of ``calculateTimesRayTracing_Bias`` (Cartesian:
    dz = |z_x - z_s|, beta = arccos(horizontal / range)) or ``..._Real`` (dz = h_s -
    h_x geodetic height difference, beta = arcsin(dz / range), with grad h the
    ellipsoid normal). The table's beta is in degrees. Where beta is +-90 degrees
    (vertical path, direction undefined) its gradient is set to 0; the table clamps
    there. Gives the -(T/c) grad c_eff term of the Jacobian, which the paper's A26
    leaves out (it holds c_eff fixed).
    """
    to_deg = 180.0 / np.pi
    d = guess - transponder_coordinates  # x - s
    abs_dist = np.sqrt(np.sum(d**2, axis=1))
    n = d.shape[0]
    dz_grad_x = np.zeros((n, 3))
    dz_grad_s = np.zeros((n, 3))
    if not real_data:
        hori_dist = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        beta = np.arccos(hori_dist / abs_dist) * to_deg
        dz = np.abs(d[:, 2])
        sign = np.sign(d[:, 2])
        dz_grad_x[:, 2] = sign
        # beta = atan2(|dz|, horizontal)
        beta_grad_x = np.empty((n, 3))
        safe_hori = np.maximum(hori_dist, 1e-12)
        beta_grad_x[:, 0] = -dz * d[:, 0] / (safe_hori * abs_dist**2)
        beta_grad_x[:, 1] = -dz * d[:, 1] / (safe_hori * abs_dist**2)
        beta_grad_x[:, 2] = sign * hori_dist / abs_dist**2
        beta_grad_x *= to_deg
        beta_grad_s = -beta_grad_x
        dz_grad_s = -dz_grad_x
    else:
        lat_s, lon_s, height_s = ECEF_Geodetic(transponder_coordinates)
        lat_x, lon_x, height_x = ECEF_Geodetic(guess[np.newaxis, :])
        normal_s = np.empty((n, 3))
        normal_s[:, 0] = np.cos(np.radians(lat_s)) * np.cos(np.radians(lon_s))
        normal_s[:, 1] = np.cos(np.radians(lat_s)) * np.sin(np.radians(lon_s))
        normal_s[:, 2] = np.sin(np.radians(lat_s))
        normal_x = np.array(
            [
                np.cos(np.radians(lat_x[0])) * np.cos(np.radians(lon_x[0])),
                np.cos(np.radians(lat_x[0])) * np.sin(np.radians(lon_x[0])),
                np.sin(np.radians(lat_x[0])),
            ]
        )
        dz = height_s - height_x[0]
        sin_beta = dz / abs_dist
        beta = np.arcsin(sin_beta) * to_deg
        los = -d / abs_dist[:, None]  # unit vector from the DOG to the transducer
        dz_grad_x[:] = -normal_x
        dz_grad_s = normal_s
        cos_beta = np.sqrt(np.maximum(1.0 - sin_beta**2, 0.0))
        scale = np.where(cos_beta > 1e-9, to_deg / np.maximum(cos_beta, 1e-9), 0.0)
        scale = scale / abs_dist
        beta_grad_x = (dz_grad_x + sin_beta[:, None] * los) * scale[:, None]
        beta_grad_s = (dz_grad_s - sin_beta[:, None] * los) * scale[:, None]
    dc_ddz, dc_dbeta = find_esv_gradient(beta, dz, dz_array, angle_array, esv_matrix)
    grad_x = dc_ddz[:, None] * dz_grad_x + dc_dbeta[:, None] * beta_grad_x
    grad_s = dc_ddz[:, None] * dz_grad_s + dc_dbeta[:, None] * beta_grad_s
    return grad_x, grad_s


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
