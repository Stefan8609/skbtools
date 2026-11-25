"""Physically motivated perturbations of a reference sound-speed profile.

Used for the ESV tables that invert the synthetic data.

Internal-wave heave displaces the temperature-salinity part of the sound speed by
eta(z) = A * phi(z), phi the first vertical mode of a Garrett-Munk N(z):
c'(z) = c(z) + c_ts(z + eta) - c_ts(z), with c_ts = c - gamma z and gamma the deep
pressure gradient (not displacing pressure keeps the deep change ~0.01 m/s). A
mixed-layer term adds dc/dT * dT tapering to zero below the layer, dc/dT = 2.5 m/s
per degC (Medwin 1975).
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal

DEEP_GRADIENT_FROM_M = 4000.0  # the profile below this depth sets gamma
SEABED_DEPTH_M = 5250.0  # heave is zero at and below this depth
GM_N0_RAD_S = 5.24e-3  # Garrett-Munk surface buoyancy frequency
GM_SCALE_DEPTH_M = 1300.0  # Garrett-Munk stratification scale depth
DC_DT_MPS_PER_DEGC = 2.5  # sound-speed sensitivity to temperature at 24 degC

# Named settings that replace the old "perturbed" and "realistic_perturbed" tables.
# See build_synthetic_esv_tables.py for the resulting change in c(z) and in ESV.
SETTINGS = {
    "moderate": {"heave_amplitude_m": 10.0, "mixed_layer_dT": 0.0},
    "strong": {"heave_amplitude_m": 25.0, "mixed_layer_dT": -0.3},
}


def mode1_shape(
    depth,
    seabed_depth_m=SEABED_DEPTH_M,
    n0_rad_s=GM_N0_RAD_S,
    scale_depth_m=GM_SCALE_DEPTH_M,
    step_m=10.0,
):
    """First vertical-displacement mode, zero at the surface and seabed, peak 1."""
    z = np.arange(step_m, seabed_depth_m, step_m)
    n_squared = (n0_rad_s * np.exp(-z / scale_depth_m)) ** 2
    # -phi'' = lambda N^2 phi, symmetrised with psi = N phi.
    diagonal = 2.0 / (step_m**2 * n_squared)
    off_diagonal = -1.0 / (step_m**2 * np.sqrt(n_squared[:-1] * n_squared[1:]))
    _, psi = eigh_tridiagonal(diagonal, off_diagonal, select="i", select_range=(0, 0))
    phi = psi[:, 0] / np.sqrt(n_squared)
    phi = phi / phi[np.argmax(np.abs(phi))]
    z_nodes = np.concatenate(([0.0], z, [seabed_depth_m]))
    phi = np.concatenate(([0.0], phi, [0.0]))
    return np.interp(depth, z_nodes, phi, left=0.0, right=0.0)


def perturb_profile(
    depth,
    c,
    heave_amplitude_m=0.0,
    mixed_layer_dT=0.0,
    shape=None,
    mixed_layer_depth_m=30.0,
    mixed_layer_taper_m=30.0,
):
    """Perturbed profile: c(z + eta(z)) plus a mixed-layer change, m/s.

    ``depth`` (m, increasing) and ``c`` (m/s) are the reference profile.
    ``heave_amplitude_m`` is the peak displacement A, positive moving the profile
    upward (the temperature-salinity structure at z comes from deeper). ``shape`` is
    the mode shape sampled on ``depth`` (default: ``mode1_shape``).
    ``mixed_layer_dT`` is the surface temperature change in degC, confined to
    ``mixed_layer_depth_m`` and tapering to zero over a further
    ``mixed_layer_taper_m``.
    """
    depth = np.asarray(depth, dtype=float)
    c = np.asarray(c, dtype=float)
    if depth.ndim != 1 or c.shape != depth.shape:
        raise ValueError(
            f"depth and c must be matching 1D arrays, got {depth.shape} and {c.shape}"
        )
    if shape is None:
        shape = mode1_shape(depth)
    eta = heave_amplitude_m * shape
    deep = depth >= DEEP_GRADIENT_FROM_M
    pressure_gradient = 0.0
    if deep.sum() > 1:
        pressure_gradient = np.polyfit(depth[deep], c[deep], 1)[0]
    c_ts = c - pressure_gradient * depth
    c_perturbed = c + (np.interp(depth + eta, depth, c_ts) - c_ts)
    below_mixed_layer = np.clip(
        (depth - mixed_layer_depth_m) / mixed_layer_taper_m, 0.0, 1.0
    )
    weight = 0.5 * (1.0 + np.cos(np.pi * below_mixed_layer))
    return c_perturbed + DC_DT_MPS_PER_DEGC * mixed_layer_dT * weight


def named_perturbation(name, depth, c):
    """Perturb ``c`` with one of the named ``SETTINGS`` ("moderate", "strong")."""
    if name not in SETTINGS:
        raise ValueError(f"name must be one of {sorted(SETTINGS)}, got {name!r}")
    return perturb_profile(depth, c, **SETTINGS[name])
