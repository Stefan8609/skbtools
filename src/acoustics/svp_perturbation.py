"""Physically motivated perturbations of a reference sound-speed profile.

Used to build the ESV tables that invert the synthetic data (the generation table
uses the unperturbed profile). Everything here is a pure function of its arguments
(no I/O, no global state), so a later time- or space-dependent experiment can drive
``heave_amplitude_m`` and ``mixed_layer_dT`` with A(t) or A(x, y), for example an M2
internal tide.

Physics
-------
Internal waves and tides move water parcels vertically, so the sound speed seen at
depth z is the reference value from a displaced depth, c'(z) = c(z + eta(z)), with
eta(z) = A * phi(z). ``phi`` is the first vertical mode of the displacement,
phi'' + lambda N(z)^2 phi = 0 with phi = 0 at the surface and the seabed, normalised
to a maximum of one, and ``A`` is the peak displacement in metres. For a
Garrett-Munk buoyancy profile N(z) = N0 exp(-z / b) the mode peaks near 1 km depth,
in the thermocline, and decays towards the seabed. The cast's own temperature and
salinity would give N(z) directly, but that needs a seawater equation of state that
this repository does not carry, and only the position of the peak depends on it.

A mixed-layer term represents a surface temperature change dT (a few tenths of a
degree), c' += dc/dT * dT * w(z), with w = 1 in the mixed layer tapering to 0 below
it. dc/dT = 2.5 m/s per degC at the 24 degC surface of the Bermuda cast (Medwin 1975).

Heave moves the temperature and salinity structure, not the pressure, so only the
temperature-salinity part of the sound speed is displaced:
c'(z) = c(z) + [c_ts(z + eta) - c_ts(z)] with c_ts = c - gamma z, where gamma is the
pressure gradient of sound speed, taken from the deep slope of the reference profile
(below ``DEEP_GRADIENT_FROM_M``, where temperature and salinity are nearly uniform).
Deep water therefore changes by only about 0.01 m/s, as observed, whereas displacing
the full c(z) would add gamma * eta = 0.017 m/s per metre of heave at any depth.
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
    """Return the perturbed sound speed c(z + eta(z)) + mixed-layer change, m/s.

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
