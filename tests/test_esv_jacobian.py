import numpy as np
import pytest
from pymap3d import enu2ecef, geodetic2ecef

from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    calculateTimesRayTracing_Bias,
    calculateTimesRayTracing_Bias_Real,
    esv_gradient,
)
from Inversion_Workflow.Forward_Model.Find_ESV import find_esv, find_esv_gradient
from Inversion_Workflow.Inversion import Numba_xAline_Geiger_bias as geiger
from Inversion_Workflow.Inversion.Numba_xAline_Geiger_bias import (
    compute_Jacobian_biased,
)
from Inversion_Workflow.Synthetic.Generate_Unaligned import generateUnaligned

# --- A. Interpolation derivatives -------------------------------------------------
GRID_DZ = np.array([0.0, 1.0, 2.0])
GRID_ANGLE = np.array([20.0, 30.0, 40.0])
# c = 1500 + 2 dz + 0.5 beta + 0.1 dz beta (beta in degrees) is bilinear, so the
# interpolant is exact: dc/ddz = 2 + 0.1 beta, dc/dbeta = 0.5 + 0.1 dz.
BILINEAR = (
    1500.0
    + 2.0 * GRID_DZ[:, None]
    + 0.5 * GRID_ANGLE[None, :]
    + 0.1 * GRID_DZ[:, None] * GRID_ANGLE[None, :]
)
# Not globally bilinear: the derivative jumps across the dz = 1 and beta = 30 lines.
KINKED = np.array([[0.0, 1.0, 2.0], [0.0, 3.0, 2.0], [1.0, 1.5, 5.0]]) + 1500.0


def _gradient(dz, beta, matrix):
    dz, beta = np.atleast_1d(dz).astype(float), np.atleast_1d(beta).astype(float)
    return find_esv_gradient(beta, dz, GRID_DZ, GRID_ANGLE, matrix)


def test_gradient_of_a_bilinear_table_is_exact_with_beta_in_degrees():
    dz = np.array([0.3, 1.7, 0.0, 2.0, 1.0])
    beta = np.array([22.0, 38.5, 20.0, 40.0, 30.0])
    dc_ddz, dc_dbeta = _gradient(dz, beta, BILINEAR)
    np.testing.assert_allclose(dc_ddz, 2.0 + 0.1 * beta, atol=1e-12)
    np.testing.assert_allclose(dc_dbeta, 0.5 + 0.1 * dz, atol=1e-12)


def test_gradient_is_zero_along_a_clamped_axis_only():
    dc_ddz, dc_dbeta = _gradient(
        [-1.0, 3.0, 1.5, 1.5], [25.0, 25.0, 10.0, 50.0], KINKED
    )
    assert dc_ddz[0] == 0.0 and dc_ddz[1] == 0.0
    assert dc_dbeta[0] != 0.0 and dc_dbeta[1] != 0.0
    assert dc_dbeta[2] == 0.0 and dc_dbeta[3] == 0.0
    assert dc_ddz[2] != 0.0 and dc_ddz[3] != 0.0


def test_gradient_matches_one_sided_differences_at_grid_lines_and_edges():
    """On a grid line: lower-cell derivative; on an edge: the inside derivative."""
    h = 1e-6

    def esv(dz, beta):
        return find_esv(np.array([beta]), np.array([dz]), GRID_DZ, GRID_ANGLE, KINKED)[
            0
        ]

    for dz, beta in ((1.0, 25.0), (0.0, 25.0), (2.0, 25.0), (1.5, 30.0), (1.5, 20.0)):
        dc_ddz, dc_dbeta = (g[0] for g in _gradient(dz, beta, KINKED))
        inside_z = -h if dz == GRID_DZ[-1] else (-h if dz == GRID_DZ[1] else h)
        inside_b = -h if beta in (30.0, 40.0) else h
        np.testing.assert_allclose(
            dc_ddz, (esv(dz + inside_z, beta) - esv(dz, beta)) / inside_z, atol=1e-6
        )
        np.testing.assert_allclose(
            dc_dbeta, (esv(dz, beta + inside_b) - esv(dz, beta)) / inside_b, atol=1e-6
        )
    # The two sides at dz = 1 really differ, so a central difference is not defined.
    left = _gradient(1.0 - 1e-9, 25.0, KINKED)[0][0]
    right = _gradient(1.0 + 1e-9, 25.0, KINKED)[0][0]
    assert abs(left - right) > 0.5


# --- B. Full travel-time derivatives ----------------------------------------------
DZ = np.arange(5100.0, 5400.0, 1.0)
ANGLE = np.arange(0.0, 90.01, 1.0)
# Globally bilinear (with a dz * beta cross term) so central differences are valid
# across cell boundaries.
SMOOTH = (
    1500.0
    + 0.017 * (DZ[:, None] - 5200.0)
    + 0.05 * ANGLE[None, :]
    + 1e-4 * (DZ[:, None] - 5200.0) * ANGLE[None, :]
)
TABLES = (DZ, ANGLE, SMOOTH)
ESV_BIAS = 3.0
# Horizontal offsets (m) of the transducer from the DOG: elevations of about 79, 60,
# 45, 30 degrees and a near-vertical 89.4 degree path.
OFFSETS = np.array([1000.0, 3000.0, 5200.0, 9000.0, 50.0])
HEIGHTS = np.array(
    [0.0, 2.0, -3.0, 1.5, 4.0]
)  # transducer height above the sea surface


def _geometry(real_data):
    """DOG position and transducer positions, 5.2 km above it."""
    if real_data:
        lat, lon = 32.0, -64.5
        dog = np.array(geodetic2ecef(lat, lon, -5200.0))
        east = OFFSETS * np.cos(np.arange(5))
        north = OFFSETS * np.sin(np.arange(5))
        transducers = np.stack(
            enu2ecef(east, north, HEIGHTS + 5200.0, lat, lon, -5200.0), axis=1
        )
        return dog, transducers
    angle = np.arange(5)
    dog = np.array([100.0, -200.0, -5200.0])
    transducers = np.stack(
        [
            dog[0] + OFFSETS * np.cos(angle),
            dog[1] + OFFSETS * np.sin(angle),
            HEIGHTS,
        ],
        axis=1,
    )
    return dog, transducers


def _forward(real_data):
    return (
        calculateTimesRayTracing_Bias_Real
        if real_data
        else calculateTimesRayTracing_Bias
    )


def _central(function, x, h):
    """Central difference of function(x) with respect to each entry of x."""
    columns = []
    for k in range(len(x)):
        step = np.zeros(len(x))
        step[k] = h
        columns.append((function(x + step) - function(x - step)) / (2 * h))
    return np.stack(columns, axis=1)


@pytest.mark.parametrize("real_data", [False, True])
@pytest.mark.parametrize("h", [0.05, 0.25, 0.5])
def test_jacobian_matches_central_differences_of_the_forward_model(real_data, h):
    dog, transducers = _geometry(real_data)
    forward = _forward(real_data)
    times, esv = forward(dog, transducers, ESV_BIAS, *TABLES)
    grad_x_c0, _ = esv_gradient(dog, transducers, *TABLES, real_data)
    J = compute_Jacobian_biased(dog, transducers, times, esv, grad_x_c0)
    J_frozen = compute_Jacobian_biased(dog, transducers, times, esv)

    numeric = _central(lambda x: forward(x, transducers, ESV_BIAS, *TABLES)[0], dog, h)
    scale = np.abs(numeric).max(axis=1, keepdims=True)
    np.testing.assert_allclose(J[:, :3], numeric, rtol=0, atol=2e-4 * scale.max())
    assert np.all(np.abs(J[:, :3] - numeric) < 2e-4 * scale)
    # The frozen-ESV Jacobian is visibly wrong on the same data.
    assert (
        np.abs(J_frozen[:, :3] - numeric).max() > 20 * np.abs(J[:, :3] - numeric).max()
    )

    # Time bias enters the residual as -1; the additive ESV bias as dT/ddc.
    np.testing.assert_array_equal(J[:, 3], -1.0)
    d_bias = (
        forward(dog, transducers, ESV_BIAS + h, *TABLES)[0]
        - forward(dog, transducers, ESV_BIAS - h, *TABLES)[0]
    ) / (2 * h)
    np.testing.assert_allclose(J[:, 4], d_bias, rtol=1e-6)


@pytest.mark.parametrize("real_data", [False, True])
@pytest.mark.parametrize("h", [0.05, 0.5])
def test_transducer_and_lever_derivatives_match_central_differences(real_data, h):
    dog, transducers = _geometry(real_data)
    forward = _forward(real_data)
    times, esv = forward(dog, transducers, ESV_BIAS, *TABLES)
    _, grad_s_c0 = esv_gradient(dog, transducers, *TABLES, real_data)
    line_of_sight = transducers - dog
    line_of_sight /= np.linalg.norm(line_of_sight, axis=1)[:, None]
    dT_ds = (line_of_sight - times[:, None] * grad_s_c0) / esv[:, None]

    # Each epoch depends only on its own transducer position.
    numeric = np.empty((5, 3))
    for k in range(3):
        step = np.zeros(3)
        step[k] = h
        plus = forward(dog, transducers + step, ESV_BIAS, *TABLES)[0]
        minus = forward(dog, transducers - step, ESV_BIAS, *TABLES)[0]
        numeric[:, k] = (plus - minus) / (2 * h)
    np.testing.assert_allclose(
        dT_ds, numeric, rtol=0, atol=2e-4 * np.abs(numeric).max()
    )

    # Lever derivative through s_i = R_i @ lever + d_i with arbitrary rotations.
    rng = np.random.default_rng(3)
    rotation = np.stack([np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(5)])
    origin = transducers - rotation @ np.array([1.0, 2.0, -3.0])
    lever = np.array([1.0, 2.0, -3.0])
    analytic = np.einsum("ni,nij->nj", dT_ds, rotation)
    numeric_lever = _central(
        lambda v: forward(dog, origin + rotation @ v, ESV_BIAS, *TABLES)[0], lever, h
    )
    np.testing.assert_allclose(
        analytic, numeric_lever, rtol=0, atol=2e-4 * np.abs(numeric_lever).max()
    )


# --- C. Inversion behaviour -------------------------------------------------------
OTHERS = np.array(
    [[0.0, 0.0, 0.0], [-2.4, -4.2, 0.03], [-12.1, -0.95, 0.004], [-8.7, 5.17, 0.025]]
)
LEVER = np.array([-12.5, 9.6, -13.3])
OFFSET, TIME_BIAS, TRUE_ESV_BIAS = 3141.5926, 0.2, 2.0
# Strong enough dependence on dz and beta that the frozen Jacobian is a poor model.
STRONG = 1500.0 + 0.05 * (DZ[:, None] - 5200.0) + 1.0 * ANGLE[None, :] / 90.0 * 30.0
STRONG_TABLES = (DZ, ANGLE, STRONG)


def _finite_difference_gradient(
    guess, transducers, dz, angle, matrix, real_data, h=0.01
):
    """Reference for ``esv_gradient``: central differences of the forward-model c0."""
    forward = _forward(real_data)
    c0 = lambda x, s: forward(x, s, 0.0, dz, angle, matrix)[1]  # noqa: E731
    grad_x, grad_s = np.empty((len(transducers), 3)), np.empty((len(transducers), 3))
    for k in range(3):
        step = np.zeros(3)
        step[k] = h
        grad_x[:, k] = (
            c0(guess + step, transducers) - c0(guess - step, transducers)
        ) / (2 * h)
        grad_s[:, k] = (
            c0(guess, transducers + step) - c0(guess, transducers - step)
        ) / (2 * h)
    return grad_x, grad_s


def _frozen_gradient(guess, transducers, *args):
    return np.zeros((len(transducers), 3)), np.zeros((len(transducers), 3))


@pytest.fixture
def data():
    np.random.seed(0)
    return generateUnaligned(
        600, 2e-5, 0.02, OFFSET, TRUE_ESV_BIAS, TIME_BIAS, *STRONG_TABLES,
        gps1_to_others=OTHERS, gps1_to_transponder=LEVER,
    )  # fmt: skip


def _run(monkeypatch, gradient, data, free_lever):
    """Fit with ``gradient`` standing in for esv_gradient; count Jacobian calls."""
    CDOG_mat, CDOG, GPS_Coordinates, GPS_time, transponder = data
    calls = []
    jacobian = compute_Jacobian_biased
    monkeypatch.setattr(geiger, "esv_gradient", gradient)
    monkeypatch.setattr(
        geiger, "compute_Jacobian_biased", lambda *a: calls.append(1) or jacobian(*a)
    )
    start = CDOG + np.array([30.0, -20.0, 20.0])
    if free_lever:
        origin, rotation = geiger.lever_basis(GPS_Coordinates, OTHERS)
        estimate, lever, rms, n_pairs = geiger.lever_bias_geiger(
            start, CDOG_mat, GPS_time, origin, rotation,
            LEVER + np.array([2.0, -2.0, 1.0]), OFFSET + 0.1, *STRONG_TABLES,
        )  # fmt: skip
        return estimate, lever, rms, len(calls)
    result = geiger.final_bias_geiger(
        start,
        CDOG_mat,
        GPS_time,
        transponder,
        OFFSET + TIME_BIAS,
        0.0,
        0.0,
        *STRONG_TABLES,  # fmt: skip
    )
    estimate = result[0]
    rms = np.sqrt(np.mean((result[1] - result[2]) ** 2))
    return estimate, LEVER, rms, len(calls)


@pytest.mark.parametrize("free_lever", [False, True])
def test_analytical_jacobian_solver_matches_finite_difference_reference(
    monkeypatch, data, free_lever
):
    CDOG = data[1]

    def reference(guess, transducers, *args):
        return _finite_difference_gradient(guess, transducers, *args)

    analytic = _run(monkeypatch, esv_gradient, data, free_lever)
    numeric = _run(monkeypatch, reference, data, free_lever)
    frozen = _run(monkeypatch, _frozen_gradient, data, free_lever)

    for estimate, lever, rms, _ in (analytic, numeric):
        np.testing.assert_allclose(estimate[:3], CDOG, atol=0.05)
        np.testing.assert_allclose(lever, LEVER, atol=0.05 if free_lever else 1e-12)
        assert rms < 1e-4
    # Same stationary point as the finite-difference solver to well below the noise.
    np.testing.assert_allclose(analytic[0], numeric[0], rtol=0, atol=1e-3)
    np.testing.assert_allclose(analytic[1], numeric[1], rtol=0, atol=1e-3)
    # No worse than the frozen-ESV Jacobian in iterations (it is the exact Newton step).
    assert analytic[3] <= frozen[3]
