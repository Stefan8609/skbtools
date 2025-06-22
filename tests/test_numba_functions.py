import numpy as np
from pymap3d import ecef2geodetic

from GeigerMethod.Synthetic.Numba_Functions.ECEF_Geodetic import (
    ECEF_Geodetic,
)
from GeigerMethod.Synthetic.Numba_Functions.Numba_RigidBodyMovementProblem import (
    findRotationAndDisplacement,
)
from GeigerMethod.Synthetic.Numba_Functions import Numba_Geiger as ng


def test_ecef_geodetic_against_pymap3d():
    coords = np.array(
        [
            [6378137.0, 0.0, 0.0],
            [3912960.83742374, 2259148.99281506, 4488055.51564711],
            [2764344.82599736, -4787985.68826758, -3170623.73538364],
        ]
    )
    lat, lon, h = ECEF_Geodetic(coords)
    expected = np.array([ecef2geodetic(*c) for c in coords])
    assert np.allclose(lat, expected[:, 0], atol=1e-6)
    assert np.allclose(lon, expected[:, 1], atol=1e-6)
    assert np.allclose(h, expected[:, 2], atol=1e-3)


def test_find_rotation_and_displacement_numba():
    rng = np.random.default_rng(0)
    points = rng.random((3, 5))

    axis = rng.random(3)
    axis /= np.linalg.norm(axis)
    angle = np.pi / 4
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    R_true = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    d_true = rng.random(3)

    transformed = R_true @ points + d_true[:, None]

    R_est, d_est = findRotationAndDisplacement(points, transformed)

    assert np.allclose(R_est @ points + d_est[:, None], transformed, atol=1e-6)
    assert np.allclose(R_est, R_true, atol=1e-6)
    assert np.allclose(d_est, d_true, atol=1e-6)


def test_find_transponder_simple():
    angle = np.deg2rad(30.0)
    R = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    gps1_to_others = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    gps1_to_transponder = np.array([0.5, 0.5, -1.0])

    translations = np.array([[10.0, -5.0, 2.0], [-4.0, 3.0, 1.0]])
    GPS_Coordinates = np.zeros((2, 4, 3))
    expected = np.zeros((2, 3))
    for i, t in enumerate(translations):
        GPS_Coordinates[i, 0] = t
        for j in range(1, 4):
            GPS_Coordinates[i, j] = t + R @ gps1_to_others[j]
        expected[i] = t + R @ gps1_to_transponder

    result = ng.findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder)
    assert np.allclose(result, expected, atol=1e-6)


def test_compute_jacobian_ray_tracing():
    guess = np.array([0.0, 0.0, 0.0])
    receivers = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    speed = np.full(3, 1500.0)
    dist = np.linalg.norm(receivers - guess, axis=1)
    times = dist / speed

    jac = ng.computeJacobianRayTracing(guess, receivers, times, speed)
    expected = -(receivers - guess) / (dist[:, None] * speed[:, None])
    assert np.allclose(jac, expected, atol=1e-6)
