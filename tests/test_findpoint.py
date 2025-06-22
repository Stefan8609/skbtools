import numpy as np
from geometry.find_point_by_plane import initializeFunction, findXyzt


def test_initialize_and_find_xyzt_simple():
    xs = np.array([1, -1, -1, 1], dtype=float)
    ys = np.array([1, 1, -1, -1], dtype=float)
    zs = np.zeros(4)
    xyzt = np.array([0.0, 0.0, 1.0])

    theta, phi, length, orientation = initializeFunction(xs, ys, zs, 0, xyzt)
    assert np.isclose(length, 1.0, atol=1e-6)
    assert np.isclose(theta, 0.0, atol=1e-6)
    assert np.isclose(phi, 0.0, atol=1e-6)

    vector, bary, norm = findXyzt(xs, ys, zs, 0, length, theta, phi, orientation)
    reconstructed = bary + vector
    assert np.allclose(reconstructed, xyzt, atol=1e-6)
