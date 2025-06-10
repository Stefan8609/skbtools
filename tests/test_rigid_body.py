import numpy as np
from RigidBodyMovementProblem import findRotationAndDisplacement


def test_find_rotation_and_displacement():
    rng = np.random.default_rng(0)
    points = rng.random((3, 5))

    axis = rng.random(3)
    axis /= np.linalg.norm(axis)
    angle = np.pi / 3
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R_true = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    d_true = rng.random(3)

    transformed = R_true @ points + d_true[:, None]

    R_est, d_est = findRotationAndDisplacement(points, transformed)

    assert np.allclose(R_est @ points + d_est[:, None], transformed, atol=1e-6)
    assert np.allclose(R_est, R_true, atol=1e-6)
    assert np.allclose(d_est, d_true, atol=1e-6)
