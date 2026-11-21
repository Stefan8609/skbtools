import numpy as np
from geometry.rigid_body import findRotationAndDisplacement


def test_find_rotation_and_displacement():
    rng = np.random.default_rng(0)
    points = rng.random((3, 5))

    axis = rng.random(3)
    axis /= np.linalg.norm(axis)
    angle = np.pi / 3
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


def test_fit_rigid_grid_recovers_grid_in_the_frame_of_the_reference():
    from scipy.spatial.transform import Rotation
    from geometry.GPS_Grid import fit_rigid_grid

    true_grid = np.array(
        [[0.0, 0.0, 0.0], [-2.4, -4.2, 0.01], [-12.1, -1.0, 0.0], [-8.7, 5.1, 0.0]]
    )
    rng = np.random.default_rng(1)
    rotations = Rotation.random(200, random_state=2).as_matrix()
    epochs = np.einsum("nij,kj->nki", rotations, true_grid)
    epochs += rng.normal(0.0, 100.0, (200, 1, 3)) + rng.normal(0, 0.01, epochs.shape)
    # Start from a grid 5 cm off and rotated by 0.2 degrees about z
    small = Rotation.from_euler("z", 0.2, degrees=True).as_matrix()
    frame_grid = true_grid @ small.T + 0.05

    grid, misfit = fit_rigid_grid(epochs, true_grid)
    assert np.allclose(grid, true_grid, atol=3e-3)
    assert misfit.shape == (200, 4)
    assert np.allclose(fit_rigid_grid(epochs, frame_grid)[0][0], 0.0)
