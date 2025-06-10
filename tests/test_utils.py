import numpy as np
from rodriguesRotationMatrix import rotationMatrix
from projectToPlane import projectToPlane
from fitPlane import fitPlane


def test_rotation_matrix_z_axis():
    vec = np.array([1.0, 0.0, 0.0])
    R = rotationMatrix(np.pi/2, np.array([0.0, 0.0, 1.0]))
    rotated = R @ vec
    assert np.allclose(rotated, np.array([0.0, 1.0, 0.0]), atol=1e-6)


def test_project_to_plane_xy():
    vect = np.array([1.0, 2.0, 3.0])
    norm = np.array([0.0, 0.0, 1.0])
    projected = projectToPlane(vect, norm)
    assert np.allclose(projected, np.array([1.0, 2.0, 0.0]), atol=1e-6)


def test_fit_plane_known_plane():
    xs = np.array([0.0, 1.0, -1.0, 2.0, -2.0])
    ys = np.array([0.0, -1.0, 1.0, 0.5, -0.5])
    zs = 2 * xs + 3 * ys + 5
    norm = fitPlane(xs, ys, zs)
    expected = np.array([2.0, 3.0, -1.0])
    norm /= np.linalg.norm(norm)
    expected /= np.linalg.norm(expected)
    assert np.allclose(np.abs(np.dot(norm, expected)), 1.0, atol=1e-6)
