import numpy as np
from acoustics.ray_tracing import ray_tracing, ray_trace_locate


def test_ray_tracing_constant_speed():
    depth = np.linspace(0, 1000, 101)
    cz = np.full_like(depth, 1500.0)
    x, dz, t = ray_tracing(45.0, 0.0, 1000.0, depth, cz)
    expected_x = 1000.0 * np.tan(np.deg2rad(45.0))
    expected_t = np.sqrt(expected_x**2 + 1000.0**2) / 1500.0
    assert np.isclose(x, expected_x, atol=15.0)
    assert np.isclose(t, expected_t, atol=0.02)

    alpha = ray_trace_locate(0.0, 1000.0, x, depth, cz)
    assert np.isclose(alpha, 45.0, atol=1e-2)


def test_ray_tracing_total_internal_reflection():
    depth = np.array([0.0, 50.0, 100.0])
    cz = np.array([1500.0, 2000.0, 2000.0])
    x, dz, t = ray_tracing(0.0, 0.0, 100.0, depth, cz)
    assert np.isnan(x)
    assert np.isnan(t)
    assert dz == 100.0
