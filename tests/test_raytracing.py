import numpy as np
import pytest
from scipy.integrate import quad
from acoustics.ray_tracing import ray_tracing, ray_trace_locate


def test_ray_tracing_constant_speed():
    depth = np.linspace(0, 1000, 101)
    cz = np.full_like(depth, 1500.0)
    x, dz, t = ray_tracing(45.0, 0.0, 1000.0, depth, cz)
    expected_x = 1000.0 * np.tan(np.deg2rad(45.0))
    expected_t = np.sqrt(expected_x**2 + 1000.0**2) / 1500.0
    assert np.isclose(x, expected_x, rtol=1e-12, atol=1e-12)
    assert np.isclose(t, expected_t, rtol=1e-12, atol=1e-12)

    alpha = ray_trace_locate(0.0, 1000.0, x, depth, cz)
    assert np.isclose(alpha, 45.0, atol=1e-2)


def test_ray_tracing_total_internal_reflection():
    depth = np.array([0.0, 50.0, 100.0])
    cz = np.array([1500.0, 2000.0, 2000.0])
    x, dz, t = ray_tracing(0.0, 0.0, 100.0, depth, cz)
    assert np.isnan(x)
    assert np.isnan(t)
    assert dz == 100.0


@pytest.mark.parametrize("z_a, z_b", [(0.0, 100.0), (2.1, 98.7), (2.1, 3.5)])
@pytest.mark.parametrize("dtype", [float, int])
def test_ray_tracing_integrates_exact_endpoints(z_a, z_b, dtype):
    depth = np.array([0, 50, 100], dtype=dtype)
    cz = np.full_like(depth, 1500)
    x, dz, travel_time = ray_tracing(90.0, z_a, z_b, depth, cz)
    assert np.isclose(x, 0.0, atol=1e-12)
    assert dz == z_b - z_a
    assert np.isclose(travel_time, (z_b - z_a) / 1500.0, rtol=1e-12)


def test_ray_trace_locate_recovers_after_a_turning_ray():
    depth = np.array([0.0, 500.0, 1000.0])
    cz = np.array([1500.0, 1650.0, 1800.0])
    target_x, _, _ = ray_tracing(40.0, 0.0, 1000.0, depth, cz)
    alpha = ray_trace_locate(0.0, 1000.0, target_x, depth, cz)
    x, _, _ = ray_tracing(alpha, 0.0, 1000.0, depth, cz)
    assert abs(x - target_x) < 0.0001
    assert np.isclose(alpha, 40.0, atol=1e-5)


def test_ray_trace_locate_vertical_path():
    depth = np.array([0.0, 100.0])
    cz = np.array([1500.0, 1500.0])
    assert ray_trace_locate(2.1, 98.7, 0.0, depth, cz) == 90.0


def test_ray_tracing_linear_profile_against_slowness_integrals():
    depth = np.linspace(0.0, 1000.0, 101)
    cz = 1500.0 + 0.02 * depth
    ray_parameter = np.cos(np.deg2rad(45.0)) / cz[0]
    expected_x = quad(
        lambda z: ray_parameter
        * (1500.0 + 0.02 * z)
        / np.sqrt(1.0 - (ray_parameter * (1500.0 + 0.02 * z)) ** 2),
        0.0,
        1000.0,
    )[0]
    expected_time = quad(
        lambda z: 1.0
        / (
            (1500.0 + 0.02 * z)
            * np.sqrt(1.0 - (ray_parameter * (1500.0 + 0.02 * z)) ** 2)
        ),
        0.0,
        1000.0,
    )[0]
    x, _, travel_time = ray_tracing(45.0, 0.0, 1000.0, depth, cz)
    assert abs(x - expected_x) < 0.0001
    assert abs(travel_time - expected_time) < 1e-7


def test_ray_trace_locate_returns_nan_for_unreachable_direct_range():
    depth = np.array([0.0, 500.0, 1000.0])
    cz = np.array([1500.0, 1650.0, 1800.0])
    assert np.isnan(ray_trace_locate(0.0, 1000.0, 1e8, depth, cz))


@pytest.mark.parametrize("z_a, z_b", [(-1.0, 90.0), (1.0, 101.0), (50.0, 50.0)])
def test_ray_tracing_rejects_invalid_endpoints(z_a, z_b):
    depth = np.array([0.0, 100.0])
    cz = np.array([1500.0, 1500.0])
    with pytest.raises(ValueError):
        ray_tracing(45.0, z_a, z_b, depth, cz)


def test_ray_tracing_rejects_mismatched_profile():
    with pytest.raises(ValueError, match="matching 1D arrays"):
        ray_tracing(45.0, 0.0, 100.0, np.array([0.0, 100.0]), np.array([1500.0]))


@pytest.mark.parametrize("target_x", [-1.0, np.nan])
def test_ray_trace_locate_rejects_invalid_range(target_x):
    depth = np.array([0.0, 100.0])
    cz = np.array([1500.0, 1500.0])
    with pytest.raises(ValueError, match="Horizontal range"):
        ray_trace_locate(0.0, 100.0, target_x, depth, cz)
