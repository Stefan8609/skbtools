import numpy as np

from Inversion_Workflow.Bermuda.Integer_Ambiguity import (
    mechanism_fit,
    radial_velocity,
    rmse,
    shifted_pairing_residuals,
)


def synthetic_pass(true_offset=3175.017, speed=2.0, closest_m=3000.0, n=1200):
    """Vessel passing a DOG at constant speed; 1 Hz emissions, known offset."""
    emission = 1000.0 + np.arange(n, dtype=float)
    along = speed * (emission - emission.mean())
    coordinates = np.column_stack([along, np.full(n, closest_m), np.zeros(n)])
    esv = np.full(n, 1500.0)
    travel_times = np.linalg.norm(coordinates, axis=1) / esv
    arrival = emission + travel_times + true_offset
    CDOG_data = np.column_stack([np.floor(arrival), arrival - np.floor(arrival)])
    return emission, coordinates, travel_times, esv, CDOG_data, true_offset


def test_rmse_rises_for_wrong_integer_and_tracks_radial_velocity():
    emission, coords, tt, esv, CDOG_data, offset = synthetic_pass()
    velocity = radial_velocity(coords, emission, np.zeros(3))
    rms = {}
    for k in (-2, -1, 0, 1, 2):
        pairing = shifted_pairing_residuals(
            offset + k, 0.4, CDOG_data, emission, tt, coords, esv
        )
        rms[k] = rmse(pairing["residuals"])
        if abs(k) == 1:
            fit = mechanism_fit(pairing, velocity)
            assert np.sign(fit["slope_s"]) == np.sign(k)
            assert abs(abs(fit["slope_s"]) - 1.0) < 0.05
    assert rms[0] < 1e-9
    assert all(rms[k] > 1e-4 for k in (-2, -1, 1, 2))
    assert rms[2] > rms[1]


def test_radial_velocity_nan_across_gaps():
    t = np.array([0.0, 1.0, 2.0, 30.0, 31.0, 32.0])
    coords = np.column_stack([2.0 * t, np.zeros(6), np.zeros(6)])
    v = radial_velocity(coords + np.array([-100.0, 0, 0]), t, np.zeros(3))
    assert np.isnan(v[[0, 2, 3, 5]]).all()
    assert np.allclose(v[[1, 4]], -2.0)
