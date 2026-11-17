import numpy as np

from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    calculateTimesRayTracing_Bias,
)
from Inversion_Workflow.Forward_Model.Find_Transponder import findTransponder
from Inversion_Workflow.Inversion.Numba_xAline_Geiger_bias import (
    final_bias_geiger,
    initial_bias_geiger,
)
from Inversion_Workflow.Synthetic.Generate_Unaligned import generateUnaligned

# Smooth synthetic ESV table: 1500 m/s at 5200 m offset, rising with angle.
DZ = np.arange(5100.0, 5400.0, 1.0)
ANGLE = np.arange(0.0, 90.01, 1.0)
ESV = 1500.0 + 0.02 * (DZ[:, None] - 5200.0) + 0.5 * ANGLE[None, :] / 90.0
OTHERS = np.array(
    [[0.0, 0.0, 0.0], [-2.4, -4.2, 0.03], [-12.1, -0.95, 0.004], [-8.7, 5.17, 0.025]]
)
LEVER = np.array([-12.5, 9.6, -13.3])


def _generate(n, time_noise, position_noise, offset, time_bias, seed):
    np.random.seed(seed)
    return generateUnaligned(
        n,
        time_noise,
        position_noise,
        offset,
        0.0,
        time_bias,
        DZ,
        ANGLE,
        ESV,
        gps1_to_others=OTHERS,
        gps1_to_transponder=LEVER,
    )


def test_unaligned_position_noise_has_requested_sd():
    """Each receiver gets the requested sd once (it was applied five times)."""
    sd = 0.05
    _, _, GPS_Coordinates, _, _ = _generate(2000, 0.0, sd, 1500.0, 0.0, seed=1)
    baselines = GPS_Coordinates[:, 1:, :] - GPS_Coordinates[:, :1, :]
    length_error = np.linalg.norm(baselines, axis=2) - np.linalg.norm(
        OTHERS[1:], axis=1
    )
    # A baseline length error combines the noise of two receivers.
    assert abs(length_error.std() / np.sqrt(2) / sd - 1) < 0.1


def test_unaligned_dropout_fraction_is_ten_to_twenty_percent():
    n = 20000
    for seed in range(3):
        CDOG_mat, _, _, _, _ = _generate(n, 0.0, 0.0, 1500.0, 0.0, seed)
        dropped = 1 - len(CDOG_mat) / n
        # Blocks of up to 500 tags can overshoot the target by 500 / n.
        assert 0.10 <= dropped <= 0.20 + 500 / n


def test_unaligned_arrivals_follow_the_inversion_forward_model():
    """Arrival = emission + T + offset + time_bias, T from the inversion's model."""
    offset, time_bias, time_noise = 1234.5, 0.25, 1e-4
    CDOG_mat, CDOG, _, GPS_time, transponder = _generate(
        2000, time_noise, 0.0, offset, time_bias, seed=2
    )
    times, _ = calculateTimesRayTracing_Bias(CDOG, transponder, 0.0, DZ, ANGLE, ESV)
    expected = GPS_time + times + offset + time_bias

    arrival = CDOG_mat[:, 0] + CDOG_mat[:, 1]
    nearest = np.abs(arrival[:, None] - expected[None, :]).argmin(axis=1)
    residual = arrival - expected[nearest]
    assert len(np.unique(nearest)) == len(nearest)
    assert abs(residual.std() / time_noise - 1) < 0.1
    assert abs(residual.mean()) < 5 * time_noise / np.sqrt(len(residual))


def test_clock_offset_recovered_without_the_true_offset():
    """Offset - time_bias recovers offset + generation time_bias, noiseless."""
    offset, time_bias = 3141.5926, 0.2
    CDOG_mat, CDOG, GPS_Coordinates, GPS_time, _ = _generate(
        8000, 0.0, 0.0, offset, time_bias, seed=0
    )
    transponder = findTransponder(GPS_Coordinates, OTHERS, LEVER)
    estimate, found_offset = initial_bias_geiger(
        # Far enough off that the circular-variance pick alone is wrong by seconds.
        CDOG + 200.0 * np.array([0.7, -0.6, 0.4]),
        CDOG_mat,
        GPS_time,
        transponder,
        DZ,
        ANGLE,
        ESV,
        real_data=False,
    )
    estimate = final_bias_geiger(
        estimate[:3],
        CDOG_mat,
        GPS_time,
        transponder,
        found_offset,
        estimate[4],
        estimate[3],
        DZ,
        ANGLE,
        ESV,
        real_data=False,
    )[0]
    assert abs((found_offset - estimate[3]) - (offset + time_bias)) < 1e-4
    assert np.linalg.norm(estimate[:3] - CDOG) < 0.01
