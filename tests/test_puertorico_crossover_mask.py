import numpy as np
import pytest

from Inversion_Workflow.PuertoRico.Crossover_Mask import (
    CROSSOVER_LEG_UTC_BOUNDS,
    crossover_leg_mask,
)


@pytest.mark.parametrize(
    ("clock_segment", "utc_hours"),
    [
        ("before", [68.579, 68.58, 70.82, 70.821, 74.10, 76.93, 76.931]),
        ("after", [81.019, 81.02, 83.62, 83.621, 85.75, 88.63, 88.631]),
    ],
)
def test_crossover_mask_includes_both_legs_and_their_boundaries(
    clock_segment, utc_hours
):
    mask = crossover_leg_mask(
        np.asarray(utc_hours) * 3600.0,
        CROSSOVER_LEG_UTC_BOUNDS[clock_segment],
    )

    assert mask.tolist() == [False, True, True, False, True, True, False]


def test_crossover_mask_uses_bounds_on_the_supplied_gps_timeline():
    clock_adjustment = 166130.6585374
    GPS_data = np.array([22.0, 22.5, 25.0, 28.5, 31.0]) * 3600.0
    crossover_leg_GPS_bounds = (
        CROSSOVER_LEG_UTC_BOUNDS["before"] - clock_adjustment
    )

    mask = crossover_leg_mask(GPS_data, crossover_leg_GPS_bounds)

    assert mask.tolist() == [False, True, False, True, False]


def test_crossover_mask_rejects_non_vector_times():
    with pytest.raises(ValueError, match=r"shape \(N,\)"):
        crossover_leg_mask(
            np.zeros((2, 2)),
            CROSSOVER_LEG_UTC_BOUNDS["before"],
        )


def test_crossover_mask_rejects_invalid_bounds():
    with pytest.raises(ValueError, match=r"shape \(2, 2\)"):
        crossover_leg_mask(np.array([0.0]), np.zeros(2))
