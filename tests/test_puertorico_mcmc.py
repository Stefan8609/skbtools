import numpy as np

from Inversion_Workflow.PuertoRico.Crossover_Mask import crossover_leg_mask


def test_trajectory_mode_selects_full_or_cross_epochs():
    GPS_hours = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    eligible = np.array([True, True, False, True, True])
    crossover_leg_GPS_bounds = np.array([[1.5, 2.5], [3.5, 4.5]]) * 3600.0

    cross = eligible & crossover_leg_mask(
        GPS_hours * 3600.0,
        crossover_leg_GPS_bounds,
    )

    assert cross.tolist() == [False, True, False, True, False]

