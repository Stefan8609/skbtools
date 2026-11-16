import numpy as np


# UTC hours since January 15, 2023 00:00. The bounds stop at the observed
# turnarounds so that each interval contains one straight pass over the DOG.
CROSSOVER_LEG_UTC_HOURS = {
    "before": ((68.58, 70.82), (74.10, 76.93)),
    "after": ((81.02, 83.62), (85.75, 88.63)),
}
CROSSOVER_LEG_UTC_BOUNDS = {
    segment: np.round(np.asarray(hours) * 3600.0, decimals=9)
    for segment, hours in CROSSOVER_LEG_UTC_HOURS.items()
}


def crossover_leg_mask(GPS_data, crossover_leg_bounds):
    """Mask GPS epochs within either interval; data and bounds are seconds."""
    GPS_data = np.asarray(GPS_data)
    crossover_leg_bounds = np.asarray(crossover_leg_bounds)
    if GPS_data.ndim != 1:
        raise ValueError(f"GPS_data must have shape (N,), received {GPS_data.shape}")
    if crossover_leg_bounds.shape != (2, 2):
        raise ValueError(
            "crossover_leg_bounds must have shape (2, 2), received "
            f"{crossover_leg_bounds.shape}"
        )

    return np.logical_or.reduce(
        [
            (GPS_data >= start_time) & (GPS_data <= end_time)
            for start_time, end_time in crossover_leg_bounds
        ]
    )
