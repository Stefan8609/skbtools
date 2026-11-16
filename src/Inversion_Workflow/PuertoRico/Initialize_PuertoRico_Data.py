import numpy as np
import scipy.io as sio
from numba import njit

from geometry.Numba_RigidBodyMovementProblem import findRotationAndDisplacement
from Inversion_Workflow.PuertoRico.Crossover_Mask import CROSSOVER_LEG_UTC_BOUNDS
from puertorico_data import gps_data_path


receiver_names = ["STPF", "STSF", "STPA", "STSA"]
clock_files = {
    "before": "DOG1-camp-before-clock-loss.mat",
    "after": "DOG1-camp-after-clock-loss.mat",
}

# Refined before-section inversion estimate in ECEF metres.
CDOG_guess = np.array([2429605.1269901, -5523195.78047007, 2050693.86754481])

# Approximate STPF-relative geometry in metres from SeaTrac_Schematic.png.
gps1_to_others = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 1.26619, 0.0],
        [-3.125978, 0.052197, 0.0],
        [-3.125978, 1.213993, 0.0],
    ]
)

initial_lever_guess = np.array([-0.948182, 0.633095, 0.3])

min_nsv = 5
max_gdop = 3
baseline_tolerance = 0.25

receiver_pairs = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
receiver_triplets = (
    (np.array([0, 1, 2]), np.array([0, 1, 3])),
    (np.array([0, 1, 3]), np.array([0, 2, 4])),
    (np.array([0, 2, 3]), np.array([1, 2, 5])),
    (np.array([1, 2, 3]), np.array([3, 4, 5])),
)


def load_and_process_data(path, GNSS_start, GNSS_end):
    """Load GNSS data within the specified time range."""
    data = sio.loadmat(path)
    days = data["days"].flatten() - 59959
    times = data["times"].flatten()
    datetimes = days * 24 * 3600 + times
    condition_GNSS = (datetimes / 3600 >= GNSS_start) & (datetimes / 3600 <= GNSS_end)

    return (
        datetimes[condition_GNSS],
        data["x"].flatten()[condition_GNSS],
        data["y"].flatten()[condition_GNSS],
        data["z"].flatten()[condition_GNSS],
        data["elev"].flatten()[condition_GNSS],
        data["nsv"].flatten()[condition_GNSS],
        data["gdop"].flatten()[condition_GNSS],
    )


def load_all_receivers(GNSS_start, GNSS_end):
    """Load all receivers and align them to the union of their epochs."""
    paths = [
        gps_data_path(f"GPS_Data/{receiver}-camp_bis.mat")
        for receiver in receiver_names
    ]
    all_data = [load_and_process_data(path, GNSS_start, GNSS_end) for path in paths]

    GPS_data = np.unique(np.concatenate([data[0] for data in all_data]))
    if len(GPS_data) == 0:
        raise ValueError("No GNSS data were found in the requested time range")

    GPS_Coordinates = np.full((len(GPS_data), 4, 3), np.nan)
    elevations = np.full((len(GPS_data), 4), np.nan)
    nsv = np.full((len(GPS_data), 4), np.nan)
    gdop = np.full((len(GPS_data), 4), np.nan)

    for receiver, data in enumerate(all_data):
        datetimes, x, y, z, elev, receiver_nsv, receiver_gdop = data
        indices = np.searchsorted(GPS_data, datetimes)
        GPS_Coordinates[indices, receiver] = np.column_stack((x, y, z))
        elevations[indices, receiver] = elev
        nsv[indices, receiver] = receiver_nsv
        gdop[indices, receiver] = receiver_gdop

    return GPS_data, GPS_Coordinates, elevations, nsv, gdop


def load_cdog_data(clock_segment):
    """Load C-DOG tags and derive timing metadata on the GPS_data timeline."""
    if clock_segment not in clock_files:
        raise ValueError(f"clock_segment must be one of {tuple(clock_files)}")

    data = sio.loadmat(gps_data_path(f"CDOG_Data/{clock_files[clock_segment]}"))
    CDOG_data = data["tags"].astype(float)
    utc_time = data["utc_time"].astype(float)
    tag_seconds = CDOG_data[:, 0] + CDOG_data[:, 1] / 1e9
    if np.any(np.diff(tag_seconds) < 0):
        raise ValueError(f"C-DOG tags are not sorted in the {clock_segment} segment")

    emission_times = np.full(len(CDOG_data), np.nan)
    hhmmss = utc_time[:, 0]
    hours = np.floor(hhmmss / 10000)
    minutes = np.floor((hhmmss - hours * 10000) / 100)
    seconds_of_day = (
        hours * 3600
        + minutes * 60
        + hhmmss
        - hours * 10000
        - minutes * 100
    )

    if clock_segment == "before":
        valid_utc = (
            (utc_time[:, 1] >= 15)
            & (utc_time[:, 1] <= 19)
            & (utc_time[:, 2] == 1)
            & (utc_time[:, 3] == 2023)
        )
        emission_times[valid_utc] = (
            utc_time[valid_utc, 1] - 15
        ) * 86400 + seconds_of_day[valid_utc]
    else:
        # The false 2010 time measures elapsed time after the reset. Its zero is
        # anchored to the final valid 2023 emission and refined later in pairing.
        before = sio.loadmat(gps_data_path(f"CDOG_Data/{clock_files['before']}"))[
            "utc_time"
        ].astype(float)
        before_hhmmss = before[:, 0]
        before_hours = np.floor(before_hhmmss / 10000)
        before_minutes = np.floor(
            (before_hhmmss - before_hours * 10000) / 100
        )
        before_seconds = (
            before_hours * 3600
            + before_minutes * 60
            + before_hhmmss
            - before_hours * 10000
            - before_minutes * 100
        )
        valid_before = (
            (before[:, 1] >= 15)
            & (before[:, 1] <= 19)
            & (before[:, 2] == 1)
            & (before[:, 3] == 2023)
        )
        reset_anchor = np.max(
            (before[valid_before, 1] - 15) * 86400 + before_seconds[valid_before]
        )
        valid_utc = (
            (utc_time[:, 1] >= 1)
            & (utc_time[:, 1] <= 2)
            & (utc_time[:, 2] == 1)
            & (utc_time[:, 3] == 2010)
        )
        emission_times[valid_utc] = (
            reset_anchor
            + (utc_time[valid_utc, 1] - 1) * 86400
            + seconds_of_day[valid_utc]
        )

    if not np.any(valid_utc):
        raise ValueError(f"No usable GPZDA times in the {clock_segment} segment")

    offsets = emission_times[valid_utc] - tag_seconds[valid_utc]
    clock_adjustment = np.median(offsets)
    clock_spread = np.median(np.abs(offsets - clock_adjustment))
    GPS_time_bounds = np.array(
        [
            np.min(emission_times[valid_utc]),
            np.max(emission_times[valid_utc]),
        ]
    ) - clock_adjustment
    CDOG_data[:, 1] /= 1e9

    return CDOG_data, GPS_time_bounds, clock_adjustment, clock_spread


def find_valid_receivers(GPS_Coordinates, nsv, gdop):
    """Prefer four coherent receivers, then the best coherent triplet."""
    quality_mask = (
        np.all(np.isfinite(GPS_Coordinates), axis=2)
        & (nsv >= min_nsv)
        & (gdop <= max_gdop)
    )

    expected_distances = np.array(
        [
            np.linalg.norm(gps1_to_others[first] - gps1_to_others[second])
            for first, second in receiver_pairs
        ]
    )
    observed_distances = np.column_stack(
        [
            np.linalg.norm(
                GPS_Coordinates[:, first] - GPS_Coordinates[:, second], axis=1
            )
            for first, second in receiver_pairs
        ]
    )
    baseline_errors = np.abs(observed_distances - expected_distances)

    used_receivers = np.zeros(quality_mask.shape, dtype=bool)
    all_four = quality_mask.all(axis=1) & np.all(
        baseline_errors <= baseline_tolerance, axis=1
    )
    used_receivers[all_four] = True

    best_error = np.full(len(GPS_Coordinates), np.inf)
    for receivers, pairs in receiver_triplets:
        errors = baseline_errors[:, pairs]
        valid = (
            ~all_four
            & quality_mask[:, receivers].all(axis=1)
            & np.all(errors <= baseline_tolerance, axis=1)
        )
        rms_error = np.sqrt(np.mean(errors**2, axis=1))
        better = valid & (rms_error < best_error)
        indices = np.where(better)[0]

        used_receivers[indices] = False
        used_receivers[indices[:, None], receivers] = True
        best_error[better] = rms_error[better]

    return used_receivers


@njit(cache=True)
def reconstruct_gps_grid(GPS_Coordinates, used_receivers):
    """Fit the four-receiver grid to each accepted three- or four-site epoch."""
    fitted_coordinates = np.empty_like(GPS_Coordinates)

    for i in range(len(GPS_Coordinates)):
        receivers = np.where(used_receivers[i])[0]
        initial_coordinates = gps1_to_others[receivers].T
        final_coordinates = GPS_Coordinates[i, receivers].T
        R_mtrx, d = findRotationAndDisplacement(initial_coordinates, final_coordinates)
        fitted_coordinates[i] = (R_mtrx @ gps1_to_others.T + d[:, None]).T

    return fitted_coordinates


def process_gnss(GNSS_start, GNSS_end):
    """Filter GNSS observations and reconstruct accepted receiver grids."""
    GPS_data, GPS_Coordinates, _, nsv, gdop = load_all_receivers(GNSS_start, GNSS_end)

    print("Filtering Data")
    used_receivers = find_valid_receivers(GPS_Coordinates, nsv, gdop)
    valid_epochs = used_receivers.sum(axis=1) >= 3
    GPS_data = GPS_data[valid_epochs]
    GPS_Coordinates = GPS_Coordinates[valid_epochs]
    used_receivers = used_receivers[valid_epochs]

    if len(GPS_data) == 0:
        raise ValueError("No coherent three- or four-receiver epochs were found")

    GPS_Coordinates = reconstruct_gps_grid(GPS_Coordinates, used_receivers)
    four_receiver_count = np.sum(used_receivers.sum(axis=1) == 4)
    triplet_count = np.sum(used_receivers.sum(axis=1) == 3)

    print(f"Four-receiver epochs: {four_receiver_count}")
    print(f"Triplet epochs: {triplet_count}")
    print("End of Filtering Data")

    return GPS_Coordinates, GPS_data, used_receivers


def initialize_puertorico(
    GNSS_start,
    GNSS_end,
    clock_segment="before",
    save=False,
):
    """Load Puerto Rico GNSS and C-DOG data for one clock segment."""
    print("Initializing Puerto Rico Data")

    GPS_Coordinates, GPS_data, used_receivers = process_gnss(GNSS_start, GNSS_end)
    (
        CDOG_data,
        GPS_time_bounds,
        clock_adjustment,
        clock_spread,
    ) = load_cdog_data(clock_segment)
    GPS_data = GPS_data - clock_adjustment
    crossover_leg_GPS_bounds = (
        CROSSOVER_LEG_UTC_BOUNDS[clock_segment] - clock_adjustment
    )

    print(f"Clock segment: {clock_segment}")
    print(f"UTC-minus-tag adjustment: {clock_adjustment:.3f} s")
    print(f"Median absolute timing spread: {clock_spread:.3f} s")

    if save:
        np.savez(
            gps_data_path(
                f"GPS_Data/Processed_GPS_Receivers_DOG_1_{clock_segment}_clock_loss"
            ),
            GPS_Coordinates=GPS_Coordinates,
            GPS_data=GPS_data,
            GPS_time_bounds=GPS_time_bounds,
            crossover_leg_GPS_bounds=crossover_leg_GPS_bounds,
            CDOG_data=CDOG_data,
            clock_adjustment=clock_adjustment,
            clock_adjustment_spread=clock_spread,
            clock_segment=clock_segment,
            CDOG_guess=CDOG_guess,
            used_receivers=used_receivers,
            gps1_to_others=gps1_to_others,
            initial_lever_guess=initial_lever_guess,
        )

    print("Puerto Rico Data Initialized")
    return GPS_Coordinates, GPS_data, CDOG_data


if __name__ == "__main__":
    clock_segments = [
        ("before", 47, 78),
        ("after", 78, 103),
    ]
    for clock_segment, GNSS_start, GNSS_end in clock_segments:
        initialize_puertorico(
            GNSS_start,
            GNSS_end,
            clock_segment=clock_segment,
            save=True,
        )
