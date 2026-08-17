import numpy as np
import scipy.io as sio
from numba import njit

from geometry.Numba_RigidBodyMovementProblem import findRotationAndDisplacement
from puertorico_data import gps_data_path


receiver_names = ["STPF", "STSF", "STPA", "STSA"]

# Approximate STPF-relative geometry in metres from SeaTrac_Schematic.png.
gps1_to_others = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 1.26619, 0.0],
        [-3.125978, 0.052197, 0.0],
        [-3.125978, 1.213993, 0.0],
    ]
)

# The schematic does not constrain the vertical component.
initial_lever_guess = np.array([-0.948182, 0.633095, 0.0])

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
        data["nsv"].flatten()[condition_GNSS],
        data["gdop"].flatten()[condition_GNSS],
    )


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


def initialize_puertorico(GNSS_start, GNSS_end, save=False):
    """Load and process the Puerto Rico vessel GNSS data."""
    print("Initializing Puerto Rico Data")

    paths = [
        gps_data_path(f"GPS_Data/{receiver}-camp_bis.mat")
        for receiver in receiver_names
    ]
    all_data = [load_and_process_data(path, GNSS_start, GNSS_end) for path in paths]

    GPS_data = np.unique(np.concatenate([data[0] for data in all_data]))
    if len(GPS_data) == 0:
        raise ValueError("No GNSS data were found in the requested time range")

    GPS_Coordinates = np.full((len(GPS_data), 4, 3), np.nan)
    nsv = np.full((len(GPS_data), 4), np.nan)
    gdop = np.full((len(GPS_data), 4), np.nan)

    for receiver, data in enumerate(all_data):
        datetimes, x, y, z, receiver_nsv, receiver_gdop = data
        indices = np.searchsorted(GPS_data, datetimes)
        GPS_Coordinates[indices, receiver] = np.column_stack((x, y, z))
        nsv[indices, receiver] = receiver_nsv
        gdop[indices, receiver] = receiver_gdop

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

    if save:
        np.savez(
            gps_data_path("GPS_Data/Processed_GPS_Receivers_DOG_1"),
            GPS_Coordinates=GPS_Coordinates,
            GPS_data=GPS_data,
            used_receivers=used_receivers,
            gps1_to_others=gps1_to_others,
            initial_lever_guess=initial_lever_guess,
        )

    print("Puerto Rico Data Initialized")
    return GPS_Coordinates, GPS_data, used_receivers


if __name__ == "__main__":
    GNSS_start = 0
    GNSS_end = 120
    GPS_Coordinates, GPS_data, used_receivers = initialize_puertorico(
        GNSS_start, GNSS_end, save=True
    )
