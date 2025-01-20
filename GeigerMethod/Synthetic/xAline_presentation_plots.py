import numpy as np
import matplotlib.pyplot as plt
from advancedGeigerMethod import calculateTimesRayTracing, findTransponder
from Generate_Unaligned_Realistic import generateUnalignedRealistic
from scipy import signal
from xAline import index_data
from GeigerMethod.Synthetic.Numba_Functions.Numba_xAline import find_subint_offset, two_pointer_index

true_offset = 1456.241
offset = 0
position_noise = 2*10**-2

CDOG_data, CDOG, GPS_Coordinates, GPS_data, true_transponder_coordinates = generateUnalignedRealistic(20000, 2*10**-5, true_offset)
GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))

gps1_to_others = np.array([[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64)
gps1_to_transponder = np.array([-10, 3, -15], dtype=np.float64)
transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder)

plt.rcParams.update({'font.size': 16})
# scatter = plt.scatter(transponder_coordinates[:, 0], transponder_coordinates[:, 1], s=3, marker="o",
#                       c=np.arange(len(transponder_coordinates)), cmap='viridis', label="Transponder")
# plt.colorbar(scatter, label='Index')
# plt.scatter(CDOG[0], CDOG[1], s=50, marker="x", color="k", label="C-DOG")
# plt.xlabel('Easting (m)')
# plt.ylabel('Northing (m)')
# plt.legend(loc="upper right")
# plt.title("Synthetic Trajectory")
# plt.axis("equal")
#plt.show()

travel_times, esv = calculateTimesRayTracing(CDOG, transponder_coordinates)
CDOG_unwrap = np.unwrap(CDOG_data[:, 1] * 2 * np.pi) / (2 * np.pi)
#
# plt.scatter(np.arange(len(CDOG_unwrap)), CDOG_unwrap, s=10, marker='x', label="CDOG Unwrapped Times")
# plt.xlabel("Index")
# plt.ylabel("Travel Times")
# plt.title("C-DOG Unwrapped Travel Series by Index")
#plt.show()

# plt.scatter(np.arange(len(travel_times)), travel_times, s=1, label="Model Travel Times")
# plt.xlabel("Index")
# plt.ylabel("Travel Times")
# plt.title("Model Travel Series by Index")
#plt.show()

# plt.scatter(np.arange(len(CDOG_unwrap)), CDOG_unwrap, s=10, marker='x', label="CDOG Unwrapped Times")
# plt.scatter(np.arange(len(travel_times)), travel_times, s=1, label="Model Travel Times")
# plt.legend()
# plt.xlabel("Index")
# plt.ylabel("Travel Times")
# plt.title("Comparison of Unwrapped C-DOG Travel Series with Inversion Travel Series by Index")
#plt.show()

CDOG_travel_times = CDOG_unwrap + travel_times[0] - CDOG_unwrap[0]
CDOG_times = np.round(CDOG_data[:, 0] + CDOG_data[:, 1] - 0)
GPS_times = np.round(GPS_data + travel_times)

plt.figure(figsize=(8, 5))
plt.scatter(CDOG_times/3600, CDOG_travel_times, s=10, marker='x', label="Observed")
plt.scatter(GPS_times/3600, travel_times, s=1, label="Model")
plt.legend()
plt.xlabel("Absolute Time (hours)")
plt.ylabel("Travel Times (s)")
# plt.title("Comparison of Time Series Indexed By Closest Integer Absolute Time")
plt.show()

# Get unique times and corresponding acoustic data for DOG
unique_CDOG_times, CDOG_indices = np.unique(CDOG_times, return_index=True)
unique_CDOG_travel_times = CDOG_travel_times[CDOG_indices]

# Get unique times and corresponding travel times for GPS
unique_GPS_times, indices_GPS = np.unique(GPS_times, return_index=True)
unique_travel_times = travel_times[indices_GPS]
unique_transponder_coordinates = transponder_coordinates[indices_GPS]
unique_esv = esv[indices_GPS]

# Create a full size array that has indices for all times covered by CDOG or GPS times
min_time = min(unique_GPS_times.min(), unique_CDOG_times.min())
max_time = max(unique_GPS_times.max(), unique_CDOG_times.max())
full_times = np.arange(min_time, max_time + 1)
CDOG_full = np.full(full_times.shape, np.nan)
GPS_full = np.full(full_times.shape, np.nan)

# Get CDOG travel times into the full time array at the correct indices
CDOG_match = np.searchsorted(full_times, unique_CDOG_times)
CDOG_mask = (CDOG_match < len(full_times)) & (full_times[CDOG_match] == unique_CDOG_times)
CDOG_full[CDOG_match[CDOG_mask]] = unique_CDOG_travel_times[CDOG_mask]

# Get GPS data into the respective full time arrays at the correct indices
GPS_match = np.searchsorted(full_times, unique_GPS_times)
GPS_mask = (GPS_match < len(full_times)) & (full_times[GPS_match] == unique_GPS_times)
GPS_full[GPS_match[GPS_mask]] = unique_travel_times[GPS_mask]

# Remove nan values from all arrays
nan_mask = ~np.isnan(CDOG_full) & ~np.isnan(GPS_full)
CDOG_full = CDOG_full[nan_mask]
GPS_full = GPS_full[nan_mask]
full_times = full_times[nan_mask]

# plt.scatter(full_times, CDOG_full, s=10, marker='x', label="CDOG Derived Travel Times")
# plt.scatter(full_times, GPS_full, s=1, label="Model Travel Times")
# plt.legend()
# plt.xlabel("Absolute Time")
# plt.ylabel("Travel Times")
# plt.title("Comparison of Time Series Indexed By Closest Integer Absolute Time \n (Retaining only absolute times containing data from both CDOG and GPS)")
#plt.show()

abs_diff = np.abs(CDOG_full - GPS_full)
indices = np.where(abs_diff >= 0.9)
CDOG_full[indices] += np.round(GPS_full[indices] - CDOG_full[indices])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))

# ax1.scatter(full_times, CDOG_full, s=10, marker="x", label="CDOG Derived Travel Times")
# ax1.scatter(full_times, GPS_full, s=1, marker="o", label="Model Travel Times")
# ax1.legend(loc="upper right")
# ax1.set_xlabel("Absolute Time")
# ax1.set_ylabel("Travel Times")
# ax1.set_title(f"Comparison of Time Series with offset: {offset} \n true offset: {true_offset}")

diff_data = CDOG_full - GPS_full
RMSE = np.sqrt(np.nanmean(diff_data ** 2))
# ax2.scatter(full_times, diff_data, s=1)
# ax2.set_xlabel("Absolute Time")
# ax2.set_ylabel("Difference between calculated and unwrapped times")
# ax2.set_title(f"Residual Plot with RMSE: {np.round(RMSE*1000000,4)} µs")

#plt.show()


for i in range(2):
    GPS_fp = np.modf(GPS_full)[0]
    CDOG_fp = np.modf(CDOG_full)[0]
    correlation = signal.correlate(CDOG_fp - np.mean(CDOG_fp), GPS_fp - np.mean(GPS_fp), mode="full", method="fft")
    lags = signal.correlation_lags(len(CDOG_fp), len(GPS_fp), mode="full")
    lag = lags[np.argmax(abs(correlation))]
    offset += lag

    full_times, CDOG_full, GPS_full, transponder_full, esv_full = index_data(offset, CDOG_data, GPS_data,
                                                                                 travel_times, transponder_coordinates, esv)

    # plt.scatter(full_times, CDOG_full, s=10, marker='x', label="CDOG Derived Travel Times")
    # plt.scatter(full_times, GPS_full, s=1, label="Model Travel Times")
    # plt.legend()
    # plt.xlabel("Absolute Time")
    # plt.ylabel("Travel Times")
    # plt.title(f"Comparison of Time Series with offset: {offset} \n true offset: {true_offset}")
    #plt.show()

    abs_diff = np.abs(CDOG_full - GPS_full)
    indices = np.where(abs_diff >= 0.9)
    CDOG_full[indices] += np.round(GPS_full[indices] - CDOG_full[indices])

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))
    #
    # ax1.scatter(full_times, CDOG_full, s=10, marker="x", label="CDOG Derived Travel Times")
    # ax1.scatter(full_times, GPS_full, s=1, marker="o", label="Model Travel Times")
    # ax1.legend(loc="upper right")
    # ax1.set_xlabel("Absolute Time")
    # ax1.set_ylabel("Travel Times")
    # ax1.set_title(f"Comparison of Time Series with offset: {offset} \n true offset: {true_offset}")
    #
    # diff_data = CDOG_full - GPS_full
    # RMSE = np.sqrt(np.nanmean(diff_data ** 2))
    # ax2.scatter(full_times, diff_data, s=1)
    # ax2.set_xlabel("Absolute Time")
    # ax2.set_ylabel("Difference between calculated and unwrapped times")
    # ax2.set_title(f"Residual Plot with RMSE: {np.round(RMSE*1000000,4)} µs")

    #plt.show()

for i in range(2):
    offset = find_subint_offset(offset, CDOG_data, GPS_data, travel_times, transponder_coordinates, esv)
    full_times, CDOG_full, GPS_clock, GPS_full, transponder_full, esv_full = two_pointer_index(offset, 0.6, CDOG_data, GPS_data,
                                                                                 travel_times, transponder_coordinates, esv)
    abs_diff = np.abs(CDOG_full - GPS_full)
    indices = np.where(abs_diff >= 0.9)
    CDOG_full[indices] += np.round(GPS_full[indices] - CDOG_full[indices])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    ax1.scatter(full_times/3600, CDOG_full, s=10, marker="x", label="Observed")
    ax1.scatter(full_times/3600, GPS_full, s=1, marker="o", label="Model")
    ax1.legend(loc="upper right")
    # ax1.set_xlabel("Absolute Time")
    ax1.set_ylabel("Travel Times (s)")
    # ax1.set_title(f"Comparison of Time Series with offset: {offset} \n true offset: {true_offset}")

    diff_data = (CDOG_full - GPS_full)*1000
    std = np.std(diff_data)
    RMSE = np.sqrt(np.nanmean(diff_data ** 2))
    ax2.scatter(full_times/3600, diff_data, s=1)
    ax2.set_ylim(-3*std, 3*std)
    ax2.set_xlabel("Absolute Time (hours)")
    ax2.set_ylabel("Model Misfit (ms)")
    # ax2.set_title(f"Residual Plot with RMSE: {np.round(RMSE*1000000,4)} µs")

    plt.show()