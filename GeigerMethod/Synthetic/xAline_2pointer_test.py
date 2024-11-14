import numpy as np
import matplotlib.pyplot as plt
from xAline import *
from advancedGeigerMethod import *
from Generate_Unaligned_Realistic import generateUnalignedRealistic

n=10000
true_offset = np.random.rand() * 10000
position_noise = 2*10**-2
time_noise = 2*10**-5

CDOG_data, CDOG, GPS_Coordinates, GPS_data, true_transponder_coordinates = generateUnalignedRealistic(n, time_noise, true_offset)
GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))

gps1_to_others = np.array([[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64)
gps1_to_transponder = np.array([-10, 3, -15], dtype=np.float64)
transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder)
travel_times, esv = calculateTimesRayTracing(CDOG, transponder_coordinates)


# Find the derived offset
offset = find_int_offset(CDOG_data, GPS_data, travel_times, transponder_coordinates, esv)
offset = find_subint_offset(offset, CDOG_data, GPS_data, travel_times, transponder_coordinates, esv)

print("True offset:", true_offset, "\nDerived offset:", offset)

[CDOG_clock, CDOG_full, GPS_clock, GPS_full, transponder_coordinates_full, esv_full] = (
    two_pointer_index(offset, 0.9, CDOG_data, GPS_data, travel_times, transponder_coordinates, esv)
)

abs_diff = np.abs(CDOG_full - GPS_full)
indices = np.where(abs_diff >= 0.9)
CDOG_full[indices] += np.round(GPS_full[indices] - CDOG_full[indices])

RMSE = np.sqrt(np.nanmean((CDOG_full - GPS_full) ** 2)) * 1515 * 100

print("RMSE:", RMSE, "cm")

full_times_true, CDOG_full_true, GPS_full_true = index_data(true_offset, CDOG_data, GPS_data, travel_times,
                                                            transponder_coordinates, esv)[:3]
abs_diff = np.abs(CDOG_full_true - GPS_full_true)
indices = np.where(abs_diff >= 0.6)
CDOG_full_true[indices] += np.round(GPS_full_true[indices] - CDOG_full_true[indices])
RMSE_true = np.sqrt(np.nanmean((CDOG_full_true - GPS_full_true) ** 2)) * 1515 * 100

fig, axes = plt.subplots(2, 2, figsize=(15, 8))

axes[0, 0].scatter(full_times_true, CDOG_full_true, s=10, marker="x",
                   label="Unwrapped/Adjusted Synthetic Dog Travel Time")
axes[0, 0].scatter(full_times_true, GPS_full_true, s=1, marker="o", label="Calculated GPS Travel Times")
axes[0, 0].legend(loc="upper right")
axes[0, 0].set_xlabel("Arrivals in Absolute Time (s)")
axes[0, 0].set_ylabel("Travel Times (s)")
axes[0, 0].set_title(f"Synthetic travel times with offset: {np.round(true_offset, 5)} and RMSE: {np.round(RMSE, 3)}")

diff_data_true = CDOG_full_true - GPS_full_true
axes[1, 0].scatter(full_times_true, diff_data_true, s=1)
axes[1, 0].set_xlabel("Absolute Time (s)")
axes[1, 0].set_ylabel("Difference between calculated and unwrapped times (s)")
axes[1, 0].set_title("Residual Plot")

axes[0, 1].scatter(CDOG_clock, CDOG_full, s=10, marker="x",
                   label="Unwrapped/Adjusted Synthetic Dog Travel Time")
axes[0, 1].scatter(GPS_clock, GPS_full, s=1, marker="o", label="Calculated GPS Travel Times")
axes[0, 1].legend(loc="upper right")
axes[0, 1].set_xlabel("Arrivals in Absolute Time (s)")
axes[0, 1].set_ylabel("Travel Times (s)")
axes[0, 1].set_title(f"Synthetic travel times with offset: {offset} and RMSE: {np.round(RMSE, 3)}")

diff_data = CDOG_full - GPS_full
axes[1, 1].scatter(CDOG_clock, diff_data, s=1)
axes[1, 1].set_xlabel("Absolute Time (s)")
axes[1, 1].set_ylabel("Difference between calculated and unwrapped times (s)")
axes[1, 1].set_title("Residual Plot")

print('Mean of residuals: ', np.mean(diff_data)*1000, "ms")
print("Diff between found and true offset: ", (offset - true_offset) * 1000, "ms")

plt.show()