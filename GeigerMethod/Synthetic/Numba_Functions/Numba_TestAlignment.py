import numpy as np
from Numba_xAline import two_pointer_index
from Numba_Geiger import findTransponder, calculateTimesRayTracing
from Generate_Unaligned_Realistic import generateUnalignedRealistic


offset = np.random.rand() * 10000
print("True offset:", offset)

#Generate data
n = 10000
time_noise = 2 * 10 ** -5
position_noise = 2 * 10 ** -2
CDOG_data, CDOG, GPS_Coordinates, GPS_data, true_transponder_coordinates = generateUnalignedRealistic(n, time_noise, offset)
GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))
gps1_to_others = np.array([[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64)
gps1_to_transponder = np.array([-10, 3, -15], dtype=np.float64)
transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, gps1_to_transponder)
travel_times, esv = calculateTimesRayTracing(CDOG, transponder_coordinates)

best_rmse = np.inf
best = 0
for i in range(10000):
    CDOG_full, GPS_clock, GPS_full = two_pointer_index(i, .9, CDOG_data, GPS_data, travel_times,
                                                       transponder_coordinates, esv)[1:4]
    RMSE = np.sqrt(np.nanmean((CDOG_full - GPS_full) ** 2)) * 1515 * 100

    if RMSE < best_rmse:
        best_rmse = RMSE
        best = i
        print(i, RMSE)
    print(i, RMSE)


print(best, best_rmse, offset)