import numpy as np
import matplotlib.pyplot as plt
from Numba_Geiger import calculateTimesRayTracing, findTransponder
from Numba_xAline import two_pointer_index, find_int_offset
from Generate_Unaligned_Realistic import generateUnalignedRealistic

true_offset = np.random.rand() * 9000 + 1000
print(true_offset)
position_noise = 2 * 10 ** -2
time_noise = 2 * 10 ** -5

CDOG_data, CDOG, GPS_Coordinates, GPS_data, true_transponder_coordinates = generateUnalignedRealistic(
    20000, time_noise, true_offset, ray=False
)
GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))

gps1_to_others = np.array([[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64)
gps1_to_transponder = np.array([-10, 3, -15], dtype=np.float64)

initial_guess = CDOG + np.array([700, -1300, -150], dtype=np.float64)
initial_lever = np.array([-2.0, 15.0, -2.0], dtype=np.float64)

transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, initial_lever)
GPS_travel_times, esv = calculateTimesRayTracing(initial_guess, true_transponder_coordinates)
offset = find_int_offset(CDOG_data, GPS_data, GPS_travel_times, transponder_coordinates, esv)
print(offset)
offset = int(true_offset - 50)
CDOG_clock, CDOG_full, GPS_clock, GPS_full, transponder_coordinates_full, esv_full = two_pointer_index(offset, 0.5,
                                                                                                       CDOG_data,
                                                                                                       GPS_data,
                                                                                                       GPS_travel_times,
                                                                                                       transponder_coordinates,
                                                                                                       esv, exact=True)
plt.scatter(GPS_clock, GPS_full, s=1, label="GPS")
plt.scatter(CDOG_clock, CDOG_full, s=1, label="CDOG")
plt.title("Attempt to simulate similar error to real data")
plt.xlabel("Time (s)")
plt.ylabel("Travel Time (s)")
plt.legend()
plt.show()


