import numpy as np
import matplotlib.pyplot as plt

from Modular_Synthetic import modular_synthetic
from ECEF_Geodetic import ECEF_Geodetic

from pymap3d import geodetic2enu

def error_ellipse(num_points, time_noise, position_noise):

    estimate_array = np.zeros((num_points, 3))
    for i in range(num_points):
        (inversion_result, CDOG_data, CDOG_full, GPS_data, GPS_full, CDOG_clock, GPS_clock, CDOG, transponder_coordinates,
         GPS_Coordinates, offset) = modular_synthetic(time_noise, position_noise, 0, 0,
                                                      esv1 = "global_table_esv", esv2="global_table_esv_perturbed",
                                                      generate_type = 1, inversion_type = 1, plot=False)
        estimate_array[i] = inversion_result[:3]
        print(f"{i+1}/{num_points} iterations complete")

    # estimate_array = estimate_array - CDOG
    # CDOG = CDOG - CDOG

    #Convert to geodetic
    CDOG_lat, CDOG_lon, CDOG_height = ECEF_Geodetic(np.array([CDOG]))
    estimate_lat, estimate_lon, estimate_height = ECEF_Geodetic(estimate_array)

    # Convert to ENU coordinates
    estimate_converted = np.zeros((num_points, 3))
    for i in range(num_points):
        enu = geodetic2enu(
            estimate_lat[i], estimate_lon[i], estimate_height[i],
            CDOG_lat, CDOG_lon, CDOG_height
        )
        estimate_converted[i] = np.squeeze(enu)

    plt.scatter(estimate_converted[:, 0]*100, estimate_converted[:, 1]*100, s=1, color="blue", label=r'CDOG Estimates')
    plt.scatter(0, 0, s=100, color="red", marker="o", label="CDOG Actual")
    plt.xlabel("Easting (cm)")
    plt.ylabel("Northing (cm)")
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.show()

    plt.scatter(estimate_converted[:, 0]*100, estimate_converted[:, 2]*100, s=1, color="blue", label=r'CDOG Estimates')
    plt.scatter(0, 0, s=100, color="red", marker="o", label="CDOG Actual")
    plt.xlabel("Easting (cm)")
    plt.ylabel("Elevation (cm)")
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.show()

    plt.scatter(estimate_converted[:, 1]*100, estimate_converted[:, 2]*100, s=1, color="blue", label=r'CDOG Estimates')
    plt.scatter(0, 0,  s=100, color="red", marker="o", label="CDOG Actual")
    plt.xlabel("Northing (cm)")
    plt.ylabel("Elevation (cm)")
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    num_points = 1000
    time_noise = 2 * 10**-5
    position_noise = 2 * 10**-2
    error_ellipse(num_points, time_noise, position_noise)




