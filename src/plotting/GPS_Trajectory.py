import numpy as np
import matplotlib.pyplot as plt

from data import gps_data_path
from GeigerMethod.Synthetic.Numba_Functions.ECEF_Geodetic import ECEF_Geodetic

"""Looks like GPS1 is GPS-PF"""


def GPS_Trajectory(GPS_Coordinates):
    # Convert to numpy array for easier manipulation
    GPS_Coordinates = np.array(GPS_Coordinates)

    # Extract coordinates for each GPS unit
    lat = GPS_Coordinates[:, :, 0]
    lon = GPS_Coordinates[:, :, 1]

    # Plot each GPS unit's trajectory
    for i in range(lat.shape[1]):
        plt.scatter(lon[:, i], lat[:, i], label=f"GPS Unit {i + 1}", s=1)

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("GPS Trajectory")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data = np.load(gps_data_path("GPS_Data/Processed_GPS_Receivers_DOG_1.npz"))
    ecef = data["GPS_Coordinates"][::50]  # shape (T, n_receivers, 3)

    # flatten to (T*n_receivers, 3), convert, then reshape back
    T, R, _ = ecef.shape
    ecef_flat = ecef.reshape(-1, 3)
    lat_flat, lon_flat, alt_flat = ECEF_Geodetic(ecef_flat)

    # now each is length T*R → reshape to (T, R)
    lat = lat_flat.reshape(T, R)
    lon = lon_flat.reshape(T, R)
    alt = alt_flat.reshape(T, R)

    # stack into (T, R, 3)
    GPS_Coordinates = np.stack((lat, lon, alt), axis=2)

    GPS_Trajectory(GPS_Coordinates)
