import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from ECEF_Geodetic import ECEF_Geodetic
from numba import njit
from pymap3d import geodetic2ecef, ecef2geodetic
from Numba_Geiger import findTransponder
from Numba_time_bias import calculateTimesRayTracing_Bias


def load_and_process_data(path):
    data = sio.loadmat(path)
    days = data['days'].flatten() - 59015
    times = data['times'].flatten()
    datetimes = (days * 24 * 3600) + times
    condition_GNSS = (datetimes/3600 >= 25) & (datetimes / 3600 <= 40.9)

    datetimes = datetimes[condition_GNSS]
    time_GNSS = datetimes
    x, y, z = data['x'].flatten()[condition_GNSS], data['y'].flatten()[condition_GNSS], data['z'].flatten()[
        condition_GNSS]
    # x,y,z = data['x'].flatten(), data['y'].flatten(), data['z'].flatten()

    return time_GNSS, x, y, z

def bermuda_trajectory(time_noise, position_noise, dz_array, angle_array, esv_matrix):
    """Calculate trajectory and synthetic arrival times for Bermuda dataset"""

    paths = [
        '../../../GPSData/Unit1-camp_bis.mat',
        '../../../GPSData/Unit2-camp_bis.mat',
        '../../../GPSData/Unit3-camp_bis.mat',
        '../../../GPSData/Unit4-camp_bis.mat'
    ]

    all_data = [load_and_process_data(path) for path in paths]
    common_datetimes = set(all_data[0][0])
    for data in all_data[1:]:
        common_datetimes.intersection_update(data[0])
    common_datetimes = sorted(common_datetimes)

    filtered_data = []
    for datetimes, x, y, z in all_data:
        mask = np.isin(datetimes, common_datetimes)
        filtered_data.append([np.array(datetimes)[mask], np.array(x)[mask], np.array(y)[mask], np.array(z)[mask]])
    filtered_data = np.array(filtered_data)

    # Initialize Coordinates in form of Geiger's Method
    GPS_Coordinates = np.zeros((len(filtered_data[0, 0]), 4, 3))
    for i in range(len(filtered_data[0, 0])):
        for j in range(4):
            GPS_Coordinates[i, j, 0] = filtered_data[j, 1, i]
            GPS_Coordinates[i, j, 1] = filtered_data[j, 2, i]
            GPS_Coordinates[i, j, 2] = filtered_data[j, 3, i]

    lat = sio.loadmat('../../../GPSData/Unit1-camp_bis.mat')['lat'].flatten()
    lon = sio.loadmat('../../../GPSData/Unit1-camp_bis.mat')['lon'].flatten()
    elev = sio.loadmat('../../../GPSData/Unit1-camp_bis.mat')['elev'].flatten()

    GPS_data = filtered_data[0, 0, :]

    gps1_to_others = np.array([[0.0, 0.0, 0.0], [-2.4054, -4.20905, 0.060621], [-12.1105, -0.956145, 0.00877],
                               [-8.70446831, 5.165195, 0.04880436]])

    CDOG_augment = np.array([974.12667502, -80.98121315, -805.07870249])
    lever = np.array([-12.48862757, 0.22622633, -15.89601934])
    offset = 1991.01236648

    CDOG_geodetic = np.array([np.mean(lat), np.mean(lon), np.mean(elev)]) + np.array([0, 0, -5200])
    CDOG_base = np.array(geodetic2ecef(CDOG_geodetic[0], CDOG_geodetic[1], CDOG_geodetic[2]))
    CDOG = CDOG_base + CDOG_augment

    transponder_coordinates = findTransponder(GPS_Coordinates, gps1_to_others, lever)
    synthetic_travel_times, esv = calculateTimesRayTracing_Bias(CDOG, transponder_coordinates, 0, dz_array, angle_array, esv_matrix)

    CDOG_time = GPS_data + synthetic_travel_times + np.random.normal(0, time_noise, len(GPS_data)) + offset
    CDOG_remain, CDOG_int = np.modf(CDOG_time)

    CDOG_data = np.stack((CDOG_int, CDOG_remain), axis=0)
    CDOG_data = CDOG_data.T

    GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))
    return CDOG_data, CDOG, GPS_Coordinates, GPS_data, transponder_coordinates




if __name__ == "__main__":
    esv_table = sio.loadmat('../../../GPSData/global_table_esv.mat')
    dz_array = esv_table['distance'].flatten()
    angle_array = esv_table['angle'].flatten()
    esv_matrix = esv_table['matrice']

    time_noise = 0
    position_noise = 0

    CDOG_data, CDOG, GPS_Coordinates, GPS_data, transponder_coordinates = bermuda_trajectory(time_noise, position_noise,
                                                                                             dz_array, angle_array, esv_matrix)


    lat = sio.loadmat('../../../GPSData/Unit1-camp_bis.mat')['lat'].flatten()
    lon = sio.loadmat('../../../GPSData/Unit1-camp_bis.mat')['lon'].flatten()
    elev = sio.loadmat('../../../GPSData/Unit1-camp_bis.mat')['elev'].flatten()
    times = sio.loadmat('../../../GPSData/Unit1-camp_bis.mat')['times'].flatten()
    days = sio.loadmat('../../../GPSData/Unit1-camp_bis.mat')['days'].flatten() - 59015

    datetimes = (days * 24 * 3600) + times
    condition_GNSS = (datetimes/3600 >= 25) & (datetimes / 3600 <= 40.9)

    datetimes = datetimes[condition_GNSS]
    lat = lat[condition_GNSS]
    lon = lon[condition_GNSS]
    elev = elev[condition_GNSS]

    points = len(lat)
    colors = plt.cm.viridis(np.linspace(0, 1, points))

    CDOG_lat, CDOG_lon,CDOG_depth = ecef2geodetic(CDOG[0], CDOG[1], CDOG[2])
    CDOG_lon += 360

    # Calculate time values in hours for proper colorbar range
    times_hours = datetimes / 3600  # Convert seconds to hours
    min_time = np.min(times_hours)
    max_time = np.max(times_hours)

    scatter = plt.scatter(lon, lat, s=1, c=times_hours, cmap='viridis', label='Surface Vessel')
    plt.scatter(CDOG_lon, CDOG_lat, marker='x', s=5, label='CDOG')
    plt.colorbar(scatter, label='Elapsed Time (hours)')
    plt.clim(min_time, max_time)  # Set the colorbar to actual time range
    plt.title('Plot of Bermuda Trajectory and CDOG location')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.show()
    print(CDOG_data, GPS_data)

