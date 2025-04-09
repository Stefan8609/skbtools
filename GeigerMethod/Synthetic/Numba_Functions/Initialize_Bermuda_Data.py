import numpy as np
import scipy.io as sio
from pymap3d import geodetic2ecef

def initialize_bermuda(GNSS_start, GNSS_end, CDOG_augment):
    # Load GNSS Data during the time of expedition (25 through 40.9) hours
    def load_and_process_data(path, GNSS_start, GNSS_end):
        data = sio.loadmat(path)
        days = data['days'].flatten() - 59015
        times = data['times'].flatten()
        datetimes = (days * 24 * 3600) + times
        condition_GNSS = (datetimes / 3600 >= GNSS_start) & (datetimes / 3600 <= GNSS_end)

        datetimes = datetimes[condition_GNSS]
        time_GNSS = datetimes
        x, y, z = data['x'].flatten()[condition_GNSS], data['y'].flatten()[condition_GNSS], data['z'].flatten()[
            condition_GNSS]
        return time_GNSS, x, y, z

    paths = [
        '../../../GPSData/Unit1-camp_bis.mat',
        '../../../GPSData/Unit2-camp_bis.mat',
        '../../../GPSData/Unit3-camp_bis.mat',
        '../../../GPSData/Unit4-camp_bis.mat'
    ]

    all_data = [load_and_process_data(path, GNSS_start, GNSS_end) for path in paths]
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

    # Initialize time-tagged data for GPS and CDOG
    GPS_data = filtered_data[0, 0, :]
    CDOG_data = sio.loadmat('../../../GPSData/DOG3-camp.mat')['tags'].astype(float)

    lat = sio.loadmat('../../../GPSData/Unit1-camp_bis.mat')['lat'].flatten()
    lon = sio.loadmat('../../../GPSData/Unit1-camp_bis.mat')['lon'].flatten()
    elev = sio.loadmat('../../../GPSData/Unit1-camp_bis.mat')['elev'].flatten()

    CDOG_guess_geodetic = np.array([np.mean(lat), np.mean(lon), np.mean(elev)]) + np.array([0, 0, -5200])
    CDOG_guess_base = np.array(geodetic2ecef(CDOG_guess_geodetic[0], CDOG_guess_geodetic[1], CDOG_guess_geodetic[2]))
    CDOG_guess = CDOG_guess_base + CDOG_augment

    gps1_to_others = np.array([[0.0, 0.0, 0.0], [-2.4054, -4.20905, 0.060621], [-12.1105, -0.956145, 0.00877],
                               [-8.70446831, 5.165195, 0.04880436]])

    # Scale GPS Clock slightly and scale CDOG clock to nanoseconds
    GPS_data = GPS_data - 68826

    CDOG_data[:, 1] = CDOG_data[:, 1] / 1e9

    return GPS_Coordinates, GPS_data, CDOG_data, CDOG_guess, gps1_to_others
