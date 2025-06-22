import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from data import gps_data_path


def experimentPathPlot(transponder_coordinates, CDog=None):
    if CDog is None:
        CDog = [None, None, None]
    # Plot path of experiment
    points = len(transponder_coordinates)
    colors = plt.cm.viridis(np.linspace(0, 1, points))

    scatter = plt.scatter(
        transponder_coordinates[:, 0],
        transponder_coordinates[:, 1],
        c=colors,
        s=1,
        marker="o",
        label="Transponder",
    )
    plt.colorbar(scatter, label="Elapsed Time (hours)")
    plt.clim(
        0, (points - 1) / 3600
    )  # Set the color scale limits from 0 to number of points
    if CDog[2]:
        plt.scatter(CDog[0], CDog[1], s=1, marker="x", color="k", label="CDog")
    plt.xlabel("Relative Easting (m)")
    plt.ylabel("Relative Northing (m)")
    plt.title("Vessel Trajectory")
    plt.axis("equal")
    plt.show()
    return


if __name__ == "__main__":

    def load_and_process_data(path):
        data = sio.loadmat(path)
        days = data["days"].flatten() - 59015
        times = data["times"].flatten()
        datetimes = (days * 24 * 3600) + times
        condition_GNSS = (datetimes / 3600 >= 25) & (datetimes / 3600 <= 40.9)
        # condition_GNSS = (datetimes/3600 >= 35.3) & (datetimes / 3600 <= 37.6)
        # condition_GNSS = (datetimes/3600 >= 31.9) & (datetimes / 3600 <= 34.75)

        datetimes = datetimes[condition_GNSS]
        time_GNSS = datetimes
        x, y, z = (
            data["x"].flatten()[condition_GNSS],
            data["y"].flatten()[condition_GNSS],
            data["z"].flatten()[condition_GNSS],
        )
        # x,y,z = data['x'].flatten(), data['y'].flatten(), data['z'].flatten()

        return time_GNSS, x, y, z

    paths = [
        gps_data_path("Unit1-camp_bis.mat"),
        gps_data_path("Unit2-camp_bis.mat"),
        gps_data_path("Unit3-camp_bis.mat"),
        gps_data_path("Unit4-camp_bis.mat"),
    ]

    all_data = [load_and_process_data(path) for path in paths]
    common_datetimes = set(all_data[0][0])
    for data in all_data[1:]:
        common_datetimes.intersection_update(data[0])
    common_datetimes = sorted(common_datetimes)

    filtered_data = []
    for datetimes, x, y, z in all_data:
        mask = np.isin(datetimes, common_datetimes)
        filtered_data.append(
            [
                np.array(datetimes)[mask],
                np.array(x)[mask],
                np.array(y)[mask],
                np.array(z)[mask],
            ]
        )
    filtered_data = np.array(filtered_data)

    # Initialize Coordinates in form of Geiger's Method
    GPS_Coordinates = np.zeros((len(filtered_data[0, 0]), 4, 3))
    for i in range(len(filtered_data[0, 0])):
        for j in range(4):
            GPS_Coordinates[i, j, 0] = filtered_data[j, 1, i]
            GPS_Coordinates[i, j, 1] = filtered_data[j, 2, i]
            GPS_Coordinates[i, j, 2] = filtered_data[j, 3, i]

    # Remove mean from GPS data
    GPS_Coordinates -= np.mean(GPS_Coordinates, axis=0)
    experimentPathPlot(GPS_Coordinates[:, 0])
