import numpy as np
from fitPlane import fitPlane
import scipy.io as sio
import matplotlib.pyplot as plt


def load_and_process_data(path):
    """Load MATLAB GPS data and return time series and coordinates.

    Parameters
    ----------
    path : str or Path
        Path to the ``.mat`` file.

    Returns
    -------
    tuple
        Arrays of datetimes in hours and the corresponding ``x``, ``y`` and ``z``
        coordinates.
    """

    data = sio.loadmat(path)
    days = data["days"].flatten() - 59015
    times = data["times"].flatten()
    datetimes = ((days * 24 * 3600) + times) / 3600
    x, y, z = data["x"].flatten(), data["y"].flatten(), data["z"].flatten()
    return datetimes, x, y, z


paths = [
    "GPSData/Unit1-camp_bis.mat",
    "GPSData/Unit2-camp_bis.mat",
    "GPSData/Unit3-camp_bis.mat",
    "GPSData/Unit4-camp_bis.mat",
]

# Charger les données de tous les fichiers
all_data = [load_and_process_data(path) for path in paths]

# Trouver les datetimes communs
common_datetimes = set(all_data[0][0])
for data in all_data[1:]:
    common_datetimes.intersection_update(data[0])
common_datetimes = sorted(common_datetimes)

# Appliquer le masque pour conserver uniquement les datetimes communs et les valeurs correspondantes
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

xyz_array = np.array(
    [
        np.array(
            [
                [filtered_data[j][1][i] for j in range(4)],
                [filtered_data[j][2][i] for j in range(4)],
                [filtered_data[j][3][i] for j in range(4)],
            ]
        )
        for i in range(len(filtered_data[0][0]))
    ]
)

residuals = np.zeros((len(xyz_array), 4))

for i in range(len(xyz_array)):
    normVect = fitPlane(xyz_array[i][0], xyz_array[i][1], xyz_array[i][2])
    barycenter = np.mean(
        np.array([xyz_array[i][0], xyz_array[i][1], xyz_array[i][2]]), axis=1
    )
    for p in range(4):
        pointVect = (
            np.array([xyz_array[i][0][p], xyz_array[i][1][p], xyz_array[i][2][p]])
            - barycenter
        )
        dot = np.dot(normVect, pointVect)
        normVect_Length = np.linalg.norm(normVect)
        residuals[i, p] = np.linalg.norm((dot * normVect / normVect_Length**2))
        if dot < 0:
            residuals[i, p] = residuals[i, p] * -1

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 4))

ax1.hist(residuals[:, 0], bins=300, range=[-0.5, 0.5], color="blue", alpha=0.7)
ax1.set_xlabel("Residual")
ax1.set_ylabel("Frequency")
ax1.set_title("Residuals of GPS1")

ax2.hist(residuals[:, 1], bins=300, range=[-0.5, 0.5], color="blue", alpha=0.7)
ax2.set_xlabel("Residual")
ax2.set_ylabel("Frequency")
ax2.set_title("Residuals of GPS2")

ax3.hist(residuals[:, 2], bins=300, range=[-0.5, 0.5], color="blue", alpha=0.7)
ax3.set_xlabel("Residual")
ax3.set_ylabel("Frequency")
ax3.set_title("Residuals of GPS3")

ax4.hist(residuals[:, 3], bins=100, range=[-0.5, 0.5], color="blue", alpha=0.7)
ax4.set_xlabel("Residual")
ax4.set_ylabel("Frequency")
ax4.set_title("Residuals of GPS4")

# Adjust spacing between subplots
plt.tight_layout()

plt.show()
