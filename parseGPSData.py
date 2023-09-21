import numpy as np
from findPointByPlane import initializeFunction, findTransponder
import scipy.io as sio


def load_and_process_data(path):
    """Charge et traite les données d'une unité."""
    data = sio.loadmat(path)
    days = data['days'].flatten() - 59015
    times = data['times'].flatten()
    datetimes = ((days * 24 * 3600) + times) / 3600
    x, y, z = data['x'].flatten(), data['y'].flatten(), data['z'].flatten()
    return datetimes, x, y, z

paths = [
    'GPSData/Unit1-camp_bis.mat',
    'GPSData/Unit2-camp_bis.mat',
    'GPSData/Unit3-camp_bis.mat',
    'GPSData/Unit4-camp_bis.mat'
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
    filtered_data.append([np.array(datetimes)[mask], np.array(x)[mask], np.array(y)[mask], np.array(z)[mask]])

# with np.printoptions(threshold=np.inf):
#     print(filtered_data[3][0])

#Goal is to make a array of 4 gps points, with each increasing index being the 4 gps points at each next time,
#Can easily initialize the function given the gps schematics from Thalia and the distance to transducer from Thalia

### Below is applying the findPointByPlane algorithm to gps data

## Initialize using the gps vectors and transponder vector that Thalia sent over

xs = [10.2, 10.2, 0, 0]
ys = [0, -4.93, -7.11, 0]
zs = [15.24, 15.24, 15.24, 15.24]
transponder = np.array([-6.4, 2.46, 0])

[theta, phi, length, orientation] = initializeFunction(xs, ys, zs, 3, transponder)

## Apply technique for each set of 4 GPS vectors
xyz_array = [np.array([
    [filtered_data[j][1][i] for j in range(4)],
    [filtered_data[j][2][i] for j in range(4)],
    [filtered_data[j][3][i] for j in range(4)]
]) for i in range(len(filtered_data[0][0]))]

final_xs = [0]*len(xyz_array)
final_ys = [0]*len(xyz_array)
final_zs = [0]*len(xyz_array)

for i in range(len(xyz_array)):
    vect, barycenter = findTransponder(xyz_array[i][0], xyz_array[i][1], xyz_array[i][2], 3, length, theta, phi, orientation)
    final_xs[i], final_ys[i], final_zs[i] = vect + barycenter

transducer_data = {'datetimes': common_datetimes, 'x': final_xs, 'y': final_ys, 'z': final_zs}
sio.savemat('transducer_pos.mat', transducer_data)

# Format the data and save as a .mat file