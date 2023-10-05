import numpy as np
from findPointByPlane import initializeFunction, findXyzt
import scipy.io as sio
import matplotlib.pyplot as plt

def load_and_process_data(path):
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

### Below is applying the findPointByPlane algorithm to gps data

## Initialize using the gps vectors and transponder vector that Thalia sent over
#
# xs = [10.2, 10.2, 0, 0]
# ys = [0, -4.93, -7.11, 0]
# zs = [15.24, 15.24, 15.24, 15.24]
# transponder = np.array([-6.4, 2.46, 0])
# [theta, phi, length, orientation] = initializeFunction(xs, ys, zs, 1, transponder)
#
# print(theta, phi, length, orientation)

#I wonder if gps are right in xyz_array

# ## Apply technique for each set of 4 GPS vectors
# xyz_array = np.array([np.array([
#     [filtered_data[j][1][i] for j in range(4)],
#     [filtered_data[j][2][i] for j in range(4)],
#     [filtered_data[j][3][i] for j in range(4)]
# ]) for i in range(len(filtered_data[0][0]))])
#
# for i in range(len(xyz_array)):
#     average = sum(xyz_array[i][2])/4
#     for j in range(4):
#         xyz_array[i][2][j]=average
#
# final_xs = [0]*len(xyz_array)
# final_ys = [0]*len(xyz_array)
# final_zs = [0]*len(xyz_array)
# vectors = [0]*len(xyz_array)
#
# for i in range(len(xyz_array)):
#     [vect, barycenter, normVect] = findXyzt(xyz_array[i][0], xyz_array[i][1], xyz_array[i][2], 1, length, theta, phi, orientation)
#     final_xs[i], final_ys[i], final_zs[i] = vect + barycenter
#     vectors[i] = np.ndarray.tolist(vect)
#     if i>44940 and i<44943:
#         print(xyz_array[i])
#         print(normVect)
#         print('here','\n')
#     # if final_xs[i] > 1981000 and final_ys[i] > -5070000:
#     #     print(i)
#
#
# print(xyz_array[44941:44943,0],xyz_array[44941:44943,1])
# print(vectors[44941:44943])
#
#
# transducer_data = {'datetimes': common_datetimes, 'x': final_xs, 'y': final_ys, 'z': final_zs}
# sio.savemat('transducer_pos.mat', transducer_data)
#
# # Format the data and save as a .mat file
#
# # plt.scatter(final_xs[44941:44943], final_ys[44941:44943], s=1, color='k', label="Transducer")
# # plt.scatter(xyz_array[44941:44943,0,0], xyz_array[44941:44943,1,0], s=1, color='b', label="GPS1")
# # plt.scatter(xyz_array[44941:44943,0,1], xyz_array[44941:44943,1,1], s=1, color='r', label="GPS2")
# # plt.scatter(xyz_array[44941:44943,0,2], xyz_array[44941:44943,1,2], s=1, color='g', label="GPS3")
# # plt.scatter(xyz_array[44941:44943,0,3], xyz_array[44941:44943,1,3], s=1, color='y', label="GPS4")
#
# plt.scatter(final_xs, final_ys, s=1, color='k')
# plt.scatter(xyz_array[:,0,0], xyz_array[:,1,0], s=1, color='b', label="GPS1")
# plt.scatter(xyz_array[:,0,1], xyz_array[:,1,1], s=1, color='r', label="GPS2")
# plt.scatter(xyz_array[:,0,2], xyz_array[:,1,2], s=1, color='g', label="GPS3")
# plt.scatter(xyz_array[:,0,3], xyz_array[:,1,3], s=1, color='y', label="GPS4")
#
#
# # Titres et légendes
# plt.title("Disposition des Antennes sur le Bateau")
# plt.xlabel("Avant <-> Arrière")
# plt.ylabel("Tribord <-> Babord")
# plt.legend(loc="upper right")
#
# # Afficher le graphe
# plt.show()
#
#
# #error is from orientation being off