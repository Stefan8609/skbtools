import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares

data_unit1 = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit1-camp_bis_GPStime_synchro.mat')
data_unit2 = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit3-camp_bis_GPStime_synchro.mat')
data_unit3 = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit1-camp_bis_GPStime_synchro.mat')
data_unit4 = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit3-camp_bis_GPStime_synchro.mat')

x1, y1, z1 = data_unit1['x'].flatten(), data_unit1['y'].flatten(), data_unit1['z'].flatten()
x2, y2, z2 = data_unit2['x'].flatten(), data_unit2['y'].flatten(), data_unit2['z'].flatten()
x3, y3, z3 = data_unit3['x'].flatten(), data_unit3['y'].flatten(), data_unit3['z'].flatten()
x4, y4, z4 = data_unit4['x'].flatten(), data_unit4['y'].flatten(), data_unit4['z'].flatten()

A1 = np.vstack((x1, y1, z1)).T
A2 = np.vstack((x2, y2, z2)).T
A3 = np.vstack((x3, y3, z3)).T
A4 = np.vstack((x4, y4, z4)).T

mask = ~np.isnan(A1).any(axis=1) & ~np.isnan(A2).any(axis=1) & ~np.isnan(A4).any(axis=1) & ~np.isnan(A3).any(axis=1)
A1 = A1[mask]
A2 = A2[mask]
A3 = A3[mask]
A4 = A4[mask]


def euler_to_rotation_matrix(params):
    roll, pitch, yaw = params
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])

    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])

    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

def rotation_error(params, A, B):
    R = euler_to_rotation_matrix(params)
    A_transformed = np.dot(A, R.T)
    error = A_transformed - B
    return error.flatten()


init_params = [0, 0, 0]

# Pour A1 et A2:
result_A2 = least_squares(rotation_error, init_params, args=(A1, A2))
R_final_A2 = euler_to_rotation_matrix(result_A2.x)

# Pour A1 et A3:
result_A3 = least_squares(rotation_error, init_params, args=(A1, A3))
R_final_A3 = euler_to_rotation_matrix(result_A3.x)

# Pour A1 et A4:
result_A4 = least_squares(rotation_error, init_params, args=(A1, A4))
R_final_A4 = euler_to_rotation_matrix(result_A4.x)
