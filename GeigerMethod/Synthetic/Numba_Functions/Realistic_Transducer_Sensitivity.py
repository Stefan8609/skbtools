import random
import matplotlib.pyplot as plt
import numpy as np
from Numba_Geiger import *
from Numba_RigidBodyMovementProblem import *
from GeigerMethod.Synthetic.Generate_Realistic_Transducer import generateRealistic_Transducer

def sensitivity_kernel(CDOG, GPS_Coordinates, transponder_coordinates, gps1_to_others, gps1_to_transponder, time_noise, position_noise):
    GPS_Coordinates += np.random.normal(0, position_noise, (len(GPS_Coordinates), 4, 3))

    x_axis = np.linspace(-5, 5, 50)
    y_axis = np.linspace(-5, 5, 50)

    [X, Y] = np.meshgrid(x_axis, y_axis)

    Z_rmse = np.zeros((len(Y[:,0]), len(X[0])))
    Z_dist = np.zeros((len(Y[:,0]), len(X[0])))

    for i in range(len(X[0])):
        print(i)
        for j in range(len(Y[:,0])):
            initial_guess = np.array([random.uniform(-10000, 10000), random.uniform(-10000, 10000), random.uniform(-4000, -6000)])
            lever = gps1_to_transponder + np.array([X[0, i], Y[j, 0], 0])

            transponder_coordinates_Found = findTransponder(GPS_Coordinates, gps1_to_others, lever)
            guess, times_known = geigersMethod(initial_guess, CDOG, transponder_coordinates, transponder_coordinates_Found, time_noise)[:2]

            times_calc = calculateTimesRayTracing(guess, transponder_coordinates_Found)[0]
            diff_data = (times_calc - times_known)*1000
            RMSE = np.sqrt(np.nanmean(diff_data ** 2))
            Z_rmse[j, i] = RMSE
            Z_dist[i, j] = np.linalg.norm(guess - CDOG)

    # Make a 2 figure plot by axes
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sensitivity to Lever-Arm in Inversion (x and y axes) (ms)")
    # Plot the RMSE
    im = axs[0].contourf(X, Y, Z_rmse)
    cbar = fig.colorbar(im, ax=axs[0])
    cbar.set_label("RMSE (ms)", rotation= 270, labelpad = 15)
    axs[0].set_xlabel("Lever error in x-axis (m)")
    axs[0].set_ylabel("Lever error in y-axis (m)")
    axs[0].set_title("RMSE Sensitivity to Lever-Arm in Inversion (x and y axes) (ms)")
    # Plot the distance
    im = axs[1].contourf(X, Y, Z_dist)
    cbar = fig.colorbar(im, ax=axs[1])
    cbar.set_label("Distance (m)", rotation= 270, labelpad = 15)
    axs[1].set_xlabel("Lever error in x-axis (m)")
    axs[1].set_ylabel("Lever error in y-axis (m)")
    axs[1].set_title("Distance Sensitivity to Lever-Arm in Inversion (x and y axes) (m)")
    plt.show()


if __name__ == "__main__":
    #Generate a realistic transducer
    n = 1000
    CDOG, GPS_Coordinates, transponder_coordinates, gps1_to_others, gps1_to_transponder = generateRealistic_Transducer(n)
    sensitivity_kernel(CDOG, GPS_Coordinates, transponder_coordinates, gps1_to_others, gps1_to_transponder, 0, 0)