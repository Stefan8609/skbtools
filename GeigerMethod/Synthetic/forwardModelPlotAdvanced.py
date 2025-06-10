import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from advancedGeigerMethod import (
    generateLine,
    calculateTimes,
    computeJacobian,
    findTransponder,
)

(
    CDog,
    GPS_Coordinates,
    transponder_coordinates_Actual,
    gps1_to_others,
    gps1_to_transponder,
) = generateLine(300)

GPS_Coordinates += np.random.normal(
    0, 2 * 10**-2, (len(GPS_Coordinates), 4, 3)
)  # Add noise to GPS
times_known = calculateTimes(
    CDog, transponder_coordinates_Actual, 1515
) + np.random.normal(0, 2 * 10**-5, len(transponder_coordinates_Actual))

transponder_coordinates_Found = findTransponder(
    GPS_Coordinates, gps1_to_others, gps1_to_transponder
)

a = np.linspace(CDog[0] - 2000, CDog[0] + 2000, 51)
b = np.linspace(CDog[1] - 2000, CDog[1] + 2000, 51)
A, B = np.meshgrid(a, b)
distances = np.sqrt((A - CDog[0]) ** 2 + (B - CDog[1]) ** 2 + (-5000 - CDog[2]) ** 2)

vals = np.zeros((51, 51))

idx1 = 0
for i in a:
    print(idx1)
    idx2 = 0
    for j in b:
        times = calculateTimes([i, j, -5000], transponder_coordinates_Found, 1515)
        jacobian = computeJacobian(
            [i, j, -5000], transponder_coordinates_Found, times, 1515
        )
        delta = (
            -1
            * np.linalg.inv(jacobian.T @ jacobian)
            @ jacobian.T
            @ (times - times_known)
        )
        vals[idx1, idx2] = np.linalg.norm(delta)
        idx2 += 1
    idx1 += 1

# Set scale of color map so contours and colorplot are the same color scale
norm = mcolors.Normalize(vmin=vals.min(), vmax=vals.max())

plt.xlim(CDog[0] - 2000, CDog[0] + 2000)
plt.ylim(CDog[1] - 2000, CDog[1] + 2000)
plt.contourf(A, B, vals, levels=100, cmap="jet", norm=norm, zorder=1)
plt.colorbar(label="Model Update Distance (m)")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.title("Absolute Model Update with 10 \u03bcs arrival time noise")
plt.legend(loc="upper left")
contour_levels = np.arange(0, 3000, 250)
contour = plt.contour(
    A, B, distances, levels=contour_levels, cmap="jet", norm=norm, zorder=2
)
plt.scatter(
    transponder_coordinates_Actual[:, 0],
    transponder_coordinates_Actual[:, 1],
    s=10,
    color="k",
    label="Transducer Coordinates",
    zorder=3,
)
plt.scatter(CDog[0], CDog[1], s=40, color="w", marker="x", label="CDOG", zorder=4)
plt.legend(loc="upper left")
plt.show()

# Plot eigenvalues of (Jt * J)^-1 (no obvious relationship)

# test_guess = [1000, 1000, -5000]
# times = calculateTimes(test_guess, GPS_Coordinates, 1515)
# jacobian = computeJacobian(test_guess, GPS_Coordinates, times, 1515)
# mtrx = np.linalg.inv(jacobian.T @ jacobian)
# eigenvalues, eigenvectors = np.linalg.eig(mtrx)
# print(CDog, test_guess)
# print(CDog-test_guess)
# print(eigenvalues, eigenvectors)


# # Create the color plot (This version is deprecated because contour version of higher priority)
# plt.scatter(GPS_Coordinates[:,0], GPS_Coordinates[:,1], s=10, color="k", label="GPS Coordinates (1 cm noise)")
# plt.scatter(CDog[0], CDog[1], s=40, color="w", marker="x", label="CDOG")
# plt.imshow(vals, extent=(CDog[0]-2000,CDog[0]+2000, CDog[1]-2000,CDog[1]+2000), origin='lower', cmap='jet')#, interpolation='bilinear')
# plt.colorbar(label="Model Update Distance (m)")
# plt.xlabel('Easting (m)')
# plt.ylabel('Northing (m)')
# plt.title('Absolute Model Update with 10 \u03BCs arrival time noise')
# plt.legend(loc="upper left")
# plt.show()


# Plot Jacobian Expressed in its units (would be good to show)
#   Determinant, Trace, Eigenvalues, etc?
#   Draw the spectrum of eigenvalues of Jt * J
#   How many eigenvalues ( plot as stem plot )
#   Is one noticably smaller than the other eigenvalues (z less impactful than x and y)

#   Need to see horizontals separated from the verticals (what is the influence of z in optimization)

# Add noise level onto the plot
