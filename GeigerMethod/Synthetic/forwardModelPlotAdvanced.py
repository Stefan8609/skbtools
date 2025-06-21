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
