import numpy as np
import matplotlib.pyplot as plt
from skbtools.geometry.rigid_body import findRotationAndDisplacement

# Define 4 GPS points as an array (each row: [x, y, z])
gps1_to_others = np.array(
    [
        [0.0, 0.0, 0.0],
        [-2.4054, -4.20905, 0.060621],
        [-12.1105, -0.956145, 0.00877],
        [-8.70446831, 5.165195, 0.04880436],
    ]
)

schematic_grid = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, -4.4, 0.060621],
        [-10.9, -6.5, 0.00877],
        [-10.9, 0.4, 0.04880436],
    ]
)

# Define an arbitrary target location for the arrow
pf_to_transducer = np.array([-17.3, 2.2, -15.3])
gps1_to_transducer = np.array([-12.4659, 9.6021, -13.2993])

# Compute the direction vector for the arrow
# arrow_vector = arrow_target - arrow_start

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the GPS points
ax.scatter(
    gps1_to_others[:, 0],
    gps1_to_others[:, 1],
    gps1_to_others[:, 2],
    color="b",
    marker="o",
    s=50,
    label="GPS Points from averaging distance",
)
ax.scatter(
    schematic_grid[:, 0],
    schematic_grid[:, 1],
    schematic_grid[:, 2],
    color="r",
    marker="o",
    s=50,
    label="GPS Points from schematic",
)

# Find the rotation between schematic and average GPS planes
xs, ys, zs = schematic_grid.T
new_xs, new_ys, new_zs = gps1_to_others.T
xyzs_init = np.vstack((xs, ys, zs))
xyzs_final = np.vstack((new_xs, new_ys, new_zs))
R_mtrx, d = findRotationAndDisplacement(xyzs_init, xyzs_final)

# Plot rotated points
rotated_schematic_grid = (R_mtrx @ schematic_grid.T).T
ax.scatter(
    rotated_schematic_grid[:, 0],
    rotated_schematic_grid[:, 1],
    rotated_schematic_grid[:, 2],
    color="g",
    marker="o",
    s=50,
    label="Rotated Schematic Points",
)

# Plot the quiver arrow from arrow_start in the direction of arrow_vector
ax.quiver(
    0,
    0,
    0,
    pf_to_transducer[0],
    pf_to_transducer[1],
    pf_to_transducer[2],
    color="r",
    label="Schematic Transducer",
)
ax.quiver(
    0,
    0,
    0,
    gps1_to_transducer[0],
    gps1_to_transducer[1],
    gps1_to_transducer[2],
    color="b",
    label="Lever Arm Estimate",
)

# Plot difference between original schematic transducer and rotated schematic transducer
rotated_pf = R_mtrx @ pf_to_transducer
diff_arrow = rotated_pf - pf_to_transducer
ax.quiver(
    pf_to_transducer[0],
    pf_to_transducer[1],
    pf_to_transducer[2],
    diff_arrow[0],
    diff_arrow[1],
    diff_arrow[2],
    color="g",
    label="Rotated Transducer",
)


# Set labels and title
ax.set_xlabel(r"X (meters)")
ax.set_ylabel(r"Y (meters)")
ax.set_zlabel(r"Z (meters)")
ax.set_title(r"Plot of Vessel GPS points from Transducer and averaged distance")

ax.set_xlim(-20, 5)
ax.set_ylim(-10, 10)
ax.set_zlim(-20, 10)
ax.legend()
plt.show()
