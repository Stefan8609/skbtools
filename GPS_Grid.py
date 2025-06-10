import numpy as np

from RigidBodyMovementProblem import findRotationAndDisplacement

# pf to sf = 4.4 in y
# pf to sa = -10.9 in x, -6.5 in y
# pf to pa = -10.9 in x, -0.4 in y
# z distances are unknown

gps1_to_others = np.array(
    [
        [0.0, 0.0, 0.0],
        [-2.4054, -4.20905, 0.060621],
        [-12.1105, -0.956145, 0.00877],
        [-8.70446831, 5.165195, 0.04880436],
    ]
)

"""pf is defined as GPS unit 1"""
schematic_grid = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, -4.4, 0.060621],
        [-10.9, -6.5, 0.00877],
        [-10.9, 0.4, 0.04880436],
    ]
)
pf_to_transducer = np.array([-17.3, 2.2, -15.3])

xs, ys, zs = schematic_grid.T
new_xs, new_ys, new_zs = gps1_to_others.T
xyzs_init = np.vstack((xs, ys, zs))
xyzs_final = np.vstack((new_xs, new_ys, new_zs))
R_mtrx, d = findRotationAndDisplacement(xyzs_init, xyzs_final)

print("Rotation Matrix:", R_mtrx, "\n", "Translation Vector:", d)

print("Transponder Coordinates:", R_mtrx @ pf_to_transducer)
