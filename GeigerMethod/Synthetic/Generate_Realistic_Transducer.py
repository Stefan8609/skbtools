import numpy as np
import random
import matplotlib.pyplot as plt

'''
Generates a realistic transducer that follows behind the trajectory of the boat
(turns are sharp and should be made gradual)

Sensitivity in y-axis is extremely low for some reason (leads to hard to resolve transponder in y-axis)
    x and z are surpisingly fine (although z has low variation?)

Will need to fix in the future
'''

def generateRealistic_Transducer(n):
    # Initialize CDog and GPS locations
    CDog = np.array([random.uniform(-5000, 5000), random.uniform(-5000, 5000), random.uniform(-5225, -5235)])
    x_coords1 = np.sort((np.random.rand(n//4) * 15000) - 7500)
    x_coords2 = -1*np.sort(-1*((np.random.rand(n//4) * 15000) - 7500))
    x_coords3 = np.sort((np.random.rand(n//4) * 15000) - 7500)
    x_coords4 = -1*np.sort(-1*((np.random.rand(n//4) * 15000) - 7500))
    y_coords1 = x_coords1 + (np.random.rand(n//4) * 50) - 25
    y_coords2 = 7500 + (np.random.rand(n//4) * 50) - 25
    y_coords3 = - x_coords3 + (np.random.rand(n//4) * 50) - 25
    y_coords4 = -7500 + (np.random.rand(n//4) * 50) - 25

    x_coords = np.concatenate((x_coords1, x_coords2, x_coords3, x_coords4))
    y_coords = np.concatenate((y_coords1, y_coords2, y_coords3, y_coords4))
    z_coords = 5 * np.sin(np.linspace(0, n//32 * np.pi, n // 4 * 4))
    GPS1_Coordinates = np.column_stack((x_coords, y_coords, z_coords))

    GPS_Coordinates = np.zeros((n//4 * 4, 4, 3))
    transponder_coordinates = np.zeros((n//4 * 4, 3))
    GPS_Coordinates[:, 0] = GPS1_Coordinates

    gps1_to_others = np.array([[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64)
    gps1_to_transponder = np.array([-10, 10, -15], dtype=np.float64)

    num_skipped = 50

    for i in range(num_skipped, n//4 * 4):
        direction = GPS1_Coordinates[i] - GPS1_Coordinates[i-num_skipped]
        direction /= np.linalg.norm(direction)  # Normalize the direction vector

        x_axis = direction
        y_axis = np.cross([0, 0, 1], x_axis)
        x_axis /= np.linalg.norm(x_axis)
        z_axis = np.cross(x_axis, y_axis)

        totalRot = np.array([x_axis, y_axis, z_axis]).T

        for j in range(1, 4):  # Add in other GPS with their rotations and translations for each time step
            GPS_Coordinates[i, j] = GPS_Coordinates[i, 0] + np.matmul(totalRot, gps1_to_others[j])
        # Initialize transponder location
        transponder_coordinates[i] = GPS_Coordinates[i, 0] + np.matmul(totalRot, gps1_to_transponder)

    #delete first "num_skipped" rows
    GPS_Coordinates = GPS_Coordinates[num_skipped:]
    transponder_coordinates = transponder_coordinates[num_skipped:]

    return CDog, GPS_Coordinates, transponder_coordinates, gps1_to_others, gps1_to_transponder

if __name__ == "__main__":
    CDOG, GPS_Coordinates, transponder_coordinates, gps1_to_others, gps1_to_transponder = generateRealistic_Transducer(1000)
    plt.scatter(GPS_Coordinates[::10, 0, 0], GPS_Coordinates[::10, 0, 1], s=1, c='r', label='GPS1')
    plt.scatter(transponder_coordinates[::10,0], transponder_coordinates[::10,1], s=1, c='b', label='Transponder')
    plt.legend()
    plt.show()