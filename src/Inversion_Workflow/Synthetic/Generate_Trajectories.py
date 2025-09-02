import numpy as np
import random
from numba import njit


def generateRandomData(n):  # Generate the random data in the form of numpy arrays
    # Generate CDog
    CDog = np.array(
        [
            random.uniform(-1000, 1000),
            random.uniform(-1000, 1000),
            random.uniform(-5225, -5235),
        ]
    )

    # Generate and initial GPS point to base all others off of
    xyz_point = np.array(
        [
            random.uniform(-1000, 1000),
            random.uniform(-1000, 1000),
            random.uniform(-10, 10),
        ]
    )

    # Generate the translations from initial point
    # (random x,y,z translation with z/100) for each time step
    translations = (np.random.rand(n, 3) * 15000) - 7500
    translations = np.matmul(
        translations, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1 / 100]])
    )

    # Generate rotations from initial point for
    # each time step (yaw, pitch, roll) between -pi/2 to pi/2
    rot = (np.random.rand(n, 3) * np.pi) - np.pi / 2

    # Have GPS coordinates for all 4 GPS at each time step.
    # Also have transponder for each time step
    GPS_Coordinates = np.zeros((n, 4, 3))
    transponder_coordinates = np.zeros((n, 3))

    # Have a displacement vectors to find other GPS
    # from first GPS. Also displacement from first GPS to transponder
    gps1_to_others = np.array(
        [[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64
    )
    gps1_to_transponder = np.array([-10, 3, -15], dtype=np.float64)

    for i in range(n):
        # Build rotation matrix at each time step
        xRot = np.array(
            [
                [1, 0, 0],
                [0, np.cos(rot[i, 0]), -np.sin(rot[i, 0])],
                [0, np.sin(rot[i, 0]), np.cos(rot[i, 0])],
            ]
        )
        yRot = np.array(
            [
                [np.cos(rot[i, 1]), 0, np.sin(rot[i, 1])],
                [0, 1, 0],
                [-np.sin(rot[i, 1]), 0, np.cos(rot[i, 1])],
            ]
        )
        zRot = np.array(
            [
                [np.cos(rot[i, 2]), -np.sin(rot[i, 2]), 0],
                [np.sin(rot[i, 2]), np.cos(rot[i, 2]), 0],
                [0, 0, 1],
            ]
        )
        totalRot = np.matmul(xRot, np.matmul(yRot, zRot))

        for j in range(
            1, 4
        ):  # Add in other GPS with their rotations and translations for each time step
            GPS_Coordinates[i, j] = xyz_point + np.matmul(totalRot, gps1_to_others[j])
            GPS_Coordinates[i, j] += translations[i]

        # Put in known transponder location to get simulated times
        transponder_coordinates[i] = xyz_point + np.matmul(
            totalRot, gps1_to_transponder
        )
        transponder_coordinates[i] += translations[i]

        GPS_Coordinates[i, 0] = xyz_point + translations[i]  # translate original point

    return (
        CDog,
        GPS_Coordinates,
        transponder_coordinates,
        gps1_to_others,
        gps1_to_transponder,
    )


def generateLine(n):
    # Initialize CDog and GPS locations
    CDog = np.array(
        [
            random.uniform(-1000, 1000),
            random.uniform(-1000, 1000),
            random.uniform(-5225, -5235),
        ]
    )
    x_coords = (np.random.rand(n) * 15000) - 7500
    y_coords = x_coords + (np.random.rand(n) * 50) - 25  # variation around x-coord
    z_coords = (np.random.rand(n) * 5) - 10
    GPS1_Coordinates = np.column_stack((x_coords, y_coords, z_coords))
    GPS1_Coordinates = sorted(GPS1_Coordinates, key=lambda k: [k[0], k[1], k[2]])

    GPS_Coordinates = np.zeros((n, 4, 3))
    transponder_coordinates = np.zeros((n, 3))
    GPS_Coordinates[:, 0] = GPS1_Coordinates

    # Randomize boat yaw, pitch, and roll at each time step
    rot = (np.random.rand(n, 3) * np.pi) - np.pi / 2
    gps1_to_others = np.array(
        [[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64
    )
    gps1_to_transponder = np.array([-10, 3, -15], dtype=np.float64)

    for i in range(n):
        # Build rotation matrix at each time step
        xRot = np.array(
            [
                [1, 0, 0],
                [0, np.cos(rot[i, 0]), -np.sin(rot[i, 0])],
                [0, np.sin(rot[i, 0]), np.cos(rot[i, 0])],
            ]
        )
        yRot = np.array(
            [
                [np.cos(rot[i, 1]), 0, np.sin(rot[i, 1])],
                [0, 1, 0],
                [-np.sin(rot[i, 1]), 0, np.cos(rot[i, 1])],
            ]
        )
        zRot = np.array(
            [
                [np.cos(rot[i, 2]), -np.sin(rot[i, 2]), 0],
                [np.sin(rot[i, 2]), np.cos(rot[i, 2]), 0],
                [0, 0, 1],
            ]
        )
        totalRot = np.matmul(xRot, np.matmul(yRot, zRot))
        for j in range(
            1, 4
        ):  # Add in other GPS with their rotations and translations for each time step
            GPS_Coordinates[i, j] = GPS_Coordinates[i, 0] + np.matmul(
                totalRot, gps1_to_others[j]
            )
        # Initialize transponder location
        transponder_coordinates[i] = GPS_Coordinates[i, 0] + np.matmul(
            totalRot, gps1_to_transponder
        )
    return (
        CDog,
        GPS_Coordinates,
        transponder_coordinates,
        gps1_to_others,
        gps1_to_transponder,
    )


def generateCross(n):
    # Initialize CDog and GPS locations
    CDog = np.array(
        [
            random.uniform(-1000, 1000),
            random.uniform(-1000, 1000),
            random.uniform(-5225, -5235),
        ]
    )
    x_coords1 = (np.random.rand(n // 2) * 15000) - 7500
    x_coords2 = (np.random.rand(n // 2) * 15000) - 7500
    x_coords = np.concatenate((np.sort(x_coords1), np.sort(x_coords2)))
    y_coords = x_coords + (np.random.rand(n) * 50) - 25  # variation around x-coord
    y_coords[n // 2 :] *= -1
    z_coords = (np.random.rand(n) * 5) - 10
    GPS1_Coordinates = np.column_stack((x_coords, y_coords, z_coords))

    GPS_Coordinates = np.zeros((n, 4, 3))
    transponder_coordinates = np.zeros((n, 3))
    GPS_Coordinates[:, 0] = GPS1_Coordinates

    # Randomize boat yaw, pitch, and roll at each time step
    rot = (np.random.rand(n, 3) * np.pi) - np.pi / 2
    gps1_to_others = np.array(
        [[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64
    )
    gps1_to_transponder = np.array([-10, 3, -15], dtype=np.float64)

    for i in range(n):
        # Build rotation matrix at each time step
        xRot = np.array(
            [
                [1, 0, 0],
                [0, np.cos(rot[i, 0]), -np.sin(rot[i, 0])],
                [0, np.sin(rot[i, 0]), np.cos(rot[i, 0])],
            ]
        )
        yRot = np.array(
            [
                [np.cos(rot[i, 1]), 0, np.sin(rot[i, 1])],
                [0, 1, 0],
                [-np.sin(rot[i, 1]), 0, np.cos(rot[i, 1])],
            ]
        )
        zRot = np.array(
            [
                [np.cos(rot[i, 2]), -np.sin(rot[i, 2]), 0],
                [np.sin(rot[i, 2]), np.cos(rot[i, 2]), 0],
                [0, 0, 1],
            ]
        )
        totalRot = np.matmul(xRot, np.matmul(yRot, zRot))
        for j in range(
            1, 4
        ):  # Add in other GPS with their rotations and translations for each time step
            GPS_Coordinates[i, j] = GPS_Coordinates[i, 0] + np.matmul(
                totalRot, gps1_to_others[j]
            )
        # Initialize transponder location
        transponder_coordinates[i] = GPS_Coordinates[i, 0] + np.matmul(
            totalRot, gps1_to_transponder
        )
    return (
        CDog,
        GPS_Coordinates,
        transponder_coordinates,
        gps1_to_others,
        gps1_to_transponder,
    )


@njit
def generateRealistic(n):
    """Create a synthetic survey with random boat motion.

    Parameters
    ----------
    n : int
        Number of time steps in the synthetic trajectory.

    Returns
    -------
    tuple of ndarray
        ``(CDog, GPS_Coordinates, transponder_coordinates, gps1_to_others,
        gps1_to_transponder)``.
    """
    # Initialize CDOG and GPS locations
    CDog = np.array(
        [
            random.uniform(-5000, 5000),
            random.uniform(-5000, 5000),
            random.uniform(-5225, -5235),
        ]
    )
    x_coords1 = np.sort((np.random.rand(n // 4) * 15000) - 7500)
    x_coords2 = -1 * np.sort(-1 * ((np.random.rand(n // 4) * 15000) - 7500))
    x_coords3 = np.sort((np.random.rand(n // 4) * 15000) - 7500)
    x_coords4 = -1 * np.sort(-1 * ((np.random.rand(n // 4) * 15000) - 7500))
    y_coords1 = x_coords1 + (np.random.rand(n // 4) * 50) - 25
    y_coords2 = 7500 + (np.random.rand(n // 4) * 50) - 25
    y_coords3 = -x_coords1 + (np.random.rand(n // 4) * 50) - 25
    y_coords4 = -7500 + (np.random.rand(n // 4) * 50) - 25
    x_coords = np.concatenate((x_coords1, x_coords2, x_coords3, x_coords4))
    y_coords = np.concatenate((y_coords1, y_coords2, y_coords3, y_coords4))
    z_coords = (np.random.rand(n // 4 * 4) * 5) - 10
    GPS1_Coordinates = np.column_stack((x_coords, y_coords, z_coords))

    GPS_Coordinates = np.zeros((n // 4 * 4, 4, 3))
    transponder_coordinates = np.zeros((n // 4 * 4, 3))
    GPS_Coordinates[:, 0] = GPS1_Coordinates

    # Randomize boat yaw, pitch, and roll at each time step
    rot = (np.random.rand(n, 3) * np.pi) - np.pi / 2
    gps1_to_others = np.array(
        [[0, 0, 0], [10, 1, -1], [11, 9, 1], [-1, 11, 0]], dtype=np.float64
    )
    gps1_to_transponder = np.array([-10, 3, -15], dtype=np.float64)

    for i in range(n // 4 * 4):
        # Build rotation matrix at each time step
        xRot = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(rot[i, 0]), -np.sin(rot[i, 0])],
                [0.0, np.sin(rot[i, 0]), np.cos(rot[i, 0])],
            ]
        )
        yRot = np.array(
            [
                [np.cos(rot[i, 1]), 0.0, np.sin(rot[i, 1])],
                [0.0, 1.0, 0.0],
                [-np.sin(rot[i, 1]), 0.0, np.cos(rot[i, 1])],
            ]
        )
        zRot = np.array(
            [
                [np.cos(rot[i, 2]), -np.sin(rot[i, 2]), 0.0],
                [np.sin(rot[i, 2]), np.cos(rot[i, 2]), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        totalRot = xRot @ yRot @ zRot
        for j in range(
            1, 4
        ):  # Add in other GPS with their rotations and translations for each time step
            GPS_Coordinates[i, j] = GPS_Coordinates[i, 0] + totalRot @ gps1_to_others[j]
        # Initialize transponder location
        transponder_coordinates[i] = (
            GPS_Coordinates[i, 0] + totalRot @ gps1_to_transponder
        )
    return (
        CDog,
        GPS_Coordinates,
        transponder_coordinates,
        gps1_to_others,
        gps1_to_transponder,
    )
