import numpy as np
from geometry.ECEF_Geodetic import ECEF_Geodetic
from pymap3d import geodetic2enu

# Add in the comparison to split inversion

CDOG_guess_base = np.array(
    [1976671.618715, -5069622.53769779, 3306330.69611698]
)

methods = {
    "Full Trajectory MCMC": np.array(
        [
            [-396.73685216, 369.21165118, 774.34118812],
            [826.22200156, -112.71197144, -733.58577345],
            [235.88302714, -1306.08869213, -2190.76178745],
        ]
    ),
    "Cross Trajectory MCMC": np.array(
        [
            [-396.80833750, 370.23261256, 774.14404697],
            [826.18512886, -111.85558415, -733.73615228],
            [235.77476616, -1305.38300727, -2190.89445398],
        ]
    ),
    "Full Trajectory Inversion": np.array(
        [
            [-396.91, 369.80, 774.24],
            [826.22, -112.94, -733.06],
            [236.20, -1306.98, -2189.99],
        ]
    ),
    "Cross Trajectory Inversion": np.array(
        [
            [-396.92874, 369.56785, 774.27707],
            [825.18983, -111.87948, -733.96880],
            [235.06533, -1303.89320, -2191.92964],
        ]
    ),
    # "ESV Split New": np.array([
    #         [-396.8083375,   370.23261256 , 774.14404697],
    #         [ 826.18512886, -111.85558415, -733.73615228],
    #         [ 235.77476616, -1305.38300727, -2190.89445398],
    #     ]
    # ),
}

names = list(methods)

for i in range(len(names)):
    for j in range(i + 1, len(names)):
        locations_1 = CDOG_guess_base + methods[names[i]]
        locations_2 = CDOG_guess_base + methods[names[j]]

        lat_1, lon_1, height_1 = ECEF_Geodetic(locations_1)
        lat_2, lon_2, height_2 = ECEF_Geodetic(locations_2)

        print(f"\n{names[i]} vs. {names[j]}")

        DOG_NAME = ["DOG 1", "DOG 3", "DOG 4"]

        for dog in range(3):
            east, north, up = geodetic2enu(
                lat_2[dog],
                lon_2[dog],
                height_2[dog],
                lat_1[dog],
                lon_1[dog],
                height_1[dog],
            )

            horizontal = np.hypot(east, north)
            distance_3d = np.sqrt(east**2 + north**2 + up**2)

            print(
                f"DOG {DOG_NAME[dog]}: "
                f"E={100*east:8.1f} cm, "
                f"N={100*north:8.1f} cm, "
                f"U={100*up:8.1f} cm, "
                f"horizontal={100*horizontal:8.1f} cm, "
                f"3D={100*distance_3d:8.1f} cm"
            )