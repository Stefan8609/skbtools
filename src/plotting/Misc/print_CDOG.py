import numpy as np
from geometry.ECEF_Geodetic import ECEF_Geodetic

CDOG_guess_base = np.array([1976671.618715, -5069622.53769779, 3306330.69611698])
CDOG_augments = np.array(
    [
        [-397.91498328, 372.60077228, 772.54988249],
        [825.12310871, -110.06128983, -734.81971248],
        [236.24358168, -1307.17257775, -2189.77395842],
    ]
)

CDOG_locations = CDOG_guess_base + CDOG_augments

lat, lon, elev = ECEF_Geodetic(CDOG_locations)
print(lat, lon, elev)
