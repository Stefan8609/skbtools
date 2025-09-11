import numpy as np
from geometry.ECEF_Geodetic import ECEF_Geodetic

CDOG_augments = np.array(
    [
        [-397.91498328, 372.60077228, 772.54988249],
        [825.12310871, -110.06128983, -734.81971248][
            236.24358168, -1307.17257775, -2189.77395842
        ],
    ]
)

lat, lon, elev = ECEF_Geodetic(CDOG_augments)
