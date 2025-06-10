import numpy as np
from GeigerMethod.GPS_Lever_Arms import GPS_Lever_arms


def test_gps_lever_arms_basic(capsys):
    coords = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        ]
    )
    GPS_Lever_arms(coords)
    captured = capsys.readouterr()
    assert captured.out
