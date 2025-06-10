import numpy as np
from SVP_Calculations import (
    depth_to_pressure_Leroy,
    depth_to_pressure,
    DelGrosso_SV,
    UNESCO_SV,
    NPL_ESV,
    Mackenzie_ESV,
    Coppens_ESV,
)


def test_depth_to_pressure_functions():
    p1 = depth_to_pressure_Leroy(1000, 30)
    p2 = depth_to_pressure(1000, 30)
    assert np.isclose(p1, 100.6729, atol=1e-3)
    assert np.isclose(p2, 100.9554, atol=1e-3)


def test_sound_speed_models():
    S, T, P, Z, lat = 35.0, 10.0, 100.0, 1000.0, 30.0
    dg = DelGrosso_SV(S, T, P)
    un = UNESCO_SV(S, T, P)
    npl = NPL_ESV(S, T, Z, lat)
    mac = Mackenzie_ESV(S, T, Z)
    cop = Coppens_ESV(S, T, Z)
    assert np.isclose(dg, 1506.138, atol=0.01)
    assert np.isclose(un, 1506.348, atol=0.01)
    assert np.isclose(npl, 1506.170, atol=0.01)
    assert np.isclose(mac, 1506.264, atol=0.01)
    assert np.isclose(cop, 1506.366, atol=0.01)
