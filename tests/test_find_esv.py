import numpy as np

from Inversion_Workflow.Forward_Model.Find_ESV import find_esv

# 3 x 3 table with ESV = 1500 + dz + 10 * beta, which bilinear interpolation
# reproduces exactly inside the table.
DZ_ARRAY = np.array([0.0, 1.0, 2.0])
ANGLE_ARRAY = np.array([20.0, 30.0, 40.0])
ESV_MATRIX = 1500.0 + DZ_ARRAY[:, None] + 10.0 * ANGLE_ARRAY[None, :]


def test_on_grid_points_return_table_values():
    dz = np.array([0.0, 1.0, 2.0, 2.0])
    beta = np.array([20.0, 30.0, 40.0, 20.0])

    esv = find_esv(beta, dz, DZ_ARRAY, ANGLE_ARRAY, ESV_MATRIX)

    assert esv.tolist() == [1700.0, 1801.0, 1902.0, 1702.0]


def test_between_grid_points_interpolates_bilinearly():
    dz = np.array([0.5, 1.25, 0.001])
    beta = np.array([25.0, 37.5, 20.0])

    esv = find_esv(beta, dz, DZ_ARRAY, ANGLE_ARRAY, ESV_MATRIX)

    np.testing.assert_allclose(esv, 1500.0 + dz + 10.0 * beta, rtol=0, atol=1e-9)


def test_queries_outside_the_table_are_clamped_to_its_edges():
    dz = np.array([-5.0, 7.0, 1.0])
    beta = np.array([25.0, 30.0, 95.0])

    esv = find_esv(beta, dz, DZ_ARRAY, ANGLE_ARRAY, ESV_MATRIX)

    np.testing.assert_allclose(esv, [1750.0, 1802.0, 1901.0], rtol=0, atol=1e-9)
