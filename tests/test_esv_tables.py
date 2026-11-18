import numpy as np
import pytest

from acoustics.build_synthetic_esv_tables import (
    ANGLE_ARRAY_DEG,
    DZ_ARRAY_M,
    TRANSDUCER_DEPTH_M,
    construct_esv_grid,
)
from Inversion_Workflow.Forward_Model.Calculate_Times_Bias import (
    check_esv_coverage,
)
from Inversion_Workflow.Synthetic.Generate_Trajectories import generateRealistic

# Bermuda DOG 1/3/4 query dz = 5198.9-5225.7 m and angle 22.8-89.8 deg (G8 report).
BERMUDA_DZ_RANGE_M = (5198.9, 5225.7)
BERMUDA_ANGLE_RANGE_DEG = (22.8, 89.8)
MARGIN_M = 50.0
LEGACY_DZ_ARRAY = np.arange(5148.0, 5198.1, 1.0)
LEGACY_ANGLE_ARRAY = np.arange(20.0, 90.1, 0.175)


def _flat_geometry(dz_m, horizontal_m):
    """DOG ``dz_m`` below one transponder that is ``horizontal_m`` away."""
    guess = np.array([0.0, 0.0, -dz_m])
    transponders = np.array([[horizontal_m, 0.0, 0.0]])
    return guess, transponders


def test_constant_sound_speed_gives_esv_equal_to_c():
    depth = np.arange(0.0, 401.0, 10.0)
    c = np.full_like(depth, 1500.0)
    dz = np.array([100.0, 150.0, 300.0])
    angle = np.array([20.0, 45.0, 90.0])

    esv = construct_esv_grid(depth, c, dz, angle, TRANSDUCER_DEPTH_M)

    np.testing.assert_allclose(esv, 1500.0, rtol=1e-6)


def test_grid_covers_bermuda_geometry_with_margin():
    assert DZ_ARRAY_M[0] <= BERMUDA_DZ_RANGE_M[0] - MARGIN_M
    assert DZ_ARRAY_M[-1] >= BERMUDA_DZ_RANGE_M[1] + MARGIN_M
    assert ANGLE_ARRAY_DEG[0] <= BERMUDA_ANGLE_RANGE_DEG[0]
    assert ANGLE_ARRAY_DEG[-1] >= BERMUDA_ANGLE_RANGE_DEG[1]


def test_grid_covers_unaligned_synthetic_geometry():
    np.random.seed(0)
    CDOG, _, transponder_coordinates, _, _ = generateRealistic(4000)[:5]

    check_esv_coverage(
        CDOG, transponder_coordinates, DZ_ARRAY_M, ANGLE_ARRAY_DEG, real_data=False
    )


def test_coverage_check_raises_for_the_legacy_perturbed_table_at_dog_3_depth():
    guess, transponders = _flat_geometry(5225.0, 15000.0)

    with pytest.raises(ValueError, match="dz queried 5225.00"):
        check_esv_coverage(
            guess, transponders, LEGACY_DZ_ARRAY, LEGACY_ANGLE_ARRAY, real_data=False
        )


def test_coverage_check_flags_angles_outside_the_table():
    guess, transponders = _flat_geometry(5225.0, 200.0)

    with pytest.raises(ValueError, match="angle queried"):
        check_esv_coverage(guess, transponders, DZ_ARRAY_M, ANGLE_ARRAY_DEG[:10], False)


def test_coverage_check_can_warn_instead_of_raising():
    guess, transponders = _flat_geometry(5225.0, 15000.0)

    with pytest.warns(UserWarning, match="does not cover"):
        check_esv_coverage(
            guess,
            transponders,
            LEGACY_DZ_ARRAY,
            LEGACY_ANGLE_ARRAY,
            real_data=False,
            on_outside="warn",
        )


def test_coverage_check_uses_geodetic_depth_for_real_data():
    cdog = np.array([1976671.618715, -5069622.53769779, 3306330.69611698])
    transponders = (cdog * (1 + 5200.0 / np.linalg.norm(cdog)))[None, :]

    check_esv_coverage(cdog, transponders, DZ_ARRAY_M, ANGLE_ARRAY_DEG, True)
    with pytest.raises(ValueError, match="dz queried"):
        check_esv_coverage(cdog, transponders, DZ_ARRAY_M + 400, ANGLE_ARRAY_DEG, True)


def test_coverage_check_rejects_unknown_mode():
    guess, transponders = _flat_geometry(5225.0, 200.0)

    with pytest.raises(ValueError, match="on_outside"):
        check_esv_coverage(
            guess, transponders, DZ_ARRAY_M, ANGLE_ARRAY_DEG, False, on_outside="ignore"
        )
