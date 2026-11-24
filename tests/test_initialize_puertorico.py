import numpy as np
import pytest
from pymap3d import Ellipsoid, ecef2geodetic

from Inversion_Workflow.PuertoRico.Initialize_PuertoRico_Data import (
    CDOG_guess,
    clock_files,
    find_valid_receivers,
    gps1_to_others,
    initial_lever_guess,
    load_cdog_data,
    reconstruct_gps_grid,
)
from puertorico_data import gps_data_path

needs_cdog_data = pytest.mark.skipif(
    not all(
        gps_data_path(f"CDOG_Data/{name}").exists() for name in clock_files.values()
    ),
    reason="Puerto Rico C-DOG data not present",
)


def test_cdog_guess_matches_before_inversion_estimate():
    latitude, longitude, elevation = ecef2geodetic(
        *CDOG_guess,
        ell=Ellipsoid.from_name("grs80"),
    )

    assert np.allclose(
        [latitude, longitude, elevation],
        [18.88839612, -66.25572919, -2999.87393207],
        atol=1e-6,
    )


def rotation_z(angle):
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_schematic_geometry_matches_surveyed_ranges():
    expected = {
        (0, 1): 1.2660,
        (2, 3): 1.1620,
        (0, 2): 3.1264,
        (1, 3): 3.1264,
        (0, 3): 3.3535,
        (1, 2): 3.3535,
    }

    for (first, second), distance in expected.items():
        separation = np.linalg.norm(gps1_to_others[first] - gps1_to_others[second])
        assert np.isclose(separation, distance, atol=2e-4)

    assert np.allclose(initial_lever_guess[:2], [-0.948182, 0.633095])
    assert initial_lever_guess[2] == 0.3


def test_four_receiver_preference_and_triplet_fallback():
    rotation = rotation_z(np.deg2rad(25.0))
    translation = np.array([1_000.0, -2_000.0, 500.0])
    exact = (rotation @ gps1_to_others.T).T + translation

    coordinates = np.repeat(exact[np.newaxis, :, :], 4, axis=0)
    coordinates[1, 3] = np.nan
    coordinates[2, 3] += np.array([10.0, 0.0, 0.0])
    coordinates[3, 2:] = np.nan

    nsv = np.full((4, 4), 10.0)
    gdop = np.full((4, 4), 2.0)
    used_receivers = find_valid_receivers(coordinates, nsv, gdop)

    assert used_receivers.tolist() == [
        [True, True, True, True],
        [True, True, True, False],
        [True, True, True, False],
        [False, False, False, False],
    ]


def test_triplet_reconstructs_four_receiver_grid():
    rotation = rotation_z(np.deg2rad(-32.0))
    translation = np.array([10.0, 20.0, -5.0])
    coordinates = (rotation @ gps1_to_others.T).T + translation
    coordinates[3] = np.nan
    used_receivers = np.array([[True, True, True, False]])

    fitted = reconstruct_gps_grid(coordinates[np.newaxis], used_receivers)
    expected = (rotation @ gps1_to_others.T).T + translation

    assert np.allclose(fitted[0], expected, atol=1e-9)


@needs_cdog_data
def test_cdog_clock_segments_have_separate_adjustments():
    before = load_cdog_data("before")
    after = load_cdog_data("after")

    before_data, before_bounds, before_adjustment, before_spread = before
    after_data, after_bounds, after_adjustment, after_spread = after

    assert np.all(np.diff(before_data.sum(axis=1)) >= 0)
    assert np.all(np.diff(after_data.sum(axis=1)) >= 0)
    assert np.isclose(before_adjustment, 166130.6585374, rtol=0, atol=1e-6)
    assert np.isclose(after_adjustment, 277222.6099865, rtol=0, atol=1e-6)
    assert np.allclose(before_bounds, [5858.3414626, 111028.3414626])
    assert np.allclose(after_bounds, [6066.3900135, 88271.3900135])
    assert np.isclose(before_spread, 0.2575852)
    assert np.isclose(after_spread, 0.3022043)
