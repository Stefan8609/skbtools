"""Helpers of plot_tests/_figure_common.py behind FIG-1, FIG-2 and FX-A3."""

import numpy as np
import pytest

from plot_tests import _figure_common as fc


def brute_force_median_mad(values, window):
    half = window // 2
    median, mad = [], []
    for i in range(values.size):
        part = values[max(0, i - half) : i + half + 1]
        median.append(np.median(part))
        mad.append(np.median(np.abs(part - median[-1])))
    return np.array(median), np.array(mad)


@pytest.mark.parametrize("window", [1, 4, 7, 50])
def test_running_median_mad_matches_the_loop_definition(window):
    values = np.random.default_rng(1).normal(size=23)
    median, mad = fc.running_median_mad(values, window)
    expected_median, expected_mad = brute_force_median_mad(values, window)
    np.testing.assert_allclose(median, expected_median)
    np.testing.assert_allclose(mad, expected_mad)


def test_running_median_mad_rejects_2d_input():
    with pytest.raises(ValueError, match="1-D series"):
        fc.running_median_mad(np.zeros((2, 3)), 3)


def test_gnss_rejection_masks_hard_limits_then_band_any_receiver():
    heights = np.full((2, 9), -36.5)
    heights[0, 4] = -36.0  # receiver 1 jumps 0.5 m: outside median ± 2 MAD (MAD 0)
    heights[1, 7] = -34.0  # receiver 2 above the -35 m hard limit
    masks = fc.gnss_rejection_masks(heights, (-38.0, -35.0), window=5)
    assert masks["hard_fail"].sum(axis=1).tolist() == [0, 1]
    assert masks["band_fail"].sum(axis=1).tolist() == [1, 0]
    assert not masks["band_fail"][:, 7].any()  # stage 1 already dropped epoch 7
    assert np.isnan(masks["median_m"][:, 7]).all()
    assert np.flatnonzero(~masks["kept"]).tolist() == [4, 7]


def test_gnss_rejection_masks_rejects_1d_input():
    with pytest.raises(ValueError, match="n_receivers, n_epochs"):
        fc.gnss_rejection_masks(np.zeros(5))


def test_read_esri_ascii_extent_and_nodata(tmp_path):
    path = tmp_path / "tile.asc"
    path.write_text(
        "ncols 3\nnrows 2\nxllcorner -68.7\nyllcorner 31.4\ncellsize 0.01\n"
        "NODATA_value -32767\n-5200 -5210 -32767\n-5220 -5230 -5240\n"
    )
    grid, extent = fc.read_esri_ascii(path)
    np.testing.assert_allclose(extent, (-68.7, -68.67, 31.4, 31.42))
    assert grid[0, 0] == -5200.0  # first row is the northern edge
    assert np.isnan(grid[0, 2])


def test_read_esri_ascii_without_nodata_line(tmp_path):
    path = tmp_path / "tid.asc"
    path.write_text(
        "ncols 2\nnrows 1\nxllcorner -68.7\nyllcorner 31.4\ncellsize 0.5\n11 40\n"
    )
    grid, extent = fc.read_esri_ascii(path)
    assert grid.tolist() == [[11.0, 40.0]]
    np.testing.assert_allclose(extent, (-68.7, -67.7, 31.4, 31.9))
