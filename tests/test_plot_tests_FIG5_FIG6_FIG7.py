import numpy as np
import pytest

from plot_tests import _figure_common as fc


def test_spread_over_sd():
    assert fc.spread_over_sd([1.0, 4.0, 2.0], [0.5, 1.0, 2.0]) == pytest.approx(3.0)


def test_spread_over_sd_rejects_one_model():
    with pytest.raises(ValueError):
        fc.spread_over_sd([1.0], [1.0])


def test_gapped_autocorrelation_without_gaps_matches_autocorrelation():
    x = np.random.default_rng(0).normal(size=200)
    t = np.arange(200.0)
    # Without gaps each lag is divided by its own pair count, not by n.
    expected = fc.autocorrelation(x, 5) * 200 / (200 - np.arange(6))
    expected /= expected[0]
    assert np.allclose(fc.gapped_autocorrelation(t, x, 5), expected)


def test_gapped_autocorrelation_uses_only_pairs_present():
    # Alternating +1/-1 every second with every third second missing: lag 1 pairs
    # always differ in sign, lag 2 pairs always agree.
    t = np.array([s for s in range(30) if s % 3 != 2], dtype=float)
    x = np.where(t % 2 == 0, 1.0, -1.0)
    acf = fc.gapped_autocorrelation(t, x, 2)
    centred = x - x.mean()
    lag1 = [
        centred[i] * centred[j]
        for i in range(len(t))
        for j in range(len(t))
        if t[j] - t[i] == 1
    ]
    assert acf[1] == pytest.approx(np.mean(lag1) / np.mean(centred**2))
    assert acf[1] < 0 < acf[2]


def test_gapped_autocorrelation_rejects_fractional_seconds():
    with pytest.raises(ValueError):
        fc.gapped_autocorrelation([0.0, 0.5, 2.0], [1.0, 2.0, 3.0], 1)


def test_abs_normal_bin_probability():
    probability = fc.abs_normal_bin_probability([0.0, 1.0, 1.959964, np.inf])
    assert probability == pytest.approx([0.6827, 0.2673, 0.05], abs=1e-4)
