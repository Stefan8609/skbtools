"""Common dc(t) model: shared GPS-time windows plus a dc offset per DOG."""

import numpy as np
import pytest

from Inversion_Workflow.Forward_Model.Calculate_Time_Split import (
    calculateTimesRayTracing_windows,
)
from Inversion_Workflow.MCMC_Functions import MCMC_sampler as ms

M2_PERIOD_S = 12.4206012 * 3600.0
TRUE_OFFSET_M_S = np.array([-0.12, -0.05])
TRUE_TIME_BIAS_S = np.array([2e-4, -1e-4])


def local_axes():
    """East, north and up unit vectors (ECEF) at CDOG_reference."""
    reference = np.array(ms.CONFIG["CDOG_reference"])
    up = reference / np.linalg.norm(reference)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    return east, np.cross(up, east), up


def true_common(edges_s):
    """Step of +0.1 m/s at 2.5 h plus a 30 mm/s M2 term, at window centres.

    Piecewise constant on the windows (so the model can represent it exactly),
    with zero mean over the windows as the model's gauge requires.
    """
    centre = 0.5 * (edges_s[:-1] + edges_s[1:]) - edges_s[0]
    series = 0.1 * (centre > 2.5 * 3600) + 0.03 * np.sin(
        2 * np.pi * centre / M2_PERIOD_S
    )
    return series - series.mean()


def two_dog_problem(window_s=1800.0, edges_s=None, noise_s=1e-7, gap=None):
    """Two DOGs 5 km below a vessel on a 4-h Lissajous track (epochs every 5 s).

    Arrivals follow the common-window model with dc = TRUE_OFFSET + true_common
    on windows of ``window_s`` from the first epoch (the truth's windows), the
    TRUE_TIME_BIAS and white noise of ``noise_s`` seconds. ``gap`` (start, end)
    in seconds from the first epoch removes GPS epochs. The problem is built with
    ``edges_s`` (absolute GPS seconds) or, if None, the sampler's own windows.
    Returns the configuration, the problem and the truth's edges.
    """
    east, north, up = local_axes()
    reference = np.array(ms.CONFIG["CDOG_reference"])
    grid = np.array(ms.CONFIG["gps1_to_others"])
    lever = np.array([-13.0, 9.5, -14.5])
    dog_enu = np.array([[-500.0, 300.0, 0.0], [700.0, -400.0, -50.0]])
    dog_aug = dog_enu @ np.vstack([east, north, up])

    t = np.arange(0.0, 4 * 3600.0, 5.0)
    if gap is not None:
        t = t[(t < gap[0]) | (t >= gap[1])]
    GPS_data = 1000.0 + t
    x = 3000.0 * np.sin(2 * np.pi * t / 5400.0)
    y = 2500.0 * np.sin(2 * np.pi * t / 3600.0)
    # Heading along the track and a 3-degree roll: the vessel turns and tilts, so
    # the lever arm (vertical part included) is not a common shift of the DOGs.
    vx = np.cos(2 * np.pi * t / 5400.0) / 5400.0 * 3000.0
    vy = np.cos(2 * np.pi * t / 3600.0) / 3600.0 * 2500.0
    roll = np.deg2rad(3.0) * np.sin(2 * np.pi * t / 37.0)
    GPS_Coordinates = np.empty((t.size, 4, 3))
    transducer = np.empty((t.size, 3))
    for i in range(t.size):
        gps1 = reference + 5000.0 * up + x[i] * east + y[i] * north
        heading = (vx[i] * east + vy[i] * north) / np.hypot(vx[i], vy[i])
        port = np.cross(up, heading)
        body_to_ecef = np.column_stack(
            [
                heading,
                np.cos(roll[i]) * port + np.sin(roll[i]) * up,
                -np.sin(roll[i]) * port + np.cos(roll[i]) * up,
            ]
        )
        GPS_Coordinates[i] = gps1 + grid @ body_to_ecef.T
        transducer[i] = gps1 + body_to_ecef @ lever

    dz_array = np.arange(4000.0, 6001.0, 100.0)
    angle_array = np.linspace(0.0, 90.0, 91)
    # Angle-dependent table, so dc is not a pure scale of the geometric range.
    esv_matrix = 1500.0 + np.tile(0.2 * angle_array, (dz_array.size, 1))
    truth_edges = GPS_data[0] + np.append(np.arange(0.0, t[-1], window_s), t[-1])
    common = true_common(truth_edges)
    epoch_window = ms.window_index(GPS_data, truth_edges)
    rng = np.random.default_rng(11)
    CDOG_all_data = []
    for j in range(2):
        times, _ = calculateTimesRayTracing_windows(
            reference + dog_aug[j],
            transducer,
            TRUE_OFFSET_M_S[j] + common,
            epoch_window,
            dz_array,
            angle_array,
            esv_matrix,
        )
        arrival = GPS_data + times + 100.0 - TRUE_TIME_BIAS_S[j]
        arrival += rng.normal(0.0, noise_s, t.size)
        CDOG_all_data.append(
            np.column_stack([np.floor(arrival), arrival - np.floor(arrival)])
        )

    config = {
        **ms.CONFIG,
        **ms.COMMON_WINDOWS,
        "run_name": "synthetic_common",
        "likelihood": "ar1",
        "dogs": [1, 3],
        "esv_window_s": window_s,
        "esv_window_min_pairs": 50,
        "esv_window_edges_s": None if edges_s is None else list(edges_s),
        "offsets_s": [100.0, 100.0],
        "outlier_mad_factor": None,
        "start_lever": (lever + 0.05).tolist(),
        "start_CDOG_aug": (dog_aug + 0.1).tolist(),
        "start_esv_bias": [0.0, 0.0],
        "prior_CDOG_aug_center_m": dog_aug.tolist(),
    }
    problem = ms.build_problem(
        config,
        GPS_Coordinates,
        GPS_data,
        CDOG_all_data,
        dz_array,
        angle_array,
        esv_matrix,
    )
    return config, problem, truth_edges, (lever, dog_aug, common)


def common_model(config, problem):
    """build_model with the edges the problem used, as setup_bermuda does."""
    return ms.build_model(
        {**config, "esv_window_edges_s": problem["esv_window_edges_s"]}
    )


def test_window_edges_merge_sparse_windows_across_a_gap():
    pair_time = np.concatenate([np.arange(0.0, 1000.0), np.arange(2500.0, 4000.0)])
    edges, n_merged = ms.common_window_edges(pair_time, 600.0, 100)
    counts = np.bincount(ms.window_index(pair_time, edges), minlength=len(edges) - 1)

    assert edges[0] == 0.0 and edges[-1] == 3999.0
    assert np.all(np.diff(edges) > 0)
    assert counts.sum() == pair_time.size
    assert np.all(counts >= 100)
    # 600-s windows: [1200, 1800) and [1800, 2400) are empty and merged.
    assert n_merged == 2
    assert edges == [0.0, 600.0, 2400.0, 3000.0, 3600.0, 3999.0]


def test_window_edges_short_record_gives_one_window():
    edges, n_merged = ms.common_window_edges(np.arange(0.0, 50.0), 600.0, 100)
    assert edges == [0.0, 49.0] and n_merged == 0


def test_window_index_puts_unpaired_epochs_in_end_windows():
    edges = [10.0, 20.0, 30.0]
    index = ms.window_index(np.array([0.0, 10.0, 19.9, 20.0, 30.0, 99.0]), edges)
    assert index.tolist() == [0, 0, 0, 1, 1, 1]


def test_given_edges_with_an_empty_window_raise():
    # GPS gap from 3600 s to 5400 s; a window inside it has no pairs.
    _, problem, _, _ = two_dog_problem(gap=(3600.0, 5400.0))
    start = 1000.0
    edges = [start, start + 3600.0, start + 5400.0, start + 4 * 3600.0 - 5.0]
    with pytest.raises(ValueError, match="hold no pairs"):
        two_dog_problem(edges_s=edges, gap=(3600.0, 5400.0))
    # The automatic windows never leave a window without data.
    counts = np.bincount(problem["pair_window"])
    assert counts.size == len(problem["esv_window_edges_s"]) - 1
    assert np.all(counts >= 50)


def test_build_model_needs_edges_and_rejects_bad_combinations():
    config = {**ms.CONFIG, **ms.COMMON_WINDOWS}
    with pytest.raises(ValueError, match="esv_window_edges_s is not set"):
        ms.build_model(config)
    edges = {"esv_window_edges_s": [0.0, 1.0, 2.0]}
    with pytest.raises(ValueError, match="num_esv_splits = 1"):
        ms.build_model({**config, **edges, "num_esv_splits": 4})
    with pytest.raises(ValueError, match="sampled-noise"):
        ms.build_model({**config, **edges, "likelihood": "iid_fixed"})
    with pytest.raises(ValueError, match="esv_model"):
        ms.build_model({**config, **edges, "esv_model": "windows"})


def test_layout_and_series_gauge():
    config = {**ms.CONFIG, **ms.COMMON_WINDOWS, "esv_window_edges_s": [0, 1, 2, 3.0]}
    model = ms.build_model(config)
    layout, names = model["layout"], model["names"]
    assert "esv_bias" not in layout
    assert [names[i] for i in range(*layout["esv_offset"].indices(len(names)))] == [
        "esv_offset_DOG1",
        "esv_offset_DOG3",
        "esv_offset_DOG4",
    ]
    assert layout["esv_common"].stop - layout["esv_common"].start == 2
    theta = model["start_theta"].copy()
    theta[layout["esv_offset"]] = [0.1, 0.2, 0.3]
    theta[layout["esv_common"]] = [0.01, -0.03]
    dc = ms.unpack(theta, model)["esv_bias"]
    np.testing.assert_allclose(dc[1], [0.21, 0.17, 0.22])
    np.testing.assert_allclose(dc.mean(axis=1), [0.1, 0.2, 0.3])
    out = ms.derived_outputs(theta[None], model, {"offsets": np.zeros(3)})
    np.testing.assert_allclose(out["esv_common"][0], [0.01, -0.03, 0.02])
    np.testing.assert_allclose(out["esv_bias"][0], dc)


def test_default_config_is_the_splits_model():
    # Switch off: no new keys in CONFIG, so existing runs build the same model.
    assert "esv_model" not in ms.CONFIG
    config = ms.run_config("ar1")
    explicit = ms.build_model({**config, "esv_model": "splits"})
    default = ms.build_model(config)
    assert default["names"] == explicit["names"]
    assert "esv_bias" in default["layout"] and "esv_offset" not in default["layout"]


def test_zero_common_series_equals_one_dc_per_dog():
    config, problem, _, _ = two_dog_problem()
    model = common_model(config, problem)
    single = ms.build_model(
        {**config, "esv_model": "splits", "esv_window_edges_s": None}
    )
    plain_problem = {
        key: value
        for key, value in problem.items()
        if key not in ("esv_window_edges_s", "epoch_window", "pair_window")
    }
    theta = model["start_theta"].copy()
    theta[model["layout"]["esv_offset"]] = [-0.11, -0.06]
    theta_single = single["start_theta"].copy()
    theta_single[single["layout"]["esv_bias"]] = [-0.11, -0.06]

    residual_common = ms.residuals_cm(theta, model, problem)
    residual_single = ms.residuals_cm(theta_single, single, plain_problem)
    np.testing.assert_array_equal(residual_common, residual_single)
    with pytest.raises(ValueError, match="disagree"):
        ms.residuals_cm(theta_single, single, problem)


@pytest.mark.slow
def test_noiseless_common_series_and_offsets_are_recovered():
    config, problem, truth_edges, (lever, dog_aug, common) = two_dog_problem()
    # Build with the truth's windows (the automatic ones are the same here).
    assert np.allclose(problem["esv_window_edges_s"], truth_edges)
    model = common_model(config, problem)
    theta, _, _, _ = ms.laplace_approximation(model, problem, max_iterations=12)
    params = ms.unpack(theta, model)
    recovered_common = params["esv_bias"][0] - params["esv_bias"][0].mean()

    # Noise of 1e-7 s (0.15 mm of range): errors are about 1e-5 of the values.
    np.testing.assert_allclose(recovered_common, common, atol=5e-5)
    offsets = params["esv_bias"].mean(axis=1)
    np.testing.assert_allclose(offsets, TRUE_OFFSET_M_S, atol=5e-4)
    np.testing.assert_allclose(params["time_bias"], TRUE_TIME_BIAS_S, atol=2e-6)
    np.testing.assert_allclose(params["CDOG_aug"], dog_aug, atol=1e-3)
    np.testing.assert_allclose(params["lever"], lever, atol=1e-3)


def test_common_window_jacobian_columns_match_analytic():
    config, problem, _, _ = two_dog_problem()
    model = common_model(config, problem)
    layout = model["layout"]
    theta = model["start_theta"].copy()
    theta[layout["esv_offset"]] = TRUE_OFFSET_M_S
    jacobian = ms.residual_jacobian(theta, model, problem)

    # residual = (observed - distance / esv) * scale, so d residual / d dc of a
    # pair is distance / esv**2 * scale = time / esv * scale.
    params = ms.unpack(theta, model)
    trans = (problem["R_stacked"] @ params["lever"]).reshape(-1, 3) + problem["d"]
    g = np.empty(problem["pair_dog"].size)
    for j, sel in enumerate(problem["pair_slices"]):
        times, esv = calculateTimesRayTracing_windows(
            problem["CDOG_reference"] + params["CDOG_aug"][j],
            trans,
            params["esv_bias"][j],
            problem["epoch_window"],
            problem["dz_array"],
            problem["angle_array"],
            problem["esv_matrix"],
        )
        index = problem["pair_gps_index"][sel]
        g[sel] = times[index] / esv[index] * ms.RANGE_CM_PER_S

    window = problem["pair_window"]
    last = model["num_splits"] - 1
    for j in range(2):
        column = jacobian[:, layout["esv_offset"].start + j]
        np.testing.assert_allclose(column, g * (problem["pair_dog"] == j), rtol=1e-5)
    for k in range(last):
        column = jacobian[:, layout["esv_common"].start + k]
        expected = g * ((window == k).astype(float) - (window == last))
        np.testing.assert_allclose(column, expected, rtol=1e-5, atol=1e-9)
