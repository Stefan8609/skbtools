import numpy as np

import Inversion_Workflow.Modular_Main as modular_main
from Inversion_Workflow.Modular_Main import (
    DEFAULT_REAL_LEVER,
    generate_synthetic,
    load_real_data,
    run,
)


def test_load_real_data_path(monkeypatch):
    captured_path = {}

    def fake_load(path):
        captured_path["value"] = str(path)
        return {
            "GPS_Coordinates": np.zeros((1, 4, 3)),
            "GPS_data": np.zeros(1),
            "CDOG_data": np.zeros((1, 2)),
        }

    monkeypatch.setattr(np, "load", fake_load)

    cdog_data, cdog_guess, gps_coordinates, gps_time = load_real_data(dog_num=4)

    assert captured_path["value"].endswith("GPS_Data/Processed_GPS_Receivers_DOG_4.npz")
    assert cdog_data.shape == (1, 2)
    assert cdog_guess.shape == (3,)
    assert gps_coordinates.shape == (1, 4, 3)
    assert gps_time.shape == (1,)


def test_generate_synthetic_dispatches_bermuda(monkeypatch):
    expected = tuple(
        np.zeros((1, size)) if size > 1 else np.zeros(1) for size in [2, 3, 12, 1, 3]
    )

    def fake_bermuda(*args, **kwargs):
        return expected

    def fail_generate(*args, **kwargs):
        raise AssertionError("generateUnaligned should not be used for bermuda")

    monkeypatch.setattr(modular_main, "TRAJECTORY", "bermuda")
    monkeypatch.setattr(modular_main, "bermuda_trajectory", fake_bermuda)
    monkeypatch.setattr(modular_main, "generateUnaligned", fail_generate)

    result = generate_synthetic(
        np.zeros(1), np.zeros(1), np.zeros((1, 1)), np.zeros((4, 3)), np.zeros(3)
    )
    assert all(np.array_equal(actual, want) for actual, want in zip(result, expected))


def test_generate_synthetic_dispatches_standard_trajectory(monkeypatch):
    expected = tuple(
        np.zeros((1, size)) if size > 1 else np.zeros(1) for size in [2, 3, 12, 1, 3]
    )

    def fake_generate(*args, **kwargs):
        assert kwargs["trajectory"] == "line"
        return expected

    def fail_bermuda(*args, **kwargs):
        raise AssertionError(
            "bermuda_trajectory should not be used for non-bermuda trajectories"
        )

    monkeypatch.setattr(modular_main, "TRAJECTORY", "line")
    monkeypatch.setattr(modular_main, "generateUnaligned", fake_generate)
    monkeypatch.setattr(modular_main, "bermuda_trajectory", fail_bermuda)

    result = generate_synthetic(
        np.zeros(1), np.zeros(1), np.zeros((1, 1)), np.zeros((4, 3)), np.zeros(3)
    )
    assert all(np.array_equal(actual, want) for actual, want in zip(result, expected))


def test_run_uses_runtime_global_defaults(monkeypatch):
    captured = {}

    monkeypatch.setattr(modular_main, "DATA_TYPE", "real")
    monkeypatch.setattr(modular_main, "DOG_NUM", 3)
    monkeypatch.setattr(modular_main, "SOLVE_LEVER", None)
    monkeypatch.setattr(modular_main, "MAKE_PLOTS", False)
    monkeypatch.setattr(modular_main, "SAVE_OUTPUT", False)

    monkeypatch.setattr(
        modular_main,
        "load_real_data",
        lambda dog_num=None: (
            np.zeros((2, 2)),
            np.array([1.0, 2.0, 3.0]),
            np.zeros((2, 4, 3)),
            np.array([0.0, 1.0]),
        ),
    )
    monkeypatch.setattr(
        modular_main,
        "load_esv_table",
        lambda name: (np.zeros(1), np.zeros(1), np.zeros((1, 1))),
    )

    def fake_run_inversion(*args):
        captured["solve_lever"] = args[5]
        return {
            "CDOG_guess": np.array([1.0, 2.0, 3.0]),
            "lever": np.array([0.0, 0.0, 0.0]),
            "offset": 0.0,
            "time_bias": None,
            "esv_bias": None,
            "CDOG_full": None,
            "GPS_full": None,
            "CDOG_clock": None,
            "GPS_clock": None,
        }

    monkeypatch.setattr(modular_main, "run_inversion", fake_run_inversion)

    run()

    assert np.array_equal(captured["solve_lever"], DEFAULT_REAL_LEVER)


def test_run_makes_plots_from_final_diagnostics(monkeypatch):
    plot_calls = {}

    monkeypatch.setattr(modular_main, "DATA_TYPE", "synthetic")
    monkeypatch.setattr(modular_main, "TRAJECTORY", "realistic")
    monkeypatch.setattr(modular_main, "MAKE_PLOTS", True)
    monkeypatch.setattr(modular_main, "PLOT_BLOCK", False)
    monkeypatch.setattr(modular_main, "PLOT_SAVE", True)
    monkeypatch.setattr(modular_main, "PLOT_PATH", "Figs/Test")
    monkeypatch.setattr(modular_main, "PLOT_SEGMENTS", 4)
    monkeypatch.setattr(modular_main, "SAVE_OUTPUT", False)

    monkeypatch.setattr(
        modular_main,
        "load_esv_table",
        lambda name: (np.zeros(1), np.zeros(1), np.zeros((1, 1))),
    )
    monkeypatch.setattr(
        modular_main,
        "generate_synthetic",
        lambda *args: (
            np.zeros((2, 2)),
            np.array([1.0, 2.0, 3.0]),
            np.ones((2, 4, 3)),
            np.array([0.0, 1.0]),
            np.ones((2, 3)),
        ),
    )
    monkeypatch.setattr(modular_main, "_report_quality_synthetic", lambda *args: None)
    monkeypatch.setattr(
        modular_main,
        "run_inversion",
        lambda *args: {
            "CDOG_guess": np.array([1.0, 2.0, 3.0]),
            "lever": np.array([0.0, 0.0, 0.0]),
            "offset": 0.0,
            "time_bias": None,
            "esv_bias": None,
            "CDOG_full": np.array([0.1, 0.2]),
            "GPS_full": np.array([0.1, 0.2]),
            "CDOG_clock": np.array([0.0, 1.0]),
            "GPS_clock": np.array([0.0, 1.0]),
        },
    )
    monkeypatch.setattr(
        modular_main,
        "ECEF_Geodetic",
        lambda coords: (
            np.arange(len(coords), dtype=float),
            np.arange(len(coords), dtype=float) + 10.0,
            np.zeros(len(coords), dtype=float),
        ),
    )

    def fake_time_series_plot(*args, **kwargs):
        plot_calls["time_series"] = kwargs

    def fake_trajectory_plot(*args, **kwargs):
        plot_calls["trajectory"] = kwargs

    monkeypatch.setattr(modular_main, "time_series_plot", fake_time_series_plot)
    monkeypatch.setattr(modular_main, "trajectory_plot", fake_trajectory_plot)

    run()

    assert plot_calls["time_series"]["save"] is True
    assert plot_calls["time_series"]["path"] == "Figs/Test"
    assert plot_calls["time_series"]["segments"] == 4
    assert plot_calls["trajectory"]["save"] is True
    assert plot_calls["trajectory"]["path"] == "Figs/Test"
