import runpy

import numpy as np
import pytest
from pymap3d import Ellipsoid, ecef2geodetic
from scipy.io import loadmat

import jacksonville_data
from GNSS_Processing import csrs2mat


def test_jacksonville_paths_are_independent_of_working_directory(tmp_path, monkeypatch):
    assert jacksonville_data.GPS_DATA_PATH == (
        jacksonville_data.PROJECT_ROOT / "Jacksonville_Data"
    )
    data_root = tmp_path / "Jacksonville_Data"
    monkeypatch.setattr(jacksonville_data, "GPS_DATA_PATH", data_root)
    monkeypatch.chdir(tmp_path)

    expected = data_root / "GPS_Data/Raw/JAX1/jax11630.pos"
    assert jacksonville_data.gps_data_path("GPS_Data/Raw/JAX1/jax11630.pos") == expected
    segmented = jacksonville_data.gps_data_path(
        "GPS_Data", "Raw", "JAX1", expected.name
    )
    assert segmented == expected
    assert not data_root.exists()

    output = jacksonville_data.gps_output_path("Chains", "chain.npz")
    assert output == data_root / "Output/Chains/chain.npz"
    assert output.parent.is_dir()


def test_script_reads_raw_and_saves_all_receivers(tmp_path, monkeypatch):
    data_root = tmp_path / "Jacksonville_Data"
    monkeypatch.setattr(jacksonville_data, "GPS_DATA_PATH", data_root)
    monkeypatch.chdir(tmp_path)

    for site in ["JAX1", "JAX2", "JAX3", "JAX4"]:
        raw_path = data_root / "GPS_Data/Raw" / site
        raw_path.mkdir(parents=True)
        for doy, day in [(163, 11), (164, 12)]:
            path = raw_path / f"{site.lower()}{doy}0.pos"
            path.write_text(
                "CSRS-PPP header\n"
                f"FIN NAD83 {site.lower()} {doy} 2020-06-{day} 17:43:27.50 "
                "5 2.4 1.222 0.0017 -0.3563 -0.4474 -9.1765 "
                "0.0391 0.0528 0.1101 30 23 27.99603 -81 27 40.35527 -16.4833\n"
            )

    runpy.run_path(csrs2mat.__file__, run_name="__main__")

    outputs = sorted((data_root / "GPS_Data").glob("*.mat"))
    assert [path.name for path in outputs] == [
        f"JAX{unit}-camp_bis.mat" for unit in range(1, 5)
    ]
    keys = {"days", "times", "lat", "lon", "elev", "x", "y", "z", "nsv", "gdop"}
    for output in outputs:
        data = loadmat(output)
        assert {key for key in data if not key.startswith("__")} == keys
        assert all(data[key].shape == (1, 2) for key in keys)
        np.testing.assert_array_equal(data["days"], [[59011, 59012]])
        np.testing.assert_array_equal(data["times"], [[63807.5, 63807.5]])
        np.testing.assert_array_equal(data["nsv"], [[5, 5]])
        np.testing.assert_array_equal(data["gdop"], [[2.4, 2.4]])
        latitude, longitude, elevation = ecef2geodetic(
            data["x"], data["y"], data["z"], ell=Ellipsoid.from_name("grs80")
        )
        np.testing.assert_allclose(latitude, data["lat"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(longitude, data["lon"] - 360, rtol=0, atol=1e-10)
        np.testing.assert_allclose(elevation, data["elev"], rtol=0, atol=1e-8)


def test_gnss_to_mat_creates_output_directory(tmp_path):
    raw_file = tmp_path / "receiver.pos"
    raw_file.write_text(
        "FIN NAD83 jax1 163 2020-06-11 17:43:27.00 "
        "5 2.4 0 0 0 0 0 0 0 0 30 23 27.99603 -81 27 40.35527 -16.4833\n"
    )
    output = tmp_path / "new/GPS_Data/JAX1-camp_bis.mat"

    csrs2mat.gnss_to_mat([raw_file], str(output))

    assert loadmat(output)["x"].shape == (1, 1)


def test_empty_input_does_not_create_output_directory(tmp_path):
    output = tmp_path / "new/GPS_Data/JAX1-camp_bis.mat"

    with pytest.raises(ValueError, match="No FIN position rows were read"):
        csrs2mat.gnss_to_mat([], output)

    assert not output.parent.exists()
