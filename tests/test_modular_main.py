import argparse
import numpy as np

from Inversion_Workflow.Modular_Main import build_arg_parser, load_real_data


def test_parser_defaults():
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.data_type == "synthetic"
    assert args.dog_num == 3


def test_load_real_data_path(monkeypatch):
    captured_path = {}

    def fake_load(path):
        captured_path["value"] = str(path)
        return {
            "GPS_Coordinates": np.zeros((1, 4, 3)),
            "GPS_data": np.zeros(1),
            "CDOG_data": np.zeros((1, 2)),
            "CDOG_guess": np.zeros(3),
        }

    monkeypatch.setattr(np, "load", fake_load)

    args = argparse.Namespace(dog_num=5)
    data = load_real_data(args)

    assert captured_path["value"].endswith("GPS_Data/Processed_GPS_Receivers_DOG_5.npz")
    assert len(data) == 5
