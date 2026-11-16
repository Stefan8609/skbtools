import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from plotting.Analysis_Plots import PuertoRico_GPS_Plots as gps_plots


def test_receiver_elevations_share_count_density_scale(monkeypatch):
    GPS_data = np.arange(6, dtype=float) * 60.0
    GPS_Coordinates = np.zeros((6, 4, 3))
    elevation = np.array([-56.0, -56.0, -55.9, -56.1, -56.0, -55.8])
    elevations = np.tile(elevation[:, None], (1, 4))
    nsv = np.full((6, 4), 10.0)
    gdop = np.ones((6, 4))
    used_receivers = np.ones((6, 4), dtype=bool)
    used_receivers[0, 0] = False

    monkeypatch.setattr(
        gps_plots,
        "load_all_receivers",
        lambda *_: (GPS_data, GPS_Coordinates, elevations, nsv, gdop),
    )
    monkeypatch.setattr(
        gps_plots,
        "find_valid_receivers",
        lambda *_: used_receivers,
    )
    monkeypatch.setattr(plt, "show", lambda: None)

    figure = gps_plots.filtered_elevation_plots(mask=False)
    receiver_axes = figure.axes[:4]
    density_plots = [axis.collections[1] for axis in receiver_axes]

    assert all(type(plot.norm) is colors.Normalize for plot in density_plots)
    assert all(plot.norm is density_plots[0].norm for plot in density_plots)
    assert [len(plot.get_offsets()) for plot in density_plots] == [5, 6, 6, 6]
    assert figure.axes[-1].get_ylabel() == "Nearby retained observations"
    plt.close(figure)


def test_barycenter_elevation_uses_count_density(monkeypatch):
    GPS_data = np.arange(6, dtype=float) * 60.0
    GPS_Coordinates = np.zeros((6, 4, 3))
    elevation = np.array([-56.0, -56.0, -55.9, -56.1, -56.0, -55.8])

    monkeypatch.setattr(
        gps_plots,
        "process_gnss",
        lambda *_: (GPS_Coordinates, GPS_data, np.ones((6, 4), dtype=bool)),
    )
    monkeypatch.setattr(
        gps_plots,
        "ECEF_Geodetic",
        lambda _: (np.zeros(6), np.zeros(6), elevation),
    )
    monkeypatch.setattr(plt, "show", lambda: None)

    figure = gps_plots.barycenter_elevation_plot(mask=False)
    density_plot = figure.axes[0].collections[0]

    assert type(density_plot.norm) is colors.Normalize
    assert len(density_plot.get_offsets()) == len(GPS_data)
    assert figure.axes[-1].get_ylabel() == "Nearby observations"
    plt.close(figure)
