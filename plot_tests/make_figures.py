"""Draw every revision figure in plot_tests/ (one script per figure).

From the repository root:
``PYTHONPATH=src:plot_tests MPLBACKEND=Agg .venv/bin/python plot_tests/make_figures.py``
"""

import importlib

import matplotlib.pyplot as plt

FIGURES = (
    "fig01_survey_geometry",
    "fig02_clock_offset",
    "fig03_residuals",
    "fig05_posterior_3d",
    "fig06_vertical_tradeoff",
    "fig07_esv_correction",
    "fig08_repeatability",
    "figG1_gnss_height",
    "figG2_gnss_baselines",
    "figG3_gnss_rigid_body",
)

if __name__ == "__main__":
    for name in FIGURES:
        importlib.import_module(name).main()
        plt.close("all")
