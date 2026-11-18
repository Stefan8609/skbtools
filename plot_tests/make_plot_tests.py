"""Run every revision figure script in plot_tests/ and print what was made.

From the repository root: ``.venv/bin/python plot_tests/make_plot_tests.py``.
Each script runs in its own process with PYTHONPATH=src and the Agg backend. A
script that exits with ``_figure_common.SKIP_EXIT_CODE`` is reported as SKIPPED
(an input does not exist yet); any other non-zero exit is FAILED.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SKIP_EXIT_CODE = 3  # same value as _figure_common.SKIP_EXIT_CODE
# Build order of REVISION_STATUS §7 / FIGURES_PROMPT; scripts not listed run last.
ORDER = [
    "fig_FX7_corner.py",
    "fig_FIG3_clock_offset.py",
    "fig_FX5_residuals_time.py",
    "fig_FIG1_flowchart.py",
    "fig_FIG2_site_map.py",
    "fig_FIG5_model_sensitivity.py",
    "fig_FIG6_realism_synthetic.py",
    "fig_FIG7_track_geometry.py",
    "fig_FX6_residuals_range.py",
    "fig_FX4_travel_times.py",
    "fig_FX8_vessel_posteriors.py",
    "fig_FIG4_dc_antenna.py",
    "fig_mcmc_enu_3d_shadows.py",
    "fig_FIG8_lever_per_dog.py",
    "fig_FIG9_mcmc_diagnostics.py",
    "fig_FXB8_synthetic_bias.py",
    "fig_FXB10_synthetic_inversion.py",
    "fig_FXB11_noise_dependence.py",
    "fig_FXA3_gnss_rejection.py",
]


def scripts():
    present = sorted(path.name for path in HERE.glob("fig_*.py"))
    return [name for name in ORDER if name in present] + [
        name for name in present if name not in ORDER
    ]


def main():
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "MPLBACKEND": "Agg"}
    rows = []
    for name in scripts():
        start = time.time()
        result = subprocess.run(
            [sys.executable, str(HERE / name)],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
        )
        seconds = time.time() - start
        if result.returncode == 0:
            status, note = "OK", ""
        elif result.returncode == SKIP_EXIT_CODE:
            status = "SKIPPED"
            note = result.stdout.strip().splitlines()[-1]
        else:
            status = "FAILED"
            note = (result.stderr.strip() or result.stdout.strip()).splitlines()[-1]
        rows.append((name, status, f"{seconds:.0f} s", note))
        print(f"{name:40s} {status:8s} {seconds:5.0f} s  {note}", flush=True)

    print("\nSummary")
    for status in ("OK", "SKIPPED", "FAILED"):
        names = [row[0] for row in rows if row[1] == status]
        print(f"  {status:8s} {len(names):2d}  {' '.join(names)}")
    return 1 if any(row[1] == "FAILED" for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
