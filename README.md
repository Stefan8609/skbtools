# SKBTOOLS
Seafloor geodetic inversion and synthetic methods as developed by:<br>
Geodetic positioning on the deep sea floor using one-way travel-time ranging to continuously operating reference stations

## Layout

- `src/Inversion_Workflow/`: forward model, alignment, inversion, MCMC and the
  campaign drivers (`Bermuda/`, `PuertoRico/`, `Jacksonville/`, `Synthetic/`).
- `src/acoustics/`: ray tracing, sound-speed profiles and ESV tables.
- `src/geometry/`: coordinate conversions and rigid-body (Procrustes) fitting.
- `src/GNSS_Processing/`: conversion of PRIDE PPP-AR / CSRS output to `.mat`.
- `src/plotting/`: figures.
- `Paper_Data/`: a small sample of the processed Bermuda data (see its README).

Campaign data live in `Bermuda_Data/`, `PuertoRico_Data/` and
`Jacksonville_Data/` at the repository root (not tracked). Paths are built with
`src/bermuda_data.py`, `src/puertorico_data.py` and `src/jacksonville_data.py`.
Scripts are run from `src/` with `src` on the Python path, and each exposes a
short configuration block under `if __name__ == "__main__":`.

## Pipeline (Bermuda)

1. **Initialize data:** `Inversion_Workflow/Bermuda/Initialize_Bermuda_Data.py`
   filters the GNSS receiver positions and writes
   `GPS_Data/Processed_GPS_Receivers_DOG_{n}.npz`.
2. **ESV table:** `acoustics/ESV_table_Bermuda.py` ray traces the CTD profile
   into `ESV_Tables/global_table_esv_extended.mat`.
3. **Alignment and time bias, Gauss-Newton and annealing:**
   `Inversion_Workflow/Bermuda/Bermuda_Data_time_bias.py` (one DOG) and
   `Bermuda_Data_Joint_Annealing.py` (shared lever arm). These call
   `Inversion/Numba_xAline.py` (alignment and offset search),
   `Inversion/Numba_xAline_Geiger_bias.py` (Gauss-Newton) and
   `Inversion/Numba_xAline_Annealing_bias.py` (simulated annealing for the
   lever arm).
4. **MCMC:** `Inversion_Workflow/MCMC_Functions/MCMC_sampler.py`.
5. **Plots:** see the figure map below.

The Puerto Rico workflow follows the same order in `Inversion_Workflow/PuertoRico/`
(`Initialize_PuertoRico_Data.py` → `PuertoRico_Data_time_bias.py` →
`PuertoRico_MCMC_sampler.py` → `PuertoRico_MCMC_best_sample.py`).

## Figures and tables of the paper

| Item | Script |
|---|---|
| Fig. 3 | `plotting/Plot_Modular.py` (`trajectory_plot`) |
| Table 1 (DOGs 1, 3, 4) | `Inversion_Workflow/Bermuda/Bermuda_Data_time_bias.py` |
| Table 1 (joint lever arm) | `Inversion_Workflow/Bermuda/Bermuda_Data_Joint_Annealing.py` |
| Figs. 5, C12, C13 | `Bermuda_Data_time_bias.py` (`time_series_plot`); re-plot with `plotting/Misc/Plot_from_saved_data.py` |
| Figs. 6, C14, C15 | `Bermuda_Data_time_bias.py` (`range_residual`) |
| Fig. 7 | `plotting/MCMC_Correlation.py` and `plotting/MCMC_plots.py` |
| Fig. 8 | `plotting/Atlantic_Explorer/Atlantic_Explorer_Plot.py` |
| Fig. 9 | `Inversion_Workflow/MCMC_Functions/MCMC_best_sample.py` |
| Fig. A2 | `GNSS_Processing/receiver_comparison.py` |
| Fig. A3 | `plotting/Misc/GNSS_Filtering_Plot.py` (data from `plotting/Analysis_Plots/Bermuda_GPS_Plots.py`) |
| Fig. A6 | `plotting/Analysis_Plots/metric_plots.py` (`plot_integer_pick_metrics_dog`) |
| Figs. B8, B9 | `plotting/Analysis_Plots/Sample_Ellipse.py` |
| Fig. B10 | `Inversion_Workflow/Synthetic/Modular_Synthetic.py` |
| Fig. B11 | `plotting/Analysis_Plots/combined_dependence.py` |
| GEBCO depth comparison (Sec. 6.1) | `plotting/Misc/GEBCO_Depth_Comparison.py` |

Figs. A1, A5 and A7 are schematics. No script in this repository currently
produces Fig. 4 or Fig. A4.

## Reproducing the submitted results

The `master` branch holds the code behind the submitted paper; work for the
revision happens on `development`. Table 1 is reproduced by `master` with:

- the full trajectory (no leg mask in `Bermuda_Data_time_bias.py`),
- the ESV table generated before the 2026-10-08 ray-tracing fix
  (`ESV_Tables/global_table_esv_extended_before_ray_fix_20261008.mat`),
- the Table 1 lever arms.

The Fig. 7 chain (`Output/mcmc_chain_1_22_new_MCMC_long.npz`) was produced by
`MCMC_sampler.py` as of commit 8f54e33.

## Tests

```bash
pytest -q
ruff check .
```
