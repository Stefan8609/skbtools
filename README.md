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
4. **MCMC:** `Inversion_Workflow/MCMC_Functions/MCMC_sampler.py` (one chain per
   process; see "Running on Adroit"). `run_local_chains.py` in the same folder
   runs a few chains in parallel on one machine. `MCMC_diagnostics.py` combines
   the chains of a run (rank-normalized split R-hat, bulk and tail ESS,
   acceptance, trace and autocorrelation plots); `MCMC_posterior_summary.py`
   writes the posterior tables (Table 1 layout, lever arm against its prior,
   prior sensitivity) and the linearized comparison at the MAP. Both write to
   `Output/MCMC_Chains/<run>/analysis/`. `plotting/MCMC_model_comparison.py`
   overlays the vertical, dt and dc posteriors of runs with different noise
   models. The Fig. 7-9 scripts read a pooled run with `pooled_run(<run>)`.
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

## Running on Adroit

The MCMC runs one chain per Slurm array task. `SLURM_ARRAY_TASK_ID` sets each
chain's seed, overdispersed start point and output file, and the environment
variable `MCMC_RUN` picks a run variant from `RUN_VARIANTS` in `MCMC_sampler.py`:
the noise models `iid_fixed`, `iid`, `ar1`, `ar1_drift` and `ar1_white` (AR(1)
plus white noise), and the ESV-split test `ar1_splits{1,8,16}` and
`ar1_white_splits{1,8,16}` (the default is 4 splits per DOG). The white-noise
variants burn in for 40k iterations and the 16-split variants run longer
(`ar1_white_splits16`: 680k iterations, about 6 h per chain on Adroit).
Everything else, including priors,
iteration counts and thinning, is the `CONFIG` block at the top of
`MCMC_sampler.py`. Edit it before submitting, not while jobs are queued: tasks
read the file when they start. Each chain records its configuration and git
commit, and a resumed chain refuses a changed configuration.

**1. Clone** (on the Adroit login node):

```bash
git clone <REPOSITORY URL> skbtools
cd skbtools
git checkout development
mkdir -p slurm/logs
```

**2. Copy the data.** The data folders are not in git. The sampler reads only
these files (paths relative to the repository root):

- `Bermuda_Data/ESV_Tables/global_table_esv_extended.mat`
- `Bermuda_Data/GPS_Data/Processed_GPS_Receivers_DOG_1.npz`
- `Bermuda_Data/CDOG_Data/DOG1-camp.mat`
- `Bermuda_Data/CDOG_Data/DOG3-camp.mat`
- `Bermuda_Data/CDOG_Data/DOG4-camp.mat`

From your laptop's repository root:

```bash
rsync -avR Bermuda_Data/ESV_Tables/global_table_esv_extended.mat Bermuda_Data/GPS_Data/Processed_GPS_Receivers_DOG_1.npz Bermuda_Data/CDOG_Data/DOG1-camp.mat Bermuda_Data/CDOG_Data/DOG3-camp.mat Bermuda_Data/CDOG_Data/DOG4-camp.mat <NETID>@adroit.princeton.edu:skbtools/
```

Each chain stores the SHA-256 of these files in its metadata.

**3. Environment** (Python 3.9-3.11):

```bash
module load anaconda3/<VERSION>
python -m venv .venv
.venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest -q tests/test_mcmc_sampler.py tests/test_find_esv.py
```

Fill in the placeholders in `slurm/mcmc_warmup.sh` and `slurm/mcmc_array.sh`:
account, partition, time, memory (2G is ample) and environment activation.

**4. Submit.** The warm-up job compiles the Numba kernels once. The arrays wait
for it, so their tasks do not all compile at the same time.

```bash
warmup=$(sbatch --parsable slurm/mcmc_warmup.sh)
sbatch --dependency=afterok:$warmup --export=ALL,MCMC_RUN=ar1 slurm/mcmc_array.sh
sbatch --dependency=afterok:$warmup --export=ALL,MCMC_RUN=ar1_drift slurm/mcmc_array.sh
```

Change the array size with `--array=0-15`. A task on a node with a different CPU
type may recompile, which takes under a minute.

**5. Monitor.** Each task logs its acceptance rate, log posterior, sigma and speed
every `log_every` iterations, and the end of every adaptation window:

```bash
squeue -u $USER
tail -f slurm/logs/skb-mcmc-<JOBID>_<TASK>.out
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS
```

Chains are written to `Bermuda_Data/Output/MCMC_Chains/<MCMC_RUN>/chain_<NN>.npz`
every `checkpoint_every` iterations. `n_done` and `n_total` in the file give
progress.

**6. Resume.** If tasks hit the time limit, resubmit the same command, or only
the unfinished indices, e.g. `--array=3,5`. Each task continues from its last
checkpoint and reproduces an uninterrupted run exactly; finished chains exit
immediately. To start a run over, use a new `run_name` or delete its folder.

**7. Copy the chains back:**

```bash
rsync -av <NETID>@adroit.princeton.edu:skbtools/Bermuda_Data/Output/MCMC_Chains/ Bermuda_Data/Output/MCMC_Chains/
```

Numba caches compiled functions next to the source files and does not notice
changes in called functions. After pulling changes to the forward model, clear
the cache with `find src -name "*.nbi" -delete -o -name "*.nbc" -delete`.

## Tests

```bash
pytest -q
ruff check .
```
