# GeigerMethod package

This directory holds research code implementing variations of the
**Geiger method** for underwater acoustic positioning.  Most scripts were
written for experiments around Bermuda and for testing algorithms on
simulated data.

The main components are:

- **Field data scripts** – `geigerMethod_Bermuda.py`,
  `Bermuda_Data_Parse.py`, `Bermuda_Data_Analyze.py`,
  `simulatedAnnealing_Bermuda.py` and related helpers for analysing the
  Bermuda deployment.
- **`GPS_Lever_Arms.py`** – compute lever arms between GPS receivers
  using rigid‑body plane fitting.
- **`Synthetic/`** – tools for generating artificial transponder and GPS
  trajectories.  Includes optimisation routines (Bayesian search,
  simulated annealing) and plotting utilities.
- **`Synthetic/Numba_Functions/`** – numba‑accelerated kernels for
  ray tracing, rigid‑body estimation and the Geiger solver.

The scripts are not packaged as part of the library but serve as
examples and utilities when developing new localisation methods.
