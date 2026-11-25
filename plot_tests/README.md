# plot_tests: candidate paper figures

Restarted 2026-11-21. The earlier design is in git history at commit
833ff90; its outputs are in `Bermuda_Data/Figs/plot_tests/previous_design_2026-11-20/`.

Each script draws one figure and answers one scientific question, stated with its
message at the top of the script. Numbers that belong in a caption are printed.

```bash
PYTHONPATH=src:plot_tests MPLBACKEND=Agg .venv/bin/python plot_tests/make_figures.py
```

Outputs (`name.pdf`, `name.png`) go to `Bermuda_Data/Figs/plot_tests/` (gitignored).

## Journal specification (Satellite Navigation submission guidelines, read 2026-11-21)

- Width 85 mm (half page) or 170 mm (full page); at most 225 mm high with the legend.
- About 300 dpi at final size, lines wider than 0.25 pt, fonts embedded.
- Keys inside the graphic; figure titles and explanations in the caption.
- Multi-panel figures are one file; panels are lettered a, b, c.

## Style (`_style.py`)

Revised after the comments of 2026-11-21 (the first set was "too minimalist").

- Arial, 8 pt, black for every text, including DOG names (never coloured).
- Axis labels read "words symbol (units)" with the paper's notation
  in mathtext: residuals ε, their sd σ_ε (c σ_ε in cm), timing bias δt, ESV bias δc,
  circular variance v°, RMSE R, (E, N, U), (ϕ, λ).
- Boxed axes, inward ticks on all sides, light grid, framed legends kept off the data.
- One colour and marker per DOG only where DOGs share a panel; observations in the
  blue of `Plot_Modular`.
- Dense scatters may be coloured by the number of nearby observations
  (`st.density_scatter`), as in the Puerto Rico elevation plots.
- `save` reports wrong sizes, text under 8 pt and overlapping text.

## Figures (iteration 7, awaiting comments)

| Script | Question | Size (mm) | Standing |
|---|---|---|---|
| `fig01_survey_geometry.py` | Where are the DOGs and how did the vessel sail over them? | 120 × 119 (between the journal widths); track coloured by time | liked; 17-h track |
| `fig02_clock_offset.py` | Do the data alone fix each DOG's timing bias, including the integer second? | 170 × 125 | liked |
| `fig03_residuals.py` | How large are the residuals of the final model, are they random in time, do the DOGs share them? (density-coloured) | 170 × 125 | liked |
| `fig05_posterior_3d.py` | How tightly are a DOG position and the lever arm constrained, and which parameters trade off? (2 × 2: DOG 3 ENU; lever arm, equal axes; δc, δt, l_z − U; east of the three DOGs; sd and correlations above each panel) | 170 × 160 | 2 × 2 layout awaiting comments |
| `fig06_vertical_tradeoff.py` | Why is the height weak and the horizontal position not? | 170 × 72 | supplementary at most |
| `fig07_esv_correction.py` | Does the ESV bias change during the survey, and does the model of it matter? | 170 × 120 | liked |
| `fig08_repeatability.py` | Do two parts of the survey give the same positions? | 170 × 76 | supplementary at most |
| `figG1_gnss_height.py` | Is the GNSS solution continuous, and how good was the satellite geometry? (density-coloured) | 170 × 105 | liked |
| `figG2_gnss_baselines.py` | How precise are the kinematic positions? (six inter-antenna distances, density-coloured) | 170 × 150 | liked |

Not yet redrawn: realistic-synthetic error budget and coverage, track geometry, MCMC
diagnostics, per-DOG lever arms, residuals against range, travel times, the
processing flowchart and the synthetic-test figures.

**Data state (2026-11-24): survey window extended from 14 h to 17 h.**
`GNSS_end = 42` in `Initialize_Bermuda_Data.py` adds the third diagonal leg
(59 992 GNSS epochs, was 49 552). Regenerated on the 17-h data: the processed GNSS
files, the Gauss-Newton solutions and the integer-second analysis, and with them
fig01, fig02, figG1 and figG2. The 17-h `ar1_common30` chains are back
(2026-11-24), and fig01, fig03, fig05 and fig06 are drawn from them. fig07 and
fig08 need the other runs (`ar1_splits1`, `ar1`, `ar1_white_common30`,
`ar1_common30_drift`; halves and legs), which are not on disk yet. The 14-h inputs are kept in
`Bermuda_Data/GPS_Data/window_14h_2026-11-24/`. The time axis of every figure is
`st.TIME_LIMITS_H` (0-17 h).

`_figure_common.py` keeps the data loaders and the tested numerical helpers.
