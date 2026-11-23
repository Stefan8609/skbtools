# plot_tests: revision figures for Paper I (SANA-D-26-00282)

Restarted 2026-11-21. The earlier design (FIG-1 … FX-B11) is in git history at commit
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

Revised after the reviews of 2026-11-21 (the first set was "too minimalist").

- Arial, 8 pt, black for every text, including DOG names (never coloured).
- Axis labels read "words symbol (units)" with the symbols of Table A1 of the paper
  in mathtext: residuals ε, their sd σ_ε (c σ_ε in cm), timing bias δt, ESV bias δc,
  circular variance v°, RMSE R, (E, N, U), (ϕ, λ).
- Boxed axes, inward ticks on all sides, light grid, framed legends kept off the data.
- One colour and marker per DOG only where DOGs share a panel; observations in the
  blue of `Plot_Modular`.
- `save` reports wrong sizes, text under 8 pt and overlapping text.

## Figures (iteration 4, awaiting review)

| Script | Question | Size (mm) | Standing |
|---|---|---|---|
| `fig01_survey_geometry.py` | Where are the DOGs and how did the vessel sail over them? | 85 × 90 | liked; hour marks added |
| `fig02_clock_offset.py` | Do the data alone fix each DOG's timing bias, including the integer second? | 170 × 125 | liked; panel d redrawn |
| `fig03_residuals.py` | How large are the residuals, are they random in time, do the DOGs share them? | 170 × 125 | liked |
| `fig05_posterior_3d.py` | How tightly is each DOG position constrained, and in which direction is it weakest? | 170 × 80 | liked |
| `fig06_vertical_tradeoff.py` | Why is the height weak and the horizontal position not? | 170 × 72 | supplementary at most |
| `fig07_esv_correction.py` | Does the ESV bias change during the survey, and does the model of it matter? | 170 × 120 | liked |
| `fig08_repeatability.py` | Do two parts of the survey give the same positions? | 170 × 76 | supplementary at most |
| `figG1_gnss_height.py` | Is the GNSS solution continuous, and does it resolve the vessel's motion? | 170 × 105 | candidate |
| `figG2_gnss_baselines.py` | How precise are the kinematic positions? (six inter-antenna distances) | 170 × 150 | candidate |
| `figG3_gnss_rigid_body.py` | How well do the four antennas fit a rigid vessel, and where is the transducer? | 170 × 82 | candidate |

Not yet redrawn: realistic-synthetic error budget and coverage, track geometry, MCMC
diagnostics, per-DOG lever arms, residuals against range, travel times, the
processing flowchart and the appendix B synthetic figures.

**Data state (2026-11-22):** GNSS reprocessed (commit 6e97be0) and every MCMC chain
on disk rerun on it (2026-11-21 20:39), so all figures use one dataset. fig03 shows
the Gauss-Newton residuals (Table 1 quantity), not the MCMC residuals.

`_figure_common.py` keeps the data loaders and the tested numerical helpers.
