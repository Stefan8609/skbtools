# plot_tests: revision figures for Paper I (SANA-D-26-00282)

Drafts of the new and remade figures for the major revision (REVISION_STATUS §7,
`revision/FIGURES_PROMPT.md`). They sit beside `src/plotting/` until promoted; nothing
in `src/plotting/` is modified. One script per figure.

Run everything from the repository root:

```bash
.venv/bin/python plot_tests/make_plot_tests.py
```

Outputs go to `Bermuda_Data/Figs/plot_tests/` (gitignored): `name.pdf` (vector, dense
point layers rasterized) and `name.png` (preview). `checks/` holds a 600 dpi PNG and
greyscale and deuteranopia previews of every figure; `iterations/` keeps the PNG of
each design round.

Every script calls `_figure_common.save`, which **fails** unless the figure is 84 or
174 mm wide and at most 225 mm high, every drawn text is ≥ 8 pt, and no two text
boxes overlap or leave the figure. Numbers printed on figures come from the pure
helpers in `_figure_common.py`, tested in `tests/test_plot_tests_helpers.py`.

## Decisions (approved 2026-11-18)

- Final solution: **MCMC posterior mean** of run `ar1_common30` (common 30-min δc(t)
  windows + per-DOG offset, AR(1), one δt per DOG): `FINAL_SOLUTION`, `FINAL_RUN`.
- δc is called the **ESV correction** (R3 #7): `DC_NAME`.
- FIG-2 (site map) and FX-3 (Fig. 3 trajectory) are one figure.
- Single-DOG 3-D posterior figures are drawn at 120 mm (supplement and response
  letter only); the three-DOG main figure is 174 mm.

## Journal figure guidelines (checked 2026-11-18)

- Satellite Navigation's guideline page now redirects to link.springer.com and needs
  a Springer login; its artwork section could not be read.
- Springer large-format journals: 84 mm (single) or 174 mm (double) wide, at most
  234 mm high. These are the `paper_style` widths.
- SpringerOpen/BMC production: 85 mm or 170 mm wide (one page says 176 mm), at most
  225 mm for figure plus legend. If production rescales 174 mm to 170 mm, 8 pt type
  prints at 7.8 pt. The scripts keep 174 mm and cap heights at 225 mm.
- Check the live journal page before upload.

## Figures

Status: OK, PROVISIONAL (data will be regenerated), BLOCKED (an input does not exist).

| ID | Script | Inputs | Reviewer comments | Status | Caption draft |
|---|---|---|---|---|---|
| FIG-1 | `fig_FIG1_flowchart.py` | `Bermuda_Data/GPS_Data/Bermuda_Testing/Antenna{1..4}-camp.mat`; `ar1_common30` config (gate γ, window) | R4 3, R1 2b, R2 general, R3 M8, R3 M1 | OK | Processing chain from PRIDE-PPPAR positions to the reported position. Steps run in numbered order; each box names its output after →. Orange: the one-way specific clock-offset step, where the circular variance fixes the integer second and the RMSE profile the fraction. The reported position is the MCMC posterior mean (thick box). |
| FIG-2 (+FX-3) | `fig_FIG2_site_map.py` | `Bermuda_Data/gebco_2025_bermuda_{survey,region}_ascii.asc` and `*_tid_ascii.asc`; `GPS_Data/Processed_GPS_Receivers_DOG_1.npz`; antenna files; `ar1_common30` chains | R1 4a, 4d, 4e, 3f; R3 #10, #18, M7(i) | OK (no deployment positions in the repo: gap, see below) | (a) Kept GNSS epochs coloured by time since survey start; (b) rejected epochs; DOGs at the final posterior mean; GEBCO 2025 heights (m, relative to mean sea level), hatched where the cell is multibeam (TID 11); (c) site relative to Bermuda. |
| FIG-3 | `fig_FIG3_clock_offset.py` | `Output/Plot_Data/integer_pick_metrics_DOG{1,3,4}.npz`; `revision/analysis/integer_ambiguity/integer_ambiguity_DOG{1,3,4}.csv`, `_DOG3_mechanism.npz` | R3 #15, R2 3, R3 M1 (absorbs Fig. A6) | OK | See `revision/FIGURE_REPORTS.md` (FIG-3 caption below). |
| FIG-4 | `fig_FIG4_dc_antenna.py` | `ar1_common30/analysis/dc_windows.csv`; antenna files | R1 3c, R2 6, R3 M3 | PROVISIONAL (antenna-4 QC, N3, pending) | Caption in `FIGURE_REPORTS.md`. |
| FIG-5 | `fig_FIG5_model_sensitivity.py` | `MCMC_Chains/<run>/analysis/posterior_summary.csv` for eight runs; `revision/analysis/realistic_synthetic/real_reference.csv` | R3 M2, R4 5, R1 3a | PROVISIONAL for runs on the old ESV table (open markers, †); the final run is OK | Posterior mean ± 1 sd (thick) and 95 % (thin) of E, N, U from the Table 1 position for each model; † old ESV table. Grey band: final ± half the difference between the two half-survey Gauss–Newton solutions (real-data repeatability, not MCMC). Titles: spread of means in cm and in median posterior sd. |
| FIG-6 | `fig_FIG6_realism_synthetic.py` | `revision/analysis/realistic_synthetic/{budget,coverage,real_reference}.csv`, `coverage/*.npz` | R1 2g, R3 M8, M3, M2; R4 4 | OK (true ACF/PSD not in saved data: binned residuals and lag-1 φ used; sources not simulated are hatched, bound or "n.q.") | (a) 15-min mean range residuals, data vs 5–95 % of synthetic surveys; (b, c, d) residual ACF, rms and lag-1 autocorrelation, data vs synthetic; (e) error budget by source; (f, g) \|error\| / formal sd against \|N(0,1)\| with 68/95 % coverage: the formal sd under-covers even with AR(1) inflation (horizontal 13 % / 25 %, vertical 6 % / 31 % at 68 / 95 %; read from the figure). |
| FIG-7 | `fig_FIG7_track_geometry.py` | `revision/analysis/realistic_synthetic/geometry.csv`; processed GPS file | R1 3e, R2 5, R4 2, R3 M7(i) | OK | Track shapes, 95 % E–N ellipses across synthetic surveys, U error, condition number and l_z error, lever z fixed vs free. The real track (rms horizontal 51 / 53 cm fixed / free) is not worse than the uniform grid (63 / 65 cm); removing the north-west part makes the real track worse (78 / 81 cm) and the grid not. |
| FIG-8 | `fig_FIG8_lever_per_dog.py` | `MCMC_Chains/ar1_dog{1,3,4}/` | R4 5, R1 4h | BLOCKED: missing chains; filled by `ar1_dog{1,3,4}` MCMC | – |
| FIG-9 | `fig_FIG9_mcmc_diagnostics.py` | `ar1_common30` chains and `analysis/{diagnostics,linearized_sd,prior_sensitivity}.csv` | R3 M5, R4 2a, 2b | OK | Caption below and in `FIGURE_REPORTS.md`. |
| FX-4 | `fig_FX4_travel_times.py` | `ar1_common30` chains; `GPS_Data/Processed_GPS_Receivers_DOG_1.npz`; `integer_pick_metrics_DOG*.npz` (cross-check) | R4 m3, R1 4e/3f, R3 #10 | OK | Observed one-way travel times (Eq. A21, δt at the posterior mean) and modelled travel times of the final model. They overlap at this scale; the residuals are in FX-5. Gaps in DOG 4 are arrivals dropped by pairing and rejection. Dashed line: vertical travel time. |
| FX-5, FX-6 | `fig_FX5_residuals_time.py`, `fig_FX6_residuals_range.py` | `ar1_common30` chains | R1 2c, 4i, 4j, 4l; R3 #5, M3; R4 m4 | OK | Captions in `FIGURE_REPORTS.md`. FX-6 is coloured by time, not elevation (elevation is a function of slant range at fixed depth; it is shown on the top axis). |
| FX-7 | `fig_FX7_corner.py` | `ar1_common30` chains | R4 m2, 2b, 2c; R1 4k; R3 #11 | OK | Corner plot for DOG 3 relative to the posterior mean: 2-D posterior with 68/95 % regions, 1-D marginals with the prior density, correlation matrix in the upper triangle. $l_x$, $l_y$, $l_z$ are the body frame; $l_z$ − U is the vertical transducer–DOG separation. ρ ≈ ±1 means the combination is well determined along the degenerate direction. U is up-positive, so ρ(ΔU, δt) > 0 corresponds to ρ($l_z$ − U, δt) < 0 (Eq. 6). Printed: ρ(δt, δc offset) = +0.989 and ρ(δt, $l_z$ − U) = −0.988, as R4 2c expects. |
| FX-8 | `fig_FX8_vessel_posteriors.py` | `ar1_common30` chains; `Processed_GPS_Receivers_DOG_1.npz`; `Atlantic_RV_Plot_Data_ar1.npz` | R3 sp. 1, R4 m6 | OK | Caption in `FIGURE_REPORTS.md`. |
| FX-A3 | `fig_FXA3_gnss_rejection.py` | antenna files; `Processed_GPS_Receivers_DOG_1.npz` | R3 M7 | OK (counts differ from the text, see below) | Antenna heights with running median ± 2 MAD; an epoch is rejected if any antenna fails (counts on the figure are recomputed from the raw files and match the processed file). |
| FX-B8/B9 | `fig_FXB8_synthetic_bias.py` | `Output/Plot_Data/synthetic_error_ellipse_global_table_esv_synth_{ref,moderate,strong}.npz` | R4 4, R1 3a | BLOCKED: `Sample_Ellipse` with 1000 realisations per table, free offset, saving a failure mask | – |
| FX-B10 | `fig_FXB10_synthetic_inversion.py` | `modular_synthetic` (seed 20261118) | R1 4n | OK | Caption in `FIGURE_REPORTS.md`. |
| FX-B11 | `fig_FXB11_noise_dependence.py` | `Output/Plot_Data/Combined_Dependence_global_table_esv_synth_moderate.npz` | R3 #12, K8 | BLOCKED: `combined_dependence.py` with ≥ 30 realisations and `solve_g=True` | – |
| 3-D posterior | `fig_mcmc_enu_3d_shadows.py` | `ar1_common30` chains | R4 2b | OK | Caption in `FIGURE_REPORTS.md`. |

## Notes

- **3-D figure encoding:** the cloud is coloured by the 3-D marginal kernel density
  (ln density / peak), not the joint log posterior (speckled, since it covers all 50
  parameters). Palette `magma_r` (truncated at 0.88): darkest core on light panes, warm
  hues separate from the grey shadows, monotone in greyscale; `viridis_r` gave yellow
  speckle outliers and `cividis_r` was too muted. Walls show projected draws as grey
  points. `fig_3d_posterior_three_dogs_encodings` and `_palettes` compare the options.
- **FX-6 colour:** by time since survey start, not elevation (elevation depends only on
  slant range at fixed depth, so the colour would repeat the x axis).
- **Stage roles (correction of the figure brief):** the circular variance fixes the
  integer second and the RMSE profile fixes the fraction (`find_int_offset`,
  `find_subint_offset`, App. A). FIG-1 and FIG-3 say so. This rebuts the premise of R3 M1
  that the circular variance identifies only the fractional part.
- **Bathymetry citation:** GEBCO Compilation Group (2025) GEBCO 2025 Grid,
  doi:10.5285/37c52e96-24ea-67ce-e063-7086abc05f29. Heights are relative to MSL.
- **Missing deployment positions:** the repo holds no DOG deployment (drop) positions,
  so R3 #18 cannot be shown against the estimates.
- **FX-A3 counts:** recomputed from the raw antenna files: 45 669 common epochs, 29 595
  kept, 16 074 rejected (6 by the hard cut-offs). The text has 46 539 and 17 499;
  the difference (870 epochs, 1 425 rejected) is not explained by the code and needs the
  authors' check.
- **Captions for FIG-3, FIG-4, FIG-9 and FX-4 corrections:** see the section below.

### Caption additions (review fixes)

- FIG-3: add that the three DOG curves in (b) overlap; the +0.3 s case in (a) has the
  same 1 − R̄ as the solution by design (the circular variance cannot see fractions,
  which is why stage 2 exists); the k = 0 RMSE in (d) (20.0 / 23.8 / 17.7 cm) differs from
  Table 1 for DOGs 1 and 4 because outliers are rejected (K1).
- FIG-9: the i.i.d. posterior/linearised sd ratios of 2–3 arise because AR(1)
  correlation reduces the effective number of data, which an i.i.d. covariance ignores;
  chains 1–3 are coloured, 4–8 grey.
- FX-4: DOG 4 gaps are dropped arrivals; observed and modelled overlap at this scale,
  residuals are in FX-5.
