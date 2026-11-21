# plot_tests: revision figures for Paper I (SANA-D-26-00282)

Drafts of the new and remade figures for the major revision (`revision/REVISION_REFERENCE.md` §6). They sit beside `src/plotting/` until promoted; nothing
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

## Design rules (redesign of 2026-11-20)

- One message per figure, written as `Message:` at the top of each script.
- Context is grey (`fc.CONTEXT`, `fc.RAW_POINTS`); one accent colour (`fc.ACCENT`)
  marks the one thing to find (the clock-offset step, the chosen offset, the
  antenna-4 steps, the epochs an antenna rejected, the transducer). The DOG colours
  of `paper_style.DOG_STYLE` are the same everywhere and are used only where DOGs
  share a panel or the brief asks for them.
- No paragraphs, tables or titles on figures. Numbers that were printed on a figure
  are printed to the terminal by its script, for the caption or a paper table.
- Left and bottom spines only (`fc.despine`), few ticks, time axis 0–14 h.

## Figures

Status: OK, PROVISIONAL (data will be regenerated), BLOCKED (an input does not exist).
Caption drafts are in `revision/FIGURE_REPORTS.md` ("Redesign 2026-11-20"); the last
column is the figure's one-sentence message.

| ID | Script | Inputs | Reviewer comments | Status | Message |
|---|---|---|---|---|---|
| FIG-1 | `fig_FIG1_flowchart.py` | `Bermuda_Data/GPS_Data/Bermuda_Testing/Antenna{1..4}-camp.mat`; `ar1_common30` config (gate γ) | R4 3, R1 2b, R2 general, R3 M8, R3 M1 | OK | Eight sequential steps lead from the GNSS positions to the reported DOG position; only the clock-offset step (accent outline) is specific to one-way ranging. |
| FIG-2 (+FX-3) | `fig_FIG2_site_map.py` | `Bermuda_Data/gebco_2025_bermuda_{survey,region}_ascii.asc` (and `*_tid_ascii.asc`, printed only); `GPS_Data/Processed_GPS_Receivers_DOG_1.npz`; `ar1_common30` chains | R1 4a, 4d, 4e, 3f; R3 #10, #18 | OK (84 mm; no deployment positions in the repo: gap, see below) | The vessel crossed above the three DOGs on an hourglass track over 14 h on a nearly flat, 5.2–5.3 km deep seafloor west of Bermuda. |
| FIG-3 | `fig_FIG3_clock_offset.py` | `Output/Plot_Data/integer_pick_metrics_DOG{1,3,4}.npz`; `revision/analysis/integer_ambiguity/integer_ambiguity_DOG{1,3,4}.csv`, `_DOG{1,3,4}_mechanism.npz` | R3 #15, R2 3, R3 M1 (absorbs Fig. A6) | OK (rebuilt for `align_offset`; no drift panel: the final model has no clock drift) | The clock offset is found in two steps, the integer second from the circular variance and the fraction from the RMSE; the integer is unambiguous because a wrong second leaves a residual proportional to the line-of-sight speed. |
| FIG-4 | `fig_FIG4_dc_antenna.py` | `ar1_common30/analysis/dc_windows.csv`, `posterior_summary.csv`; the run's GNSS file | R1 3c, R2 6, R3 M3 | PROVISIONAL (antenna-4 QC, N3, pending) | The ESV correction changes level where the height of GNSS antenna 4 steps, so it absorbs vessel-side GNSS error as well as sound speed. |
| FIG-5 | `fig_FIG5_model_sensitivity.py` | `MCMC_Chains/<run>/analysis/posterior_summary.csv` for eight runs; `revision/analysis/realistic_synthetic/real_reference.csv` | R3 M2, R4 5, R1 3a | OK (a run on an old ESV table would be drawn with an open marker and †; none at present) | The DOG positions move between ESV-correction and noise models by many times the posterior sd of any one model, most of all in U. |
| FIG-6 | `fig_FIG6_realism_synthetic.py` → `fig_FIG6_realism_synthetic` | `revision/analysis/realistic_synthetic/{budget,coverage}.csv` | R1 2g, R3 M8, M2; R4 4 | OK (sources not simulated: hatched bound or "not quantified") | Under realistic errors the time-varying ESV correction dominates the position error, and the formal uncertainty covers the true error far less often than its nominal level. |
| FIG-6S (supplement) | same script → `fig_FIG6S_residual_realism` | `coverage/*.npz`, `real_reference.{csv,npz}`; `cache/FIG6_residual_acf.npz` | R3 M3, R4 4 | OK (true ACF/PSD not in the saved data: binned residuals and regenerated series) | The realistic synthetics reproduce the size, time structure and autocorrelation of the real residuals. |
| FIG-7 | `fig_FIG7_track_geometry.py` | `revision/analysis/realistic_synthetic/geometry.csv`; processed GPS file | R1 3e, R2 5, R4 2, R3 M7(i) | OK | The real Bermuda track locates the DOGs about as well as a uniform grid; estimating the vertical lever arm widens the vertical error on the Bermuda track but hardly changes the horizontal error. |
| FIG-8 | `fig_FIG8_lever_per_dog.py` | `MCMC_Chains/ar1_dog{1,3,4}/` | R4 5, R1 4h | BLOCKED: missing chains; filled by `ar1_dog{1,3,4}` MCMC (not edited in the redesign) | – |
| FIG-9 | `fig_FIG9_mcmc_diagnostics.py` | `ar1_common30` chains and `analysis/{diagnostics,linearized_sd,prior_sensitivity}.csv` | R3 M5, R4 2a, 2b | OK | The eight chains mix and agree after burn-in, and the posterior widths match the linearised widths once the AR(1) weighting is used. |
| FX-4 | `fig_FX4_travel_times.py` | `ar1_common30` chains; `GPS_Data/Processed_GPS_Receivers_DOG_1.npz`; `integer_pick_metrics_DOG*.npz` (cross-check) | R4 m3, R1 4e/3f, R3 #10 | OK (light touch) | The modelled travel times follow the observed ones through every pass, and none is shorter than the vertical travel time. |
| FX-5 | `fig_FX5_residuals_time.py` | `ar1_common30` chains | R1 2c, 4i, 4l; R3 #5, M3 | OK (light touch) | The residuals of the final model stay within a few tens of centimetres of zero all survey long; the largest excursions are next to antenna-4 height steps. |
| FX-6 | `fig_FX6_residuals_range.py` | `ar1_common30` chains | R1 4j; R4 m4; R2 7 | OK | The residuals show no trend with slant range or elevation angle; their median stays within about 10 cm at all ranges. |
| FX-7 | `fig_FX7_corner.py` | `ar1_common30` chains | R4 m2, 2b, 2c; R1 4k; R3 #11 | OK | The data, not the priors, determine every parameter; the only strong correlations tie U, the clock offset and the ESV-correction offset together. |
| FX-8 | `fig_FX8_vessel_posteriors.py` | `ar1_common30` chains; `Processed_GPS_Receivers_DOG_1.npz`; `Atlantic_RV_Plot_Data_ar1.npz` | R3 sp. 1, R4 m6 | OK | The transducer position in the vessel frame is known far better than the GNSS antennas agree with a rigid body; the DOG positions are an order of magnitude better determined horizontally than vertically. |
| FX-A3 | `fig_FXA3_gnss_rejection.py` | antenna files; `Processed_GPS_Receivers_DOG_1.npz` | R3 M7 | OK (counts differ from the text, see below) | The height filter rejects an epoch when any antenna leaves its running median ± 2 MAD band, which removes about a third of the epochs, spread over the survey. |
| FX-B8/B9 | `fig_FXB8_synthetic_bias.py` | `Output/Plot_Data/synthetic_error_ellipse_global_table_esv_synth_{ref,moderate,strong}.npz` | R4 4, R1 3a | BLOCKED: `Sample_Ellipse` with 1000 realisations per table, free offset, saving a failure mask (not edited) | – |
| FX-B10 | `fig_FXB10_synthetic_inversion.py` | `modular_synthetic` (seed 20261118) | R1 4n | OK (light touch) | One synthetic inversion on the real track recovers the travel times to a few centimetres of range, with residuals close to normal. |
| FX-B11 | `fig_FXB11_noise_dependence.py` | `Output/Plot_Data/Combined_Dependence_global_table_esv_synth_moderate.npz` | R3 #12, K8 | BLOCKED: `combined_dependence.py` with ≥ 30 realisations and `solve_g=True` (not edited) | – |
| 3-D posterior | `fig_mcmc_enu_3d_shadows.py` | `ar1_common30` chains | R4 2b | OK (minor polish only) | Each DOG position is a compact, nearly uncorrelated cloud a few centimetres wide horizontally and 10–15 cm vertically. |
| Workflow | `fig_WORKFLOW_repo.py` | none (names of modules in `src/`, checked to exist) | R1 3h, R4 3 (reproducibility; response letter and supplement) | OK (new) | The repository takes PRIDE-PPPAR antenna positions and DOG arrival times through eight named steps to the DOG positions; the synthetic and model-sensitivity tests feed the same inversion and sampler. |

Workflow caption draft: "Inputs, processing steps, validation and outcomes of the
code repository. Each box names the module or script that does the work (paths
relative to `src/`; the sub-packages `Bermuda`, `Forward_Model`, `Inversion`,
`MCMC_Functions` and `Synthetic` belong to `Inversion_Workflow/`). White boxes are
inputs; the dashed box is the GNSS processing with PRIDE-PPPAR, which is done
outside the repository. Grey boxes are the processing steps in order. Blue boxes
are the validation: the integer-second test of the clock alignment, the synthetic
generators and the ensemble tests that invert synthetic surveys with the same
Gauss–Newton code, and the MCMC model variants. Orange boxes are the outcomes."

## Notes

- **Material that left the figures** (and must now be in a caption, a table or the
  supplement) is listed per figure in `revision/FIGURE_REPORTS.md`.
- **3-D figure encoding:** the cloud is coloured by the 3-D marginal kernel density
  (ln density / peak), not the joint log posterior (speckled, since it covers all 50
  parameters). Palette `magma`: bright = dense; walls show projected draws coloured by
  2-D density (`GnBu`). `fig_3d_posterior_encodings` and `_palettes` compare the
  options (design records, not paper figures). The position prior (σ = 5 m per axis)
  is stated in the caption, not on the figure.
- **FX-6 colour:** the points are grey; the time colourbar is gone. Single passes at
  particular times show as clumps away from the median (FX-5 has the time axis).
- **Stage roles:** the circular variance fixes the integer second and the RMSE
  profile fixes the fraction (`align_offset`: `find_int_offset`, then
  `find_subint_offset`, App. A). FIG-1 and FIG-3 say so. This rebuts the premise of
  R3 M1 that the circular variance identifies only the fractional part. When the
  offset is unknown, `gauss_newton_offset_search` also refits at the neighbouring
  seconds and keeps the lowest RMS (re-fit RMSE ratio ≥ 5 for every wrong second).
- **FIG-3 inputs:** (a), (b) use the current ESV table and Gauss–Newton lever; (c)
  uses the state of the submitted paper (table before the ray fix, Table 1 lever).
- **Bathymetry citation:** GEBCO Compilation Group (2025) GEBCO 2025 Grid,
  doi:10.5285/37c52e96-24ea-67ce-e063-7086abc05f29. Heights are relative to MSL. The
  GEBCO export ends at about 31.53° N, so the northernmost leg has no bathymetry.
- **Missing deployment positions:** the repo holds no DOG deployment (drop) positions,
  so R3 #18 cannot be shown against the estimates.
- **FX-A3 counts:** recomputed from the raw antenna files: 45 669 common epochs, 29 595
  kept, 16 074 rejected (6 by the hard cut-offs). The text has 46 539 and 17 499;
  the difference (870 epochs, 1 425 rejected) is not explained by the code and needs the
  authors' check.
