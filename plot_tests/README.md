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
