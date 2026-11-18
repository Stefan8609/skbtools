"""Names of the ESV tables behind the synthetic experiments (App. B, Figs. B8-B11).

``SYNTHETIC_ESV_TABLES`` selects the set every synthetic driver uses:

- ``"new"``: tables from ``build_synthetic_esv_tables.py``. Generation and inversion
  tables share one grid that covers the synthetic geometry, and the inversion tables
  differ from the generation table only by a physically motivated sound-speed
  perturbation (``svp_perturbation.py``).
- ``"submitted"``: the tables behind the submitted figures. They do not cover the DOG
  depth, so inversion lookups are clamped to the table edge (SYNTHETIC_AUDIT.md, G8).

``moderate`` replaces ``global_table_esv_perturbed`` (B8, B9, B11); ``strong``
replaces ``global_table_esv_realistic_perturbed`` (B10, Geometric_Dilution_Plot).
"""

SYNTHETIC_ESV_TABLES = "new"

ESV_TABLE_SETS = {
    "new": {
        "generate": "global_table_esv_synth_ref",
        "moderate": "global_table_esv_synth_moderate",
        "strong": "global_table_esv_synth_strong",
    },
    "submitted": {
        "generate": "global_table_esv",
        "moderate": "global_table_esv_perturbed",
        "strong": "global_table_esv_realistic_perturbed",
    },
}

ESV_GENERATE = ESV_TABLE_SETS[SYNTHETIC_ESV_TABLES]["generate"]
ESV_MODERATE = ESV_TABLE_SETS[SYNTHETIC_ESV_TABLES]["moderate"]
ESV_STRONG = ESV_TABLE_SETS[SYNTHETIC_ESV_TABLES]["strong"]
