"""Names of the ESV tables behind the synthetic experiments.

``SYNTHETIC_ESV_TABLES`` selects the set every synthetic driver uses: ``"new"``
(``build_synthetic_esv_tables.py``: one grid, inversion tables perturbed with
``svp_perturbation.py``) or ``"submitted"`` (the original tables, which do not cover
the DOG depth, so lookups clamp to the table edge).
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
