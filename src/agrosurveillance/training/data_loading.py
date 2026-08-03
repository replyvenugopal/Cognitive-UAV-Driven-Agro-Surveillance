"""Shared data-loading helper used by every training/eval script.

Tries the real Kaggle CSV first (see scripts/download_dataset.py); if it is
not present and `synthetic_fallback` is enabled in the config, transparently
generates a synthetic dataset with the same schema so the pipeline stays
runnable without network access or Kaggle credentials.
"""
from __future__ import annotations

import os
import warnings

import pandas as pd

from ..data.schema import DEFAULT_SCHEMA, derive_health_label
from ..data.synthetic import generate_synthetic_dataset


def load_dataframe(cfg) -> pd.DataFrame:
    schema = DEFAULT_SCHEMA
    raw_csv = cfg.data.raw_csv
    if os.path.exists(raw_csv):
        df = pd.read_csv(raw_csv)
    elif cfg.data.get("synthetic_fallback", True):
        warnings.warn(
            f"'{raw_csv}' not found. Falling back to a synthetic dataset with the same "
            "schema (see docs/dataset.md). Run scripts/download_dataset.py to fetch the "
            "real Kaggle dataset and reproduce paper-comparable results."
        )
        df = generate_synthetic_dataset(n_fields=10, rows_per_field=300, seed=cfg.get("seed", 42))
    else:
        raise FileNotFoundError(f"Raw dataset '{raw_csv}' not found and synthetic_fallback is disabled.")

    df[schema.health_label_col] = derive_health_label(df[schema.stress_indicator_col])
    return df
