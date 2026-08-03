# Data

This project consumes the Kaggle **"Crop Health and Environmental Stress
Dataset"** (`datasetengineer/crop-health-and-environmental-stress-dataset`),
referenced as [26] in the paper this repo implements.

## Getting the real dataset

```bash
pip install kaggle
# one-time: place your Kaggle API token at ~/.kaggle/kaggle.json (see
# https://www.kaggle.com/settings -> "Create New Token")
python scripts/download_dataset.py --output data/raw
```

This places `data/raw/crop_health_and_environmental_stress.csv`, which
every training script (`configs/default.yaml: data.raw_csv`) picks up
automatically.

## Running without the dataset

If `data/raw/crop_health_and_environmental_stress.csv` is missing and
`data.synthetic_fallback: true` (the default), every training script
transparently falls back to `agrosurveillance.data.synthetic`, which
generates a same-schema synthetic table with non-trivial temporal stress
dynamics. This keeps the full pipeline runnable end-to-end (see
`scripts/run_demo.py`) without Kaggle credentials or network access - but it
is a stand-in for testing/demoing the code, **not** a substitute for
validating against the real dataset.

See `docs/dataset.md` for the exact column schema and the modeling
assumptions made when reconstructing zone-time sequences from the flat
table.
