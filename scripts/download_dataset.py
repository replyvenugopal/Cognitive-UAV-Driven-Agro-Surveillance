#!/usr/bin/env python3
"""Downloads the Kaggle "Crop Health and Environmental Stress Dataset" used
by the paper (Section 3.1 / reference [26]) into data/raw/.

Requires a Kaggle API token. One-time setup:
    1. Create an API token at https://www.kaggle.com/settings -> "Create New Token"
       (downloads kaggle.json).
    2. Place it at ~/.kaggle/kaggle.json (chmod 600) or set the
       KAGGLE_USERNAME / KAGGLE_KEY environment variables.
    3. pip install kaggle

Usage:
    python scripts/download_dataset.py [--output data/raw]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

DATASET_SLUG = "datasetengineer/crop-health-and-environmental-stress-dataset"
OUTPUT_CSV_NAME = "crop_health_and_environmental_stress.csv"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw", help="Directory to place the extracted CSV in.")
    args = parser.parse_args(argv)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print(
            "The 'kaggle' package is not installed. Run `pip install kaggle` and configure "
            "your API token as described in this script's docstring, then re-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # noqa: BLE001
        print(
            f"Kaggle authentication failed ({exc}). Make sure ~/.kaggle/kaggle.json exists "
            "(see this script's docstring) and re-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Downloading '{DATASET_SLUG}' ...")
    api.dataset_download_files(DATASET_SLUG, path=str(out_dir), unzip=False)

    zip_path = out_dir / (DATASET_SLUG.split("/")[-1] + ".zip")
    if not zip_path.exists():
        # Kaggle sometimes names the zip differently; fall back to whatever landed.
        candidates = list(out_dir.glob("*.zip"))
        if not candidates:
            print("Download appears to have failed: no .zip file found.", file=sys.stderr)
            sys.exit(1)
        zip_path = candidates[0]

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)

    csvs = list(out_dir.glob("*.csv"))
    if csvs:
        target = out_dir / OUTPUT_CSV_NAME
        if csvs[0] != target:
            shutil.move(str(csvs[0]), str(target))
        print(f"Dataset ready at {target}")
    else:
        print(f"Extracted files are in {out_dir}, but no top-level CSV was found - check the archive contents.")


if __name__ == "__main__":
    main()
