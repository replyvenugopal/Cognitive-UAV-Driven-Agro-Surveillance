"""Synthetic data generator matching the Kaggle dataset schema.

Used as an automatic fallback (see training/*.py) so the entire pipeline -
data -> GeoSpatio-TRiNet -> SAC irrigation control -> evaluation - can be
run end-to-end without Kaggle credentials or a network connection. This is
NOT a substitute for validating against the real dataset; it exists purely
so the codebase is testable/demoable out of the box.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import ColumnSchema, DEFAULT_SCHEMA


def generate_synthetic_dataset(
    n_fields: int = 6,
    rows_per_field: int = 200,
    seed: int = 42,
    schema: ColumnSchema = DEFAULT_SCHEMA,
) -> pd.DataFrame:
    """Generate a plausible synthetic table with the same columns/ranges as
    the real Kaggle "Crop Health and Environmental Stress Dataset".

    Stress dynamics are simulated with a simple autoregressive process per
    field so that later pipeline stages (event detection, diffusion,
    vulnerability trajectories) have non-trivial temporal structure to work
    with, rather than i.i.d. noise.
    """
    rng = np.random.RandomState(seed)
    rows = []

    for field_id in range(n_fields):
        # Latent stress process: mean-reverting with occasional shocks.
        stress = np.zeros(rows_per_field)
        stress[0] = rng.uniform(10, 30)
        for t in range(1, rows_per_field):
            shock = rng.normal(0, 3)
            if rng.rand() < 0.05:
                shock += rng.uniform(15, 35)  # sudden stress event
            stress[t] = np.clip(0.9 * stress[t - 1] + shock, 0, 100)

        soil_moisture = np.clip(60 - 0.4 * stress + rng.normal(0, 4, rows_per_field), 5, 95)
        temperature = np.clip(25 + 0.15 * stress + rng.normal(0, 2, rows_per_field), 5, 48)
        humidity = np.clip(55 - 0.2 * stress + rng.normal(0, 5, rows_per_field), 10, 95)
        rainfall = np.clip(rng.exponential(2.0, rows_per_field) - 0.02 * stress, 0, None)
        wind_speed = np.clip(rng.normal(8, 3, rows_per_field), 0, None)

        ndvi = np.clip(0.85 - 0.006 * stress + rng.normal(0, 0.03, rows_per_field), 0, 1)
        savi = np.clip(0.75 - 0.005 * stress + rng.normal(0, 0.03, rows_per_field), 0, 1)
        canopy_coverage = np.clip(80 - 0.4 * stress + rng.normal(0, 4, rows_per_field), 0, 100)
        chlorophyll = np.clip(45 - 0.25 * stress + rng.normal(0, 3, rows_per_field), 0, 60)
        lai = np.clip(4.5 - 0.02 * stress + rng.normal(0, 0.3, rows_per_field), 0, 6)
        elevation = np.full(rows_per_field, rng.uniform(50, 400))

        soil_ph = np.clip(rng.normal(6.5, 0.4, rows_per_field), 4.5, 8.5)
        organic_matter = np.clip(rng.normal(3.0, 0.6, rows_per_field), 0.5, 6.0)

        pest_hotspots = (rng.rand(rows_per_field) < (0.05 + 0.003 * stress)).astype(int)
        weed_coverage = np.clip(10 + 0.15 * stress + rng.normal(0, 5, rows_per_field), 0, 100)
        pest_damage = np.clip(0.3 * stress + rng.normal(0, 5, rows_per_field), 0, 100).astype(int)

        expected_yield = np.clip(9.0 - 0.05 * stress + rng.normal(0, 0.4, rows_per_field), 0, None)

        for t in range(rows_per_field):
            rows.append(
                {
                    "Field_Boundaries": field_id,
                    "Elevation_Data": elevation[t],
                    "Canopy_Coverage": canopy_coverage[t],
                    "NDVI": ndvi[t],
                    "SAVI": savi[t],
                    "Chlorophyll_Content": chlorophyll[t],
                    "Leaf_Area_Index": lai[t],
                    "Crop_Stress_Indicator": stress[t],
                    "Temperature": temperature[t],
                    "Humidity": humidity[t],
                    "Rainfall": rainfall[t],
                    "Wind_Speed": wind_speed[t],
                    "Soil_Moisture": soil_moisture[t],
                    "Soil_pH": soil_ph[t],
                    "Organic_Matter": organic_matter[t],
                    "Pest_Hotspots": pest_hotspots[t],
                    "Weed_Coverage": weed_coverage[t],
                    "Pest_Damage": pest_damage[t],
                    "Crop_Growth_Stage": rng.choice(["seedling", "vegetative", "flowering", "maturity"]),
                    "Crop_Type": rng.choice(["wheat", "maize", "cotton", "rice"]),
                    "Expected_Yield": expected_yield[t],
                }
            )

    df = pd.DataFrame(rows)
    return df
