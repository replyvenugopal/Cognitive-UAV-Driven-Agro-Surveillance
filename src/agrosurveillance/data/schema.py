"""Column schema for the Kaggle "Crop Health and Environmental Stress Dataset".

The public dataset (kaggle.com/datasets/datasetengineer/crop-health-and-
environmental-stress-dataset) is a flat table of ~212k rows combining
UAV-derived vegetation indices, soil measurements, weather observations and
pest indicators. It does not ship explicit zone-id / timestamp columns, so
this module documents exactly which raw columns feed which part of the
paper's formulation (Eq. 2: X^RGB, X^MS, X^TH, S, E) and how missing pieces
are approximated. See docs/dataset.md for the full rationale.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ColumnSchema:
    # Used as a proxy spatial-zone grouping key (Eq. 17 clusters within this).
    zone_group_col: str = "Field_Boundaries"

    # UAV / remote-sensing derived features -> stand in for U_RGB, U_MS, U_TH
    # (the public dataset ships vegetation indices rather than raw imagery,
    # which is consistent with most operational UAV pipelines that deliver
    # orthomosaic-derived indices, not raw frames).
    uav_rgb_proxy_cols: tuple = ("Canopy_Coverage", "NDVI")
    uav_multispectral_proxy_cols: tuple = ("SAVI", "Leaf_Area_Index")
    uav_thermal_proxy_cols: tuple = ("Chlorophyll_Content",)
    elevation_col: str = "Elevation_Data"

    # Soil variables -> S_z,t
    soil_cols: tuple = ("Soil_Moisture", "Soil_pH", "Organic_Matter")

    # Environmental / meteorological variables -> E^t
    environment_cols: tuple = ("Temperature", "Humidity", "Rainfall", "Wind_Speed")

    # Pest / crop condition covariates (not modeled explicitly in the paper's
    # equations but retained as auxiliary context features, and used to
    # motivate the "future work: pest/disease stressors" limitation noted in
    # the discussion section of the manuscript).
    pest_cols: tuple = ("Pest_Hotspots", "Weed_Coverage", "Pest_Damage")

    categorical_cols: tuple = ("Field_Boundaries", "Crop_Growth_Stage", "Crop_Type")

    # Target / label columns
    stress_indicator_col: str = "Crop_Stress_Indicator"   # 0 (none) - 100 (extreme)
    expected_yield_col: str = "Expected_Yield"

    # Derived at load time: 1 = healthy, 0 = unhealthy (Figure 5 in the paper)
    health_label_col: str = "health_label"

    def all_numeric_feature_cols(self) -> tuple:
        return (
            self.uav_rgb_proxy_cols
            + self.uav_multispectral_proxy_cols
            + self.uav_thermal_proxy_cols
            + (self.elevation_col,)
            + self.soil_cols
            + self.environment_cols
            + self.pest_cols
        )


DEFAULT_SCHEMA = ColumnSchema()


def derive_health_label(stress_indicator, healthy_threshold: int = 40):
    """Binarize the 0-100 Crop_Stress_Indicator into a health label.

    Matches the paper's framing of `Y_z^t` as stress labels (Eq. 1) and the
    healthy/unhealthy class distribution discussed around Figure 5. Rows with
    stress_indicator < healthy_threshold are treated as healthy (label = 1).
    """
    import numpy as np

    stress_indicator = np.asarray(stress_indicator, dtype=float)
    return (stress_indicator < healthy_threshold).astype(int)
