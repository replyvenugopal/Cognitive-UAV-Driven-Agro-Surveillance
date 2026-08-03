# Dataset schema and modeling assumptions

## Source

Kaggle: [`datasetengineer/crop-health-and-environmental-stress-dataset`](https://www.kaggle.com/datasets/datasetengineer/crop-health-and-environmental-stress-dataset)
(~212k rows). It is a **flat table**, not pre-packaged zone/time sequences
or raw UAV imagery - see below for how this repo bridges that gap to match
the paper's formulation (Eq. 1-4).

## Columns used (`agrosurveillance/data/schema.py`)

| Group | Columns | Paper symbol |
|---|---|---|
| UAV/remote-sensing proxy | `Canopy_Coverage`, `NDVI`, `SAVI`, `Leaf_Area_Index`, `Chlorophyll_Content` | `U_RGB`, `U_MS`, `U_TH` |
| Elevation | `Elevation_Data` | - |
| Soil | `Soil_Moisture`, `Soil_pH`, `Organic_Matter` | `S_z,t` |
| Environment/weather | `Temperature`, `Humidity`, `Rainfall`, `Wind_Speed` | `E^t` |
| Pest/crop condition | `Pest_Hotspots`, `Weed_Coverage`, `Pest_Damage` | auxiliary context |
| Categorical | `Field_Boundaries`, `Crop_Growth_Stage`, `Crop_Type` | zone/crop metadata |
| Target | `Crop_Stress_Indicator` (0-100) | `Y_z^t` |

`health_label` (1 = healthy, 0 = unhealthy) is derived at load time by
thresholding `Crop_Stress_Indicator` (`schema.derive_health_label`), mirroring
the healthy/unhealthy class split discussed around Figure 5 in the paper.

## Modeling assumptions (and why)

The public dataset does not ship the same structure the paper's equations
assume. Rather than silently pretending otherwise, every assumption below is
implemented explicitly and is overridable:

1. **No raw UAV imagery.** The dataset ships vegetation indices
   (NDVI/SAVI/canopy coverage/chlorophyll/LAI) rather than pixel-level
   RGB/multispectral/thermal frames. These indices are treated as the UAV
   feature vector `X_z^t` (Eq. 2). If you have access to the raw imagery this
   dataset was derived from, swap in a CNN feature extractor ahead of
   `agrosurveillance.models.geospatio_trinet.GeoSpatioTriNet`.

2. **No explicit timestamp.** Rows are treated as chronologically ordered
   within their `Field_Boundaries` group, and split into fixed-length
   windows of `data.sequence_length` to form zone-time sequences
   (`agrosurveillance/data/dataset.py::build_zone_sequences`). The causal
   train/test split (Eq. 3-4) operates on this row order.

3. **No GPS coordinates -> no ground-truth spatial adjacency.** The spatial
   attention (Eq. 20-23) and diffusion (Eq. 26-27) modules default to
   treating all other zones in a training batch as candidate neighbors and
   let attention learn the weighting. `agrosurveillance/data/spatial_zoning.py`
   also provides a `zone_adjacency()` helper (ring topology over
   feature-similarity-sorted zones) that can be passed as an explicit
   `adjacency_mask` if you have real spatial layout information.

4. **No canopy-vs-ambient thermal channel pair.** `Chlorophyll_Content` is
   used as a thermal-stress proxy against the recorded `Temperature` column
   to compute the thermal deviation term (Eq. 6/13). This is documented as
   an approximation in `normalization.thermal_stress_deviation`.

5. **RL environment uses ground-truth trajectories, not model predictions.**
   Following the paper's own description ("the agent is interfaced with an
   artificial agro-environment that is created on the basis of the
   dataset", Section 4), `agrosurveillance/rl/environment.py` replays each
   zone's *observed* stress/moisture/environment trajectory as the
   exogenous driver, with irrigation actions perturbing simulated soil
   moisture and vulnerability. It does not call GeoSpatio-TRiNet inside the
   RL loop. Wiring the two together (live model predictions feeding RL
   state) is a natural next step - see the README's Limitations section.

All of the above are implemented as clearly-named functions/parameters
specifically so they can be swapped out if you have a richer dataset (real
UAV frames, real timestamps, real GPS zone boundaries) available.
