# Cognitive UAV-Driven Agro-Surveillance

Reference implementation of **GeoSpatio-TRiNet** (a diffusion-aware
spatio-temporal stress-forecasting model) and a **Soft Actor-Critic (SAC)**
adaptive irrigation controller, from:

> Selvakumar, S. & Venugopal, D. *"Cognitive UAV-Driven Agro-Surveillance
> Framework for Predicting Crop Stress-Induced Yield Loss Using
> Spatio-Temporal Learning and Adaptive Irrigation Control."*

The framework forecasts crop-stress-induced **yield vulnerability**
trajectories (rather than regressing yield directly) from UAV-derived
vegetation indices, soil, and weather data, and feeds those forecasts into a
reinforcement-learning agent that makes proactive, zone-level irrigation
decisions.

## Architecture

```
raw table (Kaggle "Crop Health and Environmental Stress Dataset")
      |
      v
 stress-aware normalization  +  event-anchored temporal alignment   (Sec. 3.2-3.3)
      |
      v
 adaptive spatial zoning -> zone-time sequences  X_z^t                (Sec. 3.1, 3.4)
      |
      v
 +----------------------------- GeoSpatio-TRiNet -----------------------------+
 |  spatial attention  ->  temporal encoder (GRU)  ->  diffusion module       |  (Sec. 3.6)
 +-------------------------------------------------------------------------- +
      |
      v
 stress probability  ->  vulnerability trajectory V_t  (Eq. 28/30)            (Sec. 3.7)
      |
      v
 cognitive RL state s_t = [V_t, M_t, T_t, E_t, P_t]                           (Sec. 3.8)
      |
      v
 Soft Actor-Critic  ->  zone-level irrigation action a_t in [0, a_max]        (Sec. 3.9)
```

See [`docs/equations.md`](docs/equations.md) for a full map from every
numbered equation in the paper to the function/class that implements it, and
[`docs/dataset.md`](docs/dataset.md) for the exact column schema and the
modeling assumptions used to bridge the public dataset to the paper's
formulation.

## Repository layout

```
src/agrosurveillance/
  data/          stress-aware normalization, event alignment, spatial zoning,
                 zone-time sequence construction, synthetic data generator
  models/        GeoSpatio-TRiNet (spatial attention / temporal encoder /
                 diffusion) + classical & deep-learning baselines
  rl/            SAC agent, artificial agro-environment, replay buffer,
                 lightweight RL baselines (Q-learning, DQN, DDPG)
  evaluation/    accuracy/F1/ROC-AUC/PR-AUC, vulnerability reduction,
                 water-use efficiency, early-detection lead time
  training/      CLI entry points: forecasting training, RL training,
                 ablation study runner
configs/         default.yaml - all hyperparameters in one place
scripts/         download_dataset.py (Kaggle), run_demo.py (quickstart)
tests/           pytest suite covering every module above
docs/            equations.md, dataset.md
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -r requirements.txt
pip install -e .                                       # makes `agrosurveillance` importable
```

Requires Python >= 3.10. `torch` is installed CPU-only by default; install a
CUDA build yourself first if you want GPU training (`training.device: auto`
in the config will then pick it up automatically).

## Quickstart (no dataset required)

```bash
python scripts/run_demo.py
```

Runs a scaled-down version of the entire pipeline - stress forecasting,
SAC irrigation control, and a 2-config ablation - on synthetic data in well
under a minute on CPU.

## Reproducing paper-comparable results

1. **Get the real dataset:**
   ```bash
   pip install kaggle   # + configure ~/.kaggle/kaggle.json, see data/README.md
   python scripts/download_dataset.py
   ```
2. **Train & evaluate the stress-forecasting models** (Table 2 style output -
   Logistic Regression / Decision Tree / Random Forest / XGBoost / CNN /
   CNN-LSTM / Temporal Transformer / GeoSpatio-TRiNet):
   ```bash
   python -m agrosurveillance.training.train_forecasting --config configs/default.yaml
   ```
3. **Train the SAC irrigation controller** (Table 4/5/7 style output - No
   Control / Threshold / Q-Learning / DQN / DDPG / SAC):
   ```bash
   python -m agrosurveillance.training.train_rl --config configs/default.yaml
   ```
4. **Run the ablation study** (Table 8 style output - without zoning /
   diffusion / temporal encoding / RL / stress-aware normalization):
   ```bash
   python -m agrosurveillance.training.ablation --config configs/default.yaml
   ```

All hyperparameters (epochs, episodes, learning rates, reward weights, model
dimensions) live in `configs/default.yaml`. Copy it and pass `--config
your_config.yaml` to sweep settings without touching code.

## Testing

```bash
pytest tests/ -v
```

22 tests cover the normalization/alignment math, zone-sequence construction
and causal splitting, GeoSpatio-TRiNet forward passes (including every
ablation-flag combination), the RL environment dynamics, and SAC's update
step. CI (`.github/workflows/tests.yml`) runs this suite plus the full demo
on every push.

## License

MIT - see [LICENSE](LICENSE).
