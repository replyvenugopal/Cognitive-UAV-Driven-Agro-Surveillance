"""Global seeding for reproducibility across numpy / torch / python's random."""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed all RNGs used across the pipeline.

    The paper reports mean +/- std over >= 5 random seeds (Section 4). Callers
    performing repeated-seed evaluation should call this once per run with a
    different seed each time.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
