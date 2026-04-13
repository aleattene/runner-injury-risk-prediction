"""Reproducibility utilities — seed management and environment snapshot."""

import random

import numpy as np

from src.config import RANDOM_SEED


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed for Python, NumPy, and scikit-learn reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)
