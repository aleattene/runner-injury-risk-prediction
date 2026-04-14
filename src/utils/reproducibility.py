"""Reproducibility utilities — seed management."""

import random

import numpy as np

from src.config import RANDOM_SEED


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed for Python's random module and NumPy."""
    random.seed(seed)
    np.random.seed(seed)
