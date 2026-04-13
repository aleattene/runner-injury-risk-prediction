"""Week-approach specific preprocessing — target binarization."""

import pandas as pd

from src.config import WEEK_INJURY_THRESHOLD


def binarize_target(
    y: pd.Series, threshold: float = WEEK_INJURY_THRESHOLD
) -> pd.Series:
    """Convert continuous injury score to binary (0/1).

    Parameters
    ----------
    y : pd.Series
        Continuous injury values (0.0 to ~1.5+).
    threshold : float
        Values >= threshold become 1, below become 0.

    Returns
    -------
    pd.Series
        Binary injury target.
    """
    return (y >= threshold).astype(int)
