"""Day-approach specific preprocessing — sentinel handling and scaling."""

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import SENTINEL_REPLACEMENT, SENTINEL_VALUE


def handle_sentinel_values(
    df: pd.DataFrame,
    sentinel: float = SENTINEL_VALUE,
    replacement: float = SENTINEL_REPLACEMENT,
) -> pd.DataFrame:
    """Replace sentinel values (-0.01) with the replacement (0.0).

    The sentinel indicates "no data recorded for that day" (rest day).
    Replacing with 0.0 reflects the correct semantic meaning: no training load.
    """
    return df.replace(sentinel, replacement)


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on training features only.

    Returns the fitted scaler (for transforming both train and test).
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def transform_scaled(
    df: pd.DataFrame, scaler: StandardScaler, feature_cols: list[str]
) -> pd.DataFrame:
    """Apply a fitted scaler to feature columns, returning a new DataFrame."""
    result = df.copy()
    result[feature_cols] = scaler.transform(df[feature_cols])
    return result
