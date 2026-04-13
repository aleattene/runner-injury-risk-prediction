"""Shared test fixtures — sample data and dummy models."""

import numpy as np
import pandas as pd
import pytest

from src.config import ATHLETE_ID_COL, DATA_SAMPLE_DIR, INJURY_COL


@pytest.fixture
def sample_day_df() -> pd.DataFrame:
    """Load day_sample.csv and rename columns via data_loading."""
    from src.data_loading import load_day_data

    return load_day_data(str(DATA_SAMPLE_DIR / "day_sample.csv"))


@pytest.fixture
def sample_week_df() -> pd.DataFrame:
    """Load week_sample.csv and rename columns via data_loading."""
    from src.data_loading import load_week_data

    return load_week_data(str(DATA_SAMPLE_DIR / "week_sample.csv"))


@pytest.fixture
def small_xy_groups() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Tiny dataset for fast model tests: 120 rows, 5 features, 6 athletes.

    Near-balanced: 39 positive (32.5%), 81 negative (67.5%), with stratification
    across athletes. Each athlete has 6-7 positive samples, ensuring every fold
    will contain both classes even with GroupKFold.
    """
    rng = np.random.default_rng(42)

    n_athletes = 6
    samples_per_athlete = 20
    n_total = n_athletes * samples_per_athlete  # 120

    # Generate features
    X = pd.DataFrame(
        rng.standard_normal((n_total, 5)), columns=[f"feat_{i}" for i in range(5)]
    )

    # Generate balanced labels: 40 positive, 80 negative
    # Distribute evenly: 7-8 positives per athlete
    y_list = []
    groups_list = []

    for athlete in range(n_athletes):
        # Alternate: athletes 0,2,4 get 7 positives; 1,3,5 get 6 positives
        n_positive = 7 if athlete % 2 == 0 else 6
        n_negative = samples_per_athlete - n_positive

        # Create balanced split for this athlete
        athlete_labels = np.concatenate([np.ones(n_positive), np.zeros(n_negative)])

        # Shuffle within athlete
        rng.shuffle(athlete_labels)
        y_list.append(athlete_labels)

        # All samples from this athlete get same group ID
        groups_list.extend([athlete] * samples_per_athlete)

    y = pd.Series(np.concatenate(y_list), name=INJURY_COL).astype(int)
    groups = pd.Series(groups_list, name=ATHLETE_ID_COL)

    # Shuffle all together while preserving group integrity
    idx = rng.permutation(n_total)
    X = X.iloc[idx].reset_index(drop=True)
    y = y.iloc[idx].reset_index(drop=True)
    groups = groups.iloc[idx].reset_index(drop=True)

    return X, y, groups
