"""Shared preprocessing utilities — train/test split respecting athlete groups."""

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from src.config import (
    ATHLETE_ID_COL,
    INJURY_COL,
    N_CV_FOLDS,
    RANDOM_SEED,
    TEST_SIZE,
)


def split_train_test(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train and test, grouping by athlete ID.

    All observations from one athlete appear in exactly one split.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    groups = df[ATHLETE_ID_COL]
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(
        drop=True
    )


def get_group_kfold(n_splits: int = N_CV_FOLDS) -> GroupKFold:
    """Return a GroupKFold splitter for cross-validation by athlete ID."""
    return GroupKFold(n_splits=n_splits)


def get_feature_target_groups(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Extract X, y, and groups from a DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, pd.Series]
        (X, y, groups)
    """
    X = df[feature_cols]
    y = df[INJURY_COL]
    groups = df[ATHLETE_ID_COL]
    return X, y, groups
