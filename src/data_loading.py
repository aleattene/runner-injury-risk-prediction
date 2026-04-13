"""Load and validate the day/week approach CSV files.

Handles column renaming from the original messy `.1`, `.2` suffixes
to structured `day_{i}_{feature}` / `week_{i}_{feature}` format.
"""

import logging

import pandas as pd

from src.config import (
    ATHLETE_ID_COL,
    DATE_COL,
    DAY_CSV,
    DAY_FEATURES,
    INJURY_COL,
    N_DAY_BLOCKS,
    N_WEEK_BLOCKS,
    RAW_DATA_DIR,
    WEEK_CSV,
    WEEK_FEATURES,
    WEEK_RELATIVE_FEATURES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Original column names as they appear in the CSV files
# ---------------------------------------------------------------------------

_DAY_ORIGINAL_FEATURES: list[str] = [
    "nr. sessions",
    "total km",
    "km Z3-4",
    "km Z5-T1-T2",
    "km sprinting",
    "strength training",
    "hours alternative",
    "perceived exertion",
    "perceived trainingSuccess",
    "perceived recovery",
]

_WEEK_ORIGINAL_FEATURES: list[str] = [
    "nr. sessions",
    "nr. rest days",
    "total kms",
    "max km one day",
    "total km Z3-Z4-Z5-T1-T2",
    "nr. tough sessions (effort in Z5, T1 or T2)",
    "nr. days with interval session",
    "total km Z3-4",
    "max km Z3-4 one day",
    "total km Z5-T1-T2",
    "max km Z5-T1-T2 one day",
    "total hours alternative training",
    "nr. strength trainings",
    "avg exertion",
    "min exertion",
    "max exertion",
    "avg training success",
    "min training success",
    "max training success",
    "avg recovery",
    "min recovery",
    "max recovery",
]


def _build_day_rename_map() -> dict[str, str]:
    """Build mapping from original day CSV column names to clean names."""
    rename: dict[str, str] = {}
    for block in range(N_DAY_BLOCKS):
        suffix = "" if block == 0 else f".{block}"
        for orig, clean in zip(_DAY_ORIGINAL_FEATURES, DAY_FEATURES):
            rename[f"{orig}{suffix}"] = f"day_{block}_{clean}"
    rename["Athlete ID"] = ATHLETE_ID_COL
    rename["injury"] = INJURY_COL
    rename["Date"] = DATE_COL
    return rename


def _build_week_rename_map() -> dict[str, str]:
    """Build mapping from original week CSV column names to clean names."""
    rename: dict[str, str] = {}
    for block in range(N_WEEK_BLOCKS):
        suffix = "" if block == 0 else f".{block}"
        for orig, clean in zip(_WEEK_ORIGINAL_FEATURES, WEEK_FEATURES):
            rename[f"{orig}{suffix}"] = f"week_{block}_{clean}"
    rename["Athlete ID"] = ATHLETE_ID_COL
    rename["injury"] = INJURY_COL
    rename["rel total kms week 0_1"] = WEEK_RELATIVE_FEATURES[0]
    rename["rel total kms week 0_2"] = WEEK_RELATIVE_FEATURES[1]
    rename["rel total kms week 1_2"] = WEEK_RELATIVE_FEATURES[2]
    rename["Date"] = DATE_COL
    return rename


def _validate_columns(
    df: pd.DataFrame, expected_cols: list[str], dataset_name: str
) -> None:
    """Raise ValueError if DataFrame columns don't match expected list."""
    actual: set[str] = set(df.columns)
    expected: set[str] = set(expected_cols)
    missing: set[str] = expected - actual
    extra: set[str] = actual - expected
    if missing or extra:
        msg = f"Column mismatch in {dataset_name}."
        if missing:
            msg += f" Missing: {sorted(missing)}."
        if extra:
            msg += f" Extra: {sorted(extra)}."
        raise ValueError(msg)


def load_day_data(path: str | None = None) -> pd.DataFrame:
    """Load the day-approach CSV and rename columns to clean format.

    Parameters
    ----------
    path : str or None
        Path to the CSV file. If None, uses the default location.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns like ``day_0_total_km``, ``day_1_nr_sessions``, etc.
    """
    csv_path = path or str(RAW_DATA_DIR / DAY_CSV)
    logger.info("Loading day-approach data from %s", csv_path)
    df = pd.read_csv(csv_path)

    rename_map = _build_day_rename_map()
    df = df.rename(columns=rename_map)

    expected_cols = list(rename_map.values())
    _validate_columns(df, expected_cols, "day_approach")

    logger.info("Day data loaded: %d rows, %d columns", len(df), len(df.columns))
    return df


def load_week_data(path: str | None = None) -> pd.DataFrame:
    """Load the week-approach CSV and rename columns to clean format.

    Parameters
    ----------
    path : str or None
        Path to the CSV file. If None, uses the default location.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns like ``week_0_total_kms``, ``week_1_avg_exertion``, etc.
    """
    csv_path = path or str(RAW_DATA_DIR / WEEK_CSV)
    logger.info("Loading week-approach data from %s", csv_path)
    df = pd.read_csv(csv_path)

    rename_map = _build_week_rename_map()
    df = df.rename(columns=rename_map)

    expected_cols = list(rename_map.values())
    _validate_columns(df, expected_cols, "week_approach")

    logger.info("Week data loaded: %d rows, %d columns", len(df), len(df.columns))
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all columns except metadata (athlete_id, injury, date, rel features)."""
    exclude = {ATHLETE_ID_COL, INJURY_COL, DATE_COL}
    return [c for c in df.columns if c not in exclude]
