"""I/O helpers for persisting processed train/test splits and fitted scalers."""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DATA_DIR

logger: logging.Logger = logging.getLogger(__name__)


def _validate_identifier(identifier: str) -> None:
    """Validate identifier to prevent path traversal attacks.

    Raises ValueError if identifier is empty, contains path separators,
    parent-directory segments (..), or is an absolute path.
    """
    if not identifier:
        msg = "Invalid identifier: must not be empty"
        raise ValueError(msg)

    if "/" in identifier or "\\" in identifier:
        msg = f"Invalid identifier '{identifier}': contains path separators"
        raise ValueError(msg)

    if Path(identifier).is_absolute():
        msg = f"Invalid identifier '{identifier}': must be relative (no absolute path)"
        raise ValueError(msg)

    # Reject ".." as a path component (not just substring)
    if ".." in Path(identifier).parts:
        msg = f"Invalid identifier '{identifier}': contains '..' segments"
        raise ValueError(msg)


def save_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    prefix: str,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> None:
    """Save train/test DataFrames as Parquet files.

    Files are named ``{prefix}_train.parquet`` and ``{prefix}_test.parquet``.
    """
    _validate_identifier(prefix)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path: Path = output_dir / f"{prefix}_train.parquet"
    test_path: Path = output_dir / f"{prefix}_test.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    logger.info(
        "Saved %s splits — train: %s (%s), test: %s (%s)",
        prefix,
        train_path.name,
        train_df.shape,
        test_path.name,
        test_df.shape,
    )


def load_splits(
    prefix: str,
    input_dir: Path = PROCESSED_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test DataFrames from Parquet files."""
    _validate_identifier(prefix)
    train_path: Path = input_dir / f"{prefix}_train.parquet"
    test_path: Path = input_dir / f"{prefix}_test.parquet"

    if not train_path.exists():
        msg = f"Train file not found: {train_path}"
        raise FileNotFoundError(msg)
    if not test_path.exists():
        msg = f"Test file not found: {test_path}"
        raise FileNotFoundError(msg)

    train_df: pd.DataFrame = pd.read_parquet(train_path)
    test_df: pd.DataFrame = pd.read_parquet(test_path)

    logger.info(
        "Loaded %s splits — train: %s, test: %s",
        prefix,
        train_df.shape,
        test_df.shape,
    )
    return train_df, test_df


def save_scaler(
    scaler: StandardScaler,
    name: str,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> None:
    """Persist a fitted StandardScaler with joblib."""
    _validate_identifier(name)
    output_dir.mkdir(parents=True, exist_ok=True)
    path: Path = output_dir / f"{name}.pkl"
    joblib.dump(scaler, path)
    logger.info("Saved scaler to %s", path.name)


def load_scaler(
    name: str,
    input_dir: Path = PROCESSED_DATA_DIR,
) -> StandardScaler:
    """Load a fitted StandardScaler from disk.

    Security note:
        ``joblib.load()`` deserializes pickle data and must only be used with
        trusted, locally generated scaler artifacts. Do not point this function
        at untrusted directories or externally supplied ``.pkl`` files.
    """
    _validate_identifier(name)
    base_dir: Path = input_dir.resolve()
    path: Path = (base_dir / f"{name}.pkl").resolve()
    try:
        path.relative_to(base_dir)
    except ValueError as exc:
        msg = f"Refusing to load scaler from outside the trusted directory: {path}"
        raise ValueError(msg) from exc
    if not path.exists():
        msg = f"Scaler file not found: {path}"
        raise FileNotFoundError(msg)
    scaler: StandardScaler = joblib.load(path)
    logger.info("Loaded scaler from %s", path.name)
    return scaler
