"""Fairness audit — per-group metrics and disparity analysis.

Since the dataset has no demographic attributes (age, sex, ethnicity are masked),
this module constructs proxy groups from training data characteristics:
- Training volume quartiles (high vs low volume athletes)
- Injury history (ever-injured vs never-injured in training set)
- Data density (many vs few observations per athlete)
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import ATHLETE_ID_COL, INJURY_COL

logger = logging.getLogger(__name__)


def create_athlete_groups(
    df: pd.DataFrame, method: str, feature_col: str | None = None
) -> pd.Series:
    """Assign each row to a group based on athlete-level characteristics.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame with athlete_id and features.
    method : str
        Grouping method: "volume", "injury_history", or "data_density".
    feature_col : str or None
        For "volume" method, the column to aggregate (e.g., "day_0_total_km").

    Returns
    -------
    pd.Series
        Group label for each row, aligned with df index.
    """
    athlete_ids = df[ATHLETE_ID_COL]

    if method == "volume":
        if feature_col is None:
            raise ValueError("feature_col required for 'volume' method")
        athlete_avg = df.groupby(ATHLETE_ID_COL)[feature_col].mean()
        median_val = athlete_avg.median()
        athlete_group = (athlete_avg >= median_val).map(
            {True: "high_volume", False: "low_volume"}
        )
        return athlete_ids.map(athlete_group).rename("group")

    elif method == "injury_history":
        athlete_injury = df.groupby(ATHLETE_ID_COL)[INJURY_COL].max()
        athlete_group = (athlete_injury > 0).map(
            {True: "ever_injured", False: "never_injured"}
        )
        return athlete_ids.map(athlete_group).rename("group")

    elif method == "data_density":
        athlete_count = df.groupby(ATHLETE_ID_COL).size()
        median_count = athlete_count.median()
        athlete_group = (athlete_count >= median_count).map(
            {True: "high_density", False: "low_density"}
        )
        return athlete_ids.map(athlete_group).rename("group")

    else:
        raise ValueError(
            f"Unknown method '{method}'. Supported: volume, injury_history, "
            "data_density"
        )


def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    groups: pd.Series,
) -> pd.DataFrame:
    """Compute classification metrics separately for each group.

    Returns
    -------
    pd.DataFrame
        One row per group, columns: group, recall, precision, f1, fpr, auc_roc, support.
    """
    results: list[dict] = []
    for group_name in sorted(groups.unique()):
        mask = groups == group_name
        yt = y_true[mask]
        yp = y_pred[mask]
        ypr = y_prob[mask]
        n_pos = int(yt.sum())
        n_total = int(mask.sum())

        row: dict = {
            "group": group_name,
            "support": n_total,
            "n_positive": n_pos,
            "prevalence": n_pos / n_total if n_total > 0 else 0.0,
        }

        if n_pos > 0 and n_pos < n_total:
            row["recall"] = recall_score(yt, yp, zero_division=0)
            row["precision"] = precision_score(yt, yp, zero_division=0)
            row["f1"] = f1_score(yt, yp, zero_division=0)
            tn = int(((yt == 0) & (yp == 0)).sum())
            fp = int(((yt == 0) & (yp == 1)).sum())
            row["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            row["auc_roc"] = roc_auc_score(yt, ypr)
        else:
            row.update(
                {
                    "recall": np.nan,
                    "precision": np.nan,
                    "f1": np.nan,
                    "fpr": np.nan,
                    "auc_roc": np.nan,
                }
            )
            logger.warning(
                "Group '%s' has %d positives out of %d — metrics may be unreliable",
                group_name,
                n_pos,
                n_total,
            )

        results.append(row)

    return pd.DataFrame(results)


def compute_disparity_ratios(
    group_metrics: pd.DataFrame, reference_group: str
) -> pd.DataFrame:
    """Compute disparity ratios relative to a reference group.

    For each metric, disparity = group_value / reference_value.
    A ratio of 1.0 means parity. Values far from 1.0 indicate disparity.

    Returns
    -------
    pd.DataFrame
        Same shape as group_metrics but with ratio values.
    """
    metric_cols = ["recall", "precision", "f1", "fpr", "auc_roc"]
    ref_row = group_metrics.loc[group_metrics["group"] == reference_group]
    if ref_row.empty:
        raise ValueError(f"Reference group '{reference_group}' not found")

    result = group_metrics.copy()
    for col in metric_cols:
        ref_val = ref_row[col].iloc[0]
        if pd.notna(ref_val) and ref_val != 0:
            result[f"{col}_ratio"] = result[col] / ref_val
        else:
            result[f"{col}_ratio"] = np.nan
    return result
