"""Fairness audit — per-group metrics and disparity analysis.

Since the dataset has no demographic attributes (age, sex, ethnicity are masked),
this module constructs proxy groups from training data characteristics:
- Training volume median split (high vs low volume athletes)
- Injury history (ever-injured vs never-injured in training set)
- Data density (many vs few observations per athlete)
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import ATHLETE_ID_COL, INJURY_COL
from src.utils.plotting import PALETTE, save_figure

logger = logging.getLogger(__name__)


def create_athlete_groups(
    df: pd.DataFrame,
    method: str,
    feature_col: str | None = None,
    reference_df: pd.DataFrame | None = None,
) -> pd.Series:
    """Assign each row to a group based on athlete-level characteristics.

    Parameters
    ----------
    df : pd.DataFrame
        Target DataFrame whose rows receive group labels.
    method : str
        Grouping method: "volume", "injury_history", or "data_density".
    feature_col : str or None
        For "volume" method, the column to aggregate (e.g., "day_0_total_km").
    reference_df : pd.DataFrame or None
        If provided, group membership is computed from this DataFrame and
        mapped onto *df* to avoid using *df* itself for group assignment
        (e.g., when deriving "injury_history"). All athletes present in *df*
        must also be present in *reference_df*; otherwise a ValueError is
        raised. In particular, with athlete-disjoint train/test splits the
        training set cannot be used as *reference_df* for the test set.
        In that case, compute groups directly from *df* instead.

    Returns
    -------
    pd.Series
        Group label for each row, aligned with df index.
    """
    source = reference_df if reference_df is not None else df
    athlete_ids = df[ATHLETE_ID_COL]

    # Validate that df athletes exist in the reference source
    if reference_df is not None:
        source_athletes = set(reference_df[ATHLETE_ID_COL].unique())
        target_athletes = set(df[ATHLETE_ID_COL].unique())
        missing = target_athletes - source_athletes
        if missing:
            raise ValueError(
                f"Athletes {missing} in df are not present in reference_df. "
                "With athlete-disjoint train/test splits, reference_df cannot "
                "map groups onto df. Compute groups directly from df instead."
            )

    if method == "volume":
        if feature_col is None:
            raise ValueError("feature_col required for 'volume' method")
        athlete_avg = source.groupby(ATHLETE_ID_COL)[feature_col].mean()
        median_val = athlete_avg.median()
        athlete_group = (athlete_avg >= median_val).map(
            {True: "high_volume", False: "low_volume"}
        )
        return athlete_ids.map(athlete_group).rename("group")

    elif method == "injury_history":
        athlete_injury = source.groupby(ATHLETE_ID_COL)[INJURY_COL].max()
        athlete_group = (athlete_injury > 0).map(
            {True: "ever_injured", False: "never_injured"}
        )
        return athlete_ids.map(athlete_group).rename("group")

    elif method == "data_density":
        athlete_count = source.groupby(ATHLETE_ID_COL).size()
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


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

_METRIC_COLS: list[str] = ["recall", "precision", "f1", "fpr", "auc_roc"]


def plot_group_metrics_bars(
    metrics_df: pd.DataFrame,
    title: str = "Per-Group Classification Metrics",
    metrics: list[str] | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Grouped horizontal bar chart comparing metrics across groups.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Output of :func:`compute_group_metrics`.
    title : str
        Plot title.
    metrics : list[str] or None
        Metric columns to plot (default: recall, precision, f1, fpr, auc_roc).
    save_path : Path or None
        If given, ``save_path.parent.name`` is the sub-directory and
        ``save_path.stem`` is the file name passed to :func:`save_figure`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if metrics_df.empty:
        raise ValueError("metrics_df is empty — nothing to plot")

    if metrics is None:
        metrics = [c for c in _METRIC_COLS if c in metrics_df.columns]

    if not metrics:
        raise ValueError(
            "No metric columns to plot. Provide a non-empty `metrics` list "
            f"or ensure metrics_df contains at least one of: {_METRIC_COLS}"
        )

    groups = metrics_df["group"].tolist()
    n_groups = len(groups)
    n_metrics = len(metrics)
    bar_colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["neutral"]]

    y = np.arange(n_metrics)
    height = 0.8 / n_groups

    fig, ax = plt.subplots(figsize=(10, max(4, n_metrics * 0.8)))
    for i, group in enumerate(groups):
        row = metrics_df.loc[metrics_df["group"] == group]
        values = [
            float(row[m].iloc[0]) if pd.notna(row[m].iloc[0]) else np.nan
            for m in metrics
        ]
        offset = (i - (n_groups - 1) / 2) * height
        ax.barh(
            y + offset,
            values,
            height,
            label=group,
            color=bar_colors[i % len(bar_colors)],
            alpha=0.85,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.set_xlabel("Score")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.set_xlim(0, 1.05)
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path.stem, save_path.parent.name or None, close=False)
    return fig


def plot_disparity_ratios(
    disparity_df: pd.DataFrame,
    title: str = "Disparity Ratios (reference = 1.0)",
    save_path: Path | None = None,
) -> plt.Figure:
    """Horizontal bar chart of disparity ratios with a 1.0 reference line.

    Rows where *all* ratio columns equal 1.0 (the reference group) are excluded
    from the plot so that only non-reference groups appear.

    Parameters
    ----------
    disparity_df : pd.DataFrame
        Output of :func:`compute_disparity_ratios`.
    title : str
        Plot title.
    save_path : Path or None
        If given, ``save_path.parent.name`` is the sub-directory and
        ``save_path.stem`` is the file name passed to :func:`save_figure`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    ratio_cols = [c for c in disparity_df.columns if c.endswith("_ratio")]
    if not ratio_cols:
        raise ValueError(
            "disparity_df must contain at least one '*_ratio' column to plot"
        )
    # Exclude reference group (all defined ratios ≈ 1.0).
    # Rows with all-NaN ratios are NOT treated as reference.
    is_ref = disparity_df[ratio_cols].apply(
        lambda row: row.notna().all() and np.isclose(row.astype(float), 1.0).all(),
        axis=1,
    )
    plot_df = disparity_df.loc[~is_ref].copy()

    metric_labels = [c.replace("_ratio", "") for c in ratio_cols]
    n_metrics = len(ratio_cols)

    fig, ax = plt.subplots(figsize=(10, max(4, n_metrics * 0.9)))
    y = np.arange(n_metrics)

    n_groups = len(plot_df)
    bar_height = 0.8 / max(n_groups, 1)
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        values = [float(row[c]) if pd.notna(row[c]) else np.nan for c in ratio_cols]
        colors = [
            (
                PALETTE["neutral"]
                if pd.isna(v)
                else PALETTE["positive"] if v >= 1.0 else PALETTE["negative"]
            )
            for v in values
        ]
        offset = (idx - (n_groups - 1) / 2) * bar_height
        ax.barh(
            y + offset,
            values,
            height=bar_height,
            color=colors,
            alpha=0.85,
            label=row["group"],
        )

    ax.axvline(
        1.0,
        color=PALETTE["neutral"],
        linestyle="--",
        linewidth=1.5,
        label="Parity (1.0)",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(metric_labels)
    ax.set_xlabel("Disparity Ratio")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path.stem, save_path.parent.name or None, close=False)
    return fig


def plot_fairness_summary_heatmap(
    all_metrics: dict[str, pd.DataFrame],
    title: str = "Fairness Metrics Across Grouping Methods",
    save_path: Path | None = None,
) -> plt.Figure:
    """Heatmap showing metrics for all grouping methods at a glance.

    Parameters
    ----------
    all_metrics : dict[str, pd.DataFrame]
        Mapping ``{method_name: metrics_df}`` from :func:`compute_group_metrics`.
    title : str
        Plot title.
    save_path : Path or None
        If given, ``save_path.parent.name`` is the sub-directory and
        ``save_path.stem`` is the file name passed to :func:`save_figure`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not all_metrics:
        raise ValueError("all_metrics is empty — provide at least one method")

    rows: list[dict] = []
    for method, df in all_metrics.items():
        for _, r in df.iterrows():
            row_label = f"{method} | {r['group']}"
            entry: dict = {"label": row_label}
            for col in _METRIC_COLS:
                if col in r.index:
                    entry[col] = r[col]
            rows.append(entry)

    if not rows:
        raise ValueError("all_metrics DataFrames are all empty — nothing to plot")

    heat_df = pd.DataFrame(rows).set_index("label")
    fig, ax = plt.subplots(figsize=(10, max(4, len(heat_df) * 0.7)))
    sns.heatmap(
        heat_df.astype(float),
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path.stem, save_path.parent.name or None, close=False)
    return fig
