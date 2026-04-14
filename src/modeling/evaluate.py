"""Model evaluation — metrics, curves, and comparison utilities."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.plotting import save_figure

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict[str, float]:
    """Compute a full suite of classification metrics.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels (at a given threshold).
    y_prob : array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    dict[str, float]
        Metric name -> value.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Guard against single-class y_true (common in small imbalanced splits)
    if np.unique(y_true).size < 2:
        logger.warning(
            "Only one class present in y_true; returning NaN for auc_roc and auc_pr."
        )
        auc_roc = np.nan
        auc_pr = np.nan
    else:
        auc_roc = roc_auc_score(y_true, y_prob)
        auc_pr = average_precision_score(y_true, y_prob)

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "brier_score": brier_score_loss(y_true, y_prob),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def find_optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1"
) -> float:
    """Find the classification threshold that maximizes the given metric.

    Parameters
    ----------
    metric : str
        "f1" or "recall" (at minimum precision of 0.05).

    Returns
    -------
    float
        Optimal threshold in (0, 1).
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_score: float = 0.0
    best_threshold: float = 0.5
    found_valid_threshold: bool = False

    # For recall metric, track best unconstrained threshold as fallback
    fallback_score: float = 0.0
    fallback_threshold: float = 0.5

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            recall = recall_score(y_true, y_pred, zero_division=0)
            prec = precision_score(y_true, y_pred, zero_division=0)

            # Track best unconstrained recall for fallback
            if recall > fallback_score:
                fallback_score = recall
                fallback_threshold = float(t)

            # Skip threshold if precision constraint not met
            if prec < 0.05:
                continue
            score = recall
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = float(t)
            found_valid_threshold = True

    # For recall metric, if no threshold satisfied precision constraint,
    # fall back to best unconstrained recall with warning
    if metric == "recall" and not found_valid_threshold:
        logger.warning(
            "No threshold satisfied the minimum precision constraint "
            "(precision >= 0.05) for metric=%s; falling back to the best "
            "unconstrained recall threshold %.2f (score=%.4f).",
            metric,
            fallback_threshold,
            fallback_score,
        )
        best_threshold = fallback_threshold
        best_score = fallback_score

    logger.info(
        "Optimal threshold (metric=%s): %.2f (score=%.4f)",
        metric,
        best_threshold,
        best_score,
    )
    return best_threshold


def plot_roc_curves(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot ROC curves for multiple models.

    Parameters
    ----------
    results : dict
        Model name -> (y_true, y_prob).
    save_path : Path or None
        If provided, its stem is used as the figure name and its parent
        directory name as the subdirectory inside ``reports/figures/``.
        The figure is not saved to the literal filesystem path.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    for name, (y_true, y_prob) in results.items():
        if np.unique(y_true).size < 2:
            logger.warning(
                "Skipping ROC curve for %s because y_true contains only one class.",
                name,
            )
            ax.plot([], [], label=f"{name} (AUC = NaN)")
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    if save_path:
        save_figure(fig, save_path.stem, save_path.parent.name or None, close=False)
    return fig


def plot_pr_curves(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot Precision-Recall curves for multiple models.

    Parameters
    ----------
    results : dict
        Model name -> (y_true, y_prob).
    save_path : Path or None
        If provided, its stem is used as the figure name and its parent
        directory name as the subdirectory inside ``reports/figures/``.
        The figure is not saved to the literal filesystem path.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    for name, (y_true, y_prob) in results.items():
        if np.unique(y_true).size < 2:
            logger.warning(
                "Skipping PR curve for %s because y_true contains only one class.",
                name,
            )
            ax.plot([], [], label=f"{name} (AP = NaN)")
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, label=f"{name} (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="upper right")
    if save_path:
        save_figure(fig, save_path.stem, save_path.parent.name or None, close=False)
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot a confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.
    title : str
        Plot title.
    save_path : Path or None
        If provided, its stem is used as the figure name and its parent
        directory name as the subdirectory inside ``reports/figures/``.
        The figure is not saved to the literal filesystem path.
    """
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Injury", "Injury"],
        yticklabels=["No Injury", "Injury"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    if save_path:
        save_figure(fig, save_path.stem, save_path.parent.name or None, close=False)
    return fig


def create_comparison_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Create a comparison table from model metrics.

    Parameters
    ----------
    results : dict
        Model name -> metrics dict (from compute_metrics).

    Returns
    -------
    pd.DataFrame
        Comparison table with models as rows and metrics as columns.
    """
    return pd.DataFrame(results).T.round(4)
