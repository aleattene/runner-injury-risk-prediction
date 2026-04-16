"""SHAP-based model interpretability — summary, dependence, and waterfall plots."""

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from src.utils.plotting import save_figure

logger = logging.getLogger(__name__)


def _extract_positive_class(shap_values: shap.Explanation) -> shap.Explanation:
    """Slice 3D SHAP Explanation to the positive class (index -1).

    Tree-based binary classifiers produce SHAP values with shape
    (n_samples, n_features, 2). This helper returns an Explanation
    with 2D values for the positive class only. If the input is
    already 2D, it is returned unchanged.
    """
    values = np.asarray(shap_values.values)
    if values.ndim == 3:
        base_values = np.asarray(shap_values.base_values)
        n_outputs = values.shape[-1]
        if base_values.ndim == 2:
            positive_base_values = base_values[:, -1]
        elif base_values.ndim == 1 and base_values.shape[0] == n_outputs:
            positive_base_values = base_values[-1]
        else:
            positive_base_values = base_values
        return shap.Explanation(
            values=values[:, :, -1],
            base_values=positive_base_values,
            data=shap_values.data,
            feature_names=shap_values.feature_names,
        )
    return shap_values


def compute_shap_values(model: BaseEstimator, X: pd.DataFrame) -> shap.Explanation:
    """Compute SHAP values using the appropriate explainer for the model type.

    Uses LinearExplainer for LogisticRegression, TreeExplainer for tree-based models.

    Parameters
    ----------
    model : BaseEstimator
        A fitted scikit-learn compatible estimator.
    X : pd.DataFrame
        Feature matrix to explain.

    Returns
    -------
    shap.Explanation
        SHAP values for each sample and feature.
    """
    if isinstance(model, LogisticRegression):
        logger.info("Using LinearExplainer for %s", type(model).__name__)
        explainer = shap.LinearExplainer(model, X)
    else:
        logger.info("Using TreeExplainer for %s", type(model).__name__)
        explainer = shap.TreeExplainer(model)

    shap_values = explainer(X)
    logger.info("SHAP values computed: shape %s", shap_values.shape)
    return shap_values


def plot_shap_summary(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    max_display: int = 20,
    save_path: Path | None = None,
) -> plt.Figure:
    """Create a SHAP summary (beeswarm) plot.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values for the dataset.
    X : pd.DataFrame
        Feature matrix used to compute SHAP values.
    max_display : int
        Maximum number of features to display.
    save_path : Path or None
        If given, ``stem`` and ``parent.name`` are extracted and passed to
        ``save_figure()`` which saves under ``reports/figures/``.

    Returns
    -------
    plt.Figure
        The summary plot figure.
    """
    pos_class = _extract_positive_class(shap_values)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The NumPy global RNG was seeded",
            category=FutureWarning,
        )
        shap.summary_plot(pos_class, X, max_display=max_display, show=False)
    fig = plt.gcf()
    if save_path:
        save_figure(fig, save_path.stem, save_path.parent.name or None, close=False)
    return fig


def plot_shap_dependence(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    feature: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """Create a SHAP dependence plot for a single feature.

    For multi-output SHAP values (e.g., binary tree classifiers that return
    shape (n_samples, n_features, 2)), this plots the positive class
    (last output) so that shap.dependence_plot receives the required 2D array.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values for the dataset.
    X : pd.DataFrame
        Feature matrix used to compute SHAP values.
    feature : str
        Name of the feature to plot.
    save_path : Path or None
        If given, ``stem`` and ``parent.name`` are extracted and passed to
        ``save_figure()`` which saves under ``reports/figures/``.

    Returns
    -------
    plt.Figure
        The dependence plot figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    is_3d = np.asarray(shap_values.values).ndim == 3
    pos_class = _extract_positive_class(shap_values)
    output_suffix = " (positive class)" if is_3d else ""

    shap.dependence_plot(feature, pos_class.values, X, show=False, ax=ax)
    ax.set_title(f"SHAP Dependence: {feature}{output_suffix}")
    if save_path:
        save_figure(fig, save_path.stem, save_path.parent.name or None, close=False)
    return fig


def get_top_features(shap_values: shap.Explanation, n: int = 10) -> list[str]:
    """Return the top N most important features by mean absolute SHAP value.

    Handles both 2D (linear models) and 3D (tree-based models) SHAP arrays.
    """
    importance = get_shap_importance_dict(shap_values)
    sorted_features = sorted(importance, key=lambda f: importance[f], reverse=True)
    return sorted_features[:n]


def get_shap_importance_dict(shap_values: shap.Explanation) -> dict[str, float]:
    """Return a dict mapping feature name → mean |SHAP| value.

    For 3D arrays (tree binary classifiers), uses the positive class only.
    """
    pos_class = _extract_positive_class(shap_values)
    mean_abs = np.abs(pos_class.values).mean(axis=0)
    return dict(zip(pos_class.feature_names, mean_abs.tolist()))


def plot_shap_waterfall(
    shap_values: shap.Explanation,
    index: int,
    max_display: int = 15,
    save_path: Path | None = None,
) -> plt.Figure:
    """Create a SHAP waterfall plot for a single prediction.

    Waterfall plots replace the deprecated force_plot, showing how each
    feature contributes to pushing the prediction from the base value.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values for the dataset.
    index : int
        Row index of the sample to explain.
    max_display : int
        Maximum number of features to display.
    save_path : Path or None
        If given, ``stem`` and ``parent.name`` are extracted and passed to
        ``save_figure()`` which saves under ``reports/figures/``.

    Returns
    -------
    plt.Figure
        The waterfall plot figure.
    """
    pos_class = _extract_positive_class(shap_values)
    shap.plots.waterfall(pos_class[index], max_display=max_display, show=False)
    fig = plt.gcf()
    if save_path:
        save_figure(fig, save_path.stem, save_path.parent.name or None, close=False)
    return fig


def compare_feature_importance(
    shap_importance: dict[str, float],
    builtin_importance: dict[str, float],
    top_n: int = 15,
    builtin_label: str = "XGBoost Gain",
    save_path: Path | None = None,
) -> plt.Figure:
    """Create a side-by-side bar chart comparing SHAP vs built-in importance.

    Parameters
    ----------
    shap_importance : dict[str, float]
        Feature name → mean |SHAP| value.
    builtin_importance : dict[str, float]
        Feature name → model built-in importance (e.g., XGBoost gain).
    top_n : int
        Number of top features to display (by SHAP rank).
    builtin_label : str
        Label for the built-in importance axis and title.
    save_path : Path or None
        If given, ``stem`` and ``parent.name`` are extracted and passed to
        ``save_figure()`` which saves under ``reports/figures/``.

    Returns
    -------
    plt.Figure
        The comparison chart figure.
    """
    # Sort by SHAP importance, take top_n
    sorted_features = sorted(
        shap_importance, key=lambda f: shap_importance[f], reverse=True
    )[:top_n]
    n_bars = len(sorted_features)

    shap_vals = [shap_importance[f] for f in sorted_features]
    builtin_vals = [builtin_importance.get(f, 0.0) for f in sorted_features]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # SHAP importance (left)
    axes[0].barh(range(n_bars), shap_vals[::-1], color="#2196F3")
    axes[0].set_yticks(range(n_bars))
    axes[0].set_yticklabels(sorted_features[::-1])
    axes[0].set_xlabel("Mean |SHAP value|")
    axes[0].set_title("SHAP Importance")

    # Built-in importance (right)
    axes[1].barh(range(n_bars), builtin_vals[::-1], color="#FF9800")
    axes[1].set_yticks(range(n_bars))
    axes[1].set_yticklabels(sorted_features[::-1])
    axes[1].set_xlabel(builtin_label)
    axes[1].set_title("Built-in Importance")

    fig.suptitle(f"Feature Importance: SHAP vs {builtin_label}", fontsize=14)
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path.stem, save_path.parent.name or None, close=False)
    return fig
