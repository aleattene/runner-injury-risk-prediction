"""SHAP-based model interpretability — summary, dependence, and force plots."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from src.utils.plotting import save_figure

logger = logging.getLogger(__name__)


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
    """Create a SHAP summary (beeswarm) plot."""
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    fig = plt.gcf()
    if save_path:
        save_figure(fig, save_path.stem, save_path.parent.name or None)
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
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Handle both 2D (linear) and 3D (tree) SHAP arrays
    dependence_values = np.asarray(shap_values.values)
    output_suffix = ""
    if dependence_values.ndim == 3:
        # For binary classification: select positive class (last output)
        # Shape: (n_samples, n_features, 2) → (n_samples, n_features)
        dependence_values = dependence_values[:, :, -1]
        output_suffix = " (positive class)"

    shap.dependence_plot(feature, dependence_values, X, show=False, ax=ax)
    ax.set_title(f"SHAP Dependence: {feature}{output_suffix}")
    if save_path:
        save_figure(fig, save_path.stem, save_path.parent.name or None)
    return fig


def get_top_features(shap_values: shap.Explanation, n: int = 10) -> list[str]:
    """Return the top N most important features by mean absolute SHAP value.

    Handles both 2D (linear models) and 3D (tree-based models) SHAP arrays.
    """
    values = np.abs(shap_values.values)
    # Flatten to (n_samples, n_features) by averaging over non-feature axes
    if values.ndim == 3:
        # For binary classification: (n_samples, n_features, 2) → (n_features,)
        mean_abs = values.mean(axis=(0, 2))
    else:
        # For linear models: (n_samples, n_features) → (n_features,)
        mean_abs = values.mean(axis=0)

    feature_names = shap_values.feature_names
    indices = np.argsort(mean_abs)[::-1][:n]
    return [feature_names[i] for i in indices]
