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
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    if save_path:
        save_figure(plt.gcf(), save_path.stem, save_path.parent.name or None)
    return plt.gcf()


def plot_shap_dependence(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    feature: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """Create a SHAP dependence plot for a single feature."""
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.dependence_plot(feature, shap_values.values, X, show=False, ax=ax)
    ax.set_title(f"SHAP Dependence: {feature}")
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
