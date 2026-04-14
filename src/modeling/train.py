"""Model training — cross-validation with GroupKFold and hyperparameter tuning."""

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_validate

from src.config import N_CV_FOLDS, RANDOM_SEED

logger = logging.getLogger(__name__)

CV_SCORING: dict[str, str] = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
}


def cross_validate_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = N_CV_FOLDS,
) -> dict[str, np.ndarray]:
    """Run cross-validation with GroupKFold, returning per-fold metrics.

    Returns
    -------
    dict[str, np.ndarray]
        Keys include ``test_{metric}``, ``fit_time``, and ``score_time``.
        Values are per-fold arrays.
    """
    cv = GroupKFold(n_splits=n_splits)
    logger.info(
        "Cross-validating %s with %d-fold GroupKFold", type(model).__name__, n_splits
    )
    results = cross_validate(
        clone(model),
        X,
        y,
        groups=groups,
        cv=cv,
        scoring=CV_SCORING,
        return_train_score=False,
        n_jobs=-1,
    )
    for metric in CV_SCORING:
        key = f"test_{metric}"
        scores = results[key]
        logger.info("  %s: %.4f +/- %.4f", metric, scores.mean(), scores.std())
    return results


def train_final_model(
    model: BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series
) -> BaseEstimator:
    """Fit a model on the full training set."""
    logger.info("Training final %s on %d samples", type(model).__name__, len(X_train))
    model.fit(X_train, y_train)
    return model


def tune_hyperparameters(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    param_distributions: dict,
    n_iter: int = 50,
    n_splits: int = N_CV_FOLDS,
    random_state: int = RANDOM_SEED,
) -> BaseEstimator:
    """Tune hyperparameters with RandomizedSearchCV and GroupKFold.

    Returns
    -------
    BaseEstimator
        The best estimator found.
    """
    cv = GroupKFold(n_splits=n_splits)
    search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    logger.info("Tuning %s with %d iterations", type(model).__name__, n_iter)
    search.fit(X, y, groups=groups)
    logger.info("Best AUC-ROC: %.4f", search.best_score_)
    logger.info("Best params: %s", search.best_params_)
    return search.best_estimator_
