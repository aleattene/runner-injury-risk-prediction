"""Model factory — creates unfitted estimators for the modeling pipeline."""

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import RANDOM_SEED

SUPPORTED_MODELS: list[str] = [
    "dummy",
    "logistic_regression",
    "random_forest",
    "xgboost",
]


def create_model(
    name: str,
    imbalance_ratio: float | None = None,
    random_state: int = RANDOM_SEED,
) -> BaseEstimator:
    """Create an unfitted classifier by name.

    Parameters
    ----------
    name : str
        One of "dummy", "logistic_regression", "random_forest", "xgboost".
    imbalance_ratio : float or None
        Ratio of negative to positive samples, used as ``scale_pos_weight``
        in XGBoost. Has no effect on logistic regression or random forest,
        which always use ``class_weight="balanced"``.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    BaseEstimator
        Unfitted scikit-learn compatible estimator.
    """
    if name == "dummy":
        return DummyClassifier(strategy="stratified", random_state=random_state)
    elif name == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
        )
    elif name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    elif name == "xgboost":
        return XGBClassifier(
            scale_pos_weight=1.0 if imbalance_ratio is None else imbalance_ratio,
            n_estimators=200,
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model '{name}'. Supported: {SUPPORTED_MODELS}")
