"""Lightweight smoke tests — verify imports and basic module structure."""

import importlib


def test_src_package_imports():
    """All src subpackages should be importable."""
    modules = [
        "src",
        "src.config",
        "src.data_loading",
        "src.preprocessing",
        "src.preprocessing.common",
        "src.preprocessing.day_preprocessor",
        "src.preprocessing.week_preprocessor",
        "src.modeling",
        "src.modeling.models",
        "src.modeling.train",
        "src.modeling.evaluate",
        "src.interpretability",
        "src.interpretability.shap_analysis",
        "src.fairness",
        "src.fairness.audit",
        "src.utils",
        "src.utils.logging_config",
        "src.utils.reproducibility",
        "src.utils.plotting",
    ]
    for mod in modules:
        importlib.import_module(mod)


def test_config_constants():
    """Config constants should have expected types and values."""
    from src.config import (
        DAY_FEATURES,
        N_CV_FOLDS,
        N_DAY_BLOCKS,
        N_WEEK_BLOCKS,
        PROJECT_ROOT,
        RANDOM_SEED,
        TEST_SIZE,
        WEEK_FEATURES,
        WEEK_RELATIVE_FEATURES,
    )

    assert PROJECT_ROOT.exists()
    assert RANDOM_SEED == 42
    assert N_CV_FOLDS == 5
    assert 0 < TEST_SIZE < 1
    assert N_DAY_BLOCKS == 7
    assert N_WEEK_BLOCKS == 3
    assert len(DAY_FEATURES) == 10
    assert len(WEEK_FEATURES) == 22
    assert len(WEEK_RELATIVE_FEATURES) == 3


def test_supported_models_list():
    """Model factory should expose a list of supported model names."""
    from src.modeling.models import SUPPORTED_MODELS

    assert "logistic_regression" in SUPPORTED_MODELS
    assert "random_forest" in SUPPORTED_MODELS
    assert "xgboost" in SUPPORTED_MODELS
