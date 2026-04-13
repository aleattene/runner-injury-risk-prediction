"""Tests for modeling modules — model factory, training, evaluation."""

import numpy as np
import pytest

from src.modeling.models import SUPPORTED_MODELS, create_model


class TestCreateModel:
    """Tests for model factory."""

    @pytest.mark.parametrize("name", SUPPORTED_MODELS)
    def test_returns_unfitted_estimator(self, name):
        model = create_model(name, imbalance_ratio=10.0)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            create_model("nonexistent_model")

    def test_xgboost_scale_pos_weight(self):
        model = create_model("xgboost", imbalance_ratio=72.0)
        assert model.get_params()["scale_pos_weight"] == pytest.approx(72.0)


class TestCrossValidateModel:
    """Tests for cross_validate_model."""

    def test_returns_expected_keys(self, small_xy_groups):
        from src.modeling.train import cross_validate_model

        X, y, groups = small_xy_groups
        model = create_model("logistic_regression")
        results = cross_validate_model(model, X, y, groups, n_splits=3)
        assert "test_roc_auc" in results
        assert "test_recall" in results
        assert len(results["test_roc_auc"]) == 3


class TestComputeMetrics:
    """Tests for compute_metrics."""

    def test_metrics_in_valid_range(self):
        from src.modeling.evaluate import compute_metrics

        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 0])
        y_prob = np.array([0.1, 0.2, 0.6, 0.8, 0.3])
        metrics = compute_metrics(y_true, y_pred, y_prob)

        assert 0 <= metrics["auc_roc"] <= 1
        assert 0 <= metrics["auc_pr"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert 0 <= metrics["specificity"] <= 1
        assert 0 <= metrics["brier_score"] <= 1

    def test_perfect_predictions(self):
        from src.modeling.evaluate import compute_metrics

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        metrics = compute_metrics(y_true, y_pred, y_prob)

        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)


class TestFindOptimalThreshold:
    """Tests for find_optimal_threshold."""

    def test_returns_float_in_range(self):
        from src.modeling.evaluate import find_optimal_threshold

        y_true = np.array([0, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.4, 0.7, 0.9])
        threshold = find_optimal_threshold(y_true, y_prob)
        assert 0.0 < threshold < 1.0


class TestComparisonTable:
    """Tests for create_comparison_table."""

    def test_shape(self):
        from src.modeling.evaluate import create_comparison_table

        results = {
            "ModelA": {"auc_roc": 0.7, "recall": 0.5},
            "ModelB": {"auc_roc": 0.8, "recall": 0.6},
        }
        table = create_comparison_table(results)
        assert table.shape == (2, 2)
        assert list(table.index) == ["ModelA", "ModelB"]
