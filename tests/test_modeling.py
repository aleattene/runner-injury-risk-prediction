"""Tests for modeling modules — model factory, training, evaluation."""

from pathlib import Path
from unittest.mock import patch

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


class TestTrainFinalModel:
    """Tests for train_final_model."""

    def test_returns_fitted_model(self, small_xy_groups):
        from src.modeling.train import train_final_model

        X, y, _ = small_xy_groups
        model = create_model("logistic_regression")
        fitted = train_final_model(model, X, y)
        assert hasattr(fitted, "predict")
        # Check that model was actually fitted
        assert fitted.coef_ is not None


class TestTuneHyperparameters:
    """Tests for tune_hyperparameters."""

    def test_returns_fitted_model(self, small_xy_groups):
        from src.modeling.train import tune_hyperparameters

        X, y, groups = small_xy_groups
        model = create_model("logistic_regression")
        param_dist = {"C": [0.1, 1.0]}
        best_model = tune_hyperparameters(
            model, X, y, groups, param_dist, n_iter=2, n_splits=2
        )
        assert hasattr(best_model, "predict")
        assert best_model.coef_ is not None


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

    def test_metric_f1(self):
        from src.modeling.evaluate import find_optimal_threshold

        y_true = np.array([0, 0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9])
        threshold = find_optimal_threshold(y_true, y_prob, metric="f1")
        assert 0.0 < threshold < 1.0

    def test_metric_recall_with_precision_filter(self):
        from src.modeling.evaluate import find_optimal_threshold

        # Create imbalanced data where many thresholds fail precision < 0.05
        y_true = np.array([0] * 100 + [1] * 5)
        # High probs for some negatives, low for positives
        y_prob = np.concatenate(
            [
                np.linspace(0.1, 0.9, 100),  # negatives spread across range
                np.array([0.3, 0.35, 0.4, 0.45, 0.5]),  # positives low
            ]
        )
        threshold = find_optimal_threshold(y_true, y_prob, metric="recall")
        assert 0.0 < threshold < 1.0

    def test_invalid_metric_raises(self):
        from src.modeling.evaluate import find_optimal_threshold

        y_true = np.array([0, 1])
        y_prob = np.array([0.3, 0.7])
        with pytest.raises(ValueError, match="Unsupported metric"):
            find_optimal_threshold(y_true, y_prob, metric="invalid")


class TestPlotRocCurves:
    """Tests for plot_roc_curves."""

    def test_returns_figure(self):
        from src.modeling.evaluate import plot_roc_curves

        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.9])
        results = {"Model A": (y_true, y_prob)}
        fig = plot_roc_curves(results)
        assert fig is not None
        assert hasattr(fig, "savefig")

    def test_with_save_path(self):
        from src.modeling.evaluate import plot_roc_curves

        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.9])
        results = {"Model A": (y_true, y_prob)}
        with patch("src.modeling.evaluate.save_figure") as mock_save:
            fig = plot_roc_curves(results, save_path=Path("dummy.png"))
            assert fig is not None
            mock_save.assert_called_once()


class TestPlotPrCurves:
    """Tests for plot_pr_curves."""

    def test_returns_figure(self):
        from src.modeling.evaluate import plot_pr_curves

        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.9])
        results = {"Model A": (y_true, y_prob)}
        fig = plot_pr_curves(results)
        assert fig is not None
        assert hasattr(fig, "savefig")

    def test_with_save_path(self):
        from src.modeling.evaluate import plot_pr_curves

        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.9])
        results = {"Model A": (y_true, y_prob)}
        with patch("src.modeling.evaluate.save_figure") as mock_save:
            fig = plot_pr_curves(results, save_path=Path("dummy.png"))
            assert fig is not None
            mock_save.assert_called_once()


class TestPlotConfusionMatrix:
    """Tests for plot_confusion_matrix."""

    def test_returns_figure(self):
        from src.modeling.evaluate import plot_confusion_matrix

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        fig = plot_confusion_matrix(y_true, y_pred)
        assert fig is not None
        assert hasattr(fig, "savefig")

    def test_with_save_path(self):
        from src.modeling.evaluate import plot_confusion_matrix

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        with patch("src.modeling.evaluate.save_figure") as mock_save:
            fig = plot_confusion_matrix(y_true, y_pred, save_path=Path("dummy.png"))
            assert fig is not None
            mock_save.assert_called_once()

    def test_with_custom_title(self):
        from src.modeling.evaluate import plot_confusion_matrix

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        fig = plot_confusion_matrix(y_true, y_pred, title="Custom Title")
        assert fig is not None


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
