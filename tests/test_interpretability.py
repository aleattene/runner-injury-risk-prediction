"""Tests for SHAP interpretability module."""

from pathlib import Path
from unittest.mock import patch

from src.modeling.models import create_model


class TestComputeShapValues:
    """Tests for compute_shap_values."""

    def test_shap_values_shape_matches_input(self, small_xy_groups):
        from src.interpretability.shap_analysis import compute_shap_values

        X, y, _ = small_xy_groups
        model = create_model("random_forest")
        model.fit(X, y)
        shap_values = compute_shap_values(model, X)
        # Tree models return (n_samples, n_features, n_classes) for binary clf
        assert shap_values.shape == (X.shape[0], X.shape[1], 2)

    def test_shap_with_logistic_regression(self, small_xy_groups):
        from src.interpretability.shap_analysis import compute_shap_values

        X, y, _ = small_xy_groups
        model = create_model("logistic_regression")
        model.fit(X, y)
        shap_values = compute_shap_values(model, X)
        # Linear models return (n_samples, n_features) shape
        assert shap_values.shape == X.shape


class TestGetTopFeatures:
    """Tests for get_top_features."""

    def test_returns_correct_count(self, small_xy_groups):
        from src.interpretability.shap_analysis import (
            compute_shap_values,
            get_top_features,
        )

        X, y, _ = small_xy_groups
        model = create_model("random_forest")
        model.fit(X, y)
        shap_values = compute_shap_values(model, X)
        top = get_top_features(shap_values, n=3)
        assert len(top) == 3
        assert all(isinstance(f, str) for f in top)


class TestPlotShapSummary:
    """Tests for plot_shap_summary."""

    def test_returns_figure(self, small_xy_groups):
        from src.interpretability.shap_analysis import (
            compute_shap_values,
            plot_shap_summary,
        )

        X, y, _ = small_xy_groups
        model = create_model("logistic_regression")
        model.fit(X, y)
        shap_values = compute_shap_values(model, X)
        fig = plot_shap_summary(shap_values, X)
        assert fig is not None
        assert hasattr(fig, "savefig")

    def test_with_save_path(self, small_xy_groups):
        from src.interpretability.shap_analysis import (
            compute_shap_values,
            plot_shap_summary,
        )

        X, y, _ = small_xy_groups
        model = create_model("logistic_regression")
        model.fit(X, y)
        shap_values = compute_shap_values(model, X)
        with patch("src.interpretability.shap_analysis.save_figure") as mock_save:
            fig = plot_shap_summary(shap_values, X, save_path=Path("dummy.png"))
            assert fig is not None
            mock_save.assert_called_once()


class TestPlotShapDependence:
    """Tests for plot_shap_dependence."""

    def test_returns_figure(self, small_xy_groups):
        from src.interpretability.shap_analysis import (
            compute_shap_values,
            plot_shap_dependence,
        )

        X, y, _ = small_xy_groups
        model = create_model("logistic_regression")
        model.fit(X, y)
        shap_values = compute_shap_values(model, X)
        feature = X.columns[0]
        fig = plot_shap_dependence(shap_values, X, feature)
        assert fig is not None
        assert hasattr(fig, "savefig")

    def test_with_save_path(self, small_xy_groups):
        from src.interpretability.shap_analysis import (
            compute_shap_values,
            plot_shap_dependence,
        )

        X, y, _ = small_xy_groups
        model = create_model("logistic_regression")
        model.fit(X, y)
        shap_values = compute_shap_values(model, X)
        feature = X.columns[0]
        with patch("src.interpretability.shap_analysis.save_figure") as mock_save:
            fig = plot_shap_dependence(
                shap_values, X, feature, save_path=Path("dummy.png")
            )
            assert fig is not None
            mock_save.assert_called_once()
