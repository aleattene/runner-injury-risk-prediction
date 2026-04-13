"""Tests for SHAP interpretability module."""

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
