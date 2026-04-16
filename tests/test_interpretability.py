"""Tests for SHAP interpretability module."""

from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import shap

from src.modeling.models import create_model


class TestExtractPositiveClass:
    """Tests for _extract_positive_class with constructed Explanation objects."""

    def test_2d_values_returned_unchanged(self):
        from src.interpretability.shap_analysis import _extract_positive_class

        values = np.random.default_rng(0).standard_normal((10, 4))
        base_values = np.zeros(10)
        explanation = shap.Explanation(
            values=values,
            base_values=base_values,
            data=np.zeros((10, 4)),
            feature_names=["a", "b", "c", "d"],
        )
        result = _extract_positive_class(explanation)
        np.testing.assert_array_equal(result.values, values)

    def test_3d_values_with_2d_base_values(self):
        from src.interpretability.shap_analysis import _extract_positive_class

        rng = np.random.default_rng(1)
        values_3d = rng.standard_normal((10, 4, 2))
        base_2d = rng.standard_normal((10, 2))
        explanation = shap.Explanation(
            values=values_3d,
            base_values=base_2d,
            data=np.zeros((10, 4)),
            feature_names=["a", "b", "c", "d"],
        )
        result = _extract_positive_class(explanation)
        assert result.values.shape == (10, 4)
        np.testing.assert_array_equal(result.values, values_3d[:, :, -1])
        np.testing.assert_array_equal(result.base_values, base_2d[:, -1])

    def test_3d_values_with_1d_base_values_n_outputs(self):
        from src.interpretability.shap_analysis import _extract_positive_class

        rng = np.random.default_rng(2)
        values_3d = rng.standard_normal((10, 4, 2))
        base_1d = np.array([0.3, 0.7])  # shape (n_outputs,)
        explanation = shap.Explanation(
            values=values_3d,
            base_values=base_1d,
            data=np.zeros((10, 4)),
            feature_names=["a", "b", "c", "d"],
        )
        result = _extract_positive_class(explanation)
        assert result.values.shape == (10, 4)
        assert result.base_values == pytest.approx(0.7)

    def test_3d_values_with_scalar_base_values(self):
        from src.interpretability.shap_analysis import _extract_positive_class

        rng = np.random.default_rng(3)
        values_3d = rng.standard_normal((10, 4, 2))
        base_scalar = np.float64(0.5)
        explanation = shap.Explanation(
            values=values_3d,
            base_values=base_scalar,
            data=np.zeros((10, 4)),
            feature_names=["a", "b", "c", "d"],
        )
        result = _extract_positive_class(explanation)
        assert result.values.shape == (10, 4)
        np.testing.assert_array_equal(result.base_values, np.full(10, 0.5))

    def test_3d_values_with_incompatible_base_values_raises(self):
        from src.interpretability.shap_analysis import _extract_positive_class

        rng = np.random.default_rng(4)
        values_3d = rng.standard_normal((10, 4, 2))
        base_bad = rng.standard_normal((5,))  # 1D but shape != n_outputs
        explanation = shap.Explanation(
            values=values_3d,
            base_values=base_bad,
            data=np.zeros((10, 4)),
            feature_names=["a", "b", "c", "d"],
        )
        with pytest.raises(ValueError, match="Unexpected base_values shape"):
            _extract_positive_class(explanation)


class TestComputeShapValues:
    """Tests for compute_shap_values."""

    def test_shap_values_shape_matches_input(self, small_xy_groups):
        from src.interpretability.shap_analysis import compute_shap_values

        X, y, _ = small_xy_groups
        model = create_model("random_forest")
        model.fit(X, y)
        shap_values = compute_shap_values(model, X)
        # Tree models may return 2D or 3D SHAP arrays depending on version.
        # Validate the essential invariants: correct sample and feature count.
        values = shap_values.values
        assert values.ndim in (2, 3), f"Expected 2D or 3D, got {values.ndim}D"
        assert values.shape[0] == X.shape[0], "Mismatch in number of samples"
        assert values.shape[1] == X.shape[1], "Mismatch in number of features"
        if values.ndim == 3:
            assert values.shape[2] == 2, "Expected binary classification (2 classes)"

    def test_shap_with_logistic_regression(self, small_xy_groups):
        from src.interpretability.shap_analysis import compute_shap_values

        X, y, _ = small_xy_groups
        model = create_model("logistic_regression")
        model.fit(X, y)
        shap_values = compute_shap_values(model, X)
        # Linear models return (n_samples, n_features) shape
        assert shap_values.values.shape == X.shape


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


class TestPlotShapWaterfall:
    """Tests for plot_shap_waterfall."""

    def test_returns_figure(self, small_xy_groups):
        from src.interpretability.shap_analysis import (
            compute_shap_values,
            plot_shap_waterfall,
        )

        X, y, _ = small_xy_groups
        model = create_model("random_forest")
        model.fit(X, y)
        shap_values = compute_shap_values(model, X)
        fig = plot_shap_waterfall(shap_values, index=0)
        assert fig is not None
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_with_save_path(self, small_xy_groups):
        from src.interpretability.shap_analysis import (
            compute_shap_values,
            plot_shap_waterfall,
        )

        X, y, _ = small_xy_groups
        model = create_model("random_forest")
        model.fit(X, y)
        shap_values = compute_shap_values(model, X)
        with patch("src.interpretability.shap_analysis.save_figure") as mock_save:
            fig = plot_shap_waterfall(shap_values, index=0, save_path=Path("dummy.png"))
            assert fig is not None
            mock_save.assert_called_once()
        plt.close(fig)


class TestGetShapImportanceDict:
    """Tests for get_shap_importance_dict."""

    def test_returns_dict_with_all_features(self, small_xy_groups):
        from src.interpretability.shap_analysis import (
            compute_shap_values,
            get_shap_importance_dict,
        )

        X, y, _ = small_xy_groups
        model = create_model("random_forest")
        model.fit(X, y)
        shap_values = compute_shap_values(model, X)
        importance = get_shap_importance_dict(shap_values)
        assert isinstance(importance, dict)
        assert len(importance) == X.shape[1]
        assert all(v >= 0 for v in importance.values())


class TestCompareFeatureImportance:
    """Tests for compare_feature_importance."""

    def test_returns_figure(self, small_xy_groups):
        from src.interpretability.shap_analysis import compare_feature_importance

        X, _, _ = small_xy_groups
        features = list(X.columns)
        shap_imp = {f: float(i + 1) for i, f in enumerate(features)}
        builtin_imp = {f: float(len(features) - i) for i, f in enumerate(features)}
        fig = compare_feature_importance(shap_imp, builtin_imp, top_n=3)
        assert fig is not None
        assert hasattr(fig, "savefig")
        plt.close(fig)
