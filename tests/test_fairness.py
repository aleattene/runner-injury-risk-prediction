"""Tests for fairness audit module."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.config import ATHLETE_ID_COL, INJURY_COL


class TestCreateAthleteGroups:
    """Tests for create_athlete_groups."""

    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                ATHLETE_ID_COL: [0, 0, 0, 1, 1, 1, 2, 2, 2],
                INJURY_COL: [0, 0, 1, 0, 0, 0, 0, 1, 1],
                "day_0_total_km": [10, 12, 11, 2, 3, 1, 5, 6, 4],
            }
        )

    def test_volume_method(self):
        from src.fairness.audit import create_athlete_groups

        df = self._make_df()
        groups = create_athlete_groups(
            df, method="volume", feature_col="day_0_total_km"
        )
        assert len(groups) == len(df)
        assert set(groups.unique()) == {"high_volume", "low_volume"}

    def test_injury_history_method(self):
        from src.fairness.audit import create_athlete_groups

        df = self._make_df()
        groups = create_athlete_groups(df, method="injury_history")
        assert set(groups.unique()) == {"ever_injured", "never_injured"}
        # Athlete 1 never injured
        assert groups.iloc[3] == "never_injured"

    def test_data_density_method(self):
        from src.fairness.audit import create_athlete_groups

        df = self._make_df()
        groups = create_athlete_groups(df, method="data_density")
        assert len(groups) == len(df)

    def test_volume_requires_feature_col(self):
        from src.fairness.audit import create_athlete_groups

        df = self._make_df()
        with pytest.raises(ValueError, match="feature_col required"):
            create_athlete_groups(df, method="volume")

    def test_unknown_method_raises(self):
        from src.fairness.audit import create_athlete_groups

        df = self._make_df()
        with pytest.raises(ValueError, match="Unknown method"):
            create_athlete_groups(df, method="nonexistent")


class TestComputeGroupMetrics:
    """Tests for compute_group_metrics."""

    def test_one_row_per_group(self):
        from src.fairness.audit import compute_group_metrics

        y_true = np.array([0, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.6, 0.8, 0.2, 0.4, 0.9])
        groups = pd.Series(["A", "A", "A", "B", "B", "B"])
        result = compute_group_metrics(y_true, y_pred, y_prob, groups)
        assert len(result) == 2
        assert set(result["group"]) == {"A", "B"}

    def test_metrics_columns_present(self):
        from src.fairness.audit import compute_group_metrics

        y_true = np.array([0, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.6, 0.8, 0.2, 0.4, 0.9])
        groups = pd.Series(["A", "A", "A", "B", "B", "B"])
        result = compute_group_metrics(y_true, y_pred, y_prob, groups)
        for col in ["recall", "precision", "f1", "fpr", "auc_roc", "support"]:
            assert col in result.columns


class TestComputeDisparityRatios:
    """Tests for compute_disparity_ratios."""

    def test_reference_group_ratio_is_one(self):
        from src.fairness.audit import compute_disparity_ratios

        group_metrics = pd.DataFrame(
            {
                "group": ["A", "B"],
                "recall": [0.8, 0.6],
                "precision": [0.7, 0.5],
                "f1": [0.75, 0.55],
                "fpr": [0.1, 0.2],
                "auc_roc": [0.85, 0.75],
            }
        )
        result = compute_disparity_ratios(group_metrics, reference_group="A")
        ref_row = result[result["group"] == "A"]
        assert ref_row["recall_ratio"].iloc[0] == pytest.approx(1.0)

    def test_unknown_reference_raises(self):
        from src.fairness.audit import compute_disparity_ratios

        group_metrics = pd.DataFrame({"group": ["A"], "recall": [0.8]})
        with pytest.raises(ValueError, match="not found"):
            compute_disparity_ratios(group_metrics, reference_group="Z")

    def test_reference_metric_is_zero_returns_nan(self):
        from src.fairness.audit import compute_disparity_ratios

        # Reference group A has recall=0
        group_metrics = pd.DataFrame(
            {
                "group": ["A", "B"],
                "recall": [0.0, 0.6],
                "precision": [0.7, 0.5],
                "f1": [0.75, 0.55],
                "fpr": [0.1, 0.2],
                "auc_roc": [0.85, 0.75],
            }
        )
        result = compute_disparity_ratios(group_metrics, reference_group="A")
        # When ref_val is 0, ratio should be NaN
        assert pd.isna(result.loc[result["group"] == "B", "recall_ratio"].iloc[0])

    def test_reference_metric_is_nan_returns_nan(self):
        from src.fairness.audit import compute_disparity_ratios

        # Reference group A has NaN recall
        group_metrics = pd.DataFrame(
            {
                "group": ["A", "B"],
                "recall": [np.nan, 0.6],
                "precision": [0.7, 0.5],
                "f1": [0.75, 0.55],
                "fpr": [0.1, 0.2],
                "auc_roc": [0.85, 0.75],
            }
        )
        result = compute_disparity_ratios(group_metrics, reference_group="A")
        # When ref_val is NaN, ratio should be NaN
        assert pd.isna(result.loc[result["group"] == "B", "recall_ratio"].iloc[0])


# ---------------------------------------------------------------------------
# Visualization function tests
# ---------------------------------------------------------------------------


def _make_metrics_df() -> pd.DataFrame:
    """Minimal metrics DataFrame for plot tests."""
    return pd.DataFrame(
        {
            "group": ["A", "B"],
            "support": [100, 80],
            "n_positive": [10, 8],
            "prevalence": [0.10, 0.10],
            "recall": [0.80, 0.60],
            "precision": [0.70, 0.50],
            "f1": [0.75, 0.55],
            "fpr": [0.10, 0.20],
            "auc_roc": [0.85, 0.75],
        }
    )


class TestPlotGroupMetricsBars:
    """Tests for plot_group_metrics_bars."""

    def test_returns_figure(self):
        from src.fairness.audit import plot_group_metrics_bars

        fig = plot_group_metrics_bars(_make_metrics_df())
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_saves_figure(self):
        from unittest.mock import patch

        from src.fairness.audit import plot_group_metrics_bars

        save_path = Path("fairness/07_test_metrics")
        with patch("src.fairness.audit.save_figure") as mock_save:
            fig = plot_group_metrics_bars(_make_metrics_df(), save_path=save_path)
            try:
                mock_save.assert_called_once_with(
                    fig, "07_test_metrics", "fairness", close=False
                )
            finally:
                plt.close(fig)


class TestPlotDisparityRatios:
    """Tests for plot_disparity_ratios."""

    def test_returns_figure(self):
        from src.fairness.audit import compute_disparity_ratios, plot_disparity_ratios

        disp = compute_disparity_ratios(_make_metrics_df(), reference_group="A")
        fig = plot_disparity_ratios(disp)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_reference_line_present(self):
        from src.fairness.audit import compute_disparity_ratios, plot_disparity_ratios

        disp = compute_disparity_ratios(_make_metrics_df(), reference_group="A")
        fig = plot_disparity_ratios(disp)
        try:
            ax = fig.axes[0]
            vlines = [
                line
                for line in ax.get_lines()
                if hasattr(line, "get_xdata") and 1.0 in line.get_xdata()
            ]
            assert len(vlines) > 0, "Expected a vertical reference line at 1.0"
        finally:
            plt.close(fig)


class TestPlotFairnessSummaryHeatmap:
    """Tests for plot_fairness_summary_heatmap."""

    def test_returns_figure(self):
        from src.fairness.audit import plot_fairness_summary_heatmap

        all_metrics = {"volume": _make_metrics_df(), "density": _make_metrics_df()}
        fig = plot_fairness_summary_heatmap(all_metrics)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_saves_figure(self):
        from unittest.mock import patch

        from src.fairness.audit import plot_fairness_summary_heatmap

        all_metrics = {"volume": _make_metrics_df()}
        save_path = Path("fairness/07_test_heatmap")
        with patch("src.fairness.audit.save_figure") as mock_save:
            fig = plot_fairness_summary_heatmap(all_metrics, save_path=save_path)
            try:
                mock_save.assert_called_once_with(
                    fig, "07_test_heatmap", "fairness", close=False
                )
            finally:
                plt.close(fig)
