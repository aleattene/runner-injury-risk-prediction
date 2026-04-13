"""Tests for fairness audit module."""

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
