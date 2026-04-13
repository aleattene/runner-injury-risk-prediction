"""Tests for CSV loading and column renaming."""

import pytest

from src.config import ATHLETE_ID_COL, DATE_COL, INJURY_COL


class TestLoadDayData:
    """Tests for load_day_data."""

    def test_loads_and_renames_columns(self, sample_day_df):
        assert ATHLETE_ID_COL in sample_day_df.columns
        assert INJURY_COL in sample_day_df.columns
        assert DATE_COL in sample_day_df.columns
        assert "day_0_total_km" in sample_day_df.columns
        assert "day_6_perceived_recovery" in sample_day_df.columns

    def test_no_original_column_names_remain(self, sample_day_df):
        for col in sample_day_df.columns:
            assert "Athlete ID" not in col
            assert "nr. sessions" not in col

    def test_column_count(self, sample_day_df):
        assert len(sample_day_df.columns) == 73

    def test_missing_file_raises(self):
        from src.data_loading import load_day_data

        with pytest.raises(FileNotFoundError):
            load_day_data("/nonexistent/path.csv")


class TestValidateColumns:
    """Tests for _validate_columns."""

    def test_missing_columns_raises(self):
        import pandas as pd

        from src.data_loading import _validate_columns

        df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        expected = ["col_a", "col_b", "col_c"]
        with pytest.raises(ValueError, match="Missing"):
            _validate_columns(df, expected, "test_dataset")

    def test_extra_columns_raises(self):
        import pandas as pd

        from src.data_loading import _validate_columns

        df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4], "col_c": [5, 6]})
        expected = ["col_a", "col_b"]
        with pytest.raises(ValueError, match="Extra"):
            _validate_columns(df, expected, "test_dataset")

    def test_valid_columns_passes(self):
        import pandas as pd

        from src.data_loading import _validate_columns

        df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        expected = ["col_a", "col_b"]
        # Should not raise
        _validate_columns(df, expected, "test_dataset")


class TestLoadWeekData:
    """Tests for load_week_data."""

    def test_loads_and_renames_columns(self, sample_week_df):
        assert ATHLETE_ID_COL in sample_week_df.columns
        assert INJURY_COL in sample_week_df.columns
        assert "week_0_total_kms" in sample_week_df.columns
        assert "week_2_max_recovery" in sample_week_df.columns
        assert "rel_total_kms_week_0_1" in sample_week_df.columns

    def test_no_original_column_names_remain(self, sample_week_df):
        for col in sample_week_df.columns:
            assert "Athlete ID" not in col
            assert "nr. sessions" not in col

    def test_column_count(self, sample_week_df):
        assert len(sample_week_df.columns) == 72


class TestGetFeatureColumns:
    """Tests for get_feature_columns."""

    def test_excludes_metadata(self, sample_day_df):
        from src.data_loading import get_feature_columns

        feature_cols = get_feature_columns(sample_day_df)
        assert ATHLETE_ID_COL not in feature_cols
        assert INJURY_COL not in feature_cols
        assert DATE_COL not in feature_cols

    def test_day_feature_count(self, sample_day_df):
        from src.data_loading import get_feature_columns

        feature_cols = get_feature_columns(sample_day_df)
        assert len(feature_cols) == 70  # 7 days x 10 features
