"""Tests for preprocessing modules — sentinel handling, splitting, binarization."""

import pandas as pd
import pytest

from src.config import ATHLETE_ID_COL, SENTINEL_VALUE


class TestSentinelHandling:
    """Tests for day_preprocessor.handle_sentinel_values."""

    def test_replaces_sentinel_with_zero(self):
        from src.preprocessing.day_preprocessor import handle_sentinel_values

        df = pd.DataFrame(
            {"a": [1.0, SENTINEL_VALUE, 3.0], "b": [SENTINEL_VALUE, 2.0, 0.0]}
        )
        result = handle_sentinel_values(df)
        assert (result["a"] == [1.0, 0.0, 3.0]).all()
        assert (result["b"] == [0.0, 2.0, 0.0]).all()

    def test_no_sentinels_unchanged(self):
        from src.preprocessing.day_preprocessor import handle_sentinel_values

        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = handle_sentinel_values(df)
        pd.testing.assert_frame_equal(result, df)


class TestTrainTestSplit:
    """Tests for common.split_train_test."""

    def test_athletes_dont_leak(self, sample_day_df):
        from src.preprocessing.common import split_train_test

        train_df, test_df = split_train_test(sample_day_df)
        train_athletes = set(train_df[ATHLETE_ID_COL].unique())
        test_athletes = set(test_df[ATHLETE_ID_COL].unique())
        assert train_athletes.isdisjoint(
            test_athletes
        ), "Athletes leaked between splits"

    def test_all_rows_preserved(self, sample_day_df):
        from src.preprocessing.common import split_train_test

        train_df, test_df = split_train_test(sample_day_df)
        assert len(train_df) + len(test_df) == len(sample_day_df)

    def test_deterministic(self, sample_day_df):
        from src.preprocessing.common import split_train_test

        train1, _ = split_train_test(sample_day_df)
        train2, _ = split_train_test(sample_day_df)
        assert set(train1[ATHLETE_ID_COL].unique()) == set(
            train2[ATHLETE_ID_COL].unique()
        )


class TestBinarizeTarget:
    """Tests for week_preprocessor.binarize_target."""

    @pytest.mark.parametrize("threshold", [0.1, 0.25, 0.5])
    def test_output_is_binary(self, threshold):
        from src.preprocessing.week_preprocessor import binarize_target

        y = pd.Series([0.0, 0.05, 0.3, 0.5, 1.0, 1.5])
        result = binarize_target(y, threshold=threshold)
        assert set(result.unique()).issubset({0, 1})

    def test_threshold_05(self):
        from src.preprocessing.week_preprocessor import binarize_target

        y = pd.Series([0.0, 0.49, 0.5, 0.51, 1.0])
        result = binarize_target(y, threshold=0.5)
        expected = pd.Series([0, 0, 1, 1, 1])
        pd.testing.assert_series_equal(result, expected)


class TestScaler:
    """Tests for day_preprocessor scaler functions."""

    def test_fit_scaler_returns_fitted(self):
        from src.preprocessing.day_preprocessor import fit_scaler

        X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        scaler = fit_scaler(X_train)
        assert hasattr(scaler, "mean_")
        assert len(scaler.mean_) == 2

    def test_transform_preserves_shape(self):
        from src.preprocessing.day_preprocessor import fit_scaler, transform_scaled

        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        scaler = fit_scaler(X)
        result = transform_scaled(X, scaler, ["a", "b"])
        assert result.shape == X.shape


class TestGetGroupKfold:
    """Tests for common.get_group_kfold."""

    def test_returns_groupkfold_splitter(self):
        from src.preprocessing.common import get_group_kfold

        splitter = get_group_kfold(n_splits=3)
        assert splitter.n_splits == 3
        assert hasattr(splitter, "split")

    def test_default_n_splits(self):
        from src.preprocessing.common import get_group_kfold

        splitter = get_group_kfold()
        # Should use N_CV_FOLDS from config (5 by default)
        assert splitter.n_splits == 5


class TestGetFeatureTargetGroups:
    """Tests for common.get_feature_target_groups."""

    def test_correct_shapes(self, sample_day_df):
        from src.data_loading import get_feature_columns
        from src.preprocessing.common import get_feature_target_groups

        feature_cols = get_feature_columns(sample_day_df)
        X, y, groups = get_feature_target_groups(sample_day_df, feature_cols)
        assert len(X) == len(sample_day_df)
        assert len(y) == len(sample_day_df)
        assert len(groups) == len(sample_day_df)
        assert X.shape[1] == len(feature_cols)
