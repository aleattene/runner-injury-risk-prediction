"""Tests for preprocessing modules — sentinel handling, splitting, binarization, I/O."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import ATHLETE_ID_COL, INJURY_COL, SENTINEL_VALUE


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


class TestIO:
    """Tests for preprocessing.io save/load helpers."""

    def test_save_load_splits_roundtrip(self, tmp_path: Path):
        from src.preprocessing.io import load_splits, save_splits

        train = pd.DataFrame(
            {ATHLETE_ID_COL: [1, 1, 2], INJURY_COL: [0, 1, 0], "feat": [0.1, 0.2, 0.3]}
        )
        test = pd.DataFrame({ATHLETE_ID_COL: [3], INJURY_COL: [0], "feat": [0.4]})
        save_splits(train, test, prefix="test", output_dir=tmp_path)
        loaded_train, loaded_test = load_splits(prefix="test", input_dir=tmp_path)

        pd.testing.assert_frame_equal(loaded_train, train)
        pd.testing.assert_frame_equal(loaded_test, test)

    def test_save_load_scaler_roundtrip(self, tmp_path: Path):
        from src.preprocessing.day_preprocessor import fit_scaler
        from src.preprocessing.io import load_scaler, save_scaler

        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        scaler = fit_scaler(X)
        save_scaler(scaler, name="test_scaler", output_dir=tmp_path)
        loaded = load_scaler(name="test_scaler", input_dir=tmp_path)

        np.testing.assert_array_almost_equal(scaler.mean_, loaded.mean_)
        np.testing.assert_array_almost_equal(scaler.scale_, loaded.scale_)

    def test_save_load_model_roundtrip(self, tmp_path: Path):
        from sklearn.linear_model import LogisticRegression

        from src.preprocessing.io import load_model, save_model

        model = LogisticRegression(random_state=42, max_iter=200)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        model.fit(X, y)

        save_model(model, name="test_model", output_dir=tmp_path)
        loaded = load_model(name="test_model", input_dir=tmp_path)

        np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
        np.testing.assert_array_almost_equal(
            model.predict_proba(X), loaded.predict_proba(X)
        )

    def test_load_model_missing_raises(self, tmp_path: Path):
        from src.preprocessing.io import load_model

        with pytest.raises(FileNotFoundError):
            load_model(name="nonexistent", input_dir=tmp_path)

    def test_load_model_path_traversal_raises(self, tmp_path: Path):
        from src.preprocessing.io import save_model

        with pytest.raises(ValueError, match="path separators"):
            save_model(None, name="../evil", output_dir=tmp_path)

    def test_load_missing_file_raises(self, tmp_path: Path):
        from src.preprocessing.io import load_scaler, load_splits

        with pytest.raises(FileNotFoundError):
            load_splits(prefix="nonexistent", input_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            load_scaler(name="nonexistent", input_dir=tmp_path)
