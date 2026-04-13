"""Tests for utility modules — logging, reproducibility, plotting."""

import logging
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np


class TestLoggingConfig:
    """Tests for logging_config.setup_logging."""

    def test_sets_log_level(self):
        from src.utils.logging_config import setup_logging

        setup_logging(level=logging.DEBUG)
        assert logging.getLogger().level == logging.DEBUG

        setup_logging(level=logging.INFO)
        assert logging.getLogger().level == logging.INFO


class TestReproducibility:
    """Tests for reproducibility.set_global_seed."""

    def test_numpy_deterministic_after_seed(self):
        rng_a = np.random.default_rng(42)
        a = rng_a.random(5)
        rng_b = np.random.default_rng(42)
        b = rng_b.random(5)
        np.testing.assert_array_equal(a, b)

    def test_set_global_seed_with_custom_seed(self):
        from src.utils.reproducibility import set_global_seed

        set_global_seed(123)
        # Just verify it runs without error

    def test_set_global_seed_with_default(self):
        from src.utils.reproducibility import set_global_seed

        set_global_seed()
        # Just verify it runs without error


class TestPlotting:
    """Tests for plotting utilities."""

    def test_set_style_does_not_raise(self):
        from src.utils.plotting import set_style

        set_style()

    def test_save_figure_without_subdir(self, tmp_path):
        from src.utils.plotting import save_figure

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        with patch("src.utils.plotting.FIGURES_DIR", tmp_path):
            path = save_figure(fig, "test_fig")
            assert path.exists()
            assert path.stat().st_size > 0
            assert path.name == "test_fig.png"

    def test_save_figure_with_subdir(self, tmp_path):
        from src.utils.plotting import save_figure

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        with patch("src.utils.plotting.FIGURES_DIR", tmp_path):
            path = save_figure(fig, "test_fig", subdir="subfolder")
            assert path.exists()
            assert path.stat().st_size > 0
            assert path.parent.name == "subfolder"

    def test_injury_palette_has_both_classes(self):
        from src.utils.plotting import INJURY_PALETTE

        assert 0 in INJURY_PALETTE
        assert 1 in INJURY_PALETTE
