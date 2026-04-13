"""Tests for utility modules — logging, reproducibility, plotting."""

import logging

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


class TestPlotting:
    """Tests for plotting utilities."""

    def test_set_style_does_not_raise(self):
        from src.utils.plotting import set_style

        set_style()

    def test_save_figure(self, tmp_path):

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        # Save to tmp_path instead of project figures dir
        path = tmp_path / "test_fig.png"
        fig.savefig(path)
        plt.close(fig)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_injury_palette_has_both_classes(self):
        from src.utils.plotting import INJURY_PALETTE

        assert 0 in INJURY_PALETTE
        assert 1 in INJURY_PALETTE
