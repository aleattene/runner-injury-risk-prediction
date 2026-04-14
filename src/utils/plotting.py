"""Shared plot helpers — consistent style, save, and color palettes."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR

# --- Project color palette ---
PALETTE: dict[str, str] = {
    "primary": "#2196F3",
    "secondary": "#FF9800",
    "positive": "#4CAF50",
    "negative": "#F44336",
    "neutral": "#9E9E9E",
}

INJURY_PALETTE: dict[int, str] = {
    0: PALETTE["primary"],
    1: PALETTE["negative"],
}


def set_style() -> None:
    """Apply consistent plot style across all notebooks."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def save_figure(
    fig: plt.Figure, name: str, subdir: str | None = None, close: bool = True
) -> Path:
    """Save a figure to reports/figures/ with consistent settings.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    name : str
        File name (without extension).
    subdir : str or None
        Optional subdirectory inside figures/.
    close : bool
        If True, close the figure after saving (default True).

    Returns
    -------
    Path
        Path to the saved file.
    """
    target_dir = FIGURES_DIR / subdir if subdir else FIGURES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{name}.png"
    fig.savefig(path)
    if close:
        plt.close(fig)
    return path
