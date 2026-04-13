"""Centralized configuration — paths, constants, and column names.

No environment variables. All paths are relative to PROJECT_ROOT.
"""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
DATA_SAMPLE_DIR: Path = PROJECT_ROOT / "data_sample"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# --- File names ---
DAY_CSV: str = "day_approach_maskedID_timeseries.csv"
WEEK_CSV: str = "week_approach_maskedID_timeseries.csv"

# --- Reproducibility ---
RANDOM_SEED: int = 42

# --- Cross-validation ---
N_CV_FOLDS: int = 5
TEST_SIZE: float = 0.2

# --- Column names (after renaming) ---
ATHLETE_ID_COL: str = "athlete_id"
INJURY_COL: str = "injury"
DATE_COL: str = "date"

# --- Day approach: 7 days x 10 features ---
DAY_FEATURES: list[str] = [
    "nr_sessions",
    "total_km",
    "km_z3_4",
    "km_z5_t1_t2",
    "km_sprinting",
    "strength_training",
    "hours_alternative",
    "perceived_exertion",
    "perceived_training_success",
    "perceived_recovery",
]
N_DAY_BLOCKS: int = 7  # days 0-6

# --- Week approach: 3 weeks x 22 features + 3 relative ratios ---
WEEK_FEATURES: list[str] = [
    "nr_sessions",
    "nr_rest_days",
    "total_kms",
    "max_km_one_day",
    "total_km_z3_z4_z5_t1_t2",
    "nr_tough_sessions",
    "nr_days_with_interval_session",
    "total_km_z3_4",
    "max_km_z3_4_one_day",
    "total_km_z5_t1_t2",
    "max_km_z5_t1_t2_one_day",
    "total_hours_alternative_training",
    "nr_strength_trainings",
    "avg_exertion",
    "min_exertion",
    "max_exertion",
    "avg_training_success",
    "min_training_success",
    "max_training_success",
    "avg_recovery",
    "min_recovery",
    "max_recovery",
]
N_WEEK_BLOCKS: int = 3  # weeks 0-2

WEEK_RELATIVE_FEATURES: list[str] = [
    "rel_total_kms_week_0_1",
    "rel_total_kms_week_0_2",
    "rel_total_kms_week_1_2",
]

# --- Week approach: target binarization ---
WEEK_INJURY_THRESHOLD: float = 0.5

# --- Sentinel value ---
SENTINEL_VALUE: float = -0.01
SENTINEL_REPLACEMENT: float = 0.0
