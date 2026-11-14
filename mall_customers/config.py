from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
PROFILE_COLUMNS = FEATURES + ["Age"]

K_RANGE = range(2, 11)
RANDOM_STATE = 42
N_INIT = 10