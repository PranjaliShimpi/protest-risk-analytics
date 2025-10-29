from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
SUSTAIN_MODEL_PATH = ARTIFACTS_DIR / "sustain_model.pkl"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
BASE_CSV_PATH = ARTIFACTS_DIR / "base.csv"
FEATURES_CSV_PATH = ARTIFACTS_DIR / "features.csv"
