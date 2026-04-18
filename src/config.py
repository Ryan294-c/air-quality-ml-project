from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "city_day.csv"
MODELS_DIR = BASE_DIR / "models"
REGRESSION_MODEL_PATH = MODELS_DIR / "regression_model.joblib"
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "classification_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = MODELS_DIR / "metrics_report.json"
