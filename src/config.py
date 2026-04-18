from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "city_day.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EDA_NOTEBOOK_PATH = BASE_DIR / "notebooks" / "01_eda_air_quality.ipynb"
REGRESSION_MODEL_PATH = MODELS_DIR / "regression_model.joblib"
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "classification_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = MODELS_DIR / "metrics_report.json"
REGRESSION_CV_RESULTS_PATH = REPORTS_DIR / "regression_cv_results.csv"
CLASSIFICATION_CV_RESULTS_PATH = REPORTS_DIR / "classification_cv_results.csv"
REGRESSION_FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "regression_feature_importance.csv"
CLASSIFICATION_FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "classification_feature_importance.csv"
REGRESSION_FEATURE_IMPORTANCE_FIGURE_PATH = FIGURES_DIR / "regression_feature_importance.png"
CLASSIFICATION_FEATURE_IMPORTANCE_FIGURE_PATH = FIGURES_DIR / "classification_feature_importance.png"
EDA_SUMMARY_PATH = REPORTS_DIR / "eda_summary.md"
