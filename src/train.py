from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    CLASSIFICATION_MODEL_PATH,
    DATA_PATH,
    METRICS_PATH,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    REGRESSION_MODEL_PATH,
)
from src.data_utils import (
    POLLUTANT_COLUMNS,
    build_feature_defaults,
    build_feature_options,
    build_preprocessor,
    load_dataset,
    prepare_datasets,
)


def train_regression_model(x_train, y_train) -> tuple[Pipeline, dict]:
    """Train and tune regression models."""
    baseline_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x_train)),
            ("model", LinearRegression()),
        ]
    )
    baseline_pipeline.fit(x_train, y_train)

    tuned_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x_train)),
            ("model", RandomForestRegressor(random_state=42)),
        ]
    )

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
    }

    grid_search = GridSearchCV(
        estimator=tuned_pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="r2",
        n_jobs=-1,
    )
    grid_search.fit(x_train, y_train)

    metadata = {
        "baseline_model": "LinearRegression",
        "final_model": "RandomForestRegressor",
        "best_params": grid_search.best_params_,
        "best_cv_score_r2": round(float(grid_search.best_score_), 4),
    }
    return grid_search.best_estimator_, metadata


def train_classification_model(x_train, y_train) -> tuple[Pipeline, dict]:
    """Train and tune classification models."""
    baseline_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x_train)),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    multi_class="auto",
                ),
            ),
        ]
    )
    baseline_pipeline.fit(x_train, y_train)

    tuned_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x_train)),
            ("model", RandomForestClassifier(random_state=42)),
        ]
    )

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
    }

    grid_search = GridSearchCV(
        estimator=tuned_pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid_search.fit(x_train, y_train)

    metadata = {
        "baseline_model": "LogisticRegression",
        "final_model": "RandomForestClassifier",
        "best_params": grid_search.best_params_,
        "best_cv_score_accuracy": round(float(grid_search.best_score_), 4),
    }
    return grid_search.best_estimator_, metadata


def evaluate_regression(model: Pipeline, x_test, y_test) -> dict:
    predictions = model.predict(x_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    return {
        "r2_score": round(float(r2_score(y_test, predictions)), 4),
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "rmse": round(float(rmse), 4),
    }


def evaluate_classification(model: Pipeline, x_test, y_test) -> dict:
    predictions = model.predict(x_test)
    labels = sorted(y_test.unique().tolist())
    report = classification_report(y_test, predictions, output_dict=True)
    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "labels": labels,
        "classification_report": report,
    }


def save_training_outputs(
    prepared,
    regression_model: Pipeline,
    classification_model: Pipeline,
    regression_training: dict,
    classification_training: dict,
    regression_metrics: dict,
    classification_metrics: dict,
) -> None:
    joblib.dump(regression_model, REGRESSION_MODEL_PATH)
    joblib.dump(classification_model, CLASSIFICATION_MODEL_PATH)

    metadata = {
        "dataset": {
            "source": "https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india/data",
            "filename": "city_day.csv",
        },
        "features": {
            "regression": prepared.regression_features.columns.tolist(),
            "classification": prepared.classification_features.columns.tolist(),
            "pollutants": POLLUTANT_COLUMNS,
        },
        "feature_defaults": {
            "regression": build_feature_defaults(prepared.regression_features),
            "classification": build_feature_defaults(prepared.classification_features),
        },
        "feature_options": {
            "regression": build_feature_options(prepared.regression_features),
            "classification": build_feature_options(prepared.classification_features),
        },
        "training": {
            "regression": regression_training,
            "classification": classification_training,
        },
    }

    metrics_report = {
        "regression": regression_metrics,
        "classification": classification_metrics,
    }

    save_json(MODEL_METADATA_PATH, metadata)
    save_json(METRICS_PATH, metrics_report)


def bootstrap_models_for_deployment() -> None:
    """Train smaller models quickly when cloud artifacts are missing."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Add city_day.csv to the data folder."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_dataset(str(DATA_PATH))
    prepared = prepare_datasets(raw_df)

    x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(
        prepared.regression_features,
        prepared.regression_target,
        test_size=0.2,
        random_state=42,
    )

    x_train_clf, x_test_clf, y_train_clf, y_test_clf = train_test_split(
        prepared.classification_features,
        prepared.classification_target,
        test_size=0.2,
        random_state=42,
        stratify=prepared.classification_target,
    )

    regression_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x_train_reg)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=40,
                    max_depth=12,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    regression_model.fit(x_train_reg, y_train_reg)

    classification_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x_train_clf)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=60,
                    max_depth=12,
                    min_samples_split=4,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    classification_model.fit(x_train_clf, y_train_clf)

    regression_metrics = evaluate_regression(regression_model, x_test_reg, y_test_reg)
    classification_metrics = evaluate_classification(
        classification_model, x_test_clf, y_test_clf
    )

    save_training_outputs(
        prepared=prepared,
        regression_model=regression_model,
        classification_model=classification_model,
        regression_training={
            "baseline_model": "LinearRegression",
            "final_model": "RandomForestRegressor",
            "best_params": {
                "model__n_estimators": 40,
                "model__max_depth": 12,
                "model__min_samples_split": 5,
            },
            "best_cv_score_r2": "quick_bootstrap",
        },
        classification_training={
            "baseline_model": "LogisticRegression",
            "final_model": "RandomForestClassifier",
            "best_params": {
                "model__n_estimators": 60,
                "model__max_depth": 12,
                "model__min_samples_split": 4,
            },
            "best_cv_score_accuracy": "quick_bootstrap",
        },
        regression_metrics=regression_metrics,
        classification_metrics=classification_metrics,
    )


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=json_safe), encoding="utf-8")


def json_safe(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Download city_day.csv from Kaggle and place it in the data folder."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_dataset(str(DATA_PATH))
    prepared = prepare_datasets(raw_df)

    x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(
        prepared.regression_features,
        prepared.regression_target,
        test_size=0.2,
        random_state=42,
    )

    x_train_clf, x_test_clf, y_train_clf, y_test_clf = train_test_split(
        prepared.classification_features,
        prepared.classification_target,
        test_size=0.2,
        random_state=42,
        stratify=prepared.classification_target,
    )

    regression_model, regression_training = train_regression_model(x_train_reg, y_train_reg)
    classification_model, classification_training = train_classification_model(
        x_train_clf, y_train_clf
    )

    regression_metrics = evaluate_regression(regression_model, x_test_reg, y_test_reg)
    classification_metrics = evaluate_classification(
        classification_model, x_test_clf, y_test_clf
    )

    save_training_outputs(
        prepared=prepared,
        regression_model=regression_model,
        classification_model=classification_model,
        regression_training=regression_training,
        classification_training=classification_training,
        regression_metrics=regression_metrics,
        classification_metrics=classification_metrics,
    )

    print("Training complete.")
    print(f"Regression model saved to: {REGRESSION_MODEL_PATH}")
    print(f"Classification model saved to: {CLASSIFICATION_MODEL_PATH}")
    print(f"Metrics report saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
