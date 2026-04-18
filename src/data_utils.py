from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

POLLUTANT_COLUMNS = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene",
    "Xylene",
]


@dataclass
class PreparedDatasets:
    regression_features: pd.DataFrame
    regression_target: pd.Series
    classification_features: pd.DataFrame
    classification_target: pd.Series


def load_dataset(data_path: str) -> pd.DataFrame:
    """Load the raw Kaggle dataset."""
    return pd.read_csv(data_path)


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Date into model-friendly numeric features."""
    prepared = df.copy()
    prepared = prepared.drop_duplicates()
    prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce")
    prepared = prepared.dropna(subset=["Date"])
    prepared["year"] = prepared["Date"].dt.year
    prepared["month"] = prepared["Date"].dt.month
    prepared["day"] = prepared["Date"].dt.day
    prepared["day_of_week"] = prepared["Date"].dt.dayofweek
    prepared["is_weekend"] = prepared["day_of_week"].isin([5, 6]).astype(int)
    return prepared.drop(columns=["Date"])


def prepare_datasets(df: pd.DataFrame) -> PreparedDatasets:
    """Create separate feature-target datasets for regression and classification."""
    prepared = add_date_features(df)

    regression_df = prepared.dropna(subset=["AQI"]).copy()
    classification_df = prepared.dropna(subset=["AQI_Bucket"]).copy()

    x_reg = regression_df.drop(columns=["AQI", "AQI_Bucket"])
    y_reg = regression_df["AQI"]

    # AQI is dropped here to avoid leakage because AQI_Bucket is derived from AQI.
    x_clf = classification_df.drop(columns=["AQI_Bucket", "AQI"])
    y_clf = classification_df["AQI_Bucket"]

    return PreparedDatasets(
        regression_features=x_reg,
        regression_target=y_reg,
        classification_features=x_clf,
        classification_target=y_clf,
    )


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Apply median imputation + scaling to numerics and mode imputation + one-hot encoding to categoricals."""
    numeric_columns = features.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_columns = features.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )


def build_feature_defaults(df: pd.DataFrame) -> dict[str, object]:
    """Create sensible defaults for the Streamlit form."""
    defaults: dict[str, object] = {}
    for column in df.columns:
        if df[column].dtype == "object":
            defaults[column] = df[column].mode().iloc[0]
        else:
            defaults[column] = float(df[column].median())
    return defaults


def build_feature_options(df: pd.DataFrame) -> dict[str, list[str]]:
    """Collect category options for text input fields."""
    options: dict[str, list[str]] = {}
    for column in df.select_dtypes(include=["object"]).columns:
        options[column] = sorted(df[column].dropna().astype(str).unique().tolist())
    return options
