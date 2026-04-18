from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_PATH,
    CLASSIFICATION_MODEL_PATH,
    METRICS_PATH,
    MODEL_METADATA_PATH,
    REGRESSION_MODEL_PATH,
)
from src.train import bootstrap_models_for_deployment

st.set_page_config(
    page_title="India Air Quality Predictor",
    page_icon="🌿",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    regression_model = joblib.load(REGRESSION_MODEL_PATH)
    classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return regression_model, classification_model, metadata, metrics


def artifacts_available() -> bool:
    return (
        REGRESSION_MODEL_PATH.exists()
        and CLASSIFICATION_MODEL_PATH.exists()
        and MODEL_METADATA_PATH.exists()
        and METRICS_PATH.exists()
    )


def build_input_frame(task_name: str, metadata: dict) -> pd.DataFrame:
    features = metadata["features"][task_name]
    defaults = metadata["feature_defaults"][task_name]
    options = metadata["feature_options"][task_name]

    values: dict[str, object] = {}
    columns = st.columns(2)

    for index, feature in enumerate(features):
        column_ui = columns[index % 2]
        with column_ui:
            if feature in options:
                default_value = str(defaults[feature])
                choices = options[feature]
                default_index = choices.index(default_value) if default_value in choices else 0
                values[feature] = st.selectbox(
                    f"{feature}",
                    options=choices,
                    index=default_index,
                    key=f"{task_name}_{feature}",
                )
            else:
                values[feature] = st.number_input(
                    f"{feature}",
                    value=float(defaults[feature]),
                    key=f"{task_name}_{feature}",
                )

    return pd.DataFrame([values])


def show_metrics(metrics: dict, task_name: str) -> None:
    if task_name == "regression":
        st.subheader("Regression Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("R2 Score", metrics["regression"]["r2_score"])
        col2.metric("MAE", metrics["regression"]["mae"])
        col3.metric("RMSE", metrics["regression"]["rmse"])
    else:
        st.subheader("Classification Metrics")
        st.metric("Accuracy", metrics["classification"]["accuracy"])
        confusion_df = pd.DataFrame(
            metrics["classification"]["confusion_matrix"],
            index=metrics["classification"]["labels"],
            columns=metrics["classification"]["labels"],
        )
        st.write("Confusion Matrix")
        st.dataframe(confusion_df, use_container_width=True)


def main():
    st.title("India Air Quality Predictor")
    st.caption(
        "End-to-end machine learning project using the Kaggle Air Quality Data in India dataset."
    )

    if not artifacts_available():
        if DATA_PATH.exists():
            with st.spinner("Preparing cloud models from the bundled dataset. This takes a minute on first launch."):
                bootstrap_models_for_deployment()
            st.cache_resource.clear()
            st.success("Cloud models prepared successfully. You can use the app now.")
        else:
            st.error(
                "Model files are missing and city_day.csv is not available in the deployed app."
            )
            return

    regression_model, classification_model, metadata, metrics = load_artifacts()

    task_name = st.radio(
        "Choose prediction task",
        options=["regression", "classification"],
        format_func=lambda name: "Predict AQI Value" if name == "regression" else "Predict AQI Bucket",
        horizontal=True,
    )

    with st.container(border=True):
        st.subheader("Enter Air Quality Details")
        input_df = build_input_frame(task_name, metadata)

    if st.button("Predict", type="primary", use_container_width=True):
        if task_name == "regression":
            prediction = regression_model.predict(input_df)[0]
            st.success(f"Predicted AQI value: {prediction:.2f}")
        else:
            prediction = classification_model.predict(input_df)[0]
            st.success(f"Predicted AQI bucket: {prediction}")

    show_metrics(metrics, task_name)

    with st.expander("Project Notes"):
        st.write(
            "The regression model predicts the numerical AQI value, while the classification model predicts the AQI category."
        )
        st.write(
            "Categorical columns are one-hot encoded, numeric columns are median-imputed and scaled, and Random Forest is used as the tuned final model for both tasks."
        )


if __name__ == "__main__":
    main()
