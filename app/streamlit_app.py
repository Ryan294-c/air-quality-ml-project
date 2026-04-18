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

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=DM+Sans:wght@400;500;700&display=swap');

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(42, 157, 143, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(233, 196, 106, 0.22), transparent 24%),
            linear-gradient(180deg, #f6fbf8 0%, #edf6f2 100%);
    }

    html, body, [class*="css"]  {
        font-family: "DM Sans", sans-serif;
    }

    h1, h2, h3 {
        font-family: "Space Grotesk", sans-serif !important;
        letter-spacing: -0.02em;
    }

    .hero-card, .panel-card, .result-card {
        border-radius: 24px;
        padding: 1.2rem 1.35rem;
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(38, 70, 83, 0.08);
        box-shadow: 0 16px 40px rgba(38, 70, 83, 0.08);
        backdrop-filter: blur(6px);
    }

    .hero-card {
        padding: 1.6rem;
        background: linear-gradient(135deg, rgba(42,157,143,0.92), rgba(38,70,83,0.92));
        color: white;
        border: none;
    }

    .hero-kicker {
        display: inline-block;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.16);
        margin-bottom: 0.9rem;
    }

    .hero-title {
        font-size: 3rem;
        line-height: 1;
        margin: 0;
    }

    .hero-subtitle {
        max-width: 720px;
        margin: 0.9rem 0 0 0;
        font-size: 1.03rem;
        color: rgba(255,255,255,0.9);
    }

    .stat-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #5c6b73;
        margin-bottom: 0.15rem;
    }

    .stat-value {
        font-family: "Space Grotesk", sans-serif;
        font-size: 1.9rem;
        color: #1f2d3d;
        margin: 0;
    }

    .stat-note {
        font-size: 0.9rem;
        color: #6c7a89;
        margin-top: 0.2rem;
    }

    .section-title {
        font-family: "Space Grotesk", sans-serif;
        font-size: 1.35rem;
        margin-bottom: 0.8rem;
        color: #1f2d3d;
    }

    .result-card {
        background: linear-gradient(135deg, rgba(233,196,106,0.16), rgba(42,157,143,0.1));
        margin-top: 1rem;
    }

    .result-title {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #5c6b73;
    }

    .result-value {
        font-family: "Space Grotesk", sans-serif;
        font-size: 2.4rem;
        color: #132a13;
        margin-top: 0.2rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.82);
        border: 1px solid rgba(38, 70, 83, 0.08);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        box-shadow: 0 8px 24px rgba(38, 70, 83, 0.06);
    }

    div[data-testid="stRadio"] > div {
        background: rgba(255,255,255,0.78);
        border-radius: 18px;
        padding: 0.6rem 0.8rem;
        border: 1px solid rgba(38, 70, 83, 0.08);
    }

    .insight-chip {
        display: inline-block;
        margin: 0.25rem 0.35rem 0 0;
        padding: 0.45rem 0.75rem;
        border-radius: 999px;
        background: rgba(38,70,83,0.08);
        color: #264653;
        font-size: 0.92rem;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
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
        st.markdown('<div class="section-title">Regression Metrics</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("R2 Score", metrics["regression"]["r2_score"])
        col2.metric("MAE", metrics["regression"]["mae"])
        col3.metric("RMSE", metrics["regression"]["rmse"])
    else:
        st.markdown('<div class="section-title">Classification Metrics</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        report = metrics["classification"]["classification_report"]
        col1.metric("Accuracy", metrics["classification"]["accuracy"])
        col2.metric("Weighted F1", round(report["weighted avg"]["f1-score"], 4))
        col3.metric("Macro F1", round(report["macro avg"]["f1-score"], 4))
        confusion_df = pd.DataFrame(
            metrics["classification"]["confusion_matrix"],
            index=metrics["classification"]["labels"],
            columns=metrics["classification"]["labels"],
        )
        st.write("Confusion Matrix")
        st.dataframe(confusion_df, use_container_width=True)


def render_header(metrics: dict) -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Deployed ML Portfolio Project</div>
            <h1 class="hero-title">India Air Quality Predictor</h1>
            <p class="hero-subtitle">
                Dual-task machine learning system for Indian city-level air quality forecasting.
                Predict both the exact AQI value and the AQI bucket from pollutant signals, city context, and date-derived features.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top1, top2, top3 = st.columns(3)
    top1.markdown(
        f"""
        <div class="panel-card">
            <div class="stat-label">Regression R2</div>
            <p class="stat-value">{metrics["regression"]["r2_score"]}</p>
            <div class="stat-note">Variance explained on held-out data</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    top2.markdown(
        f"""
        <div class="panel-card">
            <div class="stat-label">Classification Accuracy</div>
            <p class="stat-value">{metrics["classification"]["accuracy"]}</p>
            <div class="stat-note">Bucket prediction performance</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    top3.markdown(
        """
        <div class="panel-card">
            <div class="stat-label">Project Focus</div>
            <p class="stat-value">2 Tasks</p>
            <div class="stat-note">Regression + classification in one deployed app</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(metrics: dict) -> None:
    with st.sidebar:
        st.markdown("## Project Snapshot")
        st.write(
            "This interface demonstrates the same dataset through two ML formulations so you can explain both numeric prediction and category prediction in one project."
        )
        st.markdown("### Strong Talking Points")
        st.markdown(
            """
            <span class="insight-chip">Dual-task framing</span>
            <span class="insight-chip">Feature engineering</span>
            <span class="insight-chip">GridSearchCV</span>
            <span class="insight-chip">Streamlit deployment</span>
            <span class="insight-chip">EDA + validation artifacts</span>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### Quick Metrics")
        st.metric("AQI MAE", metrics["regression"]["mae"])
        st.metric("Weighted F1", round(metrics["classification"]["classification_report"]["weighted avg"]["f1-score"], 4))


def main():
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
    render_sidebar(metrics)
    render_header(metrics)

    task_name = st.radio(
        "Choose prediction task",
        options=["regression", "classification"],
        format_func=lambda name: "Predict AQI Value" if name == "regression" else "Predict AQI Bucket",
        horizontal=True,
    )

    with st.container(border=True):
        st.markdown('<div class="section-title">Enter Air Quality Details</div>', unsafe_allow_html=True)
        input_df = build_input_frame(task_name, metadata)

    prediction_value = None
    if st.button("Predict", type="primary", use_container_width=True):
        if task_name == "regression":
            prediction_value = f"{regression_model.predict(input_df)[0]:.2f}"
        else:
            prediction_value = str(classification_model.predict(input_df)[0])

    if prediction_value is not None:
        label = "Predicted AQI Value" if task_name == "regression" else "Predicted AQI Bucket"
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">{label}</div>
                <div class="result-value">{prediction_value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
