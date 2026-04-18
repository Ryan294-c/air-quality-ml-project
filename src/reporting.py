from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    CLASSIFICATION_FEATURE_IMPORTANCE_FIGURE_PATH,
    CLASSIFICATION_FEATURE_IMPORTANCE_PATH,
    FIGURES_DIR,
    REGRESSION_FEATURE_IMPORTANCE_FIGURE_PATH,
    REGRESSION_FEATURE_IMPORTANCE_PATH,
    REPORTS_DIR,
)


def ensure_report_directories() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_cv_results(cv_results: dict, output_path: Path, score_column: str) -> list[dict]:
    ensure_report_directories()
    cv_frame = pd.DataFrame(cv_results)
    ranked = cv_frame.sort_values(by="rank_test_score").copy()
    ranked.to_csv(output_path, index=False)
    summary_columns = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        score_column,
    ]
    available = [column for column in summary_columns if column in ranked.columns]
    top_rows = ranked.head(5)[available + ["params"]].to_dict(orient="records")
    return top_rows


def _plot_feature_importance(importance_df: pd.DataFrame, title: str, output_path: Path) -> None:
    top_features = importance_df.head(12).sort_values("importance", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature"], top_features["importance"], color="#2a9d8f")
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_feature_importance(
    model,
    feature_names: list[str],
    csv_output_path: Path,
    figure_output_path: Path,
    title: str,
) -> list[dict]:
    ensure_report_directories()
    estimator = model.named_steps["model"]
    importances = estimator.feature_importances_
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(csv_output_path, index=False)
    _plot_feature_importance(importance_df, title, figure_output_path)
    return importance_df.head(10).to_dict(orient="records")


def build_eda_markdown(summary: dict) -> str:
    lines = [
        "# Exploratory Data Analysis Summary",
        "",
        f"- Rows after duplicate removal: `{summary['rows_after_cleaning']}`",
        f"- Columns used for modeling: `{summary['modeling_columns']}`",
        f"- Regression rows with AQI: `{summary['regression_rows']}`",
        f"- Classification rows with AQI_Bucket: `{summary['classification_rows']}`",
        "",
        "## Missing Values Before Imputation",
        "",
    ]
    for column, count in summary["missing_values"].items():
        lines.append(f"- `{column}`: {count}")
    lines.extend(
        [
            "",
            "## AQI Bucket Distribution",
            "",
        ]
    )
    for bucket, count in summary["aqi_bucket_distribution"].items():
        lines.append(f"- `{bucket}`: {count}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- PM2.5 and PM10 show strong alignment with AQI and should be defended as core predictors.",
            "- AQI MAE is meaningful in practice because a 20-point error can shift category boundaries near 50, 100, or 200.",
            "- This project uses the same dataset for both regression and classification, which is the main framing difference from a standard beginner AQI notebook.",
        ]
    )
    return "\n".join(lines)


def write_json(path: Path, payload: dict) -> None:
    ensure_report_directories()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
