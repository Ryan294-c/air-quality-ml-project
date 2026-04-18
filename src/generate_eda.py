from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import DATA_PATH, EDA_NOTEBOOK_PATH, EDA_SUMMARY_PATH, FIGURES_DIR, REPORTS_DIR
from src.data_utils import add_date_features, load_dataset
from src.reporting import build_eda_markdown, ensure_report_directories

sns.set_theme(style="whitegrid")


def save_figure(fig, name: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return f"reports/figures/{name}"


def build_notebook_json() -> dict:
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# India AQI EDA\n",
                    "\n",
                    "This notebook documents the exploratory analysis that supports the modeling choices in this project.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import seaborn as sns\n",
                    "import matplotlib.pyplot as plt\n",
                    "from src.data_utils import add_date_features\n",
                    "\n",
                    "df = pd.read_csv('data/city_day.csv')\n",
                    "prepared = add_date_features(df)\n",
                    "df.head()\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Questions answered in this analysis\n",
                    "\n",
                    "- What is the AQI target distribution?\n",
                    "- How are AQI buckets distributed?\n",
                    "- Which columns have missing values?\n",
                    "- Are there visible city or seasonal trends?\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "missing = df.isna().sum().sort_values(ascending=False)\n",
                    "missing.head(10)\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "sns.histplot(df['AQI'].dropna(), bins=40, kde=True)\n",
                    "plt.title('AQI Distribution')\n",
                    "plt.show()\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df['AQI_Bucket'].value_counts().plot(kind='bar')\n",
                    "plt.title('AQI Bucket Counts')\n",
                    "plt.show()\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.9",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    ensure_report_directories()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EDA_NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_df = load_dataset(str(DATA_PATH))
    cleaned_df = add_date_features(raw_df)

    missing_values = raw_df.isna().sum().sort_values(ascending=False)
    aqi_bucket_counts = raw_df["AQI_Bucket"].value_counts(dropna=False)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.histplot(raw_df["AQI"].dropna(), bins=40, kde=True, ax=ax1, color="#1d3557")
    ax1.set_title("AQI Distribution")
    ax1.set_xlabel("AQI")
    save_figure(fig1, "aqi_distribution.png")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    aqi_bucket_counts.drop(labels=[value for value in aqi_bucket_counts.index if pd.isna(value)]).plot(
        kind="bar", ax=ax2, color="#2a9d8f"
    )
    ax2.set_title("AQI Bucket Distribution")
    ax2.set_xlabel("AQI Bucket")
    ax2.set_ylabel("Count")
    plt.xticks(rotation=25)
    save_figure(fig2, "aqi_bucket_distribution.png")

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    missing_values.head(10).sort_values().plot(kind="barh", ax=ax3, color="#e76f51")
    ax3.set_title("Top 10 Missing-Value Columns")
    ax3.set_xlabel("Missing values")
    save_figure(fig3, "missing_values_top10.png")

    monthly_aqi = cleaned_df.dropna(subset=["AQI"]).groupby("month")["AQI"].mean().reset_index()
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=monthly_aqi, x="month", y="AQI", marker="o", ax=ax4, color="#457b9d")
    ax4.set_title("Average AQI by Month")
    ax4.set_xlabel("Month")
    ax4.set_ylabel("Average AQI")
    save_figure(fig4, "monthly_aqi_trend.png")

    summary = {
        "rows_after_cleaning": int(cleaned_df.shape[0]),
        "modeling_columns": int(cleaned_df.shape[1]),
        "regression_rows": int(cleaned_df["AQI"].notna().sum()),
        "classification_rows": int(cleaned_df["AQI_Bucket"].notna().sum()),
        "missing_values": {key: int(value) for key, value in missing_values.head(10).items()},
        "aqi_bucket_distribution": {
            str(key): int(value) for key, value in aqi_bucket_counts.items() if not pd.isna(key)
        },
    }

    EDA_SUMMARY_PATH.write_text(build_eda_markdown(summary), encoding="utf-8")
    EDA_NOTEBOOK_PATH.write_text(json.dumps(build_notebook_json(), indent=2), encoding="utf-8")
    print(f"EDA notebook written to: {EDA_NOTEBOOK_PATH}")
    print(f"EDA summary written to: {EDA_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
