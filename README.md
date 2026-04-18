# India Air Quality Predictor

An end-to-end Machine Learning project that predicts both the **Air Quality Index (AQI)** value and the **AQI category** for Indian cities using pollutant concentration data from the Kaggle **Air Quality Data in India** dataset.

This project was built as a portfolio-ready ML system, not just a notebook. It covers the full pipeline from data preprocessing and feature engineering to model training, evaluation, artifact saving, and deployment through a Streamlit web app.

## Why this project matters

Air pollution is a major public health challenge in India. AQI is a simple way to summarize how polluted the air is, but the underlying data contains many pollutant measurements such as PM2.5, PM10, NO2, CO, SO2, and O3.

This project solves two real-world tasks:

- **Regression:** predict the exact AQI value
- **Classification:** predict the AQI bucket such as `Good`, `Moderate`, `Poor`, or `Severe`

That makes the project useful from both a technical and practical point of view:

- the regression model gives a detailed numeric estimate
- the classification model gives a human-friendly air quality label

## Resume-ready highlights

- Built a complete end-to-end ML pipeline on a real Indian environmental dataset
- Trained and compared both regression and classification models in one project
- Performed preprocessing, feature engineering, train-test split, and hyperparameter tuning
- Built and deployed a professional Streamlit app for interactive predictions
- Saved trained artifacts and evaluation reports for reproducibility and deployment

## Live Demo

- Live app: [https://aqi-india-ml.streamlit.app](https://aqi-india-ml.streamlit.app)
- Local app: `streamlit run app/streamlit_app.py`

## Dataset

- Source: [Air Quality Data in India - Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india/data)
- File used in this project: `city_day.csv`
- Coverage: daily city-level air quality measurements across multiple Indian cities

Place the dataset here before training:

```text
air-quality-ml-project/data/city_day.csv
```

## Problem Statement

Given pollutant levels and date-based context for an Indian city, predict:

1. the **AQI value**
2. the **AQI bucket**

This creates a strong end-to-end ML case study because the same dataset supports two different machine learning problem types:

- regression for numeric prediction
- classification for category prediction

## Project Workflow

### 1. Data understanding

The raw dataset includes:

- city names
- date
- pollutant measurements
- AQI
- AQI bucket

### 2. Data preprocessing

The preprocessing pipeline includes:

- duplicate removal
- date conversion
- feature extraction from `Date`
- separate datasets for regression and classification
- missing value handling with:
  - median imputation for numerical features
  - mode imputation for categorical features
- one-hot encoding for city names
- feature scaling for numerical columns

### 3. Feature engineering

The original `Date` column is converted into:

- `year`
- `month`
- `day`
- `day_of_week`
- `is_weekend`

These features help the model learn seasonal and time-based pollution patterns.

### 4. Model training

Two baseline models and two final tuned models were used:

- Regression baseline: `LinearRegression`
- Regression final model: `RandomForestRegressor`
- Classification baseline: `LogisticRegression`
- Classification final model: `RandomForestClassifier`

### 5. Hyperparameter tuning

`GridSearchCV` was used to tune Random Forest hyperparameters such as:

- number of estimators
- maximum tree depth
- minimum samples required to split

### 6. Deployment

A Streamlit web app was built so that users can:

- select a prediction task
- enter pollutant values and city
- get the predicted AQI value or AQI bucket instantly

## Model Performance

These are the actual saved results from the trained models in `models/metrics_report.json`.

### Regression

- **Model:** `RandomForestRegressor`
- **R2 Score:** `0.9124`
- **MAE:** `20.2686`
- **RMSE:** `40.06`

What this means:

- the model explains about **91% of the variation** in AQI values
- the average prediction error is about **20 AQI points**
- larger errors are penalized more strongly through RMSE

### Classification

- **Model:** `RandomForestClassifier`
- **Accuracy:** `0.8181`
- **Weighted F1-score:** `0.8161`
- **Macro F1-score:** `0.7886`

What this means:

- the model predicts the correct AQI category about **81.8%** of the time
- the weighted F1-score shows good overall balance across the dataset
- the macro F1-score shows the model is reasonably balanced even across smaller classes

### Best tuned parameters

Regression:

- `n_estimators = 200`
- `max_depth = None`
- `min_samples_split = 5`

Classification:

- `n_estimators = 200`
- `max_depth = None`
- `min_samples_split = 2`

## Key Insights

- Pollutants such as **PM2.5** and **PM10** are strong indicators of AQI
- City-level context matters because pollution patterns differ widely across Indian cities
- Time-based features improve prediction by helping the model capture seasonal trends
- Classification is easier to communicate to non-technical users, while regression gives more precise estimates

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## Project Structure

```text
air-quality-ml-project/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── city_day.csv
│   └── other raw Kaggle files
├── docs/
│   ├── deployment_guide.md
│   ├── interview_questions.md
│   ├── project_walkthrough.md
│   └── streamlit_code_explained.md
├── models/
│   ├── classification_model.joblib
│   ├── metrics_report.json
│   ├── model_metadata.json
│   └── regression_model.joblib
├── notebooks/
├── src/
│   ├── config.py
│   ├── data_utils.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/air-quality-ml-project.git
cd air-quality-ml-project
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download `city_day.csv` from Kaggle and place it inside the `data/` folder.

### 5. Train the models

```bash
python -m src.train
```

### 6. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

## Streamlit App Features

- clean and simple UI
- two prediction modes in one app
- dynamic input form generation
- model metrics shown inside the app
- ready for local demo and Streamlit Community Cloud deployment

## Deployment

This project can be deployed using Streamlit Community Cloud.

Deployment notes are available in:

- [docs/deployment_guide.md](docs/deployment_guide.md)

## Interview / Viva Preparation

Beginner-friendly project explanations and viva preparation documents are included:

- [docs/project_walkthrough.md](docs/project_walkthrough.md)
- [docs/streamlit_code_explained.md](docs/streamlit_code_explained.md)
- [docs/interview_questions.md](docs/interview_questions.md)

## What I learned from this project

- how to handle a real-world tabular dataset
- how to separate regression and classification pipelines cleanly
- how to avoid data leakage
- how preprocessing impacts model performance
- how to package ML work into a deployable product

## Future Improvements

- add visual EDA charts to the Streamlit app
- compare more advanced boosting models such as XGBoost or LightGBM
- deploy a hosted live demo and attach it here
- add feature importance charts for better interpretability
- add CI checks and automated retraining workflow

## Author

**Aryan Rai**

If you are viewing this for internship, placement, or project evaluation purposes, this repository demonstrates my ability to build and explain a complete applied machine learning solution from scratch.
