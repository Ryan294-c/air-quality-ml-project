# Streamlit Community Cloud Deployment Guide

## Files you need before deployment

Make sure your GitHub repository contains:

- `app/streamlit_app.py`
- `src/train.py`
- `src/data_utils.py`
- `src/config.py`
- `requirements.txt`
- `README.md`
- `models/regression_model.joblib`
- `models/classification_model.joblib`
- `models/model_metadata.json`
- `models/metrics_report.json`

Do not forget to train the models locally first so the `models/` files exist.

## How to deploy

1. Push the project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in with GitHub.
4. Click **New app**.
5. Select your repository.
6. Set the main file path to:

```text
app/streamlit_app.py
```

7. Click **Deploy**.

## Common errors and fixes

### 1. `ModuleNotFoundError`

Cause:

- missing package in `requirements.txt`

Fix:

- add the package
- commit and push again

### 2. Model files not found

Cause:

- trained `.joblib` files or `.json` metadata files were not pushed to GitHub

Fix:

- run training locally
- verify the files exist in `models/`
- commit and push them

### 3. Dataset file missing on Streamlit Cloud

Cause:

- the app expects only saved models at runtime, not training data

Fix:

- deploy the saved models, not the raw dataset
- do training locally before deployment

### 4. Wrong app path

Cause:

- incorrect entry file selected during deployment

Fix:

- make sure the app path is exactly `app/streamlit_app.py`

## Best practice

For Streamlit Cloud, it is better to:

- train models locally
- save them into `models/`
- deploy only the app and saved model artifacts

This makes deployment faster and more stable.
