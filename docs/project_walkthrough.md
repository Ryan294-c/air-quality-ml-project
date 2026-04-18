# Project Walkthrough

## Step 1: Understand the problem

We are solving two machine learning problems with one dataset:

- Regression: predict the exact AQI value
- Classification: predict the AQI category

Regression answers "how much?" while classification answers "which class?".

## Step 2: Understand the dataset

We use the Kaggle file `city_day.csv`. Important columns:

- `City`: the city name
- `Date`: the date of observation
- Pollutant columns such as `PM2.5`, `PM10`, `NO2`, `CO`, and `SO2`
- `AQI`: numerical target for regression
- `AQI_Bucket`: categorical target for classification

## Step 3: Why preprocessing matters

Models cannot learn well from messy data. We preprocess because:

- Missing target values cannot teach the model
- Text columns must be encoded into numbers
- Missing feature values must be filled
- Date text is not very useful until we extract year and month
- Features with very different scales can affect some models unfairly

If we skip preprocessing, many models either fail completely or produce weak results.

## Step 4: What the code in `src/data_utils.py` does

### `load_dataset`

Loads the CSV file into a pandas DataFrame.

### `add_date_features`

Turns the `Date` column into:

- `year`
- `month`
- `day`
- `day_of_week`
- `is_weekend`

We do this because pollution often changes with season, month, and weekdays versus weekends.

### `prepare_datasets`

Creates two datasets:

- regression dataset with target `AQI`
- classification dataset with target `AQI_Bucket`

For classification, `AQI` is removed from the inputs to avoid **data leakage**. Leakage means the model gets information that is too close to the answer.

### `build_preprocessor`

Creates a clean preprocessing pipeline:

- numeric columns: median imputation + standard scaling
- categorical columns: most-frequent imputation + one-hot encoding

This keeps the training workflow safe and reusable.

## Step 5: What the code in `src/train.py` does

### Train-test split

The data is split into:

- training set: teaches the model
- testing set: checks how well the model performs on unseen data

We split before fitting preprocessing steps to avoid leakage.

### Regression model

We train:

- Baseline: `LinearRegression`
- Final tuned model: `RandomForestRegressor`

Why Random Forest:

- handles nonlinear patterns well
- usually performs strongly on tabular datasets
- does not need complex feature assumptions

### Classification model

We train:

- Baseline: `LogisticRegression`
- Final tuned model: `RandomForestClassifier`

Why Random Forest:

- works well on mixed tabular data
- can model complex relationships
- is a strong and interview-friendly benchmark

### Hyperparameter tuning

Hyperparameters are the settings we choose before training, such as:

- number of trees
- maximum tree depth
- minimum samples required to split a node

`GridSearchCV` tries different combinations and finds the best one using cross-validation.

## Step 6: Model evaluation in simple words

### Regression metrics

- `R2 Score`: how well the model explains the variation in AQI
- `MAE`: average absolute prediction error
- `RMSE`: similar to MAE but punishes large mistakes more strongly

### Classification metrics

- `Accuracy`: percentage of correct predictions
- `Confusion Matrix`: shows where the model is confusing one AQI bucket with another
- `Precision`: when the model predicts a class, how often it is correct
- `Recall`: how well the model finds the actual examples of a class
- `F1-score`: balance between precision and recall

## Step 7: What the Streamlit app does

The app:

1. loads the saved models
2. shows a choice between AQI value prediction and AQI bucket prediction
3. creates input fields for each feature
4. runs the selected model when the user clicks **Predict**
5. shows evaluation metrics so the project looks complete and professional

## Step 8: Why this project is good for interviews

- It solves a real Indian problem
- It includes the full ML lifecycle, not just one notebook
- It demonstrates regression and classification together
- It shows deployment skills using Streamlit
- It gives you strong talking points about preprocessing, leakage, evaluation, and model tuning
