# Streamlit Code Explained Line by Line

## `st.set_page_config(...)`

This sets the browser tab title, icon, and page width. It makes the app feel like a real product instead of a default demo.

## `@st.cache_resource`

This tells Streamlit to load the models once and reuse them. Without caching, the app would reload model files on every interaction and feel slower.

## `load_artifacts()`

This function loads:

- the regression model
- the classification model
- the metadata JSON file
- the metrics JSON file

The metadata file helps the app know which input fields to show and what default values to use.

## `build_input_frame(...)`

This function creates the form dynamically.

- If a feature is categorical, it creates a dropdown.
- If a feature is numeric, it creates a number input.

At the end, it returns a one-row pandas DataFrame because scikit-learn models expect tabular input.

## `show_metrics(...)`

This displays model performance inside the app.

- For regression, it shows R2, MAE, and RMSE.
- For classification, it shows accuracy and the confusion matrix.

This makes the app look more professional and also helps you explain model quality during demos.

## `main()`

This is the entry point of the Streamlit app.

It:

1. shows the title and short description
2. checks if saved model files exist
3. loads the models and metadata
4. lets the user choose regression or classification
5. builds the input form
6. predicts on button click
7. shows metrics and project notes

## Why this UI design is good for a student project

- It is simple enough to understand
- It is professional enough to demo
- It supports two machine learning tasks in one place
- It keeps the code modular instead of putting everything in one giant file
