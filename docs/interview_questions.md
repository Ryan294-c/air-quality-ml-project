# 10 Interview or Viva Questions With Answers

## 1. Why did you choose this dataset?

I chose the Air Quality Data in India dataset because it is relevant to a real public-health issue in India and it supports both regression and classification tasks using the same data.

## 2. What is the difference between regression and classification in your project?

Regression predicts the exact AQI value, which is a number. Classification predicts the AQI bucket, which is a category such as Good, Moderate, or Poor.

## 3. Why did you create date-based features?

The original `Date` column is text, so most ML models cannot use it directly. I extracted year, month, day, day of week, and weekend information so the model can learn seasonal and time-based pollution patterns.

## 4. How did you handle missing values?

I dropped rows where the target was missing because those rows cannot teach the model. For numeric features, I used median imputation, and for categorical features, I used the most frequent value.

## 5. Why did you use one-hot encoding?

Columns like `City` are text labels. One-hot encoding converts them into numeric columns without creating a false ranking between cities.

## 6. Why did you scale the numerical features?

Scaling helps models such as Linear Regression and Logistic Regression work more reliably when numerical columns have very different ranges.

## 7. What is data leakage and how did you avoid it?

Data leakage happens when the model gets information it would not have in a real prediction setting. I avoided it by dropping `AQI` from the classification inputs because `AQI_Bucket` is derived from AQI.

## 8. Why did you use Random Forest as the final model?

Random Forest works very well on tabular data, can capture nonlinear relationships, and is a strong choice when feature interactions are complex.

## 9. What metrics did you use to evaluate the models?

For regression, I used R2 score, MAE, and RMSE. For classification, I used accuracy, confusion matrix, precision, recall, and F1-score.

## 10. How did you deploy the project?

I built a Streamlit app, pushed the project to GitHub, and deployed it using Streamlit Community Cloud by selecting `app/streamlit_app.py` as the entry point.
