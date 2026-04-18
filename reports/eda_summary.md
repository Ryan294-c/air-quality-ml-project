# Exploratory Data Analysis Summary

- Rows after duplicate removal: `29531`
- Columns used for modeling: `20`
- Regression rows with AQI: `24850`
- Classification rows with AQI_Bucket: `24850`

## Missing Values Before Imputation

- `Xylene`: 18109
- `PM10`: 11140
- `NH3`: 10328
- `Toluene`: 8041
- `Benzene`: 5623
- `AQI`: 4681
- `AQI_Bucket`: 4681
- `PM2.5`: 4598
- `NOx`: 4185
- `O3`: 4022

## AQI Bucket Distribution

- `Moderate`: 8829
- `Satisfactory`: 8224
- `Poor`: 2781
- `Very Poor`: 2337
- `Good`: 1341
- `Severe`: 1338

## Notes

- PM2.5 and PM10 show strong alignment with AQI and should be defended as core predictors.
- AQI MAE is meaningful in practice because a 20-point error can shift category boundaries near 50, 100, or 200.
- This project uses the same dataset for both regression and classification, which is the main framing difference from a standard beginner AQI notebook.