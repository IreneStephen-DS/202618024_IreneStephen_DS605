# DS605: Fundamentals of Machine Learning
## Lab Assignment 3
### Scikit-learn: Data Preprocessing and Model Performance Evaluation

Name: Irene Stephen  
Course: DS605 - Fundamentals of Machine Learning  
Lab: 3

## Dataset

The dataset used for this assignment is the Kaggle Hotel Booking Demand dataset.

Dataset:
Kaggle Hotel Booking Demand

File:
`hotel_bookings.csv`

## Objective

The objective of this lab is to build and compare Scikit-learn preprocessing pipelines and evaluate Logistic Regression and Decision Tree classification models for predicting whether a hotel booking will be canceled.

## Preprocessing

The following preprocessing steps were performed:

1. Loaded and explored the dataset using `head()`, `shape`, `info()`, `describe()` and `dtypes`.
2. Used `is_canceled` as the target variable.
3. Analyzed missing values and their percentages.
4. Removed `reservation_status` and `reservation_status_date` because they directly reveal the final booking outcome and would cause data leakage.
5. Removed the `company` column because it contained a very high percentage of missing values.
6. Checked numerical variables for outliers.
7. Removed clearly invalid records such as impossible negative values and bookings with zero adults, children and babies.
8. Used an 80:20 stratified train-test split with `random_state=42`.

## Pipeline A

Numerical features:

- KNNImputer with `n_neighbors=5`
- StandardScaler

Categorical features:

- SimpleImputer with `strategy="most_frequent"`
- OneHotEncoder with `handle_unknown="ignore"`

## Pipeline B

Numerical features:

- KNNImputer with `n_neighbors=5`
- MinMaxScaler

Categorical features:

- SimpleImputer with `strategy="most_frequent"`
- OneHotEncoder with `handle_unknown="ignore"`

## Models

Four model-pipeline combinations were evaluated:

1. Logistic Regression + StandardScaler
2. Logistic Regression + MinMaxScaler
3. Decision Tree + StandardScaler
4. Decision Tree + MinMaxScaler

Logistic Regression was trained using:

`LogisticRegression(max_iter=1000)`

Decision Tree was trained using:

`DecisionTreeClassifier(random_state=42)`

## Evaluation Metrics

The following metrics were calculated:

- Training Accuracy
- Testing Accuracy
- Precision
- Recall
- F1-score

## Results

The final comparison table is included in the notebook as well as `model_comparison.csv`.

The best overall model was selected based on the highest F1-score.

## Confusion Matrices

Confusion matrices were generated for:

- Best Logistic Regression model
- Best Decision Tree model

The figures are stored in the `figures` folder.

## Final Observations

1. The best overall model is determined using the F1-score on the test dataset.
2. StandardScaler and MinMaxScaler can produce different results for Logistic Regression because Logistic Regression is sensitive to feature scaling.
3. Decision Trees generally do not require feature scaling because they split observations based on feature thresholds.
4. The difference between training and testing accuracy provides an indication of possible overfitting.
5. The confusion matrices show the numbers of correctly and incorrectly classified canceled and non-canceled bookings.