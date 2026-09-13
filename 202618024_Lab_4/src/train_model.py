import os
import warnings
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor


warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

DATA_PATH = "data/AB_NYC_2019.csv"

MODEL_DIR = "models"

MODEL_PATH = (
    "models/airbnb_price_pipeline.pkl"
)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH
)

print(
    "Dataset shape:",
    df.shape
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates()

print(
    "Shape after duplicate removal:",
    df.shape
)


# ============================================================
# REMOVE INVALID PRICES
# ============================================================

df = df[
    df["price"] > 0
].copy()


# ============================================================
# HANDLE PRICE OUTLIERS
# ============================================================

upper_limit = df["price"].quantile(
    0.99
)

df = df[
    df["price"] <= upper_limit
].copy()


print(
    "Price upper limit:",
    upper_limit
)

print(
    "Shape after outlier removal:",
    df.shape
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

# Convert date column

df["last_review"] = pd.to_datetime(
    df["last_review"],
    errors="coerce"
)


# Review year

df["review_year"] = (
    df["last_review"]
    .dt.year
)


# Review month

df["review_month"] = (
    df["last_review"]
    .dt.month
)


# Days since last review

reference_date = pd.Timestamp(
    "2019-07-01"
)

df["review_days_old"] = (
    reference_date -
    df["last_review"]
).dt.days


# Missing date information

df["review_year"] = (
    df["review_year"]
    .fillna(0)
)

df["review_month"] = (
    df["review_month"]
    .fillna(0)
)

df["review_days_old"] = (
    df["review_days_old"]
    .fillna(
        df["review_days_old"].median()
    )
)


# Reviews per month

df["reviews_per_month"] = (
    df["reviews_per_month"]
    .fillna(0)
)


# Total reviews

df["total_reviews"] = (
    df["number_of_reviews"]
)


# Availability ratio

df["availability_ratio"] = (
    df["availability_365"] / 365
)


# Log transformations

df["minimum_nights_log"] = np.log1p(
    df["minimum_nights"]
)

df["number_of_reviews_log"] = np.log1p(
    df["number_of_reviews"]
)

df["availability_365_log"] = np.log1p(
    df["availability_365"]
)


# ============================================================
# REMOVE UNUSED COLUMNS
# ============================================================

columns_to_drop = [
    "id",
    "name",
    "host_id",
    "host_name",
    "last_review"
]


df = df.drop(
    columns=columns_to_drop,
    errors="ignore"
)


# ============================================================
# TARGET AND FEATURES
# ============================================================

X = df.drop(
    columns=["price"]
)

y = df["price"]


print(
    "\nNumber of observations:",
    len(X)
)

print(
    "Number of features:",
    X.shape[1]
)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=RANDOM_STATE
)


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# IDENTIFY FEATURE TYPES
# ============================================================

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()


print(
    "\nNumeric features:",
    numeric_features
)

print(
    "\nCategorical features:",
    categorical_features
)


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[

        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=150,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        ),

    "XGBoost":
        XGBRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
}


# ============================================================
# MODEL COMPARISON
# ============================================================

results = []

trained_models = {}


for name, model in models.items():

    print("\n" + "=" * 60)

    print(
        "Training:",
        name
    )

    print("=" * 60)


    pipeline = Pipeline(
        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )
        ]
    )


    pipeline.fit(
        X_train,
        y_train
    )


    train_pred = pipeline.predict(
        X_train
    )

    test_pred = pipeline.predict(
        X_test
    )


    train_mae = mean_absolute_error(
        y_train,
        train_pred
    )

    test_mae = mean_absolute_error(
        y_test,
        test_pred
    )


    train_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            train_pred
        )
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_pred
        )
    )


    train_r2 = r2_score(
        y_train,
        train_pred
    )

    test_r2 = r2_score(
        y_test,
        test_pred
    )


    print(
        "Train MAE:",
        round(train_mae, 2)
    )

    print(
        "Test MAE:",
        round(test_mae, 2)
    )

    print(
        "Train RMSE:",
        round(train_rmse, 2)
    )

    print(
        "Test RMSE:",
        round(test_rmse, 2)
    )

    print(
        "Train R2:",
        round(train_r2, 4)
    )

    print(
        "Test R2:",
        round(test_r2, 4)
    )


    results.append({

        "Model": name,

        "Train_MAE": train_mae,

        "Test_MAE": test_mae,

        "Train_RMSE": train_rmse,

        "Test_RMSE": test_rmse,

        "Train_R2": train_r2,

        "Test_R2": test_r2

    })


    trained_models[name] = pipeline


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    "Test_RMSE"
)


print("\n" + "=" * 60)

print("MODEL COMPARISON")

print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SELECT XGBOOST FOR TUNING
# ============================================================

print("\nStarting XGBoost hyperparameter tuning...")


xgb_pipeline = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)


param_grid = {

    "model__n_estimators": [
        200,
        300,
        400
    ],

    "model__max_depth": [
        3,
        5,
        7
    ],

    "model__learning_rate": [
        0.03,
        0.05,
        0.1
    ],

    "model__subsample": [
        0.8,
        1.0
    ],

    "model__colsample_bytree": [
        0.8,
        1.0
    ]
}


search = RandomizedSearchCV(

    estimator=xgb_pipeline,

    param_distributions=param_grid,

    n_iter=10,

    scoring="neg_root_mean_squared_error",

    cv=3,

    random_state=RANDOM_STATE,

    n_jobs=-1,

    verbose=1
)


search.fit(
    X_train,
    y_train
)


# ============================================================
# FINAL MODEL
# ============================================================

final_model = search.best_estimator_


print("\nBest parameters:")

print(
    search.best_params_
)


# ============================================================
# FINAL EVALUATION
# ============================================================

train_pred = final_model.predict(
    X_train
)

test_pred = final_model.predict(
    X_test
)


train_mae = mean_absolute_error(
    y_train,
    train_pred
)

test_mae = mean_absolute_error(
    y_test,
    test_pred
)


train_rmse = np.sqrt(
    mean_squared_error(
        y_train,
        train_pred
    )
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_pred
    )
)


train_r2 = r2_score(
    y_train,
    train_pred
)

test_r2 = r2_score(
    y_test,
    test_pred
)


print("\n" + "=" * 60)

print("FINAL MODEL PERFORMANCE")

print("=" * 60)

print(
    "Train MAE:",
    round(train_mae, 2)
)

print(
    "Test MAE:",
    round(test_mae, 2)
)

print(
    "Train RMSE:",
    round(train_rmse, 2)
)

print(
    "Test RMSE:",
    round(test_rmse, 2)
)

print(
    "Train R2:",
    round(train_r2, 4)
)

print(
    "Test R2:",
    round(test_r2, 4)
)


# ============================================================
# OVERFITTING CHECK
# ============================================================

r2_gap = train_r2 - test_r2


print(
    "\nTrain-Test R2 gap:",
    round(r2_gap, 4)
)


if r2_gap > 0.15:

    print(
        "Possible overfitting."
    )

elif train_r2 < 0.50 and test_r2 < 0.50:

    print(
        "Possible underfitting."
    )

else:

    print(
        "No severe overfitting detected."
    )


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    final_model,
    MODEL_PATH
)


print("\n" + "=" * 60)

print(
    "MODEL SAVED SUCCESSFULLY!"
)

print("=" * 60)

print(
    "Location:",
    MODEL_PATH
)


# ============================================================
# VERIFY MODEL FILE
# ============================================================

if os.path.exists(MODEL_PATH):

    file_size = os.path.getsize(
        MODEL_PATH
    )

    print(
        f"Model file size: {file_size:,} bytes"
    )

else:

    print(
        "ERROR: Model file was not created."
    )


print("\nTraining completed.")