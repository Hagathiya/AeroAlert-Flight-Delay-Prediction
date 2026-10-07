import os
import pandas as pd
import joblib

from src.features import create_features

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np


print("=" * 60)
print("AEROALERT - DELAY DURATION MODEL")
print("=" * 60)


# =========================================================
# 1. LOAD DATA
# =========================================================

print("\nLoading dataset...")

df = pd.read_csv(
    "dataset/flights.csv",
    nrows=500000
)

print("Original shape:", df.shape)


# =========================================================
# 2. KEEP ONLY OPERATED FLIGHTS
# =========================================================

df = df[
    (df["CANCELLED"] == 0)
    &
    (df["ARR_DELAY"].notna())
].copy()

print(
    "\nAfter removing cancelled/missing-delay flights:",
    df.shape
)


# =========================================================
# 3. SELECT PRE-FLIGHT FEATURES
# =========================================================

features = [
    "FL_DATE",
    "AIRLINE_CODE",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "CRS_ELAPSED_TIME",
    "DISTANCE"
]

df = df[
    features + ["ARR_DELAY"]
].dropna()


# =========================================================
# 4. TARGET & PRE-FLIGHT FEATURE ENGINEERING (Centralized)
# =========================================================

y = df["ARR_DELAY"].copy()
X = create_features(df[features])


# =========================================================
# 9. FEATURE GROUPS
# =========================================================

categorical_features = [
    "AIRLINE_CODE",
    "ORIGIN",
    "DEST",
    "TIME_OF_DAY",
    "ROUTE",
    "AIRLINE_ROUTE"
]

numerical_features = [
    "CRS_ELAPSED_TIME",
    "DISTANCE",
    "YEAR",
    "MONTH",
    "DAY_OF_WEEK",
    "DAY_OF_MONTH",
    "IS_WEEKEND",
    "DEP_HOUR",
    "DEP_MINUTE",
    "ARR_HOUR",
    "ARR_MINUTE"
]


# =========================================================
# 10. PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_features
        ),

        (
            "numerical",

            "passthrough",

            numerical_features
        )
    ]
)


# =========================================================
# 11. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42
)


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples :",
    len(X_test)
)


# =========================================================
# 12. MODEL
# =========================================================

model = Pipeline(

    steps=[

        (
            "preprocessing",

            preprocessor
        ),

        (
            "model",

            RandomForestRegressor(

                n_estimators=100,

                max_depth=20,

                min_samples_leaf=5,

                random_state=42,

                n_jobs=-1
            )
        )
    ]
)


# =========================================================
# 13. TRAIN
# =========================================================

print(
    "\nTraining delay-duration model..."
)

model.fit(
    X_train,
    y_train
)

print(
    "Training completed!"
)


# =========================================================
# 14. PREDICT
# =========================================================

predictions = model.predict(
    X_test
)


# =========================================================
# 15. METRICS
# =========================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


# =========================================================
# 16. RESULTS
# =========================================================

print("\n" + "=" * 60)
print("DELAY DURATION MODEL RESULTS")
print("=" * 60)

print(
    f"MAE : {mae:.2f} minutes"
)

print(
    f"RMSE: {rmse:.2f} minutes"
)

print(
    f"R²  : {r2:.3f}"
)


# =========================================================
# 17. EXAMPLE PREDICTIONS
# =========================================================

print("\nExample predictions:")

comparison = pd.DataFrame({

    "Actual Delay": y_test.iloc[:10].values,

    "Predicted Delay": predictions[:10]

})

print(
    comparison.round(2)
)


# =========================================================
# 18. SAVE MODEL
# =========================================================

os.makedirs(
    "model",
    exist_ok=True
)

joblib.dump(
    model,
    "model/delay_duration_model.pkl"
)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(
    "model/delay_duration_model.pkl"
)