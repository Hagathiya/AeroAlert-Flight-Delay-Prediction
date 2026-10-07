"""
AeroAlert - Final Delay Probability Model

Target:
    ARR_DELAY > 15 minutes

Only pre-flight information is used as input.

No post-flight information such as ARR_DELAY, DEP_DELAY,
actual arrival/departure times, taxi times, wheels-off/on,
or delay-cause fields is used as a feature.
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import create_features, ALLOWED_RAW_INPUTS, LEAKAGE_COLUMNS

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "dataset/flights.csv"
MODEL_PATH = "model/delay_classifier.pkl"
META_PATH = "model/delay_classifier_meta.json"

TEST_YEAR = 2023
EXCLUDE_YEAR = 2020

# Keep training manageable while still using substantial data.
ROWS_PER_YEAR = 300_000


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AEROALERT - DELAY CLASSIFIER TRAINING")
print("=" * 70)

print("\nLoading dataset...")

usecols = ALLOWED_RAW_INPUTS + ["ARR_DELAY"]

df = pd.read_csv(
    DATASET_PATH,
    usecols=usecols,
    low_memory=False
)

print(f"Total rows loaded: {len(df):,}")


# ============================================================
# CREATE TARGET
# ============================================================

df["ARR_DELAY"] = pd.to_numeric(
    df["ARR_DELAY"],
    errors="coerce"
)

# Remove flights where actual arrival delay is unavailable.
df = df.dropna(subset=["ARR_DELAY"]).copy()

# 1 = arrival delay greater than 15 minutes
# 0 = otherwise
df["TARGET"] = (df["ARR_DELAY"] > 15).astype(int)

df["_YEAR"] = pd.to_datetime(
    df["FL_DATE"],
    errors="coerce"
).dt.year


print("\nOverall target distribution:")

print(
    df["TARGET"]
    .value_counts(normalize=True)
    .sort_index()
    .rename({
        0: "15 min or less",
        1: "more than 15 min"
    })
)


# ============================================================
# TEMPORAL SPLIT
# ============================================================

print("\nApplying temporal split...")

train_df = df[
    (~df["_YEAR"].isin([EXCLUDE_YEAR])) &
    (df["_YEAR"] != TEST_YEAR)
].copy()

test_df = df[
    df["_YEAR"] == TEST_YEAR
].copy()


print(
    "Training years:",
    sorted(train_df["_YEAR"].dropna().unique().tolist())
)

print(
    "Testing year:",
    TEST_YEAR
)

print(
    f"Full training rows: {len(train_df):,}"
)

print(
    f"Test rows: {len(test_df):,}"
)


# ============================================================
# SAMPLE TRAINING DATA
# ============================================================

if ROWS_PER_YEAR is not None:

    parts = []

    for year, group in train_df.groupby("_YEAR"):

        n = min(
            ROWS_PER_YEAR,
            len(group)
        )

        parts.append(
            group.sample(
                n=n,
                random_state=42
            )
        )

    train_df = pd.concat(
        parts,
        ignore_index=True
    )


print(
    f"Training rows after sampling: {len(train_df):,}"
)


# ============================================================
# CREATE FEATURES
# ============================================================

print("\nCreating leak-free features...")

X_train = create_features(
    train_df[ALLOWED_RAW_INPUTS]
)

X_test = create_features(
    test_df[ALLOWED_RAW_INPUTS]
)

y_train = train_df["TARGET"]
y_test = test_df["TARGET"]


# ============================================================
# LEAKAGE CHECK
# ============================================================

print("\nChecking for leakage...")

for column in LEAKAGE_COLUMNS:

    if column in X_train.columns:

        raise RuntimeError(
            f"DATA LEAKAGE DETECTED: {column}"
        )

print("Leakage check passed.")


# ============================================================
# FEATURE GROUPS
# ============================================================

categorical_features = [
    "AIRLINE_CODE",
    "ORIGIN",
    "DEST",
    "TIME_OF_DAY",
    "ROUTE",
    "AIRLINE_ROUTE",
    "AIRLINE_MONTH",
    "ORIGIN_MONTH",
]


numerical_features = [
    "CRS_ELAPSED_TIME",
    "DISTANCE",
    "DISTANCE_PER_MINUTE",
    "CRS_SPEED_MPH",
    "YEAR",
    "MONTH",
    "DAY_OF_MONTH",
    "DAY_OF_WEEK",
    "IS_WEEKEND",
    "DEP_HOUR",
    "DEP_MINUTE",
    "ARR_HOUR",
    "ARR_MINUTE",
    "DEP_MINUTES_OF_DAY",
    "ARR_MINUTES_OF_DAY",
    "CLOCK_ELAPSED_MINUTES",
    "IS_OVERNIGHT",
]


# ============================================================
# PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


numerical_pipeline = Pipeline(
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


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        )
    ],
    remainder="drop"
)


# ============================================================
# MODEL
# ============================================================

print("\nBuilding Logistic Regression model...")

model = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=1.0,
                random_state=42
            )
        )
    ]
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining delay classifier...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# PREDICTIONS
# ============================================================

probabilities = model.predict_proba(
    X_test
)[:, 1]

predictions = (
    probabilities >= 0.50
).astype(int)


# ============================================================
# METRICS
# ============================================================

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

pr_auc = average_precision_score(
    y_test,
    probabilities
)

cm = confusion_matrix(
    y_test,
    predictions
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("DELAY CLASSIFIER RESULTS")
print("=" * 70)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1        : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

print(
    f"PR-AUC    : {pr_auc:.4f}"
)

print("\nConfusion Matrix:")

print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "15 min or less",
            "More than 15 min"
        ],
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    "model",
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "model_type":
        "Logistic Regression",

    "target":
        "ARR_DELAY > 15 minutes",

    "train_years":
        sorted(
            train_df["_YEAR"]
            .dropna()
            .unique()
            .astype(int)
            .tolist()
        ),

    "test_year":
        TEST_YEAR,

    "train_rows":
        int(len(train_df)),

    "test_rows":
        int(len(test_df)),

    "metrics": {

        "precision":
            round(float(precision), 4),

        "recall":
            round(float(recall), 4),

        "f1":
            round(float(f1), 4),

        "roc_auc":
            round(float(roc_auc), 4),

        "pr_auc":
            round(float(pr_auc), 4)
    },

    "features":
        ALLOWED_RAW_INPUTS,

    "leakage_excluded":
        LEAKAGE_COLUMNS
}


with open(
    META_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=2
    )


print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(
    MODEL_PATH
)

print(
    META_PATH
)

print("\nDelay classifier training complete.")