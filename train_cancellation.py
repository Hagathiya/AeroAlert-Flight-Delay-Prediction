import os
import pandas as pd
import joblib

from src.features import create_features

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score
)


# =========================================================
# 1. LOAD DATASET
# =========================================================

print("=" * 60)
print("AEROALERT - CANCELLATION MODEL")
print("=" * 60)

print("\nLoading dataset...")

# Start with 500,000 records so training is manageable.
# We can increase this later if necessary.
df = pd.read_csv(
    "dataset/flights.csv",
    nrows=500000
)

print("Dataset loaded!")
print("Original shape:", df.shape)


# =========================================================
# 2. CREATE CANCELLATION TARGET
# =========================================================

# 1 = Cancelled
# 0 = Not cancelled

df["TARGET"] = df["CANCELLED"].astype(int)

print("\nCancellation distribution:")
print(df["TARGET"].value_counts())

print("\nCancellation percentage:")
print(
    df["TARGET"].value_counts(normalize=True) * 100
)


# =========================================================
# 3. SELECT ONLY PRE-FLIGHT FEATURES
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

df = df[features + ["TARGET"]]


# =========================================================
# 4. REMOVE MISSING VALUES
# =========================================================

df = df.dropna()

print("\nAfter cleaning:", df.shape)


# =========================================================
# 5. FEATURE ENGINEERING (Centralized from src.features)
# =========================================================

df = create_features(df)


# =========================================================
# 6. SEPARATE X AND Y
# =========================================================

X = df.drop(columns=["TARGET"])
y = df["TARGET"]


# =========================================================
# 7. FEATURE TYPES
# =========================================================

categorical_features = [
    "AIRLINE_CODE",
    "ORIGIN",
    "DEST",
    "TIME_OF_DAY"
]

numerical_features = [
    "CRS_ELAPSED_TIME",
    "DISTANCE",
    "MONTH",
    "DAY_OF_WEEK",
    "IS_WEEKEND",
    "DEP_HOUR",
    "DEP_MINUTE",
    "ARR_HOUR",
    "ARR_MINUTE",
    "DISTANCE_PER_MINUTE"
]


# =========================================================
# 8. PREPROCESSING
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
            StandardScaler(),
            numerical_features
        )
    ]
)


# =========================================================
# 9. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# =========================================================
# 10. MODEL
# =========================================================

model = Pipeline(
    steps=[

        (
            "preprocessing",
            preprocessor
        ),

        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ]
)


# =========================================================
# 11. TRAIN
# =========================================================

print("\nTraining cancellation model...")

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# =========================================================
# 12. EVALUATION
# =========================================================

predictions = model.predict(X_test)

probabilities = model.predict_proba(
    X_test
)[:, 1]


accuracy = accuracy_score(
    y_test,
    predictions
)

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

auc = roc_auc_score(
    y_test,
    probabilities
)


print("\n" + "=" * 60)
print("CANCELLATION MODEL RESULTS")
print("=" * 60)

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision:.3f}"
)

print(
    f"Recall   : {recall:.3f}"
)

print(
    f"F1 Score : {f1:.3f}"
)

print(
    f"ROC-AUC  : {auc:.3f}"
)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# =========================================================
# 13. SAVE MODEL
# =========================================================

os.makedirs(
    "model",
    exist_ok=True
)

joblib.dump(
    model,
    "model/cancellation_model.pkl"
)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(
    "model/cancellation_model.pkl"
)