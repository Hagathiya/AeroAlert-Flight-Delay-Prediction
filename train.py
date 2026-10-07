import pandas as pd
import joblib

from src.features import create_features

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# =========================================================
# 1. LOAD DATASET
# =========================================================

print("Loading Dataset...")

df = pd.read_csv("dataset/flights.csv", nrows=100000)

print("Dataset Loaded Successfully!")
print("Original Shape:", df.shape)


# =========================================================
# 2. CREATE TARGET
# =========================================================

# 1 = Flight delayed by more than 15 minutes
# 0 = Flight on time

df["TARGET"] = (df["ARR_DELAY"] > 15).astype(int)


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

df = df[features + ["TARGET"]]


# =========================================================
# 4. REMOVE MISSING VALUES
# =========================================================

df = df.dropna()

print("Cleaned Shape:", df.shape)


# =========================================================
# 5. FEATURE ENGINEERING (Centralized from src.features)
# =========================================================

df = create_features(df)


# =========================================================
# 6. SEPARATE FEATURES AND TARGET
# =========================================================

X = df.drop(columns=["TARGET"])
y = df["TARGET"]


# =========================================================
# 7. DEFINE FEATURE TYPES
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
            OneHotEncoder(handle_unknown="ignore"),
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
# 10. MODEL COMPARISON
# =========================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=15,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


results = {}


print("\n==========================================")
print("           MODEL COMPARISON")
print("==========================================")


for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

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

    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall   : {recall:.2f}")
    print(f"F1 Score : {f1:.2f}")


# =========================================================
# 11. HYPERPARAMETER TUNING
# =========================================================

print("\n==========================================")
print("       LOGISTIC REGRESSION TUNING")
print("==========================================")


logistic_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),

        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ]
)


parameters = {
    "model__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ]
}


grid_search = GridSearchCV(
    logistic_pipeline,
    parameters,
    cv=5,
    scoring="f1",
    n_jobs=-1
)


print("Running 5-Fold Cross Validation...")

grid_search.fit(X_train, y_train)


print("\nBest Parameters:")
print(grid_search.best_params_)

print(
    f"Best Cross-Validation F1: "
    f"{grid_search.best_score_:.2f}"
)


# =========================================================
# 12. EVALUATE TUNED MODEL
# =========================================================

tuned_model = grid_search.best_estimator_

tuned_predictions = tuned_model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    tuned_predictions
)

precision = precision_score(
    y_test,
    tuned_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    tuned_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    tuned_predictions,
    zero_division=0
)


print("\n==========================================")
print("        TUNED MODEL RESULTS")
print("==========================================")

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall   : {recall:.2f}")
print(f"F1 Score : {f1:.2f}")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        tuned_predictions
    )
)


# =========================================================
# 13. SAVE FINAL MODEL
# =========================================================

joblib.dump(
    tuned_model,
    "model/aeroalert_model.pkl"
)


print("\n==========================================")
print(" FINAL AEROALERT MODEL SAVED SUCCESSFULLY")
print("==========================================")

print(
    "File: model/aeroalert_model.pkl"
)