import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.features import create_features


DATA_PATH = "dataset/flights.csv"
MODEL_PATH = "model/delay_duration_conditional.pkl"


print("=" * 60)
print("AEROALERT - CONDITIONAL DELAY DURATION MODEL")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Total rows loaded: {len(df):,}")


# ---------------------------------------------------------
# 1. Keep flights that actually operated
# ---------------------------------------------------------

df = df[df["CANCELLED"] == 0].copy()


# ---------------------------------------------------------
# 2. Keep only flights delayed by more than 15 minutes
# ---------------------------------------------------------

df = df[df["ARR_DELAY"].notna()].copy()
df = df[df["ARR_DELAY"] > 15].copy()
# Use a representative sample to keep training practical on a laptop
if len(df) > 100000:
    df = df.sample(n=100000, random_state=42)

print(f"Flights with >15 min arrival delay: {len(df):,}")


# ---------------------------------------------------------
# 3. Select pre-flight inputs only
# ---------------------------------------------------------

features = [
    "FL_DATE",
    "AIRLINE_CODE",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
]


# ---------------------------------------------------------
# 4. Create leak-free features
# ---------------------------------------------------------

print("\nCreating features...")

X = create_features(df[features])

y = df["ARR_DELAY"].astype(float)


# ---------------------------------------------------------
# 5. Select model features
# ---------------------------------------------------------

categorical_features = [
    "AIRLINE_CODE",
    "ORIGIN",
    "DEST",
    "TIME_OF_DAY",
    "ROUTE",
    "AIRLINE_ROUTE",
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


# ---------------------------------------------------------
# 6. Keep only available columns
# ---------------------------------------------------------

categorical_features = [
    c for c in categorical_features if c in X.columns
]

numerical_features = [
    c for c in numerical_features if c in X.columns
]


# ---------------------------------------------------------
# 7. Train/test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ---------------------------------------------------------
# 8. Preprocessing
# ---------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numerical_features
        )
    ]
)


# ---------------------------------------------------------
# 9. Model
# ---------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=50,
    max_depth=20,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)


# ---------------------------------------------------------
# 10. Transform data
# ---------------------------------------------------------

print("\nPreparing training data...")

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# ---------------------------------------------------------
# 11. Train
# ---------------------------------------------------------

print("Training conditional delay-duration model...")

model.fit(X_train_processed, y_train)


# ---------------------------------------------------------
# 12. Evaluate
# ---------------------------------------------------------

predictions = model.predict(X_test_processed)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n" + "=" * 60)
print("CONDITIONAL DELAY MODEL RESULTS")
print("=" * 60)

print(f"MAE  : {mae:.2f} minutes")
print(f"RMSE : {rmse:.2f} minutes")
print(f"R²   : {r2:.3f}")

print("\nExample predictions:")

for actual, predicted in zip(y_test.iloc[:10], predictions[:10]):
    print(
        f"Actual: {actual:.1f} min"
        f"   Predicted: {predicted:.1f} min"
    )


# ---------------------------------------------------------
# 13. Save model + preprocessor together
# ---------------------------------------------------------

os.makedirs("model", exist_ok=True)

artifact = {
    "model": model,
    "preprocessor": preprocessor,
    "categorical_features": categorical_features,
    "numerical_features": numerical_features,
}

joblib.dump(
    artifact,
    MODEL_PATH
)

print("\nMODEL SAVED:")
print(MODEL_PATH)

print("\nTraining complete.")