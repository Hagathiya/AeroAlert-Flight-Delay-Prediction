"""
train_cancellation_final.py
===========================
Trains the AeroAlert flight-cancellation classifier.

Design decisions
----------------
* 2020 excluded from training – COVID-19 inflated cancellation rates 3-4x
  above normal; including it would teach the model patterns that do not hold
  for a "will my flight be cancelled?" query in 2023+.
* Temporal split (not random shuffle):
    train  – 2019, 2021, 2022
    test   – 2023 (held out entirely)
  This prevents information from the same storm/disruption day leaking across
  the split boundary, giving an honest estimate of generalisation.
* Imbalance handling: GradientBoostingClassifier with scale_pos_weight
  equivalent (sample_weight) + CalibratedClassifierCV(isotonic) so that
  predict_proba() reflects true probabilities (~2.6 % base rate) instead of
  the artificially inflated values produced by class_weight="balanced".
* All features come from src.features.create_features() – zero post-flight
  leakage fields.
* Unseen airport/airline/route categories are handled by the pipeline's
  OneHotEncoder(handle_unknown="ignore") – it emits an all-zero row rather
  than crashing.
"""

import os
import sys
import json
import time
import warnings
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# ── Make sure src/ is importable when run from the project root ──────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.features import create_features, LEAKAGE_COLUMNS, ALLOWED_RAW_INPUTS

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATASET_PATH   = "dataset/flights.csv"
OLD_MODEL_PATH = "model/cancellation_model.pkl"
NEW_MODEL_PATH = "model/cancellation_model.pkl"  # overwrites old one

# Years deliberately excluded from training because of pandemic anomaly
EXCLUDE_TRAIN_YEARS = {2020}
TEST_YEAR  = 2023

# How many rows to sample from each training year (keeps runtime manageable
# while using a representative slice of >2 M rows).
# Set to None to use all rows.
ROWS_PER_YEAR = 200_000

RAW_COLS = ALLOWED_RAW_INPUTS  # ['FL_DATE','AIRLINE_CODE',...]

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("AEROALERT – CANCELLATION MODEL TRAINING")
print("=" * 65)

t0 = time.time()
print("\n[1] Loading dataset …")

usecols = RAW_COLS + ["CANCELLED"]
df_full = pd.read_csv(DATASET_PATH, usecols=usecols, low_memory=False)
df_full["CANCELLED"] = pd.to_numeric(df_full["CANCELLED"], errors="coerce").fillna(0).astype(int)
df_full["_YEAR"] = pd.to_datetime(df_full["FL_DATE"], errors="coerce").dt.year

print(f"  Total rows loaded : {len(df_full):,}")
print(f"  Cancellation rate : {df_full['CANCELLED'].mean()*100:.3f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 3. TEMPORAL SPLIT  (train = 2019/2021/2022, test = 2023)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2] Applying temporal split …")

train_mask = (~df_full["_YEAR"].isin(EXCLUDE_TRAIN_YEARS)) & (df_full["_YEAR"] != TEST_YEAR)
test_mask  = df_full["_YEAR"] == TEST_YEAR

df_train_full = df_full[train_mask].copy()
df_test       = df_full[test_mask].copy()

print(f"  Train years : {sorted(df_train_full['_YEAR'].unique())}")
print(f"  Test year   : {TEST_YEAR}")
print(f"  Train rows  : {len(df_train_full):,}")
print(f"  Test rows   : {len(df_test):,}")

# Per-year stratified sample (avoids 1M+ rows making GBM very slow)
if ROWS_PER_YEAR:
    parts = []
    for yr, grp in df_train_full.groupby("_YEAR"):
        n = min(ROWS_PER_YEAR, len(grp))
        parts.append(grp.sample(n=n, random_state=42))
    df_train = pd.concat(parts, ignore_index=True)
    print(f"  After sampling : {len(df_train):,} train rows")
else:
    df_train = df_train_full

# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING  (centralized, no leakage)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[3] Running feature engineering …")

y_train = df_train["CANCELLED"].values
y_test  = df_test["CANCELLED"].values

X_train = create_features(df_train[RAW_COLS])
X_test  = create_features(df_test[RAW_COLS])

# Sanity-check: no post-flight leakage survived
for col in LEAKAGE_COLUMNS:
    assert col not in X_train.columns, f"LEAKAGE: {col} found in features!"

print(f"  Train feature matrix : {X_train.shape}")
print(f"  Test  feature matrix : {X_test.shape}")
print(f"  Train cancellation   : {y_train.sum():,} / {len(y_train):,}  ({y_train.mean()*100:.2f}%)")
print(f"  Test  cancellation   : {y_test.sum():,} / {len(y_test):,}  ({y_test.mean()*100:.2f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. FEATURE COLUMN GROUPS
# ─────────────────────────────────────────────────────────────────────────────

CATEGORICAL = ["AIRLINE_CODE", "ORIGIN", "DEST", "TIME_OF_DAY",
               "ROUTE", "AIRLINE_ROUTE", "AIRLINE_MONTH", "ORIGIN_MONTH"]
NUMERICAL   = ["CRS_ELAPSED_TIME", "DISTANCE", "DISTANCE_PER_MINUTE",
               "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "IS_WEEKEND",
               "DEP_HOUR", "DEP_MINUTE", "ARR_HOUR", "ARR_MINUTE",
               "DEP_MINUTES_OF_DAY", "ARR_MINUTES_OF_DAY",
               "CLOCK_ELAPSED_MINUTES", "IS_OVERNIGHT", "CRS_SPEED_MPH"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CATEGORICAL),
        ("num", num_pipeline, NUMERICAL),
    ],
    remainder="drop",
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. CLASS WEIGHTS FOR IMBALANCE
# ─────────────────────────────────────────────────────────────────────────────

pos = y_train.sum()
neg = len(y_train) - pos
scale_pos_weight = neg / pos          # e.g. ~37 for 2.6% cancellation rate
print(f"\n[4] Class imbalance ratio  neg/pos = {scale_pos_weight:.1f}")

# sample_weight array: positive class samples get weight = scale_pos_weight
sample_weights = np.where(y_train == 1, scale_pos_weight, 1.0)

# ─────────────────────────────────────────────────────────────────────────────
# 7. BUILD MODELS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[5] Building candidate models …")

# ── 7a. Baseline LogisticRegression (mirrors old model for comparison) ───────
lr_pipe = Pipeline([
    ("preprocessing", ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True),
             ["AIRLINE_CODE", "ORIGIN", "DEST", "TIME_OF_DAY"]),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ]),
             ["CRS_ELAPSED_TIME", "DISTANCE", "MONTH", "DAY_OF_WEEK",
              "IS_WEEKEND", "DEP_HOUR", "DEP_MINUTE", "ARR_HOUR",
              "ARR_MINUTE", "DISTANCE_PER_MINUTE"]),
        ],
        remainder="drop"
    )),
    ("model", LogisticRegression(class_weight="balanced", max_iter=500, C=1.0, random_state=42)),
])

# ── 7b. Improved: GBM with isotonic calibration ─────────────────────────────
gbm_base = GradientBoostingClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.8,
    min_samples_leaf=40,
    random_state=42,
)

gbm_pipe = Pipeline([
    ("preprocessing", preprocessor),
    ("model", gbm_base),
])

# ─────────────────────────────────────────────────────────────────────────────
# 8. TRAIN BASELINE  (LogisticRegression)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[6] Training baseline LogisticRegression …")
t1 = time.time()
lr_pipe.fit(X_train, y_train)
print(f"  Done in {time.time()-t1:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 9. TRAIN GBM
# ─────────────────────────────────────────────────────────────────────────────

print("\n[7] Training GradientBoosting classifier …")
t1 = time.time()
gbm_pipe.fit(X_train, y_train, model__sample_weight=sample_weights)
print(f"  Done in {time.time()-t1:.1f}s")

# ── Calibrate GBM so predict_proba reflects true probabilities ───────────────
print("\n[8] Calibrating GBM with isotonic regression …")
t1 = time.time()

# Build a new pipeline that wraps the already-fitted preprocessor + calibrated gbm
gbm_preprocessor_fitted = gbm_pipe.named_steps["preprocessing"]
X_train_transformed = gbm_preprocessor_fitted.transform(X_train)

calibrated_gbm = CalibratedClassifierCV(
    gbm_pipe.named_steps["model"],
    method="isotonic",
    cv="prefit",
)
calibrated_gbm.fit(X_train_transformed, y_train)
print(f"  Done in {time.time()-t1:.1f}s")

# Wrap into a single predict-able object for inference
class CalibratedPipeline:
    """Thin wrapper that applies the fitted preprocessor then calibrated clf."""
    def __init__(self, preprocessor, calibrated_clf):
        self.preprocessor     = preprocessor
        self.calibrated_clf   = calibrated_clf

    def predict_proba(self, X):
        Xt = self.preprocessor.transform(X)
        return self.calibrated_clf.predict_proba(Xt)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

calib_pipeline = CalibratedPipeline(gbm_preprocessor_fitted, calibrated_gbm)

# ─────────────────────────────────────────────────────────────────────────────
# 10. EVALUATE  (test year 2023)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(name, model, X, y, threshold=0.5, use_sample_weight=False):
    """Print a full suite of classification metrics."""
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= threshold).astype(int)

    p  = precision_score(y, pred, zero_division=0)
    r  = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    roc = roc_auc_score(y, proba)
    pr  = average_precision_score(y, proba)
    cm  = confusion_matrix(y, pred)

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  Precision : {p:.4f}")
    print(f"  Recall    : {r:.4f}")
    print(f"  F1 score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc:.4f}")
    print(f"  PR-AUC    : {pr:.4f}")
    print(f"  Threshold : {threshold}")
    print(f"  Confusion matrix (rows=actual, cols=pred):")
    print(f"    TN={cm[0,0]:>7,}  FP={cm[0,1]:>6,}")
    print(f"    FN={cm[1,0]:>7,}  TP={cm[1,1]:>6,}")
    print(classification_report(y, pred, target_names=["On time", "Cancelled"],
                                 zero_division=0))
    return {"precision": p, "recall": r, "f1": f1, "roc_auc": roc, "pr_auc": pr}


print("\n" + "=" * 65)
print("EVALUATION ON TEST YEAR 2023")
print("=" * 65)

lr_metrics   = evaluate("Baseline LogisticRegression (balanced)",   lr_pipe,       X_test, y_test)
gbm_raw_metrics = evaluate("GBM (pre-calibration, threshold=0.5)",  gbm_pipe,      X_test, y_test)
cal_metrics  = evaluate("GBM + Isotonic Calibration (threshold=0.5)", calib_pipeline, X_test, y_test)

# Calibration sanity check: actual cancellation rate vs mean predicted probability
raw_pred_mean   = gbm_pipe.predict_proba(X_test)[:, 1].mean()
cal_pred_mean   = calib_pipeline.predict_proba(X_test)[:, 1].mean()
actual_rate     = y_test.mean()
print(f"\n  Actual cancellation rate       : {actual_rate*100:.3f}%")
print(f"  GBM mean predicted probability : {raw_pred_mean*100:.3f}%")
print(f"  Calibrated mean probability    : {cal_pred_mean*100:.3f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 11. SELECT FINAL MODEL
# ─────────────────────────────────────────────────────────────────────────────
# We pick the GBM + isotonic calibration pipeline.
# Rationale:
#   – Calibrated probabilities ≈ true cancellation rate (critical for passengers)
#   – GBM captures non-linear feature interactions LR misses
#   – sample_weight handles imbalance without distorting probability estimates
# The final model object wraps everything needed for inference.

print("\n" + "=" * 65)
print("SAVING FINAL MODEL")
print("=" * 65)

os.makedirs("model", exist_ok=True)
joblib.dump(calib_pipeline, NEW_MODEL_PATH)
print(f"  Saved : {NEW_MODEL_PATH}")

# Also persist metadata for the dashboard / about page
meta = {
    "model_type": "GradientBoostingClassifier + Isotonic Calibration",
    "train_years": [2019, 2021, 2022],
    "test_year": TEST_YEAR,
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "cancellation_rate_train": float(round(y_train.mean() * 100, 3)),
    "cancellation_rate_test":  float(round(y_test.mean()  * 100, 3)),
    "metrics": {
        "precision": round(cal_metrics["precision"], 4),
        "recall":    round(cal_metrics["recall"],    4),
        "f1":        round(cal_metrics["f1"],        4),
        "roc_auc":   round(cal_metrics["roc_auc"],   4),
        "pr_auc":    round(cal_metrics["pr_auc"],    4),
    },
    "baseline_metrics": {
        "precision": round(lr_metrics["precision"], 4),
        "recall":    round(lr_metrics["recall"],    4),
        "f1":        round(lr_metrics["f1"],        4),
        "roc_auc":   round(lr_metrics["roc_auc"],   4),
        "pr_auc":    round(lr_metrics["pr_auc"],    4),
    },
    "categorical_features": CATEGORICAL,
    "numerical_features": NUMERICAL,
    "leakage_excluded": LEAKAGE_COLUMNS,
}

meta_path = "model/cancellation_model_meta.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"  Metadata : {meta_path}")

print(f"\nTotal training time: {time.time()-t0:.1f}s")
print("\n✓ CANCELLATION MODEL TRAINING COMPLETE")
