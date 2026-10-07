import pandas as pd
import joblib
from src.features import create_features

# Load trained model
model = joblib.load("model/aeroalert_model.pkl")


# =========================================================
# USER INPUT
# =========================================================

print("\n==========================================")
print("             AEROALERT")
print("       Flight Delay Prediction")
print("==========================================\n")


airline = input(
    "Airline Code (Example: UA, DL, AA): "
).upper()

origin = input(
    "Origin Airport (Example: JFK): "
).upper()

destination = input(
    "Destination Airport (Example: LAX): "
).upper()

date = input(
    "Flight Date (YYYY-MM-DD): "
)

departure_time = int(
    input(
        "Scheduled Departure Time (HHMM): "
    )
)

arrival_time = int(
    input(
        "Scheduled Arrival Time (HHMM): "
    )
)

elapsed_time = float(
    input(
        "Scheduled Flight Duration (minutes): "
    )
)

distance = float(
    input(
        "Distance (miles): "
    )
)


# =========================================================
# CREATE INPUT DATA
# =========================================================

flight = {

    "FL_DATE": date,

    "AIRLINE_CODE": airline,

    "ORIGIN": origin,

    "DEST": destination,

    "CRS_DEP_TIME": departure_time,

    "CRS_ARR_TIME": arrival_time,

    "CRS_ELAPSED_TIME": elapsed_time,

    "DISTANCE": distance
}


# =========================================================
# FEATURE ENGINEERING
# =========================================================

input_data = create_features(flight)


# =========================================================
# PREDICTION
# =========================================================

prediction = model.predict(input_data)[0]

probability = model.predict_proba(
    input_data
)[0][1]


# =========================================================
# DISPLAY RESULT
# =========================================================

print("\n==========================================")
print("             AEROALERT RESULT")
print("==========================================")

if prediction == 1:

    print("⚠️ Prediction : FLIGHT LIKELY DELAYED")

else:

    print("✅ Prediction : FLIGHT LIKELY ON TIME")


print(
    f"Delay Probability: {probability * 100:.2f}%"
)

print("==========================================")