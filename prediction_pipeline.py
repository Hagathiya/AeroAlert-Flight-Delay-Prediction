"""
AeroAlert - Unified Passenger Prediction Pipeline

Combines:
1. Cancellation probability
2. Probability of arrival delay > 15 minutes
3. Historical delay-duration context
4. Estimated arrival-time window

All predictions use only pre-flight information.
"""

import os
import joblib

from src.features import create_features


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

CANCELLATION_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "cancellation_model_v2.pkl"
)

DELAY_CLASSIFIER_PATH = os.path.join(
    MODEL_DIR,
    "delay_classifier.pkl"
)


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading AeroAlert models...")

cancellation_model = joblib.load(
    CANCELLATION_MODEL_PATH
)

delay_classifier = joblib.load(
    DELAY_CLASSIFIER_PATH
)

print("All models loaded successfully.")


# ============================================================
# HISTORICAL DELAY STATISTICS
# ============================================================
#
# Calculated from the AeroAlert dataset for flights with:
# CANCELLED == 0
# ARR_DELAY > 15 minutes
#
# These are NOT personalized predictions.
# They are historical context used to communicate the
# typical delay magnitude when a flight is delayed.
#

TYPICAL_DELAY_MIN = 25
TYPICAL_DELAY_MEDIAN = 41
TYPICAL_DELAY_MAX = 78


# ============================================================
# PASSENGER INPUT VALIDATION
# ============================================================

REQUIRED_INPUTS = [
    "FL_DATE",
    "AIRLINE_CODE",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
]


def validate_input(flight_data):
    """
    Validate passenger-provided pre-flight information.
    """

    missing = []

    for field in REQUIRED_INPUTS:
        if field not in flight_data:
            missing.append(field)

    if missing:
        return False, f"Missing required fields: {missing}"

    if str(flight_data["ORIGIN"]).upper() == str(
        flight_data["DEST"]
    ).upper():

        return False, "Origin and destination cannot be the same."

    try:

        elapsed = float(
            flight_data["CRS_ELAPSED_TIME"]
        )

        if elapsed <= 0:
            return False, (
                "Scheduled flight duration must be greater than 0."
            )

    except (ValueError, TypeError):

        return False, (
            "Invalid scheduled flight duration."
        )

    try:

        distance = float(
            flight_data["DISTANCE"]
        )

        if distance <= 0:
            return False, (
                "Distance must be greater than 0."
            )

    except (ValueError, TypeError):

        return False, (
            "Invalid flight distance."
        )

    return True, "Valid input."


# ============================================================
# TIME HELPERS
# ============================================================

def convert_to_minutes(time_value):
    """
    Convert HHMM / HH:MM into minutes after midnight.
    """

    value = str(time_value).strip()

    if ":" in value:

        parts = value.split(":")

        hour = int(parts[0])
        minute = int(parts[1])

    else:

        number = int(float(value))

        hour = number // 100
        minute = number % 100

    if hour < 0 or hour > 23:
        raise ValueError("Invalid hour.")

    if minute < 0 or minute > 59:
        raise ValueError("Invalid minute.")

    return hour * 60 + minute


def format_time(minutes):
    """
    Convert minutes after midnight to HH:MM AM/PM.
    """

    minutes = int(round(minutes)) % 1440

    hour = minutes // 60
    minute = minutes % 60

    suffix = "AM"

    if hour >= 12:
        suffix = "PM"

    display_hour = hour % 12

    if display_hour == 0:
        display_hour = 12

    return f"{display_hour}:{minute:02d} {suffix}"


# ============================================================
# ARRIVAL WINDOW
# ============================================================

def calculate_arrival_window(
    scheduled_arrival,
    lower_delay,
    upper_delay
):
    """
    Calculate a historical-context arrival window.

    The window is based on the historical delay range used
    by AeroAlert, not an exact personalized delay prediction.
    """

    scheduled_arrival_minutes = convert_to_minutes(
        scheduled_arrival
    )

    lower_arrival = (
        scheduled_arrival_minutes
        + lower_delay
    )

    upper_arrival = (
        scheduled_arrival_minutes
        + upper_delay
    )
    
    return {

    "scheduled_arrival":
        format_time(
            scheduled_arrival_minutes
        ),

    "estimated_arrival":
        format_time(
            scheduled_arrival_minutes
            + TYPICAL_DELAY_MEDIAN
        ),

    "arrival_window_start":
        format_time(
            lower_arrival
        ),

    "arrival_window_end":
        format_time(
            upper_arrival
        )
}


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict_flight(flight_data):
    """
    Generate all passenger-facing AeroAlert predictions.

    Parameters
    ----------
    flight_data : dict

    Returns
    -------
    dict
    """

    # --------------------------------------------------------
    # 1. VALIDATE INPUT
    # --------------------------------------------------------

    valid, message = validate_input(
        flight_data
    )

    if not valid:
        raise ValueError(message)

    # --------------------------------------------------------
    # 2. CREATE CENTRALIZED FEATURES
    # --------------------------------------------------------

    features = create_features(
        flight_data
    )

    # --------------------------------------------------------
    # 3. CANCELLATION PROBABILITY
    # --------------------------------------------------------

    cancellation_probability = float(
        cancellation_model.predict_proba(
            features
        )[0][1]
    )

    # --------------------------------------------------------
    # 4. DELAY > 15 MINUTES PROBABILITY
    # --------------------------------------------------------

    delay_probability = float(
        delay_classifier.predict_proba(
            features
        )[0][1]
    )

    # --------------------------------------------------------
    # 5. HISTORICAL DELAY CONTEXT
    # --------------------------------------------------------
    #
    # We deliberately do NOT use the weak regression model
    # to claim an exact personalized delay.
    #
    # These values describe historical delay magnitude among
    # flights that actually experienced >15 minutes of delay.
    #

    typical_delay_min = TYPICAL_DELAY_MIN
    typical_delay_median = TYPICAL_DELAY_MEDIAN
    typical_delay_max = TYPICAL_DELAY_MAX

    # --------------------------------------------------------
    # 6. ARRIVAL WINDOW
    # --------------------------------------------------------

    arrival_info = calculate_arrival_window(
        flight_data["CRS_ARR_TIME"],
        typical_delay_min,
        typical_delay_max
    )

    # --------------------------------------------------------
    # 7. STATUS
    # --------------------------------------------------------

    if cancellation_probability >= 0.50:

        cancellation_status = (
            "High cancellation risk"
        )

    elif cancellation_probability >= 0.20:

        cancellation_status = (
            "Moderate cancellation risk"
        )

    else:

        cancellation_status = (
            "Lower cancellation risk"
        )

    if delay_probability >= 0.50:

        delay_status = (
            "Higher probability of >15 min delay"
        )

    elif delay_probability >= 0.30:

        delay_status = (
            "Moderate probability of >15 min delay"
        )

    else:

        delay_status = (
            "Lower probability of >15 min delay"
        )

    # --------------------------------------------------------
    # 8. FINAL RESULT
    # --------------------------------------------------------

    result = {

        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        "cancellation_probability":
            cancellation_probability,

        "cancellation_percentage":
            round(
                cancellation_probability * 100,
                1
            ),

        "delay_probability":
            delay_probability,

        "delay_percentage":
            round(
                delay_probability * 100,
                1
            ),

        # ----------------------------------------------------
        # Historical delay information
        # ----------------------------------------------------

        "typical_delay_min":
            typical_delay_min,

        "typical_delay_median":
            typical_delay_median,

        "typical_delay_max":
            typical_delay_max,

        # ----------------------------------------------------
        # Compatibility fields
        # ----------------------------------------------------
        #
        # These are kept so the current result.html does not
        # immediately break while we update the wording.
        #

        "predicted_delay":
            None,

        "expected_delay":
            None,

        "delay_range_min":
            typical_delay_min,

        "delay_range_max":
            typical_delay_max,

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        "cancellation_status":
            cancellation_status,

        "delay_status":
            delay_status,

        # ----------------------------------------------------
        # Arrival information
        # ----------------------------------------------------

        **arrival_info,

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        "is_estimate":
            True,

        "disclaimer":
            "AeroAlert provides model-based probabilities and "
            "historical delay context, not guaranteed outcomes."
    }

    # --------------------------------------------------------
    # 9. ARRIVAL NOTE
    # --------------------------------------------------------

    if cancellation_probability >= 0.50:

        result["arrival_note"] = (
            "Arrival information is conditional on the "
            "flight operating."
        )

    elif delay_probability >= 0.50:

        result["arrival_note"] = (
            "The arrival window uses historical delay "
            "patterns for flights delayed by more than "
            "15 minutes. It is not an exact prediction."
        )

    else:

        result["arrival_note"] = (
            "Arrival information uses scheduled arrival "
            "time and historical delay context."
        )

    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    sample_flight = {

        "FL_DATE": "2023-08-15",

        "AIRLINE_CODE": "AA",

        "ORIGIN": "JFK",

        "DEST": "LAX",

        "CRS_DEP_TIME": 900,

        "CRS_ARR_TIME": 1200,

        "CRS_ELAPSED_TIME": 360,

        "DISTANCE": 2475
    }

    print(
        "\nRunning sample prediction...\n"
    )

    try:

        result = predict_flight(
            sample_flight
        )

        for key, value in result.items():

            print(
                f"{key}: {value}"
            )

    except Exception as error:

        print(
            "\nPrediction failed:"
        )

        print(error)