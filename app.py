from flask import Flask, render_template, request
import json
import os
from datetime import datetime

from airport_utils import get_distance, get_airport_info
from validation import validate_flight_input
from prediction_pipeline import predict_flight


# =========================================================
# CREATE FLASK APP
# =========================================================

app = Flask(__name__)


# =========================================================
# RECENT PREDICTIONS
# =========================================================
# In-memory only. Resets when Flask restarts.

recent_predictions = []
MAX_HISTORY = 10


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    # Preserve submitted values if validation fails
    form_data = request.form.to_dict()

    try:

        # -------------------------------------------------
        # 1. READ FORM INPUT
        # -------------------------------------------------

        airline = request.form.get(
            "airline", ""
        ).strip().upper()

        origin = request.form.get(
            "origin", ""
        ).strip().upper()

        destination = request.form.get(
            "destination", ""
        ).strip().upper()

        flight_date = request.form.get(
            "flight_date", ""
        ).strip()


        # -------------------------------------------------
        # 2. READ NUMERIC INPUTS
        # -------------------------------------------------

        try:

            dep_time = int(
                request.form["departure_time"]
            )

            arr_time = int(
                request.form["arrival_time"]
            )

            duration = float(
                request.form["duration"]
            )

        except (ValueError, KeyError):

            return render_template(
                "index.html",
                error=(
                    "Times and duration must be numbers. "
                    "Use HHMM for times, like 0730."
                ),
                form=form_data,
            )


        # -------------------------------------------------
        # 3. DATE VALIDATION
        # -------------------------------------------------

        if not flight_date:

            return render_template(
                "index.html",
                error="Pick a flight date.",
                form=form_data,
            )


        # -------------------------------------------------
        # 4. CALCULATE DISTANCE
        # -------------------------------------------------

        distance = get_distance(
            origin,
            destination
        )

        if distance is None:

            return render_template(
                "index.html",
                error=(
                    f"Couldn't find '{origin}' or "
                    f"'{destination}'. Use 3-letter "
                    f"IATA codes, like JFK."
                ),
                form=form_data,
            )


        # -------------------------------------------------
        # 5. EXISTING INPUT VALIDATION
        # -------------------------------------------------

        error = validate_flight_input(
            airline,
            origin,
            destination,
            dep_time,
            arr_time,
            duration,
            distance
        )

        if error:

            return render_template(
                "index.html",
                error=error,
                form=form_data,
            )


        # -------------------------------------------------
        # 6. BUILD PRE-FLIGHT INPUT
        # -------------------------------------------------

        flight = {

            "FL_DATE": flight_date,

            "AIRLINE_CODE": airline,

            "ORIGIN": origin,

            "DEST": destination,

            "CRS_DEP_TIME": dep_time,

            "CRS_ARR_TIME": arr_time,

            "CRS_ELAPSED_TIME": duration,

            "DISTANCE": distance,
        }


        # -------------------------------------------------
        # 7. RUN ALL THREE MODELS
        # -------------------------------------------------

        prediction = predict_flight(
            flight
        )


        # -------------------------------------------------
        # 8. AIRPORT INFORMATION
        # -------------------------------------------------

        origin_info = get_airport_info(
            origin
        )

        dest_info = get_airport_info(
            destination
        )

        origin_name = (
            origin_info[3]
            if origin_info
            else origin
        )

        dest_name = (
            dest_info[3]
            if dest_info
            else destination
        )


        # -------------------------------------------------
        # 9. RESULT STATUS
        # -------------------------------------------------

        cancellation_percentage = prediction[
            "cancellation_percentage"
        ]

        delay_percentage = prediction[
            "delay_percentage"
        ]


        # Overall status is based primarily on
        # cancellation risk, then delay probability.

        if cancellation_percentage >= 50:

            status = "high-risk"

            result = "High cancellation risk"

        elif delay_percentage >= 50:

            status = "delayed"

            result = "Higher delay probability"

        elif cancellation_percentage >= 20:

            status = "moderate"

            result = "Moderate cancellation risk"

        else:

            status = "ontime"

            result = "Lower disruption risk"


        # -------------------------------------------------
        # 10. RECENT PREDICTION HISTORY
        # -------------------------------------------------

        recent_predictions.insert(
            0,
            {
                "time": datetime.now().strftime("%H:%M"),

                "route": (
                    f"{origin} - {destination}"
                ),

                "airline": airline,

                "status": status,

                "result": result,

                "probability": delay_percentage,

                "cancellation_probability":
                    cancellation_percentage,
            }
        )

        del recent_predictions[
            MAX_HISTORY:
        ]


        # -------------------------------------------------
        # 11. SEND RESULTS TO TEMPLATE
        # -------------------------------------------------

        return render_template(

            "result.html",

            # Existing/basic information
            result=result,

            status=status,

            airline=airline,

            origin=origin,

            destination=destination,

            origin_name=origin_name,

            dest_name=dest_name,

            distance=int(distance),

            duration=int(duration),

            dep_time=f"{dep_time:04d}",

            arr_time=f"{arr_time:04d}",

            flight_date=flight_date,


            # Cancellation
            cancellation_probability=(
                prediction[
                    "cancellation_probability"
                ]
            ),

            cancellation_percentage=(
                prediction[
                    "cancellation_percentage"
                ]
            ),

            cancellation_status=(
                prediction[
                    "cancellation_status"
                ]
            ),


            # Delay probability
            delay_probability=(
                prediction[
                    "delay_probability"
                ]
            ),

            delay_percentage=(
                prediction[
                    "delay_percentage"
                ]
            ),

            delay_status=(
                prediction[
                    "delay_status"
                ]
            ),

            typical_delay_min=prediction["typical_delay_min"],
            typical_delay_median=prediction["typical_delay_median"],
            typical_delay_max=prediction["typical_delay_max"],
            # Duration
            predicted_delay=(
                prediction[
                    "predicted_delay"
                ]
            ),

            expected_delay=(
                prediction[
                    "expected_delay"
                ]
            ),

            delay_range_min=(
                prediction[
                    "delay_range_min"
                ]
            ),

            delay_range_max=(
                prediction[
                    "delay_range_max"
                ]
            ),


            # Arrival
            scheduled_arrival=(
                prediction[
                    "scheduled_arrival"
                ]
            ),

            estimated_arrival=(
                prediction[
                    "estimated_arrival"
                ]
            ),

            arrival_window_start=(
                prediction[
                    "arrival_window_start"
                ]
            ),

            arrival_window_end=(
                prediction[
                    "arrival_window_end"
                ]
            ),

            arrival_note=(
                prediction[
                    "arrival_note"
                ]
            ),


            # General
            disclaimer=(
                prediction[
                    "disclaimer"
                ]
            ),

            is_estimate=(
                prediction[
                    "is_estimate"
                ]
            ),
        )


    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as e:

        print(
            "Prediction error:",
            repr(e)
        )

        return render_template(
            "index.html",
            error=(
                "Something went wrong processing "
                f"that flight: {e}"
            ),
            form=form_data,
        )


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():

    stats = {}

    stats_path = os.path.join(
        "static",
        "charts",
        "stats.json"
    )

    if os.path.exists(stats_path):

        with open(
            stats_path,
            encoding="utf-8"
        ) as f:

            stats = json.load(f)


    return render_template(
        "dashboard.html",

        stats=stats,

        charts_ready=bool(stats),

        recent=recent_predictions,
    )


# =========================================================
# ABOUT / MODEL INFO
# =========================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5050
    )