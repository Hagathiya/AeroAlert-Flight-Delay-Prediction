"""
validation.py
Validates user input before it ever reaches the model.

This is what stops the "impossible input" problem: a 4-minute
transatlantic flight, or a 7am departure with a 10pm arrival but a
5-hour duration. The model becomes wildly overconfident on inputs it
never saw in training, so we reject them up front.
"""

from airport_utils import get_airport_info


# Rough sanity limits for commercial flights
MIN_DURATION_MIN = 20        # shortest realistic commercial hop
MAX_DURATION_MIN = 1200      # 20 hours, covers ultra-long-haul
MAX_TZ_OFFSET_HOURS = 14     # widest plausible timezone shift
DURATION_TOLERANCE_MIN = 90  # slack for timezone + scheduling padding


def validate_time(value, field_name):
    """Check a HHMM time is a real clock time."""
    try:
        t = int(value)
    except (TypeError, ValueError):
        return f"{field_name} must be a number in HHMM format, like 0730."

    if t < 0 or t > 2359:
        return f"{field_name} must be between 0000 and 2359."

    if t % 100 > 59:
        return f"{field_name} has invalid minutes. Use HHMM format, like 0730."

    return None


def validate_airport(code, field_name):
    """Check an airport code exists in our lookup."""
    if not code or len(code.strip()) != 3:
        return f"{field_name} must be a 3-letter airport code, like JFK."

    if get_airport_info(code) is None:
        return f"'{code.upper()}' isn't a recognised airport code. Check the spelling."

    return None


def validate_airline(code):
    """Check an airline code looks plausible."""
    if not code or not (2 <= len(code.strip()) <= 3):
        return "Airline code must be 2 or 3 letters, like UA or AAL."

    if not code.strip().isalnum():
        return "Airline code should contain only letters and numbers."

    return None


def minutes_since_midnight(hhmm):
    """Convert HHMM integer to minutes past midnight."""
    return (hhmm // 100) * 60 + (hhmm % 100)


def validate_duration_consistency(dep_time, arr_time, duration, distance):
    """
    The important one.

    Compares the scheduled duration against the clock gap between
    departure and arrival. These won't match exactly because arrival
    time is local to the destination, so a timezone shift is expected.

    The allowance is tied to distance: you can't cross ten timezones on
    a short flight. A 300-mile hop gets very little slack, a 5000-mile
    haul gets a lot.
    """
    dep_min = minutes_since_midnight(dep_time)
    arr_min = minutes_since_midnight(arr_time)

    clock_gap = arr_min - dep_min
    if clock_gap < 0:
        # arrival is the next day
        clock_gap += 24 * 60

    difference = abs(clock_gap - duration)

    # How many timezones could this route plausibly cross?
    max_tz_hours = min(MAX_TZ_OFFSET_HOURS, (distance / 500.0) + 2)
    allowed = (max_tz_hours * 60) + DURATION_TOLERANCE_MIN

    if difference > allowed:
        gap_h, gap_m = divmod(int(clock_gap), 60)
        return (
            f"These times don't add up. You've entered a {int(duration)}-minute "
            f"flight, but {dep_time:04d} to {arr_time:04d} is a {gap_h}h {gap_m}m gap "
            f"on the clock. Even allowing for timezone changes on a "
            f"{int(distance)}-mile route, that isn't possible. "
            f"Check your duration and arrival time."
        )

    return None



def validate_speed(distance, duration):
    """
    Check whether the entered flight duration is physically plausible.

    This is only a sanity check. Actual flight speed varies because of
    aircraft type, wind, routing, altitude, and other operational factors.
    """

    if duration <= 0:
        return "Flight duration must be greater than zero."

    mph = (distance / duration) * 60

    # Extremely high speed: almost certainly an input error.
    if mph > 900:
        return (
            f"The entered duration gives an estimated speed of about "
            f"{int(mph)} mph, which is unusually high for a commercial flight. "
            f"Please check the flight duration."
        )

    # Extremely low speed for a reasonably long route.
    if distance > 200 and mph < 100:
        return (
            f"The entered duration gives an estimated speed of only "
            f"{int(mph)} mph for a {int(distance)}-mile route. "
            f"Please check the flight duration."
        )

    return None


def validate_flight_input(airline, origin, destination, dep_time, arr_time, duration, distance):
    """
    Run every check and return the first error found, or None if the
    input is usable. Returns a plain string so the route can pass it
    straight to the template.
    """
    checks = [
        validate_airline(airline),
        validate_airport(origin, "Origin airport"),
        validate_airport(destination, "Destination airport"),
    ]

    for error in checks:
        if error:
            return error

    if origin.strip().upper() == destination.strip().upper():
        return "Origin and destination can't be the same airport."

    for error in [
        validate_time(dep_time, "Departure time"),
        validate_time(arr_time, "Arrival time"),
    ]:
        if error:
            return error

    if duration < MIN_DURATION_MIN:
        return f"Flight duration looks too short. Minimum is {MIN_DURATION_MIN} minutes."

    if duration > MAX_DURATION_MIN:
        return f"Flight duration looks too long. Maximum is {MAX_DURATION_MIN} minutes (20 hours)."

    for error in [
        validate_duration_consistency(int(dep_time), int(arr_time), duration, distance),
        validate_speed(distance, duration),
    ]:
        if error:
            return error

    return None