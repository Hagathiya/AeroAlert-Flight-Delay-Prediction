"""
airport_utils.py
Loads airport coordinates and computes flight distance automatically,
so the user never has to type in miles by hand.

Place airport_coords.csv in the same folder as this file (or update CSV_PATH).
"""

import csv
import math
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "airport_coords.csv")

_airport_cache = None


def _load_airports():
    """Load the IATA -> (lat, lon) lookup once and cache it in memory."""
    global _airport_cache
    if _airport_cache is not None:
        return _airport_cache

    lookup = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lookup[row["IATA"].upper()] = (
                float(row["LAT"]),
                float(row["LON"]),
                row["NAME"],
                row["CITY"],
                row["COUNTRY"],
            )
    _airport_cache = lookup
    return lookup


def get_airport_info(iata_code: str):
    """Return (lat, lon, name, city, country) for an IATA code, or None if unknown."""
    lookup = _load_airports()
    return lookup.get(iata_code.strip().upper())


def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points, in miles."""
    R = 3958.8  # Earth's radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def get_distance(origin_iata: str, dest_iata: str):
    """
    Compute flight distance in miles between two IATA airport codes.
    Returns None if either code isn't found (so the caller can show
    a friendly 'unknown airport' error instead of crashing).
    """
    origin = get_airport_info(origin_iata)
    dest = get_airport_info(dest_iata)

    if origin is None or dest is None:
        return None

    lat1, lon1 = origin[0], origin[1]
    lat2, lon2 = dest[0], dest[1]
    return round(haversine_miles(lat1, lon1, lat2, lon2), 1)


if __name__ == "__main__":
    # quick manual test
    print("MAA -> AUX:", get_distance("MAA", "AUX"), "miles")
    print("JFK -> LAX:", get_distance("JFK", "LAX"), "miles")
    print("Unknown code test:", get_distance("XXX", "LAX"))
