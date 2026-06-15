"""
Flight information module for Pakistan Travel Intelligence.
Loads PIA flight dataset and provides real route search + airport information.
All data is sourced from the PIA_2026_Advanced_Kaggle_Dataset.csv file.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── Airport Reference Data ─────────────────────────────────────────────────────
# Source: Pakistan Airports Authority + user specification

AIRPORTS: dict[str, dict[str, str]] = {
    "Karachi": {
        "code": "KHI",
        "name": "Jinnah International Airport",
        "city": "Karachi",
        "province": "Sindh",
        "url": "https://paa.gov.pk",
        "schedule_url": "https://paa.gov.pk/flightSchedule",
    },
    "Lahore": {
        "code": "LHE",
        "name": "Allama Iqbal International Airport",
        "city": "Lahore",
        "province": "Punjab",
        "url": "https://lahoreairport.com.pk",
        "schedule_url": "https://lahoreairport.com.pk/flight-info/flight-schedule",
    },
    "Islamabad": {
        "code": "ISB",
        "name": "Islamabad International Airport",
        "city": "Islamabad",
        "province": "ICT",
        "url": "https://paa.gov.pk",
        "schedule_url": "https://paa.gov.pk/flightSchedule",
    },
    "Peshawar": {
        "code": "PEW",
        "name": "Bacha Khan International Airport",
        "city": "Peshawar",
        "province": "Khyber Pakhtunkhwa",
        "url": "https://paa.gov.pk",
        "schedule_url": "https://paa.gov.pk/flightSchedule",
    },
    "Sialkot": {
        "code": "SKT",
        "name": "Sialkot International Airport",
        "city": "Sialkot",
        "province": "Punjab",
        "url": "https://paa.gov.pk",
        "schedule_url": "https://paa.gov.pk/flightSchedule",
    },
    "Faisalabad": {
        "code": "FSD",
        "name": "Faisalabad International Airport",
        "city": "Faisalabad",
        "province": "Punjab",
        "url": "https://paa.gov.pk",
        "schedule_url": "https://paa.gov.pk/flightSchedule",
    },
    "Multan": {
        "code": "MUX",
        "name": "Multan International Airport",
        "city": "Multan",
        "province": "Punjab",
        "url": "https://paa.gov.pk",
        "schedule_url": "https://paa.gov.pk/flightSchedule",
    },
    "Quetta": {
        "code": "UET",
        "name": "Quetta International Airport",
        "city": "Quetta",
        "province": "Balochistan",
        "url": "https://paa.gov.pk",
        "schedule_url": "https://paa.gov.pk/flightSchedule",
    },
    "Bahawalpur": {
        "code": "BHV",
        "name": "Bahawalpur Airport",
        "city": "Bahawalpur",
        "province": "Punjab",
        "url": "https://paa.gov.pk",
        "schedule_url": "https://paa.gov.pk/flightSchedule",
    },
    "Gwadar": {
        "code": "GWD",
        "name": "Gwadar International Airport",
        "city": "Gwadar",
        "province": "Balochistan",
        "url": "https://paa.gov.pk",
        "schedule_url": "https://paa.gov.pk/flightSchedule",
    },
    "Dera Ghazi Khan": {
        "code": "DEA",
        "name": "Dera Ghazi Khan Airport",
        "city": "Dera Ghazi Khan",
        "province": "Punjab",
        "url": "https://paa.gov.pk",
        "schedule_url": "https://paa.gov.pk/flightSchedule",
    },
    "Rahim Yar Khan": {
        "code": "RYK",
        "name": "Sheikh Zayed International Airport",
        "city": "Rahim Yar Khan",
        "province": "Punjab",
        "url": "https://paa.gov.pk",
        "schedule_url": "https://paa.gov.pk/flightSchedule",
    },
    "Chitral": {
        "code": "CJL",
        "name": "Chitral Airport",
        "city": "Chitral",
        "province": "Khyber Pakhtunkhwa",
        "url": "https://paa.gov.pk",
        "schedule_url": "https://paa.gov.pk/flightSchedule",
    },
    "Gilgit": {
        "code": "GIL",
        "name": "Gilgit Airport",
        "city": "Gilgit",
        "province": "Gilgit-Baltistan",
        "url": "https://paa.gov.pk",
        "schedule_url": "https://paa.gov.pk/flightSchedule",
    },
    "Skardu": {
        "code": "KDU",
        "name": "Skardu Airport",
        "city": "Skardu",
        "province": "Gilgit-Baltistan",
        "url": "https://paa.gov.pk",
        "schedule_url": "https://paa.gov.pk/flightSchedule",
    },
}

# ── Booking Platforms ──────────────────────────────────────────────────────────

BOOKING_PLATFORMS = [
    {
        "name": "SastaTicket.pk",
        "url": "https://sastaticket.pk/flights-booking",
        "icon": "🎫",
        "desc": "Cheap flights, instant booking",
    },
    {
        "name": "Wego Pakistan",
        "url": "https://wego.pk/schedules/pk/pakistan-flight-schedules",
        "icon": "🌐",
        "desc": "Compare flight schedules",
    },
    {
        "name": "Skyscanner Pakistan",
        "url": "https://skyscanner.pk/flights/domestic-country-flights/pk",
        "icon": "🔍",
        "desc": "Search and compare prices",
    },
]

# ── Airline Reference ──────────────────────────────────────────────────────────

AIRLINES = [
    {
        "name": "Pakistan International Airlines (PIA)",
        "url": "https://www.piac.com.pk",
        "icon": "🟢",
        "routes": "Domestic + International",
    },
    {
        "name": "AirSial",
        "url": "https://www.airsial.com",
        "icon": "🔵",
        "routes": "Domestic",
    },
    {
        "name": "Fly Jinnah",
        "url": "https://www.flyjinnah.com",
        "icon": "🟡",
        "routes": "Domestic",
    },
]

# ── Data Loading ───────────────────────────────────────────────────────────────

_DATA_PATH = Path(__file__).parent.parent / "data" / "PIA_2026_Advanced_Kaggle_Dataset.csv"
_df_cache: pd.DataFrame | None = None


def _load_pia_df() -> pd.DataFrame:
    """Load and cache the PIA flight dataset."""
    global _df_cache
    if _df_cache is not None:
        return _df_cache
    try:
        df = pd.read_csv(_DATA_PATH, encoding="utf-8", on_bad_lines="skip")
        _df_cache = df
        logger.info("Loaded PIA flight dataset: %d records", len(df))
    except Exception as exc:
        logger.error("Failed to load PIA flight data: %s", exc)
        _df_cache = pd.DataFrame()
    return _df_cache


# ── Public API ─────────────────────────────────────────────────────────────────

def get_airport_info(city: str) -> dict[str, str] | None:
    """Return airport information for a given city name."""
    return AIRPORTS.get(city)


def get_all_airports() -> list[str]:
    """Return sorted list of all airport cities."""
    return sorted(AIRPORTS.keys())


def get_domestic_cities() -> list[str]:
    """Return cities that have domestic flight data in the PIA dataset."""
    df = _load_pia_df()
    if df.empty:
        return []
    domestic = df[df["Route_Type"] == "Domestic"]
    cities = set(domestic["Departure_City"].tolist()) | set(domestic["Arrival_City"].tolist())
    return sorted(cities)


def search_flights(
    departure: str,
    arrival: str,
    route_type: str = "Domestic",
) -> dict[str, Any]:
    """
    Search for flights between two cities.

    Returns a dict with:
        found: bool
        routes: list of dicts with price stats, duration, aircraft types
        avg_price_usd: float
        min_price_usd: float
        max_price_usd: float
        avg_duration_min: float
        sample_count: int
    """
    df = _load_pia_df()
    if df.empty:
        return {"found": False, "routes": []}

    mask = (
        (df["Departure_City"].str.strip().str.lower() == departure.strip().lower())
        & (df["Arrival_City"].str.strip().str.lower() == arrival.strip().lower())
    )
    if route_type:
        mask &= df["Route_Type"].str.strip().str.lower() == route_type.lower()

    results = df[mask]
    if results.empty:
        return {"found": False, "routes": [], "departure": departure, "arrival": arrival}

    aircraft_types = results["Aircraft_Type"].dropna().unique().tolist()
    avg_price = float(results["Ticket_Price_USD"].mean())
    min_price = float(results["Ticket_Price_USD"].min())
    max_price = float(results["Ticket_Price_USD"].max())
    avg_duration = float(results["Flight_Duration_Minutes"].mean())
    on_time_pct = results["On_Time_Status"].value_counts(normalize=True).get("On Time", 0) * 100

    return {
        "found": True,
        "departure": departure,
        "arrival": arrival,
        "departure_code": AIRPORTS.get(departure, {}).get("code", "???"),
        "arrival_code": AIRPORTS.get(arrival, {}).get("code", "???"),
        "avg_price_usd": round(avg_price, 0),
        "min_price_usd": round(min_price, 0),
        "max_price_usd": round(max_price, 0),
        "avg_duration_min": round(avg_duration, 0),
        "aircraft_types": aircraft_types,
        "on_time_pct": round(on_time_pct, 1),
        "sample_count": len(results),
    }


def get_all_domestic_routes() -> list[dict[str, Any]]:
    """Return summary stats for all unique domestic routes."""
    df = _load_pia_df()
    if df.empty:
        return []

    domestic = df[df["Route_Type"] == "Domestic"]
    routes = []
    for (dep, arr), grp in domestic.groupby(["Departure_City", "Arrival_City"]):
        routes.append({
            "departure": dep,
            "arrival": arr,
            "departure_code": AIRPORTS.get(dep, {}).get("code", "???"),
            "arrival_code": AIRPORTS.get(arr, {}).get("code", "???"),
            "avg_price_usd": round(float(grp["Ticket_Price_USD"].mean()), 0),
            "min_price_usd": round(float(grp["Ticket_Price_USD"].min()), 0),
            "avg_duration_min": round(float(grp["Flight_Duration_Minutes"].mean()), 0),
            "flights": len(grp),
        })
    return sorted(routes, key=lambda r: (r["departure"], r["arrival"]))
