"""
Flight information module for Pakistan Travel Intelligence.
Integrates AviationStack real-time API + PIA historical dataset.
Falls back to OpenAI GPT when live API is unavailable.
"""

from __future__ import annotations

import logging
import os
import json
import requests
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── API Keys ───────────────────────────────────────────────────────────────────
AVIATIONSTACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY", "a4410461eb7725414fcf7be3b7aa61c1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
AVIATIONSTACK_BASE_URL = "http://api.aviationstack.com/v1"

# ── Airport Reference Data ─────────────────────────────────────────────────────
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


# ── AviationStack Real-Time API ────────────────────────────────────────────────

def fetch_real_time_flights(
    dep_iata: str,
    arr_iata: str,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Fetch live real-time flight data from AviationStack API.

    Args:
        dep_iata: Departure airport IATA code (e.g., 'KHI')
        arr_iata: Arrival airport IATA code (e.g., 'LHE')
        limit: Max number of flights to return

    Returns:
        dict with keys:
            success: bool
            source: 'aviationstack' | 'openai_fallback' | 'error'
            flights: list of flight dicts
            error: str (if error)
    """
    try:
        params = {
            "access_key": AVIATIONSTACK_API_KEY,
            "dep_iata": dep_iata,
            "arr_iata": arr_iata,
            "limit": limit,
        }
        response = requests.get(
            f"{AVIATIONSTACK_BASE_URL}/flights",
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            err_msg = data["error"].get("message", "Unknown API error")
            logger.warning("AviationStack API error: %s", err_msg)
            return _openai_flight_fallback(dep_iata, arr_iata, error=err_msg)

        raw_flights = data.get("data", [])
        if not raw_flights:
            logger.info("AviationStack returned 0 flights for %s→%s", dep_iata, arr_iata)
            return _openai_flight_fallback(dep_iata, arr_iata, error="No live flights found")

        flights = []
        for f in raw_flights:
            dep = f.get("departure") or {}
            arr = f.get("arrival") or {}
            airline = f.get("airline") or {}
            flt = f.get("flight") or {}
            aircraft = f.get("aircraft") or {}

            # Format departure/arrival times
            dep_time = dep.get("estimated") or dep.get("scheduled") or dep.get("actual") or "—"
            arr_time = arr.get("estimated") or arr.get("scheduled") or arr.get("actual") or "—"

            # Parse times cleanly
            def fmt_time(t: str) -> str:
                if not t or t == "—":
                    return "—"
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
                    return dt.strftime("%d %b %Y, %H:%M UTC")
                except Exception:
                    return t[:16] if len(t) >= 16 else t

            status = f.get("flight_status", "scheduled").title()
            status_color = {
                "Active": "#4caf50",
                "Scheduled": "#2196f3",
                "Landed": "#9e9e9e",
                "Cancelled": "#f44336",
                "Diverted": "#ff9800",
                "Delayed": "#ff5722",
            }.get(status, "#9ca3af")

            delay_dep = dep.get("delay")
            delay_arr = arr.get("delay")

            flights.append({
                "flight_number": flt.get("iata") or flt.get("icao") or "N/A",
                "airline_name": airline.get("name", "N/A"),
                "airline_iata": airline.get("iata", ""),
                "status": status,
                "status_color": status_color,
                "departure_airport": dep.get("airport", "N/A"),
                "departure_iata": dep.get("iata", dep_iata),
                "departure_time": fmt_time(dep_time),
                "departure_terminal": dep.get("terminal") or "—",
                "departure_gate": dep.get("gate") or "—",
                "arrival_airport": arr.get("airport", "N/A"),
                "arrival_iata": arr.get("iata", arr_iata),
                "arrival_time": fmt_time(arr_time),
                "arrival_terminal": arr.get("terminal") or "—",
                "arrival_gate": arr.get("gate") or "—",
                "aircraft_type": (aircraft.get("iata") or aircraft.get("icao") or "—"),
                "aircraft_registration": aircraft.get("registration") or "—",
                "delay_departure_min": delay_dep,
                "delay_arrival_min": delay_arr,
            })

        return {
            "success": True,
            "source": "aviationstack",
            "dep_iata": dep_iata,
            "arr_iata": arr_iata,
            "total_found": data.get("pagination", {}).get("total", len(flights)),
            "flights": flights,
        }

    except requests.exceptions.RequestException as e:
        logger.warning("AviationStack request failed: %s", e)
        return _openai_flight_fallback(dep_iata, arr_iata, error=str(e))
    except Exception as e:
        logger.error("Unexpected error fetching flights: %s", e)
        return _openai_flight_fallback(dep_iata, arr_iata, error=str(e))


def _openai_flight_fallback(dep_iata: str, arr_iata: str, error: str = "") -> dict[str, Any]:
    """
    Use OpenAI GPT to generate flight information when AviationStack fails.
    Returns AI-generated flight insights clearly marked as AI-generated.
    """
    if not OPENAI_API_KEY:
        return {
            "success": False,
            "source": "error",
            "error": f"AviationStack unavailable ({error}) and no OpenAI key configured.",
            "flights": [],
        }

    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        prompt = f"""You are a Pakistan aviation expert. The user wants flight information for the route {dep_iata} → {arr_iata}.

The live flight API is currently unavailable (reason: {error}).

Please provide realistic, helpful information about:
1. Airlines that typically operate this route (PIA, AirSial, Fly Jinnah)
2. Typical flight frequency (flights per day)
3. Average flight duration
4. Approximate ticket price range in PKR
5. Best time to book
6. Any important notes about this route

Format your response as a JSON object with these keys:
{{
  "route_summary": "one sentence about this route",
  "airlines": ["list of airlines"],
  "frequency": "X flights per day",
  "duration": "X hours Y minutes",
  "price_range_pkr": "PKR X - PKR Y",
  "best_booking_time": "advice string",
  "notes": "any important notes",
  "disclaimer": "This information is AI-generated based on typical Pakistan aviation patterns. Verify with airlines for current schedules."
}}

Only return valid JSON, no other text."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3,
        )

        raw = response.choices[0].message.content.strip()
        # Try to extract JSON
        try:
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            ai_data = json.loads(raw)
        except Exception:
            ai_data = {
                "route_summary": raw[:200],
                "airlines": ["PIA", "AirSial", "Fly Jinnah"],
                "frequency": "Multiple daily",
                "duration": "Check with airline",
                "price_range_pkr": "Contact airline for current fares",
                "best_booking_time": "Book 2-4 weeks in advance",
                "notes": "Schedules vary by season",
                "disclaimer": "AI-generated estimate. Verify with airlines.",
            }

        return {
            "success": True,
            "source": "openai_fallback",
            "dep_iata": dep_iata,
            "arr_iata": arr_iata,
            "ai_insight": ai_data,
            "flights": [],
        }

    except Exception as e:
        logger.error("OpenAI fallback also failed: %s", e)
        return {
            "success": False,
            "source": "error",
            "error": f"Both AviationStack and OpenAI unavailable. {str(e)}",
            "flights": [],
        }


# ── Public API ─────────────────────────────────────────────────────────────────

def get_airport_info(city: str) -> dict[str, str] | None:
    """Return airport information for a given city name."""
    return AIRPORTS.get(city)


def get_all_airports() -> list[str]:
    """Return sorted list of all airport cities."""
    return sorted(AIRPORTS.keys())


def get_city_iata(city: str) -> str:
    """Return IATA code for a city, or empty string."""
    airport = AIRPORTS.get(city, {})
    return airport.get("code", "")


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
    Search for historical flights between two cities using PIA dataset.
    Returns aggregate statistics for the route.
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
