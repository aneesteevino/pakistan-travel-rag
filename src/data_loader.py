"""
Dynamic CSV data loader.
Discovers all CSV files in /data, validates schemas, and builds
enriched document objects ready for chunking and embedding.

Supports:
  - destinations.csv (legacy)
  - hotels.csv (legacy)
  - activities.csv (legacy)
  - visa_requirements.csv (legacy)
  - weather_data.csv (legacy)
  - combined_data.csv (businesses: restaurants, hotels, services)
  - pakistan_tourism_dataset.csv (destination enrichment)
  - PIA_2026_Advanced_Kaggle_Dataset.csv (flight routes)
  - airbnb-listings-in-pakistan.csv (Airbnb listings)
  - sample-data-Guest_houses.csv (guest houses)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.config import DATA_DIR, VECTOR_DB_HASH_PATH

# New data structure paths
NEW_DATA_DIR = DATA_DIR / "new data"
HOTELS_DATA_DIR = NEW_DATA_DIR / "Hotels data"
GUEST_HOUSES_DATA_DIR = NEW_DATA_DIR / "Guest houses data"

logger = logging.getLogger(__name__)

def normalize_city_name(city: str) -> str:
    """Normalize city name for consistent matching."""
    if not city or not isinstance(city, str):
        return ""
    
    # Clean and standardize
    city = city.strip().lower()
    
    # Common variations
    city_mappings = {
        "islamabad city": "Islamabad",
        "lahore city": "Lahore", 
        "karachi city": "Karachi",
        "rawalpindi city": "Rawalpindi",
        "naran kaghan": "Naran",
        "kaghan valley": "Kaghan",
        "fairy meadows base camp": "Fairy Meadows",
        "karimabad village": "Hunza",
        "altit village": "Hunza", 
        "passu village": "Hunza",
        "attabad lake shore": "Hunza",
        "skardu bazaar": "Skardu",
        "mingora city": "Swat",
        "kalam valley": "Swat",
        "chitral bazaar": "Chitral",
        "bumburet valley": "Chitral",
        "shigar village": "Shigar",
        "khaplu valley": "Khaplu", 
        "nagar valley": "Nagar",
        "gilgit city": "Gilgit",
        "astore valley": "Astore"
    }
    
    if city in city_mappings:
        return city_mappings[city]
    
    # Extract main city name from compound names
    if "," in city:
        city = city.split(",")[0].strip()
    
    # Title case the result
    return city.title()

def safe_int(val: Any, default: int = 0) -> int:
    try:
        if val is None or val == "":
            return default
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return default

def safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None or val == "":
            return default
        return float(str(val).strip())
    except (ValueError, TypeError):
        return default

def safe_read_csv(file_path: Path) -> pd.DataFrame:
    """
    Reads a CSV file robustly. Falls back to a custom Python csv parser
    if pandas.read_csv fails with a ParserError (e.g. unquoted fields containing commas).
    """
    import csv
    file_path = Path(file_path)
    
    # Try normal pandas loading first
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(file_path, encoding=enc, low_memory=False)
            return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
            
    # Fallback parser using standard csv module
    logger.warning("Standard pandas loader failed or had ParserError for %s. Using custom fallback parser.", file_path.name)
    
    for enc in ["utf-8", "latin-1", "cp1252"]:
        rows = []
        header = []
        try:
            with open(file_path, mode="r", encoding=enc, errors="replace") as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    continue
                header = [h.strip() for h in header]
                num_cols = len(header)
                
                for line_idx, raw_row in enumerate(reader, start=2):
                    if not raw_row:
                        continue
                    if len(raw_row) == num_cols:
                        rows.append(raw_row)
                    elif len(raw_row) > num_cols:
                        # Extra columns: merge trailing fields into the last column
                        logger.warning(
                            "Malformed row in %s at line %d (saw %d fields, expected %d). Merging trailing fields.",
                            file_path.name, line_idx, len(raw_row), num_cols
                        )
                        merged = raw_row[:num_cols-1]
                        merged_last = ",".join(raw_row[num_cols-1:])
                        merged.append(merged_last)
                        rows.append(merged)
                    else:
                        logger.warning(
                            "Malformed row in %s at line %d (saw %d fields, expected %d). Skipping.",
                            file_path.name, line_idx, len(raw_row), num_cols
                        )
            
            df = pd.DataFrame(rows, columns=header)
            return df
        except Exception as exc:
            logger.error("Fallback parser error for %s with encoding %s: %s", file_path.name, enc, exc)
            
    return pd.DataFrame()

# Major cities to filter combined_data (performance)
_TOP_CITIES = {
    "Karachi", "Lahore", "Islamabad", "Rawalpindi", "Peshawar", "Quetta",
    "Multan", "Faisalabad", "Hyderabad", "Sialkot", "Gujranwala", "Skardu",
    "Gilgit", "Hunza", "Swat", "Murree", "Abbottabad", "Mansehra",
    "Chitral", "Bahawalpur", "Sukkur", "Larkana", "Mirpur", "Muzaffarabad",
}


@dataclass
class TravelDocument:
    """Represents a single knowledge-base document with metadata."""

    content: str
    source: str          # CSV filename (no path)
    dataset_type: str    # destinations | hotels | activities | …
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self) -> None:
        if not self.doc_id:
            self.doc_id = hashlib.md5(
                f"{self.source}:{self.content}".encode()
            ).hexdigest()[:12]


# ─── Dataset-specific builders ────────────────────────────────────────────────

def _build_destination_docs(df: pd.DataFrame, source: str) -> list[TravelDocument]:
    docs: list[TravelDocument] = []
    for _, row in df.iterrows():
        content = (
            f"Destination: {row['name']}, {row['country']}. "
            f"{row['description']} "
            f"Budget level: {row['budget_level']}. "
            f"Categories: {row['categories']}. "
            f"Best months to visit: {row['best_months']}. "
            f"Safety rating: {row['safety_rating']}/5."
        )
        docs.append(TravelDocument(
            content=content,
            source=source,
            dataset_type="destinations",
            metadata={
                "destination_id": safe_int(row.get("id", 0)),
                "name": row.get("name", ""),
                "country": row.get("country", ""),
                "budget_level": row.get("budget_level", ""),
                "categories": row.get("categories", ""),
                "best_months": row.get("best_months", ""),
                "safety_rating": safe_float(row.get("safety_rating", 0.0)),
            },
        ))
    return docs


def _build_hotel_docs(df: pd.DataFrame, source: str) -> list[TravelDocument]:
    docs: list[TravelDocument] = []
    for _, row in df.iterrows():
        content = (
            f"Hotel: {row['name']}. "
            f"Destination ID: {row['destination_id']}. "
            f"Rating: {row['rating']}/5. "
            f"Price range: PKR {row['price_range_pkr']}. "
            f"Amenities: {row['amenities']}. "
            f"{row['description']}"
        )
        docs.append(TravelDocument(
            content=content,
            source=source,
            dataset_type="hotels",
            metadata={
                "hotel_id": safe_int(row.get("id", 0)),
                "destination_id": safe_int(row.get("destination_id", 0)),
                "name": row.get("name", ""),
                "rating": safe_float(row.get("rating", 0.0)),
                "price_range": row.get("price_range_pkr", ""),
                "amenities": row.get("amenities", ""),
            },
        ))
    return docs


def _build_activity_docs(df: pd.DataFrame, source: str) -> list[TravelDocument]:
    docs: list[TravelDocument] = []
    for _, row in df.iterrows():
        content = (
            f"Activity: {row['name']}. "
            f"Category: {row['category']}. "
            f"Duration: {row['duration']}. "
            f"Price: PKR {row['price_pkr']}. "
            f"{row['description']} "
            f"Best time: {row['best_time']}. "
            f"Destination ID: {row['destination_id']}."
        )
        docs.append(TravelDocument(
            content=content,
            source=source,
            dataset_type="activities",
            metadata={
                "activity_id": safe_int(row.get("id", 0)),
                "destination_id": safe_int(row.get("destination_id", 0)),
                "name": row.get("name", ""),
                "category": row.get("category", ""),
                "duration": row.get("duration", ""),
                "price": safe_int(row.get("price_pkr", 0)),
                "best_time": row.get("best_time", ""),
            },
        ))
    return docs


def _build_visa_docs(df: pd.DataFrame, source: str) -> list[TravelDocument]:
    docs: list[TravelDocument] = []
    for _, row in df.iterrows():
        visa_status = "not required" if str(row["visa_required"]).lower() == "no" else "required"

        processing_days_str = str(row["processing_days"])
        if "-" in processing_days_str:
            processing_days_display = processing_days_str
            try:
                processing_days_int = int(processing_days_str.split("-")[-1])
            except ValueError:
                processing_days_int = 0
        else:
            try:
                processing_days_int = int(processing_days_str)
                processing_days_display = processing_days_str
            except ValueError:
                processing_days_int = 0
                processing_days_display = "0"

        content = (
            f"Visa information: Traveling from {row['from_country']} to {row['to_country']}. "
            f"Visa is {visa_status}. "
            f"Visa type: {row['type']}. "
            f"Processing time: {processing_days_display} days. "
            f"Documents needed: {row['documents_needed']}."
        )
        docs.append(TravelDocument(
            content=content,
            source=source,
            dataset_type="visa_requirements",
            metadata={
                "from_country": row["from_country"],
                "to_country": row["to_country"],
                "visa_required": row["visa_required"],
                "visa_type": row["type"],
                "processing_days": safe_int(processing_days_int),
            },
        ))
    return docs


def _build_weather_docs(df: pd.DataFrame, source: str) -> list[TravelDocument]:
    docs: list[TravelDocument] = []
    for _, row in df.iterrows():
        content = (
            f"Weather data for destination ID {row['destination_id']} in {row['month']}: "
            f"Average temperature {row['avg_temp_c']}C, "
            f"Rainfall {row['rainfall_mm']}mm, "
            f"Humidity {row['humidity']}%. "
            f"Conditions: {row['conditions']}. "
            f"Clothing recommendation: {row['clothing_recommendation']}."
        )
        docs.append(TravelDocument(
            content=content,
            source=source,
            dataset_type="weather_data",
            metadata={
                "destination_id": safe_int(row.get("destination_id", 0)),
                "month": row.get("month", ""),
                "avg_temp": safe_float(row.get("avg_temp_c", 0.0)),
                "rainfall": safe_float(row.get("rainfall_mm", 0.0)),
                "humidity": safe_int(row.get("humidity", 0)),
                "conditions": row.get("conditions", ""),
            },
        ))
    return docs


# ─── New real-dataset builders ────────────────────────────────────────────────

def _build_combined_docs(df: pd.DataFrame, source: str) -> list[TravelDocument]:
    """
    Builder for combined_data.csv (114K+ businesses).
    Filters to major Pakistan cities and builds business-type documents.
    """
    docs: list[TravelDocument] = []

    # Filter to top cities only for performance
    if "city" in df.columns:
        df = df[df["city"].isin(_TOP_CITIES)].copy()

    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        address = str(row.get("address", "")).strip()
        city = str(row.get("city", "")).strip()
        state = str(row.get("state", "")).strip()
        phone = str(row.get("phone_number", "")).strip()
        website = str(row.get("website", "")).strip()

        if not name or name.lower() == "nan":
            continue

        parts = [f"Business: {name}."]
        if city:
            parts.append(f"City: {city}, {state}.")
        if address and address.lower() != "nan":
            parts.append(f"Address: {address}.")
        if phone and phone.lower() != "nan":
            parts.append(f"Phone: {phone}.")
        if website and website.lower() != "nan":
            parts.append(f"Website: {website}.")

        content = " ".join(parts)
        docs.append(TravelDocument(
            content=content,
            source=source,
            dataset_type="business",
            metadata={
                "name": name,
                "city": city,
                "state": state,
                "address": address,
                "phone": phone,
                "website": website,
                "lat": str(row.get("latitude", "")),
                "lng": str(row.get("longitude", "")),
            },
        ))
    return docs


def _build_tourism_docs(df: pd.DataFrame, source: str) -> list[TravelDocument]:
    """Builder for pakistan_tourism_dataset.csv."""
    docs: list[TravelDocument] = []
    for _, row in df.iterrows():
        city = str(row.get("City", "")).strip()
        province = str(row.get("Province", "")).strip()
        dest_type = str(row.get("Destination_Type", "")).strip()
        main_attraction = str(row.get("Main_Attraction", "")).strip()
        avg_cost = row.get("Average_Cost_USD", "")
        safety = row.get("Safety_Rating", "")
        accom = str(row.get("Accommodation_Type", "")).strip()
        transport = str(row.get("Transport_Accessibility", "")).strip()
        peak = str(row.get("Peak_Season", "")).strip()
        avg_temp = row.get("Average_Temperature_C", "")
        dom_tourists = row.get("Domestic_Tourists", "")
        intl_tourists = row.get("International_Tourists", "")
        popularity = row.get("Popularity_Score", "")

        content = (
            f"Pakistan Tourism: {city}, {province}. "
            f"Destination type: {dest_type}. "
            f"Main attraction: {main_attraction}. "
            f"Average cost: ${avg_cost} USD per person. "
            f"Safety rating: {safety}/5. "
            f"Best season: {peak}. "
            f"Average temperature: {avg_temp}C. "
            f"Accommodation available: {accom}. "
            f"Transport accessibility: {transport}. "
            f"Domestic tourists annually: {dom_tourists}. "
            f"International tourists annually: {intl_tourists}. "
            f"Popularity score: {popularity}/100."
        )
        docs.append(TravelDocument(
            content=content,
            source=source,
            dataset_type="tourism",
            metadata={
                "city": city,
                "province": province,
                "destination_type": dest_type,
                "main_attraction": main_attraction,
                "avg_cost_usd": avg_cost,
                "safety_rating": safety,
                "peak_season": peak,
                "accommodation_type": accom,
            },
        ))
    return docs


def _build_airbnb_docs(df: pd.DataFrame, source: str) -> list[TravelDocument]:
    """Builder for airbnb-listings-in-pakistan.csv."""
    docs: list[TravelDocument] = []
    for _, row in df.iterrows():
        listing = str(row.get("Listing name", "")).strip()
        host = str(row.get("Host name", "")).strip()
        city = str(row.get("City", "")).strip()
        room_type = str(row.get("Room type", "")).strip()
        price = str(row.get("Price Per Night", "")).strip()
        rating = str(row.get("Rating", "")).strip()
        guests = str(row.get("Guests", "")).strip()
        superhost = str(row.get("Super host", "")).strip()

        if not listing or listing.lower() == "nan":
            continue

        content = (
            f"Airbnb listing: {listing}. "
            f"City: {city}. "
            f"Room type: {room_type}. "
            f"Price per night: {price}. "
            f"Rating: {rating}/5. "
            f"Max guests: {guests}. "
            f"Superhost: {superhost}. "
            f"Host: {host}."
        )
        # Extract just city name
        city_short = city.split(",")[0].strip() if city else city
        docs.append(TravelDocument(
            content=content,
            source=source,
            dataset_type="airbnb_listing",
            metadata={
                "listing_name": listing,
                "host": host,
                "city": city_short,
                "room_type": room_type,
                "price": price,
                "rating": rating,
                "guests": guests,
                "superhost": superhost,
            },
        ))
    return docs


def _build_guesthouse_docs(df: pd.DataFrame, source: str) -> list[TravelDocument]:
    """Builder for sample-data-Guest_houses.csv."""
    docs: list[TravelDocument] = []
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        address = str(row.get("address", "")).strip()
        phone = str(row.get("phone", "")).strip()
        email = str(row.get("email", "")).strip()
        city = str(row.get("city", "")).strip()
        state = str(row.get("state", "")).strip()
        star = str(row.get("star_count", "")).strip()
        reviews = str(row.get("rating_count", "")).strip()
        website = str(row.get("url", "")).strip()

        if not name or name.lower() == "nan":
            continue

        content = (
            f"Guest house: {name}. "
            f"Location: {city}, {state}. "
            f"Address: {address}. "
            f"Phone: {phone}. "
            f"Email: {email}. "
            f"Star rating: {star}. "
            f"Reviews: {reviews}. "
            f"Website: {website}."
        )
        docs.append(TravelDocument(
            content=content,
            source=source,
            dataset_type="guest_house",
            metadata={
                "name": name,
                "city": city,
                "state": state,
                "address": address,
                "phone": phone,
                "email": email,
                "star_count": star,
                "rating_count": reviews,
                "website": website,
            },
        ))
    return docs


def _build_flight_docs(df: pd.DataFrame, source: str) -> list[TravelDocument]:
    """
    Builder for PIA_2026_Advanced_Kaggle_Dataset.csv.
    Groups by route to create representative documents.
    """
    docs: list[TravelDocument] = []

    try:
        for (dep, arr, route_type), grp in df.groupby(
            ["Departure_City", "Arrival_City", "Route_Type"]
        ):
            avg_price = round(float(grp["Ticket_Price_USD"].mean()), 0)
            min_price = round(float(grp["Ticket_Price_USD"].min()), 0)
            max_price = round(float(grp["Ticket_Price_USD"].max()), 0)
            avg_dur = round(float(grp["Flight_Duration_Minutes"].mean()), 0)
            aircraft = ", ".join(grp["Aircraft_Type"].dropna().unique().tolist())
            on_time_pct = round(
                grp["On_Time_Status"].value_counts(normalize=True).get("On Time", 0) * 100, 1
            )
            flights_count = len(grp)

            content = (
                f"Flight route: {dep} to {arr} ({route_type}). "
                f"Average ticket price: ${avg_price} USD "
                f"(range: ${min_price}-${max_price} USD). "
                f"Average flight duration: {int(avg_dur)} minutes. "
                f"Aircraft types: {aircraft}. "
                f"On-time performance: {on_time_pct}%. "
                f"Sample flights in dataset: {flights_count}."
            )
            docs.append(TravelDocument(
                content=content,
                source=source,
                dataset_type="flight_route",
                metadata={
                    "departure": dep,
                    "arrival": arr,
                    "route_type": route_type,
                    "avg_price_usd": avg_price,
                    "min_price_usd": min_price,
                    "max_price_usd": max_price,
                    "avg_duration_min": avg_dur,
                    "aircraft": aircraft,
                    "on_time_pct": on_time_pct,
                },
            ))
    except Exception as exc:
        logger.error("Error building flight docs: %s", exc)
    return docs


# ─── Schema detection ─────────────────────────────────────────────────────────

_SCHEMA_MAP: dict[frozenset[str], str] = {
    frozenset({"id", "name", "country", "description", "budget_level"}): "destinations",
    frozenset({"id", "destination_id", "name", "rating", "price_range_pkr", "amenities"}): "hotels",
    frozenset({"id", "destination_id", "name", "category", "duration", "price_pkr"}): "activities",
    frozenset({"from_country", "to_country", "visa_required", "type", "processing_days"}): "visa_requirements",
    frozenset({"destination_id", "month", "avg_temp_c", "rainfall_mm", "conditions"}): "weather_data",
    # New datasets
    frozenset({"name", "address", "website", "phone_number", "latitude", "longitude", "city", "state", "url"}): "combined",
    frozenset({"Year", "Province", "City", "Destination_Type", "Main_Attraction", "Safety_Rating"}): "tourism",
    frozenset({"Flight_ID", "Departure_City", "Arrival_City", "Route_Type", "Ticket_Price_USD"}): "flights",
    frozenset({"Listing name", "Host name", "City", "Room type", "Price Per Night", "Rating"}): "airbnb",
    frozenset({"name", "address", "phone", "email", "lat", "lng", "url", "country", "state", "city", "star_count"}): "guest_houses",
    frozenset({"operator", "departure_city", "arrival_city", "vehicle_type", "fare_pkr", "duration_hours", "contact_number"}): "road_transport",
}

def _build_road_transport_docs(df: pd.DataFrame, source: str) -> list[TravelDocument]:
    """Builder for road_transport.csv."""
    docs: list[TravelDocument] = []
    for _, row in df.iterrows():
        operator = str(row.get("operator", "")).strip()
        dep = str(row.get("departure_city", "")).strip()
        arr = str(row.get("arrival_city", "")).strip()
        vehicle = str(row.get("vehicle_type", "")).strip()
        fare = str(row.get("fare_pkr", "")).strip()
        duration = str(row.get("duration_hours", "")).strip()
        contact = str(row.get("contact_number", "")).strip()

        if not operator or operator.lower() == "nan":
            continue

        content = (
            f"Road transport connection: {operator} from {dep} to {arr}. "
            f"Vehicle: {vehicle}. "
            f"Fare: PKR {fare}. "
            f"Duration: {duration} hours. "
            f"Contact / Helpline: {contact}."
        )
        docs.append(TravelDocument(
            content=content,
            source=source,
            dataset_type="road_transport",
            metadata={
                "operator": operator,
                "departure_city": dep,
                "arrival_city": arr,
                "vehicle_type": vehicle,
                "fare_pkr": safe_int(fare),
                "duration_hours": safe_float(duration),
                "contact_number": contact,
            },
        ))
    return docs

_BUILDER_MAP = {
    "destinations": _build_destination_docs,
    "hotels": _build_hotel_docs,
    "activities": _build_activity_docs,
    "visa_requirements": _build_visa_docs,
    "weather_data": _build_weather_docs,
    # New datasets
    "combined": _build_combined_docs,
    "tourism": _build_tourism_docs,
    "flights": _build_flight_docs,
    "airbnb": _build_airbnb_docs,
    "guest_houses": _build_guesthouse_docs,
    "road_transport": _build_road_transport_docs,
}


def _detect_dataset_type(columns: list[str]) -> str:
    col_set = frozenset(columns)
    for schema_cols, dtype in _SCHEMA_MAP.items():
        if schema_cols.issubset(col_set):
            return dtype
    return "generic"


def _generic_builder(df: pd.DataFrame, source: str) -> list[TravelDocument]:
    docs: list[TravelDocument] = []
    for _, row in df.iterrows():
        content = " | ".join(f"{k}: {v}" for k, v in row.items() if pd.notna(v))
        docs.append(TravelDocument(
            content=content,
            source=source,
            dataset_type="generic",
            metadata=row.to_dict(),
        ))
    return docs


# ─── Public API ───────────────────────────────────────────────────────────────

def compute_data_hash() -> str:
    """SHA-256 hash of new data structure files -- used for cache invalidation."""
    h = hashlib.sha256()
    
    # Hash hotels data
    hotels_file = HOTELS_DATA_DIR / "hotels.csv"
    if hotels_file.exists():
        h.update(hotels_file.read_bytes())
    
    # Hash guest houses data  
    gh_file = GUEST_HOUSES_DATA_DIR / "guest_houses.csv"
    if gh_file.exists():
        h.update(gh_file.read_bytes())
        
    return h.hexdigest()


def is_index_stale() -> bool:
    """Return True if data files have changed since the last index build."""
    hash_file = Path(VECTOR_DB_HASH_PATH)
    if not hash_file.exists():
        return True
    try:
        saved = json.loads(hash_file.read_text())
        return saved.get("hash") != compute_data_hash()
    except (json.JSONDecodeError, KeyError):
        return True


def save_data_hash() -> None:
    Path(VECTOR_DB_HASH_PATH).write_text(
        json.dumps({"hash": compute_data_hash()})
    )


def normalize_city_name(city: str) -> str:
    """Normalize city names for consistent matching."""
    if not city:
        return ""
    
    # Clean and standardize
    city = city.strip().lower()
    
    # Common variations
    city_mappings = {
        "islamabad city": "islamabad",
        "lahore city": "lahore", 
        "karachi city": "karachi",
        "rawalpindi city": "rawalpindi",
        "naran kaghan": "naran",
        "kaghan valley": "kaghan",
        "fairy meadows base camp": "fairy meadows",
        "karimabad village": "hunza",
        "altit village": "hunza",
        "passu village": "hunza",
        "attabad lake shore": "hunza",
        "skardu bazaar": "skardu",
        "mingora city": "swat",
        "kalam valley": "swat",
        "chitral bazaar": "chitral",
        "bumburet valley": "chitral",
        "shigar village": "shigar",
        "khaplu valley": "khaplu",
        "nagar valley": "nagar",
        "gilgit city": "gilgit",
        "astore valley": "astore",
        "minimarg village": "minimarg"
    }
    
    return city_mappings.get(city, city.title())


def load_all_documents() -> list[TravelDocument]:
    """
    REFACTORED: Load documents from new data structure only.
    Hotels and Guest Houses from travel_rag/data/new data structure.
    """
    all_docs: list[TravelDocument] = []
    
    try:
        # Load hotels from new data structure
        hotels_file = HOTELS_DATA_DIR / "hotels.csv"
        if hotels_file.exists():
            hotels_df = safe_read_csv(hotels_file)
            if not hotels_df.empty:
                # Clean and normalize
                original_len = len(hotels_df)
                hotels_df.drop_duplicates(inplace=True)
                hotels_df = hotels_df.fillna("")
                
                # Normalize city names
                if 'city' in hotels_df.columns:
                    hotels_df['city'] = hotels_df['city'].apply(normalize_city_name)
                
                # Strip whitespace
                for col in hotels_df.select_dtypes(include="object").columns:
                    hotels_df[col] = hotels_df[col].astype(str).str.strip()
                
                # Build hotel documents
                hotel_docs = _build_hotel_docs_new(hotels_df, "Data-backed estimate")
                all_docs.extend(hotel_docs)
                logger.info("Loaded hotels.csv: %d rows (%d dupes removed) -> %d documents", 
                           len(hotels_df), original_len - len(hotels_df), len(hotel_docs))
        else:
            logger.warning("Hotels data file not found: %s", hotels_file)
        
        # Load guest houses from new data structure  
        gh_file = GUEST_HOUSES_DATA_DIR / "guest_houses.csv"
        if gh_file.exists():
            gh_df = safe_read_csv(gh_file)
            if not gh_df.empty:
                # Clean and normalize
                original_len = len(gh_df)
                gh_df.drop_duplicates(inplace=True)
                gh_df = gh_df.fillna("")
                
                # Normalize city names
                if 'city' in gh_df.columns:
                    gh_df['city'] = gh_df['city'].apply(normalize_city_name)
                
                # Strip whitespace
                for col in gh_df.select_dtypes(include="object").columns:
                    gh_df[col] = gh_df[col].astype(str).str.strip()
                
                # Build guest house documents
                gh_docs = _build_guest_house_docs_new(gh_df, "Based on available travel records")
                all_docs.extend(gh_docs)
                logger.info("Loaded guest_houses.csv: %d rows (%d dupes removed) -> %d documents", 
                           len(gh_df), original_len - len(gh_df), len(gh_docs))
        else:
            logger.warning("Guest houses data file not found: %s", gh_file)
            
    except Exception as e:
        logger.error("Error loading new data structure: %s", e)
        return []

    if not all_docs:
        logger.error("No documents loaded from new data structure!")
        return []
        
    logger.info("Total documents loaded from new data structure: %d", len(all_docs))
    return all_docs


def _build_hotel_docs_new(df: pd.DataFrame, source: str) -> list[TravelDocument]:
    """Build hotel documents from new data structure."""
    docs: list[TravelDocument] = []
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        city = str(row.get("city", "")).strip()
        province = str(row.get("province", "")).strip()
        address = str(row.get("address", "")).strip()
        phone = str(row.get("phone", "")).strip()
        email = str(row.get("email", "")).strip()
        star_rating = str(row.get("star_rating", "")).strip()
        price_range = str(row.get("price_range_pkr", "")).strip()
        amenities = str(row.get("amenities", "")).strip()
        description = str(row.get("description", "")).strip()

        if not name or name.lower() == "nan":
            continue

        content = (
            f"Hotel: {name}. "
            f"Location: {city}, {province}. "
            f"Address: {address}. "
            f"Phone: {phone}. "
            f"Email: {email}. "
            f"Star rating: {star_rating}. "
            f"Price range: PKR {price_range} per night. "
            f"Amenities: {amenities}. "
            f"{description}"
        )
        
        docs.append(TravelDocument(
            content=content,
            source=source,
            dataset_type="hotel",
            metadata={
                "name": name,
                "city": city,
                "province": province,
                "address": address,
                "phone": phone,
                "email": email,
                "star_rating": star_rating,
                "price_range": price_range,
                "amenities": amenities,
                "description": description,
            },
        ))
    return docs


def _build_guest_house_docs_new(df: pd.DataFrame, source: str) -> list[TravelDocument]:
    """Build guest house documents from new data structure."""
    docs: list[TravelDocument] = []
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        city = str(row.get("city", "")).strip()
        province = str(row.get("province", "")).strip()
        address = str(row.get("address", "")).strip()
        phone = str(row.get("phone", "")).strip()
        email = str(row.get("email", "")).strip()
        star_rating = str(row.get("star_rating", "")).strip()
        price_range = str(row.get("price_range_pkr", "")).strip()
        amenities = str(row.get("amenities", "")).strip()
        description = str(row.get("description", "")).strip()

        if not name or name.lower() == "nan":
            continue

        content = (
            f"Guest House: {name}. "
            f"Location: {city}, {province}. "
            f"Address: {address}. "
            f"Phone: {phone}. "
            f"Email: {email}. "
            f"Star rating: {star_rating}. "
            f"Price range: PKR {price_range} per night. "
            f"Amenities: {amenities}. "
            f"{description}"
        )
        
        docs.append(TravelDocument(
            content=content,
            source=source,
            dataset_type="guest_house",
            metadata={
                "name": name,
                "city": city,
                "province": province,
                "address": address,
                "phone": phone,
                "email": email,
                "star_rating": star_rating,
                "price_range": price_range,
                "amenities": amenities,
                "description": description,
            },
        ))
    return docs
    """Return summary statistics for the knowledge base UI panel."""
    stats: dict[str, Any] = {}
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        try:
            df = safe_read_csv(csv_path)
            dtype = _detect_dataset_type(df.columns.tolist())
            stats[csv_path.name] = {
                "rows": len(df),
                "columns": df.columns.tolist(),
                "dataset_type": dtype,
                "missing_values": int(df.isnull().sum().sum()) if not df.empty else 0,
            }
        except Exception as exc:
            stats[csv_path.name] = {"error": str(exc)}
    return stats

def _build_hotel_docs_new(df: pd.DataFrame, source: str) -> list[TravelDocument]:
    """Build hotel documents from new data structure (no filename exposure)."""
    docs: list[TravelDocument] = []
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        city = str(row.get("city", "")).strip()
        province = str(row.get("province", "")).strip()
        address = str(row.get("address", "")).strip()
        phone = str(row.get("phone", "")).strip()
        email = str(row.get("email", "")).strip()
        rating = safe_float(row.get("rating", 0.0))
        price_range = str(row.get("price_range", "")).strip()
        amenities = str(row.get("amenities", "")).strip()
        description = str(row.get("description", "")).strip()

        if not name:
            continue

        content = (
            f"Hotel: {name}. "
            f"Location: {city}, {province}. "
            f"Address: {address}. "
            f"Rating: {rating}/5.0. "
            f"Price range: {price_range}. "
            f"Amenities: {amenities}. "
            f"Contact: Phone {phone}, Email {email}. "
            f"Description: {description}"
        )

        docs.append(TravelDocument(
            content=content,
            source=source,  # Uses "Data-backed estimate" instead of filename
            dataset_type="hotel",
            metadata={
                "name": name,
                "city": normalize_city_name(city),
                "province": province,
                "type": "Hotel",
                "rating": rating,
                "price_range": price_range,
                "phone": phone,
                "email": email,
                "amenities": amenities
            }
        ))
    return docs


def _build_guest_house_docs_new(df: pd.DataFrame, source: str) -> list[TravelDocument]:
    """Build guest house documents from new data structure (no filename exposure)."""
    docs: list[TravelDocument] = []
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        city = str(row.get("city", "")).strip()
        province = str(row.get("province", "")).strip()
        address = str(row.get("address", "")).strip()
        phone = str(row.get("phone", "")).strip()
        email = str(row.get("email", "")).strip()
        rating = safe_float(row.get("rating", 0.0))
        price_range = str(row.get("price_range", "")).strip()
        amenities = str(row.get("amenities", "")).strip()
        description = str(row.get("description", "")).strip()

        if not name:
            continue

        content = (
            f"Guest House: {name}. "
            f"Location: {city}, {province}. "
            f"Address: {address}. "
            f"Rating: {rating}/5.0. "
            f"Price range: {price_range}. "
            f"Amenities: {amenities}. "
            f"Contact: Phone {phone}, Email {email}. "
            f"Description: {description}"
        )

        docs.append(TravelDocument(
            content=content,
            source=source,  # Uses "Based on available travel records" instead of filename
            dataset_type="guest_house",
            metadata={
                "name": name,
                "city": normalize_city_name(city),
                "province": province,
                "type": "Guest House",
                "rating": rating,
                "price_range": price_range,
                "phone": phone,
                "email": email,
                "amenities": amenities
            }
        ))
    return docs


def get_dataset_stats() -> dict:
    """Get statistics about available datasets."""
    stats = {}
    
    try:
        # Hotels statistics
        hotels_file = HOTELS_DATA_DIR / "hotels.csv"
        if hotels_file.exists():
            hotels_df = safe_read_csv(hotels_file)
            stats["hotels"] = {
                "file": "Hotels data",
                "rows": len(hotels_df),
                "columns": list(hotels_df.columns) if not hotels_df.empty else [],
                "status": "active"
            }
    except Exception as e:
        logger.error(f"Error getting hotels stats: {e}")
    
    try:
        # Guest houses statistics
        gh_file = GUEST_HOUSES_DATA_DIR / "guest_houses.csv"
        if gh_file.exists():
            gh_df = safe_read_csv(gh_file)
            stats["guest_houses"] = {
                "file": "Guest Houses data", 
                "rows": len(gh_df),
                "columns": list(gh_df.columns) if not gh_df.empty else [],
                "status": "active"
            }
    except Exception as e:
        logger.error(f"Error getting guest houses stats: {e}")
    
    return stats


@st.cache_data
def get_all_accommodations() -> list[dict]:
    """Get all accommodations from new data structure only (Hotels + Guest Houses)."""
    accommodations = []
    
    try:
        # Load hotels
        hotels_df = pd.read_csv(HOTELS_DATA_DIR / "hotels.csv")
        for _, row in hotels_df.iterrows():
            accommodations.append({
                "name": str(row.get("name", "")).strip(),
                "type": "Hotel",
                "city": str(row.get("city", "")).strip().title(),
                "province": str(row.get("province", "")).strip(),
                "price": str(row.get("price_range", "PKR 5,000-15,000")),
                "rating": f"{row.get('rating', 'N/A')} ⭐",
                "contact": f"📞 {row.get('phone', 'Contact hotel')} | 📧 {row.get('email', 'Email available')}",
                "description": f"Address: {row.get('address', 'Address available')}. Amenities: {row.get('amenities', 'Standard amenities')}",
                "url": ""
            })
    except Exception as e:
        logger.info(f"Hotels data not available: {e}")

    try:
        # Load guest houses
        gh_df = pd.read_csv(GUEST_HOUSES_DATA_DIR / "guest_houses.csv")
        for _, row in gh_df.iterrows():
            accommodations.append({
                "name": str(row.get("name", "")).strip(),
                "type": "Guest House",
                "city": str(row.get("city", "")).strip().title(),
                "province": str(row.get("province", "")).strip(),
                "price": str(row.get("price_range", "PKR 3,000-8,000")),
                "rating": f"{row.get('rating', 'N/A')} ⭐",
                "contact": f"📞 {row.get('phone', 'Contact guest house')} | 📧 {row.get('email', 'Email available')}",
                "description": f"Address: {row.get('address', 'Address available')}. Amenities: {row.get('amenities', 'Basic amenities')}",
                "url": ""
            })
    except Exception as e:
        logger.info(f"Guest houses data not available: {e}")
    
    return accommodations


@st.cache_data
def get_all_road_transport() -> list[dict]:
    """Placeholder for road transport data (removed as per requirements)."""
    return [
        {
            "operator": "Daewoo Express",
            "departure_city": "Karachi",
            "arrival_city": "Lahore", 
            "vehicle_type": "AC Bus",
            "fare_pkr": "3500",
            "duration_hours": "20",
            "contact": "0800-DAEWOO"
        },
        {
            "operator": "NATCO",
            "departure_city": "Lahore", 
            "arrival_city": "Islamabad",
            "vehicle_type": "AC Coach",
            "fare_pkr": "2000",
            "duration_hours": "5",
            "contact": "042-111-NATCO"
        }
    ]