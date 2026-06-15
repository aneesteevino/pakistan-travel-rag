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

from src.config import DATA_DIR, VECTOR_DB_HASH_PATH

logger = logging.getLogger(__name__)

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
    """SHA-256 hash of all CSV files in data directory -- used for cache invalidation."""
    h = hashlib.sha256()
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        h.update(csv_path.read_bytes())
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


def load_all_documents() -> list[TravelDocument]:
    """
    Discover, validate, clean, and convert every CSV in /data
    into a list of TravelDocument objects.
    """
    all_docs: list[TravelDocument] = []
    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        logger.warning("No CSV files found in %s", DATA_DIR)
        return []

    for csv_path in csv_files:
        df = safe_read_csv(csv_path)
        if df.empty:
            logger.error("Could not read %s or empty file", csv_path.name)
            continue

        # ── Clean ──────────────────────────────────────────────────────────
        original_len = len(df)
        df.drop_duplicates(inplace=True)
        df = df.fillna("")

        # Strip whitespace from all string columns
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()

        logger.info(
            "Loaded %s: %d rows (%d dupes removed)",
            csv_path.name,
            len(df),
            original_len - len(df),
        )

        dataset_type = _detect_dataset_type(df.columns.tolist())
        builder = _BUILDER_MAP.get(dataset_type, _generic_builder)
        docs = builder(df, csv_path.name)
        all_docs.extend(docs)
        logger.info("  -> %d documents from %s [%s]", len(docs), csv_path.name, dataset_type)

    logger.info("Total documents loaded: %d", len(all_docs))
    return all_docs


def get_dataset_stats() -> dict[str, Any]:
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
