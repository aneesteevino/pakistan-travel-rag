"""
Dynamic CSV data loader.
Discovers all CSV files in /data, validates schemas, and builds
enriched document objects ready for chunking and embedding.
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
                "destination_id": int(row["id"]),
                "name": row["name"],
                "country": row["country"],
                "budget_level": row["budget_level"],
                "categories": row["categories"],
                "best_months": row["best_months"],
                "safety_rating": float(row["safety_rating"]),
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
                "hotel_id": int(row["id"]),
                "destination_id": int(row["destination_id"]),
                "name": row["name"],
                "rating": float(row["rating"]),
                "price_range": row["price_range_pkr"],
                "amenities": row["amenities"],
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
                "activity_id": int(row["id"]),
                "destination_id": int(row["destination_id"]),
                "name": row["name"],
                "category": row["category"],
                "duration": row["duration"],
                "price": int(row["price_pkr"]),
                "best_time": row["best_time"],
            },
        ))
    return docs


def _build_visa_docs(df: pd.DataFrame, source: str) -> list[TravelDocument]:
    docs: list[TravelDocument] = []
    for _, row in df.iterrows():
        visa_status = "not required" if str(row["visa_required"]).lower() == "no" else "required"
        
        # Handle processing days - could be a range like "7-10" or "0"
        processing_days_str = str(row["processing_days"])
        if "-" in processing_days_str:
            # It's a range, use as-is in content but take the max for metadata
            processing_days_display = processing_days_str
            try:
                processing_days_int = int(processing_days_str.split("-")[-1])  # Take the max value
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
                "processing_days": processing_days_int,
            },
        ))
    return docs


def _build_weather_docs(df: pd.DataFrame, source: str) -> list[TravelDocument]:
    docs: list[TravelDocument] = []
    for _, row in df.iterrows():
        content = (
            f"Weather data for destination ID {row['destination_id']} in {row['month']}: "
            f"Average temperature {row['avg_temp_c']}°C, "
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
                "destination_id": int(row["destination_id"]),
                "month": row["month"],
                "avg_temp": float(row["avg_temp_c"]),
                "rainfall": float(row["rainfall_mm"]),
                "humidity": int(row["humidity"]),
                "conditions": row["conditions"],
            },
        ))
    return docs


# ─── Schema detection ─────────────────────────────────────────────────────────

_SCHEMA_MAP: dict[frozenset[str], str] = {
    frozenset({"id", "name", "country", "description", "budget_level"}): "destinations",
    frozenset({"id", "destination_id", "name", "rating", "price_range_pkr", "amenities"}): "hotels",
    frozenset({"id", "destination_id", "name", "category", "duration", "price_pkr"}): "activities",
    frozenset({"from_country", "to_country", "visa_required", "type", "processing_days"}): "visa_requirements",
    frozenset({"destination_id", "month", "avg_temp_c", "rainfall_mm", "conditions"}): "weather_data",
}

_BUILDER_MAP = {
    "destinations": _build_destination_docs,
    "hotels": _build_hotel_docs,
    "activities": _build_activity_docs,
    "visa_requirements": _build_visa_docs,
    "weather_data": _build_weather_docs,
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
    """SHA-256 hash of all CSV files in data directory — used for cache invalidation."""
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
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            logger.error("Failed to read %s: %s", csv_path.name, exc)
            continue

        # ── Clean ──────────────────────────────────────────────────────────
        original_len = len(df)
        df.drop_duplicates(inplace=True)
        df.fillna("", inplace=True)

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
        logger.info("  → %d documents from %s", len(docs), csv_path.name)

    logger.info("Total documents loaded: %d", len(all_docs))
    return all_docs


def get_dataset_stats() -> dict[str, Any]:
    """Return summary statistics for the knowledge base UI panel."""
    stats: dict[str, Any] = {}
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
            dtype = _detect_dataset_type(df.columns.tolist())
            stats[csv_path.name] = {
                "rows": len(df),
                "columns": df.columns.tolist(),
                "dataset_type": dtype,
                "missing_values": int(df.isnull().sum().sum()),
            }
        except Exception as exc:
            stats[csv_path.name] = {"error": str(exc)}
    return stats
