"""
Configuration module for the Pakistan Travel Intelligence RAG System.
All settings are loaded from Streamlit Secrets (cloud) or environment variables (local).
"""

import os
from pathlib import Path

# Load .env for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Streamlit Secrets Support (Streamlit Cloud deployment) ──────────────────
# When running on Streamlit Cloud, secrets are in st.secrets.
# We inject them into os.environ so the rest of the code works unchanged.
try:
    import streamlit as st
    if hasattr(st, 'secrets') and len(st.secrets) > 0:
        for _key, _val in st.secrets.items():
            if isinstance(_val, str):
                os.environ.setdefault(_key, _val)
except Exception:
    pass

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
CACHE_DIR = BASE_DIR / "cache"
LOGS_DIR = BASE_DIR / "logs"

for _dir in [VECTOR_DB_DIR, CACHE_DIR, LOGS_DIR]:
    try:
        _dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass  # Read-only filesystem on cloud platforms

# ─── Pakistan Travel Settings ───────────────────────────────────────────────
COUNTRY_FOCUS = "Pakistan"
CURRENCY = "PKR"
CURRENCY_SYMBOL = "₨"

# Interest categories for Pakistan travel
TRAVEL_INTERESTS = [
    "Adventure", "Nature", "Mountains", "Historical Sites", "Food", 
    "Culture", "Photography", "Shopping", "Family Travel", "Luxury Travel",
    "Budget Travel", "Hiking", "Road Trips", "Beaches", "Wildlife",
    "Trekking", "Camping", "Religious Sites", "Desert", "Lakes"
]

TRAVEL_STYLES = [
    "Budget", "Standard", "Luxury", "Family", "Adventure", "Solo", 
    "Cultural", "Nature Lover", "Photographer"
]

# ─── LLM ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")  # "gemini" | "groq"
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ─── Embeddings ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ─── Chunking ────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

# ─── Retrieval ───────────────────────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "6"))
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.03"))

# ─── Vector DB ───────────────────────────────────────────────────────────────
VECTOR_DB_PATH: str = str(VECTOR_DB_DIR / "faiss_index")
VECTOR_DB_HASH_PATH: str = str(CACHE_DIR / "data_hash.json")

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: str = str(LOGS_DIR / "travel_rag.log")
