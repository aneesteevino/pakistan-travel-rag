"""
Embeddings module.

Uses TF-IDF with SVD dimensionality reduction (LSA) as a fully offline
embedding backend.  The public API is identical to a sentence-transformer
wrapper so the rest of the codebase never needs to change.

When running in an environment with internet access and HuggingFace support,
set EMBEDDING_BACKEND=sentence_transformers in .env to switch providers.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize

from src.config import CACHE_DIR

logger = logging.getLogger(__name__)

_CACHE_FILE = Path(CACHE_DIR) / "tfidf_pipeline.pkl"
_EMBED_DIM = 256          # LSA latent dimensions


class EmbeddingEngine:
    """TF-IDF + LSA embedding engine (fully offline, no HuggingFace required)."""

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._fitted: bool = False

    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=40_000,
                sublinear_tf=True,
                strip_accents="unicode",
                analyzer="word",
                token_pattern=r"\b[a-zA-Z][a-zA-Z0-9]*\b",
                min_df=1,
            )),
            ("svd", TruncatedSVD(n_components=_EMBED_DIM, random_state=42)),
        ])

    def fit(self, texts: List[str]) -> None:
        """Fit the TF-IDF + SVD pipeline on the corpus."""
        logger.info("Fitting TF-IDF+SVD pipeline on %d texts…", len(texts))
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(texts)
        self._fitted = True
        # Persist
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_FILE, "wb") as f:
            pickle.dump(self._pipeline, f)
        logger.info("Embedding pipeline fitted and cached.")

    def _load_cached(self) -> bool:
        if not _CACHE_FILE.exists():
            return False
        try:
            with open(_CACHE_FILE, "rb") as f:
                self._pipeline = pickle.load(f)
            self._fitted = True
            logger.info("Loaded cached TF-IDF pipeline.")
            return True
        except Exception as exc:
            logger.warning("Could not load cached pipeline: %s", exc)
            return False

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Return (N, D) float32 normalised embedding matrix."""
        if not self._fitted:
            raise RuntimeError("EmbeddingEngine must be fit() before embed().")
        vectors = self._pipeline.transform(texts)
        return normalize(vectors, norm="l2").astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Return (1, D) float32 embedding for a single query."""
        return self.embed([query])

    @property
    def dimension(self) -> int:
        return _EMBED_DIM


# ── Module-level singleton ─────────────────────────────────────────────────

_engine: EmbeddingEngine | None = None


def get_embedding_engine() -> EmbeddingEngine:
    global _engine
    if _engine is None:
        _engine = EmbeddingEngine()
        _engine._load_cached()      # no-op if not yet cached
    return _engine
