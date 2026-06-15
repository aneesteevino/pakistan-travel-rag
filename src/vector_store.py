"""
FAISS-based persistent vector store.
Fits the embedding pipeline on all documents, then stores/searches
the dense vectors with cosine similarity (inner product on L2-normalised vecs).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, List

import faiss
import numpy as np

from src.config import VECTOR_DB_PATH
from src.data_loader import TravelDocument
from src.embeddings import get_embedding_engine

logger = logging.getLogger(__name__)

_INDEX_FILE = Path(VECTOR_DB_PATH + ".faiss")
_META_FILE  = Path(VECTOR_DB_PATH + ".meta.pkl")


class VectorStore:
    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP | None = None
        self._documents: list[TravelDocument] = []
        self._engine = get_embedding_engine()

    # ── Build ──────────────────────────────────────────────────────────────

    def build(self, documents: list[TravelDocument]) -> None:
        if not documents:
            raise ValueError("Cannot build an empty index.")

        texts = [doc.content for doc in documents]

        # Fit the embedding model on the corpus first
        self._engine.fit(texts)

        vectors = self._engine.embed(texts)          # (N, D) float32 normalised

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        self._index     = index
        self._documents = documents

        self._save()
        logger.info("Built FAISS index: %d vectors, dim=%d", len(documents), dim)

    # ── Persistence ────────────────────────────────────────────────────────

    def _save(self) -> None:
        _INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(_INDEX_FILE))
        with open(_META_FILE, "wb") as f:
            pickle.dump(self._documents, f)
        logger.info("Saved FAISS index → %s", _INDEX_FILE)

    def load(self) -> bool:
        if not _INDEX_FILE.exists() or not _META_FILE.exists() or not self._engine.is_fitted:
            return False
        try:
            self._index = faiss.read_index(str(_INDEX_FILE))
            with open(_META_FILE, "rb") as f:
                self._documents = pickle.load(f)
            logger.info("Loaded FAISS index: %d docs", len(self._documents))
            return True
        except Exception as exc:
            logger.error("Failed to load index: %s", exc)
            return False

    def is_loaded(self) -> bool:
        return self._index is not None and bool(self._documents)

    # ── Search ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 6,
        dataset_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.is_loaded():
            raise RuntimeError("Vector store not loaded.")

        q_vec = self._engine.embed_query(query)                # (1, D)
        fetch = min(top_k * 6, len(self._documents))
        scores, indices = self._index.search(q_vec, fetch)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self._documents[idx]
            if dataset_filter and doc.dataset_type != dataset_filter:
                continue
            results.append({"document": doc, "score": float(score)})
            if len(results) >= top_k:
                break
        return results

    def search_multi_filter(
        self,
        query: str,
        top_k: int = 6,
        allowed_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.is_loaded():
            raise RuntimeError("Vector store not loaded.")

        q_vec  = self._engine.embed_query(query)
        fetch  = min(top_k * 6, len(self._documents))
        scores, indices = self._index.search(q_vec, fetch)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self._documents[idx]
            if allowed_types and doc.dataset_type not in allowed_types:
                continue
            results.append({"document": doc, "score": float(score)})
            if len(results) >= top_k:
                break
        return results

    # ── Stats ──────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        if not self.is_loaded():
            return {"status": "not loaded"}
        counts: dict[str, int] = {}
        for doc in self._documents:
            counts[doc.dataset_type] = counts.get(doc.dataset_type, 0) + 1
        return {
            "total_documents": len(self._documents),
            "by_type": counts,
            "index_size": self._index.ntotal,
        }


# ── Module-level singleton ─────────────────────────────────────────────────

_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
