"""
Retrieval module — context-compressed, re-ranked results with source citations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.config import SIMILARITY_THRESHOLD, TOP_K
from src.data_loader import TravelDocument
from src.vector_store import get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    """Structured container returned to the generator."""

    chunks: list[dict[str, Any]]          # [{content, score, source, dataset_type, metadata}]
    context_text: str                      # Concatenated text for LLM prompt
    sources: list[str]                     # Unique source filenames
    has_results: bool


def _format_chunk(rank: int, doc: TravelDocument, score: float) -> dict[str, Any]:
    return {
        "rank": rank,
        "content": doc.content,
        "score": score,
        "source": doc.source,
        "dataset_type": doc.dataset_type,
        "metadata": doc.metadata,
        "doc_id": doc.doc_id,
    }


def _rerank(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Simple score-threshold re-ranking.
    Removes low-confidence chunks and deduplicates by doc_id.
    """
    seen: set[str] = set()
    filtered: list[dict[str, Any]] = []
    for r in results:
        doc: TravelDocument = r["document"]
        if r["score"] < SIMILARITY_THRESHOLD:
            continue
        if doc.doc_id in seen:
            continue
        seen.add(doc.doc_id)
        filtered.append(r)
    return filtered


class Retriever:
    """High-level retriever with optional type filtering and context compression."""

    def __init__(self) -> None:
        self._store = get_vector_store()

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        dataset_filter: str | None = None,
        allowed_types: list[str] | None = None,
    ) -> RetrievedContext:
        """
        Run a similarity search against the vector store, re-rank results,
        and return a RetrievedContext ready for the generator.
        """
        if not self._store.is_loaded():
            return RetrievedContext(
                chunks=[], context_text="", sources=[], has_results=False
            )

        if dataset_filter:
            raw = self._store.search(query, top_k=top_k * 2, dataset_filter=dataset_filter)
        elif allowed_types:
            raw = self._store.search_multi_filter(
                query, top_k=top_k * 2, allowed_types=allowed_types
            )
        else:
            raw = self._store.search(query, top_k=top_k * 2)

        reranked = _rerank(raw)[:top_k]

        chunks = [
            _format_chunk(i + 1, r["document"], r["score"])
            for i, r in enumerate(reranked)
        ]

        if not chunks:
            return RetrievedContext(
                chunks=[], context_text="", sources=[], has_results=False
            )

        context_parts: list[str] = []
        for c in chunks:
            context_parts.append(
                f"[Source: {c['source']} | Type: {c['dataset_type']} | Score: {c['score']:.3f}]\n"
                f"{c['content']}"
            )

        context_text = "\n\n---\n\n".join(context_parts)
        sources = list({c["source"] for c in chunks})

        return RetrievedContext(
            chunks=chunks,
            context_text=context_text,
            sources=sources,
            has_results=True,
        )

    def retrieve_for_destination(
        self, destination_name: str, top_k: int = TOP_K
    ) -> RetrievedContext:
        """Retrieve all knowledge about a specific destination."""
        query = f"travel information about {destination_name} destination hotels activities weather"
        return self.retrieve(query, top_k=top_k)

    def retrieve_for_comparison(
        self, destination_a: str, destination_b: str
    ) -> tuple[RetrievedContext, RetrievedContext]:
        return (
            self.retrieve_for_destination(destination_a),
            self.retrieve_for_destination(destination_b),
        )


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
