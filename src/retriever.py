"""
Retrieval module — enhanced with reranking, context compression, and fallback behavior.
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
    sources: list[str]                     # Unique source filenames (hidden from user)
    has_results: bool
    confidence_score: float = 0.0          # Average confidence across chunks
    fallback_used: bool = False            # Whether LLM fallback was triggered


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


def _semantic_rerank(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Enhanced reranking with score thresholding, deduplication, and semantic scoring.
    """
    seen: set[str] = set()
    filtered: list[dict[str, Any]] = []
    
    # Sort by score descending
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
    
    for r in results_sorted:
        doc: TravelDocument = r["document"]
        
        # Apply confidence threshold
        if r["score"] < SIMILARITY_THRESHOLD:
            logger.debug(f"Filtered doc {doc.doc_id} due to low score: {r['score']:.3f}")
            continue
            
        # Remove duplicates by doc_id
        if doc.doc_id in seen:
            logger.debug(f"Filtered duplicate doc {doc.doc_id}")
            continue
            
        seen.add(doc.doc_id)
        filtered.append(r)
    
    logger.info(f"Reranking: {len(results)} -> {len(filtered)} documents after filtering")
    return filtered


def _compress_context(chunks: list[dict[str, Any]]) -> str:
    """
    Context compression: aggregate, deduplicate, and format for LLM.
    Hide dataset filenames from user-facing content.
    """
    if not chunks:
        return ""
    
    # Aggregate by type for better organization
    hotels = []
    guest_houses = []
    
    for chunk in chunks:
        if chunk["dataset_type"] == "hotel":
            hotels.append(chunk)
        elif chunk["dataset_type"] == "guest_house":
            guest_houses.append(chunk)
    
    context_parts = []
    
    if hotels:
        context_parts.append("=== HOTELS ===")
        for chunk in hotels:
            context_parts.append(
                f"[Data-backed source | Confidence: {chunk['score']:.3f}]\n{chunk['content']}"
            )
    
    if guest_houses:
        context_parts.append("=== GUEST HOUSES ===")  
        for chunk in guest_houses:
            context_parts.append(
                f"[Data-backed source | Confidence: {chunk['score']:.3f}]\n{chunk['content']}"
            )
    
    return "\n\n---\n\n".join(context_parts)


class Retriever:
    """Enhanced retriever with reranking, compression, and fallback detection."""

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
        Enhanced retrieval with semantic reranking and context compression.
        """
        if not self._store.is_loaded():
            logger.warning("Vector store not loaded")
            return RetrievedContext(
                chunks=[], context_text="", sources=[], has_results=False,
                confidence_score=0.0, fallback_used=True
            )

        # Retrieve more candidates for better reranking
        fetch_size = min(top_k * 4, 50)
        
        if dataset_filter:
            raw = self._store.search(query, top_k=fetch_size, dataset_filter=dataset_filter)
        elif allowed_types:
            raw = self._store.search_multi_filter(
                query, top_k=fetch_size, allowed_types=allowed_types
            )
        else:
            raw = self._store.search(query, top_k=fetch_size)

        # Apply semantic reranking
        reranked = _semantic_rerank(raw)[:top_k]

        if not reranked:
            logger.info(f"No results after reranking for query: {query}")
            return RetrievedContext(
                chunks=[], context_text="", sources=[], has_results=False,
                confidence_score=0.0, fallback_used=True
            )

        # Format chunks
        chunks = [
            _format_chunk(i + 1, r["document"], r["score"])
            for i, r in enumerate(reranked)
        ]

        # Calculate confidence score
        avg_confidence = sum(c["score"] for c in chunks) / len(chunks)
        
        # Detect if fallback should be used
        fallback_needed = avg_confidence < (SIMILARITY_THRESHOLD * 2)

        # Compress and format context
        context_text = _compress_context(chunks)
        sources = list({c["source"] for c in chunks})

        logger.info(
            f"Retrieved {len(chunks)} documents, avg confidence: {avg_confidence:.3f}, "
            f"fallback needed: {fallback_needed}"
        )

        return RetrievedContext(
            chunks=chunks,
            context_text=context_text,
            sources=sources,
            has_results=True,
            confidence_score=avg_confidence,
            fallback_used=fallback_needed,
        )

    def retrieve_accommodations(
        self, city: str, accommodation_type: str = None, top_k: int = TOP_K
    ) -> RetrievedContext:
        """Retrieve accommodation options for a specific city."""
        if accommodation_type and accommodation_type.lower() in ["hotel", "guest house"]:
            allowed_types = [accommodation_type.lower().replace(" ", "_")]
        else:
            allowed_types = ["hotel", "guest_house"]
            
        query = f"accommodation {city} Pakistan hotel guest house"
        return self.retrieve(query, top_k=top_k, allowed_types=allowed_types)

    def retrieve_for_destination(
        self, destination_name: str, top_k: int = TOP_K
    ) -> RetrievedContext:
        """Retrieve all accommodation knowledge about a specific destination."""
        query = f"accommodation {destination_name} Pakistan hotel guest house"
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
