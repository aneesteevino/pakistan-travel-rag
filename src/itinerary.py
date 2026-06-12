"""
Pakistan Itinerary module — orchestrates retrieval + generation for day-wise Pakistan travel plans.
"""

from __future__ import annotations

import logging

from src.generator import generate_itinerary, generate_trip_plan
from src.retriever import RetrievedContext, get_retriever

logger = logging.getLogger(__name__)


def build_itinerary(
    destination: str,
    duration_days: int,
    budget_pkr: int,
    travel_style: str,
    interests: str,
) -> tuple[str, RetrievedContext]:
    """
    Retrieve relevant Pakistan knowledge then generate a day-wise itinerary.
    Returns (itinerary_text, retrieved_context).
    """
    retriever = get_retriever()

    # Create a comprehensive query to get relevant information
    query = (
        f"Pakistan {destination} activities hotels weather things to do "
        f"{interests} PKR budget {travel_style} travel attractions accommodation"
    )
    
    # Retrieve broader context to ensure we get data
    context = retriever.retrieve(
        query,
        top_k=15,  # Increased to get more relevant data
    )

    # If no context found, try a simpler query
    if not context.has_results:
        simple_query = f"{destination} Pakistan travel hotels activities"
        context = retriever.retrieve(simple_query, top_k=10)

    itinerary_text = generate_itinerary(
        destination=destination,
        duration_days=duration_days,
        budget_pkr=budget_pkr,
        travel_style=travel_style,
        interests=interests,
        context=context,
    )

    return itinerary_text, context


def build_trip_plan(
    budget_pkr: int,
    duration: int,
    interests: str,
    travel_style: str,
) -> tuple[str, RetrievedContext]:
    """
    Build a personalized Pakistan trip plan using user preferences.
    Returns (plan_text, retrieved_context).
    """
    retriever = get_retriever()

    # Create a comprehensive query
    query = (
        f"Pakistan travel destinations PKR budget {travel_style} style "
        f"{interests} {duration} days hotels activities attractions"
    )
    
    # Get broader context
    context = retriever.retrieve(
        query,
        top_k=20,  # Increased to get comprehensive data
    )

    # If no results, try simpler query
    if not context.has_results:
        simple_query = f"Pakistan destinations hotels activities travel"
        context = retriever.retrieve(simple_query, top_k=15)

    plan_text = generate_trip_plan(
        budget_pkr=budget_pkr,
        duration=duration,
        interests=interests,
        travel_style=travel_style,
        context=context,
    )

    return plan_text, context
