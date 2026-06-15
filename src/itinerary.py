"""
Pakistan Travel plan module -- orchestrates retrieval + generation for Pakistan travel plans.
"""

from __future__ import annotations

import logging

from src.generator import generate_trip_plan, generate_wizard_trip_plan
from src.retriever import RetrievedContext, get_retriever

logger = logging.getLogger(__name__)


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

    query = (
        f"Pakistan travel destinations PKR budget {travel_style} style "
        f"{interests} {duration} days hotels activities attractions"
    )

    context = retriever.retrieve(query, top_k=20)

    if not context.has_results:
        context = retriever.retrieve("Pakistan destinations hotels activities travel", top_k=15)

    plan_text = generate_trip_plan(
        budget_pkr=budget_pkr,
        duration=duration,
        interests=interests,
        travel_style=travel_style,
        context=context,
    )

    return plan_text, context


def build_wizard_trip_plan(
    destination: str,
    departure_city: str,
    group_type: str,
    num_travelers: int,
    duration_days: int,
    travel_style: str,
    interests: list[str],
    accommodation_type: str,
    room_type: str,
    food_prefs: list[str],
    food_notes: str,
    transport_type: str,
    transport_class: str,
    budget_pkr: int,
    split_budget: bool,
) -> tuple[str, RetrievedContext]:
    """
    Full wizard trip plan: retrieve RAG context then generate comprehensive plan.
    Returns (plan_text, retrieved_context).
    """
    retriever = get_retriever()

    interests_str = ", ".join(interests) if interests else "sightseeing"
    food_str = ", ".join(food_prefs) if food_prefs else "local cuisine"
    query = (
        f"Pakistan {destination} travel {interests_str} {travel_style} style "
        f"{accommodation_type} accommodation {food_str} food restaurants activities attractions "
        f"budget PKR hotels transport from {departure_city}"
    )

    context = retriever.retrieve(query, top_k=25)
    if not context.has_results:
        context = retriever.retrieve(f"{destination} Pakistan hotels activities travel", top_k=15)

    plan_text = generate_wizard_trip_plan(
        destination=destination,
        departure_city=departure_city,
        group_type=group_type,
        num_travelers=num_travelers,
        duration_days=duration_days,
        travel_style=travel_style,
        interests=interests,
        accommodation_type=accommodation_type,
        room_type=room_type,
        food_prefs=food_prefs,
        food_notes=food_notes,
        transport_type=transport_type,
        transport_class=transport_class,
        budget_pkr=budget_pkr,
        split_budget=split_budget,
        context=context,
    )

    return plan_text, context
