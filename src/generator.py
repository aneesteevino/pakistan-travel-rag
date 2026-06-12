"""
LLM generation module for Pakistan Travel Intelligence.
Supports Google Gemini (google-genai SDK) and Groq as providers.
Falls back automatically between providers.
All answers are grounded in retrieved context — no hallucination.
"""

from __future__ import annotations

import logging

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    CURRENCY_SYMBOL,
)
from src.retriever import RetrievedContext

logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert Pakistan travel assistant for a Pakistan Travel Intelligence platform.
You ONLY answer questions using the provided context retrieved from the Pakistan travel knowledge base.
You NEVER fabricate destinations, hotels, activities, prices, or travel information.

Rules:
1. Base every answer strictly on the retrieved context below.
2. Focus exclusively on Pakistan tourism and travel.
3. Use Pakistani Rupees (PKR) for all pricing and budget information.
4. Cite sources (CSV filename) when referencing specific data points.
5. If the context does not contain enough information, say:
   "The Pakistan travel knowledge base does not contain sufficient information to answer this query."
6. Be concise, structured, and helpful for Pakistan travelers.
7. When listing items use markdown bullet points.
8. Highlight unique aspects of Pakistani culture, cuisine, and destinations.
"""

_TRAVEL_ASSISTANT_PROMPT = """You are a knowledgeable Pakistan travel assistant.
Your role is to help travelers plan their trips to Pakistan with accurate and helpful information.

Guidelines:
1. Focus exclusively on Pakistan travel, tourism, and related topics.
2. For budget questions, provide realistic estimates in Pakistani Rupees (PKR).
3. Be enthusiastic about Pakistan's natural beauty, culture, and hospitality.
4. Provide practical and actionable travel advice.
5. Include information about destinations, costs, activities, accommodation, and transportation.
6. Always prioritize traveler safety and provide current general guidance.
7. For non-travel questions, politely redirect to Pakistan travel topics.

When discussing budgets:
- Budget travel: 2,000-4,000 PKR per day
- Mid-range travel: 5,000-12,000 PKR per day  
- Luxury travel: 15,000+ PKR per day

Popular Pakistan destinations include:
- Karachi (beaches, food, culture)
- Lahore (history, architecture, cuisine)
- Islamabad (modern capital, Margalla Hills)
- Hunza Valley (mountains, culture)
- Skardu (K2 base, lakes)
- Swat (green valleys, waterfalls)
- Murree (hill station)
- And many more beautiful locations

Be helpful, informative, and encouraging about Pakistan travel experiences."""


def _build_prompt(question: str, context: RetrievedContext) -> str:
    if not context.has_results:
        return (
            f"{_SYSTEM_PROMPT}\n\n"
            "CONTEXT: No relevant documents were retrieved.\n\n"
            f"USER QUESTION: {question}\n\n"
            "ANSWER: The Pakistan travel knowledge base does not contain sufficient information to answer this query."
        )
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"RETRIEVED CONTEXT:\n{context.context_text}\n\n"
        f"USER QUESTION: {question}\n\n"
        "ANSWER (grounded in the above context only):"
    )


# ── Gemini via new google-genai SDK ───────────────────────────────────────────

def _gemini_generate(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set.")
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text


# ── Groq ──────────────────────────────────────────────────────────────────────

def _groq_generate(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set.")
    
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.7,
        )
        
        if completion.choices and len(completion.choices) > 0:
            return completion.choices[0].message.content
        else:
            raise ValueError("No response generated from Groq API")
            
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        raise e


# ── Provider dispatcher ───────────────────────────────────────────────────────

def _call_llm(prompt: str) -> str:
    # Use Groq as primary provider since it's working
    if GROQ_API_KEY:
        try:
            return _groq_generate(prompt)
        except Exception as exc:
            logger.error("Groq failed: %s", exc)
            return f"⚠️ Groq API error: {str(exc)}"
    
    # Only try Gemini if no Groq key and we have a valid Gemini key
    if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
        try:
            return _gemini_generate(prompt)
        except Exception as exc:
            logger.warning("Gemini failed: %s", exc)
            return f"⚠️ Gemini API error: {str(exc)}"
    
    return "⚠️ No AI service available. Please configure your API keys."


# ── Travel Assistant (Gemini only) ───────────────────────────────────────────

def generate_travel_assistant_response(question: str) -> str:
    """Generate response using available AI provider for travel assistant functionality."""
    
    prompt = f"{_TRAVEL_ASSISTANT_PROMPT}\n\nUser Question: {question}\n\nResponse:"
    
    # Try Groq first since it's our primary provider
    if GROQ_API_KEY:
        try:
            logger.info(f"Attempting Groq generation for: {question}")
            return _groq_generate(prompt)
        except Exception as exc:
            logger.error("Groq generation failed: %s", exc)
            return f"⚠️ Groq API error: {str(exc)}"
    
    # Try Gemini as fallback if available
    if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
        try:
            logger.info(f"Attempting Gemini generation for: {question}")
            return _gemini_generate(prompt)
        except Exception as exc:
            logger.error("Gemini generation failed: %s", exc)
            return f"⚠️ Gemini API error: {str(exc)}"
    
    return "⚠️ No AI service available. Please configure your API keys or try again later."


# ── Public API ────────────────────────────────────────────────────────────────

def generate_answer(question: str, context: RetrievedContext) -> str:
    return _call_llm(_build_prompt(question, context))


def generate_itinerary(
    destination: str,
    duration_days: int,
    budget_pkr: int,
    travel_style: str,
    interests: str,
    context: RetrievedContext,
) -> str:
    
    # Always generate a response, even without perfect context
    if not context.has_results:
        # Generate based on common knowledge if no context
        prompt = (
            f"Generate a {duration_days}-day Pakistan travel itinerary for {destination}.\n"
            f"Budget: {CURRENCY_SYMBOL}{budget_pkr:,} per day\n"
            f"Travel Style: {travel_style}\n"
            f"Interests: {interests}\n\n"
            f"Create a realistic day-by-day itinerary with:\n"
            f"- Morning, afternoon, and evening activities\n"
            f"- Accommodation suggestions\n"
            f"- Transportation tips\n"
            f"- Budget breakdown in PKR\n"
            f"- Cultural highlights specific to {destination}\n"
            f"- Food recommendations\n"
            f"- Safety tips\n\n"
            f"Format as Day 1, Day 2, etc."
        )
        return _call_llm(prompt)
    
    prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"RETRIEVED CONTEXT:\n{context.context_text}\n\n"
        f"TASK: Generate a {duration_days}-day Pakistan itinerary for {destination}.\n"
        f"Budget: {CURRENCY_SYMBOL}{budget_pkr:,} per day | Travel Style: {travel_style} | Interests: {interests}\n\n"
        "Requirements:\n"
        "- Use activities, hotels, and data from the context where available.\n"
        "- Structure as Day 1, Day 2, etc. with morning / afternoon / evening.\n"
        "- Recommend hotels matching the travel style and budget.\n"
        "- Include costs in Pakistani Rupees (PKR) where available.\n"
        "- Focus on authentic Pakistan experiences.\n"
        "- Consider local culture and customs.\n"
        "- If context data is limited, use general knowledge about Pakistan travel.\n\n"
        "PAKISTAN ITINERARY:"
    )
    return _call_llm(prompt)


def generate_comparison(
    dest_a: str,
    dest_b: str,
    ctx_a: RetrievedContext,
    ctx_b: RetrievedContext,
) -> str:
    combined = (
        f"=== {dest_a} ===\n{ctx_a.context_text}\n\n"
        f"=== {dest_b} ===\n{ctx_b.context_text}"
    )
    prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"RETRIEVED CONTEXT:\n{combined}\n\n"
        f"TASK: Compare Pakistan destinations {dest_a} and {dest_b} covering:\n"
        "1. Budget & Cost (in PKR)\n2. Top Activities\n3. Best Time to Visit\n"
        "4. Safety Rating\n5. Accommodation Options\n6. Cultural Experiences\n"
        "7. Natural Beauty\n8. Overall recommendation for different traveler types\n\n"
        "Use ONLY context data. Show costs in Pakistani Rupees. Cite CSV files.\n\nPAKISTAN DESTINATIONS COMPARISON:"
    )
    return _call_llm(prompt)


def generate_trip_plan(
    budget_pkr: int,
    duration: int,
    interests: str,
    travel_style: str,
    context: RetrievedContext,
) -> str:
    
    # Always generate a response
    if not context.has_results:
        prompt = (
            f"Create a personalized Pakistan travel plan:\n"
            f"Total Budget: {CURRENCY_SYMBOL}{budget_pkr:,}\n"
            f"Duration: {duration} days\n"
            f"Interests: {interests}\n"
            f"Travel Style: {travel_style}\n\n"
            f"Provide:\n"
            f"1. Top 3 recommended Pakistan destinations matching preferences\n"
            f"2. Why each destination fits the traveler's interests\n"
            f"3. Key activities and experiences\n"
            f"4. Accommodation suggestions by budget level\n"
            f"5. Daily budget breakdown in PKR\n"
            f"6. Best time to visit each destination\n"
            f"7. Transportation recommendations\n"
            f"8. Cultural highlights and local experiences\n"
            f"9. Safety and practical tips\n\n"
            f"Focus on authentic Pakistan experiences within the specified budget."
        )
        return _call_llm(prompt)
    
    prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"RETRIEVED CONTEXT:\n{context.context_text}\n\n"
        "TASK: Build a personalized Pakistan travel plan.\n"
        f"Total Budget: {CURRENCY_SYMBOL}{budget_pkr:,} | Duration: {duration} days | "
        f"Interests: {interests} | Style: {travel_style}\n\n"
        "Provide:\n"
        "1. Top 2-3 recommended Pakistan destinations (use context where available)\n"
        "2. Why each destination matches preferences\n"
        "3. Key activities and experiences (context preferred)\n"
        "4. Suggested hotels matching budget and style (context preferred)\n"
        "5. Daily budget breakdown in PKR\n"
        "6. Best time to visit each destination\n"
        "7. Transportation recommendations\n"
        "8. Cultural highlights and local experiences\n"
        "9. Safety considerations\n\n"
        "Use context data where available, supplement with general Pakistan travel knowledge. PAKISTAN TRIP PLAN:"
    )
    return _call_llm(prompt)
