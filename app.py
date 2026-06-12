"""
Pakistan Travel Intelligence RAG System — Streamlit Application
Premium Pakistan travel assistant powered by semantic search and LLM generation.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(ROOT / "logs" / "travel_rag.log"), mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ── Streamlit page config — MUST be first Streamlit call ─────────────────────
st.set_page_config(
    page_title="Pakistan Travel Intelligence RAG",
    page_icon="🇵🇰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Global ──────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a4e 40%, #24243e 100%);
    min-height: 100vh;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ── Hero ─────────────────────────────────────────────────── */
.hero-container {
    background: linear-gradient(135deg,
        rgba(99,102,241,0.25) 0%,
        rgba(168,85,247,0.2) 50%,
        rgba(236,72,153,0.15) 100%);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #f472b6, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0;
}
.hero-subtitle {
    color: rgba(226,232,240,0.75);
    font-size: 1.05rem;
    font-weight: 300;
    margin: 0;
}

/* ── Glass cards ─────────────────────────────────────────── */
.glass-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.glass-card-accent {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.1));
    backdrop-filter: blur(16px);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* ── Section headers ─────────────────────────────────────── */
.section-header {
    font-size: 1.2rem;
    font-weight: 600;
    color: #a78bfa;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Stat chips ──────────────────────────────────────────── */
.stat-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 999px;
    padding: 0.3rem 0.9rem;
    font-size: 0.82rem;
    color: #c4b5fd;
    margin: 0.25rem;
}

/* ── Source badge ────────────────────────────────────────── */
.source-badge {
    display: inline-block;
    background: rgba(52,211,153,0.15);
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 6px;
    padding: 0.15rem 0.6rem;
    font-size: 0.75rem;
    color: #6ee7b7;
    margin-right: 0.4rem;
}

/* ── Score bar ───────────────────────────────────────────── */
.score-bar-container { margin-bottom: 0.4rem; }
.score-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    height: 6px;
    overflow: hidden;
    margin-top: 2px;
}
.score-bar-fill {
    height: 6px;
    border-radius: 999px;
    background: linear-gradient(90deg, #6366f1, #a78bfa);
}

/* ── Inputs ──────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > select {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
.stTextInput label, .stSelectbox label, .stTextArea label,
.stSlider label, .stRadio label {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
}

/* ── Chat messages ───────────────────────────────────────── */
.chat-user {
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 14px 14px 4px 14px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    color: #e2e8f0;
}
.chat-assistant {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px 14px 14px 4px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    color: #e2e8f0;
}

/* ── Expander ────────────────────────────────────────────── */
.streamlit-expanderHeader {
    color: #a78bfa !important;
    font-weight: 500 !important;
}
.streamlit-expanderContent {
    background: rgba(0,0,0,0.15) !important;
    border-radius: 0 0 10px 10px !important;
}

/* ── Tabs ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab"] {
    color: #94a3b8 !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom: 2px solid #a78bfa !important;
}

/* ── Markdown text ───────────────────────────────────────── */
.stMarkdown, .element-container p, .stMarkdown p {
    color: #cbd5e1 !important;
}
h1, h2, h3, h4 { color: #e2e8f0 !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Lazy imports (after path setup) ──────────────────────────────────────────
from src.config import GEMINI_API_KEY, GROQ_API_KEY, TOP_K, TRAVEL_INTERESTS, TRAVEL_STYLES, CURRENCY_SYMBOL
from src.data_loader import get_dataset_stats, is_index_stale, load_all_documents, save_data_hash
from src.generator import generate_answer, generate_comparison
from src.itinerary import build_itinerary, build_trip_plan
from src.retriever import get_retriever
from src.vector_store import get_vector_store


# ── Index initialisation ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def initialise_rag() -> bool:
    """Load or rebuild the FAISS index. Cached for the session."""
    store = get_vector_store()

    if not is_index_stale() and store.load():
        logger.info("Loaded existing FAISS index.")
        return True

    logger.info("Rebuilding FAISS index from CSV files…")
    docs = load_all_documents()
    if not docs:
        logger.error("No documents loaded — check /data directory.")
        return False

    store.build(docs)
    save_data_hash()
    return True


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 1.2rem 0 1rem;">
                <div style="font-size:2.4rem;">🇵🇰</div>
                <div style="font-size:1.1rem; font-weight:700; color:#a78bfa;">
                    Pakistan Travel
                </div>
                <div style="font-size:0.75rem; color:#64748b;">
                    Intelligence Platform
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        page = st.radio(
            "Navigation",
            [
                "🤖  Travel Assistant",
                "📅  Itinerary Builder", 
                "⚖️  Destination Compare",
                "🧳  Trip Planner",
                "📊  Knowledge Base",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # KB stats
        stats = get_dataset_stats()
        total_rows = sum(
            v.get("rows", 0) for v in stats.values() if isinstance(v, dict)
        )
        st.markdown(
            f"""
            <div class="glass-card" style="padding:1rem;">
                <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase;
                            letter-spacing:0.08em; margin-bottom:0.6rem;">
                    Knowledge Base
                </div>
                <div>
                    <span class="stat-chip">📄 {len(stats)} datasets</span>
                    <span class="stat-chip">📝 {total_rows} records</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # API key status
        gemini_ok = bool(GEMINI_API_KEY)
        groq_ok = bool(GROQ_API_KEY)
        st.markdown(
            f"""
            <div style="font-size:0.72rem; color:#64748b; margin-top:0.5rem;">
                {'✅' if gemini_ok else '❌'} Gemini &nbsp;
                {'✅' if groq_ok else '❌'} Groq
            </div>
            """,
            unsafe_allow_html=True,
        )

    return page


# ── Source transparency panel ─────────────────────────────────────────────────

def render_sources(chunks: list[dict]) -> None:
    if not chunks:
        return
    with st.expander("🔍 Retrieved Sources & Similarity Scores", expanded=False):
        for c in chunks:
            score_pct = int(c["score"] * 100)
            st.markdown(
                f"""
                <div class="glass-card" style="margin-bottom:0.6rem; padding:1rem;">
                    <div style="display:flex; justify-content:space-between;
                                align-items:center; margin-bottom:0.5rem;">
                        <div>
                            <span class="source-badge">{c['source']}</span>
                            <span style="color:#94a3b8; font-size:0.78rem;">
                                {c['dataset_type']}
                            </span>
                        </div>
                        <div style="color:#a78bfa; font-weight:600; font-size:0.85rem;">
                            {score_pct}%
                        </div>
                    </div>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width:{score_pct}%;"></div>
                    </div>
                    <div style="color:#94a3b8; font-size:0.8rem; margin-top:0.6rem;
                                line-height:1.5;">
                        {c['content'][:280]}{'…' if len(c['content']) > 280 else ''}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Pages ──────────────────────────────────────────────────────────────────────

def page_itinerary() -> None:
    st.markdown(
        """
        <div class="hero-container">
            <p class="hero-title">📅 Pakistan Itinerary Builder</p>
            <p class="hero-subtitle">
                Generate personalized day-wise travel plans for Pakistan destinations using real knowledge-base data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    import pandas as pd
    dest_df = pd.read_csv(ROOT / "data" / "destinations.csv")
    destination_names = dest_df["name"].tolist()

    col1, col2 = st.columns(2)
    with col1:
        destination = st.selectbox("Select Pakistan Destination", destination_names)
        duration = st.slider("Trip Duration (days)", 1, 14, 5)
        budget_pkr = st.slider(f"Budget per day ({CURRENCY_SYMBOL})", 2000, 25000, 8000, step=500)
        
    with col2:
        travel_style = st.selectbox("Travel Style", TRAVEL_STYLES)
        
        st.markdown("**Select Your Interests:**")
        # Create clickable interest chips
        selected_interests = []
        cols = st.columns(3)
        for i, interest in enumerate(TRAVEL_INTERESTS):
            with cols[i % 3]:
                if st.checkbox(interest, key=f"interest_{interest}"):
                    selected_interests.append(interest)

    if st.button("🗓️ Generate Pakistan Itinerary", use_container_width=True):
        if not selected_interests:
            st.warning("Please select at least one interest to personalize your itinerary.")
            return

        interests_str = ", ".join(selected_interests)
        
        with st.spinner(f"✨ Building {duration}-day itinerary for {destination}…"):
            itinerary_text, context = build_itinerary(
                destination=destination,
                duration_days=duration,
                budget_pkr=budget_pkr,
                travel_style=travel_style,
                interests=interests_str,
            )

        st.markdown(
            '<div class="glass-card-accent">',
            unsafe_allow_html=True,
        )
        st.markdown(itinerary_text)
        st.markdown("</div>", unsafe_allow_html=True)
        render_sources(context.chunks)


def page_comparison() -> None:
    st.markdown(
        """
        <div class="hero-container">
            <p class="hero-title">⚖️ Pakistan Destination Comparison</p>
            <p class="hero-subtitle">
                Compare Pakistan destinations side-by-side using authentic travel data with ratings, costs, and recommendations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    import pandas as pd
    dest_df = pd.read_csv(ROOT / "data" / "destinations.csv")
    destination_names = dest_df["name"].tolist()

    col1, col2 = st.columns(2)
    with col1:
        dest_a = st.selectbox("First Pakistan Destination", destination_names, index=0)
    with col2:
        dest_b = st.selectbox(
            "Second Pakistan Destination",
            [d for d in destination_names if d != dest_a],
            index=1 if len(destination_names) > 1 else 0,
        )

    if st.button("⚖️ Compare Pakistan Destinations", use_container_width=True):
        if dest_a == dest_b:
            st.warning("Please select two different Pakistan destinations.")
            return

        retriever = get_retriever()
        with st.spinner(f"📡 Retrieving data for {dest_a} and {dest_b}…"):
            ctx_a, ctx_b = retriever.retrieve_for_comparison(dest_a, dest_b)

        with st.spinner("🤖 Generating Pakistan destinations comparison…"):
            comparison_text = generate_comparison(dest_a, dest_b, ctx_a, ctx_b)

        st.markdown('<div class="glass-card-accent">', unsafe_allow_html=True)
        st.markdown(comparison_text)
        st.markdown("</div>", unsafe_allow_html=True)

        # Show destination details cards
        st.markdown("---")
        c1, c2 = st.columns(2)
        
        # Get destination details from CSV
        dest_a_info = dest_df[dest_df["name"] == dest_a].iloc[0]
        dest_b_info = dest_df[dest_df["name"] == dest_b].iloc[0]
        
        with c1:
            st.markdown(f"### 🏔️ {dest_a}")
            st.markdown(f"**Safety Rating:** {'⭐' * int(dest_a_info['safety_rating'])} ({dest_a_info['safety_rating']}/5)")
            st.markdown(f"**Budget Level:** {dest_a_info['budget_level'].title()}")
            st.markdown(f"**Best Time:** {dest_a_info['best_months']}")
            st.markdown(f"**Categories:** {dest_a_info['categories']}")
            st.markdown(f"**Description:** {dest_a_info['description']}")
            render_sources(ctx_a.chunks[:3])
            
        with c2:
            st.markdown(f"### 🏔️ {dest_b}")
            st.markdown(f"**Safety Rating:** {'⭐' * int(dest_b_info['safety_rating'])} ({dest_b_info['safety_rating']}/5)")
            st.markdown(f"**Budget Level:** {dest_b_info['budget_level'].title()}")
            st.markdown(f"**Best Time:** {dest_b_info['best_months']}")
            st.markdown(f"**Categories:** {dest_b_info['categories']}")
            st.markdown(f"**Description:** {dest_b_info['description']}")
            render_sources(ctx_b.chunks[:3])


def page_trip_planner() -> None:
    st.markdown(
        """
        <div class="hero-container">
            <p class="hero-title">🧳 Pakistan Trip Planner</p>
            <p class="hero-subtitle">
                Enter your preferences and get curated Pakistan travel recommendations from our knowledge base.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        budget_pkr = st.slider(f"Total Budget ({CURRENCY_SYMBOL})", 10000, 500000, 50000, step=5000)
        duration = st.slider("Trip Duration (days)", 3, 21, 7)
    with col2:
        travel_style = st.selectbox("Travel Style", TRAVEL_STYLES)
        
    st.markdown("**Select Your Interests:**")
    selected_interests = []
    cols = st.columns(4)
    for i, interest in enumerate(TRAVEL_INTERESTS):
        with cols[i % 4]:
            if st.checkbox(interest, key=f"planner_interest_{interest}"):
                selected_interests.append(interest)

    if st.button("🌍 Plan My Pakistan Trip", use_container_width=True):
        if not selected_interests:
            st.warning("Please select your interests to personalize your trip plan.")
            return

        interests_str = ", ".join(selected_interests)

        with st.spinner("🔍 Finding the best Pakistan destinations for you…"):
            plan_text, context = build_trip_plan(
                budget_pkr=budget_pkr,
                duration=duration,
                interests=interests_str,
                travel_style=travel_style,
            )

        st.markdown('<div class="glass-card-accent">', unsafe_allow_html=True)
        st.markdown(plan_text)
        st.markdown("</div>", unsafe_allow_html=True)
        render_sources(context.chunks)


def page_travel_assistant() -> None:
    st.markdown(
        """
        <div class="hero-container">
            <p class="hero-title">🤖 AI Travel Assistant</p>
            <p class="hero-subtitle">
                Your personal Pakistan travel expert. Ask anything about travel planning, budgets, destinations, and more.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    if "assistant_history" not in st.session_state:
        st.session_state.assistant_history = []

    # Display chat history
    for msg in st.session_state.assistant_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user">👤 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-assistant">🤖 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )

    # Chat input
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Ask your Pakistan travel question...",
            placeholder="e.g. What's the best time to visit Hunza? How much should I budget for Lahore?",
            key="assistant_input",
            label_visibility="collapsed"
        )
    with col2:
        send_btn = st.button("Send 💬", use_container_width=True)

    if send_btn and user_input.strip():
        # Add user message
        st.session_state.assistant_history.append({"role": "user", "content": user_input})
        
        with st.spinner("🤖 Thinking..."):
            try:
                # Test direct Groq API call
                from groq import Groq
                from src.config import GROQ_API_KEY, GROQ_MODEL
                
                if not GROQ_API_KEY:
                    response = "API key not configured. Please check your .env file."
                else:
                    client = Groq(api_key=GROQ_API_KEY)
                    
                    # Simple travel assistant prompt
                    prompt = f"""You are a Pakistan travel assistant. Help with travel questions about Pakistan.
                    
                    Question: {user_input}
                    
                    Please provide a helpful response about Pakistan travel, including costs in PKR when relevant."""
                    
                    completion = client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1000,
                        temperature=0.7,
                    )
                    
                    response = completion.choices[0].message.content
                    
            except Exception as e:
                response = f"Error: {str(e)}"
                st.error(f"API Error: {str(e)}")
            
        st.session_state.assistant_history.append({"role": "assistant", "content": response})
        st.rerun()
        
    if st.button("🗑️ Clear Assistant Chat"):
        st.session_state.assistant_history = []
        st.rerun()


def page_knowledge_base() -> None:
    st.markdown(
        """
        <div class="hero-container">
            <p class="hero-title">📊 Pakistan Travel Knowledge Base</p>
            <p class="hero-subtitle">
                Browse and inspect the Pakistan travel datasets powering this RAG system with destinations, hotels, activities, and travel information.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stats = get_dataset_stats()
    vs = get_vector_store()
    vs_stats = vs.stats()

    # Index stats bar
    cols = st.columns(4)
    tiles = [
        ("📄", "Datasets", str(len(stats))),
        ("🔢", "Total Records", str(sum(v.get("rows", 0) for v in stats.values() if isinstance(v, dict)))),
        ("🧠", "Vector Docs", str(vs_stats.get("total_documents", "–"))),
        ("🗄️", "Index Size", str(vs_stats.get("index_size", "–"))),
    ]
    for col, (icon, label, value) in zip(cols, tiles):
        with col:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align:center; padding:1rem;">
                    <div style="font-size:1.6rem;">{icon}</div>
                    <div style="font-size:1.4rem; font-weight:700; color:#a78bfa;">{value}</div>
                    <div style="font-size:0.75rem; color:#64748b;">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Per-dataset cards
    import pandas as pd
    for fname, info in stats.items():
        if "error" in info:
            st.error(f"{fname}: {info['error']}")
            continue

        with st.expander(f"📁 {fname}  ·  {info['rows']} rows  ·  {info['dataset_type']}", expanded=False):
            df = pd.read_csv(ROOT / "data" / fname)
            st.markdown(
                f"""
                <div style="display:flex; flex-wrap:wrap; gap:0.4rem; margin-bottom:0.8rem;">
                    {''.join(f'<span class="stat-chip">{c}</span>' for c in info['columns'])}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(
                df.head(10),
                use_container_width=True,
                hide_index=True,
            )

    # Rebuild button
    st.markdown("---")
    st.markdown(
        "<p style='color:#64748b; font-size:0.85rem;'>Rebuild the vector index if you update the CSV files:</p>",
        unsafe_allow_html=True,
    )
    if st.button("🔄 Rebuild Vector Index", key="rebuild_idx"):
        with st.spinner("Rebuilding…"):
            get_vector_store.cache_clear() if hasattr(get_vector_store, "cache_clear") else None
            store = get_vector_store()
            docs = load_all_documents()
            store.build(docs)
            save_data_hash()
        st.success(f"✅ Index rebuilt with {len(docs)} documents.")
        st.cache_resource.clear()
        st.rerun()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # Initialise RAG
    with st.spinner("⚙️ Initialising Travel Intelligence Engine…"):
        ok = initialise_rag()

    if not ok:
        st.error(
            "❌ Failed to initialise the knowledge base. "
            "Please ensure CSV files are present in the /data directory."
        )
        st.stop()

    page = render_sidebar()

    if "Assistant" in page:
        page_travel_assistant()
    elif "Itinerary" in page:
        page_itinerary()
    elif "Compare" in page:
        page_comparison()
    elif "Planner" in page:
        page_trip_planner()
    elif "Knowledge" in page:
        page_knowledge_base()


if __name__ == "__main__":
    main()
