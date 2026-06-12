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
/* ── Global Dark Obsidian Theme ──────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0D0D0D;
    background-image: 
        radial-gradient(circle at 25% 25%, rgba(255, 87, 34, 0.03) 0%, transparent 50%),
        radial-gradient(circle at 75% 75%, rgba(255, 111, 0, 0.02) 0%, transparent 50%);
    min-height: 100vh;
    color: #e2e8f0;
}

/* ── Sidebar Dark Glassmorphism ───────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(13, 13, 13, 0.85);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 87, 34, 0.15);
    box-shadow: inset -1px 0 0 rgba(255, 87, 34, 0.08);
}

[data-testid="stSidebar"] * { 
    color: #e2e8f0 !important; 
}

[data-testid="stSidebar"] .stRadio > div {
    background: rgba(18, 18, 18, 0.6);
    border-radius: 12px;
    padding: 0.5rem;
}

/* ── Hero Section with Premium Glow ──────────────────────── */
.hero-container {
    background: rgba(18, 18, 18, 0.7);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 87, 34, 0.15);
    border-radius: 24px;
    padding: 3rem 3.5rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 
        0 0 40px rgba(255, 87, 34, 0.08),
        inset 0 1px 0 rgba(255, 255, 255, 0.05);
    position: relative;
    overflow: hidden;
}

.hero-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, 
        transparent, 
        rgba(255, 87, 34, 0.4), 
        transparent);
}

.hero-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #ffffff 0%, #ff5722 40%, #ff6f00 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.8rem 0;
    text-shadow: 0 0 30px rgba(255, 87, 34, 0.3);
}

.hero-subtitle {
    color: rgba(226, 232, 240, 0.7);
    font-size: 1.1rem;
    font-weight: 400;
    margin: 0;
    letter-spacing: 0.02em;
}

/* ── Dark Glass Cards with Rim Lighting ──────────────────── */
.glass-card {
    background: rgba(18, 18, 18, 0.6);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: 
        0 8px 32px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.05);
    transition: all 0.3s ease;
}

.glass-card:hover {
    border-color: rgba(255, 87, 34, 0.2);
    box-shadow: 
        0 8px 32px rgba(0, 0, 0, 0.4),
        0 0 20px rgba(255, 87, 34, 0.1),
        inset 0 1px 0 rgba(255, 255, 255, 0.08);
    transform: translateY(-2px);
}

.glass-card-accent {
    background: rgba(18, 18, 18, 0.8);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 87, 34, 0.25);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 
        0 0 40px rgba(255, 87, 34, 0.15),
        0 8px 32px rgba(0, 0, 0, 0.4),
        inset 0 1px 0 rgba(255, 87, 34, 0.1);
    position: relative;
}

.glass-card-accent::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, 
        transparent, 
        rgba(255, 87, 34, 0.6), 
        transparent);
}

/* ── Section Headers with Orange Glow ────────────────────── */
.section-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #ff6f00;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    text-shadow: 0 0 10px rgba(255, 111, 0, 0.3);
}

/* ── Stat Chips with Dark Theme ──────────────────────────── */
.stat-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(18, 18, 18, 0.8);
    border: 1px solid rgba(255, 87, 34, 0.2);
    border-radius: 999px;
    padding: 0.4rem 1rem;
    font-size: 0.85rem;
    color: #ffa726;
    margin: 0.25rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    transition: all 0.2s ease;
}

.stat-chip:hover {
    border-color: rgba(255, 87, 34, 0.4);
    box-shadow: 0 0 15px rgba(255, 87, 34, 0.2);
}

/* ── Source Badge Premium Style ──────────────────────────── */
.source-badge {
    display: inline-block;
    background: rgba(18, 18, 18, 0.9);
    border: 1px solid rgba(255, 167, 38, 0.3);
    border-radius: 8px;
    padding: 0.2rem 0.8rem;
    font-size: 0.75rem;
    color: #ffa726;
    margin-right: 0.5rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

/* ── Score Bar with Orange Gradient ──────────────────────── */
.score-bar-container { margin-bottom: 0.5rem; }
.score-bar-bg {
    background: rgba(18, 18, 18, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
    margin-top: 3px;
}
.score-bar-fill {
    height: 8px;
    border-radius: 999px;
    background: linear-gradient(90deg, #ff5722, #ff6f00, #ffab00);
    box-shadow: 0 0 10px rgba(255, 87, 34, 0.4);
}

/* ── Dark Input Styling ──────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > select {
    background: rgba(18, 18, 18, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2) !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus,
.stSelectbox > div > div > select:focus {
    border-color: rgba(255, 87, 34, 0.4) !important;
    box-shadow: 
        inset 0 2px 4px rgba(0, 0, 0, 0.2),
        0 0 0 2px rgba(255, 87, 34, 0.1) !important;
}

.stTextInput label, .stSelectbox label, .stTextArea label,
.stSlider label, .stRadio label, .stCheckbox label {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
}

/* ── Premium Orange Glow Buttons ─────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #ff5722 0%, #ff6f00 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 0.7rem 2rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 
        0 4px 15px rgba(255, 87, 34, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #e64a19 0%, #f57c00 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 
        0 8px 25px rgba(255, 87, 34, 0.4),
        0 0 20px rgba(255, 87, 34, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.25) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Chat Messages Cyberpunk Style ───────────────────────── */
.chat-user {
    background: rgba(18, 18, 18, 0.8);
    border: 1px solid rgba(255, 87, 34, 0.3);
    border-radius: 16px 16px 4px 16px;
    padding: 1rem 1.4rem;
    margin: 0.8rem 0;
    color: #e2e8f0;
    box-shadow: 
        0 4px 20px rgba(0, 0, 0, 0.3),
        0 0 15px rgba(255, 87, 34, 0.1);
}

.chat-assistant {
    background: rgba(18, 18, 18, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px 16px 16px 4px;
    padding: 1rem 1.4rem;
    margin: 0.8rem 0;
    color: #e2e8f0;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

/* ── Expander Dark Theme ─────────────────────────────────── */
.streamlit-expanderHeader {
    color: #ff6f00 !important;
    font-weight: 600 !important;
    background: rgba(18, 18, 18, 0.6) !important;
    border: 1px solid rgba(255, 87, 34, 0.15) !important;
    border-radius: 12px !important;
}

.streamlit-expanderContent {
    background: rgba(13, 13, 13, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
}

/* ── Tabs with Orange Accent ─────────────────────────────── */
.stTabs [data-baseweb="tab"] {
    color: #94a3b8 !important;
    font-weight: 500 !important;
    background: rgba(18, 18, 18, 0.4) !important;
    border-radius: 8px 8px 0 0 !important;
}

.stTabs [aria-selected="true"] {
    color: #ff6f00 !important;
    border-bottom: 3px solid #ff5722 !important;
    background: rgba(18, 18, 18, 0.8) !important;
    box-shadow: 0 0 15px rgba(255, 87, 34, 0.2) !important;
}

/* ── Typography with High Contrast ───────────────────────── */
.stMarkdown, .element-container p, .stMarkdown p {
    color: #cbd5e1 !important;
    line-height: 1.6 !important;
}

h1, h2, h3, h4 { 
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* ── Dataframe Dark Theme ────────────────────────────────── */
.stDataFrame {
    background: rgba(18, 18, 18, 0.8) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}

/* ── Slider Dark Theme ───────────────────────────────────── */
.stSlider > div > div > div > div {
    background: #ff5722 !important;
}

.stSlider > div > div > div > div > div {
    background: #ff6f00 !important;
    box-shadow: 0 0 10px rgba(255, 87, 34, 0.4) !important;
}

/* ── Checkbox Dark Theme ─────────────────────────────────── */
.stCheckbox > label {
    background: rgba(18, 18, 18, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    padding: 0.5rem 0.8rem !important;
    margin: 0.2rem 0 !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

.stCheckbox > label:hover {
    border-color: rgba(255, 87, 34, 0.3) !important;
    background: rgba(18, 18, 18, 0.8) !important;
}

.stCheckbox > label > span:first-child {
    background: rgba(18, 18, 18, 0.8) !important;
    border: 2px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 4px !important;
    width: 18px !important;
    height: 18px !important;
}

.stCheckbox > label[data-testid="stCheckbox"] > span:first-child[data-checked="true"] {
    background: linear-gradient(135deg, #ff5722, #ff6f00) !important;
    border-color: #ff5722 !important;
    box-shadow: 0 0 8px rgba(255, 87, 34, 0.4) !important;
}

.stCheckbox > label > div {
    color: #e2e8f0 !important;
    font-weight: 500 !important;
    margin-left: 0.5rem !important;
}

/* ── Radio Dark Theme ────────────────────────────────────── */
.stRadio > label > div {
    background: rgba(18, 18, 18, 0.6) !important;
    border-radius: 8px !important;
    padding: 0.3rem !important;
}

.stRadio > label > div[data-checked="true"] {
    background: rgba(255, 87, 34, 0.15) !important;
    border: 1px solid rgba(255, 87, 34, 0.3) !important;
}

/* ── Spinner Dark Theme ──────────────────────────────────── */
.stSpinner > div {
    border-color: #ff5722 !important;
}

/* ── Success/Error/Warning Dark Theme ────────────────────── */
.stSuccess {
    background: rgba(76, 175, 80, 0.1) !important;
    border: 1px solid rgba(76, 175, 80, 0.3) !important;
    color: #81c784 !important;
}

.stError {
    background: rgba(244, 67, 54, 0.1) !important;
    border: 1px solid rgba(244, 67, 54, 0.3) !important;
    color: #e57373 !important;
}

.stWarning {
    background: rgba(255, 152, 0, 0.1) !important;
    border: 1px solid rgba(255, 152, 0, 0.3) !important;
    color: #ffb74d !important;
}

/* ── Scrollbar Dark Theme ────────────────────────────────── */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(18, 18, 18, 0.8);
}

::-webkit-scrollbar-thumb {
    background: rgba(255, 87, 34, 0.4);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 87, 34, 0.6);
}

/* ── Mobile Touch Optimization ───────────────────────────── */
* {
    -webkit-tap-highlight-color: rgba(255, 87, 34, 0.2);
    -webkit-touch-callout: none;
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
}

input, textarea, [contenteditable] {
    -webkit-user-select: text;
    -moz-user-select: text;
    -ms-user-select: text;
    user-select: text;
}

.stButton > button,
.stCheckbox > label,
.stRadio > label,
.stSelectbox > div {
    touch-action: manipulation;
}

/* ── Improved Focus States for Accessibility ──────────────── */
.stButton > button:focus-visible,
.stCheckbox > label:focus-visible,
.stRadio > label:focus-visible {
    outline: 2px solid #ff6f00;
    outline-offset: 2px;
}
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
            <div style="text-align:center; padding: 1.5rem 0 1.2rem;">
                <div style="font-size:3rem; filter: drop-shadow(0 0 10px rgba(255, 87, 34, 0.4));">🇵🇰</div>
                <div style="font-size:1.2rem; font-weight:700; color:#ff6f00; text-shadow: 0 0 8px rgba(255, 111, 0, 0.3);">
                    Pakistan Travel
                </div>
                <div style="font-size:0.8rem; color:#9ca3af; letter-spacing: 0.05em;">
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
            <div class="glass-card" style="padding:1.2rem;">
                <div style="font-size:0.7rem; color:#9ca3af; text-transform:uppercase;
                            letter-spacing:0.1em; margin-bottom:0.8rem; font-weight: 500;">
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
            <div style="font-size:0.75rem; color:#9ca3af; margin-top:0.8rem; 
                        padding: 0.8rem; background: rgba(18, 18, 18, 0.6); 
                        border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05);">
                <div style="margin-bottom: 0.3rem; color: #ff6f00; font-weight: 500;">API Status</div>
                {'✅' if gemini_ok else '❌'} Gemini &nbsp;&nbsp;
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
        # Show currently selected interests
        if st.session_state.get('selected_interests'):
            st.markdown(
                f"""
                <div style="margin-bottom: 0.8rem;">
                    <span style="font-size: 0.8rem; color: #9ca3af;">Currently selected: </span>
                    {' '.join([f'<span class="stat-chip" style="font-size: 0.7rem; padding: 0.2rem 0.6rem;">{interest}</span>' for interest in st.session_state.selected_interests])}
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Initialize shared session state for interests if not exists
        if 'selected_interests' not in st.session_state:
            st.session_state.selected_interests = []
        
        # Create clickable interest chips
        selected_interests = []
        cols = st.columns(3)
        for i, interest in enumerate(TRAVEL_INTERESTS):
            with cols[i % 3]:
                is_checked = st.checkbox(
                    interest, 
                    key=f"itinerary_interest_{i}_{interest.replace(' ', '_')}",
                    value=interest in st.session_state.selected_interests
                )
                if is_checked and interest not in st.session_state.selected_interests:
                    st.session_state.selected_interests.append(interest)
                elif not is_checked and interest in st.session_state.selected_interests:
                    st.session_state.selected_interests.remove(interest)
                
                if is_checked:
                    selected_interests.append(interest)
        
        # Add a clear button
        if st.button("🗑️ Clear Interest Selections", key="clear_itinerary_interests"):
            st.session_state.selected_interests = []
            st.rerun()

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
    # Show currently selected interests
    if st.session_state.get('selected_interests'):
        st.markdown(
            f"""
            <div style="margin-bottom: 0.8rem;">
                <span style="font-size: 0.8rem; color: #9ca3af;">Currently selected: </span>
                {' '.join([f'<span class="stat-chip" style="font-size: 0.7rem; padding: 0.2rem 0.6rem;">{interest}</span>' for interest in st.session_state.selected_interests])}
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Use the same shared session state for interests
    if 'selected_interests' not in st.session_state:
        st.session_state.selected_interests = []
    
    selected_interests = []
    cols = st.columns(4)
    for i, interest in enumerate(TRAVEL_INTERESTS):
        with cols[i % 4]:
            is_checked = st.checkbox(
                interest, 
                key=f"planner_interest_{i}_{interest.replace(' ', '_')}",
                value=interest in st.session_state.selected_interests
            )
            if is_checked and interest not in st.session_state.selected_interests:
                st.session_state.selected_interests.append(interest)
            elif not is_checked and interest in st.session_state.selected_interests:
                st.session_state.selected_interests.remove(interest)
            
            if is_checked:
                selected_interests.append(interest)
    
    # Add a clear button
    if st.button("🗑️ Clear Interest Selections", key="clear_planner_interests"):
        st.session_state.selected_interests = []
        st.rerun()

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
                <div class="glass-card" style="text-align:center; padding:1.5rem;">
                    <div style="font-size:2rem; margin-bottom: 0.5rem; filter: drop-shadow(0 0 8px rgba(255, 87, 34, 0.3));">{icon}</div>
                    <div style="font-size:1.6rem; font-weight:700; color:#ff6f00; text-shadow: 0 0 8px rgba(255, 111, 0, 0.3);">{value}</div>
                    <div style="font-size:0.8rem; color:#9ca3af; margin-top: 0.3rem;">{label}</div>
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
        "<p style='color:#9ca3af; font-size:0.9rem; margin-bottom: 1rem;'>Rebuild the vector index if you update the CSV files:</p>",
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
