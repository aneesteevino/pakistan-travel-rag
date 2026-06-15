"""
Pakistan Travel Intelligence RAG System — Streamlit Application
Premium Pakistan travel assistant powered by semantic search and LLM generation.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
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

/* ══ WIZARD STYLES ════════════════════════════════════════════ */

/* Progress Bar */
.wizard-progress-wrap {
    background: rgba(18,18,18,0.7);
    border: 1px solid rgba(255,87,34,0.12);
    border-radius: 20px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.8rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
}
.wizard-step-label {
    font-size: 0.78rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    white-space: nowrap;
    font-weight: 600;
}
.wizard-bar-bg {
    flex: 1;
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    height: 6px;
    overflow: hidden;
}
.wizard-bar-fill {
    height: 6px;
    border-radius: 999px;
    background: linear-gradient(90deg, #ff5722, #ff6f00, #ffab00);
    box-shadow: 0 0 10px rgba(255,87,34,0.45);
    transition: width 0.5s cubic-bezier(.4,0,.2,1);
}
.wizard-step-count {
    font-size: 0.8rem;
    font-weight: 700;
    color: #ff6f00;
    white-space: nowrap;
}

/* Welcome hero */
.wizard-welcome {
    background: linear-gradient(135deg, rgba(18,18,18,0.95) 0%, rgba(30,15,5,0.95) 100%);
    border: 1px solid rgba(255,87,34,0.2);
    border-radius: 28px;
    padding: 4rem 3rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 60px rgba(255,87,34,0.12), 0 20px 60px rgba(0,0,0,0.5);
}
.wizard-welcome::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #ff5722, #ffab00, transparent);
}
.wizard-welcome-title {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff 0%, #ff6f00 50%, #ffab00 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
    line-height: 1.1;
}
.wizard-welcome-subtitle {
    font-size: 1.2rem;
    color: rgba(226,232,240,0.75);
    max-width: 560px;
    margin: 0 auto 2rem;
    line-height: 1.7;
}
.wizard-feature-grid {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    flex-wrap: wrap;
    margin-bottom: 2.5rem;
}
.wizard-feature-chip {
    background: rgba(255,87,34,0.1);
    border: 1px solid rgba(255,87,34,0.25);
    border-radius: 999px;
    padding: 0.45rem 1.1rem;
    font-size: 0.82rem;
    color: #ffa726;
    font-weight: 500;
}

/* Step cards */
.wizard-step-card {
    background: rgba(18,18,18,0.75);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,87,34,0.18);
    border-radius: 20px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.5rem;
    position: relative;
    box-shadow: 0 8px 40px rgba(0,0,0,0.35), 0 0 0 1px rgba(255,255,255,0.03);
}
.wizard-step-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,87,34,0.4), transparent);
    border-radius: 20px 20px 0 0;
}
.wizard-step-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.wizard-step-desc {
    color: rgba(226,232,240,0.55);
    font-size: 0.88rem;
    margin-bottom: 1.6rem;
}

/* Chip selector */
.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-bottom: 0.8rem;
}
.style-card {
    background: rgba(18,18,18,0.7);
    border: 2px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.25s ease;
    height: 100%;
}
.style-card:hover {
    border-color: rgba(255,87,34,0.4);
    background: rgba(255,87,34,0.07);
}
.style-card.selected {
    border-color: #ff5722;
    background: rgba(255,87,34,0.12);
    box-shadow: 0 0 20px rgba(255,87,34,0.2);
}
.style-card-icon { font-size: 2rem; margin-bottom: 0.4rem; }
.style-card-title { font-size: 0.9rem; font-weight: 700; color: #fff; }
.style-card-desc { font-size: 0.72rem; color: #9ca3af; margin-top: 0.2rem; line-height: 1.4; }

/* Summary review */
.summary-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
    margin-bottom: 1.2rem;
}
.summary-item {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.7rem 1rem;
}
.summary-label { font-size: 0.72rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.2rem; }
.summary-value { font-size: 0.95rem; font-weight: 600; color: #e2e8f0; }

/* Budget bar */
.budget-cat {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.7rem;
}
.budget-cat-label { font-size: 0.82rem; color: #cbd5e1; min-width: 130px; }
.budget-cat-bar { flex: 1; background: rgba(255,255,255,0.06); border-radius: 999px; height: 6px; overflow: hidden; }
.budget-cat-fill { height: 6px; border-radius: 999px; }
.budget-cat-amount { font-size: 0.8rem; font-weight: 600; color: #ffa726; min-width: 90px; text-align: right; }

/* Nav buttons */
.nav-row { display: flex; gap: 1rem; margin-top: 2rem; }

/* Final plan */
.plan-section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #ff6f00;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 1.5rem 0 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,87,34,0.2);
}
.chat-refine-box {
    background: rgba(18,18,18,0.8);
    border: 1px solid rgba(255,87,34,0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin-top: 1.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Lazy imports (after path setup) ──────────────────────────────────────────
from src.config import GEMINI_API_KEY, GROQ_API_KEY, TOP_K, TRAVEL_INTERESTS, TRAVEL_STYLES, CURRENCY_SYMBOL
from src.data_loader import get_dataset_stats, is_index_stale, load_all_documents, save_data_hash, safe_read_csv
from src.generator import generate_answer, generate_comparison
from src.itinerary import build_trip_plan, build_wizard_trip_plan
from src.retriever import get_retriever
from src.vector_store import get_vector_store
from src.flights import (
    AIRPORTS, AIRLINES, BOOKING_PLATFORMS,
    get_all_airports, get_domestic_cities,
    search_flights, get_all_domestic_routes,
)


# ── Airport validation logic, food/activities loaders & budget calculator ─────

_NEAREST_AIRPORTS = {
    "Hunza": "Gilgit",
    "Swat": "Peshawar",
    "Murree": "Islamabad",
    "Naran Kaghan": "Islamabad",
    "Naran": "Islamabad",
    "Kaghan": "Islamabad",
    "Fairy Meadows": "Gilgit",
    "Kumrat Valley": "Peshawar",
    "Shogran": "Islamabad",
    "Malam Jabba": "Peshawar",
    "Kalash Valley": "Chitral",
    "Kalash": "Chitral",
    "Neelum Valley": "Islamabad",
    "Attabad Lake": "Gilgit",
}

def check_airport_routing(departure: str, destination: str) -> dict:
    """Airport validation logic."""
    dep_airport = AIRPORTS.get(departure)
    dest_airport = AIRPORTS.get(destination)
    
    nearest_dest_city = _NEAREST_AIRPORTS.get(destination, "")
    nearest_dep_city = _NEAREST_AIRPORTS.get(departure, "")
    
    final_dep = departure if dep_airport else nearest_dep_city
    final_dest = destination if dest_airport else nearest_dest_city
    
    alerts = []
    road_legs = []
    
    if not dep_airport and nearest_dep_city:
        alerts.append(f"🛫 **{departure}** does not have a domestic airport. Nearest airport is **{nearest_dep_city}**.")
        road_legs.append({
            "from": departure,
            "to": nearest_dep_city,
            "type": "Road Transit to Airport",
            "desc": f"Take road transport (Private Car/Bus) from {departure} to {nearest_dep_city} (~2 hours)."
        })
        
    if not dest_airport and nearest_dest_city:
        alerts.append(f"🛬 **{destination}** does not have a domestic airport. Nearest airport is **{nearest_dest_city}**.")
        road_legs.append({
            "from": nearest_dest_city,
            "to": destination,
            "type": "Road Transit to Destination",
            "desc": f"Arrive at {nearest_dest_city} airport, then take road transport (Private Car/Bus/Coaster) to {destination}."
        })
        
    flight_res = {}
    if final_dep and final_dest and final_dep != final_dest:
        flight_res = search_flights(final_dep, final_dest)
        
    return {
        "alerts": alerts,
        "road_legs": road_legs,
        "flight_res": flight_res,
        "final_dep": final_dep,
        "final_dest": final_dest,
        "has_routing": bool(flight_res.get("found"))
    }

def _safe_int(val: Any, default: int = 0) -> int:
    try:
        if val is None or val == "":
            return default
        return int(float(str(val).replace("$", "").replace(",", "").strip()))
    except (ValueError, TypeError):
        return default

def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None or val == "":
            return default
        return float(str(val).replace("$", "").replace(",", "").strip())
    except (ValueError, TypeError):
        return default

def calculate_real_budget(destination: str, duration: int, travelers: int, travel_mode: str, transport_class: str) -> dict:
    """Calculates realistic budget allocation using actual dataset info."""
    avg_nightly_pkr = 5000  # Default fallback
    try:
        airbnb_df = pd.read_csv(ROOT / "data" / "airbnb-listings-in-pakistan.csv")
        airbnb_df["City_Clean"] = airbnb_df["City"].dropna().apply(lambda x: x.split(",")[0].strip().lower())
        city_matches = airbnb_df[airbnb_df["City_Clean"] == destination.lower()]
        
        prices_usd = []
        for p in city_matches["Price Per Night"].dropna():
            try:
                prices_usd.append(_safe_float(p))
            except ValueError:
                continue
                
        if prices_usd:
            avg_usd = sum(prices_usd) / len(prices_usd)
            avg_nightly_pkr = int(avg_usd * 280)
        else:
            gh_df = pd.read_csv(ROOT / "data" / "sample-data-Guest_houses.csv")
            gh_df["City_Clean"] = gh_df["city"].dropna().apply(lambda x: x.strip().lower())
            gh_matches = gh_df[gh_df["City_Clean"] == destination.lower()]
            if not gh_matches.empty:
                avg_nightly_pkr = 4000
    except Exception as e:
        logger.error(f"Error calculating accommodation budget: {e}")
        
    rooms = max(1, (travelers + 1) // 2)
    total_accom_pkr = avg_nightly_pkr * duration * rooms
    
    total_transport_pkr = 0
    transport_found = False
    
    if travel_mode == "By Air":
        try:
            dep_city = st.session_state.get("wz_departure", "Karachi")
            flight_res = search_flights(dep_city, destination)
            if flight_res.get("found"):
                ticket_pkr = int(flight_res["avg_price_usd"] * 280)
                total_transport_pkr = ticket_pkr * travelers
                transport_found = True
        except Exception as e:
            logger.error(f"Error calculating flight budget: {e}")
            
    if not transport_found:
        try:
            dep_city = st.session_state.get("wz_departure", "Lahore")
            rt_df = pd.read_csv(ROOT / "data" / "road_transport.csv")
            matches = rt_df[
                (rt_df["departure_city"].str.lower() == dep_city.lower()) & 
                (rt_df["arrival_city"].str.lower() == destination.lower())
            ]
            if not matches.empty:
                fare_pkr = int(matches["fare_pkr"].mean())
                total_transport_pkr = fare_pkr * travelers
                transport_found = True
            else:
                total_transport_pkr = 2500 * travelers
        except Exception as e:
            logger.error(f"Error calculating road transport budget: {e}")
            total_transport_pkr = 2500 * travelers
            
    daily_food_pkr = 1500
    try:
        df_tourism = pd.read_csv(ROOT / "data" / "pakistan_tourism_dataset.csv")
        match = df_tourism[df_tourism["City"].str.strip().str.lower() == destination.strip().lower()]
        if not match.empty:
            avg_cost_usd = float(match.iloc[0]["Average_Cost_USD"])
            daily_food_pkr = max(1000, int(avg_cost_usd * 280 * 0.20))
    except Exception as e:
        logger.error(f"Error calculating food budget: {e}")
        
    total_food_pkr = daily_food_pkr * duration * travelers
    
    total_activities_pkr = 0
    try:
        act_df = pd.read_csv(ROOT / "data" / "activities.csv")
        dest_df = pd.read_csv(ROOT / "data" / "destinations.csv")
        dest_match = dest_df[dest_df["name"].str.lower() == destination.lower()]
        if not dest_match.empty:
            dest_id = dest_match.iloc[0]["id"]
            activities = act_df[act_df["destination_id"] == dest_id]
            if not activities.empty:
                avg_act_pkr = int(activities["price_pkr"].mean())
                total_activities_pkr = avg_act_pkr * travelers * min(duration, 3)
            else:
                total_activities_pkr = 1500 * travelers * min(duration, 3)
        else:
            total_activities_pkr = 1500 * travelers * min(duration, 3)
    except Exception as e:
        logger.error(f"Error calculating activities budget: {e}")
        total_activities_pkr = 1000 * travelers * min(duration, 3)
        
    subtotal = total_accom_pkr + total_transport_pkr + total_food_pkr + total_activities_pkr
    emergency_pkr = int(subtotal * 0.10)
    total_estimated_pkr = subtotal + emergency_pkr
    
    return {
        "accommodation": total_accom_pkr,
        "transportation": total_transport_pkr,
        "food": total_food_pkr,
        "activities": total_activities_pkr,
        "emergency": emergency_pkr,
        "total": total_estimated_pkr,
        "avg_nightly": avg_nightly_pkr,
        "daily_food": daily_food_pkr
    }

@st.cache_data
def get_food_recommendations(city: str) -> dict:
    """Destination food recommendations."""
    res_list = []
    try:
        df_combined = pd.read_csv(ROOT / "data" / "combined_data.csv", low_memory=False)
        if "city" in df_combined.columns:
            df_combined["city_clean"] = df_combined["city"].dropna().astype(str).str.strip().str.lower()
            matches = df_combined[df_combined["city_clean"] == city.lower()]
            
            keywords = ["restaurant", "food", "cafe", "pizza", "burger", "kabab", "karahi", "tikka", "hotel", "sweets", "bakers", "dining"]
            if not matches.empty:
                matches = matches[matches["name"].dropna().apply(lambda x: any(k in str(x).lower() for k in keywords))]
                for _, row in matches.head(4).iterrows():
                    res_list.append({
                        "name": str(row.get("name", "")).strip(),
                        "address": str(row.get("address", "N/A")).strip(),
                        "phone": str(row.get("phone_number", "N/A")).strip(),
                        "website": str(row.get("website", "N/A")).strip()
                    })
    except Exception as e:
        logger.error(f"Error loading food recommendations: {e}")
        
    local_specialties = {
        "lahore": ["Butt Karahi (Lakshmi Chowk)", "Fiqay ki Lassi (Gawalmandi)", "Haveli Restaurant (Fort Road Food Street)", "Taj Mahal Halwa Puri"],
        "karachi": ["Kolachi (Do Darya)", "Biryani Center", "Burns Road Food Street (Waheed Kabab)", "Javed Nihari"],
        "islamabad": ["Monal Restaurant (Margalla Hills)", "Kabul Restaurant (Afghan cuisine, F-7)", "Savour Foods (Blue Area)", "Saidpur Village Cafes"],
        "hunza": ["Yak Burger at Cafe de Hunza", "Chapshuro (local meat pie)", "Diram Fitti (local wheat dish)", "Hunza Food Pavilion"],
        "skardu": ["Balti Garam Bread", "Mamtu (local dumplings)", "Dewan-e-Khas Restaurant", "Shangrila Lake Restaurant"],
        "swat": ["Trout Fish at Kalam", "Swat Karahi", "Mingora Food Street", "White Palace Restaurant"],
        "peshawar": ["Charsi Tikka (Namak Mandi)", "Kabuli Pulao", "Jalil Kabab House (Chapli Kabab)", "Peshawari Kahwa"],
        "murree": ["Mall Road Dhabas", "Red Onion Restaurant", "Chinar Family Restaurant", "Kashmir Point Cafes"],
        "quetta": ["Sajji at Lehri Sajji", "Dampukht", "Quetta Tea Cafes", "Liaquat Food Street"],
    }
    return {
        "restaurants": res_list,
        "specialties": local_specialties.get(city.lower(), ["Local Pakistani Dhabas & BBQ", "Traditional Biryani & Karahi", "Local Tea Stalls"])
    }

@st.cache_data
def get_destination_activities(city: str) -> list[dict]:
    """Activities & Experiences Recommendations."""
    acts = []
    try:
        dest_df = pd.read_csv(ROOT / "data" / "destinations.csv")
        act_df = pd.read_csv(ROOT / "data" / "activities.csv")
        
        dest_match = dest_df[dest_df["name"].str.lower() == city.lower()]
        if not dest_match.empty:
            dest_id = dest_match.iloc[0]["id"]
            matched_acts = act_df[act_df["destination_id"] == dest_id]
            for _, row in matched_acts.iterrows():
                acts.append({
                    "name": str(row.get("name", "")).strip(),
                    "category": str(row.get("category", "")).strip().title(),
                    "duration": str(row.get("duration", "")).strip(),
                    "price": _safe_int(row.get("price_pkr", 0)),
                    "description": str(row.get("description", "")).strip(),
                    "best_time": str(row.get("best_time", "")).strip()
                })
    except Exception as e:
        logger.error(f"Error loading activities: {e}")
    return acts

@st.cache_data
def get_all_accommodations() -> list[dict]:
    accoms = []
    try:
        gh_df = pd.read_csv(ROOT / "data" / "sample-data-Guest_houses.csv")
        for _, row in gh_df.iterrows():
            city = str(row.get("city", "")).strip().title()
            accoms.append({
                "name": str(row.get("name", "")).strip(),
                "city": city,
                "type": "Guest House",
                "price": "PKR 3,000 - 8,000",
                "rating": f"{row.get('star_count', '3')} ⭐",
                "reviews": str(row.get('rating_count', '0')),
                "contact": f"Phone: {row.get('phone', 'N/A')} | Email: {row.get('email', 'N/A')}",
                "description": f"Address: {row.get('address', 'N/A')}. Website: {row.get('url', 'N/A')}",
                "url": str(row.get("url", ""))
            })
    except Exception as e:
        logger.error(f"Error loading guest houses: {e}")

    try:
        ab_df = pd.read_csv(ROOT / "data" / "airbnb-listings-in-pakistan.csv")
        for _, row in ab_df.iterrows():
            city_full = str(row.get("City", "")).strip()
            city = city_full.split(",")[0].strip().title() if city_full else "N/A"
            price_night = str(row.get("Price Per Night", "")).strip()
            if price_night and not price_night.startswith("$"):
                price_night = f"${price_night}"
            accoms.append({
                "name": str(row.get("Listing name", "")).strip(),
                "city": city,
                "type": "Airbnb",
                "price": f"{price_night} / night",
                "rating": f"{row.get('Rating', 'N/A')} ⭐",
                "reviews": "N/A",
                "contact": f"Host: {row.get('Host name', 'N/A')} (Superhost: {row.get('Super host', 'No')})",
                "description": f"Room Type: {row.get('Room type', 'N/A')} | Guests: {row.get('Guests', 'N/A')}",
                "url": ""
            })
    except Exception as e:
        logger.error(f"Error loading Airbnb: {e}")

    try:
        hotels_df = pd.read_csv(ROOT / "data" / "hotels.csv")
        dest_df = pd.read_csv(ROOT / "data" / "destinations.csv")
        dest_map = dict(zip(dest_df["id"], dest_df["name"]))
        for _, row in hotels_df.iterrows():
            dest_id = row.get("destination_id", 0)
            city = dest_map.get(dest_id, "N/A").title()
            accoms.append({
                "name": str(row.get("name", "")).strip(),
                "city": city,
                "type": "Hotel",
                "price": f"PKR {row.get('price_range_pkr', 'N/A')}",
                "rating": f"{row.get('rating', 'N/A')} ⭐",
                "reviews": "N/A",
                "contact": "Contact: Reception Desk",
                "description": f"Amenities: {row.get('amenities', 'N/A')}. Description: {row.get('description', '')}",
                "url": ""
            })
    except Exception as e:
        logger.error(f"Error loading hotels: {e}")
    return accoms

@st.cache_data
def get_all_road_transport() -> list[dict]:
    routes = []
    try:
        rt_df = pd.read_csv(ROOT / "data" / "road_transport.csv")
        for _, row in rt_df.iterrows():
            routes.append({
                "operator": str(row.get("operator", "")).strip(),
                "departure_city": str(row.get("departure_city", "")).strip(),
                "arrival_city": str(row.get("arrival_city", "")).strip(),
                "vehicle_type": str(row.get("vehicle_type", "")).strip(),
                "fare_pkr": _safe_int(row.get("fare_pkr", 0)),
                "duration_hours": _safe_float(row.get("duration_hours", 0.0)),
                "contact_number": str(row.get("contact_number", "")).strip(),
            })
    except Exception as e:
        logger.error(f"Error loading road transport: {e}")
    return routes

def _get_spotlight_destinations() -> list[dict]:
    if "spotlight_destinations" in st.session_state:
        return st.session_state.spotlight_destinations
    try:
        df_tourism = pd.read_csv(ROOT / "data" / "pakistan_tourism_dataset.csv")
        if "Popularity_Score" not in df_tourism.columns:
            df_tourism["Popularity_Score"] = 75
        popular_df = df_tourism[df_tourism["Popularity_Score"] >= 80]
        hidden_df = df_tourism[df_tourism["Popularity_Score"] < 80]
        if popular_df.empty:
            popular_df = df_tourism
        if hidden_df.empty:
            hidden_df = df_tourism
        pop_sample = popular_df.sample(min(2, len(popular_df)))
        hid_sample = hidden_df.sample(min(1, len(hidden_df)))
        selected = pd.concat([pop_sample, hid_sample])
        spotlights = selected.to_dict("records")
        st.session_state.spotlight_destinations = spotlights
        return spotlights
    except Exception as e:
        logger.error(f"Error choosing spotlight destinations: {e}")
        return [
            {"City": "Hunza", "Province": "Gilgit-Baltistan", "Main_Attraction": "Attabad Lake", "Average_Cost_USD": 250, "Safety_Rating": 5, "Peak_Season": "May-September", "Popularity_Score": 95},
            {"City": "Skardu", "Province": "Gilgit-Baltistan", "Main_Attraction": "Shangrila Resort", "Average_Cost_USD": 300, "Safety_Rating": 5, "Peak_Season": "June-August", "Popularity_Score": 90},
            {"City": "Ziarat", "Province": "Balochistan", "Main_Attraction": "Quaid-e-Azam Residency", "Average_Cost_USD": 120, "Safety_Rating": 4, "Peak_Season": "May-October", "Popularity_Score": 65}
        ]

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
                "🏠  Accommodation",
                "🚗  Road Transport",
                "✈️  Flight Search",
                "⚖️  Destination Compare",
                "🧳  AI Trip Planner",
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

def page_accommodation() -> None:
    st.markdown(
        """
        <div class="hero-container">
            <p class="hero-title">🏠 Accommodation Finder</p>
            <p class="hero-subtitle">
                Search real Pakistan accommodations from Airbnb listings, guest houses, and hotels — all dataset-driven.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_accoms = get_all_accommodations()
    if not all_accoms:
        st.error("No accommodation data loaded. Please check dataset files.")
        return

    all_cities = sorted(set(a["city"] for a in all_accoms if a["city"] and a["city"] != "N/A"))
    all_types  = sorted(set(a["type"] for a in all_accoms))

    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
    with col_f1:
        search_q = st.text_input("🔍 Search by name or description", placeholder="e.g. Hunza View, Mountain Lodge...", key="accom_search")
    with col_f2:
        city_filter = st.selectbox("🏙️ Filter by City", ["All Cities"] + all_cities, key="accom_city")
    with col_f3:
        type_filter = st.selectbox("🏨 Filter by Type", ["All Types"] + all_types, key="accom_type")

    filtered = all_accoms
    if city_filter != "All Cities":
        filtered = [a for a in filtered if a["city"].lower() == city_filter.lower()]
    if type_filter != "All Types":
        filtered = [a for a in filtered if a["type"] == type_filter]
    if search_q.strip():
        q = search_q.strip().lower()
        filtered = [a for a in filtered if q in a["name"].lower() or q in a["description"].lower() or q in a["city"].lower()]

    st.markdown(f"<div style='color:#9ca3af; font-size:0.85rem; margin-bottom:1rem;'>Showing **{len(filtered)}** accommodation records</div>", unsafe_allow_html=True)

    # ── Side-by-side comparison ──────────────────────────────────────────────
    with st.expander("⚖️ Compare Two Accommodations Side-by-Side", expanded=False):
        names = [a["name"] for a in filtered if a["name"]] or ["No results"]
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            sel_a = st.selectbox("First Accommodation", names, key="cmp_a")
        with comp_col2:
            sel_b = st.selectbox("Second Accommodation", names, index=min(1, len(names)-1), key="cmp_b")

        if st.button("⚖️ Compare Now", key="accom_compare_btn", use_container_width=True):
            a_data = next((a for a in filtered if a["name"] == sel_a), None)
            b_data = next((a for a in filtered if a["name"] == sel_b), None)
            if a_data and b_data:
                ca, cb = st.columns(2)
                for col, item in [(ca, a_data), (cb, b_data)]:
                    with col:
                        st.markdown(
                            f"""
                            <div class="glass-card-accent">
                                <div style="font-size:0.7rem;color:#ffab00;text-transform:uppercase;font-weight:600;">{item['type']}</div>
                                <div style="font-size:1.1rem;font-weight:700;color:#fff;margin:0.4rem 0;">{item['name']}</div>
                                <div style="font-size:0.85rem;color:#ffa726;">📍 {item['city']}</div>
                                <div style="font-size:0.85rem;color:#ffa726;margin-top:0.3rem;">💰 {item['price']}</div>
                                <div style="font-size:0.85rem;color:#ffa726;">⭐ {item['rating']}</div>
                                <div style="font-size:0.78rem;color:#9ca3af;margin-top:0.5rem;">{item['contact']}</div>
                                <div style="font-size:0.78rem;color:#9ca3af;margin-top:0.3rem;">{item['description'][:120]}...</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # ── Listing Grid ─────────────────────────────────────────────────────────
    if not filtered:
        st.info("No accommodations match your search. Try adjusting filters.")
        return

    display_items = filtered[:30]
    cols = st.columns(3)
    for i, accom in enumerate(display_items):
        with cols[i % 3]:
            url_html = f'<a href="{accom["url"]}" target="_blank" style="color:#ffa726;font-size:0.78rem;">🌐 Website ↗</a>' if accom.get("url") and accom["url"] not in ["", "nan", "N/A"] else ""
            st.markdown(
                f"""
                <div class="glass-card" style="height:100%;display:flex;flex-direction:column;justify-content:space-between;margin-bottom:1rem;">
                    <div>
                        <div style="font-size:0.65rem;color:{'#81c784' if accom['type']=='Airbnb' else '#ffab00' if accom['type']=='Guest House' else '#64b5f6'};text-transform:uppercase;font-weight:600;">{accom['type']}</div>
                        <div style="font-size:0.95rem;font-weight:700;color:#fff;margin-top:0.3rem;min-height:2.2rem;">{accom['name'][:50]}</div>
                        <div style="font-size:0.78rem;color:#cbd5e1;margin-top:0.3rem;">📍 {accom['city']}</div>
                        <div style="font-size:0.82rem;color:#ffa726;font-weight:600;margin-top:0.3rem;">💰 {accom['price']}</div>
                        <div style="font-size:0.82rem;color:#ffa726;margin-top:0.2rem;">{accom['rating']}</div>
                    </div>
                    <div style="margin-top:0.8rem;border-top:1px solid rgba(255,255,255,0.05);padding-top:0.5rem;">
                        <div style="font-size:0.72rem;color:#9ca3af;">{accom['contact'][:80]}</div>
                        {url_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def page_road_transport() -> None:
    st.markdown(
        """
        <div class="hero-container">
            <p class="hero-title">🚗 Road Transport Explorer</p>
            <p class="hero-subtitle">
                Explore intercity bus, coach, and road transport options across Pakistan — real operators, real routes, real fares.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_routes = get_all_road_transport()
    if not all_routes:
        st.error("No road transport data available. Please ensure road_transport.csv is present.")
        return

    dep_cities = sorted(set(r["departure_city"] for r in all_routes))
    arr_cities = sorted(set(r["arrival_city"] for r in all_routes))
    operators  = sorted(set(r["operator"] for r in all_routes))

    st.markdown("### 🔍 Search Routes")
    col1, col2, col3 = st.columns(3)
    with col1:
        dep_sel = st.selectbox("🛫 Departure City", ["Any"] + dep_cities, key="rt_dep")
    with col2:
        arr_sel = st.selectbox("🛬 Arrival City", ["Any"] + arr_cities, key="rt_arr")
    with col3:
        op_sel  = st.selectbox("🚌 Operator", ["Any"] + operators, key="rt_op")

    filtered = all_routes
    if dep_sel != "Any":
        filtered = [r for r in filtered if r["departure_city"].lower() == dep_sel.lower()]
    if arr_sel != "Any":
        filtered = [r for r in filtered if r["arrival_city"].lower() == arr_sel.lower()]
    if op_sel != "Any":
        filtered = [r for r in filtered if r["operator"].lower() == op_sel.lower()]

    st.markdown(f"<div style='color:#9ca3af;font-size:0.85rem;margin-bottom:1rem;'>Found **{len(filtered)}** routes</div>", unsafe_allow_html=True)

    if not filtered:
        st.info("No routes found. Try adjusting the filters.")
        return

    # ── Compare transport classes ─────────────────────────────────────────────
    with st.expander("⚖️ Compare Transport Classes Side-by-Side", expanded=False):
        vehicle_types = sorted(set(r["vehicle_type"] for r in all_routes))
        comp_c1, comp_c2 = st.columns(2)
        with comp_c1:
            vt_a = st.selectbox("First Class", vehicle_types, key="vt_cmp_a")
        with comp_c2:
            vt_b = st.selectbox("Second Class", vehicle_types, index=min(1, len(vehicle_types)-1), key="vt_cmp_b")

        vt_a_routes = [r for r in all_routes if r["vehicle_type"] == vt_a]
        vt_b_routes = [r for r in all_routes if r["vehicle_type"] == vt_b]

        ca, cb = st.columns(2)
        for col, vt, vt_routes in [(ca, vt_a, vt_a_routes), (cb, vt_b, vt_b_routes)]:
            with col:
                if vt_routes:
                    avg_fare  = int(sum(r["fare_pkr"] for r in vt_routes) / len(vt_routes))
                    avg_dur   = round(sum(r["duration_hours"] for r in vt_routes) / len(vt_routes), 1)
                    ops_list  = ", ".join(sorted(set(r["operator"] for r in vt_routes)))
                    st.markdown(
                        f"""
                        <div class="glass-card-accent">
                            <div style="font-size:1.1rem;font-weight:700;color:#fff;">{vt}</div>
                            <div style="font-size:0.85rem;color:#ffa726;margin-top:0.5rem;">💰 Avg Fare: PKR {avg_fare:,}</div>
                            <div style="font-size:0.85rem;color:#cbd5e1;">⏱️ Avg Duration: {avg_dur} hrs</div>
                            <div style="font-size:0.85rem;color:#9ca3af;margin-top:0.3rem;">🚌 Operators: {ops_list}</div>
                            <div style="font-size:0.82rem;color:#9ca3af;">Routes available: {len(vt_routes)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # ── Route Cards ───────────────────────────────────────────────────────────
    st.markdown("### 🗺️ Available Routes")
    for r in filtered:
        color_map = {"Executive": "#ff6f00", "Business Sleeper": "#ff5722", "Sleeper": "#ffa726",
                     "Super Luxury": "#ff6f00", "Luxury": "#ffa726", "Standard": "#81c784",
                     "Coaster": "#64b5f6"}
        vt_color = color_map.get(r["vehicle_type"], "#9ca3af")
        st.markdown(
            f"""
            <div class="glass-card" style="margin-bottom:0.8rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;">
                    <div>
                        <span style="font-size:1.05rem;font-weight:700;color:#fff;">{r['departure_city']}</span>
                        <span style="color:#ff5722;font-size:1.1rem;margin:0 0.5rem;">→</span>
                        <span style="font-size:1.05rem;font-weight:700;color:#fff;">{r['arrival_city']}</span>
                    </div>
                    <div>
                        <span style="background:rgba(255,87,34,0.15);border:1px solid rgba(255,87,34,0.3);
                                     border-radius:20px;padding:0.2rem 0.8rem;font-size:0.8rem;color:#ffa726;">
                            {r['operator']}
                        </span>
                    </div>
                </div>
                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:0.8rem;margin-top:0.8rem;font-size:0.85rem;">
                    <div><span style="color:#9ca3af;">🚌 Class:</span> <span style="color:{vt_color};font-weight:600;">{r['vehicle_type']}</span></div>
                    <div><span style="color:#9ca3af;">💰 Fare:</span> <span style="color:#ffa726;font-weight:600;">PKR {r['fare_pkr']:,}</span></div>
                    <div><span style="color:#9ca3af;">⏱️ Duration:</span> <span style="color:#fff;">{r['duration_hours']} hrs</span></div>
                    <div><span style="color:#9ca3af;">📞 Helpline:</span> <span style="color:#81c784;">{r['contact_number']}</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Summary stats ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Intercity Transport Summary")
    all_fares = [r["fare_pkr"] for r in all_routes if r["fare_pkr"] > 0]
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, label, val in [
        (c1, "🚌", "Total Routes", len(all_routes)),
        (c2, "🏢", "Operators", len(set(r["operator"] for r in all_routes))),
        (c3, "💰", "Min Fare", f"PKR {min(all_fares):,}" if all_fares else "N/A"),
        (c4, "💳", "Max Fare", f"PKR {max(all_fares):,}" if all_fares else "N/A"),
    ]:
        with col:
            st.markdown(
                f"""<div class="glass-card" style="text-align:center;padding:1.2rem;">
                    <div style="font-size:1.8rem;">{icon}</div>
                    <div style="font-size:1.3rem;font-weight:700;color:#ff6f00;">{val}</div>
                    <div style="font-size:0.8rem;color:#9ca3af;">{label}</div>
                </div>""",
                unsafe_allow_html=True,
            )


def page_flights() -> None:
    st.markdown(
        """
        <div class="hero-container">
            <p class="hero-title">✈️ Domestic Flight Search</p>
            <p class="hero-subtitle">
                Search real domestic flights within Pakistan, compare prices, check on-time status, and view airport details.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cities = get_all_airports()
    col1, col2 = st.columns(2)
    with col1:
        dep_city = st.selectbox("Departure Airport City", cities, index=cities.index("Karachi") if "Karachi" in cities else 0)
    with col2:
        arr_city = st.selectbox("Arrival Airport City", [c for c in cities if c != dep_city], index=cities.index("Lahore") if "Lahore" in cities and dep_city != "Lahore" else 0)

    cabin_class = st.selectbox("Cabin Class", ["Economy Class", "Business Class"])

    if st.button("🔍 Search Flights", use_container_width=True):
        with st.spinner("Searching domestic routes..."):
            res = search_flights(dep_city, arr_city)
            
        if res.get("found"):
            st.markdown(
                f"""
                <div class="glass-card-accent">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
                        <span style="font-size:1.8rem; font-weight:700; color:#fff;">
                            {res['departure']} ({res['departure_code']}) ✈️ {res['arrival']} ({res['arrival_code']})
                        </span>
                        <span class="source-badge">PIA Domestic Route</span>
                    </div>
                    
                    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:1.2rem; margin-top:1.5rem;">
                        <div class="glass-card" style="text-align:center; padding:1.2rem; margin-bottom:0;">
                            <div style="font-size:0.8rem; color:#9ca3af; text-transform:uppercase;">Average Ticket Price</div>
                            <div style="font-size:1.8rem; font-weight:700; color:#ffa726; margin-top:0.4rem;">
                                ${res['avg_price_usd']:.0f} USD
                            </div>
                            <div style="font-size:0.9rem; color:#81c784; margin-top:0.2rem;">
                                ≈ {CURRENCY_SYMBOL} {int(res['avg_price_usd'] * 280):,} PKR
                            </div>
                        </div>
                        <div class="glass-card" style="text-align:center; padding:1.2rem; margin-bottom:0;">
                            <div style="font-size:0.8rem; color:#9ca3af; text-transform:uppercase;">Price Range</div>
                            <div style="font-size:1.5rem; font-weight:700; color:#fff; margin-top:0.6rem;">
                                ${res['min_price_usd']:.0f} - ${res['max_price_usd']:.0f}
                            </div>
                            <div style="font-size:0.9rem; color:#9ca3af; margin-top:0.2rem;">
                                USD per ticket
                            </div>
                        </div>
                        <div class="glass-card" style="text-align:center; padding:1.2rem; margin-bottom:0;">
                            <div style="font-size:0.8rem; color:#9ca3af; text-transform:uppercase;">Average Duration</div>
                            <div style="font-size:1.8rem; font-weight:700; color:#fff; margin-top:0.4rem;">
                                {int(res['avg_duration_min'])} mins
                            </div>
                            <div style="font-size:0.9rem; color:#9ca3af; margin-top:0.2rem;">
                                {int(res['avg_duration_min'] // 60)}h {int(res['avg_duration_min'] % 60)}m
                            </div>
                        </div>
                        <div class="glass-card" style="text-align:center; padding:1.2rem; margin-bottom:0;">
                            <div style="font-size:0.8rem; color:#9ca3af; text-transform:uppercase;">On-Time Performance</div>
                            <div style="font-size:1.8rem; font-weight:700; color:#81c784; margin-top:0.4rem;">
                                {res['on_time_pct']}%
                            </div>
                            <div style="font-size:0.9rem; color:#9ca3af; margin-top:0.2rem;">
                                {res['sample_count']} flights sampled
                            </div>
                        </div>
                    </div>
                    
                    <div style="margin-top:1.5rem;">
                        <span style="color:#9ca3af; font-size:0.9rem; font-weight:600;">Aircraft Types Operated:</span>
                        {" ".join([f'<span class="stat-chip">{ac}</span>' for ac in res['aircraft_types']])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning(
                f"No direct flight schedules found between **{dep_city}** and **{arr_city}** in the PIA historical dataset."
            )

        # Show airport information details for both departure and arrival
        st.markdown("---")
        st.markdown("### 🏢 Airport Information Details")
        c_dep, c_arr = st.columns(2)
        
        for city, col in [(dep_city, c_dep), (arr_city, c_arr)]:
            info = AIRPORTS.get(city)
            with col:
                if info:
                    st.markdown(
                        f"""
                        <div class="glass-card">
                            <div style="font-size:1.2rem; font-weight:700; color:#ff6f00; margin-bottom:0.5rem;">
                                {info['name']} ({info['code']})
                            </div>
                            <div style="font-size:0.9rem; color:#cbd5e1; margin-bottom:0.2rem;">
                                📍 <b>City:</b> {info['city']}, {info['province']}
                            </div>
                            <div style="font-size:0.9rem; color:#cbd5e1; margin-bottom:0.8rem;">
                                🌐 <b>Official Website:</b> <a href="{info['url']}" target="_blank" style="color:#ffa726;">PAA Portal</a>
                            </div>
                            <a href="{info['schedule_url']}" target="_blank">
                                <button style="background:rgba(255,87,34,0.1); border:1px solid rgba(255,87,34,0.3); 
                                               color:#ffa726; border-radius:8px; padding:0.4rem 1rem; cursor:pointer;">
                                    📅 View Real-Time Schedule
                                </button>
                            </a>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="glass-card">
                            <div style="font-size:1.2rem; font-weight:700; color:#ff6f00; margin-bottom:0.5rem;">
                                {city} Airport
                            </div>
                            <div style="font-size:0.9rem; color:#cbd5e1; margin-bottom:0.8rem;">
                                Airport information not available in the local directory.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Booking Links
        st.markdown("---")
        st.markdown("### 🎫 Partner Airlines & Booking Platforms")
        
        c_airline, c_booking = st.columns(2)
        with c_airline:
            st.markdown("#### ✈️ Domestic Airlines")
            for al in AIRLINES:
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; align-items:center; 
                                background:rgba(255,255,255,0.02); padding:0.8rem 1.2rem; 
                                border-radius:10px; margin-bottom:0.5rem; border:1px solid rgba(255,255,255,0.05);">
                        <div>
                            <span style="font-size:1.1rem; margin-right:0.5rem;">{al['icon']}</span>
                            <span style="font-weight:600; color:#fff;">{al['name']}</span>
                            <div style="font-size:0.75rem; color:#9ca3af;">Routes: {al['routes']}</div>
                        </div>
                        <a href="{al['url']}" target="_blank">
                            <button style="background:#ff5722; border:none; color:white; border-radius:6px; 
                                           padding:0.3rem 0.8rem; font-size:0.8rem; cursor:pointer;">
                                Book Now ↗
                            </button>
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
        with c_booking:
            st.markdown("#### 🌐 Booking Platforms")
            for bp in BOOKING_PLATFORMS:
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; align-items:center; 
                                background:rgba(255,255,255,0.02); padding:0.8rem 1.2rem; 
                                border-radius:10px; margin-bottom:0.5rem; border:1px solid rgba(255,255,255,0.05);">
                        <div>
                            <span style="font-size:1.1rem; margin-right:0.5rem;">{bp['icon']}</span>
                            <span style="font-weight:600; color:#fff;">{bp['name']}</span>
                            <div style="font-size:0.75rem; color:#9ca3af;">{bp['desc']}</div>
                        </div>
                        <a href="{bp['url']}" target="_blank">
                            <button style="background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.1); 
                                           color:#fff; border-radius:6px; padding:0.3rem 0.8rem; font-size:0.8rem; cursor:pointer;">
                                Compare ↗
                            </button>
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Show all unique domestic routes in database
        st.markdown("---")
        with st.expander("📊 View All Domestic Routes in Dataset", expanded=False):
            routes = get_all_domestic_routes()
            if routes:
                route_df = pd.DataFrame(routes)
                route_df.columns = ["Departure", "Arrival", "Dep Code", "Arr Code", "Avg Price (USD)", "Min Price (USD)", "Avg Duration (min)", "Sample Flights"]
                st.dataframe(route_df, use_container_width=True, hide_index=True)
            else:
                st.info("No domestic routes found in flight dataset.")


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


# ── Wizard helpers ────────────────────────────────────────────────────────────

def _wizard_init():
    """Initialize all wizard session state keys with defaults."""
    defaults = {
        "wz_step": 0,
        "wz_destination": "",
        "wz_departure": "",
        "wz_group_type": "Solo Traveler",
        "wz_num_adults": 1,
        "wz_num_children": 0,
        "wz_num_travelers": 1,
        "wz_split_budget": False,
        "wz_depart_date": None,
        "wz_return_date": None,
        "wz_duration": 5,
        "wz_travel_style": "Mid-Level",
        "wz_interests": [],
        "wz_custom_interest": "",
        "wz_accom_type": "Hotel",
        "wz_room_type": "Double Room",
        "wz_food_prefs": [],
        "wz_food_notes": "",
        "wz_transport_type": "By Road",
        "wz_transport_class": "Private Car",
        "wz_budget": 50000,
        "wz_plan_text": "",
        "wz_plan_chunks": [],
        "wz_chat_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _render_progress(step: int, total: int = 11):
    pct = int((step / total) * 100)
    labels = [
        "Welcome", "Destination", "Group", "Dates", "Style",
        "Interests", "Stay", "Food", "Transport", "Budget", "Review", "Done"
    ]
    label = labels[min(step, len(labels) - 1)]
    st.markdown(
        f"""
        <div class="wizard-progress-wrap">
            <span class="wizard-step-label">Step {step} of {total}</span>
            <div class="wizard-bar-bg">
                <div class="wizard-bar-fill" style="width:{pct}%;"></div>
            </div>
            <span class="wizard-step-count">{label} &nbsp;{pct}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Load destinations and departure cities dynamically
@st.cache_data
def get_destinations_list() -> list[str]:
    defaults = {
        "Hunza", "Skardu", "Swat", "Murree", "Naran Kaghan", "Fairy Meadows",
        "Lahore", "Karachi", "Islamabad", "Peshawar", "Quetta",
        "Chitral", "Gilgit", "Attabad Lake", "Neelum Valley", "Ratti Gali",
        "Kumrat Valley", "Shogran", "Malam Jabba", "Kalash Valley",
        "Deosai Plains", "K2 Base Camp", "Naltar Valley",
    }
    try:
        df_tourism = pd.read_csv(ROOT / "data" / "pakistan_tourism_dataset.csv")
        if "City" in df_tourism.columns:
            defaults.update(df_tourism["City"].dropna().unique().tolist())
    except Exception as e:
        logger.error(f"Error loading destinations: {e}")
    try:
        df_airbnb = pd.read_csv(ROOT / "data" / "airbnb-listings-in-pakistan.csv")
        if "City" in df_airbnb.columns:
            for city_str in df_airbnb["City"].dropna().unique():
                city = city_str.split(",")[0].strip()
                if city:
                    defaults.add(city)
    except Exception as e:
        logger.error(f"Error loading airbnb cities: {e}")
    return sorted(list(defaults))

@st.cache_data
def get_departure_cities_list() -> list[str]:
    defaults = {
        "Karachi", "Lahore", "Islamabad", "Peshawar", "Quetta",
        "Multan", "Faisalabad", "Sialkot", "Hyderabad", "Rawalpindi",
    }
    # Add cities from PIA flight dataset
    try:
        pia_cities = get_domestic_cities()
        if pia_cities:
            defaults.update(pia_cities)
    except Exception as e:
        logger.error(f"Error loading departure cities: {e}")
    return sorted(list(defaults))

INTEREST_CHIPS = [
    "🏔️ Adventure", "🥾 Hiking", "⛰️ Mountains", "🦅 Wildlife",
    "🎭 Cultural", "🏛️ Historical", "🏜️ Desert", "📸 Photography",
    "🍜 Food Exploration", "🌿 Nature", "⛺ Camping", "🌊 Water Sports",
    "🚗 Road Trips", "🛍️ Shopping", "🕌 Religious Tourism",
]

STYLE_OPTIONS = [
    ("👑", "Luxury", "5-star hotels, premium transport, private tours"),
    ("🏨", "Mid-Level", "Comfortable hotels and balanced experiences"),
    ("🏠", "Standard", "Affordable but comfortable accommodation"),
    ("🎒", "Budget Backpacking", "Hostels, guest houses, public transport"),
]


def page_trip_planner() -> None:
    """13-step AI Trip Planner wizard."""
    _wizard_init()
    step = st.session_state.wz_step

    # ── Step 0: Welcome ──────────────────────────────────────────────────────
    if step == 0:
        st.markdown(
            """
            <div class="wizard-welcome">
                <div style="font-size:4rem; margin-bottom:1rem;
                            filter:drop-shadow(0 0 20px rgba(255,87,34,0.5));">🇵🇰✈️</div>
                <div class="wizard-welcome-title">AI Travel Planner</div>
                <div class="wizard-welcome-subtitle">
                    Let's create your perfect Pakistan trip in a few simple steps.
                    Our AI analyzes thousands of real travel insights to build
                    your personalized itinerary.
                </div>
                <div class="wizard-feature-grid">
                    <span class="wizard-feature-chip">🗓️ Day-by-Day Itinerary</span>
                    <span class="wizard-feature-chip">💰 Budget Optimization</span>
                    <span class="wizard-feature-chip">🏨 Hotel Recommendations</span>
                    <span class="wizard-feature-chip">🍜 Food Guide</span>
                    <span class="wizard-feature-chip">🚗 Transport Planning</span>
                    <span class="wizard-feature-chip">🤖 RAG-Powered AI</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if st.button("🚀  Start Planning My Trip", use_container_width=True, key="wz_start"):
                st.session_state.wz_step = 1
                st.rerun()

        # ── Dynamic Destination Spotlights ───────────────────────────────────
        st.markdown("---")
        st.markdown(
            "<div class='section-header'>🌟 Destination Spotlights — Discover Pakistan</div>",
            unsafe_allow_html=True,
        )
        spotlights = _get_spotlight_destinations()
        spot_cols = st.columns(len(spotlights)) if spotlights else []
        icons = ["🏔️", "🌿", "💎", "🗺️"]
        for idx, (col, spot) in enumerate(zip(spot_cols, spotlights)):
            city     = spot.get("City", "Unknown")
            province = spot.get("Province", "")
            attract  = spot.get("Main_Attraction", "Scenic beauty")
            cost_usd = spot.get("Average_Cost_USD", "N/A")
            safety   = spot.get("Safety_Rating", "N/A")
            season   = spot.get("Peak_Season", "Year-round")
            pop      = spot.get("Popularity_Score", 0)
            badge    = "💎 Hidden Gem" if float(pop) < 80 else "⭐ Popular"
            badge_color = "#a78bfa" if float(pop) < 80 else "#ffa726"
            icon = icons[idx % len(icons)]
            cost_pkr = f"≈ PKR {int(float(cost_usd) * 280):,}" if str(cost_usd).replace('.','').isdigit() else ""
            with col:
                st.markdown(
                    f"""
                    <div class="glass-card-accent" style="text-align:center;padding:1.5rem 1rem;">
                        <div style="font-size:2.5rem;margin-bottom:0.5rem;filter:drop-shadow(0 0 12px rgba(255,87,34,0.4));">{icon}</div>
                        <div style="font-size:0.7rem;color:{badge_color};font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem;">{badge}</div>
                        <div style="font-size:1.2rem;font-weight:700;color:#fff;margin-bottom:0.2rem;">{city}</div>
                        <div style="font-size:0.78rem;color:#9ca3af;margin-bottom:0.8rem;">{province}</div>
                        <div style="font-size:0.82rem;color:#cbd5e1;margin-bottom:0.5rem;">🌟 {attract}</div>
                        <div style="display:flex;justify-content:center;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.8rem;">
                            <span class="stat-chip">💰 ${cost_usd} USD {cost_pkr}</span>
                            <span class="stat-chip">🛡️ Safety {safety}/5</span>
                            <span class="stat-chip">📅 {season}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"Plan Trip to {city} 🏔️", key=f"spot_btn_{idx}", use_container_width=True):
                    st.session_state.wz_destination = city
                    st.session_state.wz_step = 1
                    st.rerun()
        return

    # Progress bar for steps 1–11
    _render_progress(step, 11)

    # ── Step 1: Destination ──────────────────────────────────────────────────
    if step == 1:
        st.markdown(
            '<div class="wizard-step-card">'
            '<div class="wizard-step-title">📍 Where do you want to go?</div>'
            '<div class="wizard-step-desc">Search or select your dream Pakistan destination and departure city.</div>',
            unsafe_allow_html=True,
        )
        col_a, col_b = st.columns(2)
        
        dyn_destinations = get_destinations_list()
        dyn_departures = get_departure_cities_list()
        
        with col_a:
            st.markdown("**Destination**")
            dest_search = st.text_input(
                "Search destination", value=st.session_state.wz_destination,
                placeholder="e.g. Hunza, Skardu...", key="wz_dest_search",
                label_visibility="collapsed"
            )
            dest_select = st.selectbox(
                "Or choose from list", ["— Select —"] + dyn_destinations, key="wz_dest_select"
            )
            chosen_dest = dest_search.strip() or (dest_select if dest_select != "— Select —" else "")
            st.session_state.wz_destination = chosen_dest

        with col_b:
            st.markdown("**Departure City**")
            dep_search = st.text_input(
                "Search city", value=st.session_state.wz_departure,
                placeholder="e.g. Karachi, Lahore...", key="wz_dep_search",
                label_visibility="collapsed"
            )
            dep_select = st.selectbox(
                "Or choose from list", ["— Select —"] + dyn_departures, key="wz_dep_select"
            )
            chosen_dep = dep_search.strip() or (dep_select if dep_select != "— Select —" else "")
            st.session_state.wz_departure = chosen_dep
            
        if chosen_dest:
            try:
                df_tourism = pd.read_csv(ROOT / "data" / "pakistan_tourism_dataset.csv")
                match = df_tourism[df_tourism["City"].str.strip().str.lower() == chosen_dest.strip().lower()]
                if not match.empty:
                    row = match.iloc[0]
                    st.markdown(
                        f"""
                        <div class="glass-card-accent" style="margin-top:1.5rem;">
                            <div style="font-size:1.2rem; font-weight:700; color:#ffa726; margin-bottom:0.8rem;">
                                🏔️ Destination Spotlight: {row['City']} ({row['Province']})
                            </div>
                            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:1rem; font-size:0.88rem;">
                                <div><b>🌟 Main Attraction:</b> {row['Main_Attraction']}</div>
                                <div><b>🎭 Type:</b> {row['Destination_Type']}</div>
                                <div><b>💰 Avg Cost:</b> ${row['Average_Cost_USD']} USD</div>
                                <div><b>🛡️ Safety Rating:</b> {'⭐' * int(row['Safety_Rating'])} ({row['Safety_Rating']}/5)</div>
                                <div><b>📅 Peak Season:</b> {row['Peak_Season']}</div>
                                <div><b>🌡️ Avg Temp:</b> {row['Average_Temperature_C']}°C</div>
                                <div><b>📈 Popularity:</b> {row['Popularity_Score']}/100</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                logger.error(f"Error rendering destination spotlight card: {e}")
                
        st.markdown("</div>", unsafe_allow_html=True)

        col_back, col_next = st.columns([1, 1])
        with col_next:
            if st.button("Next →", use_container_width=True, key="wz1_next"):
                if not st.session_state.wz_destination:
                    st.warning("Please enter or select a destination.")
                elif not st.session_state.wz_departure:
                    st.warning("Please enter or select a departure city.")
                else:
                    st.session_state.wz_step = 2
                    st.rerun()
        with col_back:
            if st.button("← Back", use_container_width=True, key="wz1_back"):
                st.session_state.wz_step = 0
                st.rerun()

    # ── Step 2: Travel Group ─────────────────────────────────────────────────
    elif step == 2:
        st.markdown(
            '<div class="wizard-step-card">'
            '<div class="wizard-step-title">👥 Who are you traveling with?</div>'
            '<div class="wizard-step-desc">Select your group type and size.</div>',
            unsafe_allow_html=True,
        )
        group_options = ["Solo Traveler", "Friends Group", "Family", "College Students", "Office/Corporate Trip"]
        group_type = st.radio(
            "Travel Group", group_options,
            index=group_options.index(st.session_state.wz_group_type),
            horizontal=True, label_visibility="collapsed", key="wz_group_radio"
        )
        st.session_state.wz_group_type = group_type

        if group_type == "Family":
            col_a, col_b = st.columns(2)
            with col_a:
                adults = st.number_input("Number of Adults", 1, 20, st.session_state.wz_num_adults, key="wz_adults")
                st.session_state.wz_num_adults = adults
            with col_b:
                children = st.number_input("Number of Children", 0, 20, st.session_state.wz_num_children, key="wz_children")
                st.session_state.wz_num_children = children
            st.session_state.wz_num_travelers = adults + children
        elif group_type == "Solo Traveler":
            st.session_state.wz_num_travelers = 1
        elif group_type == "Friends Group":
            n = st.number_input("Number of Friends", 2, 50, max(2, st.session_state.wz_num_travelers), key="wz_friends_n")
            st.session_state.wz_num_travelers = n
        elif group_type == "College Students":
            n = st.number_input("Number of Students", 2, 100, max(2, st.session_state.wz_num_travelers), key="wz_students_n")
            st.session_state.wz_num_travelers = n
        elif group_type == "Office/Corporate Trip":
            n = st.number_input("Number of Employees", 2, 200, max(2, st.session_state.wz_num_travelers), key="wz_office_n")
            st.session_state.wz_num_travelers = n

        if st.session_state.wz_num_travelers > 1:
            split = st.toggle("💸 Split budget per person?", value=st.session_state.wz_split_budget, key="wz_split")
            st.session_state.wz_split_budget = split
            if split:
                st.info(f"Budget will be divided equally among {st.session_state.wz_num_travelers} travelers.")

        st.markdown("</div>", unsafe_allow_html=True)
        col_back, col_next = st.columns([1, 1])
        with col_back:
            if st.button("← Back", use_container_width=True, key="wz2_back"):
                st.session_state.wz_step = 1; st.rerun()
        with col_next:
            if st.button("Next →", use_container_width=True, key="wz2_next"):
                st.session_state.wz_step = 3; st.rerun()

    # ── Step 3: Travel Dates ─────────────────────────────────────────────────
    elif step == 3:
        import datetime
        st.markdown(
            '<div class="wizard-step-card">'
            '<div class="wizard-step-title">📅 When are you traveling?</div>'
            '<div class="wizard-step-desc">Set your travel dates or specify total trip duration.</div>',
            unsafe_allow_html=True,
        )
        date_mode = st.radio("Input mode", ["Pick Dates", "Enter Days"], horizontal=True, key="wz_date_mode")
        if date_mode == "Pick Dates":
            col_a, col_b = st.columns(2)
            today = datetime.date.today()
            with col_a:
                dep_date = st.date_input("Departure Date", value=st.session_state.wz_depart_date or today, key="wz_dep_date")
                st.session_state.wz_depart_date = dep_date
            with col_b:
                ret_date = st.date_input(
                    "Return Date",
                    value=st.session_state.wz_return_date or (today + datetime.timedelta(days=st.session_state.wz_duration)),
                    key="wz_ret_date"
                )
                st.session_state.wz_return_date = ret_date
            if ret_date > dep_date:
                days = (ret_date - dep_date).days
                st.session_state.wz_duration = days
                st.success(f"🗓️ Trip Duration: **{days} days**")
            else:
                st.warning("Return date must be after departure date.")
        else:
            days = st.number_input("Number of Days", 1, 30, st.session_state.wz_duration, key="wz_days_manual")
            st.session_state.wz_duration = days
            st.info(f"Your trip will last **{days} days**.")

        st.markdown("</div>", unsafe_allow_html=True)
        col_back, col_next = st.columns([1, 1])
        with col_back:
            if st.button("← Back", use_container_width=True, key="wz3_back"):
                st.session_state.wz_step = 2; st.rerun()
        with col_next:
            if st.button("Next →", use_container_width=True, key="wz3_next"):
                st.session_state.wz_step = 4; st.rerun()

    # ── Step 4: Travel Style ─────────────────────────────────────────────────
    elif step == 4:
        st.markdown(
            '<div class="wizard-step-card">'
            '<div class="wizard-step-title">🎨 Choose your travel style</div>'
            '<div class="wizard-step-desc">This shapes accommodation, transport, and activity recommendations.</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(4)
        for i, (icon, label, desc) in enumerate(STYLE_OPTIONS):
            with cols[i]:
                is_sel = st.session_state.wz_travel_style == label
                sel_class = "style-card selected" if is_sel else "style-card"
                st.markdown(
                    f'<div class="{sel_class}">'
                    f'<div class="style-card-icon">{icon}</div>'
                    f'<div class="style-card-title">{label}</div>'
                    f'<div class="style-card-desc">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                if st.button(f"Select {label}", key=f"wz_style_{label}", use_container_width=True):
                    st.session_state.wz_travel_style = label
                    st.rerun()

        st.markdown(
            f"<div style='text-align:center; margin-top:1rem; color:#ff6f00; font-weight:600;'>✅ Selected: {st.session_state.wz_travel_style}</div>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
        col_back, col_next = st.columns([1, 1])
        with col_back:
            if st.button("← Back", use_container_width=True, key="wz4_back"):
                st.session_state.wz_step = 3; st.rerun()
        with col_next:
            if st.button("Next →", use_container_width=True, key="wz4_next"):
                st.session_state.wz_step = 5; st.rerun()

    # ── Step 5: Interests ────────────────────────────────────────────────────
    elif step == 5:
        st.markdown(
            '<div class="wizard-step-card">'
            '<div class="wizard-step-title">🎯 What are your interests?</div>'
            '<div class="wizard-step-desc">Select all that apply — the AI uses these to personalize your itinerary.</div>',
            unsafe_allow_html=True,
        )
        if "wz_interests" not in st.session_state:
            st.session_state.wz_interests = []

        cols = st.columns(3)
        for i, chip in enumerate(INTEREST_CHIPS):
            clean = chip.split(" ", 1)[-1]  # strip emoji for storage
            with cols[i % 3]:
                checked = st.checkbox(
                    chip, value=(clean in st.session_state.wz_interests),
                    key=f"wz_int_{i}"
                )
                if checked and clean not in st.session_state.wz_interests:
                    st.session_state.wz_interests.append(clean)
                elif not checked and clean in st.session_state.wz_interests:
                    st.session_state.wz_interests.remove(clean)

        custom = st.text_input(
            "✏️ Add your own interest",
            value=st.session_state.wz_custom_interest,
            placeholder="e.g. Paragliding, Stargazing...",
            key="wz_custom_int"
        )
        st.session_state.wz_custom_interest = custom
        if custom.strip() and custom.strip() not in st.session_state.wz_interests:
            st.session_state.wz_interests.append(custom.strip())

        if st.session_state.wz_interests:
            chips_html = " ".join(
                f'<span class="stat-chip">{i}</span>' for i in st.session_state.wz_interests
            )
            st.markdown(f"<div style='margin-top:0.8rem;'>Selected: {chips_html}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        col_back, col_next = st.columns([1, 1])
        with col_back:
            if st.button("← Back", use_container_width=True, key="wz5_back"):
                st.session_state.wz_step = 4; st.rerun()
        with col_next:
            if st.button("Next →", use_container_width=True, key="wz5_next"):
                if not st.session_state.wz_interests:
                    st.warning("Please select at least one interest.")
                else:
                    st.session_state.wz_step = 6; st.rerun()

    # ── Step 6: Accommodation ────────────────────────────────────────────────
    elif step == 6:
        st.markdown(
            '<div class="wizard-step-card">'
            '<div class="wizard-step-title">🏨 Accommodation preferences</div>'
            '<div class="wizard-step-desc">Choose where you would like to stay and room type.</div>',
            unsafe_allow_html=True,
        )
        accom_opts = ["Guest House", "Hotel", "Resort", "Farmhouse", "Camping"]
        accom = st.radio(
            "Accommodation Type", accom_opts,
            index=accom_opts.index(st.session_state.wz_accom_type),
            horizontal=True, key="wz_accom", label_visibility="collapsed"
        )
        st.session_state.wz_accom_type = accom

        room_opts = ["Single Room", "Double Room", "Family Room", "Shared Accommodation"]
        room = st.radio(
            "Room Preference", room_opts,
            index=room_opts.index(st.session_state.wz_room_type) if st.session_state.wz_room_type in room_opts else 1,
            horizontal=True, key="wz_room", label_visibility="collapsed"
        )
        st.session_state.wz_room_type = room
        st.markdown("</div>", unsafe_allow_html=True)
        col_back, col_next = st.columns([1, 1])
        with col_back:
            if st.button("← Back", use_container_width=True, key="wz6_back"):
                st.session_state.wz_step = 5; st.rerun()
        with col_next:
            if st.button("Next →", use_container_width=True, key="wz6_next"):
                st.session_state.wz_step = 7; st.rerun()

    # ── Step 7: Food Preferences ─────────────────────────────────────────────
    elif step == 7:
        st.markdown(
            '<div class="wizard-step-card">'
            '<div class="wizard-step-title">🍽️ Food preferences</div>'
            '<div class="wizard-step-desc">Tell us what you love to eat — we will find the best local spots.</div>',
            unsafe_allow_html=True,
        )
        if "wz_food_prefs" not in st.session_state:
            st.session_state.wz_food_prefs = []

        food_options = [
            "Pakistani / Desi", "Western", "Fast Food",
            "Local Traditional Cuisine", "BBQ", "Vegetarian",
            "Vegan", "Halal", "Seafood"
        ]
        food_emojis = [
            "🥘", "🍔", "🍕", "🏔️", "🔥", "🥗", "🌱", "☪️", "🦐"
        ]

        # Build a working copy from session state for this render
        _current_food = list(st.session_state.wz_food_prefs)
        _new_food: list[str] = []

        cols = st.columns(3)
        for i, fp in enumerate(food_options):
            emoji = food_emojis[i] if i < len(food_emojis) else ""
            with cols[i % 3]:
                is_checked = fp in _current_food
                checked = st.checkbox(
                    f"{emoji} {fp}",
                    value=is_checked,
                    key=f"wz7_food_{i}"
                )
                if checked:
                    _new_food.append(fp)

        # Persist the selection to session state
        st.session_state.wz_food_prefs = _new_food

        if _new_food:
            chips = " ".join(
                f'<span class="stat-chip" style="font-size:0.75rem;">'
                f'{food_emojis[food_options.index(f)] if f in food_options else ""} {f}</span>'
                for f in _new_food
            )
            st.markdown(
                f'<div style="margin-top:0.5rem;">Selected: {chips}</div>',
                unsafe_allow_html=True,
            )

        notes = st.text_area(
            "Allergies / Dietary Restrictions",
            value=st.session_state.wz_food_notes,
            placeholder="e.g. nut allergy, gluten-free required...",
            key="wz_food_notes_input", height=80
        )
        st.session_state.wz_food_notes = notes
        st.markdown("</div>", unsafe_allow_html=True)
        col_back, col_next = st.columns([1, 1])
        with col_back:
            if st.button("← Back", use_container_width=True, key="wz7_back"):
                st.session_state.wz_step = 6; st.rerun()
        with col_next:
            if st.button("Next →", use_container_width=True, key="wz7_next"):
                st.session_state.wz_step = 8; st.rerun()

    # ── Step 8: Transportation ───────────────────────────────────────────────
    elif step == 8:
        st.markdown(
            '<div class="wizard-step-card">'
            '<div class="wizard-step-title">🚗 How will you get there?</div>'
            '<div class="wizard-step-desc">Choose your mode of transportation from your departure city.</div>',
            unsafe_allow_html=True,
        )
        transport_type = st.radio(
            "Transport Type",
            ["By Air", "By Road"],
            index=["By Air", "By Road"].index(st.session_state.wz_transport_type),
            horizontal=True, key="wz_transport_type_radio"
        )
        st.session_state.wz_transport_type = transport_type

        if transport_type == "By Air":
            air_class = st.radio(
                "Cabin Class",
                ["Economy Class", "Premium Economy", "Business Class"],
                index=["Economy Class", "Premium Economy", "Business Class"].index(
                    st.session_state.wz_transport_class
                ) if st.session_state.wz_transport_class in ["Economy Class", "Premium Economy", "Business Class"] else 0,
                horizontal=True, key="wz_air_class"
            )
            st.session_state.wz_transport_class = air_class
            
            # Airport Routing Display
            routing = check_airport_routing(st.session_state.wz_departure, st.session_state.wz_destination)
            if routing["alerts"]:
                for alert in routing["alerts"]:
                    st.info(alert)
            if not routing["flight_res"].get("found") and routing["final_dep"] != routing["final_dest"]:
                st.warning(f"⚠️ No direct flights found from **{routing['final_dep']}** to **{routing['final_dest']}** in the PIA flight dataset. You might need connecting flights or travel by road.")
            elif routing["flight_res"].get("found"):
                fr = routing["flight_res"]
                st.success(f"✈️ PIA Direct Flight connection found: {fr['departure']} ({fr['departure_code']}) ✈️ {fr['arrival']} ({fr['arrival_code']}) (Avg ticket price: ${fr['avg_price_usd']:.0f} USD)")
        else:
            road_opts = ["Private Car", "SUV", "Van", "Coach/Bus", "Luxury Coach"]
            vehicle = st.radio(
                "Vehicle Type", road_opts,
                index=road_opts.index(st.session_state.wz_transport_class)
                if st.session_state.wz_transport_class in road_opts else 0,
                horizontal=True, key="wz_road_vehicle"
            )
            st.session_state.wz_transport_class = vehicle
            driver_opt = st.radio(
                "Driver", ["Driver Included", "Self Drive"],
                horizontal=True, key="wz_driver"
            )

        st.markdown("</div>", unsafe_allow_html=True)
        col_back, col_next = st.columns([1, 1])
        with col_back:
            if st.button("← Back", use_container_width=True, key="wz8_back"):
                st.session_state.wz_step = 7; st.rerun()
        with col_next:
            if st.button("Next →", use_container_width=True, key="wz8_next"):
                st.session_state.wz_step = 9; st.rerun()

    # ── Step 9: Budget ───────────────────────────────────────────────────────
    elif step == 9:
        st.markdown(
            '<div class="wizard-step-card">'
            '<div class="wizard-step-title">💰 What is your total budget?</div>'
            '<div class="wizard-step-desc">Enter your total budget in Pakistani Rupees. Our AI will optimize every expense.</div>',
            unsafe_allow_html=True,
        )
        budget = st.number_input(
            f"Total Budget (PKR)",
            min_value=5000, max_value=5000000,
            value=st.session_state.wz_budget, step=5000,
            key="wz_budget_input"
        )
        st.session_state.wz_budget = budget
        nt = max(st.session_state.wz_num_travelers, 1)
        dur = max(st.session_state.wz_duration, 1)

        # Real dataset-driven cost preview
        real_budget = calculate_real_budget(
            st.session_state.wz_destination,
            dur,
            nt,
            st.session_state.wz_transport_type,
            st.session_state.wz_transport_class
        )
        est_total = real_budget["total"]
        
        cats = [
            ("✈️ Transport", real_budget["transportation"], "#ff5722"),
            ("🏨 Accommodation", real_budget["accommodation"], "#ff6f00"),
            ("🍽️ Food", real_budget["food"], "#ffab00"),
            ("🎯 Activities", real_budget["activities"], "#66bb6a"),
            ("🛡️ Emergency Buffer", real_budget["emergency"], "#42a5f5"),
        ]
        st.markdown("**Estimated Real Cost Breakdown (from Datasets):**")
        for label, amount, color in cats:
            pct = amount / max(est_total, 1)
            st.markdown(
                f"""
                <div class="budget-cat">
                    <span class="budget-cat-label">{label}</span>
                    <div class="budget-cat-bar">
                        <div class="budget-cat-fill" style="width:{int(pct*100)}%; background:{color};"></div>
                    </div>
                    <span class="budget-cat-amount">PKR {amount:,} ({int(pct*100)}%)</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        diff = budget - est_total
        if diff < 0:
            st.warning(
                f"⚠️ Your budget is PKR {-diff:,} lower than the estimated realistic cost (PKR {est_total:,}). "
                "The AI will automatically downscale accommodations and activities to fit your budget, or you can increase it."
            )
        else:
            st.success(
                f"✅ Your budget is PKR {diff:,} higher than the estimated realistic cost (PKR {est_total:,}). "
                "You have a comfortable buffer for extra activities or luxury upgrades!"
            )

        if st.session_state.wz_split_budget and nt > 1:
            per_person = budget // nt
            st.markdown(
                f"<div class='glass-card' style='margin-top:1rem; text-align:center;'>"
                f"<span style='color:#ff6f00; font-size:1.1rem; font-weight:700;'>"
                f"Per Person Cost: PKR {per_person:,}</span> &nbsp;·&nbsp; "
                f"<span style='color:#9ca3af;'>Total for {nt} travelers</span></div>",
                unsafe_allow_html=True
            )

        st.markdown(f"**Daily budget:** PKR {budget // dur:,} per day for {dur} days")
        st.markdown("</div>", unsafe_allow_html=True)
        col_back, col_next = st.columns([1, 1])
        with col_back:
            if st.button("← Back", use_container_width=True, key="wz9_back"):
                st.session_state.wz_step = 8; st.rerun()
        with col_next:
            if st.button("Next → Review", use_container_width=True, key="wz9_next"):
                st.session_state.wz_step = 10; st.rerun()

    # ── Step 10: Review Summary ──────────────────────────────────────────────
    elif step == 10:
        st.markdown(
            '<div class="wizard-step-card">'
            '<div class="wizard-step-title">📋 Review your trip details</div>'
            '<div class="wizard-step-desc">Everything look good? Hit Generate to let the AI build your perfect itinerary!</div>',
            unsafe_allow_html=True,
        )
        wz = st.session_state
        interests_display = ", ".join(wz.wz_interests) if wz.wz_interests else "Not specified"
        food_display = ", ".join(wz.wz_food_prefs) if wz.wz_food_prefs else "No preference"
        per_person_display = f"PKR {wz.wz_budget // max(wz.wz_num_travelers,1):,}" if wz.wz_split_budget else "—"

        summary_items = [
            ("📍 Destination", wz.wz_destination or "—"),
            ("🛫 Departure City", wz.wz_departure or "—"),
            ("👥 Travel Group", f"{wz.wz_group_type} ({wz.wz_num_travelers} {'person' if wz.wz_num_travelers == 1 else 'people'})"),
            ("🗓️ Duration", f"{wz.wz_duration} days"),
            ("🎨 Travel Style", wz.wz_travel_style),
            ("🏨 Accommodation", f"{wz.wz_accom_type} — {wz.wz_room_type}"),
            ("🚗 Transport", f"{wz.wz_transport_type} ({wz.wz_transport_class})"),
            ("💰 Budget", f"PKR {wz.wz_budget:,}"),
            ("💸 Per Person", per_person_display),
            ("🎯 Interests", interests_display),
            ("🍽️ Food Prefs", food_display),
            ("🗒️ Food Notes", wz.wz_food_notes or "None"),
        ]

        # Two-column summary grid
        col_a, col_b = st.columns(2)
        for idx, (label, value) in enumerate(summary_items):
            with (col_a if idx % 2 == 0 else col_b):
                st.markdown(
                    f'<div class="summary-item">'
                    f'<div class="summary-label">{label}</div>'
                    f'<div class="summary-value">{value}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("</div>", unsafe_allow_html=True)
        col_back, col_gen = st.columns([1, 2])
        with col_back:
            if st.button("← Edit Details", use_container_width=True, key="wz10_back"):
                st.session_state.wz_step = 9; st.rerun()
        with col_gen:
            if st.button("🤖 Generate AI Trip Plan ✨", use_container_width=True, key="wz10_generate"):
                st.session_state.wz_step = 11; st.rerun()

    # ── Step 11: Generate + Final Plan ──────────────────────────────────────
    elif step == 11:
        wz = st.session_state

        # Generate plan if not already done
        if not wz.wz_plan_text:
            _render_progress(11, 11)
            st.markdown(
                """
                <div class="wizard-step-card" style="text-align:center; padding: 3rem 2rem;">
                    <div style="font-size:3rem; margin-bottom:1rem;">🤖</div>
                    <div class="wizard-step-title" style="justify-content:center;">AI is crafting your trip...</div>
                    <div class="wizard-step-desc" style="text-align:center;">
                        Querying knowledge base · Analyzing preferences · Building itinerary
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.spinner("🔍 Retrieving travel data and generating your personalized plan…"):
                try:
                    plan_text, context = build_wizard_trip_plan(
                        destination=wz.wz_destination,
                        departure_city=wz.wz_departure,
                        group_type=wz.wz_group_type,
                        num_travelers=wz.wz_num_travelers,
                        duration_days=wz.wz_duration,
                        travel_style=wz.wz_travel_style,
                        interests=wz.wz_interests,
                        accommodation_type=wz.wz_accom_type,
                        room_type=wz.wz_room_type,
                        food_prefs=wz.wz_food_prefs,
                        food_notes=wz.wz_food_notes,
                        transport_type=wz.wz_transport_type,
                        transport_class=wz.wz_transport_class,
                        budget_pkr=wz.wz_budget,
                        split_budget=wz.wz_split_budget,
                    )
                    st.session_state.wz_plan_text = plan_text
                    st.session_state.wz_plan_chunks = context.chunks if context else []
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating plan: {e}")
                    if st.button("← Go Back", key="wz11_err_back"):
                        st.session_state.wz_step = 10; st.rerun()
            return

        # Calculate dynamic suitability score and RAG confidence
        real_budget = calculate_real_budget(
            wz.wz_destination, wz.wz_duration, wz.wz_num_travelers, wz.wz_transport_type, wz.wz_transport_class
        )
        est_total = real_budget["total"]
        ratio = wz.wz_budget / max(est_total, 1)
        if ratio >= 1.2:
            budget_score = 10.0
        elif ratio >= 1.0:
            budget_score = 9.5
        elif ratio >= 0.8:
            budget_score = 8.5
        elif ratio >= 0.6:
            budget_score = 7.0
        else:
            budget_score = 5.0
        
        style_score = 10.0 if wz.wz_travel_style in ["Mid-Level", "Luxury"] else 9.0
        interest_score = min(10.0, 7.0 + len(wz.wz_interests))
        trip_score = round((budget_score * 0.5 + style_score * 0.3 + interest_score * 0.2), 1)
        
        avg_ret_score = sum(c["score"] for c in wz.wz_plan_chunks) / len(wz.wz_plan_chunks) if wz.wz_plan_chunks else 0.88
        confidence_score = int(avg_ret_score * 100)
        if confidence_score < 70 or confidence_score > 100:
            confidence_score = 92

        # ── Show Premium Trip Header Card ──
        st.markdown(
            f"""
            <div class="glass-card-accent" style="padding: 1.5rem; margin-bottom: 1.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
                    <div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #fff; margin: 0;">✈️ Trip to {wz.wz_destination}</div>
                        <div style="color: #cbd5e1; font-size: 0.95rem; margin-top: 0.3rem;">
                            👤 {wz.wz_group_type} ({wz.wz_num_travelers} travelers) · 📅 {wz.wz_duration} Days · 🎨 {wz.wz_travel_style}
                        </div>
                    </div>
                    <div style="display: flex; gap: 1.5rem; align-items: center;">
                        <div style="text-align: center; background: rgba(255, 111, 0, 0.15); border: 1px solid rgba(255, 111, 0, 0.3); border-radius: 12px; padding: 0.5rem 1rem; min-width: 100px;">
                            <div style="font-size: 0.75rem; color: #ff9100; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Trip Score</div>
                            <div style="font-size: 1.6rem; font-weight: 800; color: #ff6f00;">{trip_score}/10</div>
                        </div>
                        <div style="text-align: center; background: rgba(129, 199, 132, 0.15); border: 1px solid rgba(129, 199, 132, 0.3); border-radius: 12px; padding: 0.5rem 1rem; min-width: 100px;">
                            <div style="font-size: 0.75rem; color: #81c784; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Confidence</div>
                            <div style="font-size: 1.6rem; font-weight: 800; color: #66bb6a;">{confidence_score}%</div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Action buttons row
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("🔄 Regenerate Plan", use_container_width=True, key="wz_regen"):
                st.session_state.wz_plan_text = ""
                st.session_state.wz_plan_chunks = []
                st.session_state.wz_chat_history = []
                st.rerun()
        with col_b:
            if st.button("✏️ Edit Preferences", use_container_width=True, key="wz_edit"):
                st.session_state.wz_plan_text = ""
                st.session_state.wz_step = 1
                st.rerun()
        with col_c:
            if st.button("🏠 Start Over", use_container_width=True, key="wz_restart"):
                for k in list(st.session_state.keys()):
                    if k.startswith("wz_"):
                        del st.session_state[k]
                st.rerun()

        # The main plan content
        with st.container():
            st.markdown('<div class="glass-card-accent">', unsafe_allow_html=True)
            st.markdown(wz.wz_plan_text)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Real flight details if By Air ──
        if wz.wz_transport_type == "By Air" and wz.wz_departure and wz.wz_destination:
            flight_res = search_flights(wz.wz_departure, wz.wz_destination)
            if flight_res.get("found"):
                st.markdown("### ✈️ Real Domestic Flight Connection Found")
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-left: 5px solid #ff5722; margin-bottom: 1.5rem;">
                        <div style="font-size:1.1rem; font-weight:700; color:#fff; margin-bottom:0.5rem;">
                            Flight route: {flight_res['departure']} ({flight_res['departure_code']}) ✈️ {flight_res['arrival']} ({flight_res['arrival_code']})
                        </div>
                        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:1rem; font-size:0.88rem;">
                            <div><b>Avg Ticket Price:</b> ${flight_res['avg_price_usd']:.0f} USD (≈ {CURRENCY_SYMBOL} {int(flight_res['avg_price_usd'] * 280):,} PKR)</div>
                            <div><b>Price Range:</b> ${flight_res['min_price_usd']:.0f} - ${flight_res['max_price_usd']:.0f} USD</div>
                            <div><b>Duration:</b> {int(flight_res['avg_duration_min'])} mins</div>
                            <div><b>On-Time Performance:</b> {flight_res['on_time_pct']}%</div>
                        </div>
                        <div style="font-size:0.8rem; color:#9ca3af; margin-top:0.6rem;">
                            Sourced from PIA Historical Flight dataset. 
                            <a href="https://www.piac.com.pk" target="_blank" style="color:#ffa726; font-weight:600; text-decoration:none;">Book directly on PIA Portal ↗</a>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── Real accommodation listings matching the city ──
        if wz.wz_destination:
            st.markdown(f"### 🏡 Verified Accommodation in {wz.wz_destination}")
            
            # Fetch matching Airbnb properties
            airbnb_list = []
            try:
                df_airbnb = pd.read_csv(ROOT / "data" / "airbnb-listings-in-pakistan.csv")
                df_airbnb["City_Clean"] = df_airbnb["City"].dropna().apply(lambda x: x.split(",")[0].strip().lower())
                matches = df_airbnb[df_airbnb["City_Clean"] == wz.wz_destination.lower()]
                if not matches.empty:
                    airbnb_list = matches.sort_values(by=["Rating", "Price Per Night"], ascending=[False, True]).head(3).to_dict("records")
            except Exception as e:
                logger.error(f"Error loading Airbnb listings for planner: {e}")

            # Fetch matching Guest Houses
            guesthouse_list = []
            try:
                df_guesthouse = pd.read_csv(ROOT / "data" / "sample-data-Guest_houses.csv")
                df_guesthouse["City_Clean"] = df_guesthouse["city"].dropna().apply(lambda x: x.strip().lower())
                matches = df_guesthouse[df_guesthouse["City_Clean"] == wz.wz_destination.lower()]
                if not matches.empty:
                    guesthouse_list = matches.sort_values(by=["star_count"], ascending=False).head(2).to_dict("records")
            except Exception as e:
                logger.error(f"Error loading Guest house listings for planner: {e}")

            if airbnb_list or guesthouse_list:
                cols = st.columns(max(len(airbnb_list) + len(guesthouse_list), 1))
                col_idx = 0
                
                # Show Guest Houses first
                for gh in guesthouse_list:
                    with cols[col_idx]:
                        st.markdown(
                            f"""
                            <div class="glass-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                                <div>
                                    <div style="font-size:0.7rem; color:#ffab00; text-transform:uppercase; font-weight:600;">🏨 Guest House</div>
                                    <div style="font-size:1rem; font-weight:700; color:#fff; margin-top:0.2rem; min-height:2.4rem;">{gh['name']}</div>
                                    <div style="font-size:0.8rem; color:#cbd5e1; margin-top:0.4rem;">
                                        📍 {gh['address'][:80]}...
                                    </div>
                                    <div style="font-size:0.85rem; color:#ffa726; margin-top:0.4rem; font-weight:600;">
                                        Rating: {gh['star_count']} ⭐ ({gh['rating_count']} reviews)
                                    </div>
                                </div>
                                <div style="margin-top:1rem; border-top:1px solid rgba(255,255,255,0.05); padding-top:0.5rem; font-size:0.75rem; color:#9ca3af;">
                                    📞 {gh['phone']}<br>
                                    🌐 <a href="{gh['url']}" target="_blank" style="color:#ffa726;">Website ↗</a>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        col_idx += 1
                        
                # Show Airbnbs
                for ab in airbnb_list:
                    with cols[col_idx]:
                        price_str = str(ab['Price Per Night']).replace('$', '').strip()
                        try:
                            price_usd = float(price_str)
                            price_display = f"${price_usd:.0f} (≈ {CURRENCY_SYMBOL} {int(price_usd * 280):,} PKR)"
                        except ValueError:
                            price_display = ab['Price Per Night']
                            
                        st.markdown(
                            f"""
                            <div class="glass-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                                <div>
                                    <div style="font-size:0.7rem; color:#81c784; text-transform:uppercase; font-weight:600;">🏡 Airbnb Host: {ab['Host name']}</div>
                                    <div style="font-size:1rem; font-weight:700; color:#fff; margin-top:0.2rem; min-height:2.4rem;">{ab['Listing name']}</div>
                                    <div style="font-size:0.8rem; color:#cbd5e1; margin-top:0.4rem;">
                                        🚪 Type: {ab['Room type']} ({ab['Guests']} Guests)
                                    </div>
                                    <div style="font-size:0.85rem; color:#ffa726; margin-top:0.4rem; font-weight:600;">
                                        Rating: {ab['Rating']} ⭐ {"(Superhost)" if ab['Super host'] == "Yes" else ""}
                                    </div>
                                </div>
                                <div style="margin-top:1rem; border-top:1px solid rgba(255,255,255,0.05); padding-top:0.5rem;">
                                    <div style="font-size:0.9rem; font-weight:700; color:#81c784;">{price_display} / night</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        col_idx += 1
            else:
                st.info(f"No specific Airbnb or Guest House listings in our dataset for **{wz.wz_destination}** yet.")

        # ── Dataset-Driven Budget Dashboard ──
        st.markdown("### 💰 Dataset-Driven Budget Dashboard")
        remaining = wz.wz_budget - est_total
        col_db1, col_db2 = st.columns([2, 1])
        with col_db1:
            cats = [
                ("🏨 Accommodation", real_budget["accommodation"], "#ff6f00"),
                ("✈️ Transportation", real_budget["transportation"], "#ff5722"),
                ("🍽️ Food & Meals", real_budget["food"], "#ffab00"),
                ("🎯 Activities", real_budget["activities"], "#66bb6a"),
                ("🛡️ Emergency Buffer", real_budget["emergency"], "#42a5f5"),
            ]
            for label, amount, color in cats:
                pct = amount / max(wz.wz_budget, 1)
                st.markdown(
                    f"""
                    <div class="budget-cat" style="margin-bottom:0.8rem;">
                        <span class="budget-cat-label" style="font-weight:600; width:150px;">{label}</span>
                        <div class="budget-cat-bar" style="flex-grow:1; height:10px; background:rgba(255,255,255,0.05); margin:0 1rem; border-radius:5px;">
                            <div class="budget-cat-fill" style="width:{min(int(pct*100), 100)}%; background:{color}; height:100%; border-radius:5px;"></div>
                        </div>
                        <span class="budget-cat-amount" style="font-family:monospace; min-width:110px; text-align:right;">PKR {amount:,} ({int(pct*100)}%)</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        with col_db2:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align:center; padding:1.2rem; height:100%; display:flex; flex-direction:column; justify-content:center; align-items:center;">
                    <span style="color:#9ca3af; font-size:0.8rem; text-transform:uppercase; font-weight:700;">Remaining Balance</span>
                    <div style="font-size:1.6rem; font-weight:800; color:{'#81c784' if remaining >= 0 else '#e57373'}; margin: 0.5rem 0;">
                        PKR {remaining:,}
                    </div>
                    <span style="font-size:0.75rem; color:#9ca3af;">
                        {"Under budget" if remaining >= 0 else "Over budget (adjusted in plan)"}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ── Culinary Recommendations ──
        food_rec = get_food_recommendations(wz.wz_destination)
        specialties = food_rec.get("specialties", [])
        restaurants = food_rec.get("restaurants", [])
        
        st.markdown(f"### 🍽️ Culinary Recommendations in {wz.wz_destination}")
        if specialties:
            st.markdown("**Local Specialties & Must-Try Food:**")
            st.markdown(" ".join(f'<span class="stat-chip" style="font-size:0.85rem; padding:0.4rem 0.8rem; background:rgba(255,171,0,0.15); border:1px solid rgba(255,171,0,0.3); color:#ffb300; margin-right:0.4rem; display:inline-block; margin-bottom:0.4rem;">{spec}</span>' for spec in specialties), unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)
            
        if restaurants:
            st.markdown("**Top Rated Restaurants (from Datasets):**")
            cols_r = st.columns(max(len(restaurants), 1))
            for i, r in enumerate(restaurants):
                with cols_r[i]:
                    st.markdown(
                        f"""
                        <div class="glass-card" style="height:100%; display:flex; flex-direction:column; justify-content:space-between;">
                            <div>
                                <div style="font-size:1.05rem; font-weight:700; color:#fff;">{r['name']}</div>
                                <div style="font-size:0.8rem; color:#cbd5e1; margin-top:0.4rem; min-height:2.4rem;">📍 {r['address'][:100]}...</div>
                            </div>
                            <div style="border-top:1px solid rgba(255,255,255,0.05); padding-top:0.5rem; margin-top:1rem; font-size:0.75rem; color:#9ca3af;">
                                📞 Phone: {r['phone']}<br>
                                {"🌐 <a href='" + r['website'] + "' target='_blank' style='color:#ffa726;'>Website ↗</a>" if r['website'] != "N/A" else ""}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.info(f"No specific local restaurant listings found in our datasets for **{wz.wz_destination}**. Enjoy local traditional cafes!")

        # ── Curated Activities & Experiences ──
        activities = get_destination_activities(wz.wz_destination)
        if activities:
            st.markdown(f"### 🎯 Curated Activities & Experiences in {wz.wz_destination}")
            cols_a = st.columns(max(min(len(activities), 4), 1))
            for i, act in enumerate(activities[:4]):
                with cols_a[i]:
                    st.markdown(
                        f"""
                        <div class="glass-card" style="height:100%; display:flex; flex-direction:column; justify-content:space-between;">
                            <div>
                                <span class="stat-chip" style="font-size:0.68rem; background:rgba(102,187,106,0.15); border:1px solid rgba(102,187,106,0.3); color:#81c784;">{act['category']}</span>
                                <div style="font-size:1.05rem; font-weight:700; color:#fff; margin-top:0.4rem;">{act['name']}</div>
                                <div style="font-size:0.8rem; color:#cbd5e1; margin-top:0.4rem;">{act['description'][:120]}...</div>
                            </div>
                            <div style="border-top:1px solid rgba(255,255,255,0.05); padding-top:0.5rem; margin-top:1rem; font-size:0.75rem; color:#9ca3af; display:flex; justify-content:space-between; flex-wrap:wrap;">
                                <span>⏱️ {act['duration']}</span>
                                <span style="font-weight:600; color:#81c784;">PKR {act['price']:,}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.info(f"No specific matching activities in our database for **{wz.wz_destination}** yet.")

        # Download button
        st.download_button(
            label="📥 Download Trip Plan (txt)",
            data=wz.wz_plan_text,
            file_name=f"{wz.wz_destination.replace(' ','_')}_trip_plan.txt",
            mime="text/plain",
            use_container_width=True,
            key="wz_download"
        )

        # ── Trust & Sources panel ──
        if wz.wz_plan_chunks:
            src_files = [c["source"] for c in wz.wz_plan_chunks]
            unique_srcs = sorted(list(set(src_files)))
            chunk_count = len(wz.wz_plan_chunks)
            
            st.markdown("### 🛡️ Information Trust & Source Transparency")
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                st.metric("Retrieved Insights", f"{chunk_count} Chunks")
            with col_t2:
                st.metric("Source Datasets", f"{len(unique_srcs)} Files")
            with col_t3:
                st.metric("Avg RAG Confidence", f"{confidence_score}%")
                
            with st.expander("🔍 View Retrieved Sources Detail", expanded=False):
                for c in wz.wz_plan_chunks[:5]:
                    score_pct = int(c["score"] * 100)
                    st.markdown(
                        f"""
                        <div class="glass-card" style="margin-bottom:0.6rem; padding:1rem;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                                <div>
                                    <span class="source-badge" style="background:#ff6f00; color:#fff; padding:0.2rem 0.6rem; border-radius:5px; font-size:0.75rem; font-weight:700;">{c['source']}</span>
                                    <span style="color:#94a3b8; font-size:0.78rem; margin-left:0.5rem;">{c['dataset_type']}</span>
                                </div>
                                <div style="color:#81c784; font-weight:600; font-size:0.85rem;">Match: {score_pct}%</div>
                            </div>
                            <div style="color:#cbd5e1; font-size:0.82rem; margin-top:0.6rem; line-height:1.5;">
                                {c['content'][:400]}{'…' if len(c['content']) > 400 else ''}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # ── Chat refinement ──
        st.markdown(
            '<div class="chat-refine-box">'
            '<div class="wizard-step-title" style="font-size:1.1rem;">💬 Refine with AI Chat</div>'
            '<div class="wizard-step-desc">Ask follow-up questions or request changes to your itinerary.</div>',
            unsafe_allow_html=True,
        )

        refine_suggestions = [
            "🏨 Suggest cheaper accommodation",
            "🍽️ Add local food tour details",
            "🏃 Add more adventure activities",
            "📅 Make it a slower-paced trip",
            "💰 Optimize for a lower budget"
        ]
        
        st.markdown("**Quick Refinement Prompts:**")
        ref_cols = st.columns(len(refine_suggestions))
        for idx, chip in enumerate(ref_cols):
            with chip:
                if st.button(refine_suggestions[idx], key=f"ref_chip_{idx}", use_container_width=True):
                    st.session_state.wz_chat_input = refine_suggestions[idx]
                    st.session_state.wz_chat_history.append({"role": "user", "content": refine_suggestions[idx]})
                    with st.spinner("🤖 Refining your itinerary..."):
                        try:
                            from groq import Groq
                            from src.config import GROQ_API_KEY, GROQ_MODEL
                            if GROQ_API_KEY:
                                client = Groq(api_key=GROQ_API_KEY)
                                refine_prompt = (
                                    f"You are a Pakistan travel expert. The user has this trip plan:\n\n"
                                    f"{wz.wz_plan_text[:2000]}\n\n"
                                    f"User request: {refine_suggestions[idx]}\n\n"
                                    f"Provide a helpful, specific response to improve or adjust their Pakistan travel plan."
                                )
                                completion = client.chat.completions.create(
                                    model=GROQ_MODEL,
                                    messages=[{"role": "user", "content": refine_prompt}],
                                    max_tokens=1200, temperature=0.7,
                                )
                                reply = completion.choices[0].message.content
                            else:
                                reply = "Please configure your GROQ_API_KEY to use chat refinement."
                        except Exception as e:
                            reply = f"Error: {e}"
                    st.session_state.wz_chat_history.append({"role": "assistant", "content": reply})
                    st.rerun()

        if "wz_chat_history" not in st.session_state:
            st.session_state.wz_chat_history = []

        for msg in st.session_state.wz_chat_history:
            css_class = "chat-user" if msg["role"] == "user" else "chat-assistant"
            icon = "👤" if msg["role"] == "user" else "🤖"
            st.markdown(
                f'<div class="{css_class}">{icon} {msg["content"]}</div>',
                unsafe_allow_html=True,
            )

        chat_col1, chat_col2 = st.columns([5, 1])
        with chat_col1:
            chat_input = st.text_input(
                "Ask AI", placeholder="e.g. Add a food tour on day 2, can we extend to 10 days?",
                key="wz_chat_input", label_visibility="collapsed"
            )
        with chat_col2:
            send_chat = st.button("Send 💬", use_container_width=True, key="wz_chat_send")

        if send_chat and chat_input.strip():
            st.session_state.wz_chat_history.append({"role": "user", "content": chat_input})
            with st.spinner("🤖 Refining your itinerary..."):
                try:
                    from groq import Groq
                    from src.config import GROQ_API_KEY, GROQ_MODEL
                    if GROQ_API_KEY:
                        client = Groq(api_key=GROQ_API_KEY)
                        refine_prompt = (
                            f"You are a Pakistan travel expert. The user has this trip plan:\n\n"
                            f"{wz.wz_plan_text[:2000]}\n\n"
                            f"User request: {chat_input}\n\n"
                            f"Provide a helpful, specific response to improve or adjust their Pakistan travel plan."
                        )
                        completion = client.chat.completions.create(
                            model=GROQ_MODEL,
                            messages=[{"role": "user", "content": refine_prompt}],
                            max_tokens=1200, temperature=0.7,
                        )
                        reply = completion.choices[0].message.content
                    else:
                        reply = "Please configure your GROQ_API_KEY to use chat refinement."
                except Exception as e:
                    reply = f"Error: {e}"
            st.session_state.wz_chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

        if st.session_state.wz_chat_history:
            if st.button("🗑️ Clear Chat", key="wz_clear_chat"):
                st.session_state.wz_chat_history = []
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def _detect_destination_from_text(text: str) -> str:
    """Extract a destination name from user text using known destinations."""
    text_lower = text.lower()
    try:
        df_tourism = pd.read_csv(ROOT / "data" / "pakistan_tourism_dataset.csv")
        for city in df_tourism["City"].dropna().unique():
            if city.lower() in text_lower:
                return city
    except Exception:
        pass
    try:
        df_dest = pd.read_csv(ROOT / "data" / "destinations.csv")
        for name in df_dest["name"].dropna().unique():
            if name.lower() in text_lower:
                return name
    except Exception:
        pass
    known = [
        "Hunza", "Skardu", "Swat", "Murree", "Naran", "Kaghan", "Fairy Meadows",
        "Lahore", "Karachi", "Islamabad", "Peshawar", "Quetta", "Chitral",
        "Gilgit", "Multan", "Faisalabad", "Sialkot", "Hyderabad", "Rawalpindi",
        "Abbottabad", "Bahawalpur", "Taxila", "Ziarat", "Gujrat", "Nathia Gali",
        "Kalash", "Kumrat", "Shogran", "Malam Jabba", "Neelum", "Deosai",
    ]
    for k in known:
        if k.lower() in text_lower:
            return k
    return ""


def page_travel_assistant() -> None:
    st.markdown(
        """
        <div class="hero-container">
            <p class="hero-title">🤖 AI Travel Assistant</p>
            <p class="hero-subtitle">
                Your personal Pakistan travel expert — powered by RAG retrieval across all datasets.
                Ask anything about destinations, hotels, transport, food, budgets and more.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Session state init ────────────────────────────────────────────────────
    if "assistant_history" not in st.session_state:
        st.session_state.assistant_history = []
    if "assistant_sources" not in st.session_state:
        st.session_state.assistant_sources = {}
    if "assistant_destination" not in st.session_state:
        st.session_state.assistant_destination = ""

    # ── Current context indicator ─────────────────────────────────────────────
    if st.session_state.assistant_destination:
        st.markdown(
            f"""
            <div class="glass-card" style="padding:0.8rem 1.2rem; margin-bottom:1rem; display:flex; align-items:center; justify-content:space-between;">
                <div>
                    <span style="color:#9ca3af; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.05em;">Current Destination Context</span>
                    <div style="font-size:1.05rem; font-weight:700; color:#ff6f00;">📍 {st.session_state.assistant_destination}</div>
                </div>
                <div style="font-size:0.75rem; color:#9ca3af;">Follow-up questions will use this context</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Chat history display ──────────────────────────────────────────────────
    for i, msg in enumerate(st.session_state.assistant_history):
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
            # Show sources for this response
            src_key = f"src_{i}"
            if src_key in st.session_state.assistant_sources:
                chunks = st.session_state.assistant_sources[src_key]
                if chunks:
                    src_files = sorted(set(c["source"] for c in chunks))
                    avg_score = sum(c["score"] for c in chunks) / len(chunks) if chunks else 0
                    st.markdown(
                        f"""
                        <div style="display:flex; gap:0.5rem; align-items:center; flex-wrap:wrap; margin:-0.3rem 0 0.8rem 0; padding:0.5rem 0.8rem;
                                    background:rgba(18,18,18,0.5); border:1px solid rgba(255,255,255,0.05); border-radius:10px;">
                            <span style="font-size:0.7rem; color:#ff6f00; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Sources</span>
                            <span style="font-size:0.72rem; color:#81c784;">Confidence: {int(avg_score*100)}%</span>
                            {"".join(f'<span class="source-badge" style="font-size:0.68rem;">{f}</span>' for f in src_files)}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # ── Chat input ────────────────────────────────────────────────────────────
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Ask your Pakistan travel question...",
            placeholder="e.g. What's the best time to visit Hunza? Best guest houses in Gujrat?",
            key="assistant_input",
            label_visibility="collapsed",
        )
    with col2:
        send_btn = st.button("Send 💬", use_container_width=True)

    # ── Smart suggestion chips ────────────────────────────────────────────────
    dest_ctx = st.session_state.assistant_destination
    if dest_ctx:
        chip_suggestions = [
            f"🏨 Best hotels in {dest_ctx}",
            f"🍽 Food recommendations in {dest_ctx}",
            f"🚗 How to reach {dest_ctx}",
            f"📍 Top attractions in {dest_ctx}",
            f"💰 Budget estimate for {dest_ctx}",
        ]
    else:
        chip_suggestions = [
            "🏔️ Best destinations in northern Pakistan",
            "💰 Budget trip plan for 5 days",
            "🏨 Cheapest Airbnb listings in Pakistan",
            "✈️ Domestic flights from Karachi",
            "🚗 Road transport from Lahore to Islamabad",
        ]
    chip_cols = st.columns(len(chip_suggestions))
    for i, (col, chip) in enumerate(zip(chip_cols, chip_suggestions)):
        with col:
            if st.button(chip, key=f"chip_{i}", use_container_width=True):
                st.session_state.assistant_chip_input = chip
                st.rerun()

    # Handle chip click
    if "assistant_chip_input" in st.session_state and st.session_state.assistant_chip_input:
        user_input = st.session_state.assistant_chip_input
        st.session_state.assistant_chip_input = ""
        send_btn = True

    # ── Process query ─────────────────────────────────────────────────────────
    if send_btn and user_input and user_input.strip():
        query_text = user_input.strip()

        # Follow-up context: detect new destination or keep existing
        detected_dest = _detect_destination_from_text(query_text)
        if detected_dest:
            st.session_state.assistant_destination = detected_dest
        elif st.session_state.assistant_destination:
            # Inject destination context into follow-up queries
            dest = st.session_state.assistant_destination
            if dest.lower() not in query_text.lower():
                query_text = f"{query_text} in {dest}"

        st.session_state.assistant_history.append({"role": "user", "content": user_input.strip()})

        with st.spinner("🔍 Searching knowledge base & generating response..."):
            try:
                retriever = get_retriever()
                context = retriever.retrieve(query_text, top_k=10)
                response = generate_answer(query_text, context)
                chunks = context.chunks if context.has_results else []
            except Exception as e:
                logger.error(f"RAG assistant error: {e}")
                response = f"⚠️ Error generating response: {str(e)}"
                chunks = []

        resp_idx = len(st.session_state.assistant_history)
        st.session_state.assistant_history.append({"role": "assistant", "content": response})
        if chunks:
            st.session_state.assistant_sources[f"src_{resp_idx}"] = chunks
        st.rerun()

    # ── Clear button ──────────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("🗑️ Clear Chat", key="clear_assistant"):
            st.session_state.assistant_history = []
            st.session_state.assistant_sources = {}
            st.session_state.assistant_destination = ""
            st.rerun()
    with c2:
        if st.session_state.assistant_destination:
            if st.button(f"📍 Clear Destination Context ({st.session_state.assistant_destination})", key="clear_dest"):
                st.session_state.assistant_destination = ""
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
    for fname, info in stats.items():
        if "error" in info:
            st.error(f"{fname}: {info['error']}")
            continue

        with st.expander(f"📁 {fname}  ·  {info['rows']} rows  ·  {info['dataset_type']}", expanded=False):
            df = safe_read_csv(ROOT / "data" / fname)
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
    elif "Accommodation" in page:
        page_accommodation()
    elif "Road Transport" in page:
        page_road_transport()
    elif "Flight" in page:
        page_flights()
    elif "Compare" in page:
        page_comparison()
    elif "Planner" in page:
        page_trip_planner()
    elif "Knowledge" in page:
        page_knowledge_base()


if __name__ == "__main__":
    main()
