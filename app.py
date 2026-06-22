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
}

/* Allow text selection for content areas */
.stMarkdown, .element-container p, .stMarkdown p,
.glass-card, .glass-card-accent, .chat-user, .chat-assistant,
.wizard-step-card, input, textarea, [contenteditable] {
    -webkit-user-select: text;
    -moz-user-select: text;
    -ms-user-select: text;
    user-select: text;
}

/* Disable selection only for interactive elements */
.stButton > button,
.stCheckbox > label,
.stRadio > label,
.stSelectbox > div {
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
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
    fetch_real_time_flights, get_city_iata,
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
        # First try to load from bus companies data
        bus_df = pd.read_csv(ROOT / "data" / "new data" / "Bus_companies_data.csv")
        
        # Extract useful bus company information with enhanced logic
        cities_from_csv = set()
        operators_from_csv = set()
        
        for _, row in bus_df.iterrows():
            title = str(row.get("title", "")).strip()
            city = str(row.get("city", "")).strip()
            address = str(row.get("address", "")).strip()
            phone = str(row.get("phone", "")).strip()
            total_score = row.get("totalScore", 0)
            category = str(row.get("categoryName", "")).strip()
            
            # Broader criteria to capture more transport companies
            is_transport_related = (
                "travel" in title.lower() or "express" in title.lower() or 
                "transport" in title.lower() or "coach" in title.lower() or
                "bus" in title.lower() or "terminal" in title.lower()
            )
            
            if title and city and is_transport_related and len(title) > 3:
                cities_from_csv.add(city)
                operators_from_csv.add(title)
                
                # Create multiple route combinations for each operator
                major_cities = ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Peshawar", "Quetta", "Multan", "Faisalabad", "Hyderabad", "Gujranwala"]
                
                for dest_city in major_cities:
                    if dest_city != city:  # Don't create routes to the same city
                        # Determine fare based on operator
                        base_fare = 2000
                        if "daewoo" in title.lower():
                            base_fare = 3500
                        elif "niazi" in title.lower():
                            base_fare = 3200
                        elif "bilal" in title.lower():
                            base_fare = 2900
                        elif "kainat" in title.lower():
                            base_fare = 2600
                        
                        # Distance multiplier
                        long_distance_routes = [("Karachi", "Lahore"), ("Karachi", "Islamabad"), ("Lahore", "Quetta")]
                        if (city, dest_city) in long_distance_routes or (dest_city, city) in long_distance_routes:
                            fare = int(base_fare * 1.2)
                            duration = "10-12"
                        else:
                            fare = base_fare
                            duration = "5-8"
                        
                        # Vehicle type based on operator
                        if "express" in title.lower():
                            vehicle_type = "AC Bus"
                        elif "luxury" in title.lower() or "daewoo" in title.lower():
                            vehicle_type = "Luxury Bus"
                        else:
                            vehicle_type = "Standard Bus"
                
                        routes.append({
                            "operator": title,
                            "departure_city": city,
                            "arrival_city": dest_city,
                            "vehicle_type": vehicle_type,
                            "fare_pkr": fare,
                            "duration_hours": duration,
                            "contact": phone if phone and len(phone) > 5 else "Contact directly",
                            "address": address if len(address) > 5 else "City terminal",
                            "rating": f"{total_score:.1f}" if isinstance(total_score, (int, float)) and total_score > 0 else "N/A"
                        })
        
        # If no bus companies found, add default operators with major Pakistani cities
        if not routes:
            major_cities = ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Peshawar", "Quetta", 
                           "Multan", "Faisalabad", "Sialkot", "Gujranwala", "Hyderabad", "Sargodha"]
            
            default_operators = [
                {
                    "operator": "Daewoo Express",
                    "departure_city": "Karachi",
                    "arrival_city": "Lahore",
                    "vehicle_type": "AC Bus",
                    "fare_pkr": "3500",
                    "duration_hours": "18",
                    "contact": "0800-DAEWOO",
                    "address": "Multiple terminals nationwide",
                    "rating": "4.2"
                },
                {
                    "operator": "Daewoo Express",
                    "departure_city": "Lahore",
                    "arrival_city": "Islamabad",
                    "vehicle_type": "AC Bus", 
                    "fare_pkr": "1800",
                    "duration_hours": "5",
                    "contact": "0800-DAEWOO",
                    "address": "Multiple terminals nationwide",
                    "rating": "4.2"
                },
                {
                    "operator": "NATCO",
                    "departure_city": "Islamabad",
                    "arrival_city": "Karachi",
                    "vehicle_type": "AC Bus",
                    "fare_pkr": "4200",
                    "duration_hours": "20",
                    "contact": "021-111-NATCO",
                    "address": "Government operated terminals",
                    "rating": "4.0"
                },
                {
                    "operator": "Faisal Movers",
                    "departure_city": "Lahore",
                    "arrival_city": "Rawalpindi",
                    "vehicle_type": "Luxury Bus",
                    "fare_pkr": "2800",
                    "duration_hours": "5",
                    "contact": "042-111-FAISAL",
                    "address": "Major city terminals",
                    "rating": "4.3"
                },
                {
                    "operator": "Bilal Travels",
                    "departure_city": "Islamabad",
                    "arrival_city": "Gilgit",
                    "vehicle_type": "Mountain Bus",
                    "fare_pkr": "3200",
                    "duration_hours": "12",
                    "contact": "051-111-BILAL",
                    "address": "Northern areas specialist",
                    "rating": "4.1"
                },
                {
                    "operator": "Niazi Express",
                    "departure_city": "Karachi",
                    "arrival_city": "Multan",
                    "vehicle_type": "AC Coach",
                    "fare_pkr": "2900",
                    "duration_hours": "14",
                    "contact": "021-111-NIAZI",
                    "address": "Southern routes specialist",
                    "rating": "4.0"
                },
                {
                    "operator": "Kohistan Express",
                    "departure_city": "Peshawar",
                    "arrival_city": "Islamabad",
                    "vehicle_type": "Standard Bus",
                    "fare_pkr": "1500",
                    "duration_hours": "3",
                    "contact": "091-111-KSTAN",
                    "address": "KPK regional routes",
                    "rating": "3.9"
                }
            ]
            routes.extend(default_operators)
        
    except Exception as e:
        logger.error(f"Error loading road transport: {e}")
        # Fallback to default operators
        routes = [
            {
                "operator": "Daewoo Express",
                "departure_city": "Karachi",
                "arrival_city": "Lahore", 
                "vehicle_type": "AC Bus",
                "fare_pkr": "3500",
                "duration_hours": "18",
                "contact": "0800-DAEWOO",
                "address": "Multiple terminals",
                "rating": "4.2"
            }
        ]
    
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
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

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
            <p class="hero-title">🏨 Accommodation Finder</p>
            <p class="hero-subtitle">
                Search Hotels and Guest Houses across Pakistan — exclusively from our curated travel database.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load accommodations from new data structure only
    hotels = []
    guest_houses = []
    
    try:
        # Load hotels
        hotels_df = pd.read_csv(ROOT / "data" / "new data" / "Hotels data" / "hotels.csv")
        for _, row in hotels_df.iterrows():
            hotels.append({
                "name": str(row.get("name", "")).strip(),
                "type": "Hotel",
                "city": str(row.get("city", "")).strip(),
                "province": str(row.get("province", "")).strip(),
                "price": str(row.get("price_range", "PKR 5,000-15,000")),
                "rating": f"{row.get('rating', 'N/A')} ⭐",
                "contact": f"📞 {row.get('phone', 'Contact hotel')} | 📧 {row.get('email', 'Email available')}",
                "description": f"Address: {row.get('address', 'Address available')}. Amenities: {row.get('amenities', 'Standard amenities')}",
                "url": ""
            })
    except Exception as e:
        logger.info(f"Hotels data not available: {e}")

    try:
        # Load guest houses
        gh_df = pd.read_csv(ROOT / "data" / "new data" / "Guest houses data" / "guest_houses.csv")
        for _, row in gh_df.iterrows():
            guest_houses.append({
                "name": str(row.get("name", "")).strip(),
                "type": "Guest House",
                "city": str(row.get("city", "")).strip(),
                "province": str(row.get("province", "")).strip(),
                "price": str(row.get("price_range", "PKR 3,000-8,000")),
                "rating": f"{row.get('rating', 'N/A')} ⭐",
                "contact": f"📞 {row.get('phone', 'Contact guest house')} | 📧 {row.get('email', 'Email available')}",
                "description": f"Address: {row.get('address', 'Address available')}. Amenities: {row.get('amenities', 'Basic amenities')}",
                "url": ""
            })
    except Exception as e:
        logger.info(f"Guest houses data not available: {e}")

    all_accoms = hotels + guest_houses
    
    if not all_accoms:
        st.warning("📍 Accommodation data is being updated from our new data sources.")
        st.info("Currently transitioning to Hotels and Guest Houses only. Airbnb listings have been removed as per system requirements.")
        return

    all_cities = sorted(set(a["city"] for a in all_accoms if a["city"] and a["city"] != "N/A"))
    all_types = ["Hotel", "Guest House"]

    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
    with col_f1:
        search_q = st.text_input("🔍 Search by name or description", placeholder="e.g. Pearl Continental, Mountain View Guest House...", key="accom_search")
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

    st.markdown(f"<div style='color:#9ca3af; font-size:0.85rem; margin-bottom:1rem;'>Showing **{len(filtered)}** accommodation records (Hotels & Guest Houses only)</div>", unsafe_allow_html=True)

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
                                <div style="font-size:0.85rem;color:#ffa726;">📍 {item['city']}, {item.get('province', 'Pakistan')}</div>
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
        st.info("No accommodations match your search. Try adjusting filters or check back as we update our database.")
        return

    display_items = filtered[:30]
    cols = st.columns(3)
    for i, accom in enumerate(display_items):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="glass-card" style="height:100%;display:flex;flex-direction:column;justify-content:space-between;margin-bottom:1rem;">
                    <div>
                        <div style="font-size:0.65rem;color:{'#ffab00' if accom['type']=='Guest House' else '#64b5f6'};text-transform:uppercase;font-weight:600;">{accom['type']}</div>
                        <div style="font-size:0.95rem;font-weight:700;color:#fff;margin-top:0.3rem;min-height:2.2rem;">{accom['name'][:50]}</div>
                        <div style="font-size:0.78rem;color:#cbd5e1;margin-top:0.3rem;">📍 {accom['city']}</div>
                        <div style="font-size:0.82rem;color:#ffa726;font-weight:600;margin-top:0.3rem;">💰 {accom['price']}</div>
                        <div style="font-size:0.82rem;color:#ffa726;margin-top:0.2rem;">{accom['rating']}</div>
                    </div>
                    <div style="margin-top:0.8rem;border-top:1px solid rgba(255,255,255,0.05);padding-top:0.5rem;">
                        <div style="font-size:0.72rem;color:#9ca3af;">{accom['contact'][:80]}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    # Source attribution (no CSV filename exposure)
    st.caption("📊 *Accommodation listings based on available travel records in our Pakistan tourism database.*")


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
        st.info("🚧 Road transport data is being updated. Major operators include Daewoo Express, NATCO, and Faisal Movers.")
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
        # Compare by operators instead of just vehicle types
        operators = sorted(set(r["operator"] for r in all_routes))
        
        if len(operators) >= 2:
            comp_c1, comp_c2 = st.columns(2)
            with comp_c1:
                op_a = st.selectbox("First Operator", operators, key="op_cmp_a")
            with comp_c2:
                op_b = st.selectbox("Second Operator", operators, index=min(1, len(operators)-1), key="op_cmp_b")

            op_a_routes = [r for r in all_routes if r["operator"] == op_a]
            op_b_routes = [r for r in all_routes if r["operator"] == op_b]

            ca, cb = st.columns(2)
            for col, op, op_routes in [(ca, op_a, op_a_routes), (cb, op_b, op_b_routes)]:
                with col:
                    if op_routes:
                        # Handle fare calculation safely - skip non-numeric values
                        numeric_fares = []
                        for r in op_routes:
                            fare = r["fare_pkr"]
                            if isinstance(fare, (int, float)):
                                numeric_fares.append(fare)
                            elif isinstance(fare, str) and fare.replace(",", "").isdigit():
                                numeric_fares.append(int(fare.replace(",", "")))
                        
                        if numeric_fares:
                            avg_fare = int(sum(numeric_fares) / len(numeric_fares))
                            min_fare = min(numeric_fares)
                            max_fare = max(numeric_fares)
                            fare_display = f"PKR {avg_fare:,} (avg)"
                            range_display = f"PKR {min_fare:,} - {max_fare:,}"
                        else:
                            fare_display = "Contact for rates"
                            range_display = "Varies"
                        
                        # Handle duration calculation safely
                        numeric_durations = []
                        for r in op_routes:
                            dur_str = str(r["duration_hours"])
                            try:
                                if "-" in dur_str:
                                    # Handle ranges like "5-8"
                                    parts = dur_str.split("-")
                                    avg_dur = (float(parts[0]) + float(parts[1])) / 2
                                    numeric_durations.append(avg_dur)
                                elif dur_str.replace(".", "").isdigit():
                                    numeric_durations.append(float(dur_str))
                            except:
                                continue
                        
                        if numeric_durations:
                            avg_dur = round(sum(numeric_durations) / len(numeric_durations), 1)
                            dur_display = f"{avg_dur} hrs (avg)"
                        else:
                            dur_display = "Varies"
                        
                        # Get unique vehicle types and cities served
                        vehicle_types = sorted(set(r["vehicle_type"] for r in op_routes))
                        cities_served = len(set(r["departure_city"] for r in op_routes) | set(r["arrival_city"] for r in op_routes))
                        
                        st.markdown(
                            f"""
                            <div class="glass-card-accent">
                                <div style="font-size:1.1rem;font-weight:700;color:#fff;margin-bottom:0.8rem;">{op}</div>
                                <div style="font-size:0.85rem;color:#ffa726;margin:0.3rem 0;">💰 Average Fare: {fare_display}</div>
                                <div style="font-size:0.8rem;color:#cbd5e1;">📊 Fare Range: {range_display}</div>
                                <div style="font-size:0.85rem;color:#cbd5e1;margin:0.3rem 0;">⏱️ Average Duration: {dur_display}</div>
                                <div style="font-size:0.85rem;color:#9ca3af;">🚌 Vehicle Types: {', '.join(vehicle_types)}</div>
                                <div style="font-size:0.82rem;color:#9ca3af;">🗺️ Cities Served: {cities_served}</div>
                                <div style="font-size:0.82rem;color:#9ca3af;">📋 Total Routes: {len(op_routes)}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            """
                            <div class="glass-card-accent">
                                <div style="color:#9ca3af;text-align:center;padding:2rem;">
                                    No routes available for this operator
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
        else:
            st.info("Not enough operators available for comparison. Need at least 2 operators.")

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
                    <div><span style="color:#9ca3af;">💰 Fare:</span> <span style="color:#ffa726;font-weight:600;">{r['fare_pkr'] if isinstance(r['fare_pkr'], str) else f"PKR {r['fare_pkr']:,}"}</span></div>
                    <div><span style="color:#9ca3af;">⏱️ Duration:</span> <span style="color:#fff;">{r['duration_hours']} hrs</span></div>
                    <div><span style="color:#9ca3af;">📞 Helpline:</span> <span style="color:#81c784;">{r.get('contact', r.get('contact_number', 'N/A'))}</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Summary stats ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Intercity Transport Summary")
    
    # Extract only numeric fares for calculations
    numeric_fares = []
    for r in all_routes:
        fare = r["fare_pkr"]
        if isinstance(fare, (int, float)) and fare > 0:
            numeric_fares.append(fare)
        elif isinstance(fare, str) and fare.isdigit():
            numeric_fares.append(int(fare))
    
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, label, val in [
        (c1, "🚌", "Total Routes", len(all_routes)),
        (c2, "🏢", "Operators", len(set(r["operator"] for r in all_routes))),
        (c3, "💰", "Min Fare", f"PKR {min(numeric_fares):,}" if numeric_fares else "Contact operators"),
        (c4, "💳", "Max Fare", f"PKR {max(numeric_fares):,}" if numeric_fares else "Contact operators"),
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
            <p class="hero-title">✈️ Real-Time Flight Search</p>
            <p class="hero-subtitle">
                Live flight data powered by AviationStack API — check real-time status, compare airlines, view schedules.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cities = get_all_airports()
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        dep_city = st.selectbox(
            "🛫 Departure City", cities,
            index=cities.index("Karachi") if "Karachi" in cities else 0,
            key="flight_dep"
        )
    with col2:
        arr_cities = [c for c in cities if c != dep_city]
        arr_city = st.selectbox(
            "🛬 Arrival City", arr_cities,
            index=arr_cities.index("Lahore") if "Lahore" in arr_cities else 0,
            key="flight_arr"
        )
    with col3:
        cabin_class = st.selectbox("💺 Class", ["Economy", "Business"], key="flight_class")

    dep_iata = get_city_iata(dep_city)
    arr_iata = get_city_iata(arr_city)
    dep_ap = AIRPORTS.get(dep_city, {})
    arr_ap = AIRPORTS.get(arr_city, {})
    dep_name = dep_ap.get('name', dep_city)
    arr_name = arr_ap.get('name', arr_city)

    st.markdown(
        f"""
        <div style="display:flex; gap:1rem; margin-bottom:1rem; flex-wrap:wrap;">
            <div style="background:rgba(255,87,34,0.08); border:1px solid rgba(255,87,34,0.2); border-radius:10px; padding:0.5rem 1.2rem;">
                <span style="font-size:0.72rem; color:#9ca3af; text-transform:uppercase;">Departure</span>
                <div style="font-size:1.3rem; font-weight:700; color:#ff6f00;">{dep_iata}</div>
                <div style="font-size:0.78rem; color:#cbd5e1;">{dep_name}</div>
            </div>
            <div style="display:flex; align-items:center; font-size:1.8rem; color:#ff5722; padding:0 0.5rem;">✈️</div>
            <div style="background:rgba(255,87,34,0.08); border:1px solid rgba(255,87,34,0.2); border-radius:10px; padding:0.5rem 1.2rem;">
                <span style="font-size:0.72rem; color:#9ca3af; text-transform:uppercase;">Arrival</span>
                <div style="font-size:1.3rem; font-weight:700; color:#ff6f00;">{arr_iata}</div>
                <div style="font-size:0.78rem; color:#cbd5e1;">{arr_name}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🔍 Search Real-Time Flights", use_container_width=True, key="flight_search_btn"):
        with st.spinner("🛰️ Fetching live flight data from AviationStack..."):
            result = fetch_real_time_flights(dep_iata, arr_iata, limit=20)

        source = result.get("source", "error")

        if source == "aviationstack" and result.get("flights"):
            flights = result["flights"]
            total = result.get("total_found", len(flights))
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:1.2rem;">
                    <span style="background:#1b5e20; border:1px solid #4caf50; color:#81c784;
                                 border-radius:20px; padding:0.3rem 0.8rem; font-size:0.78rem; font-weight:600;">
                        🟢 LIVE DATA — AviationStack
                    </span>
                    <span style="color:#9ca3af; font-size:0.82rem;">
                        {total} flight(s) found for {dep_iata} → {arr_iata}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("### ✈️ Live Flights")
            for f in flights:
                delay_dep = f.get("delay_departure_min")
                delay_str = ""
                if delay_dep and isinstance(delay_dep, (int, float)) and delay_dep > 0:
                    delay_str = f'<span style="color:#ff5722; font-size:0.75rem; font-weight:600;">⚠️ +{int(delay_dep)}min delay</span>'
                st.markdown(
                    f"""
                    <div class="glass-card" style="margin-bottom:0.9rem;">
                        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:0.5rem;">
                            <div>
                                <span style="font-size:1.2rem; font-weight:700; color:#fff;">{f['flight_number']}</span>
                                <span style="font-size:0.85rem; color:#ffa726; margin-left:0.6rem;">{f['airline_name']}</span>
                                {delay_str}
                            </div>
                            <span style="background:rgba(0,0,0,0.3); border:1px solid {f['status_color']};
                                         color:{f['status_color']}; border-radius:20px;
                                         padding:0.2rem 0.8rem; font-size:0.78rem; font-weight:600;">
                                {f['status']}
                            </span>
                        </div>
                        <div style="display:grid; grid-template-columns:1fr auto 1fr; gap:1rem; margin-top:1rem; align-items:center;">
                            <div>
                                <div style="font-size:1.4rem; font-weight:700; color:#ff6f00;">{f['departure_iata']}</div>
                                <div style="font-size:0.8rem; color:#cbd5e1;">{f['departure_airport'][:30]}</div>
                                <div style="font-size:0.75rem; color:#9ca3af; margin-top:0.2rem;">🕐 {f['departure_time']}</div>
                                <div style="font-size:0.72rem; color:#9ca3af;">Terminal: {f['departure_terminal']} | Gate: {f['departure_gate']}</div>
                            </div>
                            <div style="text-align:center; color:#ff5722; font-size:1.4rem;">→</div>
                            <div style="text-align:right;">
                                <div style="font-size:1.4rem; font-weight:700; color:#ff6f00;">{f['arrival_iata']}</div>
                                <div style="font-size:0.8rem; color:#cbd5e1;">{f['arrival_airport'][:30]}</div>
                                <div style="font-size:0.75rem; color:#9ca3af; margin-top:0.2rem;">🕐 {f['arrival_time']}</div>
                                <div style="font-size:0.72rem; color:#9ca3af;">Terminal: {f['arrival_terminal']} | Gate: {f['arrival_gate']}</div>
                            </div>
                        </div>
                        <div style="display:flex; gap:1rem; margin-top:0.8rem; flex-wrap:wrap; font-size:0.8rem; color:#9ca3af;
                                    border-top:1px solid rgba(255,255,255,0.05); padding-top:0.6rem;">
                            <span>✈️ Aircraft: <strong style="color:#cbd5e1;">{f['aircraft_type']}</strong></span>
                            <span>🪪 Reg: <strong style="color:#cbd5e1;">{f['aircraft_registration']}</strong></span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        elif source == "openai_fallback" and result.get("ai_insight"):
            ai = result["ai_insight"]
            st.markdown(
                """
                <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:1.2rem;">
                    <span style="background:rgba(255,111,0,0.15); border:1px solid rgba(255,111,0,0.4);
                                 color:#ffa726; border-radius:20px; padding:0.3rem 0.8rem;
                                 font-size:0.78rem; font-weight:600;">
                        🤖 AI-GENERATED INSIGHTS — Live API unavailable
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            airlines_str = ", ".join(ai.get("airlines", ["PIA", "AirSial"]))
            st.markdown(
                f"""
                <div class="glass-card-accent">
                    <div style="font-size:1rem; font-weight:600; color:#fff; margin-bottom:1rem;">
                        ✈️ Route: {dep_city} ({dep_iata}) → {arr_city} ({arr_iata})
                    </div>
                    <div style="font-size:0.88rem; color:#cbd5e1; margin-bottom:0.5rem;">
                        📝 {ai.get('route_summary', '')}
                    </div>
                    <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:0.8rem; margin-top:1rem;">
                        <div style="background:rgba(255,255,255,0.04); border-radius:10px; padding:0.8rem;">
                            <div style="font-size:0.7rem; color:#9ca3af; text-transform:uppercase;">Airlines</div>
                            <div style="color:#ffa726; font-weight:600; margin-top:0.3rem;">{airlines_str}</div>
                        </div>
                        <div style="background:rgba(255,255,255,0.04); border-radius:10px; padding:0.8rem;">
                            <div style="font-size:0.7rem; color:#9ca3af; text-transform:uppercase;">Frequency</div>
                            <div style="color:#fff; font-weight:600; margin-top:0.3rem;">{ai.get('frequency', 'Daily')}</div>
                        </div>
                        <div style="background:rgba(255,255,255,0.04); border-radius:10px; padding:0.8rem;">
                            <div style="font-size:0.7rem; color:#9ca3af; text-transform:uppercase;">Duration</div>
                            <div style="color:#fff; font-weight:600; margin-top:0.3rem;">{ai.get('duration', '~1-2 hrs')}</div>
                        </div>
                        <div style="background:rgba(255,255,255,0.04); border-radius:10px; padding:0.8rem;">
                            <div style="font-size:0.7rem; color:#9ca3af; text-transform:uppercase;">Price Range</div>
                            <div style="color:#81c784; font-weight:600; margin-top:0.3rem;">{ai.get('price_range_pkr', 'Contact airline')}</div>
                        </div>
                    </div>
                    <div style="margin-top:1rem; font-size:0.8rem; color:#9ca3af; padding:0.6rem;
                                background:rgba(255,87,34,0.06); border-radius:8px; border-left:3px solid rgba(255,87,34,0.4);">
                        💡 Best time to book: {ai.get('best_booking_time', '2-4 weeks in advance')}
                    </div>
                    <div style="margin-top:0.5rem; font-size:0.75rem; color:#9ca3af; font-style:italic;">
                        ⚠️ {ai.get('disclaimer', 'AI-generated. Verify with airlines.')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning(
                f"⚠️ Could not retrieve flight data: {result.get('error', 'Unknown error')}. "
                "Please check airline websites directly."
            )

        # ── Supplementary PIA Historical Stats ────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 Historical Route Statistics (PIA Dataset)")
        hist_res = search_flights(dep_city, arr_city)
        if hist_res.get("found"):
            c1, c2, c3, c4 = st.columns(4)
            for col, icon, label, val in [
                (c1, "💰", "Avg Price", f"${hist_res['avg_price_usd']:.0f} USD"),
                (c2, "📉", "Min Price", f"${hist_res['min_price_usd']:.0f} USD"),
                (c3, "⏱️", "Avg Duration", f"{int(hist_res['avg_duration_min'])} min"),
                (c4, "✅", "On-Time %", f"{hist_res['on_time_pct']}%"),
            ]:
                with col:
                    st.markdown(
                        f"""<div class="glass-card" style="text-align:center; padding:1rem;">
                            <div style="font-size:1.4rem;">{icon}</div>
                            <div style="font-size:1.2rem; font-weight:700; color:#ff6f00;">{val}</div>
                            <div style="font-size:0.75rem; color:#9ca3af;">{label}</div>
                            <div style="font-size:0.7rem; color:#4a5568; margin-top:0.2rem;">from {hist_res['sample_count']} PIA records</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
            if hist_res.get("aircraft_types"):
                st.markdown(
                    "**Aircraft Types:** " + " ".join(f'<span class="stat-chip">{ac}</span>' for ac in hist_res["aircraft_types"]),
                    unsafe_allow_html=True,
                )
        else:
            st.info("No historical PIA data for this specific route.")

        # ── Airport Info ──────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🏢 Airport Information")
        col_dep_info, col_arr_info = st.columns(2)
        for col, city_name in [(col_dep_info, dep_city), (col_arr_info, arr_city)]:
            with col:
                ap = AIRPORTS.get(city_name, {})
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div style="font-size:1.1rem; font-weight:700; color:#ff6f00;">{ap.get('name', city_name)}</div>
                        <div style="font-size:0.85rem; color:#ffa726; margin-top:0.3rem;">
                            🛬 IATA: <strong>{ap.get('code', '???')}</strong> | 📍 {ap.get('province', '')}
                        </div>
                        <a href="{ap.get('schedule_url', 'https://paa.gov.pk')}" target="_blank"
                           style="display:inline-block; margin-top:0.8rem; background:rgba(255,87,34,0.1);
                                  border:1px solid rgba(255,87,34,0.3); color:#ffa726;
                                  border-radius:8px; padding:0.3rem 0.8rem; font-size:0.8rem; text-decoration:none;">
                            📅 Live Schedule ↗
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── Booking Links ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🎫 Book Your Flight")
        c_airline, c_booking = st.columns(2)
        with c_airline:
            st.markdown("#### ✈️ Airlines")
            for al in AIRLINES:
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; align-items:center;
                                background:rgba(255,255,255,0.02); padding:0.8rem 1.2rem;
                                border-radius:10px; margin-bottom:0.5rem; border:1px solid rgba(255,255,255,0.05);">
                        <div>
                            <span style="font-size:1.1rem; margin-right:0.5rem;">{al['icon']}</span>
                            <span style="font-weight:600; color:#fff;">{al['name']}</span>
                            <div style="font-size:0.75rem; color:#9ca3af;">{al['routes']}</div>
                        </div>
                        <a href="{al['url']}" target="_blank">
                            <button style="background:#ff5722; border:none; color:white; border-radius:6px;
                                           padding:0.3rem 0.8rem; font-size:0.8rem; cursor:pointer;">Book ↗</button>
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with c_booking:
            st.markdown("#### 🌐 Compare Platforms")
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
                                           color:#fff; border-radius:6px; padding:0.3rem 0.8rem; font-size:0.8rem; cursor:pointer;">Compare ↗</button>
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        with st.expander("📊 All Domestic Routes in PIA Dataset", expanded=False):
            routes = get_all_domestic_routes()
            if routes:
                route_df = pd.DataFrame(routes)
                route_df.columns = ["Departure", "Arrival", "Dep Code", "Arr Code",
                                    "Avg Price (USD)", "Min Price (USD)", "Avg Duration (min)", "Sample Flights"]
                st.dataframe(route_df, use_container_width=True, hide_index=True)
            else:
                st.info("No domestic routes found in dataset.")

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

    # Load destinations from new data structure only
    destination_names = [
        "Hunza", "Skardu", "Swat", "Murree", "Naran", "Kaghan", "Fairy Meadows",
        "Lahore", "Karachi", "Islamabad", "Peshawar", "Quetta", "Multan",
        "Chitral", "Gilgit", "Neelum Valley", "Kumrat Valley", "Shogran",
        "Malam Jabba", "Kalash Valley", "Deosai Plains", "Naltar Valley"
    ]

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
        
        # Progressive Disclosure - Show recommendations first
        st.markdown("### 🎯 Quick Recommendations")
        
        # Create recommendation cards
        col_rec_a, col_rec_b = st.columns(2)
        
        # Simple recommendations based on destination characteristics
        recommendations = {
            "Hunza": {"type": "Best for Adventure", "budget": "Mid-Range", "season": "Apr-Oct", "highlight": "Mountain views & culture"},
            "Skardu": {"type": "Best for Trekking", "budget": "Mid-Range", "season": "May-Sep", "highlight": "K2 base & lakes"},
            "Swat": {"type": "Best for Families", "budget": "Budget-Friendly", "season": "Mar-Nov", "highlight": "Green valleys & waterfalls"},
            "Murree": {"type": "Best for Quick Getaway", "budget": "Budget-Friendly", "season": "Year-round", "highlight": "Hill station & accessibility"},
            "Lahore": {"type": "Best for Culture", "budget": "Budget-Friendly", "season": "Oct-Mar", "highlight": "History & cuisine"},
            "Karachi": {"type": "Best for Business", "budget": "All Levels", "season": "Nov-Feb", "highlight": "Beaches & urban life"},
            "Islamabad": {"type": "Best Overall", "budget": "Mid-Range", "season": "Year-round", "highlight": "Modern capital & nature"}
        }
        
        rec_a = recommendations.get(dest_a, {"type": "Great Choice", "budget": "Mid-Range", "season": "Check locally", "highlight": "Unique experiences"})
        rec_b = recommendations.get(dest_b, {"type": "Great Choice", "budget": "Mid-Range", "season": "Check locally", "highlight": "Unique experiences"})
        
        with col_rec_a:
            st.markdown(
                f"""
                <div class="glass-card" style="border-left: 4px solid #4CAF50;">
                    <div style="font-size: 1.2rem; font-weight: 700; color: #4CAF50; margin-bottom: 0.5rem;">
                        🏆 {dest_a}
                    </div>
                    <div style="color: #fff; font-size: 0.9rem; margin-bottom: 0.3rem;">
                        <strong>{rec_a['type']}</strong>
                    </div>
                    <div style="color: #cbd5e1; font-size: 0.8rem;">
                        💰 Budget: {rec_a['budget']}<br>
                        📅 Season: {rec_a['season']}<br>
                        ✨ Highlight: {rec_a['highlight']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col_rec_b:
            st.markdown(
                f"""
                <div class="glass-card" style="border-left: 4px solid #FF9800;">
                    <div style="font-size: 1.2rem; font-weight: 700; color: #FF9800; margin-bottom: 0.5rem;">
                        🏆 {dest_b}
                    </div>
                    <div style="color: #fff; font-size: 0.9rem; margin-bottom: 0.3rem;">
                        <strong>{rec_b['type']}</strong>
                    </div>
                    <div style="color: #cbd5e1; font-size: 0.8rem;">
                        💰 Budget: {rec_b['budget']}<br>
                        📅 Season: {rec_b['season']}<br>
                        ✨ Highlight: {rec_b['highlight']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Comparison Matrix
        st.markdown("### 📊 Quick Comparison")
        
        comparison_data = {
            "Factor": ["Budget Level", "Safety", "Weather", "Activities", "Accessibility"],
            dest_a: ["Mid-Range", "⭐⭐⭐⭐", "Seasonal", "Adventure", "Moderate"],
            dest_b: ["Mid-Range", "⭐⭐⭐⭐", "Seasonal", "Mixed", "Good"]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.table(comparison_df.set_index("Factor"))

        # Attribution (no CSV filename exposure)
        st.caption("📊 *Comparison based on available travel records in our Pakistan tourism database.*")


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
        "wz_transport_class": "Private Van",
        "wz_budget": 50000,
        "wz_plan_text": "",
        "wz_plan_chunks": [],
        "wz_chat_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _render_progress(step: int, total: int = 10):
    pct = int((step / total) * 100)
    labels = [
        "Welcome", "Destination", "Group", "Dates", "Style",
        "Interests", "Stay", "Food", "Transport", "Review", "Done"
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

    # Progress bar for steps 1–10
    _render_progress(step, 10)

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
        # Accommodation restrictions: Only Hotels and Guest Houses
        accom_opts = ["Hotel", "Guest House"]
        accom = st.radio(
            "Accommodation Type (Only Hotels & Guest Houses available)", accom_opts,
            index=accom_opts.index(st.session_state.wz_accom_type) if st.session_state.wz_accom_type in accom_opts else 0,
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
                ["Economy Class", "Business Class"],
                index=["Economy Class", "Business Class"].index(
                    st.session_state.wz_transport_class
                ) if st.session_state.wz_transport_class in ["Economy Class", "Business Class"] else 0,
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
            road_opts = ["Private Van", "Bus"]
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
            if st.button("Next → Review", use_container_width=True, key="wz8_next"):
                st.session_state.wz_step = 9; st.rerun()

    # ── Step 9: Review Summary ──────────────────────────────────────────────
    elif step == 9:
        st.markdown(
            '<div class="wizard-step-card">'
            '<div class="wizard-step-title">📋 Review your trip details</div>'
            '<div class="wizard-step-desc">Everything look good? Hit Generate to let the AI build your perfect itinerary!</div>',
            unsafe_allow_html=True,
        )
        wz = st.session_state
        interests_display = ", ".join(wz.wz_interests) if wz.wz_interests else "Not specified"
        food_display = ", ".join(wz.wz_food_prefs) if wz.wz_food_prefs else "No preference"

        summary_items = [
            ("📍 Destination", wz.wz_destination or "—"),
            ("🛫 Departure City", wz.wz_departure or "—"),
            ("👥 Travel Group", f"{wz.wz_group_type} ({wz.wz_num_travelers} {'person' if wz.wz_num_travelers == 1 else 'people'})"),
            ("🗓️ Duration", f"{wz.wz_duration} days"),
            ("🎨 Travel Style", wz.wz_travel_style),
            ("🏨 Accommodation", f"{wz.wz_accom_type} — {wz.wz_room_type}"),
            ("🚗 Transport", f"{wz.wz_transport_type} ({wz.wz_transport_class})"),
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
                st.session_state.wz_step = 8; st.rerun()
        with col_gen:
            if st.button("🤖 Generate AI Trip Plan ✨", use_container_width=True, key="wz9_generate"):
                st.session_state.wz_step = 10; st.rerun()

    # ── Step 10: Generate + Final Plan ──────────────────────────────────────
    elif step == 10:
        wz = st.session_state

        # Generate plan if not already done
        if not wz.wz_plan_text:
            # Show progress at 100% during generation
            st.markdown(
                """
                <div class="wizard-progress-wrap">
                    <span class="wizard-step-label">Generating Trip Plan</span>
                    <div class="wizard-bar-bg">
                        <div class="wizard-bar-fill" style="width: 100%;"></div>
                    </div>
                    <span class="wizard-step-count">Step 10 of 10</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
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
                        budget_pkr=100000,  # Default budget
                        split_budget=False  # Default split budget setting
                    )
                    st.session_state.wz_plan_text = plan_text
                    st.session_state.wz_plan_chunks = context.chunks if context else []
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating plan: {e}")
                    if st.button("← Go Back", key="wz10_err_back"):
                        st.session_state.wz_step = 9; st.rerun()
            return

        # Show the completed plan without progress bar - step 10 is complete
        # NO PROGRESS BAR HERE - plan generation is complete
        style_score = 10.0 if wz.wz_travel_style in ["Mid-Level", "Luxury"] else 9.0
        interest_score = min(10.0, 7.0 + len(wz.wz_interests))
        trip_score = round((style_score * 0.6 + interest_score * 0.4), 1)
        
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

        # Progressive Disclosure Implementation
        if not hasattr(st.session_state, 'show_full_itinerary'):
            st.session_state.show_full_itinerary = False
        
        # Extract summary from plan text (first section only)
        plan_lines = wz.wz_plan_text.split('\n')
        summary_lines = []
        in_overview = False
        
        for line in plan_lines:
            if '## ✈️ Trip Overview' in line or '## Trip Overview' in line:
                in_overview = True
                summary_lines.append(line)
                continue
            if in_overview:
                if line.startswith('## ') and 'Overview' not in line:
                    break
                summary_lines.append(line)
        
        summary_text = '\n'.join(summary_lines) if summary_lines else wz.wz_plan_text[:500] + "..."
        
        # Show summary first with expand option
        with st.container():
            st.markdown('<div class="glass-card-accent">', unsafe_allow_html=True)
            
            if not st.session_state.show_full_itinerary:
                # Summary view
                st.markdown("### 📋 Trip Summary")
                st.markdown(summary_text)
                
                # Extract key highlights
                st.markdown("### 🌟 Key Highlights")
                highlights = [
                    f"🎯 **Destination**: {wz.wz_destination}, Pakistan",
                    f"📅 **Duration**: {wz.wz_duration} days",
                    f"👥 **Group**: {wz.wz_group_type} ({wz.wz_num_travelers} travelers)",
                    f"🎨 **Style**: {wz.wz_travel_style}"
                ]
                
                for highlight in highlights:
                    st.markdown(highlight)
                
                # Major activities preview
                st.markdown("### 🎪 Major Activities")
                interests_text = ", ".join(wz.wz_interests) if wz.wz_interests else "General sightseeing and cultural exploration"
                st.markdown(f"• **Interests**: {interests_text}")
                st.markdown(f"• **Accommodation**: {wz.wz_accom_type}")
                st.markdown(f"• **Transport**: {wz.wz_transport_type}")
                
                # Call to action
                st.markdown("---")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("**Want the complete travel document with detailed itinerary?**")
                with col2:
                    if st.button("📄 Get Full Travel Document", use_container_width=True, type="primary"):
                        # Ask for email
                        st.session_state.show_email_form = True
                        st.rerun()
                
                # Email collection form
                if hasattr(st.session_state, 'show_email_form') and st.session_state.show_email_form:
                    st.markdown("---")
                    st.markdown("### 📧 Email Delivery")
                    
                    email_input = st.text_input("Enter your email to receive the complete travel document:", 
                                              placeholder="your-email@example.com")
                    
                    col_send, col_preview = st.columns(2)
                    with col_send:
                        if st.button("📨 Send to Email", disabled=not email_input):
                            # Here we would integrate with email service
                            with st.spinner("Sending travel document..."):
                                # Simulate email sending
                                import time
                                time.sleep(2)
                                st.success(f"✅ Complete travel document sent to {email_input}")
                                st.session_state.show_email_form = False
                                st.balloons()
                    
                    with col_preview:
                        if st.button("👁️ Preview Full Document"):
                            st.session_state.show_full_itinerary = True
                            st.session_state.show_email_form = False
                            st.rerun()
            
            else:
                # Full itinerary view
                st.markdown("### 📖 Complete Travel Document")
                st.markdown(wz.wz_plan_text)
                
                # Option to collapse back to summary
                if st.button("📄 Back to Summary", key="collapse_itinerary"):
                    st.session_state.show_full_itinerary = False
                    st.rerun()
            
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

        # ── Real accommodation listings from new data structure ──
        if wz.wz_destination:
            st.markdown(f"### 🏨 Available Accommodation in {wz.wz_destination}")
            
            # Load only Hotels and Guest Houses from new data structure
            hotel_list = []
            guesthouse_list = []
            
            try:
                # Load hotels from new data structure
                hotels_df = pd.read_csv(ROOT / "data" / "new data" / "Hotels data" / "hotels.csv")
                if 'city' in hotels_df.columns:
                    hotels_df['city_clean'] = hotels_df['city'].str.strip().str.lower()
                    hotel_matches = hotels_df[hotels_df['city_clean'] == wz.wz_destination.lower()]
                    if not hotel_matches.empty:
                        hotel_list = hotel_matches.head(2).to_dict("records")
            except Exception as e:
                logger.info(f"Hotels data not available: {e}")

            try:
                # Load guest houses from new data structure
                gh_df = pd.read_csv(ROOT / "data" / "new data" / "Guest houses data" / "guest_houses.csv") 
                if 'city' in gh_df.columns:
                    gh_df['city_clean'] = gh_df['city'].str.strip().str.lower()
                    gh_matches = gh_df[gh_df['city_clean'] == wz.wz_destination.lower()]
                    if not gh_matches.empty:
                        guesthouse_list = gh_matches.head(2).to_dict("records")
            except Exception as e:
                logger.info(f"Guest houses data not available: {e}")

            if hotel_list or guesthouse_list:
                cols = st.columns(max(len(hotel_list) + len(guesthouse_list), 1))
                col_idx = 0
                
                # Show Hotels first
                for hotel in hotel_list:
                    with cols[col_idx]:
                        st.markdown(
                            f"""
                            <div class="glass-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                                <div>
                                    <div style="font-size:0.7rem; color:#ffab00; text-transform:uppercase; font-weight:600;">🏨 Hotel</div>
                                    <div style="font-size:1rem; font-weight:700; color:#fff; margin-top:0.2rem; min-height:2.4rem;">{hotel.get('name', 'Hotel Name')}</div>
                                    <div style="font-size:0.8rem; color:#cbd5e1; margin-top:0.4rem;">
                                        📍 {str(hotel.get('address', 'Address available'))[:60]}...
                                    </div>
                                    <div style="font-size:0.85rem; color:#ffa726; margin-top:0.4rem; font-weight:600;">
                                        Rating: {hotel.get('rating', 'N/A')} ⭐ | {hotel.get('price_range', 'PKR 5,000-15,000')}
                                    </div>
                                </div>
                                <div style="margin-top:1rem; border-top:1px solid rgba(255,255,255,0.05); padding-top:0.5rem; font-size:0.75rem; color:#9ca3af;">
                                    📞 {hotel.get('phone', 'Contact hotel directly')}<br>
                                    📧 {hotel.get('email', 'Email available on request')}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        col_idx += 1
                        
                # Show Guest Houses
                for gh in guesthouse_list:
                    with cols[col_idx]:
                        st.markdown(
                            f"""
                            <div class="glass-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                                <div>
                                    <div style="font-size:0.7rem; color:#ffab00; text-transform:uppercase; font-weight:600;">🏡 Guest House</div>
                                    <div style="font-size:1rem; font-weight:700; color:#fff; margin-top:0.2rem; min-height:2.4rem;">{gh.get('name', 'Guest House Name')}</div>
                                    <div style="font-size:0.8rem; color:#cbd5e1; margin-top:0.4rem;">
                                        📍 {str(gh.get('address', 'Address available'))[:60]}...
                                    </div>
                                    <div style="font-size:0.85rem; color:#ffa726; margin-top:0.4rem; font-weight:600;">
                                        Rating: {gh.get('rating', 'N/A')} ⭐ | {gh.get('price_range', 'PKR 3,000-8,000')}
                                    </div>
                                </div>
                                <div style="margin-top:1rem; border-top:1px solid rgba(255,255,255,0.05); padding-top:0.5rem; font-size:0.75rem; color:#9ca3af;">
                                    📞 {gh.get('phone', 'Contact guest house directly')}<br>
                                    📧 {gh.get('email', 'Email available on request')}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        col_idx += 1
            else:
                st.info(f"📍 Accommodation data for {wz.wz_destination} is being updated. Please check with local tourism offices for current options.")
                
            # Source attribution (no CSV filename exposure)
            st.caption("📊 *Accommodation options based on available travel records in our Pakistan tourism database.*")

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
            st.markdown("🍽️ **Discover local culinary experiences and traditional Pakistani cuisine during your visit.**")

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
            st.markdown("🎯 **Explore amazing activities and experiences that await you during your trip.**")

        # ── PDF Download ──────────────────────────────────────────────────────
        def create_trip_pdf_bytes():
            """Generate a well-formatted PDF trip plan using fpdf2."""
            try:
                from fpdf import FPDF, XPos, YPos
                import textwrap

                def clean_pdf_text(text: str) -> str:
                    if not text:
                        return ""
                    # Replace common unicode punctuation/symbols with safe ASCII equivalents
                    replacements = {
                        "\u2013": "-",  # en dash
                        "\u2014": "-",  # em dash
                        "\u2018": "'",  # left single quote
                        "\u2019": "'",  # right single quote
                        "\u201c": '"',  # left double quote
                        "\u201d": '"',  # right double quote
                        "\u2022": "*",  # bullet point
                        "\u2026": "...",  # ellipsis
                        "✈": "[Flight]",
                        "🏨": "[Hotel]",
                        "🚗": "[Transport]",
                        "⚖": "[Compare]",
                        "🧳": "[Trip]",
                        "🏠": "[Home]",
                        "📍": "[Location]",
                        "👤": "[User]",
                        "🤖": "[AI]",
                        "🟢": "[Live]",
                        "⚠️": "[Warning]",
                        "💡": "[Tip]",
                        "🕐": "[Time]",
                        "💺": "[Seat]",
                        "🔍": "[Search]",
                        "🛰️": "[Satellite]",
                        "🛰": "[Satellite]",
                        "⬇️": "[Download]",
                        "⬇": "[Download]",
                        "📄": "[Document]",
                        "🎯": "[Target]",
                        "⭐": "[Star]",
                        "🌟": "[Star]",
                        "✓": "[Yes]",
                        "✔": "[Yes]",
                        "❌": "[No]",
                        "—": "-",
                        "•": "-",
                    }
                    for k, v in replacements.items():
                        text = text.replace(k, v)
                    # Encode to latin-1 and ignore any leftover unsupported characters, then decode back to string
                    try:
                        return text.encode("latin-1", "ignore").decode("latin-1")
                    except Exception:
                        return "".join(c for c in text if ord(c) < 256)

                class TripPDF(FPDF):
                    def header(self):
                        self.set_fill_color(18, 24, 38)
                        self.rect(0, 0, 210, 30, 'F')
                        self.set_font("Helvetica", "B", 16)
                        self.set_text_color(255, 111, 0)
                        self.set_xy(10, 8)
                        self.cell(190, 10, "Pakistan Travel Intelligence", align="L")
                        self.set_font("Helvetica", "", 9)
                        self.set_text_color(180, 180, 180)
                        self.set_xy(10, 18)
                        self.cell(190, 6, "AI Trip Planner - Powered by Travel RAG System", align="L")
                        self.set_xy(0, 30)

                    def footer(self):
                        self.set_y(-18)
                        self.set_fill_color(18, 24, 38)
                        self.rect(0, self.get_y(), 210, 20, 'F')
                        self.set_font("Helvetica", "I", 8)
                        self.set_text_color(120, 120, 120)
                        self.cell(0, 10, f"Page {self.page_no()} | Visit Pakistan - Explore Beauty - Create Memories", align="C")

                pdf = TripPDF()
                pdf.set_auto_page_break(auto=True, margin=20)
                pdf.add_page()
                pdf.set_margins(15, 35, 15)

                # ── Title Section ──
                pdf.set_font("Helvetica", "B", 22)
                pdf.set_text_color(255, 111, 0)
                pdf.ln(4)
                pdf.cell(0, 12, clean_pdf_text(f"{wz.wz_destination}, Pakistan"), align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

                pdf.set_font("Helvetica", "", 12)
                pdf.set_text_color(80, 80, 80)
                pdf.cell(0, 7, clean_pdf_text(f"{wz.wz_duration}-Day Travel Itinerary"), align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(4)

                # ── Orange separator line ──
                pdf.set_draw_color(255, 87, 34)
                pdf.set_line_width(0.8)
                pdf.line(15, pdf.get_y(), 195, pdf.get_y())
                pdf.ln(6)

                # ── Trip Overview Table ──
                pdf.set_font("Helvetica", "B", 13)
                pdf.set_text_color(40, 40, 40)
                pdf.cell(0, 8, "Trip Overview", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(2)

                overview_rows = [
                    ("Destination", f"{wz.wz_destination}, Pakistan"),
                    ("Duration", f"{wz.wz_duration} Days"),
                    ("Group Type", f"{wz.wz_group_type} ({wz.wz_num_travelers} traveler(s))"),
                    ("Travel Style", wz.wz_travel_style),
                    ("Accommodation", f"{wz.wz_accom_type} - {wz.wz_room_type}"),
                    ("Transport", f"{wz.wz_transport_type} ({wz.wz_transport_class})"),
                    ("Interests", ", ".join(wz.wz_interests) if wz.wz_interests else "General sightseeing"),
                    ("Food Preferences", ", ".join(wz.wz_food_prefs) if wz.wz_food_prefs else "No preference"),
                ]

                for i, (key, value) in enumerate(overview_rows):
                    if i % 2 == 0:
                        pdf.set_fill_color(248, 249, 250)
                    else:
                        pdf.set_fill_color(255, 255, 255)
                    pdf.set_font("Helvetica", "B", 10)
                    pdf.set_text_color(60, 60, 60)
                    pdf.cell(52, 7, clean_pdf_text(f"  {key}"), border=1, fill=True)
                    pdf.set_font("Helvetica", "", 10)
                    pdf.set_text_color(30, 30, 30)
                    # Truncate long values
                    v = str(value)
                    if len(v) > 70:
                        v = v[:68] + "..."
                    pdf.cell(128, 7, clean_pdf_text(f"  {v}"), border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

                pdf.ln(8)

                # ── Itinerary Section ──
                pdf.set_line_width(0.8)
                pdf.set_draw_color(255, 87, 34)
                pdf.line(15, pdf.get_y(), 195, pdf.get_y())
                pdf.ln(5)

                pdf.set_font("Helvetica", "B", 13)
                pdf.set_text_color(40, 40, 40)
                pdf.cell(0, 8, "Detailed Itinerary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(3)

                # Parse and render plan text
                plan_lines = wz.wz_plan_text.replace("\r\n", "\n").split("\n")
                for line in plan_lines:
                    stripped = line.strip()
                    if not stripped:
                        pdf.ln(2)
                        continue
                    # Heading detection: ## or # or starts with Day
                    if stripped.startswith("## ") or stripped.startswith("# "):
                        pdf.set_font("Helvetica", "B", 12)
                        pdf.set_text_color(200, 80, 0)
                        heading = stripped.lstrip("#").strip()
                        if len(heading) > 90:
                            heading = heading[:88] + "..."
                        pdf.cell(0, 7, clean_pdf_text(heading), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                        pdf.set_draw_color(255, 160, 80)
                        pdf.set_line_width(0.3)
                        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
                        pdf.ln(2)
                    elif stripped.startswith("**") and stripped.endswith("**"):
                        pdf.set_font("Helvetica", "B", 10)
                        pdf.set_text_color(40, 40, 40)
                        text = stripped.strip("*")
                        if len(text) > 100:
                            text = text[:98] + "..."
                        pdf.cell(0, 6, clean_pdf_text(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    else:
                        pdf.set_font("Helvetica", "", 9)
                        pdf.set_text_color(50, 50, 50)
                        # Clean markdown
                        clean = stripped.replace("**", "").replace("*", "").replace("###", "").replace("##", "").replace("#", "")
                        # Word wrap
                        wrapped = textwrap.wrap(clean, width=100)
                        for wline in wrapped:
                            pdf.cell(0, 5, clean_pdf_text(wline), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

                pdf.ln(6)

                # ── Travel Tips Section ──
                pdf.add_page()
                pdf.set_draw_color(255, 87, 34)
                pdf.set_line_width(0.8)
                pdf.line(15, pdf.get_y(), 195, pdf.get_y())
                pdf.ln(5)
                pdf.set_font("Helvetica", "B", 13)
                pdf.set_text_color(40, 40, 40)
                pdf.cell(0, 8, "Travel Tips & Important Information", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(3)

                tips = [
                    "Always carry valid identification documents (CNIC or Passport)",
                    "Respect local customs and dress modestly in religious sites",
                    "Try authentic Pakistani cuisine at local restaurants and dhabas",
                    "Keep emergency contacts saved on your phone",
                    "Check weather forecasts before planning outdoor activities",
                    "Book accommodations in advance, especially during peak season (Jul-Sep)",
                    "Exchange currency only at authorized banks or exchange centers",
                    "Stay hydrated and carry essentials for day trips in northern areas",
                    "Download offline maps - some remote areas have limited connectivity",
                    "Join guided tours for remote trekking areas for safety",
                ]
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(50, 50, 50)
                for tip in tips:
                    pdf.cell(6, 6, "-")
                    pdf.cell(0, 6, clean_pdf_text(tip), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

                pdf.ln(8)

                # ── Emergency Contacts ──
                pdf.set_draw_color(220, 50, 50)
                pdf.set_line_width(0.8)
                pdf.line(15, pdf.get_y(), 195, pdf.get_y())
                pdf.ln(5)
                pdf.set_font("Helvetica", "B", 13)
                pdf.set_text_color(180, 30, 30)
                pdf.cell(0, 8, "Emergency Contacts - Pakistan", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(3)

                contacts = [
                    ("Police Emergency", "15"),
                    ("Rescue Services", "1122"),
                    ("Tourist Helpline", "1422 / 051-9215538"),
                    ("Fire Brigade", "16"),
                    ("Edhi Foundation", "115"),
                    ("Chippa Rescue", "1020"),
                    ("PTDC Tourism", "+92-51-9204766"),
                ]
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(40, 40, 40)
                for name, number in contacts:
                    pdf.set_font("Helvetica", "B", 10)
                    pdf.cell(80, 7, clean_pdf_text(f"  {name}"), border=1, fill=False)
                    pdf.set_font("Helvetica", "", 10)
                    pdf.set_text_color(200, 60, 0)
                    pdf.cell(100, 7, clean_pdf_text(f"  {number}"), border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    pdf.set_text_color(40, 40, 40)

                pdf.ln(8)

                # ── Footer note ──
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(120, 120, 120)
                pdf.cell(0, 6, clean_pdf_text("Generated by Pakistan Travel Intelligence RAG System | For information only - verify details locally."), align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

                return bytes(pdf.output())
            except Exception as e:
                # Log actual generation error
                logging.getLogger("travel_rag").error(f"Error in PDF generation: {e}", exc_info=True)
                raise e

            except ImportError:
                # Fallback to plain text if fpdf2 not installed
                plan_text = f"""PAKISTAN TRAVEL ITINERARY
Destination: {wz.wz_destination}, Pakistan
Duration: {wz.wz_duration} Days | Group: {wz.wz_group_type} ({wz.wz_num_travelers} travelers)
Style: {wz.wz_travel_style} | Stay: {wz.wz_accom_type} | Transport: {wz.wz_transport_type}

DETAILED ITINERARY
==================
{wz.wz_plan_text}

EMERGENCY CONTACTS
Police: 15 | Rescue: 1122 | Tourist Helpline: 1422
"""
                return plan_text.encode("utf-8")

        # Generate and offer PDF download
        st.markdown("---")
        pdf_col1, pdf_col2 = st.columns([3, 1])
        with pdf_col1:
            st.markdown(
                """
                <div style="display:flex; align-items:center; gap:0.8rem;">
                    <span style="font-size:1.5rem;">📄</span>
                    <div>
                        <div style="font-weight:600; color:#fff;">Download Complete Trip Plan as PDF</div>
                        <div style="font-size:0.8rem; color:#9ca3af;">Well-formatted PDF with itinerary, tips & emergency contacts</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with pdf_col2:
            try:
                pdf_bytes = create_trip_pdf_bytes()
                st.download_button(
                    label="⬇️ Download PDF",
                    data=pdf_bytes,
                    file_name=f"{wz.wz_destination.replace(' ', '_')}_Trip_Plan.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="wz_download"
                )
            except Exception as pdf_err:
                st.error(f"PDF generation error: {pdf_err}")


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


def detect_intent(query: str) -> str:
    """
    Detect the intent of a travel assistant query.
    Returns one of:
      'accommodation', 'road_transport', 'flights',
      'destination_compare', 'trip_planner', 'destination_food',
      'out_of_scope'
    """
    q = query.lower().strip()

    # Accommodation keywords
    accommodation_kw = [
        "hotel", "guest house", "hostel", "stay", "room", "lodge",
        "accommodation", "airbnb", "resort", "motel", "inn",
        "bed", "check in", "check-in", "booking", "book a room",
        "where to stay", "place to stay", "overnight"
    ]
    if any(kw in q for kw in accommodation_kw):
        return "accommodation"

    # Road transport keywords
    road_transport_kw = [
        "bus", "coach", "road", "drive", "car", "daewoo",
        "natco", "niazi", "faisal movers", "bilal travels", "van",
        "coaster", "intercity", "route", "transport", "travel by",
        "lahore to", "karachi to", "islamabad to", "road trip",
        "highway", "motorway", "passenger"
    ]
    if any(kw in q for kw in road_transport_kw):
        return "road_transport"

    # Flight search keywords
    flight_kw = [
        "flight", "airline", "pia", "air sial", "airsial", "fly jinnah",
        "ticket", "airport", "fly", "flying", "plane", "aircraft",
        "khi", "lhe", "isb", "departure", "arrival", "booking flight",
        "book flight", "domestic flight", "air travel", "airways"
    ]
    if any(kw in q for kw in flight_kw):
        return "flights"

    # Destination compare keywords
    compare_kw = [
        "compare", "vs", "versus", "difference between", "which is better",
        "better destination", "choose between", "prefer", "both",
        "comparison", "or", "which one"
    ]
    if any(kw in q for kw in compare_kw) and any(dest in q for dest in [
        "hunza", "skardu", "swat", "murree", "lahore", "karachi",
        "islamabad", "peshawar", "quetta", "chitral", "gilgit", "naran"
    ]):
        return "destination_compare"

    # Trip planner keywords
    planner_kw = [
        "plan", "itinerary", "schedule", "trip plan", "day by day",
        "days in", "day trip", "travel plan", "plan my trip", "plan a trip",
        "7 days", "5 days", "10 days", "week in", "tour plan"
    ]
    if any(kw in q for kw in planner_kw):
        return "trip_planner"

    # Destination food keywords
    food_kw = [
        "food", "restaurant", "cuisine", "eat", "dine", "dish",
        "local food", "must eat", "must try", "street food", "biryani",
        "karahi", "tikka", "kabab", "nihari", "halwa puri", "paya",
        "breakfast", "lunch", "dinner", "cafe", "dhaba", "chai",
        "local specialty", "best food"
    ]
    if any(kw in q for kw in food_kw):
        return "destination_food"

    # Destination info (general travel queries about known Pakistan places)
    destination_kw = [
        "hunza", "skardu", "swat", "murree", "naran", "kaghan",
        "fairy meadows", "lahore", "karachi", "islamabad", "peshawar",
        "quetta", "chitral", "gilgit", "neelum", "kumrat", "shogran",
        "attabad", "k2", "deosai", "naltar", "kalash", "malam jabba",
        "bahawalpur", "multan", "faisalabad", "taxila", "ziarat"
    ]
    if any(dest in q for dest in destination_kw):
        # If asking about a destination generally (not covered above), allow it
        return "accommodation"  # Treat as general destination query

    return "out_of_scope"


def page_travel_assistant() -> None:
    st.markdown(
        """
        <div class="hero-container">
            <p class="hero-title">🤖 AI Travel Assistant</p>
            <p class="hero-subtitle">
                Your Pakistan travel expert — ask about accommodation, flights, road transport,
                destination food, comparing destinations, or planning your trip.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show supported topics
    st.markdown(
        """
        <div style="display:flex; flex-wrap:wrap; gap:0.5rem; margin-bottom:1.2rem;">
            <span style="background:rgba(255,87,34,0.1); border:1px solid rgba(255,87,34,0.3);
                         border-radius:20px; padding:0.3rem 0.8rem; font-size:0.78rem; color:#ffa726; font-weight:500;">
                🏠 Accommodation
            </span>
            <span style="background:rgba(255,87,34,0.1); border:1px solid rgba(255,87,34,0.3);
                         border-radius:20px; padding:0.3rem 0.8rem; font-size:0.78rem; color:#ffa726; font-weight:500;">
                🚗 Road Transport
            </span>
            <span style="background:rgba(255,87,34,0.1); border:1px solid rgba(255,87,34,0.3);
                         border-radius:20px; padding:0.3rem 0.8rem; font-size:0.78rem; color:#ffa726; font-weight:500;">
                ✈️ Flight Search
            </span>
            <span style="background:rgba(255,87,34,0.1); border:1px solid rgba(255,87,34,0.3);
                         border-radius:20px; padding:0.3rem 0.8rem; font-size:0.78rem; color:#ffa726; font-weight:500;">
                ⚖️ Destination Compare
            </span>
            <span style="background:rgba(255,87,34,0.1); border:1px solid rgba(255,87,34,0.3);
                         border-radius:20px; padding:0.3rem 0.8rem; font-size:0.78rem; color:#ffa726; font-weight:500;">
                🧳 AI Trip Planner
            </span>
            <span style="background:rgba(255,87,34,0.1); border:1px solid rgba(255,87,34,0.3);
                         border-radius:20px; padding:0.3rem 0.8rem; font-size:0.78rem; color:#ffa726; font-weight:500;">
                🍽️ Destination Food
            </span>
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

        # Intent detection - restrict assistant to supported topics
        intent = detect_intent(query_text)
        if intent == "out_of_scope":
            st.session_state.assistant_history.append({"role": "user", "content": user_input.strip()})
            out_of_scope_msg = (
                "🤖 **I'm specialized for Pakistan travel topics only!** Here's what I can help with:\n\n"
                "- 🏠 **Accommodation** — Hotels, guest houses, where to stay\n"
                "- 🚗 **Road Transport** — Bus operators, routes, fares (Daewoo, NATCO, etc.)\n"
                "- ✈️ **Flight Search** — Domestic flights, PIA, AirSial, Fly Jinnah\n"
                "- ⚖️ **Destination Compare** — Compare two Pakistan destinations\n"
                "- 🧳 **AI Trip Planner** — Complete day-by-day travel itineraries\n"
                "- 🍽️ **Destination Food** — Local cuisine, restaurants, must-try dishes\n\n"
                "*Try: 'Best hotels in Hunza?', 'Flights from Karachi to Lahore?', "
                "'What to eat in Peshawar?', or 'Plan a 5-day trip to Swat'*"
            )
            st.session_state.assistant_history.append({"role": "assistant", "content": out_of_scope_msg})
            st.rerun()

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

        with st.spinner("🔍 Searching knowledge base & generating enhanced response..."):
            try:
                retriever = get_retriever()
                context = retriever.retrieve(query_text, top_k=10)
                
                # Generate professional response
                response = generate_answer(query_text, context, confidence_threshold=0.6)
                chunks = context.chunks if context.has_results else []

            except Exception as e:
                logger.error(f"Travel assistant error: {e}")
                response = f"⚠️ Something went wrong: {str(e)}\n\nPlease try rephrasing your question."
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
