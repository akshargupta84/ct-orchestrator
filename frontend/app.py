"""
CT Orchestrator - Streamlit Frontend

Main application entry point with navigation and shared state.
Supports two modes:
  - Demo mode (DEMO_MODE=true): Pre-scored videos, demo chat, limited queries
  - Local mode (DEMO_MODE=false): Full agent system, file processing, unlimited
"""

import streamlit as st
from pathlib import Path
import sys
import json
import os
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file (from project root)
load_dotenv(Path(__file__).parent.parent / ".env")

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="CT Orchestrator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Check if running in demo mode (HF Spaces) or local mode
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"

# Get API key for chat (if available)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Demo query limit
MAX_DEMO_QUERIES = 5

# Custom CSS
st.markdown("""
<style>
    [data-testid="stSidebarNav"] { display: none; }
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E3A5F; margin-bottom: 0; }
    .sub-header { font-size: 1.1rem; color: #4A90A4; margin-top: 0; }
    .status-pass { color: #10B981; font-weight: 600; }
    .status-fail { color: #EF4444; font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "campaigns" not in st.session_state:
        st.session_state.campaigns = []
    if "current_campaign" not in st.session_state:
        st.session_state.current_campaign = None
    if "test_plans" not in st.session_state:
        st.session_state.test_plans = {}
    if "test_results" not in st.session_state:
        st.session_state.test_results = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "rules_pdf_path" not in st.session_state:
        st.session_state.rules_pdf_path = None
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "🏠 Home"
    if "demo_videos_loaded" not in st.session_state:
        st.session_state.demo_videos_loaded = False
    if "selected_video" not in st.session_state:
        st.session_state.selected_video = None
    if "demo_query_count" not in st.session_state:
        st.session_state.demo_query_count = 0
    # Hub chat (used in demo mode)
    if "hub_chat_messages" not in st.session_state:
        st.session_state.hub_chat_messages = []
    if "hub_uploaded_videos" not in st.session_state:
        st.session_state.hub_uploaded_videos = []
    if "hub_uploaded_media_plan" not in st.session_state:
        st.session_state.hub_uploaded_media_plan = None

    # Non-demo: try to initialize agent system + persistence
    if not DEMO_MODE:
        _init_local_state()


def _init_local_state():
    """Initialize local-mode state: agent system, persistence."""
    if "persistence_loaded" not in st.session_state:
        st.session_state.persistence_loaded = False

    if "agent_hub_state" not in st.session_state:
        try:
            from agents.state import create_initial_state
            st.session_state.agent_hub_state = create_initial_state()
        except ImportError:
            st.session_state.agent_hub_state = None

    if "agent_hub_system" not in st.session_state:
        try:
            from agents.graph import get_multi_agent_system
            st.session_state.agent_hub_system = get_multi_agent_system()
        except ImportError:
            st.session_state.agent_hub_system = None

    if not st.session_state.persistence_loaded:
        _load_from_persistence()
        st.session_state.persistence_loaded = True


def _load_from_persistence():
    """Load saved data from persistence on app startup (local mode only)."""
    try:
        from services.persistence import get_persistence_service
        persistence = get_persistence_service()

        plans = persistence.list_plans(is_approved=True)
        for plan_info in plans:
            plan_data = persistence.load_plan(plan_info['plan_id'], is_approved=True)
            if plan_data:
                st.session_state.test_plans[plan_info['plan_id']] = plan_data
                campaign = plan_data.get('campaign', {})
                if campaign:
                    existing_ids = [c.get('id') for c in st.session_state.campaigns]
                    if campaign.get('id') not in existing_ids:
                        st.session_state.campaigns.append(campaign)

        results_list = persistence.list_results()
        for result_info in results_list:
            result_data = persistence.load_results(result_info['results_id'])
            if result_data:
                campaign_id = result_data.get('campaign_id', result_data.get('plan_id', ''))
                if campaign_id:
                    st.session_state.test_results[campaign_id] = result_data

        chat_messages = persistence.load_chat_history('global')
        if chat_messages:
            st.session_state.chat_history = chat_messages

        try:
            from services.prediction_model import CreativePredictionModel
            CreativePredictionModel.load_benchmarks_from_historical_data()
        except Exception:
            pass
    except ImportError:
        pass
    except Exception as e:
        print(f"Error loading from persistence: {e}")


# =============================================================================
# Demo Data
# =============================================================================

def load_demo_data():
    """Load pre-scored demo videos and results."""
    if st.session_state.demo_videos_loaded:
        return
    demo_data_path = Path(__file__).parent.parent / "demo_data"
    if demo_data_path.exists():
        features_file = demo_data_path / "prescored_features.json"
        if features_file.exists():
            with open(features_file) as f:
                st.session_state.demo_features = json.load(f)
        results_file = demo_data_path / "sample_results.json"
        if results_file.exists():
            with open(results_file) as f:
                st.session_state.demo_results = json.load(f)
        st.session_state.demo_videos_loaded = True


def get_sample_videos():
    """Return sample video data for demo mode."""
    return [
        {
            "id": "summer_hero_30s", "name": "Summer_Hero_30s", "duration": "30 sec",
            "prediction": "PASS", "confidence": "87%",
            "key_features": "Human presence, Product demo, Clear CTA",
            "diagnostics": {"attention_score": 78, "brand_recall_score": 82, "message_clarity_score": 75, "emotional_resonance_score": 71, "uniqueness_score": 68},
            "features": {"human_frame_ratio": 0.73, "logo_in_first_3_sec": True, "has_cta": True, "has_emotional_content": True, "product_visible_ratio": 0.45},
            "recommendation": "RUN - Strong creative with excellent human presence and early brand integration."
        },
        {
            "id": "brand_story_15s", "name": "Brand_Story_15s", "duration": "15 sec",
            "prediction": "PASS", "confidence": "72%",
            "key_features": "Emotional content, Logo in first 3s",
            "diagnostics": {"attention_score": 70, "brand_recall_score": 75, "message_clarity_score": 68, "emotional_resonance_score": 74, "uniqueness_score": 62},
            "features": {"human_frame_ratio": 0.55, "logo_in_first_3_sec": True, "has_cta": True, "has_emotional_content": True, "product_visible_ratio": 0.30},
            "recommendation": "RUN - Emotionally engaging short-form content. Consider A/B testing with longer format."
        },
        {
            "id": "product_focus_30s", "name": "Product_Focus_30s", "duration": "30 sec",
            "prediction": "FAIL", "confidence": "65%",
            "key_features": "Low human presence, No clear CTA",
            "diagnostics": {"attention_score": 45, "brand_recall_score": 38, "message_clarity_score": 62, "emotional_resonance_score": 41, "uniqueness_score": 55},
            "features": {"human_frame_ratio": 0.12, "logo_in_first_3_sec": False, "has_cta": False, "has_emotional_content": False, "product_visible_ratio": 0.85},
            "recommendation": "DO NOT RUN - Critical issues: low human presence, late logo, no CTA. Major revisions needed.",
            "improvements": ["Add human talent interacting with product", "Move logo to opening 3 seconds", "Add clear CTA in final 5 seconds", "Include emotional storytelling elements"]
        },
        {
            "id": "lifestyle_60s", "name": "Lifestyle_60s", "duration": "60 sec",
            "prediction": "PASS", "confidence": "79%",
            "key_features": "High engagement, Strong brand recall",
            "diagnostics": {"attention_score": 76, "brand_recall_score": 71, "message_clarity_score": 73, "emotional_resonance_score": 80, "uniqueness_score": 72},
            "features": {"human_frame_ratio": 0.68, "logo_in_first_3_sec": False, "has_cta": True, "has_emotional_content": True, "product_visible_ratio": 0.40},
            "recommendation": "RUN - Strong lifestyle creative with excellent emotional resonance. Consider earlier logo placement."
        }
    ]


# =============================================================================
# Main + Navigation
# =============================================================================

def main():
    """Main application."""
    init_session_state()

    if DEMO_MODE:
        load_demo_data()

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("## 🎬 CT Orchestrator")

        if DEMO_MODE:
            st.warning("📺 **Demo Mode** — Pre-scored videos")

        st.markdown("---")

        # Build nav options based on mode
        if DEMO_MODE:
            nav_options = [
                "🏠 Home",
                "🧠 Agent Hub",
                "📊 Results & Predictions",
            ]
        else:
            nav_options = [
                "🏠 Home",
                "🧠 Agent Hub",
                "🤖 Planning Agent",
                "📋 CT Planner",
                "🔮 Creative Scorer",
                "📊 Results",
                "💬 Insights",
                "⚙️ Admin",
            ]

        for option in nav_options:
            if st.button(
                option,
                key=f"nav_{option}",
                use_container_width=True,
                type="primary" if st.session_state.nav_page == option else "secondary"
            ):
                st.session_state.nav_page = option
                st.rerun()

        st.markdown("---")

        # Quick Stats
        st.markdown("### 📊 Stats")
        col1, col2 = st.columns(2)
        with col1:
            if DEMO_MODE:
                st.metric("Videos", 4)
            else:
                st.metric("Campaigns", len(st.session_state.campaigns))
        with col2:
            if DEMO_MODE:
                remaining = MAX_DEMO_QUERIES - st.session_state.demo_query_count
                st.metric("Queries Left", f"{remaining}/{MAX_DEMO_QUERIES}")
            else:
                st.metric("Plans", len(st.session_state.test_plans))

        st.markdown("---")
        st.markdown("""
        **🔗 Resources**
        - [GitHub Repo](https://github.com/akshargupta84/ct-orchestrator)
        - [Documentation](https://github.com/akshargupta84/ct-orchestrator#readme)
        """)
        st.caption("CT Orchestrator v1.0 | Apache 2.0")

    # Route to pages
    page = st.session_state.nav_page

    if page == "🏠 Home":
        show_home()
    elif page == "🧠 Agent Hub":
        show_agent_hub()
    elif page == "🤖 Planning Agent":
        show_planning_agent()
    elif page == "📋 CT Planner":
        show_planner()
    elif page == "🔮 Creative Scorer":
        show_creative_scorer()
    elif page in ("📊 Results", "📊 Results & Predictions"):
        show_results()
    elif page == "💬 Insights":
        show_insights()
    elif page == "⚙️ Admin":
        show_admin()


# =============================================================================
# Home Page
# =============================================================================

def show_home():
    """Home page with overview."""
    st.title("🎬 Creative Testing Orchestrator")
    st.markdown("*AI-powered multi-agent system for automating creative testing workflows in media agencies*")

    if DEMO_MODE:
        st.info(
            "**🎯 You're viewing the interactive demo** — "
            "Pre-scored sample videos let you explore all features instantly. "
            "For live scoring with your own creatives, "
            "[clone the repo](https://github.com/akshargupta84/ct-orchestrator) and run locally with Ollama."
        )

    st.markdown("---")

    # What is this?
    st.markdown("## 🎯 What is CT Orchestrator?")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **CT Orchestrator** helps media agencies predict which video ads will perform well 
        in brand lift studies — *before* spending $10K-$25K on actual testing.
        
        **The Problem:** Brand lift studies cost $10K-$25K each, ~65% of creatives fail, 
        results take 2-4 weeks, and there's no systematic learning from past tests.
        
        **Our Solution:**
        - 🎥 **Video Analysis**: Extract creative features using local AI vision models
        - 🤖 **Multi-Agent System**: Specialized agents for planning, analysis, and recommendations
        - 📊 **Predictive Modeling**: ML models trained on historical results
        - 💬 **Insights Chat**: Ask questions about your data and past learnings
        """)
    with col2:
        st.metric("Potential Annual Savings", "~$150K")
        st.metric("Reduce Failure Rate", "65% → 30%")

    st.markdown("---")

    # Quick navigation cards
    st.markdown("## 🚀 Get Started")

    if not DEMO_MODE:
        # Non-demo: show all features
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🧠 Try Agent Hub", type="primary", use_container_width=True):
                st.session_state.nav_page = "🧠 Agent Hub"
                st.rerun()
        with col_b:
            if st.button("🤖 Planning Agent (Single Agent)", use_container_width=True):
                st.session_state.nav_page = "🤖 Planning Agent"
                st.rerun()

        st.markdown("---")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown("### 📋 Planner")
            st.markdown("Form-based testing plans with rule validation.")
            if st.button("Create Plan", key="home_plan", use_container_width=True):
                st.session_state.nav_page = "📋 CT Planner"
                st.rerun()
        with col2:
            st.markdown("### 🔮 Scorer")
            st.markdown("Predict results before testing with local AI.")
            if st.button("Score Creative", key="home_scorer", use_container_width=True):
                st.session_state.nav_page = "🔮 Creative Scorer"
                st.rerun()
        with col3:
            st.markdown("### 📊 Results")
            st.markdown("Analyze performance and get recommendations.")
            if st.button("View Results", key="home_results", use_container_width=True):
                st.session_state.nav_page = "📊 Results"
                st.rerun()
        with col4:
            st.markdown("### 💬 Insights")
            st.markdown("Ask questions about data and learnings.")
            if st.button("Ask Question", key="home_insights", use_container_width=True):
                st.session_state.nav_page = "💬 Insights"
                st.rerun()
        with col5:
            st.markdown("### ⚙️ Admin")
            st.markdown("Configure rules, API keys, and settings.")
            if st.button("Settings", key="home_admin", use_container_width=True):
                st.session_state.nav_page = "⚙️ Admin"
                st.rerun()

    else:
        # Demo: streamlined navigation
        col1, col2, col3 = st.columns(3)
        with col1:
            with st.container(border=True):
                st.markdown("### 1️⃣ Agent Hub")
                st.markdown("Chat with AI agents, upload videos & media plans, generate test plans.")
                if st.button("Go to Agent Hub →", key="goto_hub"):
                    st.session_state.nav_page = "🧠 Agent Hub"
                    st.rerun()
        with col2:
            with st.container(border=True):
                st.markdown("### 2️⃣ Results & Predictions")
                st.markdown("Explore video predictions, diagnostic scores, and recommendations.")
                if st.button("Go to Results →", key="goto_results"):
                    st.session_state.nav_page = "📊 Results & Predictions"
                    st.rerun()
        with col3:
            with st.container(border=True):
                st.markdown("### 3️⃣ Chat with Agents")
                st.markdown("Ask about brand recall drivers, budget rules, and creative best practices.")
                st.caption("*5 free queries in demo*")
                if st.button("Try Chat →", key="goto_chat"):
                    st.session_state.nav_page = "🧠 Agent Hub"
                    st.rerun()

    # Sample videos (demo only)
    if DEMO_MODE:
        st.markdown("---")
        st.markdown("## 📹 Pre-Scored Sample Videos")
        sample_videos = get_sample_videos()
        cols = st.columns(4)
        for i, video in enumerate(sample_videos):
            with cols[i]:
                with st.container(border=True):
                    st.markdown(f"**🎬 {video['name']}**")
                    st.caption(video['duration'])
                    if video["prediction"] == "PASS":
                        st.success(f"{video['prediction']} ({video['confidence']})")
                    else:
                        st.error(f"{video['prediction']} ({video['confidence']})")
                    st.caption(video['key_features'])
                    if st.button("View Details", key=f"view_{video['id']}", use_container_width=True):
                        st.session_state.selected_video = video['id']
                        st.session_state.nav_page = "📊 Results & Predictions"
                        st.rerun()

    # Architecture
    st.markdown("---")
    st.markdown("## 🏗️ Architecture Overview")
    st.code("""
┌─────────────────────────────────────────────────────────────────┐
│                     FRONTEND (Streamlit)                        │
│   Home │ Agent Hub │ Planner │ Scorer │ Results │ Insights     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ORCHESTRATION LAYER (LangGraph)                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Planning │  │ Analysis │  │  Video   │  │Knowledge │       │
│  │  Agent   │  │  Agent   │  │ Analyzer │  │  Agent   │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│   SQLite │ ChromaDB (RAG) │ Video Files │ Pre-scored Cache     │
└─────────────────────────────────────────────────────────────────┘
    """, language="text")

    # Run locally (demo only)
    if DEMO_MODE:
        st.markdown("---")
        st.markdown("## 💻 Run Locally with Your Own Videos")
        with st.expander("📋 Quick Start Guide", expanded=False):
            st.code("""
# Clone the repository
git clone https://github.com/akshargupta84/ct-orchestrator.git
cd ct-orchestrator

# Create virtual environment
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements-local.txt

# Set up Ollama (in a separate terminal)
ollama serve && ollama pull llava:13b

# Configure environment
cp .env.example .env
# Edit .env: set DEMO_MODE=false and add your ANTHROPIC_API_KEY

# Run the app
streamlit run frontend/app.py
            """, language="bash")


# =============================================================================
# Agent Hub (Demo vs Local)
# =============================================================================

def show_agent_hub():
    """Agent Hub — routes to real multi-agent system (local) or demo chat."""
    if not DEMO_MODE:
        # Check if the actual multi-agent dependencies exist before routing
        try:
            import agents.graph  # noqa: F401
            import agents.state  # noqa: F401
            from pages.agent_hub import render
            render()
            return
        except ImportError:
            pass
        try:
            import agents.graph  # noqa: F401
            import agents.state  # noqa: F401
            from frontend.pages.agent_hub import render
            render()
            return
        except ImportError:
            pass
        # Dependencies not built yet — fall through to demo chat (unlimited)

    # Demo mode or fallback
    _show_demo_agent_hub()


def _show_demo_agent_hub():
    """Demo-mode Agent Hub with persistent chat, file uploads, and demo responses."""
    st.markdown("# 🧠 Agent Hub")
    st.markdown("Chat with our AI agents to analyze creatives, generate test plans, and explore CT rules.")

    # Agent architecture (collapsed)
    with st.expander("🏗️ Agent Architecture", expanded=False):
        st.code("""
┌───────────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR (LangGraph)                         │
│                 Routes requests to appropriate agents                  │
└───────────────────────────────────────────────────────────────────────┘
      │                    │                    │                    │
      ▼                    ▼                    ▼                    ▼
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│  PLANNING  │    │  ANALYSIS  │    │   VIDEO    │    │ KNOWLEDGE  │
│   AGENT    │    │   AGENT    │    │  ANALYZER  │    │   AGENT    │
│            │    │            │    │            │    │            │
│ • Rules    │    │ • Pass/Fail│    │ • Frames   │    │ • RAG      │
│ • Budget   │    │ • Scores   │    │ • LLaVA    │    │ • Q&A      │
│ • Costs    │    │ • Recs     │    │ • Features │    │ • Learnings│
└────────────┘    └────────────┘    └────────────┘    └────────────┘
        """, language="text")

    st.markdown("---")

    # File upload section
    with st.expander("📁 Upload Files (Videos & Media Plan)", expanded=False):
        up_col1, up_col2 = st.columns(2)
        with up_col1:
            st.markdown("**🎥 Video Creatives**")
            uploaded_videos = st.file_uploader(
                "Upload video files", type=["mp4", "mov", "avi"],
                accept_multiple_files=True, key="hub_video_uploader",
                help="Upload MP4/MOV/AVI files for analysis."
            )
            if uploaded_videos:
                st.session_state.hub_uploaded_videos = uploaded_videos
                st.success(f"✅ {len(uploaded_videos)} video(s) ready")
                for v in uploaded_videos:
                    st.caption(f"  • {v.name} ({v.size / 1024:.0f} KB)")
        with up_col2:
            st.markdown("**📊 Media Plan**")
            uploaded_plan = st.file_uploader(
                "Upload media plan", type=["csv", "xlsx", "xls"],
                key="hub_plan_uploader",
                help="Upload a media plan to auto-generate a CT plan."
            )
            if uploaded_plan:
                st.session_state.hub_uploaded_media_plan = uploaded_plan
                st.success(f"✅ {uploaded_plan.name} ready")
        if st.session_state.hub_uploaded_videos or st.session_state.hub_uploaded_media_plan:
            st.info("💡 Files attached! Ask the agents to generate a test plan, analyze videos, or match creatives.")

    # Chat interface
    st.markdown("## 💬 Chat with the Agents")

    # Render full chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.hub_chat_messages:
            avatar = "🧠" if msg["role"] == "assistant" else None
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

    # Query limit — right above chat input
    remaining = MAX_DEMO_QUERIES - st.session_state.demo_query_count
    chat_disabled = (remaining <= 0 and DEMO_MODE)

    if DEMO_MODE:
        if remaining <= 0:
            st.error("❌ **Query limit reached.** [Clone the repo](https://github.com/akshargupta84/ct-orchestrator) and run locally for unlimited queries.")
        elif remaining <= 2:
            st.warning(f"⚠️ **{remaining}/{MAX_DEMO_QUERIES} queries remaining** · [Run locally](https://github.com/akshargupta84/ct-orchestrator) for unlimited access")
        else:
            st.caption(f"💬 {remaining}/{MAX_DEMO_QUERIES} demo queries remaining")

    placeholder = "e.g., What drives brand recall? / Generate a test plan / What are the budget rules?"
    if chat_disabled:
        placeholder = "Query limit reached — run locally for unlimited access"

    user_query = st.chat_input(placeholder, disabled=chat_disabled)

    if user_query:
        st.session_state.hub_chat_messages.append({"role": "user", "content": user_query})
        if DEMO_MODE:
            st.session_state.demo_query_count += 1
        upload_context = _build_upload_context()
        response = _generate_demo_response(user_query, upload_context)
        st.session_state.hub_chat_messages.append({"role": "assistant", "content": response})
        st.rerun()

    # Quick action buttons (only before first message)
    if not st.session_state.hub_chat_messages:
        st.markdown("---")
        st.markdown("### 🎮 Try These")
        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            if st.button("🎬 Analyze Summer_Hero_30s", use_container_width=True, key="qa_video"):
                _send_quick_query("Analyze the Summer_Hero_30s video")
        with qc2:
            if st.button("❓ What drives brand recall?", use_container_width=True, key="qa_recall"):
                _send_quick_query("What drives brand recall?")
        with qc3:
            if st.button("📋 Show budget rules", use_container_width=True, key="qa_budget"):
                _send_quick_query("What are the budget tier rules?")

    # Clear chat
    if st.session_state.hub_chat_messages:
        st.markdown("---")
        if st.button("🗑️ Clear Chat", key="clear_hub_chat"):
            st.session_state.hub_chat_messages = []
            st.rerun()


# =============================================================================
# Demo Chat Helpers
# =============================================================================

def _send_quick_query(query: str):
    """Send a quick-action query through the demo chat."""
    st.session_state.hub_chat_messages.append({"role": "user", "content": query})
    if DEMO_MODE:
        st.session_state.demo_query_count += 1
    upload_context = _build_upload_context()
    response = _generate_demo_response(query, upload_context)
    st.session_state.hub_chat_messages.append({"role": "assistant", "content": response})
    st.rerun()


def _build_upload_context() -> dict:
    """Build context dict from any uploaded files."""
    context = {"videos": [], "media_plan": None}
    if st.session_state.hub_uploaded_videos:
        for v in st.session_state.hub_uploaded_videos:
            context["videos"].append({"name": v.name, "size_kb": v.size / 1024})
    if st.session_state.hub_uploaded_media_plan:
        try:
            import pandas as pd
            plan_file = st.session_state.hub_uploaded_media_plan
            if plan_file.name.endswith(".csv"):
                df = pd.read_csv(plan_file)
            else:
                df = pd.read_excel(plan_file)
            plan_file.seek(0)
            context["media_plan"] = {
                "filename": plan_file.name,
                "columns": list(df.columns),
                "rows": len(df),
                "preview": df.head(5).to_dict(orient="records"),
            }
        except Exception as e:
            context["media_plan"] = {"filename": st.session_state.hub_uploaded_media_plan.name, "error": str(e)}
    return context


def _generate_demo_response(query: str, upload_context: dict) -> str:
    """Generate a demo response based on query keywords, uploads, and conversation context."""
    query_lower = query.lower()

    # Check if user is asking about uploaded files / test plan generation
    has_uploads = bool(upload_context["videos"]) or upload_context["media_plan"] is not None
    if has_uploads and any(w in query_lower for w in ["plan", "test plan", "generate", "match", "analyze my", "uploaded", "videos"]):
        return _generate_demo_test_plan(upload_context)

    # Follow-up detection
    recent_context = " ".join([m["content"].lower() for m in st.session_state.hub_chat_messages[-6:]])
    is_followup = any(w in query_lower for w in [
        "why", "how", "tell me more", "explain", "what about", "can you",
        "elaborate", "details", "example", "compared", "vs", "also",
        "and what", "follow up", "more on", "expand"
    ])

    # Video analysis queries
    if "summer" in query_lower or "hero" in query_lower:
        return _resp_video("Summer_Hero_30s", "30 seconds", "73%", "2.1s ✓", True, True, "Positive", "PASS", "87%", "RUN - Strong creative with good brand integration")
    if "brand" in query_lower and "story" in query_lower:
        return _resp_video("Brand_Story_15s", "15 seconds", "55%", "1.8s ✓", False, True, "Strong positive", "PASS", "72%", "RUN - Emotionally engaging short-form content")
    if "product" in query_lower and "focus" in query_lower:
        return _resp_video("Product_Focus_30s", "30 seconds", "12% ⚠️", "18s ⚠️", True, False, "Neutral", "FAIL", "65%", "DO NOT RUN - Major revisions needed")

    # Brand recall
    if "recall" in query_lower or ("brand" in query_lower and "recall" in recent_context):
        base = (
            "**Knowledge Agent** activated (RAG search)\n\n"
            "Based on analysis of 47 historical tests, the key drivers of **brand recall** are:\n\n"
            "1. **Logo in First 3 Seconds** (r=0.52, p<0.01) — 2.1x higher brand recall\n"
            "2. **Human Presence >50%** (r=0.45, p<0.01) — prominent faces drive memorability\n"
            "3. **Product Demonstration** (r=0.38, p<0.05) — product-in-use improves recall\n\n"
            "*Sources: CT Rules v2.3, Historical Analysis*"
        )
        if is_followup and "recall" in recent_context:
            base += (
                "\n\n**Follow-up context:** The logo timing effect is non-linear — 0-3s yields the strongest recall, "
                "but even 3-5s still outperforms >5s placements. For 15s creatives, logo placement is especially "
                "critical since viewers have less time to form brand associations."
            )
        return base

    # Budget / rules
    if any(w in query_lower for w in ["budget", "cost", "rules", "tier"]):
        base = (
            "**Planning Agent** activated\n\n"
            "**Budget Tier Rules:**\n\n"
            "| Budget Range | Video Limit | Display Limit | Cost/Video | Cost/Display |\n"
            "|-------------|-------------|---------------|------------|-------------|\n"
            "| $0-5M | 2 | 5 | $5,000 | $3,000 |\n"
            "| $5M-35M | 8 | 15 | $5,000 | $3,000 |\n"
            "| $35M-100M | 15 | 30 | $5,000 | $3,000 |\n"
            "| $100M+ | 25 | 50 | $5,000 | $3,000 |\n\n"
            "**Expedited Testing:** +50% cost, -3 business days\n\n"
            "*Source: CT Rules v2.3*"
        )
        if is_followup and any(w in recent_context for w in ["budget", "cost"]):
            base += (
                "\n\n**Follow-up context:** The budget tier is determined by total annual media spend for the brand, "
                "not the individual campaign budget. A brand spending $40M annually uses the $35M-100M tier limits "
                "regardless of individual campaign size."
            )
        return base

    # Attention
    if "attention" in query_lower:
        return (
            "**Knowledge Agent** activated\n\n"
            "**Attention Score Drivers:**\n\n"
            "1. **Scene Diversity** (r=0.48) — multiple scene types maintain viewer interest\n"
            "2. **Human Eye Contact** (r=0.42) — direct camera gaze captures attention\n"
            "3. **Motion/Action** (r=0.35) — dynamic content outperforms static\n\n"
            "Creatives with attention scores >70 have 2.3x higher completion rates.\n\n"
            "*Sources: Historical Analysis, Q2 Performance Report*"
        )

    # Generic follow-up
    if is_followup and recent_context:
        return (
            "**Knowledge Agent** activated\n\n"
            "Good follow-up! Based on our conversation, here's additional context:\n\n"
            "The CT Orchestrator combines insights from CT rules, historical test results, and creative feature "
            "analysis. Could you specify which aspect you'd like me to dig deeper on?\n\n"
            "For example:\n"
            "- \"Tell me more about logo timing data\"\n"
            "- \"How does this apply to a $20M campaign?\"\n"
            "- \"Show me an example test plan\""
        )

    # Default
    return (
        "**Knowledge Agent** activated\n\n"
        "I can help you with:\n"
        "- **Video Analysis**: \"Analyze Summer_Hero_30s\" — predictions and recommendations\n"
        "- **Performance Drivers**: \"What drives brand recall?\" — key factors\n"
        "- **Budget Rules**: \"What are the budget tier rules?\" — limits and costs\n"
        "- **Test Plan Generation**: Upload videos + media plan, then ask \"Generate a test plan\"\n"
        "- **Follow-ups**: Ask follow-up questions on any topic!\n\n"
        "Try asking a specific question about creative testing!"
    )


def _resp_video(name, duration, human_pct, logo_timing, has_product, has_cta, emotion, pred, conf, rec):
    """Format a video analysis response."""
    icon = "✅" if pred == "PASS" else "❌"
    return (
        f"**Video Analyzer → Analysis Agent** pipeline for {name}\n\n"
        f"```\n"
        f"Features Extracted:\n"
        f"├── Duration: {duration}\n"
        f"├── Human Presence: {human_pct}\n"
        f"├── Logo Visibility: First appears at {logo_timing}\n"
        f"├── Product Demo: {'Yes' if has_product else 'No'}\n"
        f"├── CTA Detected: {'Yes' if has_cta else 'No ⚠️'}\n"
        f"└── Emotional Content: {emotion}\n\n"
        f"Prediction: {pred} ({conf} confidence) {icon}\n"
        f"Recommendation: {rec}\n"
        f"```"
    )


def _generate_demo_test_plan(upload_context: dict) -> str:
    """Generate a demo test plan from uploaded file context."""
    videos = upload_context.get("videos", [])
    media_plan = upload_context.get("media_plan")
    lines = ["**Planning Agent** activated — generating creative testing plan\n"]

    if videos:
        lines.append(f"**📹 {len(videos)} Video(s) Detected:**\n")
        for i, v in enumerate(videos, 1):
            lines.append(f"{i}. **{v['name']}** ({v['size_kb']:.0f} KB)")
        lines.append("")

    if media_plan and "error" not in media_plan:
        lines.append(f"**📊 Media Plan:** {media_plan['filename']} ({media_plan['rows']} line items)")
        lines.append(f"Columns: {', '.join(media_plan['columns'][:8])}\n")

    num_videos = len(videos) if videos else 3
    lines.append("---\n")
    lines.append("### 📋 Recommended Creative Testing Plan\n")
    lines.append(f"**Budget Tier:** $5M-35M (assumed) → up to **8 videos**, **15 display** assets")
    lines.append(f"**Videos for Testing:** {min(num_videos, 8)} of {num_videos} uploaded")
    lines.append(f"**Estimated Cost:** ${min(num_videos, 8) * 5000:,}")
    lines.append(f"**Timeline:** ~10 business days\n")

    lines.append("**Prioritization (by impressions):**\n")
    if videos:
        for i, v in enumerate(videos[:8], 1):
            lines.append(f"{i}. **{v['name']}** — ✅ Selected")
    else:
        lines.append("Upload videos to see prioritized creative selection.\n")

    lines.append("\n**Next Steps:**")
    lines.append("1. Confirm budget tier and campaign details")
    lines.append("2. Videos analyzed for features (human presence, logo timing, CTA, etc.)")
    lines.append("3. Ensemble model predicts pass/fail with diagnostic scores")
    lines.append("4. Final test plan generated with recommendations\n")
    lines.append("*Ask follow-up questions like \"What if our budget is $50M?\" or \"Analyze the first video\"*")
    return "\n".join(lines)


# =============================================================================
# Page Routing (non-demo pages delegate to real implementations)
# =============================================================================

def show_planning_agent():
    """Planning Agent page — conversational test planning."""
    try:
        import agents.planning_agent  # noqa: F401
        from pages.planning_agent import render
        render()
    except ImportError:
        try:
            import agents.planning_agent  # noqa: F401
            from frontend.pages.planning_agent import render
            render()
        except ImportError:
            st.markdown("# 🤖 Planning Agent")
            st.info(
                "The Planning Agent requires the multi-agent system modules (`agents.graph`, `agents.planning_agent`) "
                "which are not yet built. Use the **Agent Hub** for chat-based planning in the meantime."
            )
            if st.button("Go to Agent Hub →"):
                st.session_state.nav_page = "🧠 Agent Hub"
                st.rerun()


def show_planner():
    """CT Planner page."""
    try:
        from pages.ct_planner import render_planner
        render_planner()
    except ImportError:
        try:
            from frontend.pages.ct_planner import render_planner
            render_planner()
        except ImportError as e:
            st.markdown("# 📋 CT Planner")
            st.warning(f"CT Planner requires full dependencies: {e}")


def show_creative_scorer():
    """Creative Scorer page."""
    try:
        import services.creative_scorer  # noqa: F401
        from pages.creative_scorer import render
        render()
    except ImportError:
        try:
            import services.creative_scorer  # noqa: F401
            from frontend.pages.creative_scorer import render
            render()
        except ImportError:
            st.markdown("# 🔮 Creative Scorer")
            st.info(
                "The Creative Scorer requires `services.creative_scorer` and Ollama for local AI vision scoring. "
                "Use the **Agent Hub** to analyze pre-scored videos in the meantime."
            )
            if st.button("Go to Agent Hub →", key="scorer_to_hub"):
                st.session_state.nav_page = "🧠 Agent Hub"
                st.rerun()


def show_results():
    """Results page — demo version or full version."""
    if DEMO_MODE:
        _show_results_demo()
    else:
        try:
            from pages.results import render_results
            render_results()
        except ImportError:
            try:
                from frontend.pages.results import render_results
                render_results()
            except ImportError:
                _show_results_demo()


def show_insights():
    """Insights/Chat page."""
    try:
        from pages.insights import render_insights
        render_insights()
    except ImportError:
        try:
            from frontend.pages.insights import render_insights
            render_insights()
        except ImportError as e:
            st.markdown("# 💬 Insights Chat")
            st.warning(f"Insights Chat requires full dependencies: {e}")


def show_admin():
    """Admin page."""
    try:
        from pages.admin import render_admin
        render_admin()
    except ImportError:
        try:
            from frontend.pages.admin import render_admin
            render_admin()
        except ImportError as e:
            st.markdown("# ⚙️ Admin")
            st.warning(f"Admin panel requires full dependencies: {e}")


# =============================================================================
# Results Demo Page
# =============================================================================

def _show_results_demo():
    """Results page with pre-scored sample videos (demo mode)."""
    st.markdown("# 📊 Results & Predictions")
    st.markdown("Explore pre-scored sample videos and their predictions.")
    st.markdown("---")

    sample_videos = get_sample_videos()
    video_names = [v["name"] for v in sample_videos]

    default_index = 0
    if st.session_state.selected_video:
        for i, v in enumerate(sample_videos):
            if v["id"] == st.session_state.selected_video:
                default_index = i
                break

    selected_name = st.selectbox("Select a video to analyze:", video_names, index=default_index)
    selected = next(v for v in sample_videos if v["name"] == selected_name)

    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"### 🎬 {selected['name']}")
        st.caption(f"Duration: {selected['duration']}")
        if selected["prediction"] == "PASS":
            st.success(f"### ✅ {selected['prediction']}")
        else:
            st.error(f"### ❌ {selected['prediction']}")
        st.metric("Confidence", selected["confidence"])

    with col2:
        st.markdown("### 📈 Diagnostic Scores")
        cols = st.columns(5)
        for i, (key, value) in enumerate(selected["diagnostics"].items()):
            with cols[i]:
                label = key.replace("_score", "").replace("_", " ").title()
                st.metric(label, f"{value}/100")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔍 Extracted Features")
        for key, value in selected["features"].items():
            label = key.replace("_", " ").title()
            if isinstance(value, bool):
                st.markdown(f"- **{label}:** {'✅' if value else '❌'}")
            elif isinstance(value, float):
                st.markdown(f"- **{label}:** {value:.0%}")
            else:
                st.markdown(f"- **{label}:** {value}")
    with col2:
        st.markdown("### 💡 Recommendation")
        st.info(selected["recommendation"])
        if "improvements" in selected:
            st.markdown("**Suggested Improvements:**")
            for imp in selected["improvements"]:
                st.markdown(f"- {imp}")

    st.markdown("---")
    st.markdown("### 📋 All Videos Summary")
    import pandas as pd
    summary_data = []
    for v in sample_videos:
        summary_data.append({
            "Video": v["name"], "Duration": v["duration"],
            "Prediction": v["prediction"], "Confidence": v["confidence"],
            "Attention": v["diagnostics"]["attention_score"],
            "Brand Recall": v["diagnostics"]["brand_recall_score"],
            "Message Clarity": v["diagnostics"]["message_clarity_score"],
        })
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
