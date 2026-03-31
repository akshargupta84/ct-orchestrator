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

# Initialize structured logging
try:
    from services.logger import setup_logging, get_logger, new_request_id
    setup_logging()
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    def new_request_id(): return ""

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
    /* Indent hub sub-navigation */
    .hub-sub-nav { padding-left: 20px; border-left: 2px solid #3f3f46; margin-left: 12px; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    # Auth state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_info" not in st.session_state:
        st.session_state.user_info = None
    
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
        logger.error(f"Error loading from persistence: {e}", exc_info=True)


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

    # Auth gate — show login if not authenticated
    if not st.session_state.authenticated:
        show_login_page()
        return

    if DEMO_MODE:
        load_demo_data()

    # Get user info
    user = st.session_state.user_info

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("## 🎬 CT Orchestrator")

        # User info
        role_badge = "🔑 Admin" if user["role"] == "admin" else "👤 Viewer"
        st.markdown(f"**{user['display_name']}** · {role_badge}")
        if st.button("🚪 Logout", width="stretch"):
            _handle_logout()

        if DEMO_MODE:
            st.warning("📺 **Demo Mode** — Pre-scored videos")

        st.markdown("---")

        # Home
        if st.button(
            "🏠 Home", key="nav_home", width="stretch",
            type="primary" if st.session_state.nav_page == "🏠 Home" else "secondary"
        ):
            st.session_state.nav_page = "🏠 Home"
            st.rerun()

        # Multi-Agent Hub — clickable header + sub-pages
        if st.button(
            "🧠 Multi-Agent Hub", key="nav_hub", width="stretch",
            type="primary" if st.session_state.nav_page == "🧠 Multi-Agent Hub" else "secondary"
        ):
            st.session_state.nav_page = "🧠 Multi-Agent Hub"
            st.rerun()

        hub_pages = [
            ("　📋 Generate Test Plan", "📋 Generate Test Plan"),
            ("　📊 Optimize Creatives", "📊 Optimize Creatives"),
            ("　💡 Best Practices", "💡 Best Practices"),
            ("　🔮 Score Creatives", "🔮 Score Creatives"),
        ]

        for label, page_key in hub_pages:
            if st.button(
                label, key=f"nav_{page_key}", width="stretch",
                type="primary" if st.session_state.nav_page == page_key else "secondary"
            ):
                st.session_state.nav_page = page_key
                st.rerun()

        # Admin (admin only)
        if user.get("role") == "admin":
            st.markdown("---")
            if st.button(
                "⚙️ Admin", key="nav_admin", width="stretch",
                type="primary" if st.session_state.nav_page == "⚙️ Admin" else "secondary"
            ):
                st.session_state.nav_page = "⚙️ Admin"
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

        # Session query count
        session_queries = _get_session_query_count()
        st.caption(f"📝 {session_queries} queries this session")

        st.markdown("---")
        st.markdown("""
        **🔗 Resources**
        - [GitHub Repo](https://github.com/akshargupta84/ct-orchestrator)
        - [Documentation](https://github.com/akshargupta84/ct-orchestrator#readme)
        """)
        st.caption("CT Orchestrator v1.0 | Apache 2.0")

    # Route to pages
    page = st.session_state.nav_page
    logger.debug(f"Routing to page: {page}", extra={"page": page, "user": user.get("username", "anon")})

    try:
        if page == "🏠 Home":
            show_home()
        elif page == "🧠 Multi-Agent Hub":
            show_agent_hub()
        elif page == "📋 Generate Test Plan":
            show_agent_hub()
        elif page == "📊 Optimize Creatives":
            show_results()
        elif page == "💡 Best Practices":
            show_insights()
        elif page == "🔮 Score Creatives":
            show_creative_scorer()
        elif page == "⚙️ Admin":
            show_admin()
    except Exception as e:
        logger.error(f"Page render failed: {e}", extra={"page": page}, exc_info=True)
        try:
            from services.error_handler import user_friendly_error
            st.error(f"⚠️ {user_friendly_error(e)}")
        except ImportError:
            st.error("Something went wrong loading this page. Please try again.")


def show_login_page():
    """Render the login page."""
    from services.auth import authenticate, get_login_page_config

    config = get_login_page_config()

    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("")
        st.markdown(f"# {config['title']}")
        st.markdown(f"*{config['subtitle']}*")

        st.markdown("---")

        # Demo note
        st.info(f"🔐 {config['demo_note']}")

        # Login form
        with st.container(border=True):
            st.markdown("### Log In")
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input(
                "Password",
                type="password",
                placeholder=config["hint"],
            )

            if st.button("Log In", type="primary", width="stretch"):
                if username and password:
                    user_info = authenticate(username, password)
                    if user_info:
                        st.session_state.authenticated = True
                        st.session_state.user_info = user_info
                        _handle_login(user_info)
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                else:
                    st.warning("Please enter both username and password.")

        # Show available demo accounts
        st.markdown("**Demo Accounts:**")
        for u in config["available_users"]:
            st.caption(f"• `{u['username']}` — {u['role']}")
        st.caption(f"💡 {config['hint']}")

        st.markdown("---")
        st.caption("CT Orchestrator v1.0 · [GitHub](https://github.com/akshargupta84/ct-orchestrator)")


def _handle_login(user_info: dict):
    """Handle post-login setup: log event, load chat history."""
    new_request_id()  # Start a new request context for this session
    logger.info("User logged in", extra={"user": user_info["username"], "action": "login", "session_id": user_info["session_id"]})
    try:
        from services.usage_tracker import get_tracker
        tracker = get_tracker()
        tracker.log_login(user_info["session_id"], user_info["username"], user_info["role"])

        # Load persisted chat history for this user
        saved_messages = tracker.load_chat_history(user_info["username"], page="agent_hub")
        if saved_messages:
            st.session_state.hub_chat_messages = saved_messages
    except Exception as e:
        logger.warning(f"Usage tracker unavailable: {e}")


def _handle_logout():
    """Handle logout: log event, clear session."""
    logger.info("User logged out", extra={"user": st.session_state.user_info.get("username", "unknown"), "action": "logout"})
    try:
        from services.usage_tracker import get_tracker
        tracker = get_tracker()
        if st.session_state.user_info:
            tracker.log_logout(st.session_state.user_info["session_id"])
    except Exception:
        pass

    # Clear auth state
    st.session_state.authenticated = False
    st.session_state.user_info = None
    st.session_state.hub_chat_messages = []
    st.session_state.demo_query_count = 0
    st.session_state.nav_page = "🏠 Home"
    st.rerun()


def _get_session_query_count() -> int:
    """Get query count for the current session."""
    try:
        from services.usage_tracker import get_tracker
        user = st.session_state.user_info
        if user:
            return get_tracker().get_session_query_count(user["session_id"])
    except Exception:
        pass
    return 0


# =============================================================================
# Home Page
# =============================================================================

def show_home():
    """Home page with lifecycle visual and overview."""
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

    # What is CT Orchestrator
    st.markdown("## 🎯 What is CT Orchestrator?")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **CT Orchestrator** helps media agencies predict which video ads will perform well 
        in brand lift studies — *before* spending $10K-$25K on actual testing.
        
        It doesn't replace the creative process — **strategy, ideation, and concept development remain human-driven.**
        Instead, it augments the stages where data and AI add the most value.
        """)
    with col2:
        st.metric("Potential Annual Savings", "~$150K")
        st.metric("Reduce Failure Rate", "65% → 30%")

    st.markdown("---")

    # Lifecycle Visual — Where CT Orchestrator Fits
    st.markdown("## Where CT Orchestrator Fits")
    st.markdown("Four AI-powered components, each designed for a specific stage of the campaign and creative lifecycle.")

    import streamlit.components.v1 as components
    lifecycle_path = Path(__file__).parent / "lifecycle_map.html"
    if lifecycle_path.exists():
        lifecycle_html = lifecycle_path.read_text()
        components.html(lifecycle_html, height=1100, scrolling=False)
    else:
        st.info("Lifecycle map not found. Place `lifecycle_map.html` in the `frontend/` directory.")

    st.markdown("---")

    # Quick navigation cards aligned to lifecycle
    st.markdown("## 🚀 Get Started")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with st.container(border=True):
            st.markdown("### 📋 Generate Test Plan")
            st.markdown("Upload media plan & videos. Auto-match, validate rules, generate plan with cost & timeline.")
            if st.button("Generate Plan →", key="home_plan", width="stretch"):
                st.session_state.nav_page = "📋 Generate Test Plan"
                st.rerun()
    with col2:
        with st.container(border=True):
            st.markdown("### 📊 Optimize Creatives")
            st.markdown("Analyze test outcomes and diagnostics. AI-driven insights on what to run, cut, or revise.")
            if st.button("View Results →", key="home_results", width="stretch"):
                st.session_state.nav_page = "📊 Optimize Creatives"
                st.rerun()
    with col3:
        with st.container(border=True):
            st.markdown("### 💡 Best Practices")
            st.markdown("Chat with CT rules and past test results. Discover what creative elements drive lift.")
            if st.button("Explore →", key="home_insights", width="stretch"):
                st.session_state.nav_page = "💡 Best Practices"
                st.rerun()
    with col4:
        with st.container(border=True):
            st.markdown("### 🔮 Score Creatives")
            st.markdown("Pre-test video scoring using vision AI. Predict pass/fail before the expensive lift study.")
            if st.button("Score Video →", key="home_scorer", width="stretch"):
                st.session_state.nav_page = "🔮 Score Creatives"
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
                    if st.button("View Details", key=f"view_{video['id']}", width="stretch"):
                        st.session_state.selected_video = video['id']
                        st.session_state.nav_page = "📊 Optimize Creatives"
                        st.rerun()

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
        new_request_id()  # New request context per query
        user = st.session_state.user_info
        logger.info("Chat query received", extra={"user": user.get("username", "anon"), "action": "agent_hub_query", "query": user_query[:100]})
        st.session_state.hub_chat_messages.append({"role": "user", "content": user_query})
        if DEMO_MODE:
            st.session_state.demo_query_count += 1
        upload_context = _build_upload_context()
        response = _generate_demo_response(user_query, upload_context)
        st.session_state.hub_chat_messages.append({"role": "assistant", "content": response})
        _track_chat_exchange(user_query, response)
        logger.info("Chat response generated", extra={"user": user.get("username", "anon"), "action": "response_sent"})
        st.rerun()

    # Quick action buttons (only before first message)
    if not st.session_state.hub_chat_messages:
        st.markdown("---")
        st.markdown("### 🎮 Try These")
        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            if st.button("🎬 Analyze Summer_Hero_30s", width="stretch", key="qa_video"):
                _send_quick_query("Analyze the Summer_Hero_30s video")
        with qc2:
            if st.button("❓ What drives brand recall?", width="stretch", key="qa_recall"):
                _send_quick_query("What drives brand recall?")
        with qc3:
            if st.button("📋 Show budget rules", width="stretch", key="qa_budget"):
                _send_quick_query("What are the budget tier rules?")

    # Clear chat
    if st.session_state.hub_chat_messages:
        st.markdown("---")
        if st.button("🗑️ Clear Chat", key="clear_hub_chat"):
            _clear_tracked_chat()
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
    _track_chat_exchange(query, response)
    st.rerun()


def _track_chat_exchange(query: str, response: str):
    """Log a query and persist chat messages for the current user."""
    try:
        from services.usage_tracker import get_tracker
        user = st.session_state.user_info
        if not user:
            return
        tracker = get_tracker()
        # Log the query action
        tracker.log_query(
            session_id=user["session_id"],
            username=user["username"],
            action="agent_hub_query",
            query=query,
            response_preview=response[:200],
            page="agent_hub",
        )
        # Persist chat messages
        tracker.save_chat_message(user["username"], "user", query)
        tracker.save_chat_message(user["username"], "assistant", response)
    except Exception as e:
        logger.warning(f"Tracking error: {e}")


def _clear_tracked_chat():
    """Clear persisted chat history for the current user."""
    try:
        from services.usage_tracker import get_tracker
        user = st.session_state.user_info
        if user:
            get_tracker().clear_chat_history(user["username"])
    except Exception:
        pass


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
    """Generate a response — uses Anthropic API if available, falls back to keyword matching."""
    
    # Try API-powered response first
    api_response = _try_api_response(query, upload_context)
    if api_response:
        return api_response
    
    # Fallback: keyword matching for when no API key is available
    return _generate_keyword_response(query, upload_context)


def _try_api_response(query: str, upload_context: dict):
    """Try to generate a response using the Anthropic API. Returns None if unavailable."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or len(api_key) < 20:
        return None
    
    try:
        from anthropic import Anthropic
    except ImportError:
        return None
    
    # --- Cache check ---
    try:
        from services.cache import get_cache, hash_upload_context, select_model, get_cost_tracker
        cache = get_cache()
        cost_tracker = get_cost_tracker()
        ctx_hash = hash_upload_context(upload_context)
        
        # Only cache if this is the first message or a standalone query (not a follow-up)
        is_first_or_short_history = len(st.session_state.hub_chat_messages) <= 2
        if is_first_or_short_history:
            cached = cache.get(query, ctx_hash)
            if cached:
                logger.info("Cache hit — returning cached response", extra={"action": "cache_hit", "query": query[:50]})
                cost_tracker.record_cache_hit(500)  # Estimate
                return cached
    except ImportError:
        cache = None
        cost_tracker = None
        ctx_hash = ""
        select_model = None
    
    # Build conversation history for context
    history_messages = []
    for msg in st.session_state.hub_chat_messages[-12:]:  # Last 12 messages
        history_messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current query
    history_messages.append({"role": "user", "content": query})
    
    # Build upload context string
    upload_text = ""
    if upload_context["videos"]:
        video_names = [v["name"] for v in upload_context["videos"]]
        upload_text += f"\n\nUser has uploaded these videos: {', '.join(video_names)}"
    if upload_context["media_plan"] and "error" not in upload_context["media_plan"]:
        mp = upload_context["media_plan"]
        upload_text += f"\n\nUser has uploaded a media plan: {mp['filename']} ({mp['rows']} rows, columns: {', '.join(mp['columns'])})"
        upload_text += f"\nPreview: {mp['preview']}"
    
    system_prompt = f"""You are the CT Orchestrator, a multi-agent AI system for creative testing in media agencies. You help brands predict which video ads will perform well in brand lift studies before spending $10K-$25K on actual testing.

You have 4 specialized agents:
1. **Planning Agent** — Validates test plans against business rules, budget tiers, costs
2. **Analysis Agent** — Predicts pass/fail using an ensemble ML model (40% Logistic Regression + 60% Random Forest), generates diagnostic scores
3. **Video Analyzer** — Extracts creative features from video files using LLaVA vision AI (human presence, logo timing, CTA, emotion, etc.)
4. **Knowledge Agent** — RAG-powered Q&A over CT rules and historical test results

When responding, indicate which agent(s) are responding, e.g. "**Knowledge Agent** activated" or "**Planning Agent → Analysis Agent** pipeline".

## Pre-Scored Sample Videos in This Demo

1. **Summer_Hero_30s** (30 sec) — PASS (87% confidence)
   - Features: Human presence 73%, logo at 2.1s ✓, product demo 45%, CTA at 27s, positive emotion
   - Diagnostics: Attention 78, Brand Recall 82, Message Clarity 75, Emotional Resonance 71, Uniqueness 68
   - Recommendation: RUN — Strong creative with excellent human presence and early brand integration

2. **Brand_Story_15s** (15 sec) — PASS (72% confidence)
   - Features: Human presence 55%, logo at 1.8s ✓, no product demo, CTA at 13s, strong positive emotion
   - Diagnostics: Attention 70, Brand Recall 75, Message Clarity 68, Emotional Resonance 74, Uniqueness 62
   - Recommendation: RUN — Emotionally engaging short-form. Consider A/B with longer format

3. **Product_Focus_30s** (30 sec) — FAIL (65% confidence)
   - Features: Human presence 12% ⚠️, logo at 18s ⚠️, product demo 85%, no CTA ⚠️, neutral emotion
   - Diagnostics: Attention 45, Brand Recall 38, Message Clarity 62, Emotional Resonance 41, Uniqueness 55
   - Recommendation: DO NOT RUN — Low human presence, late logo, no CTA. Major revisions needed
   - Improvements: Add human talent, move logo to first 3s, add CTA in final 5s, add emotional storytelling

4. **Lifestyle_60s** (60 sec) — PASS (79% confidence)
   - Features: Human presence 68%, no logo in first 3s ⚠️, product demo 40%, CTA present, positive emotion
   - Diagnostics: Attention 76, Brand Recall 71, Message Clarity 73, Emotional Resonance 80, Uniqueness 72
   - Recommendation: RUN — Strong emotional resonance. Consider earlier logo placement

## Historical Data & Performance Drivers (from 47 historical tests)

**Brand Recall drivers:**
- Logo in first 3 seconds (r=0.52, p<0.01) — 2.1x higher brand recall
- Human presence >50% (r=0.45, p<0.01) — prominent faces drive memorability
- Product demonstration (r=0.38, p<0.05) — product-in-use improves recall
- Logo timing effect is non-linear: 0-3s strongest, 3-5s still good, >5s poor

**Attention Score drivers:**
- Scene diversity (r=0.48) — multiple scene types maintain interest
- Human eye contact (r=0.42) — direct camera gaze captures attention
- Motion/Action (r=0.35) — dynamic > static. Attention >70 → 2.3x completion rate

**Overall pass rate:** ~35% (17 of 47 tests passed brand lift threshold)

## Budget Tier Rules (CT Rules v2.3)

| Budget Range | Video Limit | Display Limit | Cost/Video | Cost/Display |
|-------------|-------------|---------------|------------|-------------|
| $0-5M | 2 | 5 | $5,000 | $3,000 |
| $5M-35M | 8 | 15 | $5,000 | $3,000 |
| $35M-100M | 15 | 30 | $5,000 | $3,000 |
| $100M+ | 25 | 50 | $5,000 | $3,000 |

Budget tier determined by total annual media spend for the brand, not individual campaign budget.
Expedited testing: +50% cost, -3 business days.
{upload_text}

## Response Guidelines
- Be specific, data-driven, and cite sources (CT Rules v2.3, Historical Analysis, etc.)
- Reference the actual sample video data when discussing specific creatives
- Support follow-up questions using conversation context
- Keep responses focused and concise (not overly long)
- When asked about historical results, reference the 47 historical tests and specific correlations
- When asked to generate test plans, use the budget tier rules and uploaded file context"""

    try:
        # --- Model routing: Haiku for simple queries, Sonnet for complex ---
        if select_model:
            model = select_model(query)
        else:
            model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        
        client = Anthropic(api_key=api_key)
        logger.info("Calling Anthropic API", extra={"action": "api_call", "agent": "demo_hub"})
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=history_messages,
        )
        input_tokens = getattr(response.usage, 'input_tokens', 0)
        output_tokens = getattr(response.usage, 'output_tokens', 0)
        total_tokens = input_tokens + output_tokens
        logger.info("API response received", extra={"action": "api_response", "tokens": total_tokens})
        
        response_text = response.content[0].text
        
        # --- Store in cache and track cost ---
        if cache:
            cache.put(query, ctx_hash, response_text, tokens_used=total_tokens)
        if cost_tracker:
            cost_tracker.record_call(input_tokens, output_tokens, model=model)
        
        return response_text
    except Exception as e:
        # API call failed — fall back to keyword matching
        logger.warning(f"API call failed: {e}", extra={"error_type": type(e).__name__})
        try:
            from services.error_handler import user_friendly_error
            logger.info(f"User-friendly error: {user_friendly_error(e)}")
        except ImportError:
            pass
        return None


def _generate_keyword_response(query: str, upload_context: dict) -> str:
    """Fallback keyword-based responses when no API key is available."""
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
                st.session_state.nav_page = "📋 Generate Test Plan"
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
                st.session_state.nav_page = "📋 Generate Test Plan"
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
