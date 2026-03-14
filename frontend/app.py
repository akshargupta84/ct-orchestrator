"""
CT Orchestrator - Streamlit Frontend
Main application entry point with navigation and shared state.

This version is designed for Hugging Face Spaces demo with pre-scored videos.
"""

import streamlit as st
from pathlib import Path
import sys
import json
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    /* Hide default Streamlit multipage navigation */
    [data-testid="stSidebarNav"] {
        display: none;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4A90A4;
        margin-top: 0;
    }
    .status-pass {
        color: #10B981;
        font-weight: 600;
    }
    .status-fail {
        color: #EF4444;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


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
        st.session_state.nav_page = "🏠 Welcome"
    
    if "demo_videos_loaded" not in st.session_state:
        st.session_state.demo_videos_loaded = False
    
    if "selected_video" not in st.session_state:
        st.session_state.selected_video = None
    
    if "demo_query_count" not in st.session_state:
        st.session_state.demo_query_count = 0


def load_demo_data():
    """Load pre-scored demo videos and results."""
    if st.session_state.demo_videos_loaded:
        return
    
    demo_data_path = Path(__file__).parent.parent / "demo_data"
    
    if demo_data_path.exists():
        # Load pre-scored video features
        features_file = demo_data_path / "prescored_features.json"
        if features_file.exists():
            with open(features_file) as f:
                st.session_state.demo_features = json.load(f)
        
        # Load sample results
        results_file = demo_data_path / "sample_results.json"
        if results_file.exists():
            with open(results_file) as f:
                st.session_state.demo_results = json.load(f)
        
        st.session_state.demo_videos_loaded = True


def get_sample_videos():
    """Return sample video data."""
    return [
        {
            "id": "summer_hero_30s",
            "name": "Summer_Hero_30s",
            "duration": "30 sec",
            "prediction": "PASS",
            "confidence": "87%",
            "key_features": "Human presence, Product demo, Clear CTA",
            "diagnostics": {
                "attention_score": 78,
                "brand_recall_score": 82,
                "message_clarity_score": 75,
                "emotional_resonance_score": 71,
                "uniqueness_score": 68
            },
            "features": {
                "human_frame_ratio": 0.73,
                "logo_in_first_3_sec": True,
                "has_cta": True,
                "has_emotional_content": True,
                "product_visible_ratio": 0.45
            },
            "recommendation": "RUN - Strong creative with excellent human presence and early brand integration."
        },
        {
            "id": "brand_story_15s",
            "name": "Brand_Story_15s",
            "duration": "15 sec",
            "prediction": "PASS",
            "confidence": "72%",
            "key_features": "Emotional content, Logo in first 3s",
            "diagnostics": {
                "attention_score": 70,
                "brand_recall_score": 75,
                "message_clarity_score": 68,
                "emotional_resonance_score": 74,
                "uniqueness_score": 62
            },
            "features": {
                "human_frame_ratio": 0.55,
                "logo_in_first_3_sec": True,
                "has_cta": True,
                "has_emotional_content": True,
                "product_visible_ratio": 0.30
            },
            "recommendation": "RUN - Emotionally engaging short-form content. Consider A/B testing with longer format."
        },
        {
            "id": "product_focus_30s",
            "name": "Product_Focus_30s",
            "duration": "30 sec",
            "prediction": "FAIL",
            "confidence": "65%",
            "key_features": "Low human presence, No clear CTA",
            "diagnostics": {
                "attention_score": 45,
                "brand_recall_score": 38,
                "message_clarity_score": 62,
                "emotional_resonance_score": 41,
                "uniqueness_score": 55
            },
            "features": {
                "human_frame_ratio": 0.12,
                "logo_in_first_3_sec": False,
                "has_cta": False,
                "has_emotional_content": False,
                "product_visible_ratio": 0.85
            },
            "recommendation": "DO NOT RUN - Critical issues: low human presence, late logo, no CTA. Major revisions needed.",
            "improvements": [
                "Add human talent interacting with product",
                "Move logo to opening 3 seconds",
                "Add clear CTA in final 5 seconds",
                "Include emotional storytelling elements"
            ]
        },
        {
            "id": "lifestyle_60s",
            "name": "Lifestyle_60s",
            "duration": "60 sec",
            "prediction": "PASS",
            "confidence": "79%",
            "key_features": "High engagement, Strong brand recall",
            "diagnostics": {
                "attention_score": 76,
                "brand_recall_score": 71,
                "message_clarity_score": 73,
                "emotional_resonance_score": 80,
                "uniqueness_score": 72
            },
            "features": {
                "human_frame_ratio": 0.68,
                "logo_in_first_3_sec": False,
                "has_cta": True,
                "has_emotional_content": True,
                "product_visible_ratio": 0.40
            },
            "recommendation": "RUN - Strong lifestyle creative with excellent emotional resonance. Consider earlier logo placement."
        }
    ]


def main():
    """Main application."""
    init_session_state()
    
    if DEMO_MODE:
        load_demo_data()
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("## 🎬 CT Orchestrator")
        
        if DEMO_MODE:
            st.warning("📺 **Demo Mode** - Using pre-scored videos")
        
        st.markdown("---")
        
        # Navigation
        nav_options = [
            "🏠 Welcome",
            "🤖 Multi-Agent Hub", 
            "📊 Results & Predictions",
        ]
        
        # Only show these in non-demo mode (they have import issues)
        if not DEMO_MODE:
            nav_options.extend([
                "📋 CT Planner",
                "💬 Insights Chat",
                "⚙️ Admin"
            ])
        
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
        st.markdown("### 📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Videos", 4)
        with col2:
            if DEMO_MODE:
                remaining = MAX_DEMO_QUERIES - st.session_state.demo_query_count
                st.metric("Queries Left", f"{remaining}/{MAX_DEMO_QUERIES}")
        
        st.markdown("---")
        
        # GitHub Link
        st.markdown("""
        **🔗 Resources**
        - [GitHub Repo](https://github.com/akshargupta84/ct-orchestrator)
        - [Documentation](https://github.com/akshargupta84/ct-orchestrator#readme)
        """)
        
        st.caption("CT Orchestrator v1.0 | Apache 2.0")
    
    # Main Content
    if st.session_state.nav_page == "🏠 Welcome":
        show_welcome()
    elif st.session_state.nav_page == "🤖 Multi-Agent Hub":
        show_multi_agent_hub()
    elif st.session_state.nav_page == "📊 Results & Predictions":
        show_results_demo()
    elif st.session_state.nav_page == "📋 CT Planner":
        show_planner()
    elif st.session_state.nav_page == "💬 Insights Chat":
        show_insights()
    elif st.session_state.nav_page == "⚙️ Admin":
        show_admin()


def show_welcome():
    """Welcome/intro page with project overview."""
    
    # Hero Section
    st.title("🎬 Creative Testing Orchestrator")
    st.markdown("*AI-powered multi-agent system for automating creative testing workflows in media agencies*")
    
    # Demo Mode Banner
    if DEMO_MODE:
        st.info(
            "**🎯 You're viewing the interactive demo** — "
            "This demo uses pre-scored sample videos so you can explore all features instantly. "
            "For live video scoring with your own creatives, "
            "[clone the GitHub repo](https://github.com/akshargupta84/ct-orchestrator) and run locally with Ollama."
        )
    
    st.markdown("---")
    
    # What is this?
    st.markdown("## 🎯 What is CT Orchestrator?")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **CT Orchestrator** helps media agencies predict which video ads will perform well 
        in brand lift studies — *before* spending $10K-$25K on actual testing.
        
        **The Problem:**
        - Brand lift studies cost $10K-$25K each
        - ~65% of creatives fail to show significant lift
        - Results take 2-4 weeks
        - No systematic learning from past tests
        
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
    
    # How to Use This Demo - Using native Streamlit components
    st.markdown("## 🚀 How to Explore This Demo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(border=True):
            st.markdown("### 1️⃣ Multi-Agent Hub")
            st.markdown("""
            See how our AI agents work together to orchestrate the creative testing workflow.
            - Planning Agent
            - Analysis Agent  
            - Video Analyzer
            - Knowledge Agent
            """)
            if st.button("Go to Multi-Agent Hub →", key="goto_hub"):
                st.session_state.nav_page = "🤖 Multi-Agent Hub"
                st.rerun()
    
    with col2:
        with st.container(border=True):
            st.markdown("### 2️⃣ Results & Predictions")
            st.markdown("""
            Explore pre-scored sample videos and see predictions in action.
            - Pass/Fail predictions
            - Diagnostic scores
            - Video feature analysis
            - Recommendations
            """)
            if st.button("Go to Results →", key="goto_results"):
                st.session_state.nav_page = "📊 Results & Predictions"
                st.rerun()
    
    with col3:
        with st.container(border=True):
            st.markdown("### 3️⃣ Chat with Agents")
            st.markdown("""
            Ask questions about creative testing in the Multi-Agent Hub.
            - "What drives brand recall?"
            - "Analyze this video"
            - "What are the budget rules?"
            
            *5 free queries in demo*
            """)
            if st.button("Try Chat →", key="goto_chat"):
                st.session_state.nav_page = "🤖 Multi-Agent Hub"
                st.rerun()
    
    st.markdown("---")
    
    # Sample Videos - Using native Streamlit with clickable buttons
    st.markdown("## 📹 Pre-Scored Sample Videos")
    st.markdown("""
    This demo includes **4 sample video creatives** that have been pre-analyzed. 
    Click on any video to explore its features, predictions, and recommendations.
    """)
    
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
    
    st.markdown("---")
    
    # Run Locally Section
    st.markdown("## 💻 Run Locally with Your Own Videos")
    
    st.markdown("""
    Want to analyze your own video creatives? Clone the repo and run locally with Ollama for live scoring.
    """)
    
    with st.expander("📋 Quick Start Guide", expanded=False):
        st.code("""
# Clone the repository
git clone https://github.com/akshargupta84/ct-orchestrator.git
cd ct-orchestrator

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-local.txt

# Set up Ollama (in a separate terminal)
ollama serve
ollama pull llava:13b

# Configure environment
cp .env.example .env
# Edit .env: set DEMO_MODE=false and add your ANTHROPIC_API_KEY

# Run the app
streamlit run frontend/app.py
        """, language="bash")
    
    # Architecture Preview
    st.markdown("---")
    st.markdown("## 🏗️ Architecture Overview")
    
    st.code("""
┌─────────────────────────────────────────────────────────────────┐
│                     FRONTEND (Streamlit)                        │
│   Welcome │ Multi-Agent Hub │ Results │ Insights │ Admin       │
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


def show_multi_agent_hub():
    """Multi-Agent Hub page showing agent orchestration and chat."""
    
    st.markdown("# 🤖 Multi-Agent Hub")
    st.markdown("See how our AI agents work together to orchestrate creative testing workflows.")
    
    # Demo mode warning with query limit
    if DEMO_MODE:
        remaining = MAX_DEMO_QUERIES - st.session_state.demo_query_count
        if remaining > 0:
            st.warning(
                f"⚠️ **Demo Mode:** {remaining}/{MAX_DEMO_QUERIES} free queries remaining. "
                "For unlimited access, [run locally](https://github.com/akshargupta84/ct-orchestrator) with your own API key."
            )
        else:
            st.error(
                "❌ **Query limit reached.** You've used all 5 demo queries. "
                "To continue exploring, [clone the repo](https://github.com/akshargupta84/ct-orchestrator), "
                "add your own Anthropic API key, and run locally for unlimited queries."
            )
    
    st.markdown("---")
    
    # Agent Overview
    st.markdown("## Agent Architecture")
    
    st.code("""
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER REQUEST                                     │
│            "Analyze this video" / "What drives brand recall?"           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR (LangGraph)                           │
│                   Routes requests to appropriate agents                  │
└─────────────────────────────────────────────────────────────────────────┘
        │                    │                    │                    │
        ▼                    ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   PLANNING   │    │   ANALYSIS   │    │    VIDEO     │    │  KNOWLEDGE   │
│    AGENT     │    │    AGENT     │    │   ANALYZER   │    │    AGENT     │
│              │    │              │    │              │    │              │
│ • Validates  │    │ • Pass/Fail  │    │ • Frame      │    │ • RAG over   │
│   rules      │    │   decisions  │    │   extraction │    │   rules      │
│ • Prioritize │    │ • Recomm-    │    │ • Vision AI  │    │ • Past       │
│   creatives  │    │   endations  │    │   (LLaVA)    │    │   learnings  │
│ • Cost calc  │    │ • Diagnostics│    │ • Features   │    │ • Q&A        │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
    """, language="text")
    
    st.markdown("---")
    
    # Agent Details
    st.markdown("## Agent Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("📋 Planning Agent", expanded=True):
            st.markdown("""
            **Purpose:** Validates test plans against business rules
            
            **Capabilities:**
            - Budget tier validation (determines creative limits)
            - Cost calculation ($5K/video, $3K/display)
            - Timeline estimation
            - Rule compliance checking
            
            **Input:** Campaign details, creative list, budget
            
            **Output:** Validated test plan with costs and timeline
            """)
        
        with st.expander("🎥 Video Analyzer", expanded=True):
            st.markdown("""
            **Purpose:** Extracts creative features from video files
            
            **Capabilities:**
            - Frame extraction (8 frames per video)
            - Vision AI analysis (Ollama/LLaVA locally)
            - Feature aggregation
            - Similarity detection (perceptual hashing)
            
            **Features Extracted:**
            - Human presence & screen percentage
            - Logo visibility & timing
            - Product demonstration
            - Call-to-action detection
            - Emotional content
            - Scene type diversity
            
            **Input:** Video file (MP4)
            
            **Output:** 14 numeric features for ML model
            """)
    
    with col2:
        with st.expander("📊 Analysis Agent", expanded=True):
            st.markdown("""
            **Purpose:** Predicts outcomes and generates recommendations
            
            **Capabilities:**
            - Pass/fail prediction (ensemble ML model)
            - Diagnostic score prediction (attention, brand recall, etc.)
            - Recommendation generation
            - Result interpretation
            
            **Model Architecture:**
            - Ensemble: 40% Logistic Regression + 60% Random Forest
            - 5 Ridge regressors for diagnostic scores
            - LOOCV validation (n≈50 samples)
            
            **Input:** Video features + historical data
            
            **Output:** Predictions, confidence, recommendations
            """)
        
        with st.expander("🧠 Knowledge Agent", expanded=True):
            st.markdown("""
            **Purpose:** RAG-powered Q&A over rules and learnings
            
            **Capabilities:**
            - Semantic search over CT rules
            - Historical results lookup
            - Best practices retrieval
            - Natural language answers
            
            **Data Sources:**
            - CT Rules PDF (business rules)
            - Past test results
            - Performance driver analysis
            - Session data
            
            **Input:** Natural language question
            
            **Output:** Contextual answer with citations
            """)
    
    st.markdown("---")
    
    # Interactive Chat Section
    st.markdown("## 💬 Chat with the Agents")
    
    remaining = MAX_DEMO_QUERIES - st.session_state.demo_query_count
    
    if remaining <= 0 and DEMO_MODE:
        st.error("Query limit reached. Run locally for unlimited access.")
    else:
        # Check for API key
        if not ANTHROPIC_API_KEY and DEMO_MODE:
            st.info("💡 Chat is available! Ask questions about creative testing.")
        
        # Chat input
        user_query = st.text_input(
            "Ask a question:",
            placeholder="e.g., What drives brand recall? / Analyze Summer_Hero_30s / What are the budget rules?",
            disabled=(remaining <= 0 and DEMO_MODE)
        )
        
        if st.button("Send", disabled=(remaining <= 0 and DEMO_MODE)):
            if user_query:
                st.session_state.demo_query_count += 1
                
                with st.spinner("Agents processing..."):
                    import time
                    time.sleep(1.5)
                    
                    # Generate response based on query
                    response = generate_agent_response(user_query)
                    
                st.success("**Agent Response:**")
                st.markdown(response)
                
                # Show remaining queries
                new_remaining = MAX_DEMO_QUERIES - st.session_state.demo_query_count
                if new_remaining > 0 and DEMO_MODE:
                    st.caption(f"Queries remaining: {new_remaining}/{MAX_DEMO_QUERIES}")
    
    st.markdown("---")
    
    # Example Scenarios
    st.markdown("## 🎮 Example Scenarios")
    st.markdown("Click a scenario to see a sample agent response:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎬 Analyze Summer_Hero_30s", use_container_width=True):
            st.markdown("""
            **Video Analyzer Response:**
            ```
            Video Analysis Complete: Summer_Hero_30s.mp4
            
            Features Extracted:
            ├── Duration: 30 seconds
            ├── Human Presence: 73% of frames
            ├── Logo Visibility: First appears at 2.1s ✓
            ├── Product Demo: Yes (45% of frames)
            ├── CTA Detected: Yes, at 27s
            └── Emotional Content: Positive
            ```
            
            **Analysis Agent Response:**
            ```
            Prediction: PASS (87% confidence)
            
            Diagnostic Scores:
            ├── Attention: 78/100
            ├── Brand Recall: 82/100
            ├── Message Clarity: 75/100
            └── Emotional Resonance: 71/100
            
            Recommendation: RUN - Strong creative with good brand integration
            ```
            """)
    
    with col2:
        if st.button("❓ What drives brand recall?", use_container_width=True):
            st.markdown("""
            **Knowledge Agent Response:**
            
            Based on analysis of 47 historical tests, the key drivers of brand recall are:
            
            **Top 3 Factors:**
            
            1. **Logo in First 3 Seconds** (r=0.52, p<0.01)
               - Creatives with early logo placement show 2.1x higher brand recall
               
            2. **Human Presence >50%** (r=0.45, p<0.01)
               - Videos with prominent human presence drive better recall
               
            3. **Product Demonstration** (r=0.38, p<0.05)
               - Showing product in use improves memorability
            
            **Recommendation:** Ensure logo appears within the first 3 seconds and 
            feature humans interacting with the product throughout the creative.
            
            *Sources: CT Rules v2.3, Q3 Performance Analysis*
            """)


def generate_agent_response(query: str) -> str:
    """Generate a response based on the user query."""
    query_lower = query.lower()
    
    # Check for video analysis queries
    if "summer" in query_lower or "hero" in query_lower:
        return """
**Video Analyzer** activated for Summer_Hero_30s

```
Features Extracted:
├── Duration: 30 seconds
├── Human Presence: 73% of frames
├── Logo Visibility: First appears at 2.1s ✓
├── Product Demo: Yes (45% of frames)
├── CTA Detected: Yes, at 27s
└── Emotional Content: Positive

Prediction: PASS (87% confidence)
Recommendation: RUN - Strong creative with good brand integration
```
"""
    
    elif "brand" in query_lower and "story" in query_lower:
        return """
**Video Analyzer** activated for Brand_Story_15s

```
Features Extracted:
├── Duration: 15 seconds
├── Human Presence: 55% of frames
├── Logo Visibility: First appears at 1.8s ✓
├── Emotional Content: Strong positive
└── CTA Detected: Yes, at 13s

Prediction: PASS (72% confidence)
Recommendation: RUN - Emotionally engaging short-form content
```
"""
    
    elif "product" in query_lower and "focus" in query_lower:
        return """
**Video Analyzer** activated for Product_Focus_30s

```
Features Extracted:
├── Duration: 30 seconds
├── Human Presence: 12% of frames ⚠️
├── Logo Visibility: First appears at 18s ⚠️
├── Product Demo: Yes (85% of frames)
├── CTA Detected: No ⚠️

Prediction: FAIL (65% confidence)

Issues Identified:
• Low human presence
• Late logo appearance
• No clear CTA

Recommendation: DO NOT RUN - Major revisions needed
```
"""
    
    elif "recall" in query_lower or "brand" in query_lower:
        return """
**Knowledge Agent** activated (RAG search)

Based on analysis of 47 historical tests, the key drivers of **brand recall** are:

1. **Logo in First 3 Seconds** (r=0.52, p<0.01)
   - Creatives with early logo placement show 2.1x higher brand recall
   
2. **Human Presence >50%** (r=0.45, p<0.01)
   - Videos with prominent human presence drive better recall
   
3. **Product Demonstration** (r=0.38, p<0.05)
   - Showing product in use improves memorability

*Sources: CT Rules v2.3, Historical Analysis*
"""
    
    elif "budget" in query_lower or "cost" in query_lower or "rules" in query_lower:
        return """
**Planning Agent** activated

**Budget Tier Rules:**

| Budget Range | Video Limit | Display Limit | Cost/Video | Cost/Display |
|-------------|-------------|---------------|------------|--------------|
| $0-5M | 2 | 5 | $5,000 | $3,000 |
| $5M-35M | 8 | 15 | $5,000 | $3,000 |
| $35M-100M | 15 | 30 | $5,000 | $3,000 |
| $100M+ | 25 | 50 | $5,000 | $3,000 |

**Expedited Testing:** +50% cost, -3 business days

*Source: CT Rules v2.3*
"""
    
    elif "attention" in query_lower:
        return """
**Knowledge Agent** activated

**Attention Score Drivers:**

1. **Scene Diversity** (r=0.48) - Multiple scene types maintain viewer interest
2. **Human Eye Contact** (r=0.42) - Direct camera gaze captures attention
3. **Motion/Action** (r=0.35) - Dynamic content outperforms static

Creatives with attention scores >70 have 2.3x higher completion rates.

*Sources: Historical Analysis, Q2 Performance Report*
"""
    
    else:
        return """
**Knowledge Agent** activated

I can help you with:
- **Video Analysis**: "Analyze [video name]" - Get predictions and recommendations
- **Performance Drivers**: "What drives [metric]?" - Understand key factors
- **Budget Rules**: "What are the budget rules?" - See tier limits and costs
- **Best Practices**: "How to improve brand recall?" - Get actionable tips

Try asking a specific question about creative testing!
"""


def show_results_demo():
    """Results page for demo mode with pre-scored videos."""
    
    st.markdown("# 📊 Results & Predictions")
    st.markdown("Explore pre-scored sample videos and their predictions.")
    
    st.markdown("---")
    
    sample_videos = get_sample_videos()
    
    # Video selector
    video_names = [v["name"] for v in sample_videos]
    
    # Check if a video was pre-selected
    default_index = 0
    if st.session_state.selected_video:
        for i, v in enumerate(sample_videos):
            if v["id"] == st.session_state.selected_video:
                default_index = i
                break
    
    selected_name = st.selectbox("Select a video to analyze:", video_names, index=default_index)
    
    # Find selected video
    selected = next(v for v in sample_videos if v["name"] == selected_name)
    
    st.markdown("---")
    
    # Main prediction display
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"### 🎬 {selected['name']}")
        st.caption(f"Duration: {selected['duration']}")
        
        # Prediction badge
        if selected["prediction"] == "PASS":
            st.success(f"### ✅ {selected['prediction']}")
        else:
            st.error(f"### ❌ {selected['prediction']}")
        
        st.metric("Confidence", selected["confidence"])
    
    with col2:
        st.markdown("### 📈 Diagnostic Scores")
        
        # Display diagnostic scores as metrics
        cols = st.columns(5)
        diagnostics = selected["diagnostics"]
        
        for i, (key, value) in enumerate(diagnostics.items()):
            with cols[i]:
                label = key.replace("_score", "").replace("_", " ").title()
                delta_color = "normal" if value >= 60 else "inverse"
                st.metric(label, f"{value}/100")
    
    st.markdown("---")
    
    # Features and Recommendation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔍 Extracted Features")
        features = selected["features"]
        
        for key, value in features.items():
            label = key.replace("_", " ").title()
            if isinstance(value, bool):
                icon = "✅" if value else "❌"
                st.markdown(f"- **{label}:** {icon}")
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
    
    # Summary table
    st.markdown("### 📋 All Videos Summary")
    
    summary_data = []
    for v in sample_videos:
        summary_data.append({
            "Video": v["name"],
            "Duration": v["duration"],
            "Prediction": v["prediction"],
            "Confidence": v["confidence"],
            "Attention": v["diagnostics"]["attention_score"],
            "Brand Recall": v["diagnostics"]["brand_recall_score"],
            "Message Clarity": v["diagnostics"]["message_clarity_score"],
        })
    
    import pandas as pd
    df = pd.DataFrame(summary_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def show_planner():
    """CT Planner page - only available in local mode."""
    st.markdown("# 📋 CT Planner")
    st.warning("CT Planner requires full dependencies. Please run locally for this feature.")
    st.markdown("[View setup instructions](https://github.com/akshargupta84/ct-orchestrator#quick-start)")


def show_insights():
    """Insights/Chat page - only available in local mode."""
    st.markdown("# 💬 Insights Chat")
    st.warning("Insights Chat requires full dependencies. Please run locally for this feature.")
    st.markdown("[View setup instructions](https://github.com/akshargupta84/ct-orchestrator#quick-start)")


def show_admin():
    """Admin page - only available in local mode."""
    st.markdown("# ⚙️ Admin")
    st.warning("Admin panel requires full dependencies. Please run locally for this feature.")
    st.markdown("[View setup instructions](https://github.com/akshargupta84/ct-orchestrator#quick-start)")


if __name__ == "__main__":
    main()
