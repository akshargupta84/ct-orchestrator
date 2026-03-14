"""
CT Orchestrator - Streamlit Frontend
Main application entry point with navigation and shared state.

This version is designed for Hugging Face Spaces demo with pre-scored videos.
"""

import streamlit as st
from pathlib import Path
import sys
import json

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
# Set DEMO_MODE=false in .env to enable full functionality with Ollama
import os
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"

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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .status-pass {
        color: #10B981;
        font-weight: 600;
    }
    .status-fail {
        color: #EF4444;
        font-weight: 600;
    }
    .demo-banner {
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .feature-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
    }
    .feature-card:hover {
        border-color: #3B82F6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
    }
    .step-number {
        background: #3B82F6;
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        margin-right: 12px;
    }
    .code-block {
        background: #1e293b;
        color: #e2e8f0;
        padding: 1rem;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.9rem;
        overflow-x: auto;
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


def main():
    """Main application."""
    init_session_state()
    
    if DEMO_MODE:
        load_demo_data()
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("## 🎬 CT Orchestrator")
        
        if DEMO_MODE:
            st.markdown("""
            <div style="background: #FEF3C7; border: 1px solid #F59E0B; border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem;">
                <strong style="color: #92400E;">📺 Demo Mode</strong><br>
                <span style="color: #78350F; font-size: 0.85rem;">Using pre-scored videos</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        nav_options = [
            "🏠 Welcome",
            "🤖 Multi-Agent Hub", 
            "📋 CT Planner",
            "📊 Results & Predictions",
            "💬 Insights Chat",
            "⚙️ Admin"
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
        st.markdown("### 📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Campaigns", len(st.session_state.campaigns))
        with col2:
            st.metric("Results", len(st.session_state.test_results))
        
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
    elif st.session_state.nav_page == "📋 CT Planner":
        show_planner()
    elif st.session_state.nav_page == "📊 Results & Predictions":
        show_results()
    elif st.session_state.nav_page == "💬 Insights Chat":
        show_insights()
    elif st.session_state.nav_page == "⚙️ Admin":
        show_admin()


def show_welcome():
    """Welcome/intro page with project overview."""
    
    # Hero Section
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">🎬 Creative Testing Orchestrator</h1>
        <p style="font-size: 1.3rem; color: #64748b; max-width: 800px; margin: 0 auto;">
            AI-powered multi-agent system for automating creative testing workflows in media agencies
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Demo Mode Banner
    if DEMO_MODE:
        st.markdown("""
        <div class="demo-banner">
            <strong>🎯 You're viewing the interactive demo</strong><br>
            This demo uses <strong>pre-scored sample videos</strong> so you can explore all features instantly.
            For live video scoring with your own creatives, clone the 
            <a href="https://github.com/akshargupta84/ct-orchestrator" style="color: white; text-decoration: underline;">GitHub repo</a> 
            and run locally with Ollama.
        </div>
        """, unsafe_allow_html=True)
    
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
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">~$150K</div>
            <div class="metric-label">Potential Annual Savings</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #10B981 0%, #059669 100%);">
            <div class="metric-value">65%→30%</div>
            <div class="metric-label">Reduce Failure Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # How to Use This Demo
    st.markdown("## 🚀 How to Explore This Demo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3><span class="step-number">1</span>Multi-Agent Hub</h3>
            <p>See how our AI agents work together to orchestrate the creative testing workflow.</p>
            <ul>
                <li>Planning Agent</li>
                <li>Analysis Agent</li>
                <li>Video Analyzer</li>
                <li>Knowledge Agent</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3><span class="step-number">2</span>Results & Predictions</h3>
            <p>Explore pre-scored sample videos and see predictions in action.</p>
            <ul>
                <li>Pass/Fail predictions</li>
                <li>Diagnostic scores</li>
                <li>Video feature analysis</li>
                <li>Recommendations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3><span class="step-number">3</span>Insights Chat</h3>
            <p>Ask questions about creative testing rules, past results, and best practices.</p>
            <ul>
                <li>"What drives brand recall?"</li>
                <li>"Which creatives passed?"</li>
                <li>"What are the budget rules?"</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sample Videos
    st.markdown("## 📹 Pre-Scored Sample Videos")
    st.markdown("""
    This demo includes **4 sample video creatives** that have been pre-analyzed. 
    You can explore their features, predictions, and recommendations without waiting for processing.
    """)
    
    # Display sample video cards
    sample_videos = [
        {
            "name": "Summer_Hero_30s",
            "duration": "30 sec",
            "prediction": "PASS",
            "confidence": "87%",
            "key_features": "Human presence, Product demo, Clear CTA"
        },
        {
            "name": "Brand_Story_15s",
            "duration": "15 sec",
            "prediction": "PASS",
            "confidence": "72%",
            "key_features": "Emotional content, Logo in first 3s"
        },
        {
            "name": "Product_Focus_30s",
            "duration": "30 sec",
            "prediction": "FAIL",
            "confidence": "65%",
            "key_features": "Low human presence, No clear CTA"
        },
        {
            "name": "Lifestyle_60s",
            "duration": "60 sec",
            "prediction": "PASS",
            "confidence": "79%",
            "key_features": "High engagement, Strong brand recall"
        }
    ]
    
    cols = st.columns(4)
    for i, video in enumerate(sample_videos):
        with cols[i]:
            status_color = "#10B981" if video["prediction"] == "PASS" else "#EF4444"
            st.markdown(f"""
            <div style="border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem; text-align: center;">
                <div style="background: #f1f5f9; border-radius: 8px; padding: 2rem; margin-bottom: 1rem;">
                    🎬
                </div>
                <strong>{video["name"]}</strong><br>
                <span style="color: #64748b; font-size: 0.85rem;">{video["duration"]}</span><br>
                <span style="color: {status_color}; font-weight: 600; font-size: 1.2rem;">
                    {video["prediction"]} ({video["confidence"]})
                </span><br>
                <span style="color: #64748b; font-size: 0.75rem;">{video["key_features"]}</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Run Locally Section
    st.markdown("## 💻 Run Locally with Your Own Videos")
    
    st.markdown("""
    Want to analyze your own video creatives? Clone the repo and run locally with Ollama for live scoring.
    """)
    
    with st.expander("📋 Quick Start Guide", expanded=False):
        st.markdown("""
        ### Prerequisites
        - Python 3.10+
        - [Ollama](https://ollama.ai/) installed and running
        - Anthropic API key (for chat features)
        
        ### Installation
        ```bash
        # Clone the repository
        git clone https://github.com/akshargupta84/ct-orchestrator.git
        cd ct-orchestrator
        
        # Create virtual environment
        python -m venv venv
        source venv/bin/activate
        
        # Install dependencies
        pip install -r requirements.txt
        
        # Set up Ollama (in a separate terminal)
        ollama serve
        ollama pull llava:13b
        
        # Configure environment
        cp .env.example .env
        # Edit .env with your ANTHROPIC_API_KEY
        
        # Run the app
        cd frontend
        streamlit run app.py
        ```
        
        ### Analyze Your Own Videos
        ```python
        from services.video_ingestion import VideoIngestionService
        from services.video_analysis import LocalVisionService
        
        # Initialize
        vision = LocalVisionService()
        ingestion = VideoIngestionService(vision_service=vision)
        
        # Analyze a video
        result = ingestion.analyze_video("path/to/your/video.mp4")
        print(f"Pass probability: {result.pass_probability:.1%}")
        ```
        """)
    
    # Architecture Preview
    st.markdown("---")
    st.markdown("## 🏗️ Architecture Overview")
    
    st.markdown("""
    ```
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
    ```
    """)
    
    # Call to Action
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Exploring → Multi-Agent Hub", use_container_width=True, type="primary"):
            st.session_state.nav_page = "🤖 Multi-Agent Hub"
            st.rerun()


def show_multi_agent_hub():
    """Multi-Agent Hub page showing agent orchestration."""
    
    st.markdown("# 🤖 Multi-Agent Hub")
    st.markdown("See how our AI agents work together to orchestrate creative testing workflows.")
    
    st.markdown("---")
    
    # Agent Overview
    st.markdown("## Agent Architecture")
    
    st.markdown("""
    ```
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
    ```
    """)
    
    st.markdown("---")
    
    # Individual Agent Details
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
    
    # Live Demo Section
    st.markdown("## 🎮 Try It: Agent Simulation")
    
    st.markdown("Select a scenario to see how the agents would respond:")
    
    scenario = st.selectbox(
        "Choose a scenario:",
        [
            "Select a scenario...",
            "🎬 Analyze a new video creative",
            "📋 Validate a test plan for $15M campaign",
            "❓ Ask: What drives brand recall?",
            "📊 Get recommendations for a failed creative"
        ]
    )
    
    if scenario != "Select a scenario...":
        with st.spinner("Agents processing..."):
            import time
            time.sleep(1)  # Simulate processing
        
        if "Analyze a new video" in scenario:
            st.success("**Video Analyzer** activated")
            st.markdown("""
            **Agent Response:**
            
            ```
            Video Analysis Complete: Summer_Hero_30s.mp4
            
            Features Extracted:
            ├── Duration: 30 seconds
            ├── Human Presence: 73% of frames
            ├── Logo Visibility: First appears at 2.1s ✓
            ├── Product Demo: Yes (45% of frames)
            ├── CTA Detected: Yes, at 27s
            └── Emotional Content: Positive
            
            → Forwarding to Analysis Agent for prediction...
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
        
        elif "Validate a test plan" in scenario:
            st.success("**Planning Agent** activated")
            st.markdown("""
            **Agent Response:**
            
            ```
            Test Plan Validation: $15M Campaign
            
            Budget Tier: TIER 2 ($5M - $35M)
            ├── Video Limit: 8 creatives
            ├── Display Limit: 15 creatives
            
            Plan Details:
            ├── Requested Videos: 6 ✓
            ├── Requested Display: 10 ✓
            
            Cost Calculation:
            ├── Videos: 6 × $5,000 = $30,000
            ├── Display: 10 × $3,000 = $30,000
            └── Total: $60,000
            
            Timeline: 10 business days (standard)
            
            Status: ✅ APPROVED - Within all limits
            ```
            """)
        
        elif "What drives brand recall" in scenario:
            st.success("**Knowledge Agent** activated (RAG)")
            st.markdown("""
            **Agent Response:**
            
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
        
        elif "failed creative" in scenario:
            st.success("**Analysis Agent** activated")
            st.markdown("""
            **Agent Response:**
            
            ```
            Creative Analysis: Product_Focus_30s
            
            Prediction: FAIL (65% confidence)
            
            Diagnostic Breakdown:
            ├── Attention: 45/100 ⚠️ Below threshold
            ├── Brand Recall: 38/100 ⚠️ Below threshold
            ├── Message Clarity: 62/100 
            └── Emotional Resonance: 41/100 ⚠️ Below threshold
            
            Root Cause Analysis:
            ├── Low human presence (12% of frames)
            ├── Logo appears late (at 18s)
            ├── No clear CTA detected
            └── Product-only shots lack engagement
            
            Recommendations:
            1. Add human talent interacting with product
            2. Move logo to opening 3 seconds
            3. Add clear CTA in final 5 seconds
            4. Include emotional storytelling elements
            
            Action: OPTIMIZE AND RETEST
            ```
            """)
    
    st.markdown("---")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("📊 Explore Results & Predictions →", use_container_width=True, type="primary"):
            st.session_state.nav_page = "📊 Results & Predictions"
            st.rerun()


def show_planner():
    """CT Planner page - imported from pages module."""
    from pages.ct_planner import render_planner
    render_planner()


def show_results():
    """Results page - imported from pages module."""
    from pages.results import render_results
    render_results()


def show_insights():
    """Insights/Chat page - imported from pages module."""
    from pages.insights import render_insights
    render_insights()


def show_admin():
    """Admin page - imported from pages module."""
    from pages.admin import render_admin
    render_admin()


if __name__ == "__main__":
    main()
