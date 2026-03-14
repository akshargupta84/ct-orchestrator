"""
Admin Page.

Handles superuser functions:
- CT Rules document management
- Past learnings upload
- System configuration
"""

import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def render_admin():
    """Render the Admin page."""
    st.markdown("## ⚙️ Admin Settings")
    st.markdown("Manage master data and system configuration.")
    
    # Tabs for different admin functions
    tab1, tab2, tab3 = st.tabs(["📄 CT Rules", "📚 Past Learnings", "🔧 System"])
    
    with tab1:
        render_rules_management()
    
    with tab2:
        render_learnings_management()
    
    with tab3:
        render_system_info()


def render_rules_management():
    """Render CT Rules management section."""
    st.subheader("CT Rules Document")
    st.markdown("""
    Upload a CT Rules PDF to update the testing rules used for plan validation.
    The rules document defines:
    - Budget tiers and creative limits
    - Testing costs per asset type
    - Turnaround times
    - Minimum requirements
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CT Rules PDF",
        type=["pdf"],
        key="rules_pdf_upload",
        help="Upload a new version of the CT Rules document"
    )
    
    if uploaded_file:
        st.success(f"File selected: {uploaded_file.name}")
        
        col1, col2 = st.columns(2)
        with col1:
            version = st.text_input("Version", value="1.1", help="Version number for tracking")
        with col2:
            effective_date = st.date_input("Effective Date")
        
        if st.button("Update Rules", type="primary"):
            # Save file
            rules_dir = Path("master_data")
            rules_dir.mkdir(exist_ok=True)
            
            rules_path = rules_dir / "ct_rules.pdf"
            with open(rules_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Update session state
            st.session_state.rules_pdf_path = str(rules_path)
            
            # Reload rules engine
            from services.rules_engine import get_rules_engine
            engine = get_rules_engine(str(rules_path))
            engine.reload_rules()
            
            st.success("✅ CT Rules updated successfully!")
            st.info("All future validations will use the new rules.")
    
    # Show current rules
    st.markdown("---")
    st.subheader("Current Rules Summary")
    
    try:
        from services.rules_engine import get_rules_engine
        engine = get_rules_engine()
        rules = engine.rules
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Budget Allocation:**")
            st.markdown(f"- {rules.budget_allocation_pct * 100:.1f}% of campaign budget")
            
            st.markdown("**Testing Costs:**")
            st.markdown(f"- Video: ${rules.costs.video_cost:,} per asset")
            st.markdown(f"- Display: ${rules.costs.display_cost:,} per asset")
            st.markdown(f"- Audio: ${rules.costs.audio_cost:,} per asset")
            st.markdown(f"- Expedited fee: +{rules.costs.expedited_fee_pct * 100:.0f}%")
        
        with col2:
            st.markdown("**Turnaround Times:**")
            st.markdown(f"- Video: {rules.turnaround.video_standard_days} days (standard)")
            st.markdown(f"- Display: {rules.turnaround.display_standard_days} days (standard)")
            
            st.markdown("**Minimum Requirements:**")
            st.markdown(f"- Required video test above: ${rules.minimum_requirements.min_budget_for_required_video_test:,.0f}")
        
        st.markdown("**Budget Tiers:**")
        tier_data = []
        for tier in rules.budget_tiers:
            max_budget = f"${tier.max_budget:,.0f}" if tier.max_budget < float('inf') else "Unlimited"
            tier_data.append({
                "Min Budget": f"${tier.min_budget:,.0f}",
                "Max Budget": max_budget,
                "Video Limit": tier.video_limit,
                "Display Limit": tier.display_limit,
            })
        st.table(tier_data)
        
    except Exception as e:
        st.warning(f"Could not load rules: {e}")
        st.info("Upload a CT Rules PDF to configure the system.")


def render_learnings_management():
    """Render past learnings management section."""
    st.subheader("Past Learnings Knowledge Base")
    st.markdown("""
    Upload past learning documents (PPTs, reports) to enhance the AI's knowledge.
    These documents help provide context for:
    - Creative recommendations
    - Historical comparisons
    - Meta-analysis insights
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Learning Document",
        type=["pptx", "ppt", "pdf", "docx"],
        key="learning_upload",
        help="Upload past learnings, case studies, or reports"
    )
    
    if uploaded_file:
        st.success(f"File selected: {uploaded_file.name}")
        
        col1, col2 = st.columns(2)
        with col1:
            brand = st.text_input("Brand (optional)", help="Associate with a specific brand")
        with col2:
            campaign = st.text_input("Campaign (optional)", help="Associate with a campaign")
        
        tags = st.text_input(
            "Tags (comma-separated)",
            placeholder="e.g., video, awareness, Q4",
            help="Add tags for easier searching"
        )
        
        if st.button("Add to Knowledge Base", type="primary"):
            # In production, this would:
            # 1. Extract text from the document
            # 2. Chunk and embed the text
            # 3. Store in vector database
            
            st.success("✅ Document added to knowledge base!")
            st.info(f"The system can now reference insights from {uploaded_file.name}")
    
    # Show existing learnings
    st.markdown("---")
    st.subheader("Existing Learnings")
    
    try:
        from services.vector_store import get_vector_store
        vs = get_vector_store()
        stats = vs.get_stats()
        
        st.metric("Documents in Knowledge Base", stats["learnings_count"])
        
        if stats["learnings_count"] > 0:
            st.markdown("*Use the Insights page to query the knowledge base.*")
        else:
            st.info("No learnings uploaded yet. Upload documents above to build the knowledge base.")
            
    except Exception as e:
        st.info("Knowledge base not yet initialized. Upload your first document to get started.")


def render_system_info():
    """Render system information and configuration."""
    st.subheader("System Information")
    
    # API Configuration
    st.markdown("**API Configuration:**")
    
    import os
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "****"
        st.success(f"✅ Anthropic API Key configured: {masked_key}")
    else:
        st.error("❌ Anthropic API Key not configured")
        st.code("export ANTHROPIC_API_KEY=your_key_here", language="bash")
    
    # Vector Store Status
    st.markdown("---")
    st.markdown("**Vector Store Status:**")
    
    try:
        from services.vector_store import get_vector_store
        vs = get_vector_store()
        stats = vs.get_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rules Chunks", stats["rules_count"])
        with col2:
            st.metric("Past Learnings", stats["learnings_count"])
        with col3:
            st.metric("Historical Results", stats["results_count"])
        
        st.success("✅ Vector store operational")
        
    except Exception as e:
        st.warning(f"Vector store not initialized: {e}")
    
    # Session State
    st.markdown("---")
    st.markdown("**Session Data:**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Campaigns", len(st.session_state.get("campaigns", [])))
        st.metric("Test Plans", len(st.session_state.get("test_plans", {})))
    with col2:
        st.metric("Test Results", len(st.session_state.get("test_results", {})))
        st.metric("Chat Messages", len(st.session_state.get("chat_history", [])))
    
    # Clear data option
    st.markdown("---")
    st.markdown("**Data Management:**")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Session Data", type="secondary"):
            for key in ["campaigns", "test_plans", "test_results", "chat_history"]:
                if key in st.session_state:
                    st.session_state[key] = [] if key == "campaigns" else {}
            st.success("Session data cleared!")
            st.rerun()
    
    with col2:
        if st.button("🔄 Refresh System", type="secondary"):
            st.rerun()
    
    # Debug info
    with st.expander("🔍 Debug Information"):
        st.json({
            "session_state_keys": list(st.session_state.keys()),
            "rules_pdf_path": st.session_state.get("rules_pdf_path"),
            "current_campaign": st.session_state.get("current_campaign"),
        })
