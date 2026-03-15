"""
Planning Agent Page - Conversational test planning with AI.

This page provides a chat-based interface to the Planning Agent,
allowing users to:
1. Upload media plans and videos
2. Chat with the agent to refine the plan
3. Resolve issues (duplicates, low scores, budget)
4. Generate and approve final test plans
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the Planning Agent
try:
    from agents.planning_agent import get_planning_agent, reset_planning_agent, PlanningAgent
    AGENT_AVAILABLE = True
except ImportError as e:
    AGENT_AVAILABLE = False
    AGENT_ERROR = str(e)

# Import persistence
try:
    from services.persistence import get_persistence_service
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False


def render_planning_agent():
    """Render the Planning Agent page."""
    st.markdown("## 🤖 Planning Agent")
    st.markdown("*Chat with an AI agent to create your creative test plan*")
    
    if not AGENT_AVAILABLE:
        st.error(f"Planning Agent not available: {AGENT_ERROR}")
        return
    
    # Initialize session state
    if "planning_agent" not in st.session_state:
        st.session_state.planning_agent = get_planning_agent()
    if "agent_files_uploaded" not in st.session_state:
        st.session_state.agent_files_uploaded = False
    if "agent_processing" not in st.session_state:
        st.session_state.agent_processing = False
    
    agent = st.session_state.planning_agent
    
    # Show current state in sidebar if available
    with st.sidebar:
        if agent.state.media_plan_info and agent.state.media_plan_info.brand:
            st.markdown("### 📊 Current State")
            info = agent.state.media_plan_info
            st.markdown(f"**Brand:** {info.brand}")
            st.markdown(f"**Campaign:** {info.campaign_name}")
            if info.total_budget:
                st.markdown(f"**Budget:** ${info.total_budget:,.0f}")
            st.markdown(f"**Videos:** {len(agent.state.videos)}")
            
            if agent.state.issues:
                st.warning(f"⚠️ {len(agent.state.issues)} issue(s)")
            
            st.markdown("---")
        
        # Reset button in sidebar
        if st.button("🔄 Start Over", use_container_width=True):
            st.session_state.planning_agent = reset_planning_agent()
            st.session_state.agent_files_uploaded = False
            st.session_state.agent_processing = False
            st.rerun()
    
    # Main content - Chat interface
    _render_chat_interface(agent)


def _render_chat_interface(agent: PlanningAgent):
    """Render the chat interface."""
    
    # File upload section at the top (in main area)
    if not agent.state.messages:
        st.markdown("""
        ### 👋 Welcome to the Planning Agent!
        
        I'll help you create an optimal creative test plan. Upload your files below to get started.
        """)
        
        st.markdown("---")
        st.markdown("### 📁 Upload Files")
        
        col1, col2 = st.columns(2)
        
        with col1:
            media_plan_file = st.file_uploader(
                "📋 Media Plan (Excel/CSV)",
                type=['xlsx', 'xls', 'csv'],
                key="agent_media_plan",
                help="Upload your media plan with campaign details, budget, and creative line items"
            )
        
        with col2:
            video_files = st.file_uploader(
                "🎬 Video Creatives",
                type=['mp4', 'mov', 'avi', 'mkv'],
                accept_multiple_files=True,
                key="agent_videos",
                help="Upload the video files you want to test"
            )
        
        # Show what's been uploaded
        if media_plan_file or video_files:
            st.markdown("---")
            st.markdown("### 📦 Files Ready")
            if media_plan_file:
                st.markdown(f"✅ Media Plan: **{media_plan_file.name}**")
            if video_files:
                st.markdown(f"✅ Videos: **{len(video_files)} files**")
                for vf in video_files:
                    st.caption(f"  • {vf.name}")
            
            st.markdown("")
            if st.button("🚀 Analyze Files & Start Planning", type="primary", use_container_width=True):
                _process_uploaded_files(agent, media_plan_file, video_files)
                st.rerun()
        
        # Show example of what agent can do
        with st.expander("💡 What can I help with?"):
            st.markdown("""
            - **Extract campaign details** from media plans automatically
            - **Pre-score videos** to predict pass probability
            - **Detect duplicates** or similar videos
            - **Match videos** to media plan line items
            - **Recommend priorities** when budget is limited
            - **Explain risks** for low-scoring videos
            - **Generate test plans** ready for approval
            """)
        return
    
    # Once conversation has started, show upload option prominently
    # Check if user's last message mentions upload/file
    show_upload_expanded = False
    if agent.state.messages:
        last_msgs = [m['content'].lower() for m in agent.state.messages[-3:]]
        upload_keywords = ['upload', 'file', 'media plan', 'try again', 'reupload', 're-upload']
        for msg in last_msgs:
            if any(kw in msg for kw in upload_keywords):
                show_upload_expanded = True
                break
    
    with st.expander("📁 Upload Files", expanded=show_upload_expanded):
        col1, col2 = st.columns(2)
        
        with col1:
            media_plan_file = st.file_uploader(
                "📋 Media Plan (Excel/CSV)",
                type=['xlsx', 'xls', 'csv'],
                key="agent_media_plan_conv"
            )
        
        with col2:
            video_files = st.file_uploader(
                "🎬 Videos",
                type=['mp4', 'mov', 'avi', 'mkv'],
                accept_multiple_files=True,
                key="agent_videos_conv"
            )
        
        if media_plan_file or video_files:
            if st.button("🚀 Analyze Files", type="primary"):
                _process_uploaded_files(agent, media_plan_file, video_files)
                st.rerun()
    
    # Display conversation history
    st.markdown("### 💬 Conversation")
    
    chat_container = st.container()
    
    with chat_container:
        for msg in agent.state.messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'user':
                with st.chat_message("user"):
                    st.markdown(content)
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(content)
    
    # Chat input
    user_input = st.chat_input("Ask me anything about your test plan...")
    
    if user_input:
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate agent response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                response = agent.chat(user_input)
            st.markdown(response)
        
        st.rerun()
    
    # Action buttons based on state
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if agent.state.videos and st.button("📊 Show Video Summary", use_container_width=True):
            _show_video_summary(agent)
    
    with col2:
        if agent.state.issues and st.button("⚠️ Review Issues", use_container_width=True):
            _show_issues(agent)
    
    with col3:
        if agent.state.videos and st.button("✅ Generate Plan", type="primary", use_container_width=True):
            _generate_and_show_plan(agent)


def _process_uploaded_files(agent: PlanningAgent, media_plan_file, video_files):
    """Process uploaded files and get agent analysis."""
    
    progress = st.progress(0, "Starting analysis...")
    
    media_plan_path = None
    video_file_list = []
    
    try:
        # Save media plan to temp file
        if media_plan_file:
            progress.progress(0.1, "Saving media plan...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(media_plan_file.name)[1]) as f:
                f.write(media_plan_file.getvalue())
                media_plan_path = f.name
        
        # Save videos to temp files
        if video_files:
            for i, vf in enumerate(video_files):
                progress.progress(0.2 + (0.2 * i / len(video_files)), f"Saving {vf.name}...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(vf.name)[1]) as f:
                    f.write(vf.getvalue())
                    video_file_list.append((vf.name, f.name))
        
        # Process with agent
        def update_progress(msg, pct):
            progress.progress(0.4 + (0.5 * pct), msg)
        
        agent.process_upload(
            media_plan_path=media_plan_path,
            video_files=video_file_list if video_file_list else None,
            progress_callback=update_progress
        )
        
        progress.progress(1.0, "Analysis complete!")
        
    except Exception as e:
        st.error(f"Error processing files: {e}")
    
    finally:
        # Clean up temp files (but keep video paths for later)
        if media_plan_path and os.path.exists(media_plan_path):
            try:
                os.unlink(media_plan_path)
            except:
                pass


def _show_video_summary(agent: PlanningAgent):
    """Show video analysis summary in a modal-like expander."""
    
    with st.expander("📹 Video Analysis Summary", expanded=True):
        if not agent.state.videos:
            st.info("No videos analyzed yet.")
            return
        
        # Create summary dataframe
        import pandas as pd
        
        data = []
        for v in agent.state.videos:
            data.append({
                'Video': v.filename,
                'Pass Prob': f"{v.pass_probability*100:.0f}%" if v.scored else "N/A",
                'Status': _get_status(v),
                'Matched To': v.matched_line_item or "❌ No match",
                'Duplicate': v.is_duplicate_of or "-"
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Duplicates section
        if agent.state.duplicates_detected:
            st.markdown("#### 🔄 Duplicates Detected")
            for v1, v2, sim in agent.state.duplicates_detected:
                st.markdown(f"- **{v1}** and **{v2}** ({sim*100:.0f}% similar)")


def _get_status(v) -> str:
    """Get status string for a video."""
    if v.is_duplicate_of:
        return "🔄 Duplicate"
    elif not v.scored:
        return "❓ Not scored"
    elif v.pass_probability >= 0.7:
        return "✅ Strong"
    elif v.pass_probability >= 0.5:
        return "⚠️ Moderate"
    else:
        return "❌ Risky"


def _show_issues(agent: PlanningAgent):
    """Show issues in an expander."""
    
    with st.expander("⚠️ Issues to Resolve", expanded=True):
        if not agent.state.issues:
            st.success("No issues found!")
            return
        
        for i, issue in enumerate(agent.state.issues, 1):
            st.markdown(f"**{i}.** {issue}")
        
        st.markdown("---")
        st.markdown("*Ask me in the chat how to resolve these issues, or tell me your decision.*")


def _generate_and_show_plan(agent: PlanningAgent):
    """Generate and display the final plan."""
    
    with st.expander("✅ Generated Test Plan", expanded=True):
        # Generate plan
        plan = agent.generate_plan()
        
        st.markdown(f"### {plan['campaign']['brand']} - {plan['campaign']['name']}")
        st.markdown(f"**Plan ID:** {plan['plan_id']}")
        st.markdown(f"**Testing Budget:** ${plan['testing_budget']:,}")
        st.markdown(f"**Estimated Turnaround:** {plan['estimated_turnaround']}")
        
        st.markdown("#### Creatives to Test")
        
        for i, creative in enumerate(plan['creatives'], 1):
            prob = creative['predicted_pass_probability'] * 100
            status = "✅" if prob >= 70 else "⚠️" if prob >= 50 else "❌"
            
            st.markdown(f"**{i}. {creative['filename']}** - {prob:.0f}% predicted pass {status}")
            if creative['matched_line_item']:
                st.caption(f"   → {creative['matched_line_item']}")
        
        # Approve button
        st.markdown("---")
        
        if st.button("✅ Approve Plan", type="primary"):
            # Save to persistence
            if PERSISTENCE_AVAILABLE:
                try:
                    persistence = get_persistence_service()
                    
                    # Convert to plan format expected by persistence
                    plan_data = {
                        'plan_id': plan['plan_id'],
                        'campaign': {
                            'id': plan['plan_id'].replace('PLAN_', ''),
                            'name': plan['campaign']['name'],
                            'brand': plan['campaign']['brand'],
                            'budget': plan['campaign']['budget'],
                            'primary_kpi': plan['campaign']['primary_kpi'],
                        },
                        'creatives': plan['creatives'],
                        'testing_budget': plan['testing_budget'],
                    }
                    
                    persistence.save_plan(plan['plan_id'], plan_data, is_approved=True)
                    st.success("✅ Plan approved and saved!")
                    
                    # Add to session state
                    if 'test_plans' not in st.session_state:
                        st.session_state.test_plans = {}
                    st.session_state.test_plans[plan['plan_id']] = plan_data
                    
                except Exception as e:
                    st.error(f"Error saving plan: {e}")
            else:
                st.success("✅ Plan approved!")


# Main render function
def render():
    """Main render function called by the app."""
    render_planning_agent()
