"""
Agent Hub Page - Multi-agent chat interface.

This page provides:
1. File upload (media plans + videos)
2. Chat with the multi-agent system
3. Toggleable reasoning view (see which agents responded)
4. Token usage tracking
"""

import streamlit as st
import tempfile
import os
import json
from pathlib import Path
import sys
import traceback

# Fix path for imports - handle running from frontend/ or project root
current_file = Path(__file__).resolve()
frontend_pages_dir = current_file.parent  # frontend/pages
frontend_dir = frontend_pages_dir.parent   # frontend
project_root = frontend_dir.parent         # ct-orchestrator

# Add project root to path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import multi-agent system
MULTI_AGENT_AVAILABLE = False
MULTI_AGENT_ERROR = ""
try:
    from agents.graph import get_multi_agent_system, reset_multi_agent_system, MultiAgentSystem
    from agents.state import create_initial_state, add_message, AgentState
    MULTI_AGENT_AVAILABLE = True
except ImportError as e:
    MULTI_AGENT_ERROR = f"Import error: {str(e)}\n{traceback.format_exc()}"

# Import file processing from planning agent
PLANNING_AGENT_AVAILABLE = False
try:
    from agents.planning_agent import PlanningAgent
    PLANNING_AGENT_AVAILABLE = True
except ImportError as e:
    pass


def render_agent_hub():
    """Render the Agent Hub page."""
    st.markdown("## 🧠 Agent Hub")
    st.markdown("*Multi-agent AI system for creative testing insights*")
    
    if not MULTI_AGENT_AVAILABLE:
        st.error(f"Multi-agent system not available: {MULTI_AGENT_ERROR}")
        return
    
    # Initialize session state (use existing if already initialized in app.py)
    if "agent_hub_state" not in st.session_state or st.session_state.agent_hub_state is None:
        st.session_state.agent_hub_state = create_initial_state()
    if "agent_hub_system" not in st.session_state or st.session_state.agent_hub_system is None:
        st.session_state.agent_hub_system = get_multi_agent_system()
    if "show_reasoning" not in st.session_state:
        st.session_state.show_reasoning = False
    
    state = st.session_state.agent_hub_state
    system = st.session_state.agent_hub_system
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🧠 Agent Hub")
        
        # Show current context
        if state.get('media_plan_info'):
            mp = state['media_plan_info']
            st.markdown("**📋 Media Plan Loaded**")
            st.caption(f"{mp.get('brand', '')} - {mp.get('campaign_name', '')}")
            if mp.get('total_budget'):
                st.caption(f"Budget: ${mp['total_budget']:,.0f}")
        
        if state.get('videos'):
            st.markdown(f"**🎬 {len(state['videos'])} Videos**")
            for v in state['videos'][:3]:
                prob = v.get('pass_probability', 0)
                st.caption(f"• {v['filename'][:25]}... ({prob*100:.0f}%)")
            if len(state['videos']) > 3:
                st.caption(f"  +{len(state['videos'])-3} more...")
        
        st.markdown("---")
        
        # Toggle reasoning view
        st.session_state.show_reasoning = st.toggle(
            "🔍 Show Agent Reasoning",
            value=st.session_state.show_reasoning
        )
        
        # Token usage
        if state.get('token_usage'):
            with st.expander("💰 Token Usage", expanded=False):
                total = state.get('total_tokens_this_turn', 0)
                st.caption(f"**Last turn:** ~{total:,} tokens")
                for agent, tokens in state.get('token_usage', {}).items():
                    st.caption(f"• {agent}: {tokens:,}")
        
        st.markdown("---")
        
        # Reset button
        if st.button("🔄 Start New Session", use_container_width=True):
            st.session_state.agent_hub_state = create_initial_state()
            st.session_state.agent_hub_system = reset_multi_agent_system()
            st.rerun()
    
    # Main content area
    _render_main_content(state, system)


def _render_main_content(state: AgentState, system: MultiAgentSystem):
    """Render the main chat and file upload area."""

    # Agent architecture (collapsed) — kept in sync with the demo-mode hub so
    # users get the same at-a-glance orientation regardless of which mode they
    # land in.
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

    # File upload section (collapsible after first upload)
    has_files = state.get('media_plan_info') or state.get('videos')
    
    with st.expander("📁 Upload Files", expanded=not has_files):
        col1, col2 = st.columns(2)
        
        with col1:
            media_plan_file = st.file_uploader(
                "📋 Media Plan (Excel/CSV)",
                type=['xlsx', 'xls', 'csv'],
                key="hub_media_plan",
                help="Upload your media plan with campaign details"
            )
        
        with col2:
            video_files = st.file_uploader(
                "🎬 Video Creatives",
                type=['mp4', 'mov', 'avi', 'mkv'],
                accept_multiple_files=True,
                key="hub_videos",
                help="Upload videos to analyze"
            )
        
        if media_plan_file or video_files:
            if st.button("🚀 Process Files", type="primary"):
                _process_files(state, media_plan_file, video_files)
                st.rerun()
    
    # Chat interface
    st.markdown("### 💬 Chat")
    
    # Display conversation
    _render_conversation(state)
    
    # Chat input
    query = st.chat_input("Ask about your creatives, test plans, or historical insights...")
    
    if query:
        _handle_query(state, system, query)
        st.rerun()

    # Starter suggestions — shown before any chat history exists so first-time
    # users have something clickable to try. Matches the "Try These" pattern
    # from the demo-mode hub for consistency.
    if not state.get('messages'):
        st.markdown("---")
        st.markdown("### 🎮 Try These")

        try_col1, try_col2, try_col3 = st.columns(3)
        with try_col1:
            if st.button("❓ What drives brand recall?", use_container_width=True, key="try_recall"):
                _handle_query(state, system, "What drives brand recall in video ads?")
                st.rerun()
        with try_col2:
            if st.button("📋 Show budget rules", use_container_width=True, key="try_budget"):
                _handle_query(state, system, "What are the budget tier rules?")
                st.rerun()
        with try_col3:
            if st.button("📈 Patterns from past tests", use_container_width=True, key="try_patterns"):
                _handle_query(state, system, "What patterns from historical tests predict creative success?")
                st.rerun()

    # Quick action buttons — contextual, shown once files are loaded
    if state.get('videos') or state.get('media_plan_info'):
        st.markdown("---")
        st.markdown("**Quick Actions:**")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("📊 Analyze Videos", use_container_width=True):
                _handle_query(state, system, "Analyze my uploaded videos and explain their scores")
                st.rerun()

        with col2:
            if st.button("📋 Create Test Plan", use_container_width=True):
                _handle_query(state, system, "Create a test plan based on my budget and videos")
                st.rerun()

        with col3:
            if st.button("📈 Show Patterns", use_container_width=True):
                _handle_query(state, system, "What patterns from historical data apply to my creatives?")
                st.rerun()

        with col4:
            if st.button("💡 Get Recommendations", use_container_width=True):
                _handle_query(state, system, "What do you recommend I do with these creatives?")
                st.rerun()


def _render_conversation(state: AgentState):
    """Render the conversation history with optional reasoning."""
    messages = state.get('messages', [])
    
    if not messages:
        st.info("👋 Upload files and ask questions to get started. I can help with:\n\n"
                "• **Analyzing creatives** - Why did this score low?\n"
                "• **Planning tests** - Which videos should I test?\n"
                "• **Historical insights** - What patterns predict success?\n"
                "• **Strategy** - What should I do next?")
        return
    
    for i, msg in enumerate(messages):
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        
        if role == 'user':
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown(content)
                
                # Show reasoning if enabled and this is the last assistant message
                if st.session_state.show_reasoning and i == len(messages) - 1:
                    _render_reasoning(state)


def _render_reasoning(state: AgentState):
    """Render the agent reasoning trace."""
    reasoning = state.get('reasoning_trace', [])
    responses = state.get('agent_responses', {})
    errors = state.get('agent_errors', {})
    
    if not reasoning and not responses:
        return
    
    with st.expander("🔍 Agent Reasoning", expanded=True):
        # Show which agents were selected
        selected = state.get('selected_agents', [])
        if selected:
            st.markdown(f"**Agents consulted:** {', '.join([a.replace('_', ' ').title() for a in selected])}")
        
        # Show each agent's response
        for agent_name, response in responses.items():
            display_name = agent_name.replace('_', ' ').title()
            content = response.get('content', '')
            
            with st.container():
                st.markdown(f"**{display_name}:**")
                st.markdown(content[:500] + "..." if len(content) > 500 else content)
                st.markdown("---")
        
        # Show errors
        for agent_name, error in errors.items():
            st.error(f"**{agent_name}** failed: {error}")
        
        # Show inter-agent requests
        requests = state.get('inter_agent_requests', [])
        if requests:
            st.markdown("**Inter-agent communication:**")
            for req in requests:
                st.caption(f"• {req['from_agent']} asked {req['to_agent']}: {req['question'][:50]}...")


def _process_files(state: AgentState, media_plan_file, video_files):
    """Process uploaded files and update state."""
    
    progress = st.progress(0, "Processing files...")
    
    # Use PlanningAgent's file processing capabilities
    if PLANNING_AGENT_AVAILABLE:
        planning_agent = PlanningAgent()
        
        media_plan_path = None
        video_file_list = []
        
        try:
            # Save media plan
            if media_plan_file:
                progress.progress(0.1, "Processing media plan...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(media_plan_file.name)[1]) as f:
                    f.write(media_plan_file.getvalue())
                    media_plan_path = f.name
                
                # Parse media plan
                file_ext = os.path.splitext(media_plan_file.name)[1].lower()
                file_type = 'xlsx' if file_ext in ['.xlsx', '.xls'] else 'csv'
                media_plan_info = planning_agent.parse_media_plan(media_plan_path, file_type)
                
                if media_plan_info:
                    state['media_plan_info'] = media_plan_info.to_dict()
                    # Calculate testing budget
                    if media_plan_info.total_budget:
                        state['media_plan_info']['testing_budget'] = media_plan_info.total_budget * 0.04
            
            # Save and process videos
            if video_files:
                for i, vf in enumerate(video_files):
                    progress.progress(0.2 + (0.6 * i / len(video_files)), f"Processing {vf.name}...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(vf.name)[1]) as f:
                        f.write(vf.getvalue())
                        video_file_list.append((vf.name, f.name))
                
                # Analyze videos
                def update_progress(msg, pct):
                    progress.progress(0.2 + (0.6 * pct), msg)
                
                videos = planning_agent.analyze_videos(video_file_list, update_progress)
                
                # Convert to state format
                state['videos'] = [
                    {
                        'filename': v.filename,
                        'filepath': v.filepath,
                        'pass_probability': v.pass_probability,
                        'risk_factors': v.risk_factors,
                        'matched_line_item': v.matched_line_item,
                        'is_duplicate_of': v.is_duplicate_of,
                        'features': v.features,
                        'diagnostics': v.diagnostics,
                        'scored': v.scored
                    }
                    for v in videos
                ]
                
                # Store duplicates
                state['duplicates_detected'] = planning_agent.state.duplicates_detected
            
            progress.progress(1.0, "Done!")
            
            # Add system message about what was loaded
            summary_parts = []
            if state.get('media_plan_info'):
                mp = state['media_plan_info']
                summary_parts.append(f"📋 Loaded media plan: {mp.get('brand', 'Unknown')} - {mp.get('campaign_name', 'Unknown')}")
                if mp.get('total_budget'):
                    summary_parts.append(f"   Budget: ${mp['total_budget']:,.0f} (testing: ${mp.get('testing_budget', 0):,.0f})")
            
            if state.get('videos'):
                summary_parts.append(f"🎬 Analyzed {len(state['videos'])} videos")
                for v in state['videos']:
                    prob = v.get('pass_probability', 0)
                    status = '✅' if prob >= 0.7 else '⚠️' if prob >= 0.5 else '❌'
                    summary_parts.append(f"   • {v['filename']}: {prob*100:.0f}% {status}")
                    
                    # Show diagnostic scores if available
                    diagnostics = v.get('diagnostics', {})
                    if diagnostics:
                        diag_parts = []
                        for name, value in diagnostics.items():
                            # Format name nicely
                            display_name = name.replace('_', ' ').title().replace(' Score', '')
                            if isinstance(value, (int, float)):
                                diag_parts.append(f"{display_name}: {value:.0f}")
                        if diag_parts:
                            summary_parts.append(f"     📊 {' | '.join(diag_parts)}")
            
            if summary_parts:
                add_message(state, 'assistant', "**Files processed:**\n\n" + "\n".join(summary_parts) + "\n\nHow can I help you analyze these?")
            
        except Exception as e:
            st.error(f"Error processing files: {e}")
        
        finally:
            # Cleanup temp files
            if media_plan_path and os.path.exists(media_plan_path):
                try:
                    os.unlink(media_plan_path)
                except:
                    pass
    else:
        st.error("File processing not available")


def _handle_query(state: AgentState, system: MultiAgentSystem, query: str):
    """Handle a user query."""

    # Add user message
    add_message(state, 'user', query)
    state['current_query'] = query

    # Track how many messages existed before processing so we can identify
    # exactly what the agents appended and render it inline.
    messages_before = len(state.get('messages', []))

    # Inline-render the user message and a thinking indicator so the query is
    # visibly acknowledged while the agents are working. _render_conversation
    # already painted the prior history above this point, so the new pair only
    # needs to appear here; the rerun in the caller will repaint cleanly from
    # state on the next pass.
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🧠"):
        response_slot = st.empty()
        with response_slot.container():
            with st.spinner("🧠 Agents are thinking…"):
                try:
                    updated_state = system.process(state)
                    st.session_state.agent_hub_state = updated_state
                except Exception as e:
                    add_message(state, 'assistant', f"I encountered an error: {str(e)}")
                    st.session_state.agent_hub_state = state

        # Replace the spinner with the actual assistant reply so the user can
        # see the response (or error) immediately, without waiting for the
        # rerun to redraw from state.
        final_state = st.session_state.agent_hub_state or state
        new_messages = final_state.get('messages', [])[messages_before:]
        assistant_reply = next(
            (m.get('content', '') for m in reversed(new_messages)
             if m.get('role') == 'assistant'),
            None,
        )
        if assistant_reply:
            response_slot.markdown(assistant_reply)
        else:
            response_slot.warning(
                "The agents didn't return a response. "
                "Check the terminal for errors (e.g., missing ANTHROPIC_API_KEY, "
                "Ollama not running, or import failures)."
            )


def render():
    """Main render function called by the app."""
    render_agent_hub()
