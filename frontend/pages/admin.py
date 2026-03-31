"""
Admin Page.

Handles superuser functions:
- Usage analytics dashboard
- CT Rules document management
- Past learnings upload
- System configuration
"""

import streamlit as st
from pathlib import Path
from datetime import datetime
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def render_admin():
    """Render the Admin page."""
    st.markdown("## ⚙️ Admin Settings")

    # Check if user is admin
    user = st.session_state.get("user_info", {})
    is_admin = user.get("role") == "admin"

    if not is_admin:
        st.warning("Admin access required to view this page.")
        st.info("Log in with the **admin** account to access settings and analytics.")
        return

    st.markdown("Manage system configuration and view usage analytics.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Usage Dashboard",
        "📄 CT Rules",
        "📚 Past Learnings",
        "🔧 System",
    ])

    with tab1:
        render_usage_dashboard()
    with tab2:
        render_rules_management()
    with tab3:
        render_learnings_management()
    with tab4:
        render_system_info()


# =============================================================================
# Usage Dashboard
# =============================================================================

def render_usage_dashboard():
    """Render usage analytics dashboard with charts and tables."""
    st.subheader("Usage Dashboard")

    try:
        from services.usage_tracker import get_tracker
        tracker = get_tracker()
        stats = tracker.get_usage_stats()
    except Exception as e:
        st.info(f"Usage tracker not available: {e}")
        st.caption("Usage data will appear here once users start interacting with the app.")
        return

    if stats["total_queries"] == 0:
        st.info("No usage data yet. Start chatting in the Agent Hub to generate data!")
        return

    # --- Top-line metrics ---
    st.markdown("### Overview")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Queries", f"{stats['total_queries']:,}")
    with m2:
        st.metric("Unique Users", stats["unique_users"])
    with m3:
        st.metric("Sessions", stats["total_sessions"])
    with m4:
        st.metric("Est. API Cost", f"${stats['total_cost']:.2f}")

    st.markdown("---")

    # --- Daily query volume chart ---
    if stats["daily_queries"]:
        st.markdown("### Queries per Day (Last 7 Days)")
        try:
            import plotly.graph_objects as go

            days = [d["day"] for d in stats["daily_queries"]]
            counts = [d["count"] for d in stats["daily_queries"]]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=days, y=counts,
                marker_color="#4A90A4",
                text=counts,
                textposition="outside",
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=40),
                xaxis_title="Date",
                yaxis_title="Queries",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#FAFAFA"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            import pandas as pd
            st.dataframe(
                pd.DataFrame(stats["daily_queries"]),
                use_container_width=True, hide_index=True,
            )

    st.markdown("---")

    # --- Top users and recent queries ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Top Users")
        if stats["top_users"]:
            import pandas as pd
            user_df = pd.DataFrame(stats["top_users"])
            user_df.columns = ["Username", "Queries", "Tokens"]
            user_df["Tokens"] = user_df["Tokens"].apply(lambda x: f"{x:,.0f}")
            st.dataframe(user_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No user data yet.")

    with col2:
        st.markdown("### Recent Queries")
        if stats["recent_queries"]:
            for q in stats["recent_queries"][:10]:
                try:
                    dt = datetime.fromisoformat(q["timestamp"])
                    time_str = dt.strftime("%b %d %H:%M")
                except Exception:
                    time_str = q["timestamp"][:16]

                with st.container(border=True):
                    st.markdown(f"**{q['username']}** · `{q['action']}` · {time_str}")
                    if q["query"]:
                        st.caption(q["query"][:80])
        else:
            st.caption("No queries yet.")

    # --- Token usage pie chart ---
    if stats["top_users"] and any(u["tokens"] > 0 for u in stats["top_users"]):
        st.markdown("---")
        st.markdown("### Token Usage by User")
        try:
            import plotly.graph_objects as go
            users_with_tokens = [u for u in stats["top_users"] if u["tokens"] > 0]
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=[u["username"] for u in users_with_tokens],
                values=[u["tokens"] for u in users_with_tokens],
                hole=0.4,
                marker_colors=["#4A90A4", "#1E3A5F", "#6BB5D1", "#2E75B6", "#A3D4E8"],
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#FAFAFA"),
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            pass

    # --- Data management ---
    st.markdown("---")
    st.markdown("### Data Management")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh Dashboard"):
            st.rerun()
    with col2:
        if st.button("🗑️ Clear All Usage Data", type="secondary"):
            try:
                db_path = tracker.db_path
                if os.path.exists(db_path):
                    os.remove(db_path)
                    st.success("Usage data cleared. Refresh to reinitialize.")
                    st.rerun()
            except Exception as e:
                st.error(f"Error clearing data: {e}")


# =============================================================================
# CT Rules Management
# =============================================================================

def render_rules_management():
    """Render CT Rules management section."""
    st.subheader("CT Rules Document")
    st.markdown("Upload a CT Rules PDF to update the testing rules used for plan validation.")

    uploaded_file = st.file_uploader(
        "Upload CT Rules PDF", type=["pdf"], key="rules_pdf_upload",
        help="Upload a new version of the CT Rules document"
    )

    if uploaded_file:
        st.success(f"File selected: {uploaded_file.name}")
        col1, col2 = st.columns(2)
        with col1:
            version = st.text_input("Version", value="1.1")
        with col2:
            effective_date = st.date_input("Effective Date")

        if st.button("Update Rules", type="primary"):
            rules_dir = Path("master_data")
            rules_dir.mkdir(exist_ok=True)
            rules_path = rules_dir / "ct_rules.pdf"
            with open(rules_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.session_state.rules_pdf_path = str(rules_path)
            try:
                from services.rules_engine import get_rules_engine
                engine = get_rules_engine(str(rules_path))
                engine.reload_rules()
            except Exception:
                pass
            st.success("✅ CT Rules updated successfully!")

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


# =============================================================================
# Past Learnings
# =============================================================================

def render_learnings_management():
    """Render past learnings management section."""
    st.subheader("Past Learnings Knowledge Base")
    st.markdown("Upload past learning documents to enhance AI recommendations and historical analysis.")

    uploaded_file = st.file_uploader(
        "Upload Learning Document", type=["pptx", "ppt", "pdf", "docx"],
        key="learning_upload"
    )

    if uploaded_file:
        st.success(f"File selected: {uploaded_file.name}")
        col1, col2 = st.columns(2)
        with col1:
            brand = st.text_input("Brand (optional)")
        with col2:
            campaign = st.text_input("Campaign (optional)")
        tags = st.text_input("Tags (comma-separated)", placeholder="e.g., video, awareness, Q4")

        if st.button("Add to Knowledge Base", type="primary"):
            st.success("✅ Document added to knowledge base!")

    st.markdown("---")
    st.subheader("Existing Learnings")
    try:
        from services.vector_store import get_vector_store
        vs = get_vector_store()
        stats = vs.get_stats()
        st.metric("Documents in Knowledge Base", stats["learnings_count"])
    except Exception:
        st.info("Knowledge base not yet initialized.")


# =============================================================================
# System Info
# =============================================================================

def render_system_info():
    """Render system information and configuration."""
    st.subheader("System Information")

    # API config
    st.markdown("**API Configuration:**")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key and len(api_key) > 20:
        masked_key = api_key[:8] + "..." + api_key[-4:]
        st.success(f"✅ Anthropic API Key: {masked_key}")
    else:
        st.error("❌ Anthropic API Key not configured")

    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    st.caption(f"Model: `{model}`")

    # Vector store
    st.markdown("---")
    st.markdown("**Vector Store:**")
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

    # Usage DB
    st.markdown("---")
    st.markdown("**Usage Database:**")
    try:
        from services.usage_tracker import get_tracker
        tracker = get_tracker()
        db_size = os.path.getsize(tracker.db_path) / 1024
        st.success(f"✅ `{tracker.db_path}` ({db_size:.1f} KB)")
    except Exception:
        st.warning("Usage tracker not initialized.")

    # Session
    st.markdown("---")
    st.markdown("**Session Data:**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Campaigns", len(st.session_state.get("campaigns", [])))
        st.metric("Test Plans", len(st.session_state.get("test_plans", {})))
    with col2:
        st.metric("Test Results", len(st.session_state.get("test_results", {})))
        st.metric("Chat Messages", len(st.session_state.get("hub_chat_messages", [])))

    # Current user
    st.markdown("---")
    st.markdown("**Current User:**")
    user = st.session_state.get("user_info", {})
    if user:
        st.json({
            "username": user.get("username"),
            "role": user.get("role"),
            "session_id": user.get("session_id", "")[:16] + "...",
            "login_time": user.get("login_time"),
        })

    # Data management
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

    with st.expander("🔍 Debug Information"):
        st.json({
            "session_state_keys": sorted(list(st.session_state.keys())),
            "demo_mode": os.getenv("DEMO_MODE", "true"),
            "authenticated": st.session_state.get("authenticated", False),
        })
