"""
Insights Page.

Chat interface for:
1. Asking questions about CT rules
2. Querying past results (from vector store)
3. Getting recommendations
4. Meta-analysis across campaigns
5. Performance driver analysis (creative features → lift)
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.rules_engine import get_rules_engine
from services.vector_store import get_vector_store
from utils.llm import get_completion, classify_question

# Try to import performance modeling
try:
    from services.performance_modeling import get_modeling_service
    MODELING_AVAILABLE = True
except ImportError:
    MODELING_AVAILABLE = False


def render_insights():
    """Render the Insights/Chat page."""
    st.markdown("## 💬 Insights & Q&A")
    st.markdown("Ask questions about CT rules, past results, creative performance drivers, or get recommendations.")
    
    # Show knowledge base status
    render_knowledge_status()
    
    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Quick question buttons
    st.markdown("#### Quick Questions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Rules & Guidelines**")
        if st.button("What are the testing limits for a $10M campaign?", use_container_width=True):
            add_question("What are the testing limits for a $10M campaign?")
        if st.button("How much does video testing cost?", use_container_width=True):
            add_question("How much does video testing cost?")
    
    with col2:
        st.markdown("**Past Results**")
        if st.button("Which creatives performed best?", use_container_width=True):
            add_question("Which creatives performed best in past tests?")
        if st.button("What's our overall pass rate?", use_container_width=True):
            add_question("What's our overall pass rate across all campaigns?")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**Performance Drivers**")
        if st.button("What drives creative performance?", use_container_width=True):
            add_question("What creative elements drive better performance?")
        if st.button("Does human presence impact lift?", use_container_width=True):
            add_question("Does human presence in videos impact awareness lift?")
    
    with col4:
        st.markdown("**Recommendations**")
        if st.button("What should we avoid in creatives?", use_container_width=True):
            add_question("What creative elements should we avoid?")
        if st.button("How should we use product shots?", use_container_width=True):
            add_question("How does showing the product in use affect performance?")
    
    st.markdown("---")
    
    # Chat interface
    st.markdown("#### Chat")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Input
    user_input = st.chat_input("Ask a question about creative testing...")
    
    if user_input:
        add_question(user_input)
    
    # Clear chat button
    if st.session_state.chat_messages:
        if st.button("Clear Chat History"):
            st.session_state.chat_messages = []
            st.rerun()


def render_knowledge_status():
    """Show status of the knowledge base."""
    try:
        vs = get_vector_store()
        stats = vs.get_stats()
        
        with st.expander("📚 Knowledge Base Status", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rules Chunks", stats["rules_count"])
            with col2:
                st.metric("Past Learnings", stats["learnings_count"])
            with col3:
                st.metric("Test Results", stats["results_count"])
            
            if stats["results_count"] == 0:
                st.info("💡 Upload test results in the Results tab to enable result-based queries.")
    except Exception:
        pass  # Silently handle if vector store not initialized


def add_question(question: str):
    """Add a question to chat and get response."""
    # Add user message
    st.session_state.chat_messages.append({
        "role": "user",
        "content": question
    })
    
    # Get comprehensive response
    response = get_comprehensive_answer(question)
    
    # Add assistant message
    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": response
    })
    
    st.rerun()


def get_comprehensive_answer(question: str) -> str:
    """
    Get a comprehensive answer by querying all available sources:
    1. CT Rules (structured + vector search)
    2. Past test results (from vector store)
    3. Session state results (current session)
    4. Past learnings (from vector store)
    5. Performance modeling insights (creative features → lift)
    """
    
    # Classify the question to understand intent
    question_type = classify_question(question)
    question_lower = question.lower()
    
    # Check if this is a performance driver question
    is_driver_question = any(word in question_lower for word in [
        'driver', 'impact', 'affect', 'influence', 'human', 'logo', 'product',
        'cta', 'call to action', 'what makes', 'best performing', 'worst',
        'avoid', 'element', 'feature', 'presence'
    ])
    
    # If it's a driver question and modeling is available, use that
    if is_driver_question and MODELING_AVAILABLE:
        try:
            modeling_service = get_modeling_service()
            if modeling_service.analyses:
                driver_answer = modeling_service.answer_question(question)
                if driver_answer and "don't have any" not in driver_answer:
                    return driver_answer
        except Exception:
            pass  # Fall through to other sources
    
    # Gather context from all sources
    context_parts = []
    
    # 1. Get rules context (always relevant)
    rules_context = get_rules_context(question)
    if rules_context:
        context_parts.append(f"**CT Rules & Guidelines:**\n{rules_context}")
    
    # 2. Get vector store results context
    vector_context = get_vector_store_context(question)
    if vector_context:
        context_parts.append(f"**Historical Test Results (from knowledge base):**\n{vector_context}")
    
    # 3. Get current session results context
    session_context = get_session_results_context(question)
    if session_context:
        context_parts.append(f"**Current Session Results:**\n{session_context}")
    
    # 4. Get past learnings context
    learnings_context = get_learnings_context(question)
    if learnings_context:
        context_parts.append(f"**Past Learnings:**\n{learnings_context}")
    
    # 5. Get performance modeling context if available
    if MODELING_AVAILABLE:
        modeling_context = get_modeling_context(question)
        if modeling_context:
            context_parts.append(f"**Performance Driver Analysis:**\n{modeling_context}")
    
    # If no context available, provide helpful guidance
    if not context_parts:
        return get_no_context_response(question)
    
    # Combine all context
    full_context = "\n\n".join(context_parts)
    
    # Generate comprehensive answer
    prompt = f"""Based on the following information about our creative testing program:

{full_context}

Please answer this question: {question}

Provide a clear, specific answer. If citing data, be precise about numbers and sources. 
If the information isn't available, say so clearly."""

    system_prompt = """You are an expert creative testing analyst helping media agency teams understand their test results and optimize creative performance. 

You have access to:
- CT Rules and guidelines
- Historical test results
- Past learnings from previous campaigns
- Performance driver analysis (which creative elements drive lift)

Be helpful, specific, and data-driven in your responses. When discussing results, cite specific creatives, campaigns, and metrics when available."""

    return get_completion(prompt=prompt, system=system_prompt)


def get_rules_context(question: str) -> str:
    """Get relevant rules context."""
    try:
        rules_engine = get_rules_engine()
        rules = rules_engine.rules
        
        # Build a summary of key rules
        rules_summary = f"""Budget Tiers and Limits:
{chr(10).join(f"- ${tier.min_budget:,.0f} to ${tier.max_budget if tier.max_budget < float('inf') else 'unlimited':,.0f}: {tier.video_limit} videos, {tier.display_limit} display" for tier in rules.budget_tiers)}

Costs:
- Video testing: ${rules.costs.video_cost:,} per asset
- Display testing: ${rules.costs.display_cost:,} per asset
- Audio testing: ${rules.costs.audio_cost:,} per asset
- Expedited fee: +{rules.costs.expedited_fee_pct * 100:.0f}%

Turnaround Times:
- Video: {rules.turnaround.video_standard_days} days standard, {rules.turnaround.video_expedited_days} days expedited
- Display: {rules.turnaround.display_standard_days} days standard, {rules.turnaround.display_expedited_days} days expedited

Budget Allocation: {rules.budget_allocation_pct * 100:.1f}% of campaign budget goes to testing"""

        return rules_summary
    except Exception as e:
        return ""


def get_vector_store_context(question: str) -> str:
    """Get relevant context from vector store."""
    try:
        vs = get_vector_store()
        
        # Query the results collection
        results = vs.query_results(question, n_results=5)
        
        if not results:
            return ""
        
        context_parts = []
        for r in results:
            context_parts.append(r["text"])
        
        return "\n\n---\n\n".join(context_parts)
    except Exception:
        return ""


def get_session_results_context(question: str) -> str:
    """Get context from current session results."""
    if not st.session_state.get("test_results"):
        return ""
    
    context_parts = []
    
    for plan_id, results in st.session_state.test_results.items():
        plan = st.session_state.test_plans.get(plan_id, {})
        campaign = plan.get("campaign", {})
        campaign_name = campaign.get("name", "Unknown Campaign")
        brand_name = campaign.get("brand", {}).get("name", "Unknown Brand")
        
        summary = [
            f"Campaign: {campaign_name} ({brand_name})",
            f"Pass Rate: {results.pass_rate * 100:.0f}% ({results.creatives_passed}/{results.total_creatives_tested})",
            "Results:"
        ]
        
        for r in results.results:
            status = "✓ PASS" if r.passed else "✗ FAIL"
            summary.append(
                f"  - {r.creative_name} ({r.asset_type.value}): "
                f"{r.primary_kpi_lift:.1f}% lift, {status}"
            )
        
        context_parts.append("\n".join(summary))
    
    return "\n\n".join(context_parts)


def get_learnings_context(question: str) -> str:
    """Get relevant learnings from vector store."""
    try:
        vs = get_vector_store()
        
        # Query the learnings collection
        results = vs.query_learnings(question, n_results=3)
        
        if not results:
            return ""
        
        context_parts = []
        for r in results:
            source = r.get("metadata", {}).get("source_file", "Unknown source")
            context_parts.append(f"From {source}:\n{r['text']}")
        
        return "\n\n".join(context_parts)
    except Exception:
        return ""


def get_modeling_context(question: str) -> str:
    """Get relevant performance modeling insights."""
    if not MODELING_AVAILABLE:
        return ""
    
    try:
        modeling_service = get_modeling_service()
        
        if not modeling_service.analyses:
            return ""
        
        # Get the most recent analysis
        analysis_id = list(modeling_service.analyses.keys())[-1]
        analysis = modeling_service.analyses[analysis_id]
        
        # Build context from key findings
        context_parts = [
            f"Based on analysis of {analysis.videos_with_results} videos:",
            f"Average lift: {analysis.avg_lift:.1f}%",
            f"Pass rate: {analysis.pass_rate:.1f}%",
            "",
            "Key Findings:"
        ]
        
        for finding in analysis.key_findings[:5]:
            context_parts.append(f"• {finding}")
        
        if analysis.recommendations:
            context_parts.append("")
            context_parts.append("Recommendations:")
            for rec in analysis.recommendations[:3]:
                context_parts.append(f"• {rec}")
        
        return "\n".join(context_parts)
    except Exception:
        return ""


def get_no_context_response(question: str) -> str:
    """Provide helpful response when no context is available."""
    return """I don't have enough information to answer that question yet.

**To help me answer questions better:**

1. **For rules questions:** The default CT rules are loaded. Ask about budget limits, costs, or turnaround times.

2. **For results questions:** Upload test results in the **Results** tab first. Once uploaded, I can analyze them and answer questions like:
   - "Which creatives performed best?"
   - "What's our average pass rate?"
   - "Why did certain creatives fail?"

3. **For performance driver questions:** Run a video analysis and performance modeling to get insights like:
   - "What creative elements drive lift?"
   - "Does human presence impact performance?"
   - "Should we show the logo early or late?"

4. **For historical insights:** Have a superuser upload past learnings (PPTs, reports) in the **Admin** tab.

What would you like to know about the CT rules?"""
