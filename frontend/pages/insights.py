"""
Insights Page.

Chat interface for:
1. Asking questions about CT rules
2. Querying past results (from vector store)
3. Getting recommendations
4. Meta-analysis across campaigns
5. Performance driver analysis (creative features → lift)
6. ML model insights (trained predictors)
7. Persistent chat history
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

# Try to import persistence service
try:
    from services.persistence import get_persistence_service
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False

# Try to import prediction model for ML insights
try:
    from services.prediction_model import get_prediction_model
    PREDICTION_MODEL_AVAILABLE = True
except ImportError:
    PREDICTION_MODEL_AVAILABLE = False


def render_insights():
    """Render the Insights/Chat page."""
    st.markdown("## 💬 Insights & Q&A")
    st.markdown("Ask questions about CT rules, past results, creative performance drivers, or get recommendations.")
    
    # Show knowledge base status
    render_knowledge_status()
    
    # Initialize chat history - load from persistence if available
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = _load_chat_history()
    
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
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("#### Chat")
    with col2:
        if st.button("🗑️ Clear", help="Clear chat history"):
            st.session_state.chat_messages = []
            if PERSISTENCE_AVAILABLE:
                try:
                    persistence = get_persistence_service()
                    persistence.save_chat_history('global', [])
                except:
                    pass
            st.rerun()
    
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
    _append_message("user", question)
    
    # Get comprehensive response
    response = get_comprehensive_answer(question)
    
    # Add assistant message
    _append_message("assistant", response)
    
    st.rerun()


def get_comprehensive_answer(question: str) -> str:
    """
    Get a comprehensive answer by querying all available sources:
    1. CT Rules (structured + vector search)
    2. Past test results (from vector store)
    3. Session state results (current session)
    4. Past learnings (from vector store)
    5. Performance modeling insights (creative features → lift)
    6. ML model insights (trained feature importance)
    """
    
    # Classify the question to understand intent
    question_type = classify_question(question)
    question_lower = question.lower()
    
    # Check if this is a performance driver question
    is_driver_question = any(word in question_lower for word in [
        'driver', 'impact', 'affect', 'influence', 'human', 'logo', 'product',
        'cta', 'call to action', 'what makes', 'best performing', 'worst',
        'avoid', 'element', 'feature', 'presence', 'predictor', 'predict',
        'important', 'matters', 'correlation'
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
    sources_used = []
    
    # 1. Get rules context (always relevant)
    rules_context = get_rules_context(question)
    if rules_context:
        context_parts.append(f"**CT Rules & Guidelines:**\n{rules_context}")
        sources_used.append("CT Rules")
    
    # 2. Get vector store results context
    vector_context = get_vector_store_context(question)
    if vector_context:
        context_parts.append(f"**Historical Test Results (from knowledge base):**\n{vector_context}")
        sources_used.append("Historical Test Results")
    
    # 3. Get current session results context
    session_context = get_session_results_context(question)
    if session_context:
        context_parts.append(f"**Current Session Results:**\n{session_context}")
        sources_used.append("Current Session Results")
    
    # 4. Get past learnings context
    learnings_context = get_learnings_context(question)
    if learnings_context:
        context_parts.append(f"**Past Learnings:**\n{learnings_context}")
        sources_used.append("Past Learnings")
    
    # 5. Get performance modeling context if available
    if MODELING_AVAILABLE:
        modeling_context = get_modeling_context(question)
        if modeling_context:
            context_parts.append(f"**Performance Driver Analysis:**\n{modeling_context}")
            sources_used.append("Performance Driver Analysis")
    
    # 6. Get ML model insights (NEW)
    if PREDICTION_MODEL_AVAILABLE:
        ml_context = get_ml_model_context(question)
        if ml_context:
            context_parts.append(f"**ML Model Insights (Trained on Historical Data):**\n{ml_context}")
            sources_used.append("Trained ML Model")
    
    # If no context available, provide helpful guidance
    if not context_parts:
        return get_no_context_response(question)
    
    # Combine all context
    full_context = "\n\n".join(context_parts)
    sources_note = ", ".join(sources_used)
    
    # Generate comprehensive answer with strict grounding
    prompt = f"""Based ONLY on the following information about our creative testing program:

{full_context}

Please answer this question: {question}

CRITICAL INSTRUCTIONS:
1. ONLY use information provided above. Do NOT make up data or statistics.
2. If the data doesn't contain enough information to fully answer the question, say "Based on available data..." and only share what you know.
3. When citing numbers, specify which source they came from (e.g., "According to ML model analysis..." or "From historical test results...").
4. If you need to provide general context beyond the data, explicitly state: "Note: The following is general industry knowledge, not from your specific data: ..."
5. Do NOT invent creative names, lift percentages, or pass rates that aren't in the provided context.

Sources available for this answer: {sources_note}"""

    system_prompt = """You are an expert creative testing analyst. You help media agency teams understand their test results and optimize creative performance.

IMPORTANT RULES:
1. You must ONLY answer based on the data provided in the context. Never invent statistics or results.
2. If asked about something not in the context, clearly state that you don't have that information.
3. When the context includes "ML Model Insights", treat those as the most reliable source for feature importance and predictors.
4. Distinguish between:
   - FACTS from data (cite the source)
   - GENERAL KNOWLEDGE (explicitly label as "industry research suggests..." or "general best practice...")
5. Be specific: use actual creative names, actual percentages, actual metrics from the context.
6. If you're uncertain, say so. It's better to say "I don't have data on that" than to guess.

Your goal is to be helpful AND accurate. Accuracy comes first."""

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
        
        # Handle both dict and object formats
        if isinstance(results, dict):
            results_list = results.get('results', [])
            total_tested = len(results_list)
            passed_count = sum(1 for r in results_list if r.get('passed', False))
            pass_rate = (passed_count / total_tested * 100) if total_tested > 0 else 0
            
            # Try to get campaign name from results if not in plan
            if campaign_name == "Unknown Campaign":
                campaign_name = results.get('campaign_name', results.get('campaign', {}).get('name', 'Unknown Campaign'))
            
            summary = [
                f"Campaign: {campaign_name} ({brand_name})",
                f"Pass Rate: {pass_rate:.0f}% ({passed_count}/{total_tested})",
                "Results:"
            ]
            
            for r in results_list:
                status = "✓ PASS" if r.get('passed', False) else "✗ FAIL"
                creative_name = r.get('creative_name', 'Unknown')
                lift = r.get('awareness_lift_pct', r.get('primary_kpi_lift', 0))
                summary.append(f"  - {creative_name}: {lift:.1f}% lift, {status}")
        else:
            # Original object format
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


def get_ml_model_context(question: str) -> str:
    """Get insights from the trained ML prediction model."""
    if not PREDICTION_MODEL_AVAILABLE:
        return ""
    
    try:
        model = get_prediction_model()
        
        if not model.is_trained:
            return ""
        
        context_parts = []
        stats = model.training_stats
        
        # Model overview
        context_parts.append(f"Model trained on {stats.get('n_samples', 0)} historical creatives")
        context_parts.append(f"Historical pass rate: {stats.get('pass_rate', 0):.1f}%")
        context_parts.append(f"Model accuracy (LOOCV): {stats.get('loocv_accuracy', 0):.1f}%")
        context_parts.append(f"Precision: {stats.get('loocv_precision', 0):.1f}%, Recall: {stats.get('loocv_recall', 0):.1f}%")
        context_parts.append("")
        
        # Top predictors of pass/fail
        if model.learned_feature_importance:
            context_parts.append("TOP PREDICTORS OF CREATIVE SUCCESS (from trained model):")
            for i, (feature, importance) in enumerate(list(model.learned_feature_importance.items())[:10], 1):
                feature_display = feature.replace('_', ' ').replace('score', '').title().strip()
                context_parts.append(f"  {i}. {feature_display}: {importance*100:.1f}% importance")
        
        # Diagnostic model insights
        if model.diagnostic_models_trained and model.diagnostic_training_stats:
            context_parts.append("")
            context_parts.append("DIAGNOSTIC PREDICTION ACCURACY (video features → diagnostic scores):")
            for diag_name, diag_stats in model.diagnostic_training_stats.items():
                diag_display = diag_name.replace('_', ' ').title()
                r2 = diag_stats.get('cv_r2', 0)
                rmse = diag_stats.get('rmse', 0)
                top_preds = diag_stats.get('top_predictors', [])[:3]
                pred_names = [p[0].replace('_', ' ') for p in top_preds]
                context_parts.append(f"  • {diag_display}: R²={r2:.2f}, RMSE=±{rmse:.1f}")
                if pred_names:
                    context_parts.append(f"    Driven by: {', '.join(pred_names)}")
        
        # Video features that matter
        context_parts.append("")
        context_parts.append("VIDEO FEATURES ANALYZED:")
        context_parts.append("  • Human presence (in opening, frame ratio, looking at camera)")
        context_parts.append("  • Brand elements (logo timing, logo frequency, product visibility)")
        context_parts.append("  • Engagement elements (CTA presence, CTA timing)")
        context_parts.append("  • Content style (emotional content, positive emotions, scene diversity)")
        
        return "\n".join(context_parts)
    except Exception as e:
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


def _load_chat_history() -> list:
    """Load chat history from persistence."""
    if PERSISTENCE_AVAILABLE:
        try:
            persistence = get_persistence_service()
            messages = persistence.load_chat_history('global')
            return messages
        except Exception:
            pass
    return []


def _save_chat_history():
    """Save chat history to persistence."""
    if PERSISTENCE_AVAILABLE:
        try:
            persistence = get_persistence_service()
            persistence.save_chat_history('global', st.session_state.chat_messages)
        except Exception:
            pass


def _append_message(role: str, content: str):
    """Append a message to chat history and save."""
    st.session_state.chat_messages.append({
        'role': role,
        'content': content,
    })
    _save_chat_history()

