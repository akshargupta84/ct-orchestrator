"""
Results Page.

Allows users to:
1. View existing test plans
2. Upload test results CSV
3. View analysis and recommendations
4. Generate reports
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.csv_parser import CSVParser, generate_sample_csv
from agents.analysis_agent import AnalysisAgent
from services.report_generator import ReportGenerator
from services.vector_store import get_vector_store
from models import KPIType, TestResults


def save_results_to_vector_store(results: TestResults, campaign: dict, plan_id: str):
    """
    Save test results to the vector store for future querying.
    
    This enables the Insights chat to reference past results.
    """
    try:
        vs = get_vector_store()
        
        # Build a comprehensive summary for embedding
        summary_parts = [
            f"Campaign: {campaign.get('name', 'Unknown')}",
            f"Brand: {campaign.get('brand', {}).get('name', 'Unknown')}",
            f"Primary KPI: {campaign.get('primary_kpi', 'awareness')}",
            f"Total Creatives Tested: {results.total_creatives_tested}",
            f"Pass Rate: {results.pass_rate * 100:.0f}%",
            f"Passed: {results.creatives_passed}, Failed: {results.creatives_failed}",
            "",
            "Individual Results:"
        ]
        
        for r in results.results:
            result_status = "PASSED" if r.passed else "FAILED"
            summary_parts.append(
                f"- {r.creative_name} ({r.asset_type.value}): "
                f"{r.primary_kpi_lift:.1f}% lift in {r.primary_kpi.value}, "
                f"{'statistically significant' if r.primary_kpi_stat_sig else 'not significant'}, "
                f"{result_status}"
            )
            
            # Add diagnostic info for failed creatives
            if not r.passed and r.diagnostics:
                low_diagnostics = [d for d in r.diagnostics if d.value < 60]
                if low_diagnostics:
                    diag_str = ", ".join([f"{d.name}={d.value:.0f}" for d in low_diagnostics[:3]])
                    summary_parts.append(f"  Low diagnostics: {diag_str}")
        
        # Add insights
        summary_parts.extend([
            "",
            "Key Insights:",
            f"- Best performer: {max(results.results, key=lambda x: x.primary_kpi_lift).creative_name}",
            f"- Worst performer: {min(results.results, key=lambda x: x.primary_kpi_lift).creative_name}",
        ])
        
        # Calculate video vs display performance
        video_results = [r for r in results.results if r.asset_type.value == "video"]
        display_results = [r for r in results.results if r.asset_type.value == "display"]
        
        if video_results:
            video_pass_rate = sum(1 for r in video_results if r.passed) / len(video_results)
            summary_parts.append(f"- Video pass rate: {video_pass_rate * 100:.0f}%")
        
        if display_results:
            display_pass_rate = sum(1 for r in display_results if r.passed) / len(display_results)
            summary_parts.append(f"- Display pass rate: {display_pass_rate * 100:.0f}%")
        
        summary = "\n".join(summary_parts)
        
        # Save to vector store
        vs.add_result_summary(
            campaign_id=campaign.get('id', plan_id),
            summary=summary,
            brand=campaign.get('brand', {}).get('name'),
            pass_rate=results.pass_rate,
            date=datetime.now().strftime("%Y-%m-%d")
        )
        
    except Exception as e:
        # Don't fail the main operation if vector store fails
        st.warning(f"Note: Could not save to knowledge base: {e}")


def render_results():
    """Render the Results page."""
    st.markdown("## 📊 Test Results & Recommendations")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Test Plans", "📤 Upload Results", "📈 Analysis"])
    
    with tab1:
        render_plans_list()
    
    with tab2:
        render_results_upload()
    
    with tab3:
        render_analysis()


def render_plans_list():
    """Render list of test plans."""
    st.markdown("### Active Test Plans")
    
    if not st.session_state.test_plans:
        st.info("No test plans created yet. Go to CT Planner to create one.")
        return
    
    for plan_id, plan in st.session_state.test_plans.items():
        campaign = plan.get("campaign", {})
        creatives = plan.get("creatives", [])
        
        with st.expander(f"**{campaign.get('name', 'Unnamed')}** - {plan.get('status', 'unknown').title()}", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Brand:** {campaign.get('brand', {}).get('name', 'N/A')}")
                st.markdown(f"**Budget:** ${campaign.get('budget', 0):,.0f}")
            
            with col2:
                videos = len([c for c in creatives if c.get("asset_type") == "video"])
                displays = len([c for c in creatives if c.get("asset_type") == "display"])
                st.markdown(f"**Creatives:** {videos} video, {displays} display")
                st.markdown(f"**Primary KPI:** {campaign.get('primary_kpi', 'N/A').replace('_', ' ').title()}")
            
            with col3:
                st.markdown(f"**Created:** {plan.get('created_at', 'N/A')}")
                
                # Check if results exist
                if plan_id in st.session_state.test_results:
                    st.success("✓ Results uploaded")
                else:
                    st.warning("⏳ Awaiting results")


def render_results_upload():
    """Render results upload interface."""
    st.markdown("### Upload Test Results")
    
    # Plan selection
    if not st.session_state.test_plans:
        st.info("No test plans available. Create a plan first.")
        return
    
    plan_options = {
        plan_id: f"{plan.get('campaign', {}).get('name', 'Unnamed')} ({plan_id})"
        for plan_id, plan in st.session_state.test_plans.items()
    }
    
    selected_plan_id = st.selectbox(
        "Select Test Plan",
        options=list(plan_options.keys()),
        format_func=lambda x: plan_options[x]
    )
    
    if not selected_plan_id:
        return
    
    selected_plan = st.session_state.test_plans[selected_plan_id]
    campaign = selected_plan.get("campaign", {})
    
    st.markdown(f"**Campaign:** {campaign.get('name')} | **Primary KPI:** {campaign.get('primary_kpi', 'awareness').replace('_', ' ').title()}")
    
    st.markdown("---")
    
    # Sample CSV download
    st.markdown("#### CSV Format")
    st.markdown("Download a sample CSV to see the expected format:")
    
    sample_csv = generate_sample_csv()
    st.download_button(
        "Download Sample CSV",
        data=sample_csv,
        file_name="sample_results.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Results CSV",
        type=["csv"],
        help="Upload the CSV file from the testing vendor"
    )
    
    if uploaded_file:
        # Parse CSV
        primary_kpi = KPIType(campaign.get("primary_kpi", "awareness"))
        parser = CSVParser(primary_kpi=primary_kpi)
        
        result = parser.parse(
            uploaded_file,
            campaign_id=campaign.get("id", "unknown"),
            test_plan_id=selected_plan_id
        )
        
        if not result.success:
            st.error("Failed to parse CSV:")
            for error in result.errors:
                st.error(f"- {error.error}")
            return
        
        if result.warnings:
            for warning in result.warnings:
                st.warning(warning)
        
        # Show preview
        st.markdown("#### Results Preview")
        
        preview_data = []
        for r in result.results.results:
            preview_data.append({
                "Creative": r.creative_name,
                "Type": r.asset_type.value.title(),
                f"{primary_kpi.value.replace('_', ' ').title()} Lift": f"{r.primary_kpi_lift:.1f}%",
                "Stat Sig": "Yes" if r.primary_kpi_stat_sig else "No",
                "Result": "✓ PASS" if r.passed else "✗ FAIL"
            })
        
        st.dataframe(preview_data, use_container_width=True, hide_index=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tested", result.results.total_creatives_tested)
        with col2:
            st.metric("Passed", result.results.creatives_passed)
        with col3:
            st.metric("Failed", result.results.creatives_failed)
        with col4:
            st.metric("Pass Rate", f"{result.results.pass_rate * 100:.0f}%")
        
        # Save button
        if st.button("Save Results & Generate Analysis", type="primary"):
            # Save to session state
            st.session_state.test_results[selected_plan_id] = result.results
            
            # Save to vector store for future querying
            save_results_to_vector_store(
                results=result.results,
                campaign=campaign,
                plan_id=selected_plan_id
            )
            
            st.success("Results saved! View the Analysis tab for recommendations.")
            st.info("💡 These results are now searchable in the Insights tab.")
            st.rerun()


def render_analysis():
    """Render analysis and recommendations."""
    st.markdown("### Analysis & Recommendations")
    
    if not st.session_state.test_results:
        st.info("No results uploaded yet. Upload results in the 'Upload Results' tab.")
        return
    
    # Select results to analyze
    result_options = {}
    for plan_id, results in st.session_state.test_results.items():
        plan = st.session_state.test_plans.get(plan_id, {})
        campaign_name = plan.get("campaign", {}).get("name", "Unknown")
        result_options[plan_id] = f"{campaign_name} ({results.total_creatives_tested} creatives)"
    
    selected_result_id = st.selectbox(
        "Select Results to Analyze",
        options=list(result_options.keys()),
        format_func=lambda x: result_options[x]
    )
    
    if not selected_result_id:
        return
    
    results = st.session_state.test_results[selected_result_id]
    plan = st.session_state.test_plans.get(selected_result_id, {})
    campaign = plan.get("campaign", {})
    
    st.markdown("---")
    
    # Run analysis
    with st.spinner("Analyzing results..."):
        agent = AnalysisAgent()
        recommendations = agent.analyze_results(results)
    
    # Summary
    st.markdown("#### Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### ✓ Run These")
        if recommendations.run_creatives:
            for cid in recommendations.run_creatives:
                rec = next((r for r in recommendations.recommendations if r.creative_id == cid), None)
                if rec:
                    st.success(f"**{rec.creative_name}**")
                    st.caption(rec.rationale[:100] + "..." if len(rec.rationale) > 100 else rec.rationale)
        else:
            st.warning("No creatives passed")
    
    with col2:
        st.markdown("##### ✗ Do Not Run")
        if recommendations.do_not_run_creatives:
            for cid in recommendations.do_not_run_creatives:
                rec = next((r for r in recommendations.recommendations if r.creative_id == cid), None)
                if rec:
                    st.error(f"**{rec.creative_name}**")
                    st.caption(rec.rationale[:100] + "..." if len(rec.rationale) > 100 else rec.rationale)
        else:
            st.success("All creatives passed!")
    
    with col3:
        st.markdown("##### 🔄 Optimize & Retest")
        if recommendations.optimize_creatives:
            for cid in recommendations.optimize_creatives:
                rec = next((r for r in recommendations.recommendations if r.creative_id == cid), None)
                if rec:
                    st.warning(f"**{rec.creative_name}**")
                    st.caption(rec.rationale[:100] + "..." if len(rec.rationale) > 100 else rec.rationale)
        else:
            st.info("No creatives flagged for optimization")
    
    st.markdown("---")
    
    # Detailed recommendations
    st.markdown("#### Detailed Analysis")
    
    for rec in recommendations.recommendations:
        status_color = "🟢" if rec.recommendation == "run" else "🔴" if rec.recommendation == "do_not_run" else "🟡"
        
        with st.expander(f"{status_color} {rec.creative_name} - {rec.recommendation.replace('_', ' ').title()}"):
            st.markdown(f"**Confidence:** {rec.confidence * 100:.0f}%")
            st.markdown(f"**Rationale:** {rec.rationale}")
            
            if rec.diagnostic_insights:
                st.markdown("**Diagnostic Issues:**")
                for insight in rec.diagnostic_insights:
                    st.markdown(f"- {insight}")
            
            if rec.suggested_improvements:
                st.markdown("**Suggested Improvements:**")
                for improvement in rec.suggested_improvements:
                    st.markdown(f"- {improvement}")
    
    st.markdown("---")
    
    # Long-term recommendations
    if recommendations.long_term_recommendations:
        st.markdown("#### Long-Term Recommendations")
        for rec in recommendations.long_term_recommendations:
            st.info(rec)
    
    # Meta insights
    if recommendations.meta_insights:
        st.markdown("#### Meta Insights")
        for insight in recommendations.meta_insights:
            st.markdown(f"- {insight}")
    
    st.markdown("---")
    
    # Report generation
    st.markdown("#### Generate Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate PowerPoint Report", type="primary"):
            try:
                generator = ReportGenerator()
                report_path = generator.generate_report(
                    results=results,
                    recommendations=recommendations,
                    campaign_name=campaign.get("name", "Campaign"),
                    brand_name=campaign.get("brand", {}).get("name", "Brand")
                )
                
                with open(report_path, "rb") as f:
                    st.download_button(
                        "Download PowerPoint",
                        data=f,
                        file_name=f"CT_Report_{campaign.get('name', 'Campaign').replace(' ', '_')}.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    )
                
                st.success("Report generated!")
                
            except Exception as e:
                st.warning(f"Could not generate PowerPoint: {e}")
                st.info("Generating markdown report instead...")
                
                markdown_report = generator.generate_simple_report(
                    results=results,
                    recommendations=recommendations,
                    campaign_name=campaign.get("name", "Campaign")
                )
                
                st.download_button(
                    "Download Markdown Report",
                    data=markdown_report,
                    file_name=f"CT_Report_{campaign.get('name', 'Campaign').replace(' ', '_')}.md",
                    mime="text/markdown"
                )
    
    with col2:
        if st.button("Generate Detailed Analysis (AI)"):
            with st.spinner("Generating detailed analysis..."):
                detailed = agent.generate_detailed_analysis(results)
            
            st.markdown("#### AI-Generated Analysis")
            st.markdown(detailed)
