"""
Results Page.

Allows users to:
1. View existing test plans
2. Upload test results CSV
3. View analysis and recommendations
4. Generate reports
5. Save results with persistence
6. Advanced AI-powered analysis
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

# Try to import persistence service
try:
    from services.persistence import get_persistence_service
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False

# Try to import advanced analytics
try:
    from services.advanced_analytics import get_analytics_service, AdvancedAnalysisResult
    ADVANCED_ANALYTICS_AVAILABLE = True
except ImportError:
    ADVANCED_ANALYTICS_AVAILABLE = False

# Try to import LLM utilities
try:
    from utils.llm import is_llm_available, synthesize_analysis
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


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
        
        # Get quarter info
        quarter = campaign.get('quarter', '')
        year = campaign.get('year', '')
        quarter_str = f" ({quarter} {year})" if quarter and year else ""
        
        with st.expander(f"**{campaign.get('name', 'Unnamed')}{quarter_str}** - {plan.get('status', 'unknown').title()}", expanded=False):
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
                
                # Check if results exist - extract campaign_id from plan_id
                campaign_id = campaign.get('id', plan_id.split('_2')[0] if '_2' in plan_id else plan_id)
                has_results = (
                    plan_id in st.session_state.test_results or 
                    campaign_id in st.session_state.test_results
                )
                
                if has_results:
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
        # Parse CSV - handle KPI type conversion gracefully
        kpi_value = campaign.get("primary_kpi", "awareness")
        
        # Map common variations to valid enum values
        kpi_mapping = {
            "brand_awareness": "awareness",
            "brand awareness": "awareness",
            "awareness": "awareness",
            "consideration": "consideration",
            "purchase_intent": "purchase_intent",
            "purchase intent": "purchase_intent",
            "ad_recall": "ad_recall",
            "ad recall": "ad_recall",
        }
        kpi_value = kpi_mapping.get(kpi_value.lower(), "awareness") if isinstance(kpi_value, str) else "awareness"
        
        try:
            primary_kpi = KPIType(kpi_value)
        except ValueError:
            primary_kpi = KPIType.AWARENESS  # Default fallback
        
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
            
            # Save to disk if persistence is available
            if PERSISTENCE_AVAILABLE:
                try:
                    persistence = get_persistence_service()
                    
                    # Convert results to dict for JSON serialization
                    results_dict = {
                        'total_creatives_tested': result.results.total_creatives_tested,
                        'creatives_passed': result.results.creatives_passed,
                        'creatives_failed': result.results.creatives_failed,
                        'pass_rate': result.results.pass_rate,
                        'results': [
                            {
                                'creative_name': r.creative_name,
                                'creative_id': r.creative_id,
                                'asset_type': r.asset_type.value,
                                'channel': r.channel.value if r.channel else None,
                                'primary_kpi_lift': r.primary_kpi_lift,
                                'primary_kpi_stat_sig': r.primary_kpi_stat_sig,
                                'passed': r.passed,
                            }
                            for r in result.results.results
                        ],
                        'campaign': campaign,
                        'plan_id': selected_plan_id,
                    }
                    
                    # Get raw CSV content
                    uploaded_file.seek(0)
                    raw_csv = uploaded_file.read()
                    
                    results_id = persistence.save_results(
                        campaign_id=campaign.get('id', 'unknown'),
                        results_data=results_dict,
                        raw_csv_content=raw_csv
                    )
                    st.success(f"Results saved! ✅ Persisted to disk (ID: {results_id})")
                except Exception as e:
                    st.warning(f"Results saved to session but disk persistence failed: {e}")
            else:
                st.success("Results saved! View the Analysis tab for recommendations.")
            
            st.info("💡 These results are now searchable in the Insights tab.")
            st.rerun()


def render_analysis():
    """Render analysis and recommendations with advanced analytics."""
    st.markdown("### Analysis & Recommendations")
    
    if not st.session_state.test_results:
        st.info("No results uploaded yet. Upload results in the 'Upload Results' tab.")
        return
    
    # Select results to analyze
    result_options = {}
    for plan_id, results in st.session_state.test_results.items():
        plan = st.session_state.test_plans.get(plan_id, {})
        campaign_name = plan.get("campaign", {}).get("name", "")
        quarter = plan.get("campaign", {}).get("quarter", "")
        year = plan.get("campaign", {}).get("year", "")
        
        # If no campaign name from plan, try to get from results
        if not campaign_name and isinstance(results, dict):
            campaign_name = results.get('campaign_name', '')
            if not campaign_name:
                campaign_name = results.get('campaign', {}).get('name', '')
        
        # Get quarter from results if not in plan
        if not quarter and isinstance(results, dict):
            quarter = results.get('campaign', {}).get('quarter', '')
            if not quarter:
                # Try to extract from test_date
                test_date = results.get('test_date', '')
                if test_date:
                    try:
                        month = int(test_date.split('-')[1])
                        quarter = f"Q{(month - 1) // 3 + 1}"
                    except:
                        pass
        
        if not year and isinstance(results, dict):
            year = results.get('campaign', {}).get('year', '')
            if not year:
                test_date = results.get('test_date', '')
                if test_date:
                    try:
                        year = test_date.split('-')[0]
                    except:
                        pass
        
        if not campaign_name:
            campaign_name = "Unknown"
        
        # Build display name with quarter
        quarter_str = f" ({quarter} {year})" if quarter and year else (f" ({quarter})" if quarter else "")
        
        # Handle both TestResults object and dict
        if hasattr(results, 'total_creatives_tested'):
            count = results.total_creatives_tested
        elif isinstance(results, dict):
            count = len(results.get('results', []))
        else:
            count = 0
            
        result_options[plan_id] = f"{campaign_name}{quarter_str} ({count} creatives)"
    
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
    
    # Convert results to dict format for analysis
    if hasattr(results, 'results'):
        results_data = {
            'results': [
                {
                    'creative_name': r.creative_name,
                    'creative_id': r.creative_id,
                    'passed': r.passed,
                    'primary_kpi_lift': r.primary_kpi_lift,
                    'primary_kpi_stat_sig': r.primary_kpi_stat_sig,
                    'awareness_lift_pct': getattr(r, 'awareness_lift', 0),
                    'awareness_significant': getattr(r, 'awareness_stat_sig', False),
                }
                for r in results.results
            ]
        }
        # Add diagnostic scores if available
        for i, r in enumerate(results.results):
            if hasattr(r, 'diagnostics') and r.diagnostics:
                for d in r.diagnostics:
                    results_data['results'][i][d.name] = d.value
    elif isinstance(results, dict):
        results_data = results
    else:
        st.error("Invalid results format")
        return
    
    st.markdown("---")
    
    # Run advanced analysis if available
    advanced_result = None
    if ADVANCED_ANALYTICS_AVAILABLE:
        with st.spinner("Running advanced analysis..."):
            try:
                analytics = get_analytics_service()
                advanced_result = analytics.analyze(results_data)
            except Exception as e:
                st.warning(f"Advanced analytics error: {e}")
    
    # Basic analysis fallback
    agent = AnalysisAgent()
    if hasattr(results, 'results'):
        recommendations = agent.analyze_results(results)
    else:
        recommendations = None
    
    # Summary using advanced analysis if available
    st.markdown("#### Summary")
    
    if advanced_result and advanced_result.creative_analyses:
        _render_advanced_summary(advanced_result)
    elif recommendations:
        _render_basic_summary(recommendations)
    
    st.markdown("---")
    
    # Detailed recommendations
    st.markdown("#### Detailed Analysis")
    
    if advanced_result and advanced_result.creative_analyses:
        _render_advanced_details(advanced_result)
    elif recommendations:
        _render_basic_details(recommendations)
    
    st.markdown("---")
    
    # Advanced insights section
    if advanced_result:
        _render_advanced_insights(advanced_result)
        st.markdown("---")
    
    # Report generation
    _render_report_section(results, recommendations, advanced_result, campaign)


def _render_advanced_summary(result: 'AdvancedAnalysisResult'):
    """Render summary using advanced analysis results."""
    col1, col2, col3 = st.columns(3)
    
    run_creatives = [a for a in result.creative_analyses if a.category == "run"]
    do_not_run = [a for a in result.creative_analyses if a.category == "do_not_run"]
    optimize = [a for a in result.creative_analyses if a.category == "optimize_retest"]
    
    with col1:
        st.markdown("##### ✓ Run These")
        if run_creatives:
            for a in run_creatives:
                st.success(f"**{a.creative_name}**")
                st.caption(f"Lift: {a.lift:.1f}% (stat sig) | {a.percentile_rank:.0f}th percentile")
                if a.strong_areas:
                    strengths = ", ".join([s['metric'].replace('_score', '').replace('_', ' ') for s in a.strong_areas[:2]])
                    st.caption(f"Strengths: {strengths}")
        else:
            st.warning("No creatives passed")
    
    with col2:
        st.markdown("##### ✗ Do Not Run")
        if do_not_run:
            for a in do_not_run:
                st.error(f"**{a.creative_name}**")
                st.caption(f"Lift: {a.lift:.1f}% | Stat sig negative or near-zero lift")
                if a.weak_areas:
                    weaknesses = ", ".join([w['metric'].replace('_score', '').replace('_', ' ') for w in a.weak_areas[:2]])
                    st.caption(f"Weak areas: {weaknesses}")
        else:
            st.success("No creatives in this category")
    
    with col3:
        st.markdown("##### 🔄 Optimize & Retest")
        if optimize:
            for a in optimize:
                st.warning(f"**{a.creative_name}**")
                st.caption(f"Lift: {a.lift:.1f}% (not stat sig)")
                if a.specific_recommendations:
                    rec = a.specific_recommendations[0]
                    st.caption(f"Fix: {rec['recommendation'][:80]}...")
        else:
            st.info("No creatives flagged for optimization")


def _render_basic_summary(recommendations):
    """Render basic summary from analysis agent."""
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
            st.success("No creatives in this category")
    
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


def _render_advanced_details(result: 'AdvancedAnalysisResult'):
    """Render detailed analysis using advanced analytics."""
    for analysis in result.creative_analyses:
        if analysis.category == "run":
            status_icon = "🟢"
            status_text = "Run"
        elif analysis.category == "do_not_run":
            status_icon = "🔴"
            status_text = "Do Not Run"
        else:
            status_icon = "🟡"
            status_text = "Optimize & Retest"
        
        with st.expander(f"{status_icon} {analysis.creative_name} - {status_text}"):
            # Metrics row
            cols = st.columns(4)
            with cols[0]:
                st.metric("Lift", f"{analysis.lift:.1f}%")
            with cols[1]:
                st.metric("Stat Sig", "Yes" if analysis.stat_sig else "No")
            with cols[2]:
                st.metric("Percentile", f"{analysis.percentile_rank:.0f}th")
            with cols[3]:
                st.metric("Pass Prob", f"{analysis.predicted_pass_probability * 100:.0f}%")
            
            # Diagnostic breakdown
            if analysis.diagnostic_benchmarks:
                st.markdown("**Diagnostic Scores vs Benchmarks:**")
                diag_data = []
                for metric, bench in analysis.diagnostic_benchmarks.items():
                    score = bench.get('score', 0)
                    benchmark = bench.get('benchmark', 50)
                    vs_bench = bench.get('vs_benchmark', 0)
                    
                    status = "✅" if vs_bench >= 0 else "⚠️" if vs_bench > -10 else "❌"
                    diag_data.append({
                        'Status': status,
                        'Metric': metric.replace('_score', '').replace('_', ' ').title(),
                        'Score': score,
                        'Benchmark': benchmark,
                        'vs Bench': f"{vs_bench:+.0f}",
                    })
                
                st.dataframe(pd.DataFrame(diag_data), hide_index=True, use_container_width=True)
            
            # Strengths and Weaknesses
            col1, col2 = st.columns(2)
            with col1:
                if analysis.strong_areas:
                    st.markdown("**Strengths:**")
                    for s in analysis.strong_areas:
                        st.markdown(f"- ✅ {s['metric'].replace('_score', '').replace('_', ' ').title()}: {s['score']:.0f}")
            
            with col2:
                if analysis.weak_areas:
                    st.markdown("**Areas for Improvement:**")
                    for w in analysis.weak_areas:
                        st.markdown(f"- ⚠️ {w['metric'].replace('_score', '').replace('_', ' ').title()}: {w['score']:.0f} (benchmark: {w['benchmark']})")
            
            # Specific recommendations
            if analysis.specific_recommendations:
                st.markdown("**Recommendations:**")
                for rec in analysis.specific_recommendations:
                    priority = rec.get('priority', 'medium')
                    if priority == 'high':
                        st.markdown(f"- 🔴 **{rec['recommendation']}**")
                    else:
                        st.markdown(f"- {rec['recommendation']}")
                    if rec.get('rationale'):
                        st.caption(f"  _{rec['rationale']}_")


def _render_basic_details(recommendations):
    """Render basic detailed analysis."""
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


def _render_advanced_insights(result: 'AdvancedAnalysisResult'):
    """Render advanced insights section."""
    st.markdown("#### 🔬 Advanced Insights")
    
    # Key findings
    if result.key_findings:
        st.markdown("##### Key Findings")
        for finding in result.key_findings:
            st.info(f"**{finding['finding']}**\n\n_{finding['implication']}_")
    
    # Statistical analysis
    tabs = st.tabs(["📊 Statistics", "📈 Patterns", "🎯 Optimization Playbook"])
    
    with tabs[0]:
        stats = result.statistical
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Lift Statistics:**")
            if stats.lift_statistics:
                for key, val in stats.lift_statistics.items():
                    st.write(f"- {key.title()}: {val}%")
            
            if stats.confidence_intervals.get('lift_95ci'):
                ci = stats.confidence_intervals['lift_95ci']
                st.write(f"- 95% CI: [{ci['lower']}, {ci['upper']}]%")
        
        with col2:
            st.markdown("**Correlations with Lift:**")
            if stats.correlations:
                corr_data = [
                    {
                        'Metric': k.replace('_score', '').replace('_', ' ').title(),
                        'Correlation': v,
                        'Strength': '🟢 Strong' if abs(v) > 0.5 else '🟡 Moderate' if abs(v) > 0.3 else '⚪ Weak'
                    }
                    for k, v in stats.correlations.items()
                    if isinstance(v, (int, float))
                ]
                if corr_data:
                    st.dataframe(pd.DataFrame(corr_data), hide_index=True)
        
        if stats.significant_differences:
            st.markdown("**Significant Differences (Pass vs Fail):**")
            diff_data = [
                {
                    'Metric': d['metric'].replace('_score', '').replace('_', ' ').title(),
                    'Passed Avg': d['passed_mean'],
                    'Failed Avg': d['failed_mean'],
                    'Difference': d['difference'],
                    'Effect': d['effect_magnitude'].title(),
                }
                for d in stats.significant_differences
            ]
            st.dataframe(pd.DataFrame(diff_data), hide_index=True, use_container_width=True)
    
    with tabs[1]:
        patterns = result.patterns
        
        if patterns.feature_importance:
            st.markdown("**Feature Importance (Random Forest):**")
            imp_data = [
                {'Feature': k.replace('_score', '').replace('_', ' ').title(), 'Importance': f"{v*100:.1f}%"}
                for k, v in patterns.feature_importance.items()
            ]
            st.dataframe(pd.DataFrame(imp_data), hide_index=True)
        
        if patterns.decision_rules:
            st.markdown("**Decision Rules:**")
            for rule in patterns.decision_rules:
                st.code(rule['readable'], language=None)
        
        if patterns.clusters:
            st.markdown("**Creative Clusters:**")
            for cluster in patterns.clusters:
                with st.expander(f"Cluster {cluster['cluster_id'] + 1}: {cluster['pass_rate']}% pass rate"):
                    st.write(f"**Size:** {cluster['size']} creatives")
                    st.write(f"**Avg Lift:** {cluster['avg_lift']}%")
                    st.write(f"**Creatives:** {', '.join(cluster['creatives'])}")
                    if cluster['characteristics']:
                        st.write("**Characteristics:**")
                        for k, v in cluster['characteristics'].items():
                            st.write(f"  - {k.replace('_score', '').replace('_', ' ').title()}: {v}")
    
    with tabs[2]:
        if result.optimization_playbook:
            st.markdown("**Optimization Playbook**")
            st.markdown("_Based on pattern analysis, focus on these areas:_")
            
            for item in result.optimization_playbook:
                with st.expander(f"📌 {item['metric']} (Importance: {item['importance']})"):
                    st.write(f"**Target Threshold:** {item.get('threshold', 'N/A')}")
                    st.write("**Recommendations:**")
                    for rec in item.get('recommendations', []):
                        st.write(f"- {rec}")
        else:
            st.info("Optimization playbook requires more data to generate meaningful recommendations.")


def _render_report_section(results, recommendations, advanced_result, campaign):
    """Render report generation section."""
    st.markdown("#### Generate Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate PowerPoint Report", type="secondary"):
            try:
                # Convert dict to TestResults if needed
                if isinstance(results, dict):
                    from models import TestResults, CreativeTestResult, AssetType, KPIType, DiagnosticMetric
                    
                    results_list = results.get('results', [])
                    creative_results = []
                    for r in results_list:
                        # Build diagnostic metrics
                        diagnostics = []
                        for score_name in ['attention_score', 'brand_recall_score', 'message_clarity_score', 
                                          'emotional_resonance_score', 'uniqueness_score']:
                            if score_name in r and r[score_name]:
                                diagnostics.append(DiagnosticMetric(
                                    name=score_name.replace('_score', ''),
                                    value=float(r[score_name]),
                                    benchmark=65,
                                ))
                        
                        creative_results.append(CreativeTestResult(
                            creative_id=r.get('creative_id', 'unknown'),
                            creative_name=r.get('creative_name', 'Unknown'),
                            asset_type=AssetType.VIDEO,
                            control_awareness=0,
                            control_consideration=0,
                            exposed_awareness=r.get('awareness_lift_pct', 0),
                            exposed_consideration=r.get('consideration_lift_pct', 0),
                            awareness_lift=r.get('awareness_lift_pct', 0),
                            consideration_lift=r.get('consideration_lift_pct', 0),
                            awareness_stat_sig=r.get('awareness_significant', False),
                            consideration_stat_sig=r.get('consideration_significant', False),
                            primary_kpi=KPIType.AWARENESS,
                            primary_kpi_lift=r.get('awareness_lift_pct', r.get('primary_kpi_lift', 0)),
                            primary_kpi_stat_sig=r.get('awareness_significant', r.get('primary_kpi_stat_sig', False)),
                            passed=r.get('passed', False),
                            diagnostics=diagnostics,
                            control_sample_size=750,
                            exposed_sample_size=400,
                        ))
                    
                    results_obj = TestResults(
                        id=results.get('campaign_id', 'unknown'),
                        campaign_id=results.get('campaign_id', 'unknown'),
                        test_plan_id=results.get('plan_id', 'unknown'),
                        results=creative_results,
                        total_creatives_tested=len(creative_results),
                        creatives_passed=sum(1 for r in creative_results if r.passed),
                        creatives_failed=sum(1 for r in creative_results if not r.passed),
                    )
                else:
                    results_obj = results
                
                generator = ReportGenerator()
                report_path = generator.generate_report(
                    results=results_obj,
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
                st.info("Try the Markdown report option or AI-generated analysis instead.")
    
    with col2:
        # AI-powered detailed analysis
        if st.button("🤖 Generate AI Analysis", type="primary"):
            if not LLM_AVAILABLE or not is_llm_available():
                st.error("AI analysis requires Anthropic API key. Set ANTHROPIC_API_KEY environment variable or configure in Admin page.")
            elif not advanced_result:
                st.error("Advanced analytics must run first. Please refresh the page.")
            else:
                with st.spinner("Generating AI-powered analysis (this may take 30-60 seconds)..."):
                    try:
                        # Prepare analysis data for Claude
                        analysis_dict = {
                            'statistical': advanced_result.statistical.__dict__ if hasattr(advanced_result.statistical, '__dict__') else {},
                            'historical': advanced_result.historical.__dict__ if hasattr(advanced_result.historical, '__dict__') else {},
                            'patterns': advanced_result.patterns.__dict__ if hasattr(advanced_result.patterns, '__dict__') else {},
                            'recommendations': [
                                a.__dict__ if hasattr(a, '__dict__') else a
                                for a in advanced_result.creative_analyses
                            ],
                            'raw_data': {
                                'primary_kpi': 'awareness',
                                'results_df': [
                                    a.__dict__ if hasattr(a, '__dict__') else a
                                    for a in advanced_result.creative_analyses
                                ],
                            }
                        }
                        
                        ai_report = synthesize_analysis(
                            analysis_results=analysis_dict,
                            campaign_name=campaign.get("name", "Unknown Campaign")
                        )
                        
                        st.markdown("#### 🤖 AI-Generated Analysis Report")
                        st.markdown(ai_report)
                        
                        # Download option
                        st.download_button(
                            "📥 Download Report (Markdown)",
                            data=ai_report,
                            file_name=f"AI_Analysis_{campaign.get('name', 'Campaign').replace(' ', '_')}.md",
                            mime="text/markdown"
                        )
                        
                    except ValueError as e:
                        st.error(f"API Error: {e}")
                    except Exception as e:
                        st.error(f"Error generating AI analysis: {e}")
