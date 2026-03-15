"""
Creative Scorer Page.

Pre-test creative scoring using local AI vision models.
Predicts test outcomes before actual testing.
"""

import streamlit as st
import os
import tempfile
import base64
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Try to import scoring services
try:
    from services.creative_scorer import (
        CreativeScorerService, 
        get_scorer_service,
        ScoringConfig,
        ScoringProgress,
        ScoringResult,
    )
    SCORER_AVAILABLE = True
except ImportError as e:
    SCORER_AVAILABLE = False
    IMPORT_ERROR = str(e)


def render():
    """Render the Creative Scorer page."""
    st.markdown("## 🔮 Pre-Test Creative Scorer")
    st.markdown("*Predict creative test outcomes before spending on testing*")
    
    if not SCORER_AVAILABLE:
        st.error(f"Creative Scorer not available: {IMPORT_ERROR}")
        st.info("Make sure all dependencies are installed: pip install ollama opencv-python")
        return
    
    # Initialize scorer
    scorer = get_scorer_service()
    
    # Check availability
    availability = scorer.check_availability()
    
    # Status bar
    _render_status_bar(availability)
    
    if not availability['ready']:
        _render_setup_instructions(availability)
        return
    
    st.markdown("---")
    
    # Main interface
    tabs = st.tabs(["📤 Score New Creative", "📊 Compare Creatives", "⚙️ Settings"])
    
    with tabs[0]:
        _render_score_tab(scorer)
    
    with tabs[1]:
        _render_compare_tab(scorer)
    
    with tabs[2]:
        _render_settings_tab(availability)


def _render_status_bar(availability: dict):
    """Render status bar showing system availability."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if availability['ollama_available']:
            st.success("✅ Ollama Running")
        else:
            st.error("❌ Ollama Not Found")
    
    with col2:
        if availability['vision_model_available']:
            model = availability['vision_models'][0] if availability['vision_models'] else 'unknown'
            st.success(f"✅ Vision: {model}")
        else:
            st.warning("⚠️ No Vision Model")
    
    with col3:
        if availability['text_models']:
            st.success(f"✅ Text: {availability['text_models'][0]}")
        else:
            st.info("ℹ️ No Text Model")
    
    with col4:
        if availability['cloud_llm_available']:
            st.success("✅ Claude API")
        else:
            st.info("ℹ️ No Claude API")
    
    with col5:
        # Check if model is trained on historical data
        try:
            from services.prediction_model import get_prediction_model
            model = get_prediction_model()
            if model.is_trained:
                n_samples = model.training_stats.get('n_samples', 0)
                st.success(f"✅ Trained ({n_samples} videos)")
            else:
                st.info("ℹ️ Using heuristics")
        except:
            st.info("ℹ️ Using heuristics")


def _render_setup_instructions(availability: dict):
    """Render setup instructions when system is not ready."""
    st.markdown("### 🛠️ Setup Required")
    
    st.markdown("""
    The Creative Scorer uses **local AI models** running on your Mac to analyze videos.
    This keeps your creative assets private and uses your Mac's powerful hardware.
    """)
    
    st.markdown("#### Installation Steps")
    
    with st.expander("1️⃣ Install Ollama", expanded=not availability['ollama_available']):
        st.code("""
# Install Ollama
brew install ollama

# Start Ollama server (keep this running)
ollama serve
        """, language="bash")
        st.info("Open a terminal and run these commands. Keep `ollama serve` running.")
    
    with st.expander("2️⃣ Download Vision Model", expanded=availability['ollama_available'] and not availability['vision_model_available']):
        st.code("""
# Download LLaVA vision model (recommended for 32GB RAM)
ollama pull llava:13b

# OR for less RAM (16GB)
ollama pull llava:7b
        """, language="bash")
        st.info("This downloads ~8GB. Only needed once.")
    
    with st.expander("3️⃣ Optional: Text Model for Summaries"):
        st.code("""
# Download Llama for text generation
ollama pull llama3.1:8b
        """, language="bash")
    
    if st.button("🔄 Refresh Status"):
        st.rerun()


def _render_video_player(video_source, height: int = 400):
    """
    Render a video player for the given source.
    
    Args:
        video_source: Can be a file path (str), uploaded file, or bytes
        height: Player height in pixels
    """
    if isinstance(video_source, str):
        # File path
        if os.path.exists(video_source):
            with open(video_source, 'rb') as f:
                video_bytes = f.read()
            st.video(video_bytes)
        else:
            st.warning(f"Video file not found: {video_source}")
    else:
        # Uploaded file or bytes
        st.video(video_source)


def _render_score_tab(scorer: CreativeScorerService):
    """Render the scoring tab."""
    
    # Configuration
    with st.expander("⚙️ Scoring Configuration", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            brand_name = st.text_input("Brand Name (optional)", value="", help="Helps AI look for brand-specific elements")
            product_name = st.text_input("Product Name (optional)", value="")
        
        with col2:
            max_frames = st.slider("Frames to Analyze", 4, 16, 12, help="More frames = better analysis but slower")
            extraction_strategy = st.selectbox(
                "Frame Selection",
                ["uniform", "scene_change"],
                help="'uniform' = evenly spaced, 'scene_change' = detect scene cuts"
            )
        
        use_cloud_summary = st.checkbox(
            "Use Claude for Enhanced Summary", 
            value=True,
            help="Uses Claude API for better narrative summary (requires API key)"
        )
    
    # File upload
    st.markdown("### Upload Creative")
    
    uploaded_file = st.file_uploader(
        "Drop your video here",
        type=['mp4', 'mov', 'avi', 'mkv', 'webm'],
        help="Supported formats: MP4, MOV, AVI, MKV, WebM"
    )
    
    if uploaded_file:
        # Show video preview
        st.markdown("#### Preview")
        _render_video_player(uploaded_file)
        
        # Score button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            score_clicked = st.button("🔮 Score This Creative", type="primary", use_container_width=True)
        
        if score_clicked:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            try:
                # Create config
                config = ScoringConfig(
                    brand_name=brand_name,
                    product_name=product_name,
                    max_frames=max_frames,
                    extraction_strategy=extraction_strategy,
                    use_cloud_llm_for_summary=use_cloud_summary,
                )
                
                # Progress container
                progress_container = st.empty()
                progress_bar = st.progress(0)
                
                def progress_callback(progress: ScoringProgress):
                    progress_bar.progress(progress.percentage / 100)
                    progress_container.info(f"**{progress.step_name}:** {progress.message}")
                
                # Run scoring
                result = scorer.score_creative(
                    video_path=tmp_path,
                    config=config,
                    progress_callback=progress_callback,
                )
                
                # Clear progress
                progress_container.empty()
                progress_bar.empty()
                
                if result.success:
                    # Store result in session state
                    if 'scoring_results' not in st.session_state:
                        st.session_state.scoring_results = []
                    
                    result.video_filename = uploaded_file.name
                    st.session_state.scoring_results.append(result)
                    st.session_state.current_scoring_result = result
                    
                    # Store video bytes for playback
                    uploaded_file.seek(0)
                    st.session_state.current_video_bytes = uploaded_file.read()
                    
                    # Render results
                    _render_scoring_results(result, st.session_state.current_video_bytes)
                else:
                    st.error(f"Scoring failed: {result.error_message}")
                    
            finally:
                # Cleanup temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
    
    # Show previous result if exists
    elif 'current_scoring_result' in st.session_state:
        st.markdown("---")
        st.markdown("### Previous Result")
        video_bytes = st.session_state.get('current_video_bytes')
        _render_scoring_results(st.session_state.current_scoring_result, video_bytes)


def _render_scoring_results(result: ScoringResult, video_bytes=None):
    """Render the complete scoring results."""
    score = result.score
    
    st.markdown("---")
    st.markdown("## 📊 Scoring Results")
    
    # Show model info
    try:
        from services.prediction_model import get_prediction_model
        model = get_prediction_model()
        if model.is_trained:
            stats = model.training_stats
            n_samples = stats.get('n_samples', 0)
            n_video = stats.get('n_with_video_features', 0)
            accuracy = stats.get('loocv_accuracy', 'N/A')
            
            st.success(f"🎯 **Prediction based on {n_samples} historical creatives** "
                      f"({n_video} with video analysis)")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("LOOCV Accuracy", f"{accuracy}%")
            with col_b:
                st.metric("Precision", f"{stats.get('loocv_precision', 'N/A')}%")
            with col_c:
                st.metric("Recall", f"{stats.get('loocv_recall', 'N/A')}%")
            
            # Show top predictors
            if model.learned_feature_importance:
                top_features = list(model.learned_feature_importance.items())[:5]
                with st.expander("📈 Top Predictors (from your data)"):
                    for feat, imp in top_features:
                        feat_display = feat.replace('_', ' ').replace('score', '').title()
                        st.caption(f"**{feat_display}**: {imp*100:.1f}% importance")
            
            # Show diagnostic model info
            if model.diagnostic_models_trained:
                with st.expander("🔬 Diagnostic Prediction Models"):
                    st.caption("*Diagnostics predicted from video features using learned models*")
                    for diag_name, diag_stats in model.diagnostic_training_stats.items():
                        diag_display = diag_name.replace('_', ' ').title()
                        r2 = diag_stats.get('cv_r2', 0)
                        rmse = diag_stats.get('rmse', 0)
                        st.caption(f"**{diag_display}**: R²={r2:.2f}, RMSE=±{rmse:.1f}")
        else:
            st.info("ℹ️ Using industry heuristics (need 10+ historical results to train custom model)")
    except Exception as e:
        st.info(f"ℹ️ Using industry heuristics")
    
    st.markdown("")
    
    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        prob = score.pass_probability * 100
        if prob >= 70:
            st.metric("Pass Probability", f"{prob:.0f}%", delta="High", delta_color="normal")
        elif prob >= 50:
            st.metric("Pass Probability", f"{prob:.0f}%", delta="Medium", delta_color="off")
        else:
            st.metric("Pass Probability", f"{prob:.0f}%", delta="Low", delta_color="inverse")
    
    with col2:
        lift = score.predicted_lift
        st.metric("Predicted Lift", f"{lift:+.1f}%", delta=f"Range: {score.lift_range[0]:.1f} to {score.lift_range[1]:.1f}")
    
    with col3:
        ci = score.confidence_interval
        st.metric("Confidence", f"{(ci[1] - ci[0]) * 100:.0f}% range", delta=f"{ci[0]*100:.0f}% - {ci[1]*100:.0f}%")
    
    with col4:
        risk_colors = {"low": "🟢", "medium": "🟡", "high": "🔴"}
        st.metric("Risk Level", f"{risk_colors.get(score.risk_level, '⚪')} {score.risk_level.title()}")
    
    st.markdown("---")
    
    # Two-column layout
    left_col, right_col = st.columns([3, 2])
    
    with left_col:
        # Predicted Diagnostics
        st.markdown("### 📈 Predicted Diagnostics")
        
        # Check if benchmarks are from data
        try:
            from services.prediction_model import CreativePredictionModel
            benchmarks_from_data = getattr(CreativePredictionModel, 'benchmarks_from_data', False)
            if benchmarks_from_data:
                st.caption("*Benchmarks based on historical test results*")
            else:
                st.caption("*Benchmarks based on industry standards (not historical data)*")
        except:
            pass
        
        for name, diag in score.predicted_diagnostics.items():
            display_name = name.replace('_', ' ').title()
            
            if diag.status == "good":
                status_icon = "✅"
            elif diag.status == "warning":
                status_icon = "⚠️"
            else:
                status_icon = "❌"
            
            col_a, col_b, col_c = st.columns([3, 1, 1])
            
            with col_a:
                progress_pct = min(100, max(0, diag.predicted_value))
                st.progress(progress_pct / 100)
            
            with col_b:
                st.markdown(f"**{diag.predicted_value:.0f}** {status_icon}")
            
            with col_c:
                st.caption(f"Benchmark: {diag.benchmark}")
            
            st.caption(display_name)
            st.markdown("")
        
        # Risk Factors
        if score.risk_factors:
            st.markdown("### ⚠️ Risk Factors")
            for risk in score.risk_factors:
                confidence_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(risk.confidence, "⚪")
                with st.expander(f"{confidence_color} {risk.factor}"):
                    st.markdown(f"**Impact:** {risk.impact}")
                    st.markdown(f"**Evidence:** {risk.evidence}")
    
    with right_col:
        # Video preview
        if video_bytes:
            st.markdown("### 🎬 Creative")
            st.video(video_bytes)
        
        # Frame timeline
        if result.video_analysis and result.video_analysis.frame_analyses:
            st.markdown("### 🎞️ Frame Analysis")
            st.caption(f"*Analyzed {len(result.video_analysis.frame_analyses)} frames*")
            
            frames = result.video_analysis.frame_analyses
            
            for fa in frames:
                with st.expander(f"**[{fa.timestamp:.1f}s]** - {fa.scene_type or 'Scene'}", expanded=False):
                    col_thumb, col_desc = st.columns([1, 2])
                    
                    with col_thumb:
                        if hasattr(fa, 'frame_path') and fa.frame_path and os.path.exists(fa.frame_path):
                            st.image(fa.frame_path, width=150)
                        else:
                            st.caption(f"Frame at {fa.timestamp:.1f}s")
                    
                    with col_desc:
                        # Detection tags
                        tags = []
                        if fa.humans_present:
                            tags.append("👤 Human")
                        if fa.logo_visible:
                            tags.append("🏷️ Logo")
                        if fa.cta_present:
                            tags.append("📢 CTA")
                        if fa.product_visible:
                            tags.append("📦 Product")
                        
                        if tags:
                            st.markdown(" | ".join(tags))
                        
                        # Full description
                        if fa.description:
                            st.markdown(f"**Description:** {fa.description}")
                        
                        # Additional details
                        if fa.setting:
                            st.caption(f"Setting: {fa.setting}")
                        if fa.mood:
                            st.caption(f"Mood: {fa.mood}")
                        if fa.text_on_screen:
                            st.caption(f"Text on screen: {', '.join(fa.text_on_screen)}")
    
    st.markdown("---")
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    
    if score.recommendations:
        for rec in score.recommendations:
            priority_style = {
                "high": "🔴",
                "medium": "🟡",
                "low": "🟢",
            }.get(rec.priority, "⚪")
            
            st.markdown(f"{priority_style} **{rec.action}**")
            st.caption(f"Expected Impact: {rec.expected_impact}")
            st.caption(f"_{rec.rationale}_")
            st.markdown("")
    else:
        st.success("No critical recommendations - creative looks good!")
    
    st.markdown("---")
    
    # AI Summary
    st.markdown("### 🤖 AI Analysis")
    st.markdown(score.ai_summary)
    
    # Similar Creatives
    if score.similar_creatives:
        st.markdown("---")
        st.markdown("### 📊 Similar Historical Creatives")
        
        similar_data = []
        for sc in score.similar_creatives:
            similar_data.append({
                "Creative": sc.name,
                "Similarity": f"{sc.similarity_score:.0f}%",
                "Lift": f"{sc.lift:+.1f}%",
                "Result": "✅ Pass" if sc.passed else "❌ Fail",
            })
        
        if similar_data:
            st.dataframe(similar_data, hide_index=True, use_container_width=True)
    
    # Export options
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        report = _generate_report_text(result)
        st.download_button(
            "📥 Download Report",
            data=report,
            file_name=f"Creative_Score_{result.video_filename.replace('.', '_')}.md",
            mime="text/markdown",
        )
    
    with col2:
        st.caption(f"Scored in {result.processing_time_seconds:.1f}s | {result.scoring_timestamp}")


def _generate_report_text(result: ScoringResult) -> str:
    """Generate markdown report text."""
    score = result.score
    
    lines = [
        f"# Creative Scoring Report: {result.video_filename}",
        f"",
        f"**Scored:** {result.scoring_timestamp}",
        f"**Processing Time:** {result.processing_time_seconds:.1f}s",
        f"",
        f"## Summary",
        f"",
        f"- **Pass Probability:** {score.pass_probability * 100:.0f}%",
        f"- **Predicted Lift:** {score.predicted_lift:+.1f}%",
        f"- **Risk Level:** {score.risk_level.title()}",
        f"",
        f"## Predicted Diagnostics",
        f"",
    ]
    
    for name, diag in score.predicted_diagnostics.items():
        lines.append(f"- **{name.replace('_', ' ').title()}:** {diag.predicted_value:.0f} (benchmark: {diag.benchmark}) - {diag.status}")
    
    lines.extend([f"", f"## Risk Factors", f""])
    
    for risk in score.risk_factors:
        lines.append(f"- **{risk.factor}**")
        lines.append(f"  - Impact: {risk.impact}")
        lines.append(f"  - Confidence: {risk.confidence}")
    
    lines.extend([f"", f"## Recommendations", f""])
    
    for rec in score.recommendations:
        lines.append(f"### [{rec.priority.upper()}] {rec.action}")
        lines.append(f"- Expected Impact: {rec.expected_impact}")
        lines.append(f"- Rationale: {rec.rationale}")
        lines.append("")
    
    lines.extend([f"", f"## AI Analysis", f"", score.ai_summary])
    
    return "\n".join(lines)


def _render_compare_tab(scorer: CreativeScorerService):
    """Render the compare creatives tab."""
    st.markdown("### Compare Multiple Creatives")
    
    if 'scoring_results' not in st.session_state or not st.session_state.scoring_results:
        st.info("Score some creatives first! They'll appear here for comparison.")
        return
    
    results = st.session_state.scoring_results
    
    st.markdown(f"**{len(results)} creatives scored**")
    
    comparison_data = []
    for r in results:
        if r.success and r.score:
            comparison_data.append({
                "Creative": r.video_filename,
                "Pass Prob": f"{r.score.pass_probability * 100:.0f}%",
                "Pred Lift": f"{r.score.predicted_lift:+.1f}%",
                "Risk": r.score.risk_level.title(),
                "Top Risk": r.score.risk_factors[0].factor if r.score.risk_factors else "None",
            })
    
    if comparison_data:
        comparison_data.sort(key=lambda x: float(x["Pass Prob"].replace("%", "")), reverse=True)
        st.dataframe(comparison_data, hide_index=True, use_container_width=True)
        
        if len(comparison_data) > 1:
            winner = comparison_data[0]
            st.success(f"🏆 **Recommended:** {winner['Creative']} ({winner['Pass Prob']} pass probability)")
    
    if len(results) >= 2:
        st.markdown("---")
        st.markdown("### Side-by-Side Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected1 = st.selectbox(
                "Creative 1",
                options=range(len(results)),
                format_func=lambda i: results[i].video_filename
            )
        
        with col2:
            selected2 = st.selectbox(
                "Creative 2",
                options=range(len(results)),
                index=min(1, len(results) - 1),
                format_func=lambda i: results[i].video_filename
            )
        
        if selected1 != selected2:
            r1, r2 = results[selected1], results[selected2]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### {r1.video_filename}")
                if r1.score:
                    st.metric("Pass Probability", f"{r1.score.pass_probability * 100:.0f}%")
                    st.metric("Predicted Lift", f"{r1.score.predicted_lift:+.1f}%")
                    st.metric("Risk Level", r1.score.risk_level.title())
            
            with col2:
                st.markdown(f"#### {r2.video_filename}")
                if r2.score:
                    delta = r2.score.pass_probability - r1.score.pass_probability
                    st.metric("Pass Probability", f"{r2.score.pass_probability * 100:.0f}%", delta=f"{delta*100:+.0f}%")
                    
                    delta_lift = r2.score.predicted_lift - r1.score.predicted_lift
                    st.metric("Predicted Lift", f"{r2.score.predicted_lift:+.1f}%", delta=f"{delta_lift:+.1f}%")
                    
                    st.metric("Risk Level", r2.score.risk_level.title())
    
    st.markdown("---")
    if st.button("🗑️ Clear All Results"):
        st.session_state.scoring_results = []
        if 'current_scoring_result' in st.session_state:
            del st.session_state.current_scoring_result
        st.rerun()


def _render_settings_tab(availability: dict):
    """Render settings tab."""
    st.markdown("### Settings")
    
    st.markdown("#### Available Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Vision Models:**")
        if availability['vision_models']:
            for model in availability['vision_models']:
                st.code(model)
        else:
            st.warning("No vision models installed")
            st.code("ollama pull llava:13b")
    
    with col2:
        st.markdown("**Text Models:**")
        if availability['text_models']:
            for model in availability['text_models']:
                st.code(model)
        else:
            st.info("No text models (optional)")
            st.code("ollama pull llama3.1:8b")
    
    st.markdown("---")
    st.markdown("#### System Information")
    
    st.markdown(f"- **Ollama Status:** {'Running ✅' if availability['ollama_available'] else 'Not Running ❌'}")
    st.markdown(f"- **Claude API:** {'Available ✅' if availability['cloud_llm_available'] else 'Not Configured'}")
    
    st.markdown("---")
    st.markdown("#### Model Recommendations for 32GB MacBook Pro")
    
    st.markdown("""
    | Model | RAM Needed | Quality | Speed |
    |-------|------------|---------|-------|
    | `llava:13b` | 16GB+ | ⭐⭐⭐ Best | Slow |
    | `llava:7b` | 8GB+ | ⭐⭐ Good | Medium |
    | `bakllava` | 8GB+ | ⭐⭐ Good | Medium |
    
    **Recommended:** `llava:13b` for best results on your machine.
    """)


# Export video player for use in other pages
def render_video_player_component(
    video_path: str = None,
    video_bytes: bytes = None,
    uploaded_file=None,
    title: str = None,
):
    """
    Reusable video player component for use across the app.
    
    Args:
        video_path: Path to video file
        video_bytes: Video as bytes
        uploaded_file: Streamlit uploaded file
        title: Optional title to display
    """
    if title:
        st.markdown(f"**{title}**")
    
    if video_path and os.path.exists(video_path):
        st.video(video_path)
    elif video_bytes:
        st.video(video_bytes)
    elif uploaded_file:
        st.video(uploaded_file)
    else:
        st.warning("No video source provided")
