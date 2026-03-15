"""
Creative Scorer Service.

Main orchestration service for pre-test creative scoring.
Combines video analysis, feature extraction, and prediction into a single pipeline.
"""

import os
import time
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

from .local_vision import LocalVisionService, get_vision_service, VideoAnalysisResult
from .frame_analyzer import FrameAnalyzer, get_frame_analyzer, ExtractionConfig
from .prediction_model import (
    CreativePredictionModel, 
    get_prediction_model, 
    PredictionFeatures,
    CreativeScore
)

# Try to import LLM for enhanced summaries
try:
    from utils.llm import is_llm_available, get_completion
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


@dataclass
class ScoringConfig:
    """Configuration for creative scoring."""
    # Frame extraction
    max_frames: int = 12  # Increased from 8 for better accuracy
    extraction_strategy: str = "uniform"  # "uniform" or "scene_change"
    frame_interval_seconds: float = 2.5  # More frequent sampling
    
    # Analysis
    brand_name: str = ""
    product_name: str = ""
    
    # Model
    use_local_vision: bool = True
    use_cloud_llm_for_summary: bool = True  # Use Claude for final summary if available
    
    # Performance
    parallel_frame_analysis: bool = False  # Not implemented yet
    
    # Vision model selection
    vision_model: str = None  # Auto-detect if None
    text_model: str = None


@dataclass
class ScoringProgress:
    """Progress update during scoring."""
    current_step: int
    total_steps: int
    step_name: str
    message: str
    percentage: float = 0.0


@dataclass
class ScoringResult:
    """Complete scoring result."""
    # Success
    success: bool = False
    error_message: str = ""
    
    # Results
    score: CreativeScore = None
    video_analysis: VideoAnalysisResult = None
    features: PredictionFeatures = None
    
    # Metadata
    video_path: str = ""
    video_filename: str = ""
    scoring_timestamp: str = ""
    processing_time_seconds: float = 0
    
    # Frame images for display (paths)
    frame_images: list = field(default_factory=list)  # List of (timestamp, path) tuples


class CreativeScorerService:
    """
    Main service for scoring creative videos before testing.
    
    Pipeline:
    1. Extract frames from video
    2. Analyze frames with local vision model
    3. Extract features from analysis
    4. Predict outcomes using ML model
    5. Generate recommendations and summary
    """
    
    def __init__(
        self,
        vision_service: LocalVisionService = None,
        prediction_model: CreativePredictionModel = None,
        config: ScoringConfig = None,
    ):
        """
        Initialize the scorer.
        
        Args:
            vision_service: Vision service for frame analysis
            prediction_model: Prediction model for scoring
            config: Scoring configuration
        """
        self.config = config or ScoringConfig()
        self.vision_service = vision_service or get_vision_service(
            vision_model=self.config.vision_model,
            text_model=self.config.text_model,
        )
        self.prediction_model = prediction_model or get_prediction_model()
        self.frame_analyzer = get_frame_analyzer(self.vision_service)
    
    def check_availability(self) -> dict:
        """
        Check if all required components are available.
        
        Returns:
            Dict with availability status for each component
        """
        status = {
            'ollama_available': False,
            'vision_model_available': False,
            'vision_models': [],
            'text_models': [],
            'cloud_llm_available': LLM_AVAILABLE and is_llm_available() if LLM_AVAILABLE else False,
            'ready': False,
            'message': '',
        }
        
        # Check Ollama
        if self.vision_service.is_available():
            status['ollama_available'] = True
            models = self.vision_service.get_available_models()
            status['vision_models'] = models.get('vision', [])
            status['text_models'] = models.get('text', [])
            status['vision_model_available'] = len(status['vision_models']) > 0
        
        # Determine readiness
        if status['vision_model_available']:
            status['ready'] = True
            status['message'] = f"Ready! Using vision model: {status['vision_models'][0] if status['vision_models'] else 'unknown'}"
        elif status['ollama_available']:
            status['message'] = "Ollama running but no vision model found. Run: ollama pull llava:13b"
        else:
            status['message'] = "Ollama not available. Install with: brew install ollama && ollama serve"
        
        return status
    
    def score_creative(
        self,
        video_path: str,
        config: ScoringConfig = None,
        progress_callback: Callable[[ScoringProgress], None] = None,
    ) -> ScoringResult:
        """
        Score a creative video.
        
        Args:
            video_path: Path to video file
            config: Optional config override
            progress_callback: Optional callback for progress updates
            
        Returns:
            ScoringResult with complete scoring
        """
        start_time = time.time()
        config = config or self.config
        
        result = ScoringResult(
            video_path=video_path,
            video_filename=Path(video_path).name,
            scoring_timestamp=datetime.now().isoformat(),
        )
        
        # Validate video exists
        if not os.path.exists(video_path):
            result.error_message = f"Video file not found: {video_path}"
            return result
        
        def update_progress(step: int, total: int, name: str, message: str):
            if progress_callback:
                progress_callback(ScoringProgress(
                    current_step=step,
                    total_steps=total,
                    step_name=name,
                    message=message,
                    percentage=(step / total) * 100,
                ))
        
        total_steps = 5
        
        try:
            # Step 1: Check availability
            update_progress(0, total_steps, "Checking", "Verifying vision model availability...")
            
            availability = self.check_availability()
            if not availability['ready']:
                result.error_message = availability['message']
                return result
            
            # Step 2: Analyze video
            update_progress(1, total_steps, "Analyzing", "Analyzing video frames with AI vision...")
            
            extraction_config = ExtractionConfig(
                max_frames=config.max_frames,
                strategy=config.extraction_strategy,
                interval_seconds=config.frame_interval_seconds,
            )
            
            # Custom progress for frame analysis
            def frame_progress(current, total, msg):
                pct = 1 + (current / total) * 2  # Steps 1-3
                update_progress(pct, total_steps, "Analyzing", msg)
            
            video_analysis = self.frame_analyzer.analyze_video(
                video_path=video_path,
                config=extraction_config,
                brand_name=config.brand_name,
                product_name=config.product_name,
                progress_callback=frame_progress,
            )
            
            result.video_analysis = video_analysis
            
            # Store frame images for display
            result.frame_images = [
                (fa.timestamp, fa.frame_path) 
                for fa in video_analysis.frame_analyses
                if os.path.exists(fa.frame_path)
            ]
            
            if video_analysis.errors:
                result.error_message = "; ".join(video_analysis.errors)
                # Continue anyway if we have some analysis
            
            # Step 3: Extract features
            update_progress(3, total_steps, "Features", "Extracting prediction features...")
            
            features = self.prediction_model.extract_features(video_analysis)
            result.features = features
            
            # Step 4: Predict
            update_progress(4, total_steps, "Predicting", "Running prediction model...")
            
            score = self.prediction_model.predict(features, video_analysis)
            
            # Step 5: Generate AI summary
            update_progress(5, total_steps, "Summary", "Generating AI summary...")
            
            if config.use_cloud_llm_for_summary and LLM_AVAILABLE and is_llm_available():
                score.ai_summary = self._generate_cloud_summary(score, video_analysis, config)
            else:
                # Use local summary from video analysis
                score.ai_summary = video_analysis.ai_summary
            
            result.score = score
            result.success = True
            
        except Exception as e:
            result.error_message = f"Scoring failed: {str(e)}"
            import traceback
            result.error_message += f"\n{traceback.format_exc()}"
        
        result.processing_time_seconds = time.time() - start_time
        
        return result
    
    def _generate_cloud_summary(
        self, 
        score: CreativeScore, 
        video_analysis: VideoAnalysisResult,
        config: ScoringConfig,
    ) -> str:
        """Generate enhanced summary using Claude API."""
        
        # Build context
        frame_descriptions = []
        for fa in video_analysis.frame_analyses:
            desc = f"[{fa.timestamp:.1f}s] {fa.description}"
            if fa.humans_present:
                emotions = ', '.join(fa.human_emotions) if fa.human_emotions else 'neutral'
                desc += f" (Human: {emotions})"
            if fa.logo_visible:
                desc += " [Logo visible]"
            if fa.cta_present:
                desc += f" [CTA: {fa.cta_text}]"
            frame_descriptions.append(desc)
        
        # Build diagnostic summary
        diag_lines = []
        for name, diag in score.predicted_diagnostics.items():
            status_emoji = "✅" if diag.status == "good" else "⚠️" if diag.status == "warning" else "❌"
            diag_lines.append(f"- {name}: {diag.predicted_value:.0f} (benchmark: {diag.benchmark}) {status_emoji}")
        
        # Build risk summary
        risk_lines = []
        for risk in score.risk_factors:
            risk_lines.append(f"- {risk.factor}: {risk.impact}")
        
        prompt = f"""Analyze this video creative and provide a brief executive summary.

Video: {video_analysis.duration_seconds:.0f} seconds{f' for {config.brand_name}' if config.brand_name else ''}

Frame-by-frame analysis:
{chr(10).join(frame_descriptions)}

Predicted Performance:
- Pass Probability: {score.pass_probability * 100:.0f}%
- Predicted Lift: {score.predicted_lift:.1f}%
- Risk Level: {score.risk_level}

Predicted Diagnostics:
{chr(10).join(diag_lines)}

Risk Factors:
{chr(10).join(risk_lines) if risk_lines else 'None identified'}

Write a 3-4 sentence executive summary that:
1. Describes what happens in the creative
2. Explains the predicted performance (why it will/won't work)
3. Gives the single most important recommendation

Be specific and actionable. Reference actual timestamps and elements from the analysis."""

        try:
            summary = get_completion(
                prompt=prompt,
                system="You are a creative strategist analyzing video advertisements. Be concise, specific, and actionable.",
                max_tokens=500,
                temperature=0,
            )
            return summary.strip()
        except Exception as e:
            return f"Summary generation failed: {e}\n\n{video_analysis.ai_summary}"
    
    def score_multiple(
        self,
        video_paths: list,
        config: ScoringConfig = None,
        progress_callback: Callable[[int, int, str], None] = None,
    ) -> list:
        """
        Score multiple videos.
        
        Args:
            video_paths: List of video file paths
            config: Optional config override
            progress_callback: Optional callback(current, total, video_name)
            
        Returns:
            List of ScoringResult
        """
        results = []
        
        for i, path in enumerate(video_paths):
            if progress_callback:
                progress_callback(i, len(video_paths), Path(path).name)
            
            result = self.score_creative(path, config)
            results.append(result)
        
        if progress_callback:
            progress_callback(len(video_paths), len(video_paths), "Complete")
        
        return results
    
    def compare_creatives(
        self,
        results: list,
    ) -> dict:
        """
        Compare multiple scored creatives.
        
        Args:
            results: List of ScoringResult
            
        Returns:
            Dict with comparison analysis
        """
        if not results:
            return {"error": "No results to compare"}
        
        successful = [r for r in results if r.success and r.score]
        
        if not successful:
            return {"error": "No successful scores to compare"}
        
        # Rank by pass probability
        ranked = sorted(successful, key=lambda r: r.score.pass_probability, reverse=True)
        
        # Build comparison
        comparison = {
            "total_scored": len(successful),
            "ranking": [],
            "summary": {},
        }
        
        for i, result in enumerate(ranked):
            comparison["ranking"].append({
                "rank": i + 1,
                "video": result.video_filename,
                "pass_probability": result.score.pass_probability,
                "predicted_lift": result.score.predicted_lift,
                "risk_level": result.score.risk_level,
                "key_risks": [r.factor for r in result.score.risk_factors[:2]],
            })
        
        # Summary stats
        probs = [r.score.pass_probability for r in successful]
        comparison["summary"] = {
            "avg_pass_probability": sum(probs) / len(probs),
            "best_creative": ranked[0].video_filename,
            "best_probability": ranked[0].score.pass_probability,
            "worst_creative": ranked[-1].video_filename,
            "worst_probability": ranked[-1].score.pass_probability,
        }
        
        return comparison


# Singleton instance
_scorer_service = None


def get_scorer_service(config: ScoringConfig = None) -> CreativeScorerService:
    """Get scorer service instance."""
    global _scorer_service
    if _scorer_service is None:
        _scorer_service = CreativeScorerService(config=config)
    return _scorer_service
