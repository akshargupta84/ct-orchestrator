"""
Video Analysis Service.

Extracts creative features from videos using:
1. Frame extraction (OpenCV)
2. Vision AI analysis (Claude) for each frame
3. Aggregation of frame-level features to video-level metrics

Features extracted:
- Human presence (% of video, # of people, demographics)
- Brand elements (logo visibility, product presence, product in use)
- Creative elements (CTA, text overlays, pacing)
- Emotional tone
- Scene composition
"""

import os
import json
import base64
import tempfile
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import statistics

# Video processing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# LLM for vision analysis
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

import pandas as pd


@dataclass
class FrameAnalysis:
    """Analysis results for a single frame."""
    frame_number: int
    timestamp_seconds: float
    
    # Human presence
    has_human: bool = False
    human_count: int = 0
    human_screen_percentage: float = 0.0  # 0-100
    human_demographics: list = field(default_factory=list)  # ["adult_male", "adult_female", etc.]
    human_emotions: list = field(default_factory=list)  # ["happy", "neutral", etc.]
    
    # Brand elements
    has_logo: bool = False
    logo_screen_percentage: float = 0.0
    has_product: bool = False
    product_in_use: bool = False
    product_screen_percentage: float = 0.0
    brand_colors_present: bool = False
    
    # Creative elements
    has_text_overlay: bool = False
    text_content: str = ""
    has_cta: bool = False
    cta_text: str = ""
    
    # Scene composition
    scene_type: str = ""  # "indoor", "outdoor", "studio", "lifestyle", etc.
    dominant_colors: list = field(default_factory=list)
    visual_complexity: str = "medium"  # "simple", "medium", "complex"
    
    # Raw analysis
    raw_analysis: str = ""


@dataclass
class VideoFeatures:
    """Aggregated features for an entire video."""
    video_id: str
    filename: str
    duration_seconds: float
    frame_count: int
    frames_analyzed: int
    
    # Human presence metrics
    human_presence_percentage: float = 0.0  # % of frames with humans
    avg_human_count: float = 0.0
    max_human_count: int = 0
    human_screen_time_percentage: float = 0.0  # Avg % of screen occupied by humans
    primary_demographics: list = field(default_factory=list)
    primary_emotions: list = field(default_factory=list)
    
    # Brand metrics
    logo_presence_percentage: float = 0.0  # % of frames with logo
    logo_avg_screen_percentage: float = 0.0
    logo_appears_in_first_3s: bool = False
    logo_appears_in_last_3s: bool = False
    product_presence_percentage: float = 0.0
    product_in_use_percentage: float = 0.0
    
    # Creative metrics
    text_overlay_percentage: float = 0.0
    has_cta: bool = False
    cta_appears_at_second: Optional[float] = None
    cta_text: str = ""
    
    # Pacing metrics
    scene_count: int = 0
    avg_scene_duration: float = 0.0
    cuts_per_second: float = 0.0
    
    # Scene composition
    primary_scene_types: list = field(default_factory=list)
    visual_complexity_score: float = 0.0  # 1-3 (simple to complex)
    
    # Timestamps for key moments
    first_human_appearance: Optional[float] = None
    first_product_appearance: Optional[float] = None
    first_logo_appearance: Optional[float] = None
    
    # Frame analyses
    frame_analyses: list = field(default_factory=list)
    
    # Metadata
    analyzed_at: str = ""
    analysis_model: str = "claude-sonnet-4-20250514"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        d = asdict(self)
        # Don't include full frame analyses in dict (too large)
        d.pop('frame_analyses', None)
        return d
    
    def to_feature_vector(self) -> dict:
        """Convert to a flat feature vector for modeling."""
        return {
            'human_presence_pct': self.human_presence_percentage,
            'avg_human_count': self.avg_human_count,
            'human_screen_time_pct': self.human_screen_time_percentage,
            'logo_presence_pct': self.logo_presence_percentage,
            'logo_first_3s': 1 if self.logo_appears_in_first_3s else 0,
            'logo_last_3s': 1 if self.logo_appears_in_last_3s else 0,
            'product_presence_pct': self.product_presence_percentage,
            'product_in_use_pct': self.product_in_use_percentage,
            'text_overlay_pct': self.text_overlay_percentage,
            'has_cta': 1 if self.has_cta else 0,
            'cuts_per_second': self.cuts_per_second,
            'visual_complexity': self.visual_complexity_score,
            'duration_seconds': self.duration_seconds,
        }


class VideoAnalysisService:
    """
    Service for extracting creative features from videos.
    
    Uses Claude Vision API to analyze sampled frames and aggregates
    results into video-level metrics.
    """
    
    # Analysis configuration
    FRAMES_PER_SECOND = 1  # Sample 1 frame per second
    MAX_FRAMES_TO_ANALYZE = 30  # Cap for long videos
    
    # Prompt for frame analysis
    FRAME_ANALYSIS_PROMPT = """Analyze this video frame and extract the following information. 
Respond in JSON format only, no other text.

{
  "has_human": true/false,
  "human_count": number,
  "human_screen_percentage": 0-100 (estimate what % of frame humans occupy),
  "human_demographics": ["adult_male", "adult_female", "child", "teenager", "elderly"],
  "human_emotions": ["happy", "neutral", "excited", "serious", "concerned"],
  
  "has_logo": true/false,
  "logo_screen_percentage": 0-100,
  "has_product": true/false,
  "product_in_use": true/false (is someone using/demonstrating the product?),
  "product_screen_percentage": 0-100,
  
  "has_text_overlay": true/false,
  "text_content": "text visible in frame",
  "has_cta": true/false (call-to-action like 'Buy Now', 'Learn More', 'Shop', etc.),
  "cta_text": "the CTA text if present",
  
  "scene_type": "indoor"/"outdoor"/"studio"/"lifestyle"/"product_shot"/"animation"/"mixed",
  "dominant_colors": ["color1", "color2"],
  "visual_complexity": "simple"/"medium"/"complex"
}

Be precise and objective. If uncertain, make your best estimate."""

    def __init__(self, anthropic_api_key: str = None):
        """
        Initialize the video analysis service.
        
        Args:
            anthropic_api_key: API key for Claude. If not provided, uses env var.
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required. Install: pip install opencv-python")
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic SDK required. Install: pip install anthropic")
        
        self.api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required for video analysis")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.analyses: dict[str, VideoFeatures] = {}
    
    def analyze_video(self, video_path: str, video_id: str = None) -> VideoFeatures:
        """
        Analyze a video and extract creative features.
        
        Args:
            video_path: Path to the video file
            video_id: Optional ID for the video
            
        Returns:
            VideoFeatures object with all extracted metrics
        """
        filename = os.path.basename(video_path)
        video_id = video_id or filename
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Calculate which frames to sample
        frames_to_sample = min(
            int(duration * self.FRAMES_PER_SECOND),
            self.MAX_FRAMES_TO_ANALYZE
        )
        
        if frames_to_sample == 0:
            frames_to_sample = 1
        
        # Sample frames evenly throughout video
        frame_indices = [
            int(i * total_frames / frames_to_sample)
            for i in range(frames_to_sample)
        ]
        
        # Analyze each frame
        frame_analyses = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            timestamp = frame_idx / fps
            
            try:
                analysis = self._analyze_frame(frame, frame_idx, timestamp)
                frame_analyses.append(analysis)
            except Exception as e:
                print(f"Error analyzing frame {frame_idx}: {e}")
                continue
        
        cap.release()
        
        # Aggregate frame analyses into video features
        features = self._aggregate_analyses(
            video_id=video_id,
            filename=filename,
            duration=duration,
            total_frames=total_frames,
            frame_analyses=frame_analyses
        )
        
        self.analyses[video_id] = features
        return features
    
    def _analyze_frame(self, frame, frame_number: int, timestamp: float) -> FrameAnalysis:
        """Analyze a single frame using Claude Vision."""
        # Convert frame to base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        # Call Claude Vision API
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image,
                            }
                        },
                        {
                            "type": "text",
                            "text": self.FRAME_ANALYSIS_PROMPT
                        }
                    ]
                }
            ]
        )
        
        # Parse response
        response_text = response.content[0].text
        
        try:
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
            else:
                data = {}
        except json.JSONDecodeError:
            data = {}
        
        return FrameAnalysis(
            frame_number=frame_number,
            timestamp_seconds=timestamp,
            has_human=data.get('has_human', False),
            human_count=data.get('human_count', 0),
            human_screen_percentage=data.get('human_screen_percentage', 0),
            human_demographics=data.get('human_demographics', []),
            human_emotions=data.get('human_emotions', []),
            has_logo=data.get('has_logo', False),
            logo_screen_percentage=data.get('logo_screen_percentage', 0),
            has_product=data.get('has_product', False),
            product_in_use=data.get('product_in_use', False),
            product_screen_percentage=data.get('product_screen_percentage', 0),
            has_text_overlay=data.get('has_text_overlay', False),
            text_content=data.get('text_content', ''),
            has_cta=data.get('has_cta', False),
            cta_text=data.get('cta_text', ''),
            scene_type=data.get('scene_type', ''),
            dominant_colors=data.get('dominant_colors', []),
            visual_complexity=data.get('visual_complexity', 'medium'),
            raw_analysis=response_text
        )
    
    def _aggregate_analyses(
        self,
        video_id: str,
        filename: str,
        duration: float,
        total_frames: int,
        frame_analyses: list[FrameAnalysis]
    ) -> VideoFeatures:
        """Aggregate frame-level analyses into video-level features."""
        
        if not frame_analyses:
            return VideoFeatures(
                video_id=video_id,
                filename=filename,
                duration_seconds=duration,
                frame_count=total_frames,
                frames_analyzed=0,
                analyzed_at=datetime.now().isoformat()
            )
        
        n = len(frame_analyses)
        
        # Human metrics
        frames_with_humans = [f for f in frame_analyses if f.has_human]
        human_presence_pct = len(frames_with_humans) / n * 100
        
        human_counts = [f.human_count for f in frame_analyses]
        avg_human_count = statistics.mean(human_counts) if human_counts else 0
        max_human_count = max(human_counts) if human_counts else 0
        
        human_screen_pcts = [f.human_screen_percentage for f in frames_with_humans]
        avg_human_screen = statistics.mean(human_screen_pcts) if human_screen_pcts else 0
        
        # Aggregate demographics and emotions
        all_demographics = []
        all_emotions = []
        for f in frames_with_humans:
            all_demographics.extend(f.human_demographics)
            all_emotions.extend(f.human_emotions)
        
        primary_demographics = self._get_top_items(all_demographics, 3)
        primary_emotions = self._get_top_items(all_emotions, 3)
        
        # Brand metrics
        frames_with_logo = [f for f in frame_analyses if f.has_logo]
        logo_presence_pct = len(frames_with_logo) / n * 100
        
        logo_screen_pcts = [f.logo_screen_percentage for f in frames_with_logo]
        avg_logo_screen = statistics.mean(logo_screen_pcts) if logo_screen_pcts else 0
        
        # Check logo timing
        logo_first_3s = any(f.has_logo and f.timestamp_seconds <= 3 for f in frame_analyses)
        logo_last_3s = any(f.has_logo and f.timestamp_seconds >= duration - 3 for f in frame_analyses)
        
        # Product metrics
        frames_with_product = [f for f in frame_analyses if f.has_product]
        product_presence_pct = len(frames_with_product) / n * 100
        
        frames_product_in_use = [f for f in frame_analyses if f.product_in_use]
        product_in_use_pct = len(frames_product_in_use) / n * 100
        
        # Text/CTA metrics
        frames_with_text = [f for f in frame_analyses if f.has_text_overlay]
        text_overlay_pct = len(frames_with_text) / n * 100
        
        frames_with_cta = [f for f in frame_analyses if f.has_cta]
        has_cta = len(frames_with_cta) > 0
        cta_timestamp = frames_with_cta[0].timestamp_seconds if frames_with_cta else None
        cta_text = frames_with_cta[0].cta_text if frames_with_cta else ""
        
        # Scene analysis
        scene_types = [f.scene_type for f in frame_analyses if f.scene_type]
        primary_scenes = self._get_top_items(scene_types, 3)
        
        # Detect scene changes (simple heuristic)
        scene_changes = sum(
            1 for i in range(1, len(frame_analyses))
            if frame_analyses[i].scene_type != frame_analyses[i-1].scene_type
        )
        scene_count = scene_changes + 1
        avg_scene_duration = duration / scene_count if scene_count > 0 else duration
        cuts_per_second = scene_changes / duration if duration > 0 else 0
        
        # Visual complexity
        complexity_map = {'simple': 1, 'medium': 2, 'complex': 3}
        complexities = [complexity_map.get(f.visual_complexity, 2) for f in frame_analyses]
        avg_complexity = statistics.mean(complexities) if complexities else 2
        
        # First appearances
        first_human = next((f.timestamp_seconds for f in frame_analyses if f.has_human), None)
        first_product = next((f.timestamp_seconds for f in frame_analyses if f.has_product), None)
        first_logo = next((f.timestamp_seconds for f in frame_analyses if f.has_logo), None)
        
        return VideoFeatures(
            video_id=video_id,
            filename=filename,
            duration_seconds=duration,
            frame_count=total_frames,
            frames_analyzed=n,
            human_presence_percentage=human_presence_pct,
            avg_human_count=avg_human_count,
            max_human_count=max_human_count,
            human_screen_time_percentage=avg_human_screen,
            primary_demographics=primary_demographics,
            primary_emotions=primary_emotions,
            logo_presence_percentage=logo_presence_pct,
            logo_avg_screen_percentage=avg_logo_screen,
            logo_appears_in_first_3s=logo_first_3s,
            logo_appears_in_last_3s=logo_last_3s,
            product_presence_percentage=product_presence_pct,
            product_in_use_percentage=product_in_use_pct,
            text_overlay_percentage=text_overlay_pct,
            has_cta=has_cta,
            cta_appears_at_second=cta_timestamp,
            cta_text=cta_text,
            scene_count=scene_count,
            avg_scene_duration=avg_scene_duration,
            cuts_per_second=cuts_per_second,
            primary_scene_types=primary_scenes,
            visual_complexity_score=avg_complexity,
            first_human_appearance=first_human,
            first_product_appearance=first_product,
            first_logo_appearance=first_logo,
            frame_analyses=frame_analyses,
            analyzed_at=datetime.now().isoformat(),
            analysis_model="claude-sonnet-4-20250514"
        )
    
    def _get_top_items(self, items: list, n: int) -> list:
        """Get the n most common items from a list."""
        if not items:
            return []
        
        from collections import Counter
        counts = Counter(items)
        return [item for item, _ in counts.most_common(n)]
    
    def get_features_dataframe(self) -> pd.DataFrame:
        """Get all analyzed videos as a DataFrame for modeling."""
        if not self.analyses:
            return pd.DataFrame()
        
        rows = []
        for video_id, features in self.analyses.items():
            row = features.to_feature_vector()
            row['video_id'] = video_id
            row['filename'] = features.filename
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save_analyses(self, path: str):
        """Save all analyses to JSON file."""
        data = {
            video_id: features.to_dict()
            for video_id, features in self.analyses.items()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_analyses(self, path: str):
        """Load analyses from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        for video_id, features_dict in data.items():
            # Convert dict back to VideoFeatures (without frame analyses)
            features_dict['frame_analyses'] = []
            self.analyses[video_id] = VideoFeatures(**features_dict)


def generate_analysis_report(features: VideoFeatures) -> str:
    """Generate a human-readable report for a video analysis."""
    report = f"""
# Video Analysis Report: {features.filename}

## Overview
- **Duration:** {features.duration_seconds:.1f} seconds
- **Frames Analyzed:** {features.frames_analyzed}
- **Analyzed At:** {features.analyzed_at}

## Human Presence
- **Presence:** {features.human_presence_percentage:.1f}% of video
- **Average People:** {features.avg_human_count:.1f} (max: {features.max_human_count})
- **Screen Coverage:** {features.human_screen_time_percentage:.1f}% average
- **Demographics:** {', '.join(features.primary_demographics) or 'N/A'}
- **Emotions:** {', '.join(features.primary_emotions) or 'N/A'}
- **First Appearance:** {f'{features.first_human_appearance:.1f}s' if features.first_human_appearance else 'N/A'}

## Brand Elements
- **Logo Visible:** {features.logo_presence_percentage:.1f}% of video
- **Logo in First 3s:** {'Yes' if features.logo_appears_in_first_3s else 'No'}
- **Logo in Last 3s:** {'Yes' if features.logo_appears_in_last_3s else 'No'}
- **Product Shown:** {features.product_presence_percentage:.1f}% of video
- **Product In Use:** {features.product_in_use_percentage:.1f}% of video

## Creative Elements
- **Text Overlays:** {features.text_overlay_percentage:.1f}% of video
- **Has CTA:** {'Yes' if features.has_cta else 'No'}
- **CTA Text:** {features.cta_text or 'N/A'}
- **CTA Timing:** {f'{features.cta_appears_at_second:.1f}s' if features.cta_appears_at_second else 'N/A'}

## Pacing & Composition
- **Scene Count:** {features.scene_count}
- **Avg Scene Duration:** {features.avg_scene_duration:.1f}s
- **Cuts Per Second:** {features.cuts_per_second:.2f}
- **Visual Complexity:** {features.visual_complexity_score:.1f}/3
- **Scene Types:** {', '.join(features.primary_scene_types) or 'N/A'}
"""
    return report
