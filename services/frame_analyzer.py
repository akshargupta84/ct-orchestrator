"""
Enhanced Frame Analyzer.

Extracts frames from videos and analyzes them using local vision models.
Builds comprehensive video understanding for creative scoring.
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

# Video processing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .local_vision import (
    LocalVisionService, 
    get_vision_service, 
    FrameAnalysis, 
    VideoAnalysisResult
)


@dataclass
class ExtractionConfig:
    """Configuration for frame extraction."""
    # How many frames to analyze
    max_frames: int = 8
    
    # Always include these
    include_first_frame: bool = True
    include_last_frame: bool = True
    
    # Extraction strategy
    strategy: str = "uniform"  # "uniform", "scene_change", "key_moments"
    
    # For uniform strategy - extract every N seconds
    interval_seconds: float = 3.0
    
    # For scene_change strategy - sensitivity
    scene_change_threshold: float = 30.0
    
    # Image output settings
    output_format: str = "jpg"
    output_quality: int = 85
    max_dimension: int = 1024  # Resize if larger


@dataclass
class ExtractedFrame:
    """A frame extracted from video."""
    frame_path: str
    timestamp: float  # Seconds
    frame_number: int
    is_first: bool = False
    is_last: bool = False
    is_scene_change: bool = False


class FrameAnalyzer:
    """
    Analyzes video creatives by extracting and understanding frames.
    """
    
    def __init__(self, vision_service: LocalVisionService = None):
        """
        Initialize the frame analyzer.
        
        Args:
            vision_service: Optional vision service instance
        """
        self.vision_service = vision_service or get_vision_service()
        self.temp_dir = None
        
    def analyze_video(
        self,
        video_path: str,
        config: ExtractionConfig = None,
        brand_name: str = "",
        product_name: str = "",
        progress_callback: callable = None,
    ) -> VideoAnalysisResult:
        """
        Analyze a video file completely.
        
        Args:
            video_path: Path to video file
            config: Extraction configuration
            brand_name: Brand name for context
            product_name: Product name for context
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            VideoAnalysisResult with complete analysis
        """
        start_time = time.time()
        config = config or ExtractionConfig()
        
        result = VideoAnalysisResult(
            video_path=video_path,
            duration_seconds=0,
            frame_count=0,
        )
        
        if not CV2_AVAILABLE:
            result.errors.append("OpenCV not installed. Run: pip install opencv-python")
            return result
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            result.errors.append(f"Could not open video: {video_path}")
            return result
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        result.duration_seconds = duration
        result.frame_count = total_frames
        
        cap.release()
        
        if progress_callback:
            progress_callback(0, config.max_frames + 2, "Extracting frames...")
        
        # Extract frames
        try:
            extracted_frames = self._extract_frames(video_path, config, duration)
        except Exception as e:
            result.errors.append(f"Frame extraction failed: {e}")
            return result
        
        if progress_callback:
            progress_callback(1, config.max_frames + 2, f"Extracted {len(extracted_frames)} frames")
        
        # Analyze each frame
        frame_analyses = []
        for i, frame in enumerate(extracted_frames):
            if progress_callback:
                progress_callback(
                    i + 2, 
                    len(extracted_frames) + 2, 
                    f"Analyzing frame {i+1}/{len(extracted_frames)} ({frame.timestamp:.1f}s)"
                )
            
            analysis = self.vision_service.analyze_frame(
                image_path=frame.frame_path,
                timestamp=frame.timestamp,
                is_opening=frame.is_first,
                is_closing=frame.is_last,
                brand_name=brand_name,
                product_name=product_name,
            )
            
            frame_analyses.append(analysis)
        
        result.frame_analyses = frame_analyses
        
        # Aggregate insights
        result = self._aggregate_insights(result)
        
        # Generate summary
        if progress_callback:
            progress_callback(
                len(extracted_frames) + 1, 
                len(extracted_frames) + 2, 
                "Generating summary..."
            )
        
        result.ai_summary = self.vision_service.generate_video_summary(
            frame_analyses=frame_analyses,
            video_duration=duration,
            brand_name=brand_name,
        )
        
        result.processing_time_seconds = time.time() - start_time
        result.model_used = self.vision_service._select_vision_model() or "unknown"
        
        # Cleanup temp files
        self._cleanup_temp_files(extracted_frames)
        
        if progress_callback:
            progress_callback(
                len(extracted_frames) + 2, 
                len(extracted_frames) + 2, 
                "Analysis complete!"
            )
        
        return result
    
    def _extract_frames(
        self, 
        video_path: str, 
        config: ExtractionConfig,
        duration: float
    ) -> List[ExtractedFrame]:
        """Extract frames from video based on configuration."""
        
        # Create temp directory for frames
        self.temp_dir = tempfile.mkdtemp(prefix="ct_frames_")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        extracted = []
        
        if config.strategy == "uniform":
            extracted = self._extract_uniform(cap, config, fps, total_frames, duration)
        elif config.strategy == "scene_change":
            extracted = self._extract_scene_changes(cap, config, fps, total_frames, duration)
        else:
            # Default to uniform
            extracted = self._extract_uniform(cap, config, fps, total_frames, duration)
        
        cap.release()
        return extracted
    
    def _extract_uniform(
        self,
        cap: 'cv2.VideoCapture',
        config: ExtractionConfig,
        fps: float,
        total_frames: int,
        duration: float
    ) -> List[ExtractedFrame]:
        """Extract frames at uniform intervals."""
        extracted = []
        
        # Calculate frame numbers to extract
        frame_times = []
        
        # First frame
        if config.include_first_frame:
            frame_times.append(0)
        
        # Frames at intervals
        t = config.interval_seconds
        while t < duration - 0.5:  # Leave room for last frame
            if len(frame_times) < config.max_frames - (1 if config.include_last_frame else 0):
                frame_times.append(t)
            t += config.interval_seconds
        
        # Last frame
        if config.include_last_frame and duration > 0:
            if duration - 0.1 not in frame_times:  # Avoid duplicate with interval
                frame_times.append(duration - 0.1)
        
        # Limit total frames
        if len(frame_times) > config.max_frames:
            # Keep first and last, sample middle
            middle = frame_times[1:-1]
            step = len(middle) // (config.max_frames - 2)
            step = max(1, step)
            sampled_middle = middle[::step][:config.max_frames - 2]
            frame_times = [frame_times[0]] + sampled_middle + [frame_times[-1]]
        
        # Extract frames
        for i, t in enumerate(sorted(set(frame_times))):
            frame_num = int(t * fps)
            frame_num = min(frame_num, total_frames - 1)
            frame_num = max(0, frame_num)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                # Save frame
                frame_path = os.path.join(
                    self.temp_dir, 
                    f"frame_{i:03d}_{t:.1f}s.{config.output_format}"
                )
                
                # Resize if needed
                frame = self._resize_frame(frame, config.max_dimension)
                
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, config.output_quality])
                
                extracted.append(ExtractedFrame(
                    frame_path=frame_path,
                    timestamp=t,
                    frame_number=frame_num,
                    is_first=(i == 0 and config.include_first_frame),
                    is_last=(t >= duration - 0.5 and config.include_last_frame),
                ))
        
        return extracted
    
    def _extract_scene_changes(
        self,
        cap: 'cv2.VideoCapture',
        config: ExtractionConfig,
        fps: float,
        total_frames: int,
        duration: float
    ) -> List[ExtractedFrame]:
        """Extract frames at scene changes."""
        extracted = []
        prev_frame = None
        scene_changes = []
        
        # Sample frames to detect scene changes
        sample_interval = max(1, int(fps / 4))  # 4 samples per second
        
        frame_num = 0
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if prev_frame is not None:
                # Calculate frame difference
                diff = self._calculate_frame_diff(prev_frame, frame)
                
                if diff > config.scene_change_threshold:
                    scene_changes.append({
                        'frame_num': frame_num,
                        'timestamp': frame_num / fps,
                        'diff': diff
                    })
            
            prev_frame = frame.copy()
            frame_num += sample_interval
        
        # Sort by diff magnitude and take top N
        scene_changes.sort(key=lambda x: x['diff'], reverse=True)
        selected = scene_changes[:config.max_frames - 2]  # Leave room for first/last
        
        # Add first and last
        frames_to_extract = []
        
        if config.include_first_frame:
            frames_to_extract.append({'frame_num': 0, 'timestamp': 0, 'is_first': True})
        
        for sc in sorted(selected, key=lambda x: x['timestamp']):
            frames_to_extract.append({
                'frame_num': sc['frame_num'], 
                'timestamp': sc['timestamp'],
                'is_scene_change': True
            })
        
        if config.include_last_frame:
            frames_to_extract.append({
                'frame_num': total_frames - 1, 
                'timestamp': duration - 0.1,
                'is_last': True
            })
        
        # Extract the frames
        for i, frame_info in enumerate(frames_to_extract):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_info['frame_num'])
            ret, frame = cap.read()
            
            if ret:
                frame_path = os.path.join(
                    self.temp_dir,
                    f"frame_{i:03d}_{frame_info['timestamp']:.1f}s.{config.output_format}"
                )
                
                frame = self._resize_frame(frame, config.max_dimension)
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, config.output_quality])
                
                extracted.append(ExtractedFrame(
                    frame_path=frame_path,
                    timestamp=frame_info['timestamp'],
                    frame_number=frame_info['frame_num'],
                    is_first=frame_info.get('is_first', False),
                    is_last=frame_info.get('is_last', False),
                    is_scene_change=frame_info.get('is_scene_change', False),
                ))
        
        return extracted
    
    def _calculate_frame_diff(self, frame1, frame2) -> float:
        """Calculate difference between two frames."""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Resize for speed
        gray1 = cv2.resize(gray1, (160, 90))
        gray2 = cv2.resize(gray2, (160, 90))
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        return diff.mean()
    
    def _resize_frame(self, frame, max_dim: int):
        """Resize frame if larger than max dimension."""
        h, w = frame.shape[:2]
        
        if max(h, w) <= max_dim:
            return frame
        
        if w > h:
            new_w = max_dim
            new_h = int(h * max_dim / w)
        else:
            new_h = max_dim
            new_w = int(w * max_dim / h)
        
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def _aggregate_insights(self, result: VideoAnalysisResult) -> VideoAnalysisResult:
        """Aggregate insights from frame analyses."""
        
        scene_types = {}
        moods = {}
        
        for fa in result.frame_analyses:
            # Track scene types
            if fa.scene_type:
                scene_types[fa.scene_type] = scene_types.get(fa.scene_type, 0) + 1
            
            # Track moods
            if fa.mood:
                moods[fa.mood] = moods.get(fa.mood, 0) + 1
            
            # First human appearance
            if fa.humans_present and result.first_human_appearance < 0:
                result.first_human_appearance = fa.timestamp
            
            # Human in opening (first 3 seconds)
            if fa.timestamp <= 3 and fa.humans_present:
                result.has_human_in_opening = True
            
            # First logo appearance
            if fa.logo_visible and result.logo_first_appearance < 0:
                result.logo_first_appearance = fa.timestamp
            
            # CTA present
            if fa.cta_present:
                result.cta_present = True
        
        result.scene_types = scene_types
        result.dominant_mood = max(moods.keys(), key=lambda k: moods[k]) if moods else ""
        
        return result
    
    def _cleanup_temp_files(self, frames: List[ExtractedFrame]):
        """Clean up temporary frame files."""
        for frame in frames:
            try:
                if os.path.exists(frame.frame_path):
                    os.remove(frame.frame_path)
            except:
                pass
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                os.rmdir(self.temp_dir)
            except:
                pass  # Directory might not be empty


def get_frame_analyzer(vision_service: LocalVisionService = None) -> FrameAnalyzer:
    """Get frame analyzer instance."""
    return FrameAnalyzer(vision_service)
