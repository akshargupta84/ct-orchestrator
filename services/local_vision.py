"""
Local Vision Service.

Interfaces with Ollama to run vision models (LLaVA, etc.) locally on Apple Silicon.
Provides semantic understanding of video frames.
"""

import base64
import json
import os
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field
import time

# Try to import ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Try to import PIL for image handling
try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class FrameAnalysis:
    """Analysis result for a single frame."""
    timestamp: float  # Seconds into video
    frame_path: str  # Path to extracted frame image
    
    # Scene description
    description: str = ""
    scene_type: str = ""  # lifestyle, product, testimonial, text, transition
    setting: str = ""  # indoor, outdoor, studio, etc.
    mood: str = ""  # energetic, calm, emotional, professional
    
    # Human analysis
    humans_present: bool = False
    human_count: int = 0
    human_emotions: list = field(default_factory=list)  # ["happy", "excited"]
    human_actions: list = field(default_factory=list)  # ["using phone", "smiling"]
    human_looking_at_camera: bool = False
    human_age_range: str = ""  # "20-30", "30-40", etc.
    
    # Brand elements
    logo_visible: bool = False
    logo_position: str = ""  # "top-left", "center", "on product"
    brand_colors_present: list = field(default_factory=list)
    product_visible: bool = False
    product_description: str = ""
    
    # Text and CTA
    text_on_screen: list = field(default_factory=list)
    cta_present: bool = False
    cta_text: str = ""
    
    # Technical
    is_opening_frame: bool = False
    is_closing_frame: bool = False
    visual_complexity: str = "medium"  # low, medium, high
    
    # Raw response
    raw_response: str = ""


@dataclass
class VideoAnalysisResult:
    """Complete analysis of a video."""
    video_path: str
    duration_seconds: float
    frame_count: int
    
    # Frame analyses
    frame_analyses: list = field(default_factory=list)  # List[FrameAnalysis]
    
    # Aggregated insights
    has_human_in_opening: bool = False
    first_human_appearance: float = -1  # Seconds, -1 if none
    logo_first_appearance: float = -1
    cta_present: bool = False
    
    # Scene breakdown
    scene_types: dict = field(default_factory=dict)  # {type: count}
    dominant_mood: str = ""
    
    # Summary
    ai_summary: str = ""
    
    # Processing info
    processing_time_seconds: float = 0
    model_used: str = ""
    errors: list = field(default_factory=list)


class LocalVisionService:
    """
    Service for analyzing video frames using local vision models via Ollama.
    """
    
    # Available vision models in order of preference
    VISION_MODELS = [
        "llava:13b",      # Best quality, needs 16GB+ RAM
        "llava:7b",       # Good quality, needs 8GB+ RAM
        "llava:latest",   # Default llava
        "bakllava",       # Alternative vision model
        "llava-llama3",   # LLaVA with Llama 3
    ]
    
    # Text generation models for summaries
    TEXT_MODELS = [
        "llama3.1:8b",
        "llama3:8b",
        "mistral:7b",
        "phi3:mini",
    ]
    
    def __init__(self, vision_model: str = None, text_model: str = None):
        """
        Initialize the vision service.
        
        Args:
            vision_model: Specific vision model to use (or auto-detect)
            text_model: Specific text model for summaries (or auto-detect)
        """
        self.vision_model = vision_model
        self.text_model = text_model
        self._available_models = None
        self._initialized = False
        
    def is_available(self) -> bool:
        """Check if Ollama and required models are available."""
        if not OLLAMA_AVAILABLE:
            return False
        
        try:
            # Check if Ollama is running
            models_response = ollama.list()
            
            # Handle different response formats (dict vs object)
            models_list = []
            if isinstance(models_response, dict):
                models_list = models_response.get('models', [])
            elif hasattr(models_response, 'models'):
                models_list = models_response.models
            
            # Extract model names - handle both dict and object formats
            self._available_models = []
            for m in models_list:
                if isinstance(m, dict):
                    name = m.get('name', m.get('model', ''))
                elif hasattr(m, 'name'):
                    name = m.name
                elif hasattr(m, 'model'):
                    name = m.model
                else:
                    name = str(m)
                if name:
                    self._available_models.append(name)
            
            return len(self._available_models) > 0
        except Exception as e:
            print(f"Ollama check error: {e}")
            return False
    
    def get_available_models(self) -> dict:
        """Get available vision and text models."""
        if not self.is_available():
            return {"vision": [], "text": [], "error": "Ollama not available"}
        
        vision = [m for m in self._available_models if any(v in m for v in ['llava', 'bakllava', 'cogvlm'])]
        text = [m for m in self._available_models if any(t in m for t in ['llama', 'mistral', 'phi', 'qwen'])]
        
        return {"vision": vision, "text": text}
    
    def _select_vision_model(self) -> Optional[str]:
        """Select the best available vision model."""
        if self.vision_model and self.vision_model in self._available_models:
            return self.vision_model
        
        for model in self.VISION_MODELS:
            # Check both exact match and partial match
            for available in self._available_models:
                if model in available or available in model:
                    return available
        
        return None
    
    def _select_text_model(self) -> Optional[str]:
        """Select the best available text model."""
        if self.text_model and self.text_model in self._available_models:
            return self.text_model
        
        for model in self.TEXT_MODELS:
            for available in self._available_models:
                if model in available or available in model:
                    return available
        
        return None
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for Ollama."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _resize_image_if_needed(self, image_path: str, max_size: int = 1024) -> str:
        """Resize image if too large to speed up processing."""
        if not PIL_AVAILABLE:
            return image_path
        
        try:
            img = Image.open(image_path)
            
            # Check if resizing needed
            if max(img.size) <= max_size:
                return image_path
            
            # Calculate new size maintaining aspect ratio
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            
            # Resize
            img_resized = img.resize(new_size, Image.LANCZOS)
            
            # Save to temp path
            resized_path = image_path.replace('.', '_resized.')
            img_resized.save(resized_path, quality=85)
            
            return resized_path
        except Exception:
            return image_path
    
    def analyze_frame(
        self, 
        image_path: str, 
        timestamp: float = 0,
        is_opening: bool = False,
        is_closing: bool = False,
        brand_name: str = "",
        product_name: str = "",
    ) -> FrameAnalysis:
        """
        Analyze a single frame using the vision model.
        
        Args:
            image_path: Path to the frame image
            timestamp: Timestamp in seconds
            is_opening: Whether this is the opening frame
            is_closing: Whether this is the closing frame
            brand_name: Brand name to look for
            product_name: Product name to look for
            
        Returns:
            FrameAnalysis with detailed understanding of the frame
        """
        analysis = FrameAnalysis(
            timestamp=timestamp,
            frame_path=image_path,
            is_opening_frame=is_opening,
            is_closing_frame=is_closing,
        )
        
        if not self.is_available():
            analysis.description = "Vision model not available"
            analysis.errors = ["Ollama not running or no vision model installed"]
            return analysis
        
        vision_model = self._select_vision_model()
        if not vision_model:
            analysis.description = "No vision model found"
            analysis.errors = ["Install a vision model: ollama pull llava:13b"]
            return analysis
        
        # Resize image if needed
        processed_path = self._resize_image_if_needed(image_path)
        
        # Build the prompt
        prompt = self._build_frame_analysis_prompt(
            is_opening, is_closing, brand_name, product_name
        )
        
        try:
            # Call Ollama vision model
            response = ollama.chat(
                model=vision_model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [processed_path]
                }],
                options={
                    'temperature': 0,  # Zero temperature for deterministic analysis
                    'num_predict': 1000,  # Limit response length
                }
            )
            
            raw_response = response['message']['content']
            analysis.raw_response = raw_response
            
            # Parse the response
            analysis = self._parse_frame_response(analysis, raw_response)
            
        except Exception as e:
            analysis.errors = [str(e)]
            analysis.description = f"Error analyzing frame: {e}"
        
        # Clean up resized image if created
        if processed_path != image_path and os.path.exists(processed_path):
            try:
                os.remove(processed_path)
            except:
                pass
        
        return analysis
    
    def _build_frame_analysis_prompt(
        self, 
        is_opening: bool, 
        is_closing: bool,
        brand_name: str,
        product_name: str
    ) -> str:
        """Build the prompt for frame analysis."""
        
        context = ""
        if is_opening:
            context = "This is the OPENING frame of a video advertisement. Pay special attention to what catches attention first."
        elif is_closing:
            context = "This is the CLOSING frame of a video advertisement. Look for call-to-action, logo, and final message."
        else:
            context = "This is a frame from a video advertisement."
        
        brand_context = ""
        if brand_name:
            brand_context = f"The brand is {brand_name}."
        if product_name:
            brand_context += f" The product is {product_name}."
        
        return f"""Analyze this video advertisement frame. {context} {brand_context}

Provide a detailed analysis in the following JSON format:
{{
    "description": "Brief description of what's shown in this frame",
    "scene_type": "one of: lifestyle, product_demo, testimonial, text_card, transition, other",
    "setting": "indoor/outdoor/studio/abstract",
    "mood": "one of: energetic, calm, emotional, professional, playful, dramatic",
    
    "humans": {{
        "present": true/false,
        "count": number,
        "emotions": ["list of emotions detected"],
        "actions": ["what they are doing"],
        "looking_at_camera": true/false,
        "age_range": "estimated age range like 20-30"
    }},
    
    "brand_elements": {{
        "logo_visible": true/false,
        "logo_position": "where the logo appears if visible",
        "brand_colors": ["colors that might be brand colors"],
        "product_visible": true/false,
        "product_description": "what the product looks like if visible"
    }},
    
    "text": {{
        "text_on_screen": ["any text visible"],
        "cta_present": true/false,
        "cta_text": "the call to action if present"
    }},
    
    "visual_complexity": "low/medium/high"
}}

Respond ONLY with the JSON, no other text."""
    
    def _parse_frame_response(self, analysis: FrameAnalysis, response: str) -> FrameAnalysis:
        """Parse the vision model response into structured data."""
        try:
            # Try to extract JSON from response
            json_str = response.strip()
            
            # Handle markdown code blocks
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            data = json.loads(json_str)
            
            # Map parsed data to analysis object
            analysis.description = data.get("description", "")
            analysis.scene_type = data.get("scene_type", "")
            analysis.setting = data.get("setting", "")
            analysis.mood = data.get("mood", "")
            
            # Humans
            humans = data.get("humans", {})
            analysis.humans_present = humans.get("present", False)
            analysis.human_count = humans.get("count", 0)
            analysis.human_emotions = humans.get("emotions", [])
            analysis.human_actions = humans.get("actions", [])
            analysis.human_looking_at_camera = humans.get("looking_at_camera", False)
            analysis.human_age_range = humans.get("age_range", "")
            
            # Brand elements
            brand = data.get("brand_elements", {})
            analysis.logo_visible = brand.get("logo_visible", False)
            analysis.logo_position = brand.get("logo_position", "")
            analysis.brand_colors_present = brand.get("brand_colors", [])
            analysis.product_visible = brand.get("product_visible", False)
            analysis.product_description = brand.get("product_description", "")
            
            # Text
            text = data.get("text", {})
            analysis.text_on_screen = text.get("text_on_screen", [])
            analysis.cta_present = text.get("cta_present", False)
            analysis.cta_text = text.get("cta_text", "")
            
            analysis.visual_complexity = data.get("visual_complexity", "medium")
            
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract key information from text
            response_lower = response.lower()
            
            analysis.description = response[:500]  # Use first 500 chars as description
            
            # Simple keyword detection
            analysis.humans_present = any(word in response_lower for word in ['person', 'people', 'human', 'man', 'woman', 'face'])
            analysis.logo_visible = any(word in response_lower for word in ['logo', 'brand', 'watermark'])
            analysis.product_visible = any(word in response_lower for word in ['product', 'phone', 'device', 'item'])
            analysis.cta_present = any(word in response_lower for word in ['call to action', 'cta', 'click', 'buy', 'shop', 'learn more'])
            
        return analysis
    
    def generate_video_summary(
        self, 
        frame_analyses: list,
        video_duration: float,
        brand_name: str = "",
    ) -> str:
        """
        Generate a summary of the entire video based on frame analyses.
        
        Args:
            frame_analyses: List of FrameAnalysis objects
            video_duration: Total duration in seconds
            brand_name: Brand name for context
            
        Returns:
            AI-generated summary of the video
        """
        if not frame_analyses:
            return "No frames analyzed"
        
        text_model = self._select_text_model()
        if not text_model:
            # Fall back to vision model
            text_model = self._select_vision_model()
        
        if not text_model:
            return "No model available for summary generation"
        
        # Build frame summary
        frame_summaries = []
        for fa in frame_analyses:
            timestamp = f"{fa.timestamp:.1f}s"
            summary = f"[{timestamp}] {fa.description}"
            if fa.humans_present:
                summary += f" (Human: {', '.join(fa.human_emotions) if fa.human_emotions else 'present'})"
            if fa.logo_visible:
                summary += " (Logo visible)"
            if fa.cta_present:
                summary += f" (CTA: {fa.cta_text})"
            frame_summaries.append(summary)
        
        prompt = f"""Based on the following frame-by-frame analysis of a {video_duration:.0f} second video advertisement{f' for {brand_name}' if brand_name else ''}, 
write a concise summary of the creative:

Frame Analysis:
{chr(10).join(frame_summaries)}

Provide a 2-3 sentence summary describing:
1. The overall narrative/story of the ad
2. The key visual elements and style
3. The main message or call to action

Keep it concise and factual."""

        try:
            response = ollama.chat(
                model=text_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0, 'num_predict': 300}
            )
            return response['message']['content'].strip()
        except Exception as e:
            return f"Summary generation failed: {e}"


# Singleton instance
_vision_service = None


def get_vision_service(vision_model: str = None, text_model: str = None) -> LocalVisionService:
    """Get singleton vision service instance."""
    global _vision_service
    if _vision_service is None:
        _vision_service = LocalVisionService(vision_model, text_model)
    return _vision_service
