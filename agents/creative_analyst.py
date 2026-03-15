"""
Creative Analyst Agent - Specialist for video/creative analysis.

This agent:
- Interprets video features extracted by Ollama
- Identifies creative strengths and weaknesses
- Explains why scores are high or low
- Suggests improvements
- Compares to best practices
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, AgentConfig
from agents.state import AgentState


class CreativeAnalystAgent(BaseAgent):
    """
    Creative Analyst - Expert at analyzing video creative content.
    
    Specialties:
    - Video feature interpretation
    - Strength/weakness identification
    - Improvement recommendations
    - Risk factor explanation
    """
    
    def __init__(self):
        config = AgentConfig(
            name="creative_analyst",
            display_name="Creative Analyst",
            description="Analyzes video creatives to identify strengths, weaknesses, and improvement opportunities",
            system_prompt=self._get_system_prompt(),
            trigger_keywords=[
                'analyze', 'creative', 'video', 'score', 'why', 'improve',
                'features', 'risk', 'strength', 'weakness', 'what\'s wrong'
            ],
            can_access_videos=True,
            can_access_ml_model=True,
            can_access_historical=True
        )
        super().__init__(config)
    
    def _get_system_prompt(self) -> str:
        return """You are the Creative Analyst, an expert at analyzing video advertising creatives.

## Your Expertise
- Interpreting video features (human presence, logo timing, CTAs, emotions, scene composition)
- Identifying what makes creatives effective or ineffective
- Explaining risk factors in plain language
- Suggesting specific, actionable improvements

## Your Data Sources
You have access to:
- Video features extracted by AI vision analysis (human detection, logo timing, emotional content, etc.)
- Predicted diagnostic scores (attention, brand recall, message clarity, emotional resonance, uniqueness)
- ML model insights showing which features drive pass/fail
- Historical patterns from similar creatives

## How to Analyze

When analyzing a creative:
1. **Start with the score** - State the pass probability and what it means
2. **Identify strengths** - What the creative does well (cite specific features)
3. **Identify risks** - What might cause it to fail (cite specific features)
4. **Explain WHY** - Connect features to likely viewer response
5. **Suggest improvements** - Specific, actionable changes

## Feature Interpretation Guide

**Human Presence:**
- Human in opening → +15-20% attention lift
- Eye contact with camera → Stronger connection
- Missing humans → Risk for awareness campaigns

**Logo/Brand:**
- Logo in first 3 seconds → Better brand recall
- Logo appearing late (>8s) → Brand attribution risk
- High logo frequency → Can feel "salesy" but aids recall

**Emotional Content:**
- Positive emotions → Generally better engagement
- No clear emotion → "Flat" creative, lower resonance

**CTA (Call to Action):**
- CTA in final frames → Drives action
- Missing CTA → Missed conversion opportunity

## Response Format

Use clear structure:
- Use **bold** for key metrics and findings
- Use bullet points for lists
- Be specific - cite actual feature values
- Be concise - focus on what matters most

## Important Rules
- Always ground analysis in actual feature data
- Don't invent features that weren't detected
- Acknowledge uncertainty when relevant
- If asked about a video not in the data, say so clearly"""

    def get_system_prompt(self) -> str:
        return self.config.system_prompt
    
    def _build_context(self, state: AgentState) -> str:
        """Build context focused on video analysis data."""
        context_parts = []
        
        # Get all videos with their features
        videos = state.get('videos', [])
        if videos:
            context_parts.append("## Uploaded Videos\n")
            for v in videos:
                prob = v.get('pass_probability', 0)
                status = '✅ Strong' if prob >= 0.7 else '⚠️ Moderate' if prob >= 0.5 else '❌ Risky'
                
                video_info = [
                    f"### {v['filename']}",
                    f"**Pass Probability:** {prob*100:.0f}% ({status})",
                ]
                
                # Add features if available
                features = v.get('features', {})
                if features:
                    video_info.append("\n**Detected Features:**")
                    
                    # Human presence
                    if features.get('has_human_in_opening'):
                        video_info.append(f"- ✅ Human in opening frame")
                    else:
                        video_info.append(f"- ⚠️ No human in opening")
                    
                    if features.get('human_frame_ratio'):
                        video_info.append(f"- Human visible in {features['human_frame_ratio']*100:.0f}% of frames")
                    
                    # Logo/brand
                    if features.get('logo_in_first_3_sec'):
                        video_info.append(f"- ✅ Logo in first 3 seconds")
                    elif features.get('logo_first_appearance_sec'):
                        video_info.append(f"- ⚠️ Logo first appears at {features['logo_first_appearance_sec']:.1f}s")
                    
                    if features.get('logo_frame_ratio'):
                        video_info.append(f"- Logo visible in {features['logo_frame_ratio']*100:.0f}% of frames")
                    
                    # Emotional content
                    if features.get('has_positive_emotion'):
                        video_info.append(f"- ✅ Positive emotion detected")
                    if features.get('has_emotional_content'):
                        video_info.append(f"- ✅ Emotional content present")
                    if not features.get('has_positive_emotion') and not features.get('has_emotional_content'):
                        video_info.append(f"- ⚠️ No clear emotional expression")
                    
                    # CTA
                    if features.get('has_cta'):
                        if features.get('cta_in_last_5_sec'):
                            video_info.append(f"- ✅ CTA present in final frames")
                        else:
                            video_info.append(f"- ⚠️ CTA present but not in final frames")
                    else:
                        video_info.append(f"- ⚠️ No CTA detected")
                    
                    # Scene info
                    if features.get('scene_type_diversity'):
                        video_info.append(f"- Scene diversity: {features['scene_type_diversity']}")
                    if features.get('visual_complexity_score'):
                        video_info.append(f"- Visual complexity: {features['visual_complexity_score']:.1f}/10")
                
                # Risk factors
                risks = v.get('risk_factors', [])
                if risks:
                    video_info.append(f"\n**Risk Factors:** {', '.join(risks[:3])}")
                
                # Diagnostics if available
                diagnostics = v.get('diagnostics', {})
                if diagnostics:
                    video_info.append("\n**Predicted Diagnostics:**")
                    for diag, value in diagnostics.items():
                        if isinstance(value, (int, float)):
                            video_info.append(f"- {diag.replace('_', ' ').title()}: {value:.0f}/100")
                
                context_parts.append("\n".join(video_info))
                context_parts.append("")  # Blank line between videos
        
        # Add ML model insights
        from agents.tools import execute_tool
        
        feature_importance = execute_tool('get_feature_importance', state)
        if feature_importance:
            context_parts.append("## ML Model Insights - Top Predictors")
            for f in feature_importance[:5]:
                context_parts.append(f"- {f['feature']}: {f['importance_pct']} importance")
        
        return "\n".join(context_parts) if context_parts else "No videos uploaded for analysis."


# Factory function
def create_creative_analyst() -> CreativeAnalystAgent:
    """Create a Creative Analyst agent instance."""
    return CreativeAnalystAgent()
