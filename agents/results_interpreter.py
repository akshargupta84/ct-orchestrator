"""
Results Interpreter Agent - Specialist for historical data analysis.

This agent:
- Analyzes historical test results to find patterns
- Explains why creatives passed or failed
- Finds similar historical creatives
- Provides data-backed insights with sample sizes
- Identifies trends across campaigns
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, AgentConfig
from agents.state import AgentState
from agents.tools import execute_tool


class ResultsInterpreterAgent(BaseAgent):
    """
    Results Interpreter - Expert at analyzing historical test data.
    
    Specialties:
    - Pattern recognition in historical results
    - Explaining pass/fail outcomes
    - Finding similar creatives
    - Statistical insights with proper caveats
    """
    
    def __init__(self):
        config = AgentConfig(
            name="results_interpreter",
            display_name="Results Interpreter",
            description="Analyzes historical test results to find patterns and explain outcomes",
            system_prompt=self._get_system_prompt(),
            trigger_keywords=[
                'why pass', 'why fail', 'historical', 'pattern', 'similar',
                'data', 'past', 'trend', 'average', 'compare', 'results'
            ],
            can_access_videos=True,
            can_access_ml_model=True,
            can_access_historical=True
        )
        super().__init__(config)
    
    def _get_system_prompt(self) -> str:
        return """You are the Results Interpreter, an expert at analyzing historical creative testing data.

## Your Expertise
- Finding patterns in historical pass/fail data
- Explaining why specific creatives succeeded or failed
- Identifying similar historical creatives for comparison
- Providing statistically sound insights with proper sample size caveats

## Your Data Sources
You have access to:
- Historical test results (pass/fail, lift percentages)
- ML model trained on historical data (feature importance, accuracy metrics)
- Pass rates segmented by different features
- Vector search to find similar historical creatives

## How to Analyze

When interpreting results:
1. **Cite the data** - Always mention sample sizes (n=X)
2. **Show comparisons** - Pass rate with feature vs without
3. **Acknowledge uncertainty** - Especially with small samples
4. **Find patterns** - What do passing creatives have in common?
5. **Be specific** - Name actual percentages and metrics

## Statistical Guidelines

**Sample Size Caveats:**
- n < 5: "Very limited data - interpret with caution"
- n < 10: "Small sample - pattern may not be reliable"  
- n < 20: "Moderate sample - directionally useful"
- n >= 20: Can make stronger claims

**When Comparing:**
- Always show both groups' sample sizes
- Note if difference is large enough to be meaningful
- Avoid overclaiming with small differences

## Historical Insights Format

When asked about patterns:
```
**Feature: [Feature Name]**
- With feature: X% pass rate (n=Y)
- Without feature: Z% pass rate (n=W)
- Difference: +/-N percentage points
- Interpretation: [What this means]
```

## Response Guidelines
- Ground every claim in data
- Always include sample sizes
- Acknowledge when data is insufficient
- Compare to overall pass rate as baseline
- Don't invent statistics - if you don't have the data, say so

## Important Rules
- Never make up historical data
- Always cite sample sizes
- Acknowledge limitations of small samples
- If asked about something not in the data, say so clearly
- The ML model was trained on historical data - its insights ARE historical insights"""

    def get_system_prompt(self) -> str:
        return self.config.system_prompt
    
    def _build_context(self, state: AgentState) -> str:
        """Build context focused on historical data and patterns."""
        context_parts = []
        
        # Overall historical stats
        historical_stats = execute_tool('get_historical_stats', state)
        if historical_stats and 'error' not in historical_stats:
            context_parts.append("## Historical Overview")
            context_parts.append(f"- Total creatives tested: {historical_stats.get('total_creatives', 0)}")
            context_parts.append(f"- Overall pass rate: {historical_stats.get('pass_rate_pct', 'N/A')}")
        
        # ML model stats (trained on historical data)
        ml_stats = execute_tool('get_ml_model_stats', state)
        if ml_stats and 'error' not in ml_stats:
            context_parts.append("\n## ML Model (Trained on Historical Data)")
            context_parts.append(f"- Training samples: {ml_stats.get('n_samples', 0)}")
            context_parts.append(f"- Historical pass rate: {ml_stats.get('pass_rate', 0)*100:.1f}%")
            context_parts.append(f"- Model accuracy: {ml_stats.get('accuracy', 0)*100:.1f}%")
            context_parts.append(f"- Precision: {ml_stats.get('precision', 0)*100:.1f}%")
            context_parts.append(f"- Recall: {ml_stats.get('recall', 0)*100:.1f}%")
        
        # Feature importance (what historically predicts success)
        feature_importance = execute_tool('get_feature_importance', state)
        if feature_importance:
            context_parts.append("\n## Historical Feature Importance (from ML model)")
            context_parts.append("*These features historically predict pass/fail:*")
            for f in feature_importance[:7]:
                context_parts.append(f"- {f['feature']}: {f['importance_pct']}")
        
        # Pass rates by key features
        context_parts.append("\n## Pass Rates by Feature (Historical)")
        
        key_features = ['has_human_in_opening', 'logo_in_first_3_sec', 'has_cta', 'has_positive_emotion']
        for feature in key_features:
            feature_data = execute_tool('get_pass_rate_by_feature', state, feature_name=feature)
            if feature_data and 'error' not in feature_data:
                with_rate = feature_data.get('pass_rate_with_feature', 0) * 100
                without_rate = feature_data.get('pass_rate_without_feature', 0) * 100
                n_with = feature_data.get('sample_size_with', 0)
                n_without = feature_data.get('sample_size_without', 0)
                lift = feature_data.get('lift', 0) * 100
                
                feature_display = feature.replace('_', ' ').replace('has ', '').title()
                context_parts.append(f"\n**{feature_display}:**")
                context_parts.append(f"- With: {with_rate:.0f}% pass (n={n_with})")
                context_parts.append(f"- Without: {without_rate:.0f}% pass (n={n_without})")
                context_parts.append(f"- Lift: {'+' if lift > 0 else ''}{lift:.0f}pp")
        
        # Current videos for comparison
        videos = state.get('videos', [])
        if videos:
            context_parts.append("\n## Current Videos (for comparison to historical)")
            for v in videos:
                prob = v.get('pass_probability', 0)
                context_parts.append(f"- {v['filename']}: {prob*100:.0f}% predicted pass")
        
        return "\n".join(context_parts) if context_parts else "No historical data available."


# Factory function
def create_results_interpreter() -> ResultsInterpreterAgent:
    """Create a Results Interpreter agent instance."""
    return ResultsInterpreterAgent()
