"""
Strategy Agent - Specialist for strategic recommendations.

This agent:
- Provides actionable recommendations
- Synthesizes insights into next steps
- Prioritizes actions based on impact
- Thinks about long-term learning and optimization
- Advises on testing strategy
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, AgentConfig
from agents.state import AgentState
from agents.tools import execute_tool


class StrategyAgent(BaseAgent):
    """
    Strategy Agent - Expert at strategic recommendations.
    
    Specialties:
    - Actionable recommendations
    - Prioritization
    - Long-term optimization
    - Testing strategy
    """
    
    def __init__(self):
        config = AgentConfig(
            name="strategy",
            display_name="Strategy Agent",
            description="Provides strategic recommendations and prioritizes actions",
            system_prompt=self._get_system_prompt(),
            trigger_keywords=[
                'recommend', 'should', 'next', 'strategy', 'advice', 'suggest',
                'optimize', 'improve', 'priority', 'action', 'what to do'
            ],
            can_access_media_plan=True,
            can_access_videos=True,
            can_access_ml_model=True,
            can_access_rules=True,
            can_access_historical=True
        )
        super().__init__(config)
    
    def _get_system_prompt(self) -> str:
        return """You are the Strategy Agent, an expert at translating data insights into actionable recommendations.

## Your Expertise
- Turning analysis into clear action items
- Prioritizing based on impact and effort
- Thinking about long-term testing program optimization
- Balancing risk and reward in creative testing
- Budget-aware recommendations

## Your Data Sources
You have access to:
- Media plan (budget, campaign objectives)
- Video scores and risk factors
- ML model insights (what predicts success)
- Historical pass rates
- CT rules and constraints

## How to Recommend

When providing recommendations:
1. **Be specific** - Not "improve the creative" but "add human in opening frame"
2. **Prioritize by impact** - What will move the needle most?
3. **Consider constraints** - Budget, timeline, effort required
4. **Explain the why** - Connect recommendation to data
5. **Provide alternatives** - If Option A isn't feasible, offer B and C

## Recommendation Framework

**Immediate Actions** (do now):
- High impact, low effort
- Clear path forward
- Based on strong data signal

**Near-term Opportunities** (plan for):
- Medium impact, medium effort
- Requires some preparation
- Good ROI

**Strategic Considerations** (think about):
- Long-term improvements
- Program-level changes
- Learning opportunities

## Response Format

Structure recommendations clearly:
```
## Recommendation: [Clear action]

**Why:** [Data-backed reasoning]

**Expected Impact:** [What will improve]

**Effort:** [Low/Medium/High]

**How to Execute:** [Specific steps]
```

## Decision Frameworks

**Should I test this creative?**
Consider:
- Pass probability (>65% = likely worth it, <40% = risky)
- Budget constraints (can you afford a failure?)
- Learning value (even failures teach something)
- Strategic importance (is this a key message?)

**Which creatives to prioritize?**
1. Highest pass probability (if budget limited)
2. Most strategically important (if learning is goal)
3. Most different from each other (for learning diversity)

**When to skip testing:**
- Very low predicted pass (<35%) AND budget is tight
- Near-duplicate of already-tested creative
- Doesn't align with campaign objectives

## Important Rules
- Always ground recommendations in data
- Acknowledge tradeoffs honestly
- Don't oversimplify complex decisions
- Respect budget constraints
- If you need information from other agents, ask with @REQUEST[agent]: question"""

    def get_system_prompt(self) -> str:
        return self.config.system_prompt
    
    def _build_context(self, state: AgentState) -> str:
        """Build context for strategic recommendations."""
        context_parts = []
        
        # Media plan / campaign context
        media_plan = state.get('media_plan_info')
        if media_plan:
            context_parts.append("## Campaign Context")
            context_parts.append(f"- Brand: {media_plan.get('brand', 'Unknown')}")
            context_parts.append(f"- Campaign: {media_plan.get('campaign_name', 'Unknown')}")
            
            budget = media_plan.get('total_budget', 0)
            if budget:
                testing_budget = budget * 0.04
                context_parts.append(f"- Total Budget: ${budget:,.0f}")
                context_parts.append(f"- Max Testing Budget (4%): ${testing_budget:,.0f}")
            
            if media_plan.get('primary_kpi'):
                context_parts.append(f"- Primary KPI: {media_plan['primary_kpi']}")
        
        # Budget status
        budget_info = execute_tool('get_testing_budget', state)
        if budget_info and 'error' not in budget_info:
            context_parts.append("\n## Budget Status")
            context_parts.append(f"- Can test up to: {budget_info.get('max_videos', 0)} videos")
            context_parts.append(f"- Currently have: {budget_info.get('current_videos', 0)} videos")
            context_parts.append(f"- Testing cost: ${budget_info.get('testing_cost', 0):,.0f}")
            context_parts.append(f"- Status: **{budget_info.get('budget_status', 'unknown').replace('_', ' ').title()}**")
        
        # Videos ranked by score
        videos = state.get('videos', [])
        if videos:
            # Sort by pass probability
            sorted_videos = sorted(videos, key=lambda v: v.get('pass_probability', 0), reverse=True)
            
            context_parts.append("\n## Videos Ranked by Pass Probability")
            for i, v in enumerate(sorted_videos, 1):
                prob = v.get('pass_probability', 0)
                status = '✅' if prob >= 0.7 else '⚠️' if prob >= 0.5 else '❌'
                dup = " (duplicate)" if v.get('is_duplicate_of') else ""
                context_parts.append(f"{i}. {v['filename']}: {prob*100:.0f}% {status}{dup}")
            
            # Summary stats
            non_dup_videos = [v for v in videos if not v.get('is_duplicate_of')]
            if non_dup_videos:
                avg_prob = sum(v.get('pass_probability', 0) for v in non_dup_videos) / len(non_dup_videos)
                high_prob = len([v for v in non_dup_videos if v.get('pass_probability', 0) >= 0.7])
                low_prob = len([v for v in non_dup_videos if v.get('pass_probability', 0) < 0.5])
                
                context_parts.append(f"\n**Summary:** {len(non_dup_videos)} unique videos")
                context_parts.append(f"- Average pass probability: {avg_prob*100:.0f}%")
                context_parts.append(f"- Strong (≥70%): {high_prob}")
                context_parts.append(f"- Risky (<50%): {low_prob}")
        
        # Duplicates
        duplicates = state.get('duplicates_detected', [])
        if duplicates:
            context_parts.append(f"\n## Duplicates Detected: {len(duplicates)}")
            for d in duplicates:
                context_parts.append(f"- {d[0]} ≈ {d[1]}")
        
        # Historical context
        ml_stats = execute_tool('get_ml_model_stats', state)
        if ml_stats and 'error' not in ml_stats:
            context_parts.append("\n## Historical Benchmarks")
            context_parts.append(f"- Historical pass rate: {ml_stats.get('pass_rate', 0)*100:.1f}%")
            context_parts.append(f"- Model accuracy: {ml_stats.get('accuracy', 0)*100:.1f}%")
        
        # CT Rules
        rules = execute_tool('get_ct_rules', state)
        if rules:
            context_parts.append("\n## CT Rules")
            context_parts.append(f"- Video test cost: {rules.get('video_test_cost_display', '$15,000')}")
            context_parts.append(f"- Turnaround: {rules.get('standard_turnaround_display', '2-3 weeks')}")
        
        return "\n".join(context_parts) if context_parts else "No context available for strategic recommendations."


# Factory function
def create_strategy_agent() -> StrategyAgent:
    """Create a Strategy Agent instance."""
    return StrategyAgent()
