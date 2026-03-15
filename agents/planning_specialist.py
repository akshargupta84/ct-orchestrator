"""
Planning Specialist Agent - Multi-agent version for test plan design.

This agent extends BaseAgent and specializes in:
- Budget optimization
- Creative prioritization
- Duplicate detection
- Test plan generation

Note: This is separate from planning_agent.py which is the standalone
conversational Planning Agent UI. This version integrates with the
multi-agent orchestration system.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, AgentConfig
from agents.state import AgentState
from agents.tools import execute_tool


class PlanningSpecialistAgent(BaseAgent):
    """
    Planning Specialist - Expert at test plan design and optimization.
    
    Specialties:
    - Budget analysis and constraints
    - Creative prioritization
    - Duplicate detection handling
    - Test plan generation
    - CT rules compliance
    """
    
    def __init__(self):
        config = AgentConfig(
            name="planning",
            display_name="Planning Agent",
            description="Designs optimal test plans with budget awareness and creative prioritization",
            system_prompt=self._get_system_prompt(),
            trigger_keywords=[
                'plan', 'test', 'budget', 'prioritize', 'which', 'how many',
                'afford', 'cost', 'duplicate', 'select', 'choose', 'schedule'
            ],
            can_access_media_plan=True,
            can_access_videos=True,
            can_access_rules=True
        )
        super().__init__(config)
    
    def _get_system_prompt(self) -> str:
        return """You are the Planning Agent, an expert at designing optimal creative test plans.

## Your Expertise
- Budget analysis and optimization
- Creative prioritization based on predicted performance
- Identifying and handling duplicate creatives
- Ensuring CT rules compliance
- Recommending which creatives to test vs skip

## Your Data Sources
You have access to:
- Media plan (brand, campaign, budget, creative line items)
- Video scores (pass probability for each creative)
- CT rules (4% budget limit, $15K per video, etc.)
- Duplicate detection results

## CT Rules Quick Reference
- Max testing budget: 4% of total media budget
- Video test cost: $15,000 per video
- Static test cost: $8,000 per static
- Turnaround: 2-3 weeks standard

## How to Plan

When creating a test plan:
1. **Check budget** - Calculate how many creatives can be tested
2. **Handle duplicates** - Don't test near-identical creatives
3. **Prioritize by score** - Higher pass probability = lower risk
4. **Consider strategy** - Sometimes test risky ones to learn
5. **Validate against rules** - Ensure compliance

## Budget Calculation

```
Max Testing Budget = Total Media Budget × 4%
Max Videos = Max Testing Budget ÷ $15,000
```

## Prioritization Logic

When budget forces a choice:
1. **Default:** Highest pass probability first (safest)
2. **Learning focus:** Include some moderate-risk for insights
3. **Strategic:** Prioritize key messages regardless of score

## Duplicate Handling

When duplicates detected:
- Recommend testing only one version
- Suggest keeping the higher-scored version
- Ask user preference if scores are similar

## Response Format

When presenting a plan:
```
## Test Plan Summary

**Campaign:** [Name]
**Budget:** $X total → $Y testing budget
**Capacity:** Can test N videos

### Recommended for Testing (N videos, $X)
1. [Creative] - XX% pass probability ✅
2. [Creative] - XX% pass probability ✅
...

### Not Recommended
- [Creative] - XX% (low score / duplicate / budget)

### Issues to Resolve
- [Any duplicates or decisions needed]
```

## Important Rules
- Always check budget constraints first
- Never recommend testing duplicates
- Cite specific pass probabilities
- Be clear about tradeoffs
- If budget is insufficient, provide options"""

    def get_system_prompt(self) -> str:
        return self.config.system_prompt
    
    def _build_context(self, state: AgentState) -> str:
        """Build context focused on planning data."""
        context_parts = []
        
        # Media plan info
        media_plan = state.get('media_plan_info')
        if media_plan:
            context_parts.append("## Media Plan")
            context_parts.append(f"- Brand: {media_plan.get('brand', 'Unknown')}")
            context_parts.append(f"- Campaign: {media_plan.get('campaign_name', 'Unknown')}")
            
            budget = media_plan.get('total_budget', 0)
            if budget:
                context_parts.append(f"- Total Media Budget: ${budget:,.0f}")
            
            if media_plan.get('flight_start') or media_plan.get('flight_end'):
                context_parts.append(f"- Flight: {media_plan.get('flight_start', '')} - {media_plan.get('flight_end', '')}")
            
            if media_plan.get('primary_kpi'):
                context_parts.append(f"- Primary KPI: {media_plan['primary_kpi']}")
            
            line_items = media_plan.get('creative_line_items', [])
            if line_items:
                context_parts.append(f"- Creative Line Items: {len(line_items)}")
                for item in line_items[:10]:
                    context_parts.append(f"  • {item.get('name', 'Unknown')}")
        
        # Budget analysis
        budget_info = execute_tool('get_testing_budget', state)
        if budget_info and 'error' not in budget_info:
            context_parts.append("\n## Budget Analysis")
            context_parts.append(f"- Max Testing Budget (4%): ${budget_info.get('max_testing_budget', 0):,.0f}")
            context_parts.append(f"- Cost per Video: ${budget_info.get('cost_per_video', 15000):,}")
            context_parts.append(f"- Can Test: {budget_info.get('max_videos', 0)} videos")
            context_parts.append(f"- Videos Uploaded: {budget_info.get('current_videos', 0)}")
            context_parts.append(f"- Projected Cost: ${budget_info.get('testing_cost', 0):,.0f}")
            
            status = budget_info.get('budget_status', '')
            if status == 'over_budget':
                context_parts.append(f"- ⚠️ **OVER BUDGET** - Need to cut {budget_info['current_videos'] - budget_info['max_videos']} videos")
            else:
                context_parts.append(f"- ✅ Within budget")
        
        # Videos ranked
        ranked_videos = execute_tool('get_videos_ranked_by_score', state)
        if ranked_videos:
            context_parts.append("\n## Videos by Pass Probability")
            for i, v in enumerate(ranked_videos, 1):
                prob = v.get('pass_probability', 0)
                status = '✅' if prob >= 0.7 else '⚠️' if prob >= 0.5 else '❌'
                matched = f" → {v['matched_line_item']}" if v.get('matched_line_item') else " (no match)"
                context_parts.append(f"{i}. {v['filename']}: {prob*100:.0f}% {status}{matched}")
        
        # Duplicates
        duplicates = execute_tool('get_duplicate_videos', state)
        if duplicates:
            context_parts.append("\n## ⚠️ Duplicates Detected")
            for d in duplicates:
                context_parts.append(f"- {d['video1']} ≈ {d['video2']} ({d['similarity']*100:.0f}% similar)")
        
        # CT Rules
        rules = execute_tool('get_ct_rules', state)
        if rules:
            context_parts.append("\n## CT Rules")
            context_parts.append(f"- Max testing: {rules.get('max_testing_budget_pct_display', '4%')} of media budget")
            context_parts.append(f"- Video cost: {rules.get('video_test_cost_display', '$15,000')}")
            context_parts.append(f"- Turnaround: {rules.get('standard_turnaround_display', '2-3 weeks')}")
        
        return "\n".join(context_parts) if context_parts else "No planning context available. Please upload a media plan and videos."


# Factory function
def create_planning_specialist() -> PlanningSpecialistAgent:
    """Create a Planning Specialist agent instance."""
    return PlanningSpecialistAgent()
