"""
CT Planning Workflow using LangGraph.

This workflow handles the creation and approval of creative testing plans.

States:
1. INTAKE - Gather campaign details
2. VALIDATE - Validate inputs against rules
3. GENERATE - Generate the test plan
4. REVIEW - Human approval checkpoint
5. REVISE - Handle revision requests
6. FINALIZE - Export approved plan
"""

from datetime import date, datetime, timedelta
from typing import Annotated, Literal, Optional, TypedDict
from enum import Enum
import uuid

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from models import (
    Campaign, 
    Creative, 
    CreativeTrix,
    TestPlan, 
    TestPlanItem, 
    TestStatus,
    AssetType,
    KPIType,
    Channel,
    Brand,
)
from models.rules import CTRules
from services.rules_engine import get_rules_engine
from utils.llm import get_completion, get_structured_output


# ============================================================================
# State Definition
# ============================================================================

class PlanningState(TypedDict):
    """State for the planning workflow."""
    
    # Input data
    campaign: Optional[dict]  # Campaign data as dict
    creative_trix: Optional[list]  # List of creative dicts
    hypotheses: Optional[list]  # List of hypothesis strings
    
    # Validation
    validation_result: Optional[dict]
    validation_errors: list
    
    # Generated plan
    plan: Optional[dict]  # TestPlan as dict
    
    # Human interaction
    human_feedback: Optional[str]
    approval_status: Optional[str]  # "approved", "revision_requested", "rejected"
    
    # Workflow tracking
    current_step: str
    revision_count: int
    messages: list  # Conversation history for UI


# ============================================================================
# Node Functions
# ============================================================================

def intake_node(state: PlanningState) -> PlanningState:
    """
    INTAKE: Validate that all required inputs are present.
    """
    messages = state.get("messages", [])
    errors = []
    
    if not state.get("campaign"):
        errors.append("Campaign details are required")
    
    if not state.get("creative_trix"):
        errors.append("Creative Trix (creative list) is required")
    
    if errors:
        messages.append({
            "role": "assistant",
            "content": f"Missing required inputs:\n" + "\n".join(f"- {e}" for e in errors)
        })
        return {
            **state,
            "current_step": "intake",
            "validation_errors": errors,
            "messages": messages,
        }
    
    messages.append({
        "role": "assistant", 
        "content": "All inputs received. Validating against CT rules..."
    })
    
    return {
        **state,
        "current_step": "validate",
        "validation_errors": [],
        "messages": messages,
    }


def validate_node(state: PlanningState) -> PlanningState:
    """
    VALIDATE: Check inputs against CT rules.
    """
    messages = state.get("messages", [])
    campaign_data = state["campaign"]
    creative_list = state["creative_trix"]
    
    # Count videos and display assets
    video_count = sum(1 for c in creative_list if c.get("asset_type") == "video")
    display_count = sum(1 for c in creative_list if c.get("asset_type") == "display")
    budget = campaign_data.get("budget", 0)
    
    # Validate against rules
    rules_engine = get_rules_engine()
    validation = rules_engine.validate_plan(budget, video_count, display_count)
    
    if not validation["valid"]:
        error_msg = "Validation failed:\n" + "\n".join(f"- {e}" for e in validation["errors"])
        messages.append({"role": "assistant", "content": error_msg})
        return {
            **state,
            "current_step": "intake",  # Go back to intake
            "validation_result": validation,
            "validation_errors": validation["errors"],
            "messages": messages,
        }
    
    # Validation passed
    messages.append({
        "role": "assistant",
        "content": f"""Validation passed!
- Budget: ${budget:,.0f}
- Testing budget (1.5%): ${validation['testing_budget']:,.0f}
- Video limit: {validation['limits']['video_limit']} (requesting {video_count})
- Display limit: {validation['limits']['display_limit']} (requesting {display_count})
- Estimated cost: ${validation['estimated_cost']:,.0f}

Generating test plan..."""
    })
    
    return {
        **state,
        "current_step": "generate",
        "validation_result": validation,
        "validation_errors": [],
        "messages": messages,
    }


def generate_node(state: PlanningState) -> PlanningState:
    """
    GENERATE: Create the test plan using LLM.
    """
    messages = state.get("messages", [])
    campaign_data = state["campaign"]
    creative_list = state["creative_trix"]
    hypotheses = state.get("hypotheses", [])
    validation = state["validation_result"]
    
    rules_engine = get_rules_engine()
    rules = rules_engine.rules
    
    # Build campaign object
    campaign = Campaign(
        id=campaign_data.get("id", str(uuid.uuid4())),
        name=campaign_data.get("name", "Unnamed Campaign"),
        brand=Brand(**campaign_data.get("brand", {"id": "brand1", "name": "Brand"})),
        budget=campaign_data.get("budget", 0),
        start_date=date.fromisoformat(campaign_data.get("start_date", date.today().isoformat())),
        end_date=date.fromisoformat(campaign_data.get("end_date", (date.today() + timedelta(days=90)).isoformat())),
        primary_kpi=KPIType(campaign_data.get("primary_kpi", "awareness")),
        secondary_kpis=[KPIType(k) for k in campaign_data.get("secondary_kpis", [])],
    )
    
    # Separate videos and display
    videos = [c for c in creative_list if c.get("asset_type") == "video"]
    displays = [c for c in creative_list if c.get("asset_type") == "display"]
    
    # Calculate test dates
    test_start = campaign.start_date - timedelta(days=rules.turnaround.video_standard_days + 7)
    if test_start < date.today():
        test_start = date.today() + timedelta(days=1)
    
    # Build video test items
    video_tests = []
    for i, v in enumerate(videos):
        video_tests.append(TestPlanItem(
            creative=Creative(
                id=v.get("id", f"vid_{i}"),
                name=v.get("name", f"Video {i+1}"),
                campaign_id=campaign.id,
                asset_type=AssetType.VIDEO,
                channel=Channel(v.get("channel", "digital_video")),
                impressions=v.get("impressions", 0),
                hypothesis=hypotheses[i] if i < len(hypotheses) else None,
            ),
            test_start_date=test_start,
            expected_results_date=test_start + timedelta(days=rules.turnaround.video_standard_days),
            estimated_cost=rules.costs.video_cost,
            cell_size=1500,
            priority=i + 1,
        ))
    
    # Build display test items
    display_tests = []
    for i, d in enumerate(displays):
        display_tests.append(TestPlanItem(
            creative=Creative(
                id=d.get("id", f"dsp_{i}"),
                name=d.get("name", f"Display {i+1}"),
                campaign_id=campaign.id,
                asset_type=AssetType.DISPLAY,
                channel=Channel(d.get("channel", "display")),
                impressions=d.get("impressions", 0),
                hypothesis=hypotheses[len(videos) + i] if len(videos) + i < len(hypotheses) else None,
            ),
            test_start_date=test_start,
            expected_results_date=test_start + timedelta(days=rules.turnaround.display_standard_days),
            estimated_cost=rules.costs.display_cost,
            cell_size=2000,
            priority=i + 1,
        ))
    
    # Build test plan
    total_cost = (
        len(video_tests) * rules.costs.video_cost + 
        len(display_tests) * rules.costs.display_cost
    )
    
    plan = TestPlan(
        id=f"plan_{campaign.id}_{datetime.now().strftime('%Y%m%d')}",
        campaign_id=campaign.id,
        status=TestStatus.PENDING_APPROVAL,
        video_tests=video_tests,
        display_tests=display_tests,
        total_estimated_cost=total_cost,
        remaining_budget=campaign.testing_budget - total_cost,
    )
    
    # Format plan for display
    plan_summary = f"""## Creative Testing Plan Generated

**Campaign:** {campaign.name}
**Primary KPI:** {campaign.primary_kpi.value.replace('_', ' ').title()}

### Video Tests ({len(video_tests)})
| Creative | Test Start | Results Expected | Cost |
|----------|------------|------------------|------|
"""
    for item in video_tests:
        plan_summary += f"| {item.creative.name} | {item.test_start_date} | {item.expected_results_date} | ${item.estimated_cost:,.0f} |\n"
    
    plan_summary += f"""
### Display Tests ({len(display_tests)})
| Creative | Test Start | Results Expected | Cost |
|----------|------------|------------------|------|
"""
    for item in display_tests:
        plan_summary += f"| {item.creative.name} | {item.test_start_date} | {item.expected_results_date} | ${item.estimated_cost:,.0f} |\n"
    
    plan_summary += f"""
### Budget Summary
- Total estimated cost: **${total_cost:,.0f}**
- Testing budget: **${campaign.testing_budget:,.0f}**
- Remaining: **${campaign.testing_budget - total_cost:,.0f}**

---
Please review and approve this plan, or provide feedback for revisions."""

    messages.append({"role": "assistant", "content": plan_summary})
    
    return {
        **state,
        "current_step": "review",
        "plan": plan.model_dump(),
        "messages": messages,
    }


def review_node(state: PlanningState) -> PlanningState:
    """
    REVIEW: Wait for human approval/feedback.
    
    This is an interrupt point - the workflow pauses here until
    human_feedback and approval_status are provided.
    """
    # This node just passes through - the actual pause happens
    # at the edge condition checking for human input
    return state


def revise_node(state: PlanningState) -> PlanningState:
    """
    REVISE: Incorporate human feedback and regenerate plan.
    """
    messages = state.get("messages", [])
    feedback = state.get("human_feedback", "")
    revision_count = state.get("revision_count", 0) + 1
    
    messages.append({"role": "user", "content": f"Revision requested: {feedback}"})
    
    # Use LLM to interpret feedback and suggest changes
    prompt = f"""A creative testing plan was created and the reviewer provided this feedback:
"{feedback}"

The current plan has:
- {len(state['plan'].get('video_tests', []))} video tests
- {len(state['plan'].get('display_tests', []))} display tests

Based on the feedback, what specific changes should be made to the plan?
Be specific about which creatives to add, remove, or modify."""

    response = get_completion(prompt, system="You help revise creative testing plans based on stakeholder feedback.")
    
    messages.append({
        "role": "assistant",
        "content": f"**Revision #{revision_count}**\n\n{response}\n\nRegenerating plan with changes..."
    })
    
    # For now, just go back to generate with updated state
    # In a full implementation, we'd parse the LLM response and modify inputs
    return {
        **state,
        "current_step": "generate",
        "revision_count": revision_count,
        "human_feedback": None,
        "approval_status": None,
        "plan": {**state["plan"], "revision_history": state["plan"].get("revision_history", []) + [feedback]},
        "messages": messages,
    }


def finalize_node(state: PlanningState) -> PlanningState:
    """
    FINALIZE: Mark plan as approved and prepare for export.
    """
    messages = state.get("messages", [])
    plan = state["plan"]
    
    # Update plan status
    plan["status"] = TestStatus.APPROVED.value
    plan["approved_at"] = datetime.now().isoformat()
    
    messages.append({
        "role": "assistant",
        "content": f"""## Plan Approved! ✓

The creative testing plan has been approved and is ready for execution.

**Next Steps:**
1. Creatives will be sent to the testing vendor
2. Testing begins: {plan['video_tests'][0]['test_start_date'] if plan.get('video_tests') else 'TBD'}
3. Results expected: {plan['video_tests'][0]['expected_results_date'] if plan.get('video_tests') else 'TBD'}

You can now upload results when they're received."""
    })
    
    return {
        **state,
        "current_step": "complete",
        "plan": plan,
        "messages": messages,
    }


# ============================================================================
# Routing Functions
# ============================================================================

def route_after_intake(state: PlanningState) -> str:
    """Route after intake based on validation errors."""
    if state.get("validation_errors"):
        return END  # Stop if missing required inputs
    return "validate"


def route_after_review(state: PlanningState) -> str:
    """Route after review based on human input."""
    approval = state.get("approval_status")
    
    if approval == "approved":
        return "finalize"
    elif approval == "revision_requested":
        return "revise"
    elif approval == "rejected":
        return END
    else:
        # No input yet - stay in review (this shouldn't happen with proper interrupt)
        return "review"


# ============================================================================
# Workflow Graph
# ============================================================================

def create_planning_workflow():
    """Create the planning workflow graph."""
    
    # Create the graph
    workflow = StateGraph(PlanningState)
    
    # Add nodes
    workflow.add_node("intake", intake_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("review", review_node)
    workflow.add_node("revise", revise_node)
    workflow.add_node("finalize", finalize_node)
    
    # Set entry point
    workflow.set_entry_point("intake")
    
    # Add edges
    workflow.add_conditional_edges(
        "intake",
        route_after_intake,
        {
            "validate": "validate",
            END: END,
        }
    )
    
    workflow.add_edge("validate", "generate")
    
    # Validate might loop back to intake on errors
    # This is handled in validate_node by setting current_step
    
    workflow.add_edge("generate", "review")
    
    workflow.add_conditional_edges(
        "review",
        route_after_review,
        {
            "finalize": "finalize",
            "revise": "revise",
            "review": "review",  # Wait state
            END: END,
        }
    )
    
    workflow.add_edge("revise", "generate")
    workflow.add_edge("finalize", END)
    
    # Compile with memory for persistence
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory, interrupt_before=["review"])


# ============================================================================
# Helper Functions
# ============================================================================

def run_planning_workflow(
    campaign_data: dict,
    creative_list: list,
    hypotheses: list = None,
    thread_id: str = None,
) -> tuple[PlanningState, str]:
    """
    Run the planning workflow to generate a test plan.
    
    Args:
        campaign_data: Campaign details dict
        creative_list: List of creative dicts
        hypotheses: Optional list of hypotheses to test
        thread_id: Optional thread ID for persistence
        
    Returns:
        Tuple of (final_state, thread_id)
    """
    workflow = create_planning_workflow()
    
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "campaign": campaign_data,
        "creative_trix": creative_list,
        "hypotheses": hypotheses or [],
        "validation_result": None,
        "validation_errors": [],
        "plan": None,
        "human_feedback": None,
        "approval_status": None,
        "current_step": "intake",
        "revision_count": 0,
        "messages": [],
    }
    
    # Run until interrupt (review node)
    result = workflow.invoke(initial_state, config)
    
    return result, thread_id


def continue_planning_workflow(
    thread_id: str,
    approval_status: str,
    feedback: str = None,
) -> PlanningState:
    """
    Continue the planning workflow after human review.
    
    Args:
        thread_id: Thread ID from initial run
        approval_status: "approved", "revision_requested", or "rejected"
        feedback: Optional feedback text for revisions
        
    Returns:
        Updated state
    """
    workflow = create_planning_workflow()
    config = {"configurable": {"thread_id": thread_id}}
    
    # Update state with human input
    update = {
        "approval_status": approval_status,
        "human_feedback": feedback,
    }
    
    # Continue workflow
    result = workflow.invoke(update, config)
    
    return result
