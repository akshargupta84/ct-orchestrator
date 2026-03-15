"""
Agent State - Shared state schema for the multi-agent system.

This module defines the state that flows through the LangGraph
and is shared across all agents.
"""

from typing import TypedDict, Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class QueryType(Enum):
    """Classification of user queries."""
    PLANNING = "planning"           # Test plan design, budget, prioritization
    ANALYSIS = "analysis"           # Video/creative analysis
    RESULTS = "results"             # Historical data, why pass/fail
    STRATEGY = "strategy"           # Recommendations, next steps
    MIXED = "mixed"                 # Requires multiple perspectives
    GENERAL = "general"             # General questions, chitchat


class AgentName(Enum):
    """Available agents in the system."""
    ORCHESTRATOR = "orchestrator"
    PLANNING = "planning"
    CREATIVE_ANALYST = "creative_analyst"
    RESULTS_INTERPRETER = "results_interpreter"
    STRATEGY = "strategy"


@dataclass
class Message:
    """A single message in the conversation."""
    role: str                       # "user" or "assistant" or agent name
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    agent_name: Optional[str] = None  # Which agent generated this (if assistant)
    
    def to_dict(self) -> dict:
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp,
            'agent_name': self.agent_name
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Message':
        return cls(
            role=data['role'],
            content=data['content'],
            timestamp=data.get('timestamp', datetime.now().isoformat()),
            agent_name=data.get('agent_name')
        )


@dataclass
class VideoInfo:
    """Information about an uploaded video."""
    filename: str
    filepath: str
    duration: float = 0
    pass_probability: float = 0
    risk_factors: List[str] = field(default_factory=list)
    matched_line_item: str = ""
    is_duplicate_of: str = ""
    features: Dict = field(default_factory=dict)
    diagnostics: Dict = field(default_factory=dict)  # Predicted diagnostic scores
    scored: bool = False
    
    def to_dict(self) -> dict:
        return {
            'filename': self.filename,
            'filepath': self.filepath,
            'duration': self.duration,
            'pass_probability': self.pass_probability,
            'risk_factors': self.risk_factors,
            'matched_line_item': self.matched_line_item,
            'is_duplicate_of': self.is_duplicate_of,
            'features': self.features,
            'diagnostics': self.diagnostics,
            'scored': self.scored
        }


@dataclass
class MediaPlanInfo:
    """Extracted information from a media plan."""
    brand: str = ""
    campaign_name: str = ""
    total_budget: float = 0
    testing_budget: float = 0      # Calculated: total_budget * 0.04
    flight_start: str = ""
    flight_end: str = ""
    markets: List[str] = field(default_factory=list)
    primary_kpi: str = ""
    creative_line_items: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'brand': self.brand,
            'campaign_name': self.campaign_name,
            'total_budget': self.total_budget,
            'testing_budget': self.testing_budget,
            'flight_start': self.flight_start,
            'flight_end': self.flight_end,
            'markets': self.markets,
            'primary_kpi': self.primary_kpi,
            'creative_line_items': self.creative_line_items
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MediaPlanInfo':
        return cls(**data)


@dataclass
class InterAgentRequest:
    """A request from one agent to another."""
    from_agent: str
    to_agent: str
    question: str
    answer: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            'from_agent': self.from_agent,
            'to_agent': self.to_agent,
            'question': self.question,
            'answer': self.answer,
            'timestamp': self.timestamp
        }


@dataclass 
class AgentResponse:
    """Response from a single agent."""
    agent_name: str
    content: str
    confidence: float = 1.0         # How confident the agent is (0-1)
    reasoning: str = ""             # Internal reasoning (for "show reasoning")
    requests_made: List[InterAgentRequest] = field(default_factory=list)
    error: Optional[str] = None
    tokens_used: int = 0
    
    def to_dict(self) -> dict:
        return {
            'agent_name': self.agent_name,
            'content': self.content,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'requests_made': [r.to_dict() for r in self.requests_made],
            'error': self.error,
            'tokens_used': self.tokens_used
        }


@dataclass
class ReasoningStep:
    """A single step in the reasoning trace for UI display."""
    agent_name: str
    action: str                     # "thinking", "responding", "requesting", "error"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            'agent_name': self.agent_name,
            'action': self.action,
            'content': self.content,
            'timestamp': self.timestamp
        }


class AgentState(TypedDict, total=False):
    """
    The shared state that flows through the LangGraph.
    
    This state is passed to each agent and updated as the
    conversation progresses.
    """
    
    # =========================================================================
    # Conversation History (persisted across turns)
    # =========================================================================
    messages: List[Dict]            # Full chat history as dicts
    current_query: str              # Latest user message
    
    # =========================================================================
    # Uploaded Files (persisted across turns)
    # =========================================================================
    media_plan_info: Optional[Dict]     # Parsed media plan (MediaPlanInfo.to_dict())
    videos: List[Dict]                  # Uploaded videos (VideoInfo.to_dict())
    duplicates_detected: List[tuple]    # [(video1, video2, similarity), ...]
    
    # =========================================================================
    # Query Classification & Routing (current turn)
    # =========================================================================
    query_classification: str           # QueryType value
    selected_agents: List[str]          # AgentName values to invoke
    _remaining_agents: List[str]        # Agents still to run (for LangGraph routing)
    
    # =========================================================================
    # Agent Responses (current turn)
    # =========================================================================
    agent_responses: Dict[str, Dict]    # {agent_name: AgentResponse.to_dict()}
    inter_agent_requests: List[Dict]    # InterAgentRequest.to_dict() list
    agent_errors: Dict[str, str]        # {agent_name: error_message}
    
    # =========================================================================
    # Final Output (current turn)
    # =========================================================================
    final_response: str                 # Synthesized response to user
    reasoning_trace: List[Dict]         # ReasoningStep.to_dict() list for UI
    
    # =========================================================================
    # Token Tracking
    # =========================================================================
    token_usage: Dict[str, int]         # {agent_name: tokens_used}
    total_tokens_this_turn: int


def create_initial_state() -> AgentState:
    """Create a fresh initial state."""
    return AgentState(
        messages=[],
        current_query="",
        media_plan_info=None,
        videos=[],
        duplicates_detected=[],
        query_classification="",
        selected_agents=[],
        _remaining_agents=[],
        agent_responses={},
        inter_agent_requests=[],
        agent_errors={},
        final_response="",
        reasoning_trace=[],
        token_usage={},
        total_tokens_this_turn=0
    )


def add_message(state: AgentState, role: str, content: str, agent_name: str = None) -> AgentState:
    """Add a message to the conversation history."""
    message = Message(role=role, content=content, agent_name=agent_name)
    state['messages'].append(message.to_dict())
    return state


def add_reasoning_step(state: AgentState, agent_name: str, action: str, content: str) -> AgentState:
    """Add a step to the reasoning trace."""
    step = ReasoningStep(agent_name=agent_name, action=action, content=content)
    state['reasoning_trace'].append(step.to_dict())
    return state


def get_conversation_history(state: AgentState, max_messages: int = 20) -> List[Dict]:
    """Get recent conversation history for context."""
    messages = state.get('messages', [])
    return messages[-max_messages:] if len(messages) > max_messages else messages


def get_video_by_filename(state: AgentState, filename: str) -> Optional[Dict]:
    """Find a video by filename."""
    for video in state.get('videos', []):
        if video.get('filename') == filename:
            return video
    return None


def update_video(state: AgentState, filename: str, updates: Dict) -> AgentState:
    """Update a video's information."""
    for i, video in enumerate(state.get('videos', [])):
        if video.get('filename') == filename:
            state['videos'][i].update(updates)
            break
    return state
