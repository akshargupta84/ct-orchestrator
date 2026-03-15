"""
Agents module - Multi-agent system for creative testing.

This module contains:
- state: Shared state definitions (AgentState, etc.)
- tools: Tool functions that agents can use
- base_agent: Abstract base class for all agents
- orchestrator: Routes queries and synthesizes responses
- planning_specialist: Test plan design specialist
- creative_analyst: Video analysis specialist  
- results_interpreter: Historical data specialist
- strategy_agent: Recommendations specialist
"""

from agents.state import (
    AgentState,
    QueryType,
    AgentName,
    Message,
    VideoInfo,
    MediaPlanInfo,
    InterAgentRequest,
    AgentResponse,
    ReasoningStep,
    create_initial_state,
    add_message,
    add_reasoning_step,
    get_conversation_history
)

from agents.tools import (
    TOOL_REGISTRY,
    get_tools_for_agent,
    execute_tool
)

from agents.base_agent import (
    BaseAgent,
    AgentConfig
)

from agents.orchestrator import (
    Orchestrator,
    get_orchestrator,
    reset_orchestrator
)

from agents.planning_specialist import (
    PlanningSpecialistAgent,
    create_planning_specialist
)

from agents.creative_analyst import (
    CreativeAnalystAgent,
    create_creative_analyst
)

from agents.results_interpreter import (
    ResultsInterpreterAgent,
    create_results_interpreter
)

from agents.strategy_agent import (
    StrategyAgent,
    create_strategy_agent
)

# Legacy imports (optional)
try:
    from agents.analysis_agent import AnalysisAgent
except ImportError:
    AnalysisAgent = None

try:
    from agents.planning_agent import PlanningAgent, get_planning_agent
except ImportError:
    PlanningAgent = None
    get_planning_agent = None


__all__ = [
    # State
    'AgentState',
    'QueryType', 
    'AgentName',
    'Message',
    'VideoInfo',
    'MediaPlanInfo',
    'InterAgentRequest',
    'AgentResponse',
    'ReasoningStep',
    'create_initial_state',
    'add_message',
    'add_reasoning_step',
    'get_conversation_history',
    
    # Tools
    'TOOL_REGISTRY',
    'get_tools_for_agent',
    'execute_tool',
    
    # Base Agent
    'BaseAgent',
    'AgentConfig',
    
    # Orchestrator
    'Orchestrator',
    'get_orchestrator',
    'reset_orchestrator',
    
    # Specialist Agents
    'PlanningSpecialistAgent',
    'create_planning_specialist',
    'CreativeAnalystAgent',
    'create_creative_analyst',
    'ResultsInterpreterAgent',
    'create_results_interpreter',
    'StrategyAgent',
    'create_strategy_agent',
    
    # Legacy
    'AnalysisAgent',
    'PlanningAgent',
    'get_planning_agent'
]
