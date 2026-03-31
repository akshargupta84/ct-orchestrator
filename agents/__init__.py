"""
Agents module - Multi-agent system for creative testing.

All imports are wrapped in try/except so the app works even when
optional dependencies (langgraph, langchain, etc.) are not installed.
"""

# State
try:
    from agents.state import (
        AgentState, QueryType, AgentName, Message, VideoInfo,
        MediaPlanInfo, InterAgentRequest, AgentResponse, ReasoningStep,
        create_initial_state, add_message, add_reasoning_step, get_conversation_history
    )
except ImportError:
    AgentState = None
    QueryType = None
    AgentName = None
    Message = None
    VideoInfo = None
    MediaPlanInfo = None
    InterAgentRequest = None
    AgentResponse = None
    ReasoningStep = None
    create_initial_state = None
    add_message = None
    add_reasoning_step = None
    get_conversation_history = None

# Tools
try:
    from agents.tools import TOOL_REGISTRY, get_tools_for_agent, execute_tool
except ImportError:
    TOOL_REGISTRY = None
    get_tools_for_agent = None
    execute_tool = None

# Base Agent
try:
    from agents.base_agent import BaseAgent, AgentConfig
except ImportError:
    BaseAgent = None
    AgentConfig = None

# Orchestrator
try:
    from agents.orchestrator import Orchestrator, get_orchestrator, reset_orchestrator
except ImportError:
    Orchestrator = None
    get_orchestrator = None
    reset_orchestrator = None

# Specialist Agents
try:
    from agents.planning_specialist import PlanningSpecialistAgent, create_planning_specialist
except ImportError:
    PlanningSpecialistAgent = None
    create_planning_specialist = None

try:
    from agents.creative_analyst import CreativeAnalystAgent, create_creative_analyst
except ImportError:
    CreativeAnalystAgent = None
    create_creative_analyst = None

try:
    from agents.results_interpreter import ResultsInterpreterAgent, create_results_interpreter
except ImportError:
    ResultsInterpreterAgent = None
    create_results_interpreter = None

try:
    from agents.strategy_agent import StrategyAgent, create_strategy_agent
except ImportError:
    StrategyAgent = None
    create_strategy_agent = None

# Legacy
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
    'AgentState', 'QueryType', 'AgentName', 'Message', 'VideoInfo',
    'MediaPlanInfo', 'InterAgentRequest', 'AgentResponse', 'ReasoningStep',
    'create_initial_state', 'add_message', 'add_reasoning_step', 'get_conversation_history',
    'TOOL_REGISTRY', 'get_tools_for_agent', 'execute_tool',
    'BaseAgent', 'AgentConfig',
    'Orchestrator', 'get_orchestrator', 'reset_orchestrator',
    'PlanningSpecialistAgent', 'create_planning_specialist',
    'CreativeAnalystAgent', 'create_creative_analyst',
    'ResultsInterpreterAgent', 'create_results_interpreter',
    'StrategyAgent', 'create_strategy_agent',
    'AnalysisAgent', 'PlanningAgent', 'get_planning_agent',
]
