"""
Multi-Agent Graph - LangGraph implementation for agent orchestration.

This module wires up the agents using LangGraph for:
- State management across agents
- Conditional routing based on query classification
- Parallel agent execution when possible
- Error handling and recovery
"""

from typing import Dict, List, Any, Literal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import langgraph
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None

from agents.state import (
    AgentState, AgentResponse, QueryType, AgentName,
    add_reasoning_step, add_message, create_initial_state
)
from agents.orchestrator import Orchestrator
from agents.planning_specialist import create_planning_specialist
from agents.creative_analyst import create_creative_analyst
from agents.results_interpreter import create_results_interpreter
from agents.strategy_agent import create_strategy_agent
from utils.llm import get_completion


class MultiAgentSystem:
    """
    Multi-agent system using LangGraph for orchestration.
    
    If LangGraph is not available, falls back to simple sequential execution.
    """
    
    def __init__(self):
        self.orchestrator = Orchestrator()
        
        # Create and register agents
        self.planning = create_planning_specialist()
        self.creative = create_creative_analyst()
        self.results = create_results_interpreter()
        self.strategy = create_strategy_agent()
        
        self.orchestrator.register_agent(self.planning)
        self.orchestrator.register_agent(self.creative)
        self.orchestrator.register_agent(self.results)
        self.orchestrator.register_agent(self.strategy)
        
        # Build graph if LangGraph available
        if LANGGRAPH_AVAILABLE:
            self.graph = self._build_graph()
            self.compiled_graph = self.graph.compile()
        else:
            self.graph = None
            self.compiled_graph = None
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""
        
        # Define the graph with AgentState
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("classify", self._classify_node)
        graph.add_node("route", self._route_node)
        graph.add_node("planning", self._planning_node)
        graph.add_node("creative_analyst", self._creative_analyst_node)
        graph.add_node("results_interpreter", self._results_interpreter_node)
        graph.add_node("strategy", self._strategy_node)
        graph.add_node("synthesize", self._synthesize_node)
        
        # Set entry point
        graph.set_entry_point("classify")
        
        # Add edges
        graph.add_edge("classify", "route")
        
        # Conditional routing based on selected agents
        graph.add_conditional_edges(
            "route",
            self._determine_next_agents,
            {
                "planning": "planning",
                "creative_analyst": "creative_analyst",
                "results_interpreter": "results_interpreter",
                "strategy": "strategy",
                "synthesize": "synthesize"  # No agents selected
            }
        )
        
        # After each agent, check if more agents need to run
        for agent in ["planning", "creative_analyst", "results_interpreter", "strategy"]:
            graph.add_conditional_edges(
                agent,
                self._check_remaining_agents,
                {
                    "planning": "planning",
                    "creative_analyst": "creative_analyst",
                    "results_interpreter": "results_interpreter",
                    "strategy": "strategy",
                    "synthesize": "synthesize"
                }
            )
        
        # Synthesize goes to END
        graph.add_edge("synthesize", END)
        
        return graph
    
    def _classify_node(self, state: AgentState) -> AgentState:
        """Classify the query type."""
        query = state.get('current_query', '')
        
        # Reset per-turn state
        state['agent_responses'] = {}
        state['agent_errors'] = {}
        state['inter_agent_requests'] = []
        state['reasoning_trace'] = []
        state['token_usage'] = {}
        state['total_tokens_this_turn'] = 0
        
        add_reasoning_step(state, "orchestrator", "thinking", f"Analyzing query: {query[:50]}...")
        
        query_type = self.orchestrator._classify_query(query, state)
        state['query_classification'] = query_type
        
        add_reasoning_step(state, "orchestrator", "thinking", f"Query type: {query_type}")
        
        return state
    
    def _route_node(self, state: AgentState) -> AgentState:
        """Select which agents to invoke."""
        query = state.get('current_query', '')
        query_type = state.get('query_classification', '')
        
        selected = self.orchestrator._select_agents(query_type, query, state)
        state['selected_agents'] = selected
        state['_remaining_agents'] = list(selected)
        
        if selected:
            add_reasoning_step(
                state, "orchestrator", "routing",
                f"Selected agents: {', '.join(selected)}"
            )
        else:
            add_reasoning_step(state, "orchestrator", "thinking", "Handling directly")
        
        return state
    
    def _determine_next_agents(self, state: AgentState) -> str:
        """Determine which agent to run first."""
        remaining = state.get('_remaining_agents', [])
        return "synthesize" if not remaining else remaining[0]
    
    def _check_remaining_agents(self, state: AgentState) -> str:
        """Check if more agents need to run."""
        remaining = state.get('_remaining_agents', [])
        return "synthesize" if not remaining else remaining[0]
    
    def _run_agent_node(self, state: AgentState, agent_name: str) -> AgentState:
        """Generic agent execution node."""
        remaining = state.get('_remaining_agents', [])
        
        # Remove this agent from remaining
        if agent_name in remaining:
            remaining.remove(agent_name)
            state['_remaining_agents'] = remaining
        
        # Get the agent
        agent = self.orchestrator.agents.get(agent_name)
        if not agent:
            state['agent_errors'][agent_name] = f"Agent {agent_name} not found"
            return state
        
        try:
            response = agent.run(state)
            state['agent_responses'][agent_name] = response.to_dict()
            state['token_usage'][agent_name] = response.tokens_used
            state['total_tokens_this_turn'] += response.tokens_used

            # base_agent.run() catches its own exceptions and returns a response
            # with empty content + error set. Surface that into agent_errors so
            # synthesis sees the failure instead of treating empty content as
            # a valid answer.
            if response.error:
                state['agent_errors'][agent_name] = response.error

            # Handle inter-agent requests
            for request in response.requests_made:
                self.orchestrator._handle_inter_agent_request(request, state)

        except Exception as e:
            state['agent_errors'][agent_name] = str(e)
            add_reasoning_step(state, agent_name, "error", str(e))
        
        return state
    
    def _planning_node(self, state: AgentState) -> AgentState:
        return self._run_agent_node(state, "planning")
    
    def _creative_analyst_node(self, state: AgentState) -> AgentState:
        return self._run_agent_node(state, "creative_analyst")
    
    def _results_interpreter_node(self, state: AgentState) -> AgentState:
        return self._run_agent_node(state, "results_interpreter")
    
    def _strategy_node(self, state: AgentState) -> AgentState:
        return self._run_agent_node(state, "strategy")
    
    def _synthesize_node(self, state: AgentState) -> AgentState:
        """Synthesize final response."""
        import logging
        log = logging.getLogger(__name__)

        try:
            final_response = self.orchestrator._synthesize_response(state)
        except Exception as e:
            log.exception("Synthesize failed")
            final_response = ""
            state['agent_errors']['orchestrator'] = f"Synthesis failed: {e}"

        # Diagnostic: log what came out of the graph so empty/blank responses
        # aren't invisible to operators running `streamlit run` in a terminal.
        selected = state.get('selected_agents', [])
        responses = state.get('agent_responses', {})
        errors = state.get('agent_errors', {})
        log.info(
            "Synthesis outcome: selected=%s responses=%s errors=%s final_response_len=%s",
            selected, list(responses.keys()), list(errors.keys()), len(final_response or ""),
        )

        # Fallback: if synthesis produced nothing usable, surface whatever
        # agent errors or reasoning exist so the user sees *something* in the
        # chat bubble instead of a silent blank message.
        if not final_response or not final_response.strip():
            parts = ["⚠️ The agents didn't produce a response."]
            if errors:
                parts.append("\n**Agent errors:**")
                for agent_name, err in errors.items():
                    parts.append(f"- `{agent_name}`: {err}")
            if selected and not responses and not errors:
                parts.append(
                    f"\nSelected agents ({', '.join(selected)}) returned nothing. "
                    "This usually means the underlying LLM call returned an empty "
                    "response — check your `ANTHROPIC_API_KEY`, model ID, and quota."
                )
            if not selected and not responses and not errors:
                parts.append(
                    "\nThe orchestrator handled this as a general query but the "
                    "LLM returned an empty completion. Check the terminal for tracebacks "
                    "and verify `ANTHROPIC_API_KEY` / model ID in `utils/llm.py`."
                )
            final_response = "\n".join(parts)

        state['final_response'] = final_response

        # Add to conversation history
        add_message(state, 'assistant', final_response, 'orchestrator')

        # Track orchestrator tokens
        state['token_usage']['orchestrator'] = len(final_response) // 4
        state['total_tokens_this_turn'] += state['token_usage']['orchestrator']

        add_reasoning_step(state, "orchestrator", "responding", "Synthesized final response")

        return state
    
    def process(self, state: AgentState) -> AgentState:
        """
        Process a query through the multi-agent system.
        
        Args:
            state: AgentState with current_query set
            
        Returns:
            Updated AgentState with final_response
        """
        if self.compiled_graph:
            return self.compiled_graph.invoke(state)
        else:
            return self.orchestrator.process_query(state)
    
    def chat(self, query: str, state: AgentState = None) -> tuple[str, AgentState]:
        """
        Convenience method to chat with the system.
        
        Args:
            query: User's query
            state: Existing state (or None to create new)
            
        Returns:
            Tuple of (response, updated_state)
        """
        if state is None:
            state = create_initial_state()
        
        # Add user message
        add_message(state, 'user', query)
        state['current_query'] = query
        
        # Process
        state = self.process(state)
        
        return state.get('final_response', ''), state


# Singleton instance
_multi_agent_system = None


def get_multi_agent_system() -> MultiAgentSystem:
    """Get or create the multi-agent system instance."""
    global _multi_agent_system
    if _multi_agent_system is None:
        _multi_agent_system = MultiAgentSystem()
    return _multi_agent_system


def reset_multi_agent_system() -> MultiAgentSystem:
    """Reset and return a fresh multi-agent system."""
    global _multi_agent_system
    _multi_agent_system = MultiAgentSystem()
    return _multi_agent_system
