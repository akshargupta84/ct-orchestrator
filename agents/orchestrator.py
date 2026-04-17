"""
Orchestrator Agent - Routes queries to specialists and synthesizes responses.

The Orchestrator is the central controller of the multi-agent system:
1. Classifies incoming queries by type
2. Selects which specialist agents to invoke (1-3)
3. Routes inter-agent requests
4. Synthesizes final response from multiple agent outputs
5. Handles errors gracefully (skips failed agents)
"""

from typing import Dict, List, Optional, Tuple
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.state import (
    AgentState, AgentResponse, QueryType, AgentName,
    add_reasoning_step, add_message, ReasoningStep
)
from agents.base_agent import BaseAgent, AgentConfig
from utils.llm import get_completion


# Agent selection based on query type
QUERY_TYPE_TO_AGENTS = {
    QueryType.PLANNING.value: [AgentName.PLANNING.value],
    QueryType.ANALYSIS.value: [AgentName.CREATIVE_ANALYST.value],
    QueryType.RESULTS.value: [AgentName.RESULTS_INTERPRETER.value],
    QueryType.STRATEGY.value: [AgentName.STRATEGY.value],
    QueryType.MIXED.value: [AgentName.CREATIVE_ANALYST.value, AgentName.RESULTS_INTERPRETER.value],
    QueryType.GENERAL.value: []  # Orchestrator handles directly
}

# Keywords for query classification
CLASSIFICATION_KEYWORDS = {
    QueryType.PLANNING.value: [
        'plan', 'test', 'budget', 'prioritize', 'which creative', 'how many',
        'afford', 'cost', 'duplicate', 'select', 'choose', 'schedule'
    ],
    QueryType.ANALYSIS.value: [
        'analyze', 'score', 'why low', 'why high', 'features', 'improve',
        'what\'s wrong', 'creative', 'video', 'risk', 'strength', 'weakness'
    ],
    QueryType.RESULTS.value: [
        'why pass', 'why fail', 'historical', 'pattern', 'similar', 'data',
        'past', 'previous', 'trend', 'average', 'compare'
    ],
    QueryType.STRATEGY.value: [
        'recommend', 'should i', 'next', 'strategy', 'advice', 'suggest',
        'what to do', 'optimize', 'improve program', 'learn'
    ]
}


class Orchestrator:
    """
    Central orchestrator for the multi-agent system.
    
    Responsibilities:
    - Classify queries
    - Route to appropriate agents
    - Handle inter-agent communication
    - Synthesize final responses
    - Track token usage
    """
    
    def __init__(self):
        self.name = AgentName.ORCHESTRATOR.value
        self.display_name = "Orchestrator"
        self.agents: Dict[str, BaseAgent] = {}
    
    def register_agent(self, agent: BaseAgent):
        """Register a specialist agent."""
        self.agents[agent.name] = agent
    
    def process_query(self, state: AgentState) -> AgentState:
        """
        Main entry point - process a user query through the multi-agent system.
        
        Args:
            state: Current shared state with current_query set
            
        Returns:
            Updated state with final_response and reasoning_trace
        """
        # Reset per-turn state
        state['agent_responses'] = {}
        state['agent_errors'] = {}
        state['inter_agent_requests'] = []
        state['reasoning_trace'] = []
        state['token_usage'] = {}
        state['total_tokens_this_turn'] = 0
        
        query = state.get('current_query', '')
        
        # Step 1: Classify the query
        add_reasoning_step(state, self.name, "thinking", f"Classifying query: {query[:100]}...")
        query_type = self._classify_query(query, state)
        state['query_classification'] = query_type
        add_reasoning_step(state, self.name, "thinking", f"Query type: {query_type}")
        
        # Step 2: Select agents to invoke
        selected_agents = self._select_agents(query_type, query, state)
        state['selected_agents'] = selected_agents
        
        if selected_agents:
            add_reasoning_step(
                state, self.name, "routing", 
                f"Routing to: {', '.join(selected_agents)}"
            )
        else:
            add_reasoning_step(state, self.name, "thinking", "Handling directly (general query)")
        
        # Step 3: Run selected agents
        for agent_name in selected_agents:
            if agent_name in self.agents:
                try:
                    agent = self.agents[agent_name]
                    response = agent.run(state)
                    state['agent_responses'][agent_name] = response.to_dict()
                    state['token_usage'][agent_name] = response.tokens_used
                    state['total_tokens_this_turn'] += response.tokens_used

                    # Surface agent-internal errors (base_agent.run() catches
                    # exceptions and returns empty content + error field).
                    if response.error:
                        state['agent_errors'][agent_name] = response.error

                    # Handle inter-agent requests
                    for request in response.requests_made:
                        self._handle_inter_agent_request(request, state)

                except Exception as e:
                    state['agent_errors'][agent_name] = str(e)
                    add_reasoning_step(state, agent_name, "error", f"Failed: {str(e)}")
        
        # Step 4: Synthesize final response
        import logging
        log = logging.getLogger(__name__)
        try:
            final_response = self._synthesize_response(state)
        except Exception as e:
            log.exception("Synthesize failed")
            final_response = ""
            state['agent_errors']['orchestrator'] = f"Synthesis failed: {e}"

        log.info(
            "Synthesis outcome: selected=%s responses=%s errors=%s final_response_len=%s",
            state.get('selected_agents', []),
            list(state.get('agent_responses', {}).keys()),
            list(state.get('agent_errors', {}).keys()),
            len(final_response or ""),
        )

        # Fallback so the assistant bubble is never a silent blank when every
        # agent errored or the LLM returned empty.
        if not final_response or not final_response.strip():
            errors = state.get('agent_errors', {})
            selected = state.get('selected_agents', [])
            parts = ["⚠️ The agents didn't produce a response."]
            if errors:
                parts.append("\n**Agent errors:**")
                for agent_name, err in errors.items():
                    parts.append(f"- `{agent_name}`: {err}")
            if selected and not state.get('agent_responses') and not errors:
                parts.append(
                    f"\nSelected agents ({', '.join(selected)}) returned nothing. "
                    "Check `ANTHROPIC_API_KEY`, the model ID in `utils/llm.py`, and quota."
                )
            if not selected and not state.get('agent_responses') and not errors:
                parts.append(
                    "\nThe orchestrator handled this as a general query but the LLM "
                    "returned an empty completion. Check the terminal for tracebacks."
                )
            final_response = "\n".join(parts)

        state['final_response'] = final_response

        # Add to conversation history
        add_message(state, 'assistant', final_response, self.name)

        # Track orchestrator tokens
        state['token_usage'][self.name] = self._estimate_tokens(final_response)
        state['total_tokens_this_turn'] += state['token_usage'][self.name]

        return state
    
    def _classify_query(self, query: str, state: AgentState) -> str:
        """
        Classify the query type using keywords and context.
        
        Args:
            query: User's query
            state: Current state (for context)
            
        Returns:
            QueryType value string
        """
        query_lower = query.lower()
        
        # Score each type based on keyword matches
        scores = {qtype: 0 for qtype in CLASSIFICATION_KEYWORDS}
        
        for qtype, keywords in CLASSIFICATION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    scores[qtype] += 1
        
        # Check for multiple high scores (mixed query)
        high_scores = [qtype for qtype, score in scores.items() if score >= 2]
        if len(high_scores) > 1:
            return QueryType.MIXED.value
        
        # Return highest scoring type
        max_score = max(scores.values())
        if max_score > 0:
            for qtype, score in scores.items():
                if score == max_score:
                    return qtype
        
        # Context-based classification
        if state.get('videos') and any(kw in query_lower for kw in ['this', 'these', 'it']):
            return QueryType.ANALYSIS.value
        
        if state.get('media_plan_info') and 'budget' in query_lower:
            return QueryType.PLANNING.value
        
        return QueryType.GENERAL.value
    
    def _select_agents(self, query_type: str, query: str, state: AgentState) -> List[str]:
        """
        Select which agents to invoke based on query type and context.
        
        Args:
            query_type: Classified query type
            query: Original query
            state: Current state
            
        Returns:
            List of agent names to invoke
        """
        # Start with default agents for this query type
        agents = list(QUERY_TYPE_TO_AGENTS.get(query_type, []))
        
        query_lower = query.lower()
        
        # Add agents based on specific needs
        if 'recommend' in query_lower or 'should' in query_lower:
            if AgentName.STRATEGY.value not in agents:
                agents.append(AgentName.STRATEGY.value)
        
        if 'historical' in query_lower or 'past' in query_lower or 'similar' in query_lower:
            if AgentName.RESULTS_INTERPRETER.value not in agents:
                agents.append(AgentName.RESULTS_INTERPRETER.value)
        
        if 'video' in query_lower or 'creative' in query_lower or 'score' in query_lower:
            if AgentName.CREATIVE_ANALYST.value not in agents:
                agents.append(AgentName.CREATIVE_ANALYST.value)
        
        # Limit to 3 agents max
        return agents[:3]
    
    def _handle_inter_agent_request(self, request, state: AgentState):
        """
        Handle a request from one agent to another.
        
        Args:
            request: InterAgentRequest object
            state: Current state
        """
        target_agent = request.to_agent
        
        if target_agent not in self.agents:
            request.answer = f"Agent {target_agent} not available"
            state['inter_agent_requests'].append(request.to_dict())
            return
        
        try:
            add_reasoning_step(
                state, self.name, "routing",
                f"Routing request from {request.from_agent} to {target_agent}: {request.question[:50]}..."
            )
            
            answer = self.agents[target_agent].answer_direct_question(request.question, state)
            request.answer = answer
            
            add_reasoning_step(
                state, target_agent, "responding",
                f"Answered: {answer[:100]}..."
            )
            
        except Exception as e:
            request.answer = f"Error: {str(e)}"
        
        state['inter_agent_requests'].append(request.to_dict())
    
    def _synthesize_response(self, state: AgentState) -> str:
        """
        Synthesize a final response from all agent outputs.
        
        Args:
            state: Current state with agent responses
            
        Returns:
            Synthesized response string
        """
        responses = state.get('agent_responses', {})
        errors = state.get('agent_errors', {})
        query = state.get('current_query', '')
        
        # If no agents were invoked, handle directly
        if not responses and not errors:
            return self._handle_general_query(query, state)
        
        # If only one agent responded successfully, use its response
        if len(responses) == 1 and not errors:
            agent_name = list(responses.keys())[0]
            return responses[agent_name].get('content', '')
        
        # Multiple agents or some errors - synthesize
        return self._llm_synthesize(query, responses, errors, state)
    
    def _handle_general_query(self, query: str, state: AgentState) -> str:
        """Handle a general query without specialist agents."""
        
        # Build context
        context_parts = []
        
        if state.get('media_plan_info'):
            mp = state['media_plan_info']
            context_parts.append(f"Campaign: {mp.get('brand', '')} - {mp.get('campaign_name', '')}")
            context_parts.append(f"Budget: ${mp.get('total_budget', 0):,.0f}")
        
        if state.get('videos'):
            context_parts.append(f"Videos uploaded: {len(state['videos'])}")
        
        context = "\n".join(context_parts) if context_parts else "No files uploaded yet."
        
        prompt = f"""Context:
{context}

User question: {query}

Provide a helpful response. If the question requires specific analysis, suggest what the user should do (upload files, ask about specific videos, etc.)."""

        system = """You are the CT Orchestrator assistant. You help users with creative testing questions.
If you don't have enough information to answer, guide the user on what to provide.
Be concise and helpful."""

        return get_completion(prompt=prompt, system=system, max_tokens=500)
    
    def _llm_synthesize(self, query: str, responses: Dict, errors: Dict, state: AgentState) -> str:
        """Use LLM to synthesize multiple agent responses."""
        
        # Format agent responses
        response_text = []
        for agent_name, response in responses.items():
            display_name = agent_name.replace('_', ' ').title()
            content = response.get('content', '')
            response_text.append(f"**{display_name}:**\n{content}")
        
        # Note any errors
        error_text = ""
        if errors:
            error_agents = list(errors.keys())
            error_text = f"\n\nNote: {', '.join(error_agents)} could not respond."
        
        prompt = f"""User query: {query}

Agent responses:

{chr(10).join(response_text)}
{error_text}

Synthesize these responses into a single, coherent answer for the user.
- Combine insights from all agents
- Resolve any conflicting information
- Present a clear recommendation if applicable
- Use markdown formatting
- Be concise but complete"""

        system = """You are synthesizing responses from multiple AI agents into a single coherent answer.
Combine the insights, highlight key points, and present a unified response.
Don't mention the individual agents unless their perspectives differ significantly."""

        return get_completion(prompt=prompt, system=system, max_tokens=1000)
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate."""
        return len(text) // 4


# Singleton instance
_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    """Get or create the orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


def reset_orchestrator() -> Orchestrator:
    """Reset and return a fresh orchestrator."""
    global _orchestrator
    _orchestrator = Orchestrator()
    return _orchestrator
