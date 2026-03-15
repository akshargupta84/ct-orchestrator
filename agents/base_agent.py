"""
Base Agent - Abstract base class for all agents in the multi-agent system.

All specialist agents (Planning, Creative Analyst, Results Interpreter, Strategy)
extend this base class.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.state import (
    AgentState, AgentResponse, InterAgentRequest, ReasoningStep,
    add_reasoning_step
)
from agents.tools import get_tools_for_agent, execute_tool, TOOL_REGISTRY
from utils.llm import get_completion


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    display_name: str
    description: str
    system_prompt: str
    trigger_keywords: List[str] = field(default_factory=list)
    
    # Data access permissions
    can_access_media_plan: bool = False
    can_access_videos: bool = False
    can_access_historical: bool = False
    can_access_ml_model: bool = False
    can_access_rules: bool = False


class BaseAgent(ABC):
    """
    Base class for all agents.
    
    Provides:
    - Tool access based on permissions
    - LLM interaction with consistent prompting
    - Inter-agent communication protocol
    - Response formatting
    - Error handling
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name
        self.display_name = config.display_name
        self.tools = get_tools_for_agent(config.name)
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the agent's system prompt."""
        pass
    
    def run(self, state: AgentState) -> AgentResponse:
        """
        Run the agent on the current state.
        
        This is the main entry point for agent execution.
        
        Args:
            state: Current shared state
            
        Returns:
            AgentResponse with the agent's output
        """
        try:
            # Add reasoning step
            add_reasoning_step(state, self.name, "thinking", f"{self.display_name} is analyzing the query...")
            
            # Build context from state and tools
            context = self._build_context(state)
            
            # Build the prompt
            prompt = self._build_prompt(state, context)
            
            # Get LLM response
            response_text = self._call_llm(prompt)
            
            # Parse response for any inter-agent requests
            requests_made, clean_response = self._parse_inter_agent_requests(response_text, state)
            
            # Add reasoning step
            add_reasoning_step(state, self.name, "responding", clean_response[:200] + "..." if len(clean_response) > 200 else clean_response)
            
            return AgentResponse(
                agent_name=self.name,
                content=clean_response,
                reasoning=response_text,
                requests_made=requests_made,
                tokens_used=self._estimate_tokens(prompt, response_text)
            )
            
        except Exception as e:
            add_reasoning_step(state, self.name, "error", str(e))
            return AgentResponse(
                agent_name=self.name,
                content="",
                error=str(e)
            )
    
    def _build_context(self, state: AgentState) -> str:
        """Build context string from available tools and data."""
        context_parts = []
        
        # Add relevant data based on what's available
        if state.get('media_plan_info') and self.config.can_access_media_plan:
            media_plan = state['media_plan_info']
            context_parts.append(f"**Media Plan:**\n{json.dumps(media_plan, indent=2)}")
        
        if state.get('videos') and self.config.can_access_videos:
            videos_summary = []
            for v in state['videos']:
                prob = v.get('pass_probability', 0)
                status = '✅' if prob >= 0.7 else '⚠️' if prob >= 0.5 else '❌'
                videos_summary.append(
                    f"- {v['filename']}: {prob*100:.0f}% pass prob {status}"
                    + (f" (duplicate of {v['is_duplicate_of']})" if v.get('is_duplicate_of') else "")
                )
            context_parts.append(f"**Uploaded Videos:**\n" + "\n".join(videos_summary))
        
        if self.config.can_access_ml_model:
            ml_stats = execute_tool('get_ml_model_stats', state)
            if ml_stats and 'error' not in ml_stats:
                context_parts.append(f"**ML Model Stats:**\n{json.dumps(ml_stats, indent=2)}")
            
            feature_importance = execute_tool('get_feature_importance', state)
            if feature_importance:
                top_features = feature_importance[:5]
                context_parts.append(
                    f"**Top Predictive Features:**\n" + 
                    "\n".join([f"- {f['feature']}: {f['importance_pct']}" for f in top_features])
                )
        
        if self.config.can_access_rules:
            rules = execute_tool('get_ct_rules', state)
            context_parts.append(f"**CT Rules:**\n{json.dumps(rules, indent=2)}")
        
        if self.config.can_access_historical:
            historical = execute_tool('get_historical_stats', state)
            if historical:
                context_parts.append(f"**Historical Stats:**\n{json.dumps(historical, indent=2)}")
        
        return "\n\n".join(context_parts) if context_parts else "No additional context available."
    
    def _build_prompt(self, state: AgentState, context: str) -> str:
        """Build the prompt to send to the LLM."""
        # Get conversation history
        messages = state.get('messages', [])
        history_text = ""
        if messages:
            recent = messages[-10:]  # Last 10 messages
            for msg in recent:
                role = "User" if msg['role'] == 'user' else "Assistant"
                history_text += f"{role}: {msg['content']}\n\n"
        
        # Get current query
        current_query = state.get('current_query', '')
        
        # Build tool descriptions
        tool_descriptions = []
        for tool_name, tool_info in self.tools.items():
            tool_descriptions.append(f"- {tool_name}: {tool_info['description']}")
        tools_text = "\n".join(tool_descriptions) if tool_descriptions else "No tools available."
        
        prompt = f"""## Available Context

{context}

## Available Tools (for reference)

{tools_text}

## Conversation History

{history_text}

## Current Query

{current_query}

## Instructions

Respond to the current query based on the context provided. Be specific and cite data when available.

If you need information from another agent, format your request as:
@REQUEST[agent_name]: Your question here

Available agents to request from:
- planning: Budget, test plan design, prioritization
- creative_analyst: Video features, creative strengths/weaknesses
- results_interpreter: Historical patterns, why things pass/fail
- strategy: Recommendations, next steps

Provide your response below:"""

        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the prompt."""
        return get_completion(
            prompt=prompt,
            system=self.get_system_prompt(),
            max_tokens=1500
        )
    
    def _parse_inter_agent_requests(self, response: str, state: AgentState) -> tuple[List[InterAgentRequest], str]:
        """
        Parse any inter-agent requests from the response.
        
        Looks for patterns like: @REQUEST[agent_name]: question
        
        Returns:
            Tuple of (list of requests, cleaned response)
        """
        import re
        
        requests = []
        clean_response = response
        
        # Find all @REQUEST patterns
        pattern = r'@REQUEST\[(\w+)\]:\s*([^\n@]+)'
        matches = re.findall(pattern, response)
        
        for agent_name, question in matches:
            requests.append(InterAgentRequest(
                from_agent=self.name,
                to_agent=agent_name,
                question=question.strip()
            ))
            # Remove the request from the response
            clean_response = re.sub(
                rf'@REQUEST\[{agent_name}\]:\s*{re.escape(question)}',
                '',
                clean_response
            )
        
        return requests, clean_response.strip()
    
    def _estimate_tokens(self, prompt: str, response: str) -> int:
        """Rough estimate of tokens used."""
        # Approximate: 1 token ≈ 4 characters
        return (len(prompt) + len(response)) // 4
    
    def answer_direct_question(self, question: str, state: AgentState) -> str:
        """
        Answer a direct question from another agent.
        
        This is used for inter-agent communication.
        
        Args:
            question: The question from another agent
            state: Current shared state
            
        Returns:
            Answer string
        """
        context = self._build_context(state)
        
        prompt = f"""Another agent is asking you a question. Answer concisely and specifically.

## Context
{context}

## Question
{question}

## Your Answer (be concise):"""

        return get_completion(
            prompt=prompt,
            system=self.get_system_prompt(),
            max_tokens=500
        )
    
    def get_tool_result(self, tool_name: str, state: AgentState, **kwargs) -> Any:
        """
        Execute a tool and return the result.
        
        Args:
            tool_name: Name of the tool
            state: Current state
            **kwargs: Tool arguments
            
        Returns:
            Tool result
        """
        if tool_name not in self.tools:
            return {'error': f'Tool {tool_name} not available to {self.name}'}
        
        return execute_tool(tool_name, state, **kwargs)


def create_agent_prompt_tools_section(agent_name: str) -> str:
    """
    Create a formatted tools section for an agent's prompt.
    
    Args:
        agent_name: The agent's name
        
    Returns:
        Formatted string describing available tools
    """
    tools = get_tools_for_agent(agent_name)
    
    if not tools:
        return "No tools available."
    
    lines = ["You have access to the following tools:\n"]
    for name, info in tools.items():
        lines.append(f"**{name}**: {info['description']}")
    
    return "\n".join(lines)
