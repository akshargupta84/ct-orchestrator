"""
LLM utility for Claude API calls.

Provides a consistent interface for making Claude API calls throughout the system.
"""

import os
from typing import Optional, Any
from anthropic import Anthropic
from pydantic import BaseModel


# Initialize client
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Model configuration
DEFAULT_MODEL = "claude-sonnet-4-20250514"
ADVANCED_MODEL = "claude-sonnet-4-20250514"  # For complex analysis


def get_completion(
    prompt: str,
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> str:
    """
    Get a text completion from Claude.
    
    Args:
        prompt: The user message/prompt
        system: Optional system prompt
        model: Model to use
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        
    Returns:
        The assistant's response text
    """
    messages = [{"role": "user", "content": prompt}]
    
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "temperature": temperature,
    }
    
    if system:
        kwargs["system"] = system
    
    response = client.messages.create(**kwargs)
    return response.content[0].text


def get_structured_output(
    prompt: str,
    output_schema: type[BaseModel],
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> BaseModel:
    """
    Get a structured output from Claude that conforms to a Pydantic model.
    
    This uses Claude's ability to output JSON and validates against the schema.
    
    Args:
        prompt: The user message/prompt
        output_schema: Pydantic model class for the expected output
        system: Optional system prompt
        model: Model to use
        max_tokens: Maximum tokens in response
        
    Returns:
        An instance of the output_schema populated with Claude's response
    """
    schema_json = output_schema.model_json_schema()
    
    structured_system = f"""You are a helpful assistant that outputs structured JSON.
Always respond with valid JSON that conforms to this schema:
{schema_json}

Do not include any text before or after the JSON. Only output the JSON object."""

    if system:
        structured_system = f"{system}\n\n{structured_system}"
    
    response_text = get_completion(
        prompt=prompt,
        system=structured_system,
        model=model,
        max_tokens=max_tokens,
        temperature=0.1,  # Lower temperature for structured output
    )
    
    # Clean up response (remove markdown code blocks if present)
    clean_text = response_text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    if clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    clean_text = clean_text.strip()
    
    # Parse and validate
    return output_schema.model_validate_json(clean_text)


def get_analysis(
    data: str,
    question: str,
    context: Optional[str] = None,
    model: str = ADVANCED_MODEL,
) -> str:
    """
    Get analysis of data with a specific question.
    
    Args:
        data: The data to analyze (CSV, JSON, or text)
        question: The analysis question
        context: Optional additional context
        model: Model to use
        
    Returns:
        Analysis text
    """
    system = """You are an expert data analyst specializing in marketing analytics 
and creative testing. Provide clear, actionable insights based on the data provided.
Be specific and cite numbers when relevant."""

    prompt = f"""Analyze the following data:

{data}

{f"Context: {context}" if context else ""}

Question: {question}

Provide a detailed analysis with specific insights and recommendations."""

    return get_completion(prompt=prompt, system=system, model=model)


def classify_question(question: str) -> str:
    """
    Classify a user question into categories for routing.
    
    Categories:
    - factual: Questions about rules, definitions, processes
    - data: Questions about specific campaign/creative results
    - meta: Questions requiring cross-campaign analysis
    - action: Requests to do something (create plan, generate report)
    
    Returns:
        One of: "factual", "data", "meta", "action"
    """
    system = """Classify user questions into one of these categories:
- factual: Questions about rules, definitions, processes, benchmarks
- data: Questions about specific campaign or creative results
- meta: Questions requiring analysis across multiple campaigns/creatives
- action: Requests to perform an action (create plan, upload, generate report)

Respond with only the category name, nothing else."""

    response = get_completion(
        prompt=f"Classify this question: {question}",
        system=system,
        temperature=0,
        max_tokens=20,
    )
    
    category = response.strip().lower()
    if category not in ["factual", "data", "meta", "action"]:
        return "factual"  # Default
    return category
