"""
LLM utility for Claude API calls.

Provides a consistent interface for making Claude API calls throughout the system.
"""

import os
import json
from typing import Optional, Any
from pathlib import Path

# Load .env from project root
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # python-dotenv not installed (e.g., on HF Spaces)

# Try to import Anthropic client
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None


# Model configuration
DEFAULT_MODEL = "claude-sonnet-4-20250514"
ADVANCED_MODEL = "claude-sonnet-4-20250514"  # For complex analysis


def _get_api_key() -> Optional[str]:
    """Get API key from environment or session state."""
    # Check environment variable first
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        return api_key
    
    # Try to check session state (set via Admin page)
    try:
        import streamlit as st
        if hasattr(st, 'session_state') and "anthropic_api_key" in st.session_state:
            return st.session_state.anthropic_api_key
    except ImportError:
        pass
    
    return None


def _get_client():
    """Get or create Anthropic client."""
    if not ANTHROPIC_AVAILABLE:
        raise ImportError(
            "anthropic package not installed. Run: pip install anthropic"
        )
    
    api_key = _get_api_key()
    if not api_key:
        raise ValueError(
            "Anthropic API key not set. Please either:\n"
            "1. Set ANTHROPIC_API_KEY environment variable, or\n"
            "2. Enter your API key in the Admin page"
        )
    
    return Anthropic(api_key=api_key)


def is_llm_available() -> bool:
    """Check if LLM functionality is available."""
    if not ANTHROPIC_AVAILABLE:
        return False
    return _get_api_key() is not None


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
        
    Raises:
        ValueError: If API key is not set
        ImportError: If anthropic package is not installed
    """
    client = _get_client()
    
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
    output_schema: type,
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> Any:
    """
    Get a structured output from Claude that conforms to a Pydantic model.
    
    Args:
        prompt: The user message/prompt
        output_schema: Pydantic model class for the expected output
        system: Optional system prompt
        model: Model to use
        max_tokens: Maximum tokens in response
        
    Returns:
        An instance of the output_schema populated with Claude's response
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError("pydantic package required for structured output")
    
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


def synthesize_analysis(
    analysis_results: dict,
    campaign_name: str = "Unknown Campaign",
    model: str = ADVANCED_MODEL,
    max_tokens: int = 2000,
) -> str:
    """
    Synthesize advanced analysis results into a comprehensive narrative report.
    
    This is Stage 4 of the analysis pipeline - Claude takes all the quantitative
    findings from statistical analysis, historical comparison, and pattern mining
    and generates human-readable insights and recommendations.
    
    Args:
        analysis_results: Dict containing all analysis outputs:
            - statistical: StatisticalFindings
            - historical: HistoricalContext  
            - patterns: PatternMiningResults
            - recommendations: List of CreativeRecommendation
            - raw_data: Original data
        campaign_name: Name of the campaign
        model: Model to use
        max_tokens: Maximum tokens in response
        
    Returns:
        Comprehensive analysis report as markdown text
    """
    system = """You are an expert creative strategist and data analyst specializing in 
brand lift studies and creative testing. Your job is to synthesize quantitative analysis 
results into clear, actionable insights for marketing teams.

Your analysis should:
1. Lead with the most important findings
2. Be specific - cite actual numbers and percentages
3. Provide actionable recommendations, not just observations
4. Explain WHY things worked or didn't work based on the data
5. Prioritize recommendations by impact and ease of implementation

Write in a professional but accessible tone. Use markdown formatting."""

    # Build the context from analysis results
    context_parts = []
    
    # Campaign overview
    raw_data = analysis_results.get('raw_data', {})
    results_df = raw_data.get('results_df', [])
    primary_kpi = raw_data.get('primary_kpi', 'awareness')
    
    context_parts.append(f"# Analysis Context\n")
    context_parts.append(f"**Campaign:** {campaign_name}")
    context_parts.append(f"**Primary KPI:** {primary_kpi}")
    context_parts.append(f"**Creatives Tested:** {len(results_df)}")
    
    # Statistical findings
    statistical = analysis_results.get('statistical', {})
    if hasattr(statistical, '__dict__'):
        statistical = statistical.__dict__
    
    context_parts.append(f"\n## Statistical Analysis Results\n")
    
    if statistical.get('correlations'):
        context_parts.append("**Correlations with Lift:**")
        for metric, corr in statistical['correlations'].items():
            pval = statistical.get('correlation_pvalues', {}).get(metric, 'N/A')
            context_parts.append(f"- {metric}: r={corr}, p={pval}")
    
    if statistical.get('regression_r_squared'):
        context_parts.append(f"\n**Regression Model R²:** {statistical['regression_r_squared']}")
        if statistical.get('regression_coefficients'):
            context_parts.append("**Coefficients:**")
            for metric, coef in statistical['regression_coefficients'].items():
                context_parts.append(f"- {metric}: {coef}")
    
    if statistical.get('significant_differences'):
        context_parts.append("\n**Significant Differences (Pass vs Fail):**")
        for diff in statistical['significant_differences']:
            context_parts.append(
                f"- {diff.get('metric_name', diff.get('metric'))}: "
                f"Passed avg={diff.get('passed_mean')}, Failed avg={diff.get('failed_mean')}, "
                f"p={diff.get('p_value')}, effect={diff.get('effect_interpretation')}"
            )
    
    if statistical.get('key_predictors'):
        context_parts.append("\n**Top Predictors of Success:**")
        for pred in statistical['key_predictors']:
            if isinstance(pred, dict):
                context_parts.append(
                    f"- {pred.get('metric_name', pred.get('metric'))}: "
                    f"correlation={pred.get('correlation')}, effect_size={pred.get('effect_size')}"
                )
    
    # Historical context
    historical = analysis_results.get('historical', {})
    if hasattr(historical, '__dict__'):
        historical = historical.__dict__
    
    context_parts.append(f"\n## Historical Context\n")
    context_parts.append(f"**Current Test Pass Rate:** {historical.get('current_pass_rate', 'N/A')}%")
    context_parts.append(f"**Historical Pass Rate:** {historical.get('historical_pass_rate', 'N/A')}%")
    context_parts.append(f"**Trend:** {historical.get('trend_direction', 'unknown')}")
    
    if historical.get('percentile_ranks'):
        context_parts.append("\n**Percentile Rankings (vs History):**")
        for creative, pct in historical['percentile_ranks'].items():
            context_parts.append(f"- {creative}: {pct}th percentile")
    
    if historical.get('pattern_matches'):
        context_parts.append("\n**Historical Pattern Matches:**")
        for pm in historical['pattern_matches']:
            context_parts.append(
                f"- {pm.get('creative_name')}: {pm.get('pattern')} "
                f"(historical pass rate: {pm.get('historical_pass_rate')}%)"
            )
    
    # Pattern mining results
    patterns = analysis_results.get('patterns', {})
    if hasattr(patterns, '__dict__'):
        patterns = patterns.__dict__
    
    context_parts.append(f"\n## Pattern Mining Results\n")
    
    if patterns.get('decision_rules'):
        context_parts.append("**Decision Rules:**")
        for rule in patterns['decision_rules']:
            context_parts.append(f"- {rule}")
    
    if patterns.get('feature_importance'):
        context_parts.append("\n**Feature Importance:**")
        for feature, importance in patterns['feature_importance'].items():
            context_parts.append(f"- {feature}: {importance}")
    
    if patterns.get('clusters'):
        context_parts.append("\n**Creative Clusters:**")
        for cluster in patterns['clusters']:
            context_parts.append(
                f"- {cluster.get('name')} ({cluster.get('size')} creatives): "
                f"{cluster.get('pass_rate')}% pass rate - {', '.join(cluster.get('characteristics', []))}"
            )
    
    if patterns.get('winning_combinations'):
        context_parts.append("\n**Winning Combinations:**")
        for combo in patterns['winning_combinations']:
            context_parts.append(
                f"- {combo.get('condition')}: {combo.get('pass_rate')}% pass rate"
            )
    
    # Per-creative data
    recommendations = analysis_results.get('recommendations', [])
    context_parts.append(f"\n## Per-Creative Data\n")
    
    for rec in recommendations:
        if hasattr(rec, '__dict__'):
            rec = rec.__dict__
        
        context_parts.append(f"\n### {rec.get('creative_name')}")
        context_parts.append(f"- **Category:** {rec.get('category')}")
        context_parts.append(f"- **Lift:** {rec.get('lift')}%")
        context_parts.append(f"- **Statistically Significant:** {rec.get('stat_sig')}")
        if rec.get('percentile_rank'):
            context_parts.append(f"- **Percentile Rank:** {rec.get('percentile_rank')}")
        if rec.get('strengths'):
            context_parts.append(f"- **Strengths:** {', '.join(rec['strengths'])}")
        if rec.get('weaknesses'):
            context_parts.append(f"- **Weaknesses:** {', '.join(rec['weaknesses'])}")
        if rec.get('pattern_match'):
            context_parts.append(f"- **Pattern Match:** {rec['pattern_match']}")
    
    context = "\n".join(context_parts)
    
    prompt = f"""Based on the following quantitative analysis results, write a CONCISE creative test analysis report.

{context}

Write a focused report with ONLY these sections (keep each section brief - 2-4 sentences max):

1. **Executive Summary** (3-4 sentences: pass rate, top performer, key driver of success)

2. **What Separates Winners from Losers** (Based on the statistical analysis, what are the key differences? Be specific with numbers.)

3. **Top 3 Recommendations** (Prioritized, actionable recommendations for future creative development)

4. **Confidence Note** (One sentence on sample size and reliability)

IMPORTANT: 
- Do NOT include a section for each individual creative
- Keep the total response under 400 words
- Focus on actionable insights, not descriptions
- Use bullet points sparingly

Be specific, cite actual numbers, and make recommendations actionable."""

    return get_completion(prompt=prompt, system=system, model=model, max_tokens=max_tokens)


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
    # Check if LLM is available
    if not is_llm_available():
        # Simple keyword-based classification as fallback
        question_lower = question.lower()
        if any(word in question_lower for word in ['create', 'generate', 'upload', 'make']):
            return "action"
        elif any(word in question_lower for word in ['compare', 'across', 'trend', 'all']):
            return "meta"
        elif any(word in question_lower for word in ['result', 'performance', 'lift', 'pass']):
            return "data"
        else:
            return "factual"
    
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
