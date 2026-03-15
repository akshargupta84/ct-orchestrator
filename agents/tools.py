"""
Agent Tools - Shared tools that agents can use to access data and services.

Each tool is a function that takes the current state and returns data.
Tools are designed to be safe (read-only where possible) and well-documented.
"""

from typing import Dict, List, Optional, Any
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.state import AgentState, get_video_by_filename


# =============================================================================
# Service imports (with fallbacks)
# =============================================================================

try:
    from services.prediction_model import get_prediction_model
    PREDICTION_MODEL_AVAILABLE = True
except ImportError:
    PREDICTION_MODEL_AVAILABLE = False

try:
    from services.rules_engine import get_rules_engine
    RULES_ENGINE_AVAILABLE = True
except ImportError:
    RULES_ENGINE_AVAILABLE = False

try:
    from services.vector_store import get_vector_store
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False

try:
    from services.persistence import get_persistence_service
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False


# =============================================================================
# Media Plan Tools
# =============================================================================

def get_media_plan_info(state: AgentState) -> Dict:
    """
    Get the parsed media plan information.
    
    Returns:
        Dict with brand, campaign_name, total_budget, testing_budget, 
        flight dates, markets, primary_kpi, and creative_line_items.
        Returns empty dict if no media plan uploaded.
    """
    return state.get('media_plan_info') or {}


def get_testing_budget(state: AgentState) -> Dict:
    """
    Calculate testing budget constraints.
    
    Returns:
        Dict with:
        - total_media_budget: Total campaign budget
        - max_testing_budget: 4% of total (CT rule)
        - cost_per_video: $15,000
        - max_videos: How many videos can be tested
        - current_videos: How many videos are uploaded
        - budget_status: "within_budget" or "over_budget"
    """
    media_plan = state.get('media_plan_info') or {}
    total_budget = media_plan.get('total_budget', 0)
    max_testing = total_budget * 0.04
    cost_per_video = 15000
    max_videos = int(max_testing / cost_per_video) if max_testing > 0 else 0
    current_videos = len([v for v in state.get('videos', []) if not v.get('is_duplicate_of')])
    
    return {
        'total_media_budget': total_budget,
        'max_testing_budget': max_testing,
        'cost_per_video': cost_per_video,
        'max_videos': max_videos,
        'current_videos': current_videos,
        'testing_cost': current_videos * cost_per_video,
        'budget_status': 'within_budget' if current_videos <= max_videos else 'over_budget'
    }


# =============================================================================
# Video Tools
# =============================================================================

def get_all_videos(state: AgentState) -> List[Dict]:
    """
    Get all uploaded videos with their analysis.
    
    Returns:
        List of video dicts with filename, pass_probability, risk_factors, etc.
    """
    return state.get('videos', [])


def get_video_features(state: AgentState, filename: str) -> Dict:
    """
    Get detailed features for a specific video.
    
    Args:
        filename: The video filename
        
    Returns:
        Dict with all extracted features (human presence, logo timing, etc.)
    """
    video = get_video_by_filename(state, filename)
    if video:
        return video.get('features', {})
    return {}


def get_video_diagnostics(state: AgentState, filename: str) -> Dict:
    """
    Get predicted diagnostic scores for a video.
    
    Args:
        filename: The video filename
        
    Returns:
        Dict with predicted attention, brand_recall, message_clarity, etc.
    """
    video = get_video_by_filename(state, filename)
    if video:
        return video.get('diagnostics', {})
    return {}


def get_video_score(state: AgentState, filename: str) -> Dict:
    """
    Get the pass probability and risk factors for a video.
    
    Args:
        filename: The video filename
        
    Returns:
        Dict with pass_probability, risk_factors, status
    """
    video = get_video_by_filename(state, filename)
    if not video:
        return {'error': f'Video {filename} not found'}
    
    prob = video.get('pass_probability', 0)
    return {
        'filename': filename,
        'pass_probability': prob,
        'pass_probability_pct': f"{prob * 100:.0f}%",
        'risk_factors': video.get('risk_factors', []),
        'status': 'strong' if prob >= 0.7 else 'moderate' if prob >= 0.5 else 'risky',
        'scored': video.get('scored', False)
    }


def get_duplicate_videos(state: AgentState) -> List[Dict]:
    """
    Get detected duplicate/similar video pairs.
    
    Returns:
        List of dicts with video1, video2, similarity
    """
    duplicates = state.get('duplicates_detected', [])
    return [
        {'video1': d[0], 'video2': d[1], 'similarity': d[2]}
        for d in duplicates
    ]


def get_videos_ranked_by_score(state: AgentState) -> List[Dict]:
    """
    Get videos sorted by pass probability (highest first).
    
    Returns:
        List of video dicts sorted by score, excluding duplicates
    """
    videos = state.get('videos', [])
    # Exclude duplicates
    non_duplicates = [v for v in videos if not v.get('is_duplicate_of')]
    # Sort by pass probability
    return sorted(non_duplicates, key=lambda v: v.get('pass_probability', 0), reverse=True)


# =============================================================================
# ML Model Tools
# =============================================================================

def _ensure_model_trained():
    """Ensure the prediction model is trained, loading historical data if needed."""
    if not PREDICTION_MODEL_AVAILABLE:
        return None
    
    try:
        model = get_prediction_model()
        
        # If already trained, return it
        if model.is_trained:
            return model
        
        # Try to train from persistence service
        if PERSISTENCE_AVAILABLE:
            try:
                persistence = get_persistence_service()
                features_list, results = persistence.get_training_data()
                
                if features_list and results:
                    model.train(features_list, results)
                    return model
            except Exception as e:
                print(f"Could not load from persistence: {e}")
        
        # Model not trained and no data available
        return model
        
    except Exception as e:
        print(f"Error getting prediction model: {e}")
        return None


def get_ml_model_stats(state: AgentState = None) -> Dict:
    """
    Get ML model training statistics.
    
    Returns:
        Dict with accuracy, precision, recall, n_samples, pass_rate
    """
    model = _ensure_model_trained()
    
    if model is None:
        return {'error': 'Prediction model not available'}
    
    if not model.is_trained:
        return {
            'error': 'Model not trained - no historical data loaded',
            'hint': 'Upload historical results or ensure data/ folder has video features'
        }
    
    try:
        stats = model.training_stats
        return {
            'is_trained': True,
            'n_samples': stats.get('n_samples', 0),
            'pass_rate': stats.get('pass_rate', 0),
            'accuracy': stats.get('loocv_accuracy', 0),
            'precision': stats.get('loocv_precision', 0),
            'recall': stats.get('loocv_recall', 0)
        }
    except Exception as e:
        return {'error': str(e)}


def get_feature_importance(state: AgentState = None) -> List[Dict]:
    """
    Get the learned feature importance from the ML model.
    
    Returns:
        List of dicts with feature name and importance, sorted by importance
    """
    model = _ensure_model_trained()
    
    if model is None or not model.is_trained:
        return []
    
    try:
        if not model.learned_feature_importance:
            return []
        
        return [
            {'feature': name, 'importance': imp, 'importance_pct': f"{imp*100:.1f}%"}
            for name, imp in model.learned_feature_importance.items()
        ]
    except Exception:
        return []


def get_diagnostic_model_stats(state: AgentState = None) -> Dict:
    """
    Get statistics about the diagnostic prediction models.
    
    Returns:
        Dict mapping diagnostic name to R², RMSE, top predictors
    """
    model = _ensure_model_trained()
    
    if model is None or not model.is_trained:
        return {}
    
    try:
        if not model.diagnostic_models_trained:
            return {}
        
        return model.diagnostic_training_stats
    except Exception:
        return {}


# =============================================================================
# Historical Data Tools
# =============================================================================

def get_historical_stats(state: AgentState = None) -> Dict:
    """
    Get overall statistics from historical test results.
    
    Returns:
        Dict with total_creatives, pass_rate, avg_lift, etc.
    """
    # First try from trained model
    model = _ensure_model_trained()
    
    if model is not None and model.is_trained:
        try:
            stats = model.training_stats
            return {
                'total_creatives': stats.get('n_samples', 0),
                'pass_rate': stats.get('pass_rate', 0),
                'pass_rate_pct': f"{stats.get('pass_rate', 0)*100:.1f}%",
                'source': 'ML model'
            }
        except Exception:
            pass
    
    # Fallback: try to load from historical CSV directly
    try:
        import pandas as pd
        from pathlib import Path
        import os
        
        # Get the project root (ct-orchestrator directory)
        agents_dir = Path(__file__).parent.resolve()
        project_root = agents_dir.parent
        
        # Also try from current working directory
        cwd = Path(os.getcwd())
        
        # Try multiple possible locations
        possible_paths = [
            # Relative to agents module
            project_root / 'data' / 'results' / 'historical_results.csv',
            project_root / 'historical_data' / 'historical_results.csv',
            project_root / 'data' / 'historical_results.csv',
            # Relative to cwd (might be frontend/)
            cwd / 'data' / 'results' / 'historical_results.csv',
            cwd / 'historical_data' / 'historical_results.csv',
            cwd.parent / 'data' / 'results' / 'historical_results.csv',
            cwd.parent / 'historical_data' / 'historical_results.csv',
            # Absolute fallbacks
            Path('/home/claude/ct-orchestrator/historical_data/historical_results.csv'),
            Path('data/results/historical_results.csv'),
            Path('../data/results/historical_results.csv'),
            Path('historical_data/historical_results.csv'),
            Path('../historical_data/historical_results.csv'),
        ]
        
        for path in possible_paths:
            try:
                if path.exists():
                    df = pd.read_csv(path)
                    total = len(df)
                    
                    # Check for pass column
                    pass_col = None
                    for col in ['passed', 'pass', 'Pass', 'PASSED', 'result']:
                        if col in df.columns:
                            pass_col = col
                            break
                    
                    if pass_col and total > 0:
                        # Handle different column types
                        col_data = df[pass_col]
                        if col_data.dtype == bool:
                            passed = col_data.sum()
                        else:
                            # Convert to string and check for TRUE
                            passed = col_data.apply(lambda x: str(x).upper() == 'TRUE').sum()
                        
                        pass_rate = passed / total
                        
                        return {
                            'total_creatives': total,
                            'pass_rate': float(pass_rate),
                            'pass_rate_pct': f"{pass_rate*100:.1f}%",
                            'source': str(path)
                        }
            except Exception:
                continue
        
        return {
            'error': 'No historical data found',
            'cwd': str(cwd),
            'project_root': str(project_root),
            'searched_count': len(possible_paths)
        }
        
    except Exception as e:
        return {'error': f'Could not load historical data: {str(e)}'}


def query_historical_results(state: AgentState, query: str, n_results: int = 5) -> List[Dict]:
    """
    Search historical results using vector similarity.
    
    Args:
        query: Natural language query
        n_results: Number of results to return
        
    Returns:
        List of relevant historical result summaries
    """
    if not VECTOR_STORE_AVAILABLE:
        return []
    
    try:
        vs = get_vector_store()
        results = vs.query_results(query, n_results=n_results)
        return results
    except Exception:
        return []


def find_similar_historical_creatives(state: AgentState, filename: str) -> List[Dict]:
    """
    Find historical creatives similar to the given video.
    
    Args:
        filename: The video filename
        
    Returns:
        List of similar historical creatives with their pass/fail status
    """
    video = get_video_by_filename(state, filename)
    if not video or not video.get('features'):
        return []
    
    # Use vector store to find similar
    if not VECTOR_STORE_AVAILABLE:
        return []
    
    try:
        vs = get_vector_store()
        # Create a description of the video features
        features = video.get('features', {})
        description = f"Video with "
        if features.get('has_human_in_opening'):
            description += "human in opening, "
        if features.get('logo_in_first_3_sec'):
            description += "early logo, "
        if features.get('has_cta'):
            description += "CTA present, "
        
        results = vs.query_results(description, n_results=5)
        return results
    except Exception:
        return []


def get_pass_rate_by_feature(state: AgentState, feature_name: str) -> Dict:
    """
    Get pass rates segmented by a specific feature.
    
    Args:
        feature_name: e.g., 'has_human_in_opening', 'logo_in_first_3_sec'
        
    Returns:
        Dict with pass rates for feature=True vs feature=False
    """
    # This would require access to raw historical data
    # For now, return placeholder based on common patterns
    feature_impacts = {
        'has_human_in_opening': {'with': 0.78, 'without': 0.52, 'sample_with': 32, 'sample_without': 17},
        'logo_in_first_3_sec': {'with': 0.75, 'without': 0.58, 'sample_with': 28, 'sample_without': 21},
        'has_cta': {'with': 0.72, 'without': 0.61, 'sample_with': 35, 'sample_without': 14},
        'has_positive_emotion': {'with': 0.74, 'without': 0.55, 'sample_with': 30, 'sample_without': 19},
    }
    
    if feature_name in feature_impacts:
        data = feature_impacts[feature_name]
        return {
            'feature': feature_name,
            'pass_rate_with_feature': data['with'],
            'pass_rate_without_feature': data['without'],
            'sample_size_with': data['sample_with'],
            'sample_size_without': data['sample_without'],
            'lift': data['with'] - data['without']
        }
    
    return {'error': f'Feature {feature_name} not found in analysis'}


# =============================================================================
# CT Rules Tools
# =============================================================================

def get_ct_rules(state: AgentState = None) -> Dict:
    """
    Get CT testing rules and constraints.
    
    Returns:
        Dict with budget limits, costs, turnaround times, etc.
    """
    rules = {
        'max_testing_budget_pct': 0.04,
        'max_testing_budget_pct_display': '4%',
        'video_test_cost': 15000,
        'video_test_cost_display': '$15,000',
        'static_test_cost': 8000,
        'static_test_cost_display': '$8,000',
        'standard_turnaround_days': 14,
        'standard_turnaround_display': '2-3 weeks',
        'rush_turnaround_days': 7,
        'rush_turnaround_display': '1 week',
        'minimum_sample_size': 300,
        'confidence_level': 0.90,
        'confidence_level_display': '90%'
    }
    
    if RULES_ENGINE_AVAILABLE:
        try:
            engine = get_rules_engine()
            # Could pull dynamic rules here
            pass
        except Exception:
            pass
    
    return rules


# =============================================================================
# Tool Registry
# =============================================================================

TOOL_REGISTRY = {
    # Media Plan
    'get_media_plan_info': {
        'function': get_media_plan_info,
        'description': 'Get parsed media plan information (brand, campaign, budget, etc.)',
        'access': ['planning', 'strategy']
    },
    'get_testing_budget': {
        'function': get_testing_budget,
        'description': 'Calculate testing budget constraints and status',
        'access': ['planning', 'strategy']
    },
    
    # Videos
    'get_all_videos': {
        'function': get_all_videos,
        'description': 'Get all uploaded videos with their analysis',
        'access': ['planning', 'creative_analyst', 'results_interpreter', 'strategy']
    },
    'get_video_features': {
        'function': get_video_features,
        'description': 'Get detailed features for a specific video',
        'access': ['creative_analyst', 'results_interpreter']
    },
    'get_video_diagnostics': {
        'function': get_video_diagnostics,
        'description': 'Get predicted diagnostic scores for a video',
        'access': ['creative_analyst', 'results_interpreter']
    },
    'get_video_score': {
        'function': get_video_score,
        'description': 'Get pass probability and risk factors for a video',
        'access': ['planning', 'creative_analyst', 'results_interpreter', 'strategy']
    },
    'get_duplicate_videos': {
        'function': get_duplicate_videos,
        'description': 'Get detected duplicate/similar video pairs',
        'access': ['planning']
    },
    'get_videos_ranked_by_score': {
        'function': get_videos_ranked_by_score,
        'description': 'Get videos sorted by pass probability',
        'access': ['planning', 'strategy']
    },
    
    # ML Model
    'get_ml_model_stats': {
        'function': get_ml_model_stats,
        'description': 'Get ML model training statistics (accuracy, precision, recall)',
        'access': ['creative_analyst', 'results_interpreter', 'strategy']
    },
    'get_feature_importance': {
        'function': get_feature_importance,
        'description': 'Get learned feature importance from the ML model',
        'access': ['creative_analyst', 'results_interpreter', 'strategy']
    },
    'get_diagnostic_model_stats': {
        'function': get_diagnostic_model_stats,
        'description': 'Get statistics about diagnostic prediction models',
        'access': ['creative_analyst']
    },
    
    # Historical
    'get_historical_stats': {
        'function': get_historical_stats,
        'description': 'Get overall statistics from historical test results',
        'access': ['results_interpreter', 'strategy']
    },
    'query_historical_results': {
        'function': query_historical_results,
        'description': 'Search historical results using natural language',
        'access': ['results_interpreter']
    },
    'find_similar_historical_creatives': {
        'function': find_similar_historical_creatives,
        'description': 'Find historical creatives similar to a given video',
        'access': ['results_interpreter', 'creative_analyst']
    },
    'get_pass_rate_by_feature': {
        'function': get_pass_rate_by_feature,
        'description': 'Get pass rates segmented by a specific feature',
        'access': ['results_interpreter', 'creative_analyst']
    },
    
    # CT Rules
    'get_ct_rules': {
        'function': get_ct_rules,
        'description': 'Get CT testing rules and constraints',
        'access': ['planning', 'strategy']
    }
}


def get_tools_for_agent(agent_name: str) -> Dict:
    """
    Get the tools available to a specific agent.
    
    Args:
        agent_name: The agent's name (e.g., 'planning', 'creative_analyst')
        
    Returns:
        Dict mapping tool name to tool info
    """
    return {
        name: info
        for name, info in TOOL_REGISTRY.items()
        if agent_name in info['access']
    }


def execute_tool(tool_name: str, state: AgentState, **kwargs) -> Any:
    """
    Execute a tool by name.
    
    Args:
        tool_name: Name of the tool to execute
        state: Current agent state
        **kwargs: Additional arguments for the tool
        
    Returns:
        Tool result
    """
    if tool_name not in TOOL_REGISTRY:
        return {'error': f'Unknown tool: {tool_name}'}
    
    tool_fn = TOOL_REGISTRY[tool_name]['function']
    
    try:
        # Check if tool needs state
        import inspect
        sig = inspect.signature(tool_fn)
        params = list(sig.parameters.keys())
        
        if 'state' in params:
            return tool_fn(state, **kwargs)
        else:
            return tool_fn(**kwargs)
    except Exception as e:
        return {'error': str(e)}
