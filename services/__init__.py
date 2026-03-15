"""
Services module.

Provides business logic services for the CT Orchestrator.
"""

from .rules_engine import RulesEngine, get_rules_engine
from .csv_parser import CSVParser, CSVParseResult, generate_sample_csv
from .vector_store import VectorStore, get_vector_store
from .report_generator import ReportGenerator
from .persistence import PersistenceService, get_persistence_service

# Try to import advanced analytics
try:
    from .advanced_analytics import AdvancedAnalyticsService, get_analytics_service
    ADVANCED_ANALYTICS_AVAILABLE = True
except ImportError:
    ADVANCED_ANALYTICS_AVAILABLE = False
    AdvancedAnalyticsService = None
    get_analytics_service = None

__all__ = [
    "RulesEngine",
    "get_rules_engine",
    "CSVParser", 
    "CSVParseResult",
    "generate_sample_csv",
    "VectorStore",
    "get_vector_store",
    "ReportGenerator",
    "PersistenceService",
    "get_persistence_service",
    "AdvancedAnalyticsService",
    "get_analytics_service",
]
