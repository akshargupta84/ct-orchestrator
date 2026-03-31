"""
Services module.

Provides business logic services for the CT Orchestrator.
All imports are wrapped in try/except so the app works even when
optional dependencies (pdfplumber, chromadb, etc.) are not installed.
"""

# Rules Engine
try:
    from .rules_engine import RulesEngine, get_rules_engine
except ImportError:
    RulesEngine = None
    get_rules_engine = None

# CSV Parser
try:
    from .csv_parser import CSVParser, CSVParseResult, generate_sample_csv
except ImportError:
    CSVParser = None
    CSVParseResult = None
    generate_sample_csv = None

# Vector Store
try:
    from .vector_store import VectorStore, get_vector_store
except ImportError:
    VectorStore = None
    get_vector_store = None

# Report Generator
try:
    from .report_generator import ReportGenerator
except ImportError:
    ReportGenerator = None

# Persistence
try:
    from .persistence import PersistenceService, get_persistence_service
except ImportError:
    PersistenceService = None
    get_persistence_service = None

# Advanced Analytics
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
