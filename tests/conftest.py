"""
Shared test fixtures for CT Orchestrator test suite.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def default_rules():
    """Provide the default CT Rules for testing."""
    from models.rules import DEFAULT_CT_RULES
    return DEFAULT_CT_RULES


@pytest.fixture
def sample_csv_content():
    """Provide sample CSV content for parser testing."""
    from services.csv_parser import generate_sample_csv
    return generate_sample_csv()


@pytest.fixture
def tmp_db(tmp_path):
    """Provide a temporary database path for usage tracker tests."""
    return str(tmp_path / "test_usage.db")
