"""
Tests for Logging and Error Handling services.

Covers:
- Logger setup and formatting
- Request ID tracking
- JSON log format
- safe_call decorator (fallback, fallback_fn, error logging)
- user_friendly_error mapping
- ServiceStatus dataclass
"""

import pytest
import json
import logging
import os


class TestLoggerSetup:
    """Tests for logger initialization."""

    def test_get_logger(self):
        """Should return a logger instance."""
        from services.logger import get_logger
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_setup_logging_creates_dir(self, tmp_path):
        """Should create log directory if it doesn't exist."""
        log_dir = str(tmp_path / "test_logs")
        from services.logger import setup_logging
        setup_logging(level="DEBUG", log_dir=log_dir, json_logs=False)
        assert os.path.isdir(log_dir)

    def test_setup_logging_creates_file(self, tmp_path):
        """Should create a log file."""
        log_dir = str(tmp_path / "test_logs2")
        from services.logger import setup_logging
        setup_logging(level="DEBUG", log_dir=log_dir)
        log_file = os.path.join(log_dir, "ct_orchestrator.log")
        assert os.path.exists(log_file)


class TestRequestIdTracking:
    """Tests for request ID context tracking."""

    def test_new_request_id(self):
        """Should generate a unique request ID."""
        from services.logger import new_request_id, get_request_id
        rid = new_request_id()
        assert len(rid) == 12
        assert rid == get_request_id()

    def test_set_request_id(self):
        """Should allow setting a specific request ID."""
        from services.logger import set_request_id, get_request_id
        set_request_id("custom_abc123")
        assert get_request_id() == "custom_abc123"

    def test_request_ids_are_unique(self):
        """Each new_request_id() should be different."""
        from services.logger import new_request_id
        ids = {new_request_id() for _ in range(100)}
        assert len(ids) == 100


class TestJSONFormatter:
    """Tests for JSON log formatting."""

    def test_json_format(self):
        """Should produce valid JSON output."""
        from services.logger import JSONFormatter
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=10, msg="Hello world", args=(), exc_info=None
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "Hello world"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test"
        assert "timestamp" in parsed

    def test_json_includes_extra_fields(self):
        """Extra fields should appear in JSON output."""
        from services.logger import JSONFormatter
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=10, msg="Query processed", args=(), exc_info=None
        )
        record.user = "demo"
        record.action = "agent_hub_query"
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["user"] == "demo"
        assert parsed["action"] == "agent_hub_query"


class TestSafeCallDecorator:
    """Tests for the safe_call error handling decorator."""

    def test_returns_result_on_success(self):
        """Should pass through return value when no error."""
        from services.error_handler import safe_call

        @safe_call(fallback="fallback_value")
        def good_fn():
            return "success"

        assert good_fn() == "success"

    def test_returns_fallback_on_error(self):
        """Should return fallback value when function raises."""
        from services.error_handler import safe_call

        @safe_call(fallback="fallback_value")
        def bad_fn():
            raise ValueError("boom")

        assert bad_fn() == "fallback_value"

    def test_returns_fallback_fn_on_error(self):
        """Should call fallback_fn with the exception."""
        from services.error_handler import safe_call

        @safe_call(fallback_fn=lambda e: f"Error: {e}")
        def bad_fn():
            raise RuntimeError("something broke")

        result = bad_fn()
        assert "something broke" in result

    def test_returns_none_fallback_by_default(self):
        """Default fallback is None."""
        from services.error_handler import safe_call

        @safe_call()
        def bad_fn():
            raise Exception("oops")

        assert bad_fn() is None

    def test_preserves_function_name(self):
        """Decorated function should preserve its name."""
        from services.error_handler import safe_call

        @safe_call(fallback=None)
        def my_important_function():
            pass

        assert my_important_function.__name__ == "my_important_function"


class TestUserFriendlyErrors:
    """Tests for user_friendly_error message mapping."""

    def test_auth_error(self):
        """401 errors should mention API key."""
        from services.error_handler import user_friendly_error
        msg = user_friendly_error(Exception("Error code: 401 - authentication failed"))
        assert "API" in msg or "key" in msg.lower()

    def test_rate_limit_error(self):
        """429 errors should mention rate limit."""
        from services.error_handler import user_friendly_error
        msg = user_friendly_error(Exception("Error code: 429 rate_limit_exceeded"))
        assert "rate limit" in msg.lower()

    def test_timeout_error(self):
        """Timeout errors should be user-friendly."""
        from services.error_handler import user_friendly_error
        msg = user_friendly_error(Exception("Connection timeout after 30s"))
        assert "timed out" in msg.lower()

    def test_import_error(self):
        """Import errors should mention the module."""
        from services.error_handler import user_friendly_error
        msg = user_friendly_error(ImportError("No module named 'ollama'"))
        assert "ollama" in msg

    def test_generic_error(self):
        """Unknown errors should give a safe generic message."""
        from services.error_handler import user_friendly_error
        msg = user_friendly_error(Exception("some internal error xyz"))
        assert "try again" in msg.lower()
        assert "xyz" not in msg  # Should NOT leak internal details


class TestServiceStatus:
    """Tests for the ServiceStatus dataclass."""

    def test_all_healthy(self):
        """all_healthy should be True when all services are up."""
        from services.error_handler import ServiceStatus
        status = ServiceStatus(
            anthropic_api=True, ollama=True, chromadb=True, persistence=True
        )
        assert status.all_healthy is True

    def test_not_all_healthy(self):
        """all_healthy should be False if any service is down."""
        from services.error_handler import ServiceStatus
        status = ServiceStatus(
            anthropic_api=True, ollama=False, chromadb=True, persistence=True
        )
        assert status.all_healthy is False

    def test_summary_format(self):
        """summary should return a dict with per-service info."""
        from services.error_handler import ServiceStatus
        status = ServiceStatus(
            anthropic_api=True,
            ollama=False, ollama_error="Connection refused",
        )
        summary = status.summary
        assert summary["anthropic_api"]["healthy"] is True
        assert summary["ollama"]["healthy"] is False
        assert summary["ollama"]["error"] == "Connection refused"
