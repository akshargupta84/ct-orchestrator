"""
Logging Service — Structured logging for CT Orchestrator.

Provides:
- JSON-formatted structured logs for machine parsing
- Request ID tracking across agent calls
- Log levels: DEBUG (dev), INFO (production), WARNING, ERROR
- File rotation with size limits
- Console + file output

Usage:
    from services.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing video", extra={"video_name": "hero_30s.mp4", "user": "demo"})
"""

import logging
import logging.handlers
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from contextvars import ContextVar
from typing import Optional

# Context variable for request ID tracking across async/threaded calls
_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_request_id() -> str:
    """Get or create a request ID for the current context."""
    rid = _request_id.get()
    if rid is None:
        rid = uuid.uuid4().hex[:12]
        _request_id.set(rid)
    return rid


def set_request_id(rid: str):
    """Set a specific request ID (e.g., from an incoming request)."""
    _request_id.set(rid)


def new_request_id() -> str:
    """Generate and set a new request ID. Returns the new ID."""
    rid = uuid.uuid4().hex[:12]
    _request_id.set(rid)
    return rid


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": get_request_id(),
        }

        # Add module/function info
        if record.funcName and record.funcName != "<module>":
            log_entry["function"] = f"{record.module}.{record.funcName}"
            log_entry["line"] = record.lineno

        # Add any extra fields passed via logger.info(..., extra={...})
        for key in ("user", "action", "page", "video_name", "session_id",
                     "query", "tokens", "duration_ms", "error_type", "agent"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class SimpleFormatter(logging.Formatter):
    """Human-readable formatter for console output."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        rid = get_request_id()
        rid_str = f"[{rid}] " if rid else ""

        # Build extra fields string
        extras = []
        for key in ("user", "action", "page", "video_name", "agent"):
            val = getattr(record, key, None)
            if val is not None:
                extras.append(f"{key}={val}")
        extra_str = f" | {', '.join(extras)}" if extras else ""

        return (
            f"{color}{record.levelname:8s}{self.RESET} "
            f"{rid_str}{record.name}: {record.getMessage()}{extra_str}"
        )


def setup_logging(
    level: str = None,
    log_dir: str = None,
    json_logs: bool = None,
):
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to env var LOG_LEVEL or INFO.
        log_dir: Directory for log files. Defaults to 'data/logs/'.
        json_logs: Use JSON format. Defaults to True in production (DEMO_MODE=true), False locally.
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    if json_logs is None:
        json_logs = os.getenv("DEMO_MODE", "true").lower() == "true"

    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "logs")

    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Root logger
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))

    # Remove existing handlers
    root.handlers.clear()

    # Console handler — always human-readable
    console = logging.StreamHandler()
    console.setFormatter(SimpleFormatter())
    console.setLevel(getattr(logging, level, logging.INFO))
    root.addHandler(console)

    # File handler — JSON for structured analysis
    log_file = os.path.join(log_dir, "ct_orchestrator.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB per file
        backupCount=3,              # Keep 3 rotated files
    )
    file_handler.setFormatter(JSONFormatter())
    file_handler.setLevel(logging.DEBUG)  # File always captures everything
    root.addHandler(file_handler)

    # Suppress noisy libraries
    for lib in ("urllib3", "httpx", "httpcore", "watchdog", "fsevents"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    root.info("Logging initialized", extra={"level": level, "log_dir": log_dir})


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module.

    Usage:
        logger = get_logger(__name__)
        logger.info("Something happened", extra={"user": "demo"})
    """
    return logging.getLogger(name)
