"""
Error Handling & Graceful Degradation for CT Orchestrator.

Provides:
- Service health checks (Anthropic API, Ollama, ChromaDB)
- Graceful fallback strategies when services are unavailable
- User-friendly error messages (never raw tracebacks)
- Decorators for common error patterns

Usage:
    from services.error_handler import safe_call, check_services, ServiceStatus

    # Decorator for graceful error handling
    @safe_call(fallback="Service unavailable", notify_user=True)
    def risky_operation():
        ...

    # Check all services
    status = check_services()
    if not status.anthropic_api:
        # Fall back to keyword responses
"""

import os
import functools
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from services.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ServiceStatus:
    """Health status of all external services."""
    anthropic_api: bool = False
    anthropic_error: str = ""
    ollama: bool = False
    ollama_error: str = ""
    chromadb: bool = False
    chromadb_error: str = ""
    persistence: bool = False
    persistence_error: str = ""

    @property
    def all_healthy(self) -> bool:
        return self.anthropic_api and self.ollama and self.chromadb and self.persistence

    @property
    def summary(self) -> dict:
        return {
            "anthropic_api": {"healthy": self.anthropic_api, "error": self.anthropic_error},
            "ollama": {"healthy": self.ollama, "error": self.ollama_error},
            "chromadb": {"healthy": self.chromadb, "error": self.chromadb_error},
            "persistence": {"healthy": self.persistence, "error": self.persistence_error},
        }


def check_services() -> ServiceStatus:
    """
    Check health of all external services.
    Returns a ServiceStatus with per-service health info.
    """
    status = ServiceStatus()

    # Anthropic API
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if api_key and len(api_key) > 20:
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)
            # Minimal API call to verify
            client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=5,
                messages=[{"role": "user", "content": "hi"}],
            )
            status.anthropic_api = True
        else:
            status.anthropic_error = "API key not configured or too short"
    except ImportError:
        status.anthropic_error = "anthropic package not installed"
    except Exception as e:
        status.anthropic_error = str(e)[:200]

    # Ollama
    try:
        import urllib.request
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        req = urllib.request.Request(f"{ollama_host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                status.ollama = True
    except Exception as e:
        status.ollama_error = str(e)[:200]

    # ChromaDB
    try:
        from services.vector_store import get_vector_store
        vs = get_vector_store()
        vs.get_stats()
        status.chromadb = True
    except Exception as e:
        status.chromadb_error = str(e)[:200]

    # Persistence (SQLite)
    try:
        from services.usage_tracker import get_tracker
        tracker = get_tracker()
        tracker.get_usage_stats()
        status.persistence = True
    except Exception as e:
        status.persistence_error = str(e)[:200]

    logger.info(
        "Service health check complete",
        extra={"action": "health_check"},
    )
    return status


def safe_call(
    fallback: Any = None,
    fallback_fn: Optional[Callable] = None,
    notify_user: bool = False,
    log_level: str = "error",
):
    """
    Decorator for graceful error handling.

    Args:
        fallback: Static fallback value to return on error.
        fallback_fn: Function to call for fallback value (receives the exception).
        notify_user: If True, shows a Streamlit warning (requires st to be importable).
        log_level: Log level for the error ("error", "warning", "info").

    Usage:
        @safe_call(fallback="Default response")
        def call_api(query):
            return client.messages.create(...)

        @safe_call(fallback_fn=lambda e: f"Error: {e}")
        def process_video(path):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the error
                log_fn = getattr(logger, log_level, logger.error)
                log_fn(
                    f"{func.__name__} failed: {e}",
                    extra={"error_type": type(e).__name__, "function": func.__name__},
                    exc_info=True,
                )

                # Notify user via Streamlit if requested
                if notify_user:
                    try:
                        import streamlit as st
                        st.warning(f"⚠️ {func.__name__} encountered an issue. Using fallback.")
                    except Exception:
                        pass

                # Return fallback
                if fallback_fn is not None:
                    return fallback_fn(e)
                return fallback
        return wrapper
    return decorator


def user_friendly_error(error: Exception) -> str:
    """
    Convert a raw exception into a user-friendly message.
    Never exposes stack traces or internal details.
    """
    error_type = type(error).__name__
    error_str = str(error)

    # Anthropic API errors
    if "401" in error_str or "authentication" in error_str.lower():
        return "API authentication failed. Please check your API key in Settings."
    if "429" in error_str or "rate_limit" in error_str.lower():
        return "API rate limit reached. Please wait a moment and try again."
    if "500" in error_str or "502" in error_str or "503" in error_str:
        return "The AI service is temporarily unavailable. Please try again shortly."
    if "timeout" in error_str.lower():
        return "The request timed out. Try a simpler question or try again."

    # Import errors (missing dependencies)
    if error_type == "ImportError" or error_type == "ModuleNotFoundError":
        module = error_str.split("'")[1] if "'" in error_str else "unknown"
        return f"Required module '{module}' is not installed. Run locally with full dependencies."

    # File errors
    if error_type in ("FileNotFoundError", "PermissionError"):
        return "A file operation failed. Please check file permissions and paths."

    # Database errors
    if "sqlite" in error_str.lower() or "database" in error_str.lower():
        return "Database operation failed. Try refreshing the page."

    # Generic fallback
    logger.error(f"Unhandled error: {error_type}: {error_str}", exc_info=True)
    return "Something went wrong. Please try again or contact support."
