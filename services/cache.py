"""
Performance & Caching Service for CT Orchestrator.

Provides:
- LRU response cache for repeated/similar queries (reduces API costs)
- Model router: Haiku for simple queries, Sonnet for complex (reduces costs)
- Token usage tracking with cost estimation
- Timing decorator for performance measurement
- Cache stats for admin dashboard

Usage:
    from services.cache import get_cache, timed, select_model

    cache = get_cache()

    # Check cache before calling API
    cached = cache.get(query, context_hash)
    if cached:
        return cached

    # Route to cheaper model for simple queries
    model = select_model(query)

    # Call API...
    cache.put(query, context_hash, response, tokens_used)

    # Measure function timing
    @timed
    def expensive_operation():
        ...
"""

import hashlib
import time
import functools
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

try:
    from services.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# Singleton
_cache_instance = None


def get_cache() -> "ResponseCache":
    """Get or create the singleton ResponseCache."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = ResponseCache()
    return _cache_instance


@dataclass
class CacheEntry:
    """A single cached response."""
    query: str
    response: str
    tokens_used: int
    created_at: str
    hit_count: int = 0


@dataclass
class CacheStats:
    """Aggregated cache statistics."""
    total_entries: int = 0
    hits: int = 0
    misses: int = 0
    tokens_saved: int = 0
    cost_saved: float = 0.0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class ResponseCache:
    """
    LRU cache for API responses.

    Caches responses keyed by (query_hash, context_hash) to avoid
    redundant API calls for repeated or similar questions.

    - Max 200 entries (configurable)
    - Entries expire after 1 hour
    - Context-aware: different file uploads = different cache keys
    """

    def __init__(self, max_size: int = 200, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats(max_size=max_size)

    def _make_key(self, query: str, context_hash: str = "") -> str:
        """Create a cache key from query + context."""
        normalized = query.strip().lower()
        raw = f"{normalized}|{context_hash}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, query: str, context_hash: str = "") -> Optional[str]:
        """
        Look up a cached response.

        Args:
            query: The user's question.
            context_hash: Hash of upload context (empty if no uploads).

        Returns:
            Cached response string, or None if not found/expired.
        """
        key = self._make_key(query, context_hash)
        entry = self._cache.get(key)

        if entry is None:
            self._stats.misses += 1
            return None

        # Check TTL
        created = datetime.fromisoformat(entry.created_at)
        age = (datetime.now() - created).total_seconds()
        if age > self.ttl_seconds:
            del self._cache[key]
            self._stats.misses += 1
            logger.debug(f"Cache expired for query", extra={"action": "cache_expired"})
            return None

        # Cache hit
        entry.hit_count += 1
        self._stats.hits += 1
        self._stats.tokens_saved += entry.tokens_used
        # Estimate cost saved: ~$9/M tokens average (Sonnet)
        self._stats.cost_saved += (entry.tokens_used / 1_000_000) * 9

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        logger.debug(f"Cache hit", extra={"action": "cache_hit", "tokens": entry.tokens_used})
        return entry.response

    def put(self, query: str, context_hash: str, response: str, tokens_used: int = 0):
        """
        Store a response in the cache.

        Args:
            query: The user's question.
            context_hash: Hash of upload context.
            response: The API response to cache.
            tokens_used: Tokens consumed by this response.
        """
        key = self._make_key(query, context_hash)

        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        self._cache[key] = CacheEntry(
            query=query[:200],
            response=response,
            tokens_used=tokens_used,
            created_at=datetime.now().isoformat(),
        )
        self._stats.total_entries = len(self._cache)
        logger.debug(f"Cached response", extra={"action": "cache_put", "tokens": tokens_used})

    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()
        self._stats.total_entries = 0
        logger.info("Cache cleared", extra={"action": "cache_clear"})

    def get_stats(self) -> dict:
        """Get cache statistics for admin dashboard."""
        return {
            "entries": len(self._cache),
            "max_size": self.max_size,
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "hit_rate": f"{self._stats.hit_rate:.1%}",
            "tokens_saved": self._stats.tokens_saved,
            "cost_saved": f"${self._stats.cost_saved:.4f}",
        }


# =============================================================================
# Model Router — Use cheaper models for simple queries
# =============================================================================

# Keywords that indicate a simple/factual question (use Haiku)
_SIMPLE_PATTERNS = {
    "what is", "what are", "how much", "how many", "show me",
    "list", "define", "rules", "budget", "cost", "tier",
    "limit", "price", "turnaround", "timeline",
}

# Keywords that indicate complex analysis (use Sonnet)
_COMPLEX_PATTERNS = {
    "analyze", "compare", "recommend", "strategy", "optimize",
    "why did", "what if", "generate plan", "test plan",
    "explain the relationship", "drivers", "correlation",
    "tell me more", "elaborate", "deep dive",
}


def select_model(query: str) -> str:
    """
    Select the best model for a query based on complexity.

    Returns the model string to use in the API call.
    Simple queries → Haiku (cheaper, faster)
    Complex queries → Sonnet (smarter, more expensive)
    """
    import os
    query_lower = query.strip().lower()

    # Check for complex patterns first (higher priority)
    if any(p in query_lower for p in _COMPLEX_PATTERNS):
        return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    # Check for simple patterns
    if any(p in query_lower for p in _SIMPLE_PATTERNS):
        return os.getenv("ANTHROPIC_MODEL_FAST", "claude-haiku-4-5-20251001")

    # Default to standard model
    return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")


# =============================================================================
# Cost Tracking
# =============================================================================

@dataclass
class CostTracker:
    """Track API token usage and costs per session."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0  # Tokens saved by cache
    calls: int = 0
    cached_calls: int = 0

    # Pricing (per million tokens, approximate)
    SONNET_INPUT_PRICE: float = 3.0
    SONNET_OUTPUT_PRICE: float = 15.0
    HAIKU_INPUT_PRICE: float = 0.80
    HAIKU_OUTPUT_PRICE: float = 4.0

    def record_call(self, input_tokens: int, output_tokens: int, model: str = "sonnet"):
        """Record an API call."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.calls += 1

    def record_cache_hit(self, tokens_saved: int):
        """Record a cache hit."""
        self.total_cached_tokens += tokens_saved
        self.cached_calls += 1

    @property
    def estimated_cost(self) -> float:
        """Estimated total cost in USD."""
        input_cost = (self.total_input_tokens / 1_000_000) * self.SONNET_INPUT_PRICE
        output_cost = (self.total_output_tokens / 1_000_000) * self.SONNET_OUTPUT_PRICE
        return input_cost + output_cost

    @property
    def estimated_savings(self) -> float:
        """Estimated savings from caching."""
        return (self.total_cached_tokens / 1_000_000) * 9  # Average price

    def get_summary(self) -> dict:
        """Get cost summary for admin dashboard."""
        return {
            "api_calls": self.calls,
            "cached_calls": self.cached_calls,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "estimated_cost": f"${self.estimated_cost:.4f}",
            "tokens_saved_by_cache": self.total_cached_tokens,
            "estimated_savings": f"${self.estimated_savings:.4f}",
        }


# Global cost tracker
_cost_tracker = CostTracker()


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker."""
    return _cost_tracker


# =============================================================================
# Timing Decorator
# =============================================================================

def timed(func):
    """
    Decorator to measure and log function execution time.

    Usage:
        @timed
        def slow_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                f"{func.__name__} completed in {elapsed_ms:.0f}ms",
                extra={"action": "timing", "duration_ms": round(elapsed_ms)},
            )
            return result
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(
                f"{func.__name__} failed after {elapsed_ms:.0f}ms: {e}",
                extra={"action": "timing_error", "duration_ms": round(elapsed_ms)},
            )
            raise
    return wrapper


# =============================================================================
# Context Hash Helper
# =============================================================================

def hash_upload_context(upload_context: dict) -> str:
    """
    Create a hash of the upload context for cache keying.
    Different uploads = different cache keys.
    """
    parts = []
    if upload_context.get("videos"):
        parts.extend(v["name"] for v in upload_context["videos"])
    if upload_context.get("media_plan") and "error" not in upload_context.get("media_plan", {}):
        mp = upload_context["media_plan"]
        parts.append(mp.get("filename", ""))
        parts.append(str(mp.get("rows", 0)))
    raw = "|".join(sorted(parts))
    return hashlib.md5(raw.encode()).hexdigest()[:12] if parts else ""
