"""
Tests for Performance & Caching Service.

Covers:
- ResponseCache: put, get, TTL expiry, LRU eviction, stats
- Model router: simple vs complex query classification
- Cost tracker: recording calls, cache hits, summaries
- Context hashing: different uploads = different cache keys
- Timing decorator
"""

import pytest
import time


class TestResponseCache:
    """Tests for the LRU response cache."""

    def test_put_and_get(self):
        """Should store and retrieve a response."""
        from services.cache import ResponseCache
        cache = ResponseCache()
        cache.put("What drives brand recall?", "", "Logo placement is key.", tokens_used=500)
        result = cache.get("What drives brand recall?", "")
        assert result == "Logo placement is key."

    def test_cache_miss(self):
        """Unknown query should return None."""
        from services.cache import ResponseCache
        cache = ResponseCache()
        result = cache.get("never asked this before", "")
        assert result is None

    def test_case_insensitive(self):
        """Cache should normalize query case."""
        from services.cache import ResponseCache
        cache = ResponseCache()
        cache.put("What Are The Budget Rules?", "", "Budget tiers...", tokens_used=300)
        result = cache.get("what are the budget rules?", "")
        assert result == "Budget tiers..."

    def test_whitespace_normalized(self):
        """Cache should strip whitespace."""
        from services.cache import ResponseCache
        cache = ResponseCache()
        cache.put("  budget rules  ", "", "Tiers...", tokens_used=200)
        result = cache.get("budget rules", "")
        assert result == "Tiers..."

    def test_different_context_different_key(self):
        """Different upload context should produce different cache keys."""
        from services.cache import ResponseCache
        cache = ResponseCache()
        cache.put("analyze", "ctx_abc", "Response A", tokens_used=100)
        cache.put("analyze", "ctx_xyz", "Response B", tokens_used=100)
        assert cache.get("analyze", "ctx_abc") == "Response A"
        assert cache.get("analyze", "ctx_xyz") == "Response B"

    def test_lru_eviction(self):
        """Should evict oldest entries when max_size reached."""
        from services.cache import ResponseCache
        cache = ResponseCache(max_size=3)
        cache.put("q1", "", "r1", tokens_used=100)
        cache.put("q2", "", "r2", tokens_used=100)
        cache.put("q3", "", "r3", tokens_used=100)
        cache.put("q4", "", "r4", tokens_used=100)  # q1 should be evicted
        assert cache.get("q1", "") is None
        assert cache.get("q4", "") == "r4"

    def test_ttl_expiry(self):
        """Entries should expire after TTL."""
        from services.cache import ResponseCache
        cache = ResponseCache(ttl_seconds=1)
        cache.put("query", "", "response", tokens_used=100)
        assert cache.get("query", "") == "response"
        time.sleep(1.1)
        assert cache.get("query", "") is None

    def test_cache_stats(self):
        """Stats should track hits and misses."""
        from services.cache import ResponseCache
        cache = ResponseCache()
        cache.put("q1", "", "r1", tokens_used=500)
        cache.get("q1", "")     # hit
        cache.get("q1", "")     # hit
        cache.get("q2", "")     # miss
        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["entries"] == 1

    def test_tokens_saved_tracking(self):
        """Should track tokens saved by cache hits."""
        from services.cache import ResponseCache
        cache = ResponseCache()
        cache.put("q1", "", "r1", tokens_used=1000)
        cache.get("q1", "")  # hit — saves 1000 tokens
        cache.get("q1", "")  # hit — saves 1000 tokens
        stats = cache.get_stats()
        assert stats["tokens_saved"] == 2000

    def test_clear(self):
        """Should clear all entries."""
        from services.cache import ResponseCache
        cache = ResponseCache()
        cache.put("q1", "", "r1", tokens_used=100)
        cache.put("q2", "", "r2", tokens_used=100)
        cache.clear()
        assert cache.get("q1", "") is None
        assert cache.get_stats()["entries"] == 0


class TestModelRouter:
    """Tests for model selection based on query complexity."""

    def test_simple_query_uses_haiku(self):
        """Simple factual questions should route to Haiku."""
        from services.cache import select_model
        model = select_model("What are the budget rules?")
        assert "haiku" in model.lower()

    def test_complex_query_uses_sonnet(self):
        """Complex analysis queries should route to Sonnet."""
        from services.cache import select_model
        model = select_model("Analyze these videos and recommend a strategy")
        assert "sonnet" in model.lower() or "claude-sonnet" in model

    def test_generate_plan_is_complex(self):
        """'generate plan' should be classified as complex."""
        from services.cache import select_model
        model = select_model("Generate a test plan for my campaign")
        assert "haiku" not in model.lower()

    def test_cost_query_is_simple(self):
        """'how much does it cost' should be classified as simple."""
        from services.cache import select_model
        model = select_model("How much does video testing cost?")
        assert "haiku" in model.lower()

    def test_ambiguous_defaults_to_sonnet(self):
        """Ambiguous queries should default to Sonnet."""
        from services.cache import select_model
        model = select_model("Tell me about the creative performance landscape")
        assert "sonnet" in model.lower() or "claude-sonnet" in model


class TestCostTracker:
    """Tests for cost tracking."""

    def test_record_call(self):
        """Should track API call tokens."""
        from services.cache import CostTracker
        tracker = CostTracker()
        tracker.record_call(input_tokens=1000, output_tokens=500)
        assert tracker.total_input_tokens == 1000
        assert tracker.total_output_tokens == 500
        assert tracker.calls == 1

    def test_multiple_calls(self):
        """Should accumulate across multiple calls."""
        from services.cache import CostTracker
        tracker = CostTracker()
        tracker.record_call(1000, 500)
        tracker.record_call(2000, 800)
        assert tracker.total_input_tokens == 3000
        assert tracker.total_output_tokens == 1300
        assert tracker.calls == 2

    def test_estimated_cost(self):
        """Should estimate cost based on token pricing."""
        from services.cache import CostTracker
        tracker = CostTracker()
        tracker.record_call(1_000_000, 100_000)  # 1M input + 100K output
        # Cost = (1M/1M * $3) + (100K/1M * $15) = $3 + $1.50 = $4.50
        assert abs(tracker.estimated_cost - 4.50) < 0.01

    def test_record_cache_hit(self):
        """Should track cache hit tokens."""
        from services.cache import CostTracker
        tracker = CostTracker()
        tracker.record_cache_hit(tokens_saved=2000)
        assert tracker.total_cached_tokens == 2000
        assert tracker.cached_calls == 1

    def test_summary(self):
        """Summary should have all required fields."""
        from services.cache import CostTracker
        tracker = CostTracker()
        tracker.record_call(1000, 500)
        summary = tracker.get_summary()
        assert "api_calls" in summary
        assert "cached_calls" in summary
        assert "total_tokens" in summary
        assert "estimated_cost" in summary
        assert "estimated_savings" in summary


class TestContextHash:
    """Tests for upload context hashing."""

    def test_empty_context(self):
        """No uploads should return empty hash."""
        from services.cache import hash_upload_context
        h = hash_upload_context({"videos": [], "media_plan": None})
        assert h == ""

    def test_videos_produce_hash(self):
        """Context with videos should produce a non-empty hash."""
        from services.cache import hash_upload_context
        ctx = {"videos": [{"name": "hero.mp4"}], "media_plan": None}
        h = hash_upload_context(ctx)
        assert len(h) == 12

    def test_different_videos_different_hash(self):
        """Different video sets should produce different hashes."""
        from services.cache import hash_upload_context
        h1 = hash_upload_context({"videos": [{"name": "hero.mp4"}], "media_plan": None})
        h2 = hash_upload_context({"videos": [{"name": "other.mp4"}], "media_plan": None})
        assert h1 != h2

    def test_same_videos_same_hash(self):
        """Same video set should produce same hash."""
        from services.cache import hash_upload_context
        ctx = {"videos": [{"name": "hero.mp4"}, {"name": "intro.mp4"}], "media_plan": None}
        h1 = hash_upload_context(ctx)
        h2 = hash_upload_context(ctx)
        assert h1 == h2


class TestTimedDecorator:
    """Tests for the @timed performance decorator."""

    def test_timed_returns_result(self):
        """Decorated function should still return its value."""
        from services.cache import timed

        @timed
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_timed_preserves_name(self):
        """Decorated function should keep its name."""
        from services.cache import timed

        @timed
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_timed_propagates_exception(self):
        """Exceptions should still raise after being logged."""
        from services.cache import timed

        @timed
        def bad_fn():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            bad_fn()
