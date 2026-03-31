"""
Tests for the Rules Engine and CTRules model.

Covers:
- Budget tier matching (all 4 tiers + edge cases)
- Cost calculation (standard + expedited)
- Plan validation (valid plans, over-limit, over-budget)
- Limit lookups
- Rules engine wrapper
"""

import pytest
from models.rules import CTRules, BudgetTier, AssetCost, Turnaround, MinimumRequirements, DEFAULT_CT_RULES


# =============================================================================
# Budget Tier Matching
# =============================================================================

class TestBudgetTiers:
    """Tests for budget tier selection logic."""

    def test_tier_1_low_budget(self, default_rules):
        """$1M should match tier 1 (0-5M)."""
        tier = default_rules.get_tier_for_budget(1_000_000)
        assert tier is not None
        assert tier.video_limit == 2
        assert tier.display_limit == 5

    def test_tier_2_mid_budget(self, default_rules):
        """$10M should match tier 2 (5M-35M)."""
        tier = default_rules.get_tier_for_budget(10_000_000)
        assert tier is not None
        assert tier.video_limit == 8
        assert tier.display_limit == 15

    def test_tier_3_high_budget(self, default_rules):
        """$50M should match tier 3 (35M-100M)."""
        tier = default_rules.get_tier_for_budget(50_000_000)
        assert tier is not None
        assert tier.video_limit == 15
        assert tier.display_limit == 30

    def test_tier_4_enterprise_budget(self, default_rules):
        """$200M should match tier 4 (100M+)."""
        tier = default_rules.get_tier_for_budget(200_000_000)
        assert tier is not None
        assert tier.video_limit == 25
        assert tier.display_limit == 50

    def test_tier_boundary_exact_5m(self, default_rules):
        """$5M exactly should match tier 2 (5M is inclusive for tier 2)."""
        tier = default_rules.get_tier_for_budget(5_000_000)
        assert tier is not None
        assert tier.video_limit == 8

    def test_tier_boundary_just_below_5m(self, default_rules):
        """$4,999,999 should match tier 1."""
        tier = default_rules.get_tier_for_budget(4_999_999)
        assert tier is not None
        assert tier.video_limit == 2

    def test_tier_zero_budget(self, default_rules):
        """$0 should match tier 1."""
        tier = default_rules.get_tier_for_budget(0)
        assert tier is not None
        assert tier.video_limit == 2

    def test_tier_boundary_100m(self, default_rules):
        """$100M exactly should match tier 4."""
        tier = default_rules.get_tier_for_budget(100_000_000)
        assert tier is not None
        assert tier.video_limit == 25

    def test_budget_tier_matches_method(self):
        """BudgetTier.matches_budget() should work correctly."""
        tier = BudgetTier(min_budget=5_000_000, max_budget=35_000_000, video_limit=8, display_limit=15)
        assert tier.matches_budget(5_000_000) is True
        assert tier.matches_budget(20_000_000) is True
        assert tier.matches_budget(34_999_999) is True
        assert tier.matches_budget(35_000_000) is False
        assert tier.matches_budget(4_999_999) is False


# =============================================================================
# Limit Lookups
# =============================================================================

class TestLimitLookups:
    """Tests for get_limits() method."""

    def test_limits_for_valid_budget(self, default_rules):
        """Should return correct limits dict."""
        limits = default_rules.get_limits(10_000_000)
        assert limits["video_limit"] == 8
        assert limits["display_limit"] == 15
        assert "audio_limit" in limits

    def test_limits_for_negative_budget(self, default_rules):
        """Negative budget should return zero limits (no tier matches)."""
        limits = default_rules.get_limits(-1)
        assert limits["video_limit"] == 0
        assert limits["display_limit"] == 0


# =============================================================================
# Cost Calculation
# =============================================================================

class TestCostCalculation:
    """Tests for cost calculation logic."""

    def test_video_only_cost(self, default_rules):
        """3 videos at $5,000 each = $15,000."""
        cost = default_rules.calculate_cost(video_count=3, display_count=0)
        assert cost == 15_000

    def test_display_only_cost(self, default_rules):
        """5 displays at $3,000 each = $15,000."""
        cost = default_rules.calculate_cost(video_count=0, display_count=5)
        assert cost == 15_000

    def test_mixed_cost(self, default_rules):
        """2 videos + 3 displays = $10,000 + $9,000 = $19,000."""
        cost = default_rules.calculate_cost(video_count=2, display_count=3)
        assert cost == 19_000

    def test_expedited_cost(self, default_rules):
        """Expedited adds 50% fee."""
        standard = default_rules.calculate_cost(video_count=2, display_count=0)
        expedited = default_rules.calculate_cost(video_count=2, display_count=0, expedited=True)
        assert expedited == standard * 1.5

    def test_zero_assets_cost(self, default_rules):
        """No assets = $0."""
        cost = default_rules.calculate_cost(video_count=0, display_count=0)
        assert cost == 0

    def test_audio_cost(self, default_rules):
        """Audio at $2,500 each."""
        cost = default_rules.calculate_cost(video_count=0, display_count=0, audio_count=4)
        assert cost == 10_000


# =============================================================================
# Plan Validation
# =============================================================================

class TestPlanValidation:
    """Tests for validate_plan() method."""

    def test_valid_plan(self, default_rules):
        """A plan within limits should be valid."""
        result = default_rules.validate_plan(
            budget=10_000_000,  # Tier 2: 8 videos, 15 displays
            video_count=4,
            display_count=10
        )
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert result["limits"]["video_limit"] == 8

    def test_plan_exceeds_video_limit(self, default_rules):
        """Exceeding video limit should produce an error."""
        result = default_rules.validate_plan(
            budget=10_000_000,  # Tier 2: max 8 videos
            video_count=12,
            display_count=0
        )
        assert result["valid"] is False
        assert any("Video count" in e for e in result["errors"])

    def test_plan_exceeds_display_limit(self, default_rules):
        """Exceeding display limit should produce an error."""
        result = default_rules.validate_plan(
            budget=1_000_000,  # Tier 1: max 5 displays
            video_count=0,
            display_count=10
        )
        assert result["valid"] is False
        assert any("Display count" in e for e in result["errors"])

    def test_plan_exceeds_testing_budget(self, default_rules):
        """Plan cost exceeding 1.5% of budget should produce an error."""
        # Budget $1M → testing budget = $15,000
        # 4 videos = $20,000 > $15,000
        result = default_rules.validate_plan(
            budget=1_000_000,
            video_count=2,  # Within tier 1 limit (2)
            display_count=5  # Within tier 1 limit (5)
        )
        # 2*5000 + 5*3000 = 25,000 > 15,000
        assert result["valid"] is False
        assert any("cost" in e.lower() for e in result["errors"])

    def test_plan_requires_video_for_large_budget(self, default_rules):
        """Campaigns over $2M must test at least 1 video."""
        result = default_rules.validate_plan(
            budget=5_000_000,
            video_count=0,
            display_count=3
        )
        assert result["valid"] is False
        assert any("must test at least 1 video" in e for e in result["errors"])

    def test_plan_small_budget_no_video_ok(self, default_rules):
        """Small campaigns don't need video testing."""
        result = default_rules.validate_plan(
            budget=1_000_000,
            video_count=0,
            display_count=2
        )
        # Cost: 2*3000 = 6000 < 15000 budget → valid
        assert result["valid"] is True

    def test_validation_returns_cost_and_budget(self, default_rules):
        """Validation result should include estimated cost and testing budget."""
        result = default_rules.validate_plan(budget=10_000_000, video_count=1, display_count=0)
        assert "estimated_cost" in result
        assert "testing_budget" in result
        assert result["estimated_cost"] == 5_000
        assert result["testing_budget"] == 10_000_000 * 0.015

    def test_multiple_errors_returned(self, default_rules):
        """A badly invalid plan should return multiple errors."""
        result = default_rules.validate_plan(
            budget=1_000_000,  # Tier 1: 2 videos, 5 displays
            video_count=10,    # Way over limit
            display_count=20   # Way over limit
        )
        assert result["valid"] is False
        assert len(result["errors"]) >= 2  # At least video + display limit errors


# =============================================================================
# Rules Engine Service (wrapper)
# =============================================================================

class TestRulesEngine:
    """Tests for the RulesEngine service wrapper."""

    def test_default_rules_loaded(self):
        """RulesEngine without PDF should use default rules."""
        from services.rules_engine import RulesEngine
        engine = RulesEngine()
        rules = engine.rules
        assert rules.version == "1.0"
        assert len(rules.budget_tiers) == 4

    def test_get_limits_for_budget(self):
        """RulesEngine.get_limits_for_budget() should delegate correctly."""
        from services.rules_engine import RulesEngine
        engine = RulesEngine()
        limits = engine.get_limits_for_budget(10_000_000)
        assert limits["video_limit"] == 8

    def test_calculate_cost(self):
        """RulesEngine.calculate_cost() should delegate correctly."""
        from services.rules_engine import RulesEngine
        engine = RulesEngine()
        cost = engine.calculate_cost(video_count=2, display_count=3)
        assert cost == 19_000

    def test_validate_plan(self):
        """RulesEngine.validate_plan() should delegate correctly."""
        from services.rules_engine import RulesEngine
        engine = RulesEngine()
        result = engine.validate_plan(budget=10_000_000, video_count=4, display_count=5)
        assert result["valid"] is True

    def test_singleton_pattern(self):
        """get_rules_engine() should return consistent instance."""
        from services.rules_engine import get_rules_engine
        engine1 = get_rules_engine()
        engine2 = get_rules_engine()
        assert engine1.rules.version == engine2.rules.version
