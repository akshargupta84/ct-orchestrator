"""
CT Rules model and parser.

This module handles the master data for creative testing rules.
The rules are stored in a PDF and used in two ways:
1. Structured extraction for rule enforcement (budget tiers, costs, limits)
2. Vector embedding for Q&A about rules
"""

from typing import Optional
from pydantic import BaseModel, Field


class BudgetTier(BaseModel):
    """A budget tier defining testing limits."""
    min_budget: float = Field(..., description="Minimum campaign budget (inclusive)")
    max_budget: float = Field(..., description="Maximum campaign budget (exclusive)")
    video_limit: int = Field(..., description="Max video assets that can be tested")
    display_limit: int = Field(..., description="Max display assets that can be tested")
    audio_limit: int = Field(default=0, description="Max audio assets that can be tested")
    
    def matches_budget(self, budget: float) -> bool:
        """Check if a budget falls within this tier."""
        return self.min_budget <= budget < self.max_budget


class AssetCost(BaseModel):
    """Cost structure for testing different asset types."""
    video_cost: float = Field(..., description="Cost per video test in USD")
    display_cost: float = Field(..., description="Cost per display test in USD")
    audio_cost: float = Field(default=0, description="Cost per audio test in USD")
    expedited_fee_pct: float = Field(default=0.5, description="Additional fee for expedited testing (0.5 = 50%)")


class Turnaround(BaseModel):
    """Turnaround times for testing."""
    video_standard_days: int = Field(default=14)
    video_expedited_days: int = Field(default=7)
    display_standard_days: int = Field(default=10)
    display_expedited_days: int = Field(default=5)
    audio_standard_days: int = Field(default=10)
    audio_expedited_days: int = Field(default=5)


class MinimumRequirements(BaseModel):
    """Minimum testing requirements based on campaign characteristics."""
    min_budget_for_required_video_test: float = Field(
        default=2_000_000, 
        description="Campaigns over this budget must test at least 1 video"
    )
    tv_spend_requires_tv_test: bool = Field(
        default=True,
        description="Campaigns with TV spend must test TV creative"
    )


class CTRules(BaseModel):
    """
    Complete Creative Testing Rules.
    
    This is parsed from the master CT Rules PDF and used to:
    1. Validate test plans against budget tiers
    2. Calculate costs
    3. Determine timelines
    4. Enforce minimum requirements
    """
    version: str = Field(default="1.0")
    effective_date: str = Field(default="2025-01-01")
    
    # Budget allocation
    budget_allocation_pct: float = Field(
        default=0.015, 
        description="Percentage of campaign budget allocated to CT (1.5%)"
    )
    
    # Budget tiers
    budget_tiers: list[BudgetTier] = Field(default_factory=list)
    
    # Costs
    costs: AssetCost = Field(default_factory=lambda: AssetCost(
        video_cost=5000,
        display_cost=3000,
        audio_cost=2500
    ))
    
    # Turnaround
    turnaround: Turnaround = Field(default_factory=Turnaround)
    
    # Minimum requirements
    minimum_requirements: MinimumRequirements = Field(default_factory=MinimumRequirements)
    
    # Raw text for RAG
    raw_text: Optional[str] = Field(default=None, exclude=True)
    
    def get_tier_for_budget(self, budget: float) -> Optional[BudgetTier]:
        """Get the budget tier that applies to a given budget."""
        for tier in self.budget_tiers:
            if tier.matches_budget(budget):
                return tier
        return None
    
    def get_limits(self, budget: float) -> dict:
        """Get video and display limits for a budget."""
        tier = self.get_tier_for_budget(budget)
        if tier:
            return {
                "video_limit": tier.video_limit,
                "display_limit": tier.display_limit,
                "audio_limit": tier.audio_limit
            }
        # Default to most restrictive if no tier matches
        return {"video_limit": 0, "display_limit": 0, "audio_limit": 0}
    
    def calculate_cost(self, video_count: int, display_count: int, audio_count: int = 0, expedited: bool = False) -> float:
        """Calculate total testing cost."""
        base_cost = (
            video_count * self.costs.video_cost +
            display_count * self.costs.display_cost +
            audio_count * self.costs.audio_cost
        )
        if expedited:
            base_cost *= (1 + self.costs.expedited_fee_pct)
        return base_cost
    
    def validate_plan(self, budget: float, video_count: int, display_count: int) -> dict:
        """
        Validate a test plan against the rules.
        
        Returns:
            dict with 'valid' boolean and 'errors' list
        """
        errors = []
        limits = self.get_limits(budget)
        
        # Check video limit
        if video_count > limits["video_limit"]:
            errors.append(
                f"Video count ({video_count}) exceeds limit ({limits['video_limit']}) "
                f"for budget ${budget:,.0f}"
            )
        
        # Check display limit
        if display_count > limits["display_limit"]:
            errors.append(
                f"Display count ({display_count}) exceeds limit ({limits['display_limit']}) "
                f"for budget ${budget:,.0f}"
            )
        
        # Check cost vs budget
        total_cost = self.calculate_cost(video_count, display_count)
        testing_budget = budget * self.budget_allocation_pct
        if total_cost > testing_budget:
            errors.append(
                f"Estimated cost (${total_cost:,.0f}) exceeds testing budget "
                f"(${testing_budget:,.0f})"
            )
        
        # Check minimum requirements
        if budget > self.minimum_requirements.min_budget_for_required_video_test and video_count == 0:
            errors.append(
                f"Campaigns over ${self.minimum_requirements.min_budget_for_required_video_test:,.0f} "
                f"must test at least 1 video"
            )
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "limits": limits,
            "testing_budget": testing_budget,
            "estimated_cost": total_cost
        }


# Default rules for development/testing
DEFAULT_CT_RULES = CTRules(
    version="1.0",
    effective_date="2025-01-01",
    budget_tiers=[
        BudgetTier(min_budget=0, max_budget=5_000_000, video_limit=2, display_limit=5),
        BudgetTier(min_budget=5_000_000, max_budget=35_000_000, video_limit=8, display_limit=15),
        BudgetTier(min_budget=35_000_000, max_budget=100_000_000, video_limit=15, display_limit=30),
        BudgetTier(min_budget=100_000_000, max_budget=float('inf'), video_limit=25, display_limit=50),
    ],
    costs=AssetCost(video_cost=5000, display_cost=3000, audio_cost=2500),
    turnaround=Turnaround(),
    minimum_requirements=MinimumRequirements()
)
