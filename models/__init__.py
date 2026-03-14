"""
Data models for Creative Testing Orchestrator.

These Pydantic models define the core data structures used throughout the system.
"""

from datetime import date, datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================

class AssetType(str, Enum):
    """Type of creative asset being tested."""
    VIDEO = "video"
    DISPLAY = "display"
    AUDIO = "audio"


class Channel(str, Enum):
    """Media channel for the creative."""
    TV = "tv"
    DIGITAL_VIDEO = "digital_video"
    SOCIAL = "social"
    DISPLAY = "display"
    AUDIO = "audio"
    OOH = "ooh"


class TestStatus(str, Enum):
    """Status of a creative test."""
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    IN_TESTING = "in_testing"
    RESULTS_RECEIVED = "results_received"
    COMPLETED = "completed"


class KPIType(str, Enum):
    """Brand KPIs that can be measured in lift studies."""
    AWARENESS = "awareness"
    CONSIDERATION = "consideration"
    PREFERENCE = "preference"
    PURCHASE_INTENT = "purchase_intent"
    BRAND_FAVORABILITY = "brand_favorability"
    AD_RECALL = "ad_recall"
    MESSAGE_ASSOCIATION = "message_association"


class LiftResult(str, Enum):
    """Result of lift measurement."""
    SIGNIFICANT_LIFT = "significant_lift"
    NO_SIGNIFICANT_LIFT = "no_significant_lift"
    NEGATIVE_LIFT = "negative_lift"
    INCONCLUSIVE = "inconclusive"


# ============================================================================
# Campaign & Creative Models
# ============================================================================

class Brand(BaseModel):
    """A brand within the client's portfolio."""
    id: str
    name: str
    category: Optional[str] = None


class Campaign(BaseModel):
    """A media campaign that requires creative testing."""
    id: str
    name: str
    brand: Brand
    budget: float = Field(..., description="Total media budget in USD")
    testing_budget: float = Field(default=0, description="1.5% of budget allocated to CT")
    start_date: date
    end_date: date
    primary_kpi: KPIType
    secondary_kpis: list[KPIType] = Field(default_factory=list, max_length=3)
    channels: list[Channel] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    
    def model_post_init(self, __context):
        """Calculate testing budget as 1.5% of total budget."""
        if self.testing_budget == 0:
            self.testing_budget = self.budget * 0.015


class Creative(BaseModel):
    """A creative asset to be tested."""
    id: str
    name: str
    campaign_id: str
    asset_type: AssetType
    channel: Channel
    impressions: int = Field(..., description="Planned impressions for this creative")
    hypothesis: Optional[str] = Field(None, description="What we're testing with this creative")
    creative_url: Optional[str] = None
    duration_seconds: Optional[int] = None  # For video/audio


class CreativeTrix(BaseModel):
    """Creative matrix showing creative × channel × impressions allocation."""
    campaign_id: str
    creatives: list[Creative]
    total_impressions: int = 0
    
    def model_post_init(self, __context):
        """Calculate total impressions."""
        self.total_impressions = sum(c.impressions for c in self.creatives)


# ============================================================================
# Test Plan Models
# ============================================================================

class TestPlanItem(BaseModel):
    """A single item in the test plan (one creative to be tested)."""
    creative: Creative
    test_start_date: date
    expected_results_date: date
    estimated_cost: float
    cell_size: int = Field(..., description="Sample size for the test cell")
    priority: int = Field(default=1, description="1=highest priority")


class TestPlan(BaseModel):
    """Complete creative testing plan for a campaign."""
    id: str
    campaign_id: str
    status: TestStatus = TestStatus.DRAFT
    
    # Plan details
    video_tests: list[TestPlanItem] = Field(default_factory=list)
    display_tests: list[TestPlanItem] = Field(default_factory=list)
    
    # Budget tracking
    total_estimated_cost: float = 0
    remaining_budget: float = 0
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    revision_history: list[str] = Field(default_factory=list)
    
    @property
    def video_count(self) -> int:
        return len(self.video_tests)
    
    @property
    def display_count(self) -> int:
        return len(self.display_tests)


# ============================================================================
# Test Results Models
# ============================================================================

class DiagnosticMetric(BaseModel):
    """A diagnostic metric from the test results."""
    name: str  # e.g., "brand_strength", "relevance", "emotional_engagement"
    value: float
    benchmark: Optional[float] = None
    percentile: Optional[int] = None


class CreativeTestResult(BaseModel):
    """Test results for a single creative."""
    creative_id: str
    creative_name: str
    asset_type: AssetType
    
    # Control cell metrics
    control_awareness: float
    control_consideration: float
    control_preference: Optional[float] = None
    control_purchase_intent: Optional[float] = None
    
    # Exposed cell metrics
    exposed_awareness: float
    exposed_consideration: float
    exposed_preference: Optional[float] = None
    exposed_purchase_intent: Optional[float] = None
    
    # Lift calculations (exposed - control)
    awareness_lift: float
    consideration_lift: float
    preference_lift: Optional[float] = None
    purchase_intent_lift: Optional[float] = None
    
    # Statistical significance at 90% confidence
    awareness_stat_sig: bool
    consideration_stat_sig: bool
    preference_stat_sig: Optional[bool] = None
    purchase_intent_stat_sig: Optional[bool] = None
    
    # Primary KPI result
    primary_kpi: KPIType
    primary_kpi_lift: float
    primary_kpi_stat_sig: bool
    passed: bool = Field(..., description="True if stat sig lift in primary KPI")
    
    # Diagnostic metrics
    diagnostics: list[DiagnosticMetric] = Field(default_factory=list)
    
    # Sample sizes
    control_sample_size: int
    exposed_sample_size: int


class TestResults(BaseModel):
    """Complete test results for a campaign."""
    id: str
    campaign_id: str
    test_plan_id: str
    
    # Separate results for video and display (different control cells)
    video_control_metrics: Optional[dict] = None  # Control cell metrics for video tests
    display_control_metrics: Optional[dict] = None  # Control cell metrics for display tests
    
    # Individual creative results
    results: list[CreativeTestResult] = Field(default_factory=list)
    
    # Summary
    total_creatives_tested: int = 0
    creatives_passed: int = 0
    creatives_failed: int = 0
    
    # Metadata
    received_at: datetime = Field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    
    @property
    def pass_rate(self) -> float:
        if self.total_creatives_tested == 0:
            return 0
        return self.creatives_passed / self.total_creatives_tested


# ============================================================================
# Recommendation Models
# ============================================================================

class CreativeRecommendation(BaseModel):
    """Recommendation for a single creative based on test results."""
    creative_id: str
    creative_name: str
    recommendation: str  # "run", "do_not_run", "optimize_and_retest"
    confidence: float = Field(..., ge=0, le=1)
    rationale: str
    diagnostic_insights: list[str] = Field(default_factory=list)
    suggested_improvements: list[str] = Field(default_factory=list)


class CampaignRecommendations(BaseModel):
    """Complete recommendations for a campaign's creative rotation."""
    campaign_id: str
    test_results_id: str
    
    # Recommendations by creative
    recommendations: list[CreativeRecommendation]
    
    # Summary
    run_creatives: list[str] = Field(default_factory=list)  # Creative IDs to run
    do_not_run_creatives: list[str] = Field(default_factory=list)
    optimize_creatives: list[str] = Field(default_factory=list)
    
    # Long-term testing suggestions
    long_term_recommendations: list[str] = Field(default_factory=list)
    
    # Meta-analysis insights (if available)
    meta_insights: list[str] = Field(default_factory=list)
    
    generated_at: datetime = Field(default_factory=datetime.now)
