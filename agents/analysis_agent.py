"""
Analysis Agent.

Responsible for:
1. Analyzing test results
2. Generating creative recommendations (run/don't run/optimize)
3. Diagnosing why creatives failed
4. Providing long-term testing insights
"""

from typing import Optional
from pydantic import BaseModel, Field

from models import (
    TestResults,
    CreativeTestResult,
    CreativeRecommendation,
    CampaignRecommendations,
    DiagnosticMetric,
    KPIType,
)
from utils.llm import get_completion, get_structured_output


class DiagnosticInsight(BaseModel):
    """Insight from diagnostic metrics analysis."""
    metric_name: str
    issue: str
    severity: str  # "high", "medium", "low"
    suggested_fix: str


class CreativeAnalysis(BaseModel):
    """Detailed analysis of a single creative's performance."""
    creative_id: str
    creative_name: str
    passed: bool
    primary_kpi_lift: float
    primary_kpi_stat_sig: bool
    recommendation: str  # "run", "do_not_run", "optimize_and_retest"
    confidence: float
    summary: str
    diagnostic_issues: list[DiagnosticInsight] = Field(default_factory=list)
    improvement_suggestions: list[str] = Field(default_factory=list)


class AnalysisAgent:
    """
    Agent for analyzing creative test results and generating recommendations.
    """
    
    # Diagnostic metric benchmarks (typical values)
    DIAGNOSTIC_BENCHMARKS = {
        "brand_strength": 65,
        "relevance": 60,
        "emotional_engagement": 55,
        "uniqueness": 50,
        "credibility": 65,
        "call_to_action_clarity": 70,
        "brand_fit": 65,
        "message_clarity": 70,
        "likability": 60,
        "memorability": 55,
    }
    
    # Thresholds for diagnostic issues
    ISSUE_THRESHOLD_LOW = 0.75  # Below 75% of benchmark = low severity
    ISSUE_THRESHOLD_HIGH = 0.60  # Below 60% of benchmark = high severity
    
    def __init__(self, past_learnings: Optional[str] = None):
        """
        Initialize the analysis agent.
        
        Args:
            past_learnings: Optional text from past learnings for context
        """
        self.past_learnings = past_learnings
    
    def analyze_results(self, results: TestResults) -> CampaignRecommendations:
        """
        Analyze test results and generate recommendations.
        
        Args:
            results: TestResults object with all creative results
            
        Returns:
            CampaignRecommendations with actionable recommendations
        """
        creative_recommendations = []
        run_creatives = []
        do_not_run_creatives = []
        optimize_creatives = []
        
        for result in results.results:
            analysis = self._analyze_creative(result)
            
            recommendation = CreativeRecommendation(
                creative_id=result.creative_id,
                creative_name=result.creative_name,
                recommendation=analysis.recommendation,
                confidence=analysis.confidence,
                rationale=analysis.summary,
                diagnostic_insights=[
                    f"{issue.metric_name}: {issue.issue}" 
                    for issue in analysis.diagnostic_issues
                ],
                suggested_improvements=analysis.improvement_suggestions,
            )
            creative_recommendations.append(recommendation)
            
            # Categorize
            if analysis.recommendation == "run":
                run_creatives.append(result.creative_id)
            elif analysis.recommendation == "do_not_run":
                do_not_run_creatives.append(result.creative_id)
            else:
                optimize_creatives.append(result.creative_id)
        
        # Generate long-term recommendations
        long_term_recs = self._generate_long_term_recommendations(results)
        
        # Generate meta insights
        meta_insights = self._generate_meta_insights(results)
        
        return CampaignRecommendations(
            campaign_id=results.campaign_id,
            test_results_id=results.id,
            recommendations=creative_recommendations,
            run_creatives=run_creatives,
            do_not_run_creatives=do_not_run_creatives,
            optimize_creatives=optimize_creatives,
            long_term_recommendations=long_term_recs,
            meta_insights=meta_insights,
        )
    
    def _analyze_creative(self, result: CreativeTestResult) -> CreativeAnalysis:
        """Analyze a single creative's results."""
        
        # Determine recommendation based on primary KPI
        if result.passed:
            recommendation = "run"
            confidence = min(0.95, 0.7 + (result.primary_kpi_lift / 20))  # Higher lift = higher confidence
            summary = f"Creative showed statistically significant lift of {result.primary_kpi_lift:.1f}% in {result.primary_kpi.value}."
        else:
            # Check if it was close or clearly failed
            if result.primary_kpi_lift > 2 and not result.primary_kpi_stat_sig:
                # Positive lift but not significant - might work with optimization
                recommendation = "optimize_and_retest"
                confidence = 0.6
                summary = f"Creative showed {result.primary_kpi_lift:.1f}% lift but did not reach statistical significance. Consider optimization and retesting."
            elif result.primary_kpi_lift <= 0:
                # Negative or no lift
                recommendation = "do_not_run"
                confidence = 0.85
                summary = f"Creative showed no positive lift ({result.primary_kpi_lift:.1f}%) in {result.primary_kpi.value}. Not recommended for campaign."
            else:
                # Small positive lift, not significant
                recommendation = "do_not_run"
                confidence = 0.7
                summary = f"Creative showed minimal lift ({result.primary_kpi_lift:.1f}%) that was not statistically significant."
        
        # Analyze diagnostic metrics
        diagnostic_issues = self._analyze_diagnostics(result.diagnostics)
        
        # Generate improvement suggestions based on issues
        improvements = self._generate_improvements(diagnostic_issues, result)
        
        return CreativeAnalysis(
            creative_id=result.creative_id,
            creative_name=result.creative_name,
            passed=result.passed,
            primary_kpi_lift=result.primary_kpi_lift,
            primary_kpi_stat_sig=result.primary_kpi_stat_sig,
            recommendation=recommendation,
            confidence=confidence,
            summary=summary,
            diagnostic_issues=diagnostic_issues,
            improvement_suggestions=improvements,
        )
    
    def _analyze_diagnostics(self, diagnostics: list[DiagnosticMetric]) -> list[DiagnosticInsight]:
        """Identify issues from diagnostic metrics."""
        issues = []
        
        for metric in diagnostics:
            benchmark = self.DIAGNOSTIC_BENCHMARKS.get(metric.name, 60)
            ratio = metric.value / benchmark if benchmark > 0 else 1
            
            if ratio < self.ISSUE_THRESHOLD_HIGH:
                severity = "high"
                issue = f"Significantly below benchmark ({metric.value:.0f} vs {benchmark})"
            elif ratio < self.ISSUE_THRESHOLD_LOW:
                severity = "medium"
                issue = f"Below benchmark ({metric.value:.0f} vs {benchmark})"
            else:
                continue  # No issue
            
            # Generate suggested fix based on metric
            suggested_fix = self._get_fix_for_metric(metric.name, severity)
            
            issues.append(DiagnosticInsight(
                metric_name=metric.name,
                issue=issue,
                severity=severity,
                suggested_fix=suggested_fix,
            ))
        
        return issues
    
    def _get_fix_for_metric(self, metric_name: str, severity: str) -> str:
        """Get a suggested fix for a specific diagnostic metric issue."""
        fixes = {
            "brand_strength": "Increase brand visibility and branding moments throughout the creative",
            "relevance": "Better align messaging with target audience needs and interests",
            "emotional_engagement": "Add more emotionally resonant storytelling or human elements",
            "uniqueness": "Differentiate from competitor messaging; add distinctive creative elements",
            "credibility": "Include proof points, testimonials, or authoritative sources",
            "call_to_action_clarity": "Make the CTA more prominent and specific",
            "brand_fit": "Ensure creative style and tone match brand guidelines",
            "message_clarity": "Simplify the main message; focus on one key takeaway",
            "likability": "Test with focus groups; consider more positive/aspirational tone",
            "memorability": "Add distinctive audio/visual hooks; strengthen branding moments",
        }
        
        base_fix = fixes.get(metric_name, f"Review and improve {metric_name}")
        
        if severity == "high":
            return f"PRIORITY: {base_fix}"
        return base_fix
    
    def _generate_improvements(
        self, 
        issues: list[DiagnosticInsight], 
        result: CreativeTestResult
    ) -> list[str]:
        """Generate improvement suggestions based on diagnostic issues."""
        
        if not issues:
            if not result.passed:
                return ["Consider testing alternative creative concepts"]
            return []
        
        # Sort by severity
        high_priority = [i for i in issues if i.severity == "high"]
        medium_priority = [i for i in issues if i.severity == "medium"]
        
        improvements = []
        
        # Add high priority fixes first
        for issue in high_priority[:3]:  # Top 3 high priority
            improvements.append(issue.suggested_fix)
        
        # Add medium priority if room
        for issue in medium_priority[:2]:  # Up to 2 medium priority
            if len(improvements) < 5:
                improvements.append(issue.suggested_fix)
        
        return improvements
    
    def _generate_long_term_recommendations(self, results: TestResults) -> list[str]:
        """Generate long-term testing recommendations."""
        recommendations = []
        
        # Check pass rate
        if results.pass_rate < 0.5:
            recommendations.append(
                "Low overall pass rate suggests reviewing creative strategy before next campaign"
            )
        
        # Check for patterns in failures
        failed = [r for r in results.results if not r.passed]
        if len(failed) > 0:
            # Check if video or display has worse performance
            video_failed = [r for r in failed if r.asset_type.value == "video"]
            display_failed = [r for r in failed if r.asset_type.value == "display"]
            
            video_total = len([r for r in results.results if r.asset_type.value == "video"])
            display_total = len([r for r in results.results if r.asset_type.value == "display"])
            
            if video_total > 0 and len(video_failed) / video_total > 0.7:
                recommendations.append(
                    "Video creatives showing high failure rate - consider reviewing video creative approach"
                )
            
            if display_total > 0 and len(display_failed) / display_total > 0.7:
                recommendations.append(
                    "Display creatives showing high failure rate - consider reviewing display creative approach"
                )
        
        # Suggest retesting optimized creatives
        optimize_candidates = [
            r for r in results.results 
            if not r.passed and r.primary_kpi_lift > 0
        ]
        if optimize_candidates:
            recommendations.append(
                f"{len(optimize_candidates)} creative(s) showed positive but non-significant lift - "
                "consider optimization and retesting in next wave"
            )
        
        return recommendations
    
    def _generate_meta_insights(self, results: TestResults) -> list[str]:
        """Generate meta-level insights from results."""
        insights = []
        
        # Overall performance summary
        insights.append(
            f"Overall pass rate: {results.pass_rate * 100:.0f}% "
            f"({results.creatives_passed}/{results.total_creatives_tested})"
        )
        
        # Best performing creative
        if results.results:
            best = max(results.results, key=lambda r: r.primary_kpi_lift)
            insights.append(
                f"Best performing: {best.creative_name} with {best.primary_kpi_lift:.1f}% lift"
            )
        
        # Common diagnostic issues across failed creatives
        failed = [r for r in results.results if not r.passed]
        if failed:
            all_issues = []
            for r in failed:
                for d in r.diagnostics:
                    benchmark = self.DIAGNOSTIC_BENCHMARKS.get(d.name, 60)
                    if d.value < benchmark * self.ISSUE_THRESHOLD_LOW:
                        all_issues.append(d.name)
            
            if all_issues:
                # Find most common issue
                from collections import Counter
                common = Counter(all_issues).most_common(1)
                if common:
                    insights.append(
                        f"Most common issue in failed creatives: {common[0][0].replace('_', ' ')}"
                    )
        
        return insights
    
    def generate_detailed_analysis(self, results: TestResults) -> str:
        """
        Generate a detailed text analysis using LLM.
        
        This provides a narrative analysis beyond the structured recommendations.
        """
        # Prepare data summary for LLM
        data_summary = f"""Creative Testing Results Summary:

Campaign: {results.campaign_id}
Total Creatives Tested: {results.total_creatives_tested}
Passed: {results.creatives_passed}
Failed: {results.creatives_failed}
Pass Rate: {results.pass_rate * 100:.0f}%

Individual Results:
"""
        for r in results.results:
            data_summary += f"""
- {r.creative_name} ({r.asset_type.value}):
  - Primary KPI ({r.primary_kpi.value}): {r.primary_kpi_lift:.1f}% lift, {'Significant' if r.primary_kpi_stat_sig else 'Not Significant'}
  - Result: {'PASSED' if r.passed else 'FAILED'}
  - Diagnostics: {', '.join(f'{d.name}={d.value:.0f}' for d in r.diagnostics[:5])}
"""
        
        context = ""
        if self.past_learnings:
            context = f"\n\nRelevant past learnings:\n{self.past_learnings}"
        
        prompt = f"""{data_summary}{context}

Provide a detailed analysis of these creative testing results. Include:
1. Executive summary (2-3 sentences)
2. Key findings by creative
3. What's working well
4. What needs improvement
5. Specific recommendations for the campaign team

Be specific and actionable."""

        return get_completion(
            prompt=prompt,
            system="""You are an expert creative strategist analyzing brand lift study results. 
Provide clear, actionable insights that help marketing teams improve their creative effectiveness.
Focus on the "so what" - not just what the numbers say, but what teams should DO about it.""",
        )
