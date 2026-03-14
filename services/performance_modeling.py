"""
Performance Modeling Service.

Analyzes the relationship between video creative features and test performance.
Uses statistical methods to identify which creative elements drive lift.

Features:
1. Correlation analysis between features and outcomes
2. Regression modeling to identify drivers
3. Segment analysis (e.g., "videos with >50% human presence")
4. Natural language insights generation
"""

import os
import json
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
import statistics

import pandas as pd
import numpy as np

# Statistical modeling
try:
    from scipy import stats
    from scipy.stats import pearsonr, spearmanr, ttest_ind
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class FeatureInsight:
    """Insight about a single feature's impact on performance."""
    feature_name: str
    feature_display_name: str
    
    # Correlation with outcome
    correlation: float
    correlation_pvalue: float
    correlation_significant: bool
    
    # Impact analysis
    impact_direction: str  # "positive", "negative", "neutral"
    impact_magnitude: str  # "strong", "moderate", "weak"
    
    # Segment analysis
    high_segment_avg_lift: float  # Avg lift when feature is high
    low_segment_avg_lift: float   # Avg lift when feature is low
    segment_difference: float      # Difference between segments
    segment_pvalue: float          # T-test p-value
    
    # Natural language insight
    insight_text: str
    confidence: str  # "high", "medium", "low"


@dataclass
class ModelResults:
    """Results from the performance model."""
    model_type: str
    r_squared: float
    adjusted_r_squared: float
    
    # Feature importance
    feature_coefficients: dict  # feature_name -> coefficient
    feature_importance_ranked: list  # [(feature, importance), ...]
    
    # Top drivers
    top_positive_drivers: list  # Top features that increase lift
    top_negative_drivers: list  # Top features that decrease lift
    
    # Model diagnostics
    sample_size: int
    features_used: list
    cross_val_score: Optional[float] = None
    
    # Timestamp
    created_at: str = ""


@dataclass
class PerformanceAnalysis:
    """Complete performance analysis results."""
    analysis_id: str
    outcome_variable: str  # e.g., "awareness_lift", "purchase_intent_lift"
    
    # Sample info
    total_videos: int
    videos_with_results: int
    
    # Summary stats
    avg_lift: float
    median_lift: float
    std_lift: float
    min_lift: float
    max_lift: float
    pass_rate: float
    
    # Feature insights
    feature_insights: list  # List of FeatureInsight
    
    # Model results
    model_results: Optional[ModelResults] = None
    
    # Key findings (natural language)
    key_findings: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    
    # Metadata
    created_at: str = ""


class PerformanceModelingService:
    """
    Service for analyzing creative feature impact on performance.
    """
    
    # Feature display names
    FEATURE_NAMES = {
        'human_presence_pct': 'Human Presence (%)',
        'avg_human_count': 'Average # of People',
        'human_screen_time_pct': 'Human Screen Coverage (%)',
        'logo_presence_pct': 'Logo Visibility (%)',
        'logo_first_3s': 'Logo in First 3 Seconds',
        'logo_last_3s': 'Logo in Last 3 Seconds',
        'product_presence_pct': 'Product Visibility (%)',
        'product_in_use_pct': 'Product Demonstration (%)',
        'text_overlay_pct': 'Text Overlay (%)',
        'has_cta': 'Has Call-to-Action',
        'cuts_per_second': 'Pacing (cuts/sec)',
        'visual_complexity': 'Visual Complexity',
        'duration_seconds': 'Duration (seconds)',
    }
    
    # Thresholds for segmentation
    SEGMENT_THRESHOLDS = {
        'human_presence_pct': 50,
        'logo_presence_pct': 30,
        'product_presence_pct': 40,
        'product_in_use_pct': 20,
        'text_overlay_pct': 30,
        'cuts_per_second': 0.5,
        'visual_complexity': 2,
        'duration_seconds': 30,
    }
    
    def __init__(self):
        """Initialize the modeling service."""
        self.analyses: dict[str, PerformanceAnalysis] = {}
    
    def analyze_performance(
        self,
        features_df: pd.DataFrame,
        results_df: pd.DataFrame,
        outcome_col: str = 'lift',
        video_id_col: str = 'video_id'
    ) -> PerformanceAnalysis:
        """
        Analyze the relationship between video features and performance.
        
        Args:
            features_df: DataFrame with video features (from VideoAnalysisService)
            results_df: DataFrame with test results (lift, pass/fail, etc.)
            outcome_col: Column name for the outcome variable
            video_id_col: Column name for video ID to join on
            
        Returns:
            PerformanceAnalysis with insights and model results
        """
        # Merge features with results
        merged_df = features_df.merge(results_df, on=video_id_col, how='inner')
        
        if len(merged_df) < 5:
            raise ValueError(f"Need at least 5 videos with both features and results. Got {len(merged_df)}")
        
        # Get feature columns (exclude non-feature columns)
        exclude_cols = [video_id_col, outcome_col, 'filename', 'passed', 'creative_name']
        feature_cols = [c for c in merged_df.columns if c not in exclude_cols and c in self.FEATURE_NAMES]
        
        # Calculate summary statistics
        outcome_values = merged_df[outcome_col].dropna()
        summary = {
            'avg_lift': outcome_values.mean(),
            'median_lift': outcome_values.median(),
            'std_lift': outcome_values.std(),
            'min_lift': outcome_values.min(),
            'max_lift': outcome_values.max(),
        }
        
        # Calculate pass rate if available
        if 'passed' in merged_df.columns:
            pass_rate = merged_df['passed'].mean() * 100
        else:
            # Assume >5% lift is a pass
            pass_rate = (outcome_values > 5).mean() * 100
        
        # Analyze each feature
        feature_insights = []
        for feature in feature_cols:
            insight = self._analyze_feature(merged_df, feature, outcome_col)
            if insight:
                feature_insights.append(insight)
        
        # Sort by absolute correlation
        feature_insights.sort(key=lambda x: abs(x.correlation), reverse=True)
        
        # Build regression model
        model_results = None
        if SKLEARN_AVAILABLE and len(merged_df) >= 10:
            model_results = self._build_model(merged_df, feature_cols, outcome_col)
        
        # Generate key findings
        key_findings = self._generate_key_findings(feature_insights, summary)
        recommendations = self._generate_recommendations(feature_insights)
        
        # Create analysis
        analysis = PerformanceAnalysis(
            analysis_id=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            outcome_variable=outcome_col,
            total_videos=len(features_df),
            videos_with_results=len(merged_df),
            avg_lift=summary['avg_lift'],
            median_lift=summary['median_lift'],
            std_lift=summary['std_lift'],
            min_lift=summary['min_lift'],
            max_lift=summary['max_lift'],
            pass_rate=pass_rate,
            feature_insights=feature_insights,
            model_results=model_results,
            key_findings=key_findings,
            recommendations=recommendations,
            created_at=datetime.now().isoformat()
        )
        
        self.analyses[analysis.analysis_id] = analysis
        return analysis
    
    def _analyze_feature(
        self,
        df: pd.DataFrame,
        feature: str,
        outcome: str
    ) -> Optional[FeatureInsight]:
        """Analyze a single feature's relationship with the outcome."""
        
        if not SCIPY_AVAILABLE:
            return None
        
        # Get clean data
        clean_df = df[[feature, outcome]].dropna()
        if len(clean_df) < 5:
            return None
        
        x = clean_df[feature].values
        y = clean_df[outcome].values
        
        # Calculate correlation
        try:
            corr, pvalue = pearsonr(x, y)
        except:
            return None
        
        significant = pvalue < 0.05
        
        # Determine impact direction and magnitude
        if abs(corr) < 0.1:
            direction = "neutral"
            magnitude = "weak"
        elif corr > 0:
            direction = "positive"
            magnitude = "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"
        else:
            direction = "negative"
            magnitude = "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"
        
        # Segment analysis
        threshold = self.SEGMENT_THRESHOLDS.get(feature, np.median(x))
        
        if feature in ['logo_first_3s', 'logo_last_3s', 'has_cta']:
            # Binary features
            high_mask = x == 1
        else:
            high_mask = x >= threshold
        
        low_mask = ~high_mask
        
        high_lift = y[high_mask].mean() if high_mask.sum() > 0 else 0
        low_lift = y[low_mask].mean() if low_mask.sum() > 0 else 0
        segment_diff = high_lift - low_lift
        
        # T-test for segment difference
        try:
            if high_mask.sum() >= 2 and low_mask.sum() >= 2:
                _, segment_pvalue = ttest_ind(y[high_mask], y[low_mask])
            else:
                segment_pvalue = 1.0
        except:
            segment_pvalue = 1.0
        
        # Generate insight text
        insight_text = self._generate_feature_insight_text(
            feature, direction, magnitude, corr, significant,
            high_lift, low_lift, segment_diff, segment_pvalue
        )
        
        # Confidence level
        if significant and abs(corr) > 0.3:
            confidence = "high"
        elif significant or abs(corr) > 0.2:
            confidence = "medium"
        else:
            confidence = "low"
        
        return FeatureInsight(
            feature_name=feature,
            feature_display_name=self.FEATURE_NAMES.get(feature, feature),
            correlation=corr,
            correlation_pvalue=pvalue,
            correlation_significant=significant,
            impact_direction=direction,
            impact_magnitude=magnitude,
            high_segment_avg_lift=high_lift,
            low_segment_avg_lift=low_lift,
            segment_difference=segment_diff,
            segment_pvalue=segment_pvalue,
            insight_text=insight_text,
            confidence=confidence
        )
    
    def _generate_feature_insight_text(
        self,
        feature: str,
        direction: str,
        magnitude: str,
        corr: float,
        significant: bool,
        high_lift: float,
        low_lift: float,
        segment_diff: float,
        segment_pvalue: float
    ) -> str:
        """Generate natural language insight for a feature."""
        
        display_name = self.FEATURE_NAMES.get(feature, feature)
        
        if direction == "neutral":
            return f"{display_name} shows no significant relationship with performance."
        
        # Build insight
        if feature in ['logo_first_3s', 'logo_last_3s', 'has_cta']:
            # Binary feature
            if direction == "positive":
                insight = f"Videos with {display_name.lower()} achieve {high_lift:.1f}% lift vs {low_lift:.1f}% without"
            else:
                insight = f"Videos with {display_name.lower()} achieve lower lift ({high_lift:.1f}%) vs without ({low_lift:.1f}%)"
        else:
            # Continuous feature
            threshold = self.SEGMENT_THRESHOLDS.get(feature, 50)
            
            if direction == "positive":
                insight = f"Videos with higher {display_name.lower()} (>{threshold:.0f}) achieve {high_lift:.1f}% lift vs {low_lift:.1f}% for lower values"
            else:
                insight = f"Videos with higher {display_name.lower()} (>{threshold:.0f}) achieve lower lift ({high_lift:.1f}%) vs {low_lift:.1f}%"
        
        # Add statistical significance
        if significant:
            insight += f" (statistically significant, p={segment_pvalue:.3f})"
        else:
            insight += f" (not statistically significant)"
        
        return insight
    
    def _build_model(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        outcome_col: str
    ) -> ModelResults:
        """Build a regression model to identify drivers."""
        
        # Prepare data
        X = df[feature_cols].fillna(0).values
        y = df[outcome_col].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit Ridge regression (handles multicollinearity)
        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y)
        
        # Calculate R-squared
        y_pred = model.predict(X_scaled)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Adjusted R-squared
        n = len(y)
        p = len(feature_cols)
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else r_squared
        
        # Feature coefficients (on standardized scale for comparison)
        coefficients = dict(zip(feature_cols, model.coef_))
        
        # Rank by absolute importance
        importance_ranked = sorted(
            [(f, abs(c)) for f, c in coefficients.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Top drivers
        top_positive = [
            (f, c) for f, c in coefficients.items() if c > 0
        ]
        top_positive.sort(key=lambda x: x[1], reverse=True)
        
        top_negative = [
            (f, c) for f, c in coefficients.items() if c < 0
        ]
        top_negative.sort(key=lambda x: x[1])
        
        # Cross-validation score
        try:
            cv_scores = cross_val_score(model, X_scaled, y, cv=min(5, len(y)//2), scoring='r2')
            cv_score = cv_scores.mean()
        except:
            cv_score = None
        
        return ModelResults(
            model_type="Ridge Regression",
            r_squared=r_squared,
            adjusted_r_squared=adj_r_squared,
            feature_coefficients=coefficients,
            feature_importance_ranked=importance_ranked,
            top_positive_drivers=top_positive[:5],
            top_negative_drivers=top_negative[:5],
            sample_size=n,
            features_used=feature_cols,
            cross_val_score=cv_score,
            created_at=datetime.now().isoformat()
        )
    
    def _generate_key_findings(
        self,
        insights: list[FeatureInsight],
        summary: dict
    ) -> list[str]:
        """Generate key findings from the analysis."""
        
        findings = []
        
        # Performance summary
        findings.append(
            f"Average lift across all videos: {summary['avg_lift']:.1f}% "
            f"(range: {summary['min_lift']:.1f}% to {summary['max_lift']:.1f}%)"
        )
        
        # Top significant positive drivers
        positive_sig = [i for i in insights 
                       if i.correlation_significant and i.impact_direction == "positive"]
        if positive_sig:
            top = positive_sig[0]
            findings.append(
                f"Strongest positive driver: {top.feature_display_name} "
                f"(r={top.correlation:.2f}, {top.impact_magnitude} impact)"
            )
        
        # Top significant negative drivers
        negative_sig = [i for i in insights 
                       if i.correlation_significant and i.impact_direction == "negative"]
        if negative_sig:
            top = negative_sig[0]
            findings.append(
                f"Strongest negative driver: {top.feature_display_name} "
                f"(r={top.correlation:.2f}, {top.impact_magnitude} impact)"
            )
        
        # Notable segment differences
        for insight in insights[:5]:
            if abs(insight.segment_difference) > 2 and insight.segment_pvalue < 0.1:
                findings.append(insight.insight_text)
        
        return findings
    
    def _generate_recommendations(self, insights: list[FeatureInsight]) -> list[str]:
        """Generate actionable recommendations from insights."""
        
        recommendations = []
        
        for insight in insights:
            if not insight.correlation_significant:
                continue
            
            if insight.impact_direction == "positive" and insight.impact_magnitude in ["strong", "moderate"]:
                if "human" in insight.feature_name.lower():
                    recommendations.append(
                        f"Increase human presence in creatives - videos with more human presence "
                        f"show {insight.segment_difference:+.1f}% higher lift"
                    )
                elif "product" in insight.feature_name.lower() and "use" in insight.feature_name.lower():
                    recommendations.append(
                        f"Show product in use/demonstration - associated with "
                        f"{insight.segment_difference:+.1f}% higher lift"
                    )
                elif "cta" in insight.feature_name.lower():
                    recommendations.append(
                        f"Include clear call-to-action - videos with CTA show "
                        f"{insight.segment_difference:+.1f}% higher lift"
                    )
            
            elif insight.impact_direction == "negative" and insight.impact_magnitude in ["strong", "moderate"]:
                if "logo" in insight.feature_name.lower() and "first" in insight.feature_name.lower():
                    recommendations.append(
                        f"Consider delaying logo appearance - early logo placement "
                        f"associated with {insight.segment_difference:.1f}% lower lift"
                    )
        
        return recommendations[:5]  # Top 5 recommendations
    
    def get_feature_comparison(
        self,
        analysis_id: str,
        feature: str
    ) -> dict:
        """Get detailed comparison for a specific feature."""
        
        if analysis_id not in self.analyses:
            return {}
        
        analysis = self.analyses[analysis_id]
        insight = next(
            (i for i in analysis.feature_insights if i.feature_name == feature),
            None
        )
        
        if not insight:
            return {}
        
        return {
            'feature': feature,
            'display_name': insight.feature_display_name,
            'correlation': insight.correlation,
            'significant': insight.correlation_significant,
            'high_segment_lift': insight.high_segment_avg_lift,
            'low_segment_lift': insight.low_segment_avg_lift,
            'difference': insight.segment_difference,
            'insight': insight.insight_text,
            'recommendations': [
                r for r in analysis.recommendations 
                if feature.replace('_', ' ') in r.lower() or 
                   insight.feature_display_name.lower() in r.lower()
            ]
        }
    
    def answer_question(self, question: str, analysis_id: str = None) -> str:
        """
        Answer a natural language question about performance drivers.
        
        This method is used by the Insights chatbot.
        """
        
        # Get the most recent analysis if not specified
        if analysis_id is None and self.analyses:
            analysis_id = list(self.analyses.keys())[-1]
        
        if not analysis_id or analysis_id not in self.analyses:
            return "I don't have any performance analysis data yet. Please run a performance analysis first by uploading video features and test results."
        
        analysis = self.analyses[analysis_id]
        question_lower = question.lower()
        
        # What drives performance?
        if any(word in question_lower for word in ['driver', 'impact', 'affect', 'influence', 'what makes']):
            response = f"Based on analysis of {analysis.videos_with_results} videos:\n\n"
            response += "**Top Performance Drivers:**\n"
            
            for finding in analysis.key_findings[:5]:
                response += f"• {finding}\n"
            
            if analysis.recommendations:
                response += "\n**Recommendations:**\n"
                for rec in analysis.recommendations[:3]:
                    response += f"• {rec}\n"
            
            return response
        
        # Human presence questions
        if 'human' in question_lower or 'people' in question_lower or 'person' in question_lower:
            insight = next(
                (i for i in analysis.feature_insights if 'human' in i.feature_name),
                None
            )
            if insight:
                return f"**Human Presence Impact:**\n\n{insight.insight_text}\n\n" \
                       f"Videos with high human presence: {insight.high_segment_avg_lift:.1f}% avg lift\n" \
                       f"Videos with low human presence: {insight.low_segment_avg_lift:.1f}% avg lift"
        
        # Logo questions
        if 'logo' in question_lower or 'brand' in question_lower:
            insights = [i for i in analysis.feature_insights if 'logo' in i.feature_name]
            if insights:
                response = "**Logo/Brand Visibility Impact:**\n\n"
                for insight in insights:
                    response += f"• {insight.insight_text}\n"
                return response
        
        # Product questions
        if 'product' in question_lower:
            insights = [i for i in analysis.feature_insights if 'product' in i.feature_name]
            if insights:
                response = "**Product Visibility Impact:**\n\n"
                for insight in insights:
                    response += f"• {insight.insight_text}\n"
                return response
        
        # CTA questions
        if 'cta' in question_lower or 'call to action' in question_lower:
            insight = next(
                (i for i in analysis.feature_insights if 'cta' in i.feature_name),
                None
            )
            if insight:
                return f"**Call-to-Action Impact:**\n\n{insight.insight_text}"
        
        # Best/worst performers
        if 'best' in question_lower or 'top' in question_lower:
            response = f"**Top Performing Creative Elements:**\n\n"
            positive = [i for i in analysis.feature_insights 
                       if i.impact_direction == "positive" and i.correlation_significant][:3]
            for i, insight in enumerate(positive, 1):
                response += f"{i}. {insight.feature_display_name}: {insight.insight_text}\n"
            return response
        
        if 'worst' in question_lower or 'avoid' in question_lower:
            response = f"**Elements to Avoid/Reduce:**\n\n"
            negative = [i for i in analysis.feature_insights 
                       if i.impact_direction == "negative" and i.correlation_significant][:3]
            for i, insight in enumerate(negative, 1):
                response += f"{i}. {insight.feature_display_name}: {insight.insight_text}\n"
            return response
        
        # Default: summary
        return f"""**Performance Analysis Summary**

• Videos analyzed: {analysis.videos_with_results}
• Average lift: {analysis.avg_lift:.1f}%
• Pass rate: {analysis.pass_rate:.1f}%

**Key Findings:**
{chr(10).join('• ' + f for f in analysis.key_findings[:3])}

Ask me about specific elements like "human presence", "logo placement", "product demos", or "what drives performance" for more details."""


# Global instance
_modeling_service: Optional[PerformanceModelingService] = None


def get_modeling_service() -> PerformanceModelingService:
    """Get or create the global modeling service."""
    global _modeling_service
    if _modeling_service is None:
        _modeling_service = PerformanceModelingService()
    return _modeling_service
