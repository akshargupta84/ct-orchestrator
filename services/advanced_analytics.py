"""
Advanced Analytics Service.

Provides multi-stage analysis pipeline:
1. Statistical Analysis (correlations, regression, significance testing)
2. Historical Comparison (percentiles, trends, benchmarks)
3. Pattern Mining (decision trees, clustering, feature importance)
4. AI Synthesis (Claude-powered narrative and recommendations)
"""

import numpy as np
import pandas as pd
from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

# Statistical libraries
try:
    from scipy import stats
    from scipy.stats import pearsonr, spearmanr, ttest_ind, chi2_contingency
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ML libraries
try:
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression, LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class StatisticalFindings:
    """Results from Stage 1: Statistical Analysis."""
    correlations: dict = field(default_factory=dict)  # metric -> correlation with lift
    correlation_pvalues: dict = field(default_factory=dict)
    regression_coefficients: dict = field(default_factory=dict)
    regression_r_squared: float = 0.0
    significant_differences: list = field(default_factory=list)  # metrics that differ between pass/fail
    effect_sizes: dict = field(default_factory=dict)  # Cohen's d for each metric
    confidence_intervals: dict = field(default_factory=dict)
    lift_statistics: dict = field(default_factory=dict)
    

@dataclass
class HistoricalContext:
    """Results from Stage 2: Historical Comparison."""
    percentile_ranks: dict = field(default_factory=dict)  # creative -> percentile
    historical_pass_rate: float = 0.0
    current_pass_rate: float = 0.0
    trend_direction: str = "stable"  # improving, declining, stable
    similar_historical_creatives: list = field(default_factory=list)
    benchmark_comparison: dict = field(default_factory=dict)
    historical_patterns: list = field(default_factory=list)


@dataclass
class PatternMiningResults:
    """Results from Stage 3: Pattern Mining."""
    decision_rules: list = field(default_factory=list)
    feature_importance: dict = field(default_factory=dict)
    clusters: list = field(default_factory=list)
    winning_combinations: list = field(default_factory=list)
    threshold_recommendations: dict = field(default_factory=dict)


@dataclass 
class CreativeAnalysis:
    """Deep analysis for a single creative."""
    creative_name: str
    creative_id: str
    passed: bool
    lift: float
    stat_sig: bool
    
    # Diagnostic breakdown
    diagnostic_scores: dict = field(default_factory=dict)
    diagnostic_benchmarks: dict = field(default_factory=dict)  # vs test average
    weak_areas: list = field(default_factory=list)
    strong_areas: list = field(default_factory=list)
    
    # Comparison
    percentile_rank: float = 50.0
    similar_historical: list = field(default_factory=list)
    
    # Recommendations
    category: str = "run"  # run, do_not_run, optimize_retest
    specific_recommendations: list = field(default_factory=list)
    predicted_pass_probability: float = 0.5


@dataclass
class AdvancedAnalysisResult:
    """Complete analysis result from all stages."""
    # Stage outputs
    statistical: StatisticalFindings = field(default_factory=StatisticalFindings)
    historical: HistoricalContext = field(default_factory=HistoricalContext)
    patterns: PatternMiningResults = field(default_factory=PatternMiningResults)
    
    # Per-creative analysis
    creative_analyses: list = field(default_factory=list)
    
    # Summary
    executive_summary: str = ""
    key_findings: list = field(default_factory=list)
    optimization_playbook: list = field(default_factory=list)
    
    # Metadata
    analysis_timestamp: str = ""
    data_quality_notes: list = field(default_factory=list)


class AdvancedAnalyticsService:
    """
    Multi-stage analytics pipeline for creative test results.
    """
    
    # Diagnostic score benchmarks (industry standards)
    DIAGNOSTIC_BENCHMARKS = {
        'attention_score': 60,
        'brand_recall_score': 55,
        'message_clarity_score': 60,
        'emotional_resonance_score': 55,
        'uniqueness_score': 50,
    }
    
    # Mapping from diagnostic to improvement recommendations
    DIAGNOSTIC_RECOMMENDATIONS = {
        'attention_score': [
            "Add human face in first 2 seconds (+12 points avg historically)",
            "Increase motion/action in opening sequence",
            "Use pattern interrupt or unexpected visual hook",
            "Consider brighter colors or higher contrast",
        ],
        'brand_recall_score': [
            "Ensure logo appears within first 3 seconds",
            "Add logo presence in final frame",
            "Use brand colors consistently throughout",
            "Include audio brand mention",
        ],
        'message_clarity_score': [
            "Reduce to single key message per creative",
            "Add clear end card with key takeaway",
            "Remove competing visual elements",
            "Use text overlay to reinforce key message",
        ],
        'emotional_resonance_score': [
            "Add human story or relatable scenario",
            "Include emotional music that matches message",
            "Show product solving real human problem",
            "Use testimonial or real customer footage",
        ],
        'uniqueness_score': [
            "Differentiate from competitor creative styles",
            "Try unexpected format or narrative structure",
            "Use distinctive visual style or color palette",
            "Add memorable catchphrase or audio hook",
        ],
    }
    
    def __init__(self, vector_store=None):
        """
        Initialize analytics service.
        
        Args:
            vector_store: Optional vector store for historical comparisons
        """
        self.vector_store = vector_store
        self._historical_data = None
    
    def analyze(self, results_data: dict, historical_results: list = None) -> AdvancedAnalysisResult:
        """
        Run complete analysis pipeline.
        
        Args:
            results_data: Current test results (dict with 'results' list)
            historical_results: Optional list of past test results for comparison
            
        Returns:
            AdvancedAnalysisResult with all findings
        """
        result = AdvancedAnalysisResult(
            analysis_timestamp=datetime.now().isoformat()
        )
        
        # Convert to DataFrame for analysis
        df = self._prepare_dataframe(results_data)
        
        if df is None or len(df) < 2:
            result.data_quality_notes.append("Insufficient data for advanced analysis (need at least 2 creatives)")
            return result
        
        # Stage 1: Statistical Analysis
        result.statistical = self._stage1_statistical_analysis(df)
        
        # Stage 2: Historical Comparison
        if historical_results:
            self._historical_data = historical_results
        result.historical = self._stage2_historical_comparison(df, results_data)
        
        # Stage 3: Pattern Mining
        result.patterns = self._stage3_pattern_mining(df)
        
        # Generate per-creative analysis
        result.creative_analyses = self._generate_creative_analyses(
            df, result.statistical, result.historical, result.patterns
        )
        
        # Generate key findings
        result.key_findings = self._generate_key_findings(result)
        
        # Generate optimization playbook
        result.optimization_playbook = self._generate_optimization_playbook(result)
        
        return result
    
    def _prepare_dataframe(self, results_data: dict) -> Optional[pd.DataFrame]:
        """Convert results to DataFrame for analysis."""
        results = results_data.get('results', [])
        if not results:
            return None
        
        rows = []
        for r in results:
            # Handle both dict and object formats
            if isinstance(r, dict):
                row = {
                    'creative_name': r.get('creative_name', ''),
                    'creative_id': r.get('creative_id', ''),
                    'passed': r.get('passed', False),
                    'lift': r.get('primary_kpi_lift', r.get('awareness_lift_pct', 0)),
                    'stat_sig': r.get('primary_kpi_stat_sig', r.get('awareness_significant', False)),
                    'attention_score': r.get('attention_score', r.get('diagnostics', {}).get('attention', 50)),
                    'brand_recall_score': r.get('brand_recall_score', r.get('diagnostics', {}).get('brand_recall', 50)),
                    'message_clarity_score': r.get('message_clarity_score', r.get('diagnostics', {}).get('message_clarity', 50)),
                    'emotional_resonance_score': r.get('emotional_resonance_score', r.get('diagnostics', {}).get('emotional_resonance', 50)),
                    'uniqueness_score': r.get('uniqueness_score', r.get('diagnostics', {}).get('uniqueness', 50)),
                }
            else:
                # Object with attributes
                row = {
                    'creative_name': getattr(r, 'creative_name', ''),
                    'creative_id': getattr(r, 'creative_id', ''),
                    'passed': getattr(r, 'passed', False),
                    'lift': getattr(r, 'primary_kpi_lift', 0),
                    'stat_sig': getattr(r, 'primary_kpi_stat_sig', False),
                }
                # Get diagnostics
                diagnostics = getattr(r, 'diagnostics', [])
                for d in diagnostics:
                    if hasattr(d, 'name') and hasattr(d, 'value'):
                        row[d.name] = d.value
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Ensure boolean columns
        df['passed'] = df['passed'].astype(bool)
        df['stat_sig'] = df['stat_sig'].astype(bool)
        
        return df
    
    def _stage1_statistical_analysis(self, df: pd.DataFrame) -> StatisticalFindings:
        """
        Stage 1: Statistical Analysis
        - Correlations between diagnostics and lift
        - Regression to predict lift
        - Significance tests between pass/fail groups
        - Effect sizes
        """
        import warnings
        findings = StatisticalFindings()
        
        if not SCIPY_AVAILABLE:
            findings.correlations = {"error": "scipy not installed"}
            return findings
        
        diagnostic_cols = [c for c in df.columns if c.endswith('_score')]
        
        # Correlation analysis - with variance checks
        for col in diagnostic_cols:
            if col in df.columns and df[col].notna().sum() >= 2:
                try:
                    col_values = df[col].fillna(df[col].mean())
                    lift_values = df['lift'].fillna(0)
                    
                    # Skip if constant values (no variance)
                    if col_values.std() < 0.001 or lift_values.std() < 0.001:
                        findings.correlations[col] = 0.0
                        findings.correlation_pvalues[col] = 1.0
                        continue
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        corr, pval = pearsonr(col_values, lift_values)
                    
                    # Handle NaN results
                    if np.isnan(corr):
                        corr = 0.0
                        pval = 1.0
                    
                    findings.correlations[col] = round(corr, 3)
                    findings.correlation_pvalues[col] = round(pval, 4)
                except Exception:
                    findings.correlations[col] = 0.0
                    findings.correlation_pvalues[col] = 1.0
        
        # Regression analysis - with variance check
        if SKLEARN_AVAILABLE and len(diagnostic_cols) > 0 and len(df) >= 3:
            try:
                X = df[diagnostic_cols].fillna(df[diagnostic_cols].mean())
                y = df['lift'].fillna(0)
                
                # Only run if y has variance
                if y.std() > 0.001:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        reg = LinearRegression()
                        reg.fit(X, y)
                    
                    r_squared = reg.score(X, y)
                    if not np.isnan(r_squared):
                        findings.regression_r_squared = round(r_squared, 3)
                        findings.regression_coefficients = {
                            col: round(coef, 4) 
                            for col, coef in zip(diagnostic_cols, reg.coef_)
                            if not np.isnan(coef)
                        }
                        if not np.isnan(reg.intercept_):
                            findings.regression_coefficients['intercept'] = round(reg.intercept_, 4)
            except Exception:
                pass
        
        # T-tests between pass/fail groups
        passed_df = df[df['passed'] == True]
        failed_df = df[df['passed'] == False]
        
        if len(passed_df) >= 1 and len(failed_df) >= 1:
            for col in diagnostic_cols:
                if col in df.columns:
                    try:
                        passed_vals = passed_df[col].dropna()
                        failed_vals = failed_df[col].dropna()
                        
                        if len(passed_vals) >= 1 and len(failed_vals) >= 1:
                            # Skip if no variance in either group
                            if passed_vals.std() < 0.001 and failed_vals.std() < 0.001:
                                continue
                            
                            # T-test with warnings suppressed
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                t_stat, p_val = ttest_ind(passed_vals, failed_vals, equal_var=False)
                            
                            if np.isnan(p_val):
                                continue
                            
                            # Effect size (Cohen's d)
                            pooled_std = np.sqrt((passed_vals.var() + failed_vals.var()) / 2)
                            if pooled_std > 0.001:
                                cohens_d = (passed_vals.mean() - failed_vals.mean()) / pooled_std
                            else:
                                cohens_d = 0
                            
                            if not np.isnan(cohens_d):
                                findings.effect_sizes[col] = round(cohens_d, 3)
                            
                            if p_val < 0.1:  # Include marginally significant
                                findings.significant_differences.append({
                                    'metric': col,
                                    'passed_mean': round(passed_vals.mean(), 1),
                                    'failed_mean': round(failed_vals.mean(), 1),
                                    'difference': round(passed_vals.mean() - failed_vals.mean(), 1),
                                    'p_value': round(p_val, 4),
                                    'effect_size': round(cohens_d, 3) if not np.isnan(cohens_d) else 0,
                                    'effect_magnitude': self._interpret_effect_size(cohens_d) if not np.isnan(cohens_d) else 'unknown',
                                })
                    except Exception:
                        pass
        
        # Lift statistics
        lift_values = df['lift'].dropna()
        if len(lift_values) > 0:
            findings.lift_statistics = {
                'mean': round(lift_values.mean(), 2),
                'std': round(lift_values.std(), 2) if len(lift_values) > 1 else 0,
                'min': round(lift_values.min(), 2),
                'max': round(lift_values.max(), 2),
                'median': round(lift_values.median(), 2),
            }
        
        # Confidence intervals for mean lift - only with enough data and variance
        if len(df) >= 3 and df['lift'].std() > 0.001:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ci = stats.t.interval(
                        0.95, 
                        len(df) - 1, 
                        loc=df['lift'].mean(), 
                        scale=stats.sem(df['lift'])
                    )
                if not np.isnan(ci[0]) and not np.isnan(ci[1]):
                    findings.confidence_intervals['lift_95ci'] = {
                        'lower': round(ci[0], 2),
                        'upper': round(ci[1], 2),
                    }
            except Exception:
                pass
        
        return findings
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _stage2_historical_comparison(self, df: pd.DataFrame, results_data: dict) -> HistoricalContext:
        """
        Stage 2: Historical Comparison
        - Percentile ranks vs history
        - Trend analysis
        - Benchmark comparison
        """
        context = HistoricalContext()
        
        # Current pass rate
        context.current_pass_rate = round(df['passed'].mean() * 100, 1)
        
        # If we have historical data
        if self._historical_data:
            try:
                # Calculate historical pass rate
                historical_passes = sum(1 for h in self._historical_data if h.get('passed', False))
                context.historical_pass_rate = round(historical_passes / len(self._historical_data) * 100, 1)
                
                # Calculate percentile ranks for each creative
                historical_lifts = [h.get('lift', 0) for h in self._historical_data]
                if historical_lifts:
                    for _, row in df.iterrows():
                        rank = stats.percentileofscore(historical_lifts, row['lift'])
                        context.percentile_ranks[row['creative_name']] = round(rank, 0)
                
                # Trend direction
                if context.current_pass_rate > context.historical_pass_rate + 5:
                    context.trend_direction = "improving"
                elif context.current_pass_rate < context.historical_pass_rate - 5:
                    context.trend_direction = "declining"
                else:
                    context.trend_direction = "stable"
                    
            except Exception:
                pass
        else:
            # No historical data - use within-test percentiles
            for _, row in df.iterrows():
                rank = stats.percentileofscore(df['lift'].tolist(), row['lift'])
                context.percentile_ranks[row['creative_name']] = round(rank, 0)
        
        # Benchmark comparison
        context.benchmark_comparison = {
            'industry_avg_pass_rate': 45,  # Industry standard
            'current_vs_industry': round(context.current_pass_rate - 45, 1),
        }
        
        # Historical patterns
        context.historical_patterns = self._identify_historical_patterns(df)
        
        return context
    
    def _identify_historical_patterns(self, df: pd.DataFrame) -> list:
        """Identify patterns from the data."""
        patterns = []
        
        diagnostic_cols = [c for c in df.columns if c.endswith('_score')]
        
        for col in diagnostic_cols:
            if col not in df.columns:
                continue
                
            # Find threshold that best separates pass/fail
            passed = df[df['passed'] == True][col].dropna()
            failed = df[df['passed'] == False][col].dropna()
            
            if len(passed) > 0 and len(failed) > 0:
                threshold = (passed.mean() + failed.mean()) / 2
                
                above_threshold = df[df[col] >= threshold]
                below_threshold = df[df[col] < threshold]
                
                if len(above_threshold) > 0 and len(below_threshold) > 0:
                    above_pass_rate = above_threshold['passed'].mean() * 100
                    below_pass_rate = below_threshold['passed'].mean() * 100
                    
                    if above_pass_rate > below_pass_rate + 20:
                        patterns.append({
                            'metric': col.replace('_score', '').replace('_', ' ').title(),
                            'threshold': round(threshold, 0),
                            'above_pass_rate': round(above_pass_rate, 0),
                            'below_pass_rate': round(below_pass_rate, 0),
                            'insight': f"Creatives with {col.replace('_', ' ')} >= {threshold:.0f} pass {above_pass_rate:.0f}% of the time vs {below_pass_rate:.0f}% below"
                        })
        
        return patterns
    
    def _stage3_pattern_mining(self, df: pd.DataFrame) -> PatternMiningResults:
        """
        Stage 3: Pattern Mining
        - Decision tree rules
        - Feature importance
        - Clustering
        """
        results = PatternMiningResults()
        
        if not SKLEARN_AVAILABLE:
            return results
        
        diagnostic_cols = [c for c in df.columns if c.endswith('_score')]
        
        if len(diagnostic_cols) == 0 or len(df) < 3:
            return results
        
        X = df[diagnostic_cols].fillna(df[diagnostic_cols].mean())
        y = df['passed'].astype(int)
        
        # Decision Tree for interpretable rules
        try:
            dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1)
            dt.fit(X, y)
            
            # Extract rules
            tree_rules = export_text(dt, feature_names=diagnostic_cols)
            results.decision_rules = self._parse_decision_tree_rules(tree_rules, diagnostic_cols)
            
        except Exception:
            pass
        
        # Random Forest for feature importance
        try:
            rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
            rf.fit(X, y)
            
            importances = rf.feature_importances_
            results.feature_importance = {
                col: round(imp, 3) 
                for col, imp in sorted(zip(diagnostic_cols, importances), key=lambda x: -x[1])
            }
        except Exception:
            pass
        
        # K-Means Clustering - only if we have enough distinct data points
        if len(df) >= 3:
            try:
                import warnings
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Use fewer clusters for small datasets
                n_clusters = min(2, len(df) - 1)  # At most 2 clusters for small data
                if n_clusters < 2:
                    n_clusters = 1
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    df = df.copy()  # Avoid SettingWithCopyWarning
                    df['cluster'] = kmeans.fit_predict(X_scaled)
                
                # Analyze clusters
                for i in range(n_clusters):
                    cluster_df = df[df['cluster'] == i]
                    if len(cluster_df) > 0:
                        cluster_info = {
                            'cluster_id': i,
                            'size': len(cluster_df),
                            'pass_rate': round(cluster_df['passed'].mean() * 100, 0),
                            'avg_lift': round(cluster_df['lift'].mean(), 1),
                            'characteristics': {},
                            'creatives': cluster_df['creative_name'].tolist(),
                        }
                        
                        # Find defining characteristics
                        for col in diagnostic_cols:
                            cluster_mean = cluster_df[col].mean()
                            overall_mean = df[col].mean()
                            if cluster_mean > overall_mean + 5:
                                cluster_info['characteristics'][col] = f"High ({cluster_mean:.0f})"
                            elif cluster_mean < overall_mean - 5:
                                cluster_info['characteristics'][col] = f"Low ({cluster_mean:.0f})"
                        
                        results.clusters.append(cluster_info)
                        
            except Exception:
                pass
        
        # Threshold recommendations
        for col in diagnostic_cols:
            if col in df.columns:
                passed = df[df['passed'] == True][col]
                if len(passed) > 0:
                    results.threshold_recommendations[col] = round(passed.mean(), 0)
        
        return results
    
    def _parse_decision_tree_rules(self, tree_text: str, feature_names: list) -> list:
        """Parse decision tree text into readable rules."""
        rules = []
        lines = tree_text.strip().split('\n')
        
        current_rule = []
        for line in lines:
            if 'class:' in line:
                # Extract class prediction
                if '1' in line.split('class:')[1]:
                    prediction = "PASS"
                else:
                    prediction = "FAIL"
                
                if current_rule:
                    rule_text = " AND ".join(current_rule)
                    rules.append({
                        'conditions': rule_text,
                        'prediction': prediction,
                        'readable': f"IF {rule_text} THEN {prediction}"
                    })
                current_rule = []
            elif '<=' in line or '>' in line:
                # Extract condition
                condition = line.strip().replace('|', '').replace('---', '').strip()
                if condition:
                    current_rule.append(condition)
        
        return rules[:5]  # Limit to top 5 rules
    
    def _generate_creative_analyses(
        self, 
        df: pd.DataFrame, 
        stats: StatisticalFindings,
        historical: HistoricalContext,
        patterns: PatternMiningResults
    ) -> list:
        """Generate detailed analysis for each creative."""
        analyses = []
        
        diagnostic_cols = [c for c in df.columns if c.endswith('_score')]
        test_averages = {col: df[col].mean() for col in diagnostic_cols}
        
        for _, row in df.iterrows():
            analysis = CreativeAnalysis(
                creative_name=row['creative_name'],
                creative_id=row.get('creative_id', ''),
                passed=row['passed'],
                lift=row['lift'],
                stat_sig=row.get('stat_sig', False),
            )
            
            # Diagnostic scores and comparison to test average
            for col in diagnostic_cols:
                if col in row:
                    score = row[col]
                    analysis.diagnostic_scores[col] = score
                    analysis.diagnostic_benchmarks[col] = {
                        'score': round(score, 0) if not pd.isna(score) else 0,
                        'test_avg': round(test_averages.get(col, 50), 0),
                        'benchmark': self.DIAGNOSTIC_BENCHMARKS.get(col, 50),
                        'vs_test': round(score - test_averages.get(col, 50), 0) if not pd.isna(score) else 0,
                        'vs_benchmark': round(score - self.DIAGNOSTIC_BENCHMARKS.get(col, 50), 0) if not pd.isna(score) else 0,
                    }
                    
                    # Identify weak and strong areas
                    if not pd.isna(score):
                        benchmark = self.DIAGNOSTIC_BENCHMARKS.get(col, 50)
                        if score < benchmark - 10:
                            analysis.weak_areas.append({
                                'metric': col,
                                'score': round(score, 0),
                                'benchmark': benchmark,
                                'gap': round(benchmark - score, 0),
                            })
                        elif score > benchmark + 10:
                            analysis.strong_areas.append({
                                'metric': col,
                                'score': round(score, 0),
                                'benchmark': benchmark,
                            })
            
            # Percentile rank
            analysis.percentile_rank = historical.percentile_ranks.get(row['creative_name'], 50)
            
            # Categorize
            if row['passed']:
                analysis.category = "run"
            elif row.get('stat_sig', False) and row['lift'] <= 0:
                analysis.category = "do_not_run"
            elif row['lift'] < -2:  # Clearly negative even if not sig
                analysis.category = "do_not_run"
            else:
                analysis.category = "optimize_retest"
            
            # Generate specific recommendations
            analysis.specific_recommendations = self._generate_recommendations(analysis, stats, patterns)
            
            # Predicted pass probability (if we have a model)
            if SKLEARN_AVAILABLE and len(diagnostic_cols) > 0:
                try:
                    X = df[diagnostic_cols].fillna(df[diagnostic_cols].mean())
                    y = df['passed'].astype(int)
                    
                    if len(X) >= 3:
                        lr = LogisticRegression(random_state=42)
                        lr.fit(X, y)
                        
                        creative_X = pd.DataFrame([row])[diagnostic_cols].fillna(X.mean())
                        prob = lr.predict_proba(creative_X)[0][1]
                        analysis.predicted_pass_probability = round(prob, 2)
                except Exception:
                    pass
            
            analyses.append(analysis)
        
        return analyses
    
    def _generate_recommendations(
        self, 
        analysis: CreativeAnalysis, 
        stats: StatisticalFindings,
        patterns: PatternMiningResults
    ) -> list:
        """Generate specific recommendations for a creative."""
        recommendations = []
        
        if analysis.passed:
            recommendations.append({
                'type': 'success',
                'priority': 'high',
                'recommendation': "Scale this creative - consider additional placements or extended run",
                'rationale': f"Achieved {analysis.lift:.1f}% statistically significant lift"
            })
            
            # Note strengths
            if analysis.strong_areas:
                strengths = ", ".join([s['metric'].replace('_score', '').replace('_', ' ') for s in analysis.strong_areas])
                recommendations.append({
                    'type': 'insight',
                    'priority': 'medium',
                    'recommendation': f"Key strengths to replicate: {strengths}",
                    'rationale': "These elements scored above benchmark"
                })
        else:
            # For failed creatives, recommend based on weak areas
            sorted_weak = sorted(analysis.weak_areas, key=lambda x: x['gap'], reverse=True)
            
            for weak in sorted_weak[:2]:  # Top 2 weakest areas
                metric = weak['metric']
                if metric in self.DIAGNOSTIC_RECOMMENDATIONS:
                    recs = self.DIAGNOSTIC_RECOMMENDATIONS[metric]
                    recommendations.append({
                        'type': 'improvement',
                        'priority': 'high',
                        'recommendation': recs[0],  # Top recommendation for this metric
                        'rationale': f"{metric.replace('_', ' ').title()} is {weak['gap']:.0f} points below benchmark ({weak['score']:.0f} vs {weak['benchmark']})"
                    })
            
            # Check if close to passing
            if analysis.lift > 3 and not analysis.stat_sig:
                recommendations.append({
                    'type': 'testing',
                    'priority': 'medium',
                    'recommendation': "Consider retesting with larger sample size",
                    'rationale': f"Lift of {analysis.lift:.1f}% is promising but didn't reach significance - may be underpowered"
                })
            
            # Use pattern insights
            if patterns.threshold_recommendations:
                for metric, threshold in patterns.threshold_recommendations.items():
                    if metric in analysis.diagnostic_scores:
                        score = analysis.diagnostic_scores[metric]
                        if not pd.isna(score) and score < threshold:
                            recommendations.append({
                                'type': 'threshold',
                                'priority': 'medium',
                                'recommendation': f"Improve {metric.replace('_', ' ')} to at least {threshold:.0f}",
                                'rationale': f"Passed creatives in this test averaged {threshold:.0f} on this metric"
                            })
                            break  # Only add one threshold recommendation
        
        return recommendations
    
    def _generate_key_findings(self, result: AdvancedAnalysisResult) -> list:
        """Generate key findings summary."""
        findings = []
        
        # Get sample size for warnings
        num_creatives = len(result.creative_analyses) if result.creative_analyses else 0
        
        # Top predictor - but only if we have enough samples
        if result.statistical.correlations and num_creatives >= 5:
            top_corr = max(result.statistical.correlations.items(), key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0)
            if isinstance(top_corr[1], (int, float)) and abs(top_corr[1]) > 0.3:
                # Use "r" for correlation, not "r²" 
                correlation_strength = "strong" if abs(top_corr[1]) > 0.7 else "moderate" if abs(top_corr[1]) > 0.5 else "weak"
                findings.append({
                    'type': 'statistical',
                    'finding': f"**{top_corr[0].replace('_', ' ').title()}** shows {correlation_strength} correlation with lift (r={top_corr[1]:.2f})",
                    'implication': "Focus optimization efforts on this metric first"
                })
        elif num_creatives < 5:
            findings.append({
                'type': 'caution',
                'finding': f"**Limited sample size** ({num_creatives} creatives)",
                'implication': "Statistical patterns require more data - interpret findings directionally only"
            })
        
        # Significant differences - with effect size context
        if result.statistical.significant_differences and num_creatives >= 4:
            top_diff = result.statistical.significant_differences[0]
            findings.append({
                'type': 'comparison',
                'finding': f"Passed vs failed creatives differ most on **{top_diff['metric'].replace('_', ' ')}** ({top_diff['passed_mean']:.0f} vs {top_diff['failed_mean']:.0f})",
                'implication': f"This is a {top_diff['effect_magnitude']} effect (Cohen's d={top_diff['effect_size']:.2f})"
            })
        
        # Pass rate vs benchmark
        if result.historical.current_pass_rate > 0:
            vs_industry = result.historical.benchmark_comparison.get('current_vs_industry', 0)
            if vs_industry > 10:
                findings.append({
                    'type': 'performance',
                    'finding': f"Test pass rate ({result.historical.current_pass_rate:.0f}%) is **above** industry average",
                    'implication': "Creative strategy is working well"
                })
            elif vs_industry < -10:
                findings.append({
                    'type': 'performance',
                    'finding': f"Test pass rate ({result.historical.current_pass_rate:.0f}%) is **below** industry average",
                    'implication': "Consider strategic creative review"
                })
        
        # Patterns - only with sufficient data
        if result.patterns.decision_rules and num_creatives >= 6:
            rule = result.patterns.decision_rules[0]
            findings.append({
                'type': 'pattern',
                'finding': f"Key rule: {rule['readable']}",
                'implication': "Use this as a creative quality checkpoint"
            })
        
        return findings
    
    def _generate_optimization_playbook(self, result: AdvancedAnalysisResult) -> list:
        """Generate optimization playbook based on all findings."""
        playbook = []
        
        # Sort features by importance
        if result.patterns.feature_importance:
            for metric, importance in list(result.patterns.feature_importance.items())[:3]:
                if metric in self.DIAGNOSTIC_RECOMMENDATIONS:
                    playbook.append({
                        'metric': metric.replace('_', ' ').title(),
                        'importance': f"{importance * 100:.0f}%",
                        'recommendations': self.DIAGNOSTIC_RECOMMENDATIONS[metric][:2],
                        'threshold': result.patterns.threshold_recommendations.get(metric, 'N/A'),
                    })
        
        return playbook


def get_analytics_service(vector_store=None) -> AdvancedAnalyticsService:
    """Get analytics service instance."""
    return AdvancedAnalyticsService(vector_store=vector_store)
