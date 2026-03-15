"""
Advanced Analysis Service.

Multi-stage analysis pipeline:
1. Statistical Analysis (correlations, regression, significance tests)
2. Historical Comparison (percentiles, trends, benchmarks)
3. Pattern Mining (decision trees, clustering, feature importance)
4. Claude Synthesis (narrative insights and recommendations)
"""

import numpy as np
import pandas as pd
from typing import Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# Statistical libraries
try:
    from scipy import stats
    from scipy.stats import pearsonr, spearmanr, ttest_ind, chi2_contingency
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ML libraries
try:
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class StatisticalFindings:
    """Results from Stage 1: Statistical Analysis."""
    correlations: dict = field(default_factory=dict)  # {metric: correlation_with_lift}
    correlation_pvalues: dict = field(default_factory=dict)
    regression_coefficients: dict = field(default_factory=dict)
    regression_r_squared: float = 0.0
    significant_differences: list = field(default_factory=list)  # Metrics where pass/fail differ significantly
    effect_sizes: dict = field(default_factory=dict)  # Cohen's d for each metric
    confidence_intervals: dict = field(default_factory=dict)  # CI for lift estimates
    key_predictors: list = field(default_factory=list)  # Ranked list of predictors
    

@dataclass
class HistoricalContext:
    """Results from Stage 2: Historical Comparison."""
    percentile_ranks: dict = field(default_factory=dict)  # {creative_name: percentile}
    historical_pass_rate: float = 0.0
    current_pass_rate: float = 0.0
    trend_direction: str = "stable"  # "improving", "declining", "stable"
    similar_historical_creatives: list = field(default_factory=list)
    benchmark_comparisons: dict = field(default_factory=dict)
    pattern_matches: list = field(default_factory=list)  # Historical patterns that match current creatives


@dataclass
class PatternMiningResults:
    """Results from Stage 3: Pattern Mining."""
    decision_rules: list = field(default_factory=list)  # Human-readable rules
    feature_importance: dict = field(default_factory=dict)  # {feature: importance_score}
    clusters: list = field(default_factory=list)  # [{name, creatives, characteristics}]
    winning_combinations: list = field(default_factory=list)  # Feature combinations that predict success
    diagnostic_thresholds: dict = field(default_factory=dict)  # Optimal thresholds for each diagnostic


@dataclass 
class CreativeRecommendation:
    """Specific recommendation for a creative."""
    creative_name: str
    creative_id: str
    category: str  # "run", "do_not_run", "optimize_retest"
    lift: float
    stat_sig: bool
    strengths: list = field(default_factory=list)
    weaknesses: list = field(default_factory=list)
    specific_fixes: list = field(default_factory=list)
    percentile_rank: Optional[float] = None
    pattern_match: Optional[str] = None
    confidence_interval: Optional[tuple] = None


@dataclass
class AdvancedAnalysisResult:
    """Complete analysis result from all stages."""
    statistical: StatisticalFindings
    historical: HistoricalContext
    patterns: PatternMiningResults
    recommendations: list  # List of CreativeRecommendation
    executive_summary: str = ""
    optimization_playbook: list = field(default_factory=list)
    raw_data: dict = field(default_factory=dict)


class AdvancedAnalysisService:
    """
    Performs multi-stage advanced analysis on creative test results.
    """
    
    # Diagnostic score columns we look for
    DIAGNOSTIC_COLUMNS = [
        'brand_recall_score', 'message_clarity_score', 'emotional_resonance_score',
        'attention_score', 'uniqueness_score', 'brand_strength', 'relevance',
        'emotional_engagement', 'credibility', 'likability', 'memorability'
    ]
    
    # Friendly names for diagnostics
    DIAGNOSTIC_NAMES = {
        'brand_recall_score': 'Brand Recall',
        'message_clarity_score': 'Message Clarity',
        'emotional_resonance_score': 'Emotional Resonance',
        'attention_score': 'Attention',
        'uniqueness_score': 'Uniqueness',
        'brand_strength': 'Brand Strength',
        'relevance': 'Relevance',
        'emotional_engagement': 'Emotional Engagement',
        'credibility': 'Credibility',
        'likability': 'Likability',
        'memorability': 'Memorability',
    }
    
    def __init__(self):
        self.historical_data = None  # Will be loaded from vector store
        
    def analyze(
        self,
        results_df: pd.DataFrame,
        historical_df: Optional[pd.DataFrame] = None,
        primary_kpi: str = "awareness"
    ) -> AdvancedAnalysisResult:
        """
        Run the complete multi-stage analysis pipeline.
        
        Args:
            results_df: DataFrame with current test results
            historical_df: Optional DataFrame with historical results for comparison
            primary_kpi: The primary KPI (awareness, consideration, etc.)
            
        Returns:
            AdvancedAnalysisResult with all findings
        """
        # Normalize column names
        results_df = self._normalize_columns(results_df.copy())
        
        # Extract lift and diagnostic columns
        lift_col = f"{primary_kpi}_lift_pct" if f"{primary_kpi}_lift_pct" in results_df.columns else f"{primary_kpi}_lift"
        if lift_col not in results_df.columns:
            # Try to find any lift column
            lift_cols = [c for c in results_df.columns if 'lift' in c.lower() and primary_kpi in c.lower()]
            lift_col = lift_cols[0] if lift_cols else 'awareness_lift_pct'
        
        # Get passed column
        if 'passed' not in results_df.columns:
            sig_col = f"{primary_kpi}_significant" if f"{primary_kpi}_significant" in results_df.columns else f"{primary_kpi}_stat_sig"
            if sig_col in results_df.columns and lift_col in results_df.columns:
                results_df['passed'] = (results_df[sig_col] == True) & (results_df[lift_col] > 0)
            else:
                results_df['passed'] = False
        
        # Stage 1: Statistical Analysis
        statistical = self._stage1_statistical_analysis(results_df, lift_col)
        
        # Stage 2: Historical Comparison
        historical = self._stage2_historical_comparison(results_df, historical_df, lift_col)
        
        # Stage 3: Pattern Mining
        patterns = self._stage3_pattern_mining(results_df, lift_col)
        
        # Generate per-creative recommendations
        recommendations = self._generate_recommendations(
            results_df, statistical, historical, patterns, lift_col
        )
        
        # Build result
        result = AdvancedAnalysisResult(
            statistical=statistical,
            historical=historical,
            patterns=patterns,
            recommendations=recommendations,
            raw_data={
                'results_df': results_df.to_dict('records'),
                'lift_column': lift_col,
                'primary_kpi': primary_kpi,
            }
        )
        
        # Generate optimization playbook
        result.optimization_playbook = self._generate_optimization_playbook(
            statistical, patterns
        )
        
        return result
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase with underscores."""
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        return df
    
    def _get_diagnostic_columns(self, df: pd.DataFrame) -> list:
        """Get list of diagnostic columns present in the dataframe."""
        present = []
        for col in self.DIAGNOSTIC_COLUMNS:
            if col in df.columns:
                present.append(col)
        return present
    
    # =========================================================================
    # Stage 1: Statistical Analysis
    # =========================================================================
    
    def _stage1_statistical_analysis(
        self, 
        df: pd.DataFrame, 
        lift_col: str
    ) -> StatisticalFindings:
        """
        Perform statistical analysis on the test results.
        
        - Correlation analysis
        - Regression analysis
        - Significance testing between pass/fail groups
        - Effect size calculation
        """
        findings = StatisticalFindings()
        
        if not SCIPY_AVAILABLE:
            findings.key_predictors = ["Statistical analysis unavailable (scipy not installed)"]
            return findings
        
        diagnostic_cols = self._get_diagnostic_columns(df)
        
        if not diagnostic_cols or lift_col not in df.columns:
            return findings
        
        # Clean data - remove NaN
        analysis_df = df[[lift_col] + diagnostic_cols + ['passed']].dropna()
        
        if len(analysis_df) < 3:
            findings.key_predictors = ["Insufficient data for statistical analysis"]
            return findings
        
        lift_values = analysis_df[lift_col].values
        
        # 1. Correlation Analysis
        correlations = {}
        pvalues = {}
        for col in diagnostic_cols:
            try:
                corr, pval = pearsonr(analysis_df[col].values, lift_values)
                correlations[col] = round(corr, 3)
                pvalues[col] = round(pval, 4)
            except Exception:
                continue
        
        findings.correlations = correlations
        findings.correlation_pvalues = pvalues
        
        # 2. Regression Analysis (if sklearn available)
        if SKLEARN_AVAILABLE and len(diagnostic_cols) >= 1:
            try:
                X = analysis_df[diagnostic_cols].values
                y = lift_values
                
                # Linear regression for lift prediction
                reg = LinearRegression()
                reg.fit(X, y)
                
                findings.regression_r_squared = round(reg.score(X, y), 3)
                findings.regression_coefficients = {
                    col: round(coef, 4) 
                    for col, coef in zip(diagnostic_cols, reg.coef_)
                }
            except Exception:
                pass
        
        # 3. Significance Testing (pass vs fail groups)
        passed_df = analysis_df[analysis_df['passed'] == True]
        failed_df = analysis_df[analysis_df['passed'] == False]
        
        if len(passed_df) >= 2 and len(failed_df) >= 2:
            significant_diffs = []
            effect_sizes = {}
            
            for col in diagnostic_cols:
                try:
                    passed_vals = passed_df[col].values
                    failed_vals = failed_df[col].values
                    
                    # t-test
                    t_stat, p_val = ttest_ind(passed_vals, failed_vals)
                    
                    # Cohen's d effect size
                    pooled_std = np.sqrt(
                        ((len(passed_vals) - 1) * np.std(passed_vals, ddof=1)**2 +
                         (len(failed_vals) - 1) * np.std(failed_vals, ddof=1)**2) /
                        (len(passed_vals) + len(failed_vals) - 2)
                    )
                    if pooled_std > 0:
                        cohens_d = (np.mean(passed_vals) - np.mean(failed_vals)) / pooled_std
                    else:
                        cohens_d = 0
                    
                    effect_sizes[col] = round(cohens_d, 2)
                    
                    if p_val < 0.1:  # Include marginally significant
                        significant_diffs.append({
                            'metric': col,
                            'metric_name': self.DIAGNOSTIC_NAMES.get(col, col),
                            'p_value': round(p_val, 4),
                            'passed_mean': round(np.mean(passed_vals), 1),
                            'failed_mean': round(np.mean(failed_vals), 1),
                            'effect_size': round(cohens_d, 2),
                            'effect_interpretation': self._interpret_effect_size(cohens_d)
                        })
                except Exception:
                    continue
            
            findings.significant_differences = sorted(
                significant_diffs, key=lambda x: x['p_value']
            )
            findings.effect_sizes = effect_sizes
        
        # 4. Rank key predictors
        key_predictors = []
        for col in diagnostic_cols:
            score = 0
            if col in correlations:
                score += abs(correlations[col]) * 100
            if col in findings.effect_sizes:
                score += abs(findings.effect_sizes[col]) * 50
            key_predictors.append((col, score))
        
        key_predictors.sort(key=lambda x: x[1], reverse=True)
        findings.key_predictors = [
            {
                'metric': col,
                'metric_name': self.DIAGNOSTIC_NAMES.get(col, col),
                'correlation': correlations.get(col, 0),
                'effect_size': findings.effect_sizes.get(col, 0),
                'importance_score': round(score, 1)
            }
            for col, score in key_predictors[:5]
        ]
        
        # 5. Confidence intervals for lift estimates
        for idx, row in df.iterrows():
            creative_name = row.get('creative_name', f'Creative {idx}')
            lift = row.get(lift_col, 0)
            # Approximate CI using standard error (simplified)
            # In practice, would use proper bootstrap or analytical CI
            se = abs(lift) * 0.3  # Rough approximation
            findings.confidence_intervals[creative_name] = (
                round(lift - 1.96 * se, 2),
                round(lift + 1.96 * se, 2)
            )
        
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
    
    # =========================================================================
    # Stage 2: Historical Comparison
    # =========================================================================
    
    def _stage2_historical_comparison(
        self,
        df: pd.DataFrame,
        historical_df: Optional[pd.DataFrame],
        lift_col: str
    ) -> HistoricalContext:
        """
        Compare current results to historical data.
        
        - Percentile rankings
        - Trend analysis
        - Pattern matching
        """
        context = HistoricalContext()
        
        # Current pass rate
        if 'passed' in df.columns:
            context.current_pass_rate = round(df['passed'].mean() * 100, 1)
        
        # If no historical data, provide context from current test only
        if historical_df is None or len(historical_df) == 0:
            context.historical_pass_rate = 50.0  # Assume industry average
            context.trend_direction = "unknown"
            
            # Calculate percentiles within current test
            if lift_col in df.columns:
                lifts = df[lift_col].values
                for idx, row in df.iterrows():
                    creative_name = row.get('creative_name', f'Creative {idx}')
                    lift = row.get(lift_col, 0)
                    percentile = (lifts < lift).sum() / len(lifts) * 100
                    context.percentile_ranks[creative_name] = round(percentile, 0)
            
            return context
        
        # With historical data
        historical_df = self._normalize_columns(historical_df.copy())
        
        # Historical pass rate
        if 'passed' in historical_df.columns:
            context.historical_pass_rate = round(historical_df['passed'].mean() * 100, 1)
        
        # Percentile rankings vs history
        hist_lift_col = lift_col if lift_col in historical_df.columns else 'awareness_lift_pct'
        if hist_lift_col in historical_df.columns:
            historical_lifts = historical_df[hist_lift_col].dropna().values
            
            for idx, row in df.iterrows():
                creative_name = row.get('creative_name', f'Creative {idx}')
                lift = row.get(lift_col, 0)
                if len(historical_lifts) > 0:
                    percentile = (historical_lifts < lift).sum() / len(historical_lifts) * 100
                    context.percentile_ranks[creative_name] = round(percentile, 0)
        
        # Trend analysis (if dates available)
        if 'test_date' in historical_df.columns or 'quarter' in historical_df.columns:
            context.trend_direction = self._analyze_trend(historical_df, df)
        
        # Pattern matching - find similar historical creatives
        diagnostic_cols = self._get_diagnostic_columns(df)
        hist_diagnostic_cols = self._get_diagnostic_columns(historical_df)
        common_diagnostics = list(set(diagnostic_cols) & set(hist_diagnostic_cols))
        
        if common_diagnostics and len(historical_df) > 0:
            context.pattern_matches = self._find_pattern_matches(
                df, historical_df, common_diagnostics
            )
        
        return context
    
    def _analyze_trend(
        self, 
        historical_df: pd.DataFrame, 
        current_df: pd.DataFrame
    ) -> str:
        """Analyze if pass rates are trending up or down."""
        if 'passed' not in historical_df.columns:
            return "unknown"
        
        # Simple comparison: current vs historical average
        hist_rate = historical_df['passed'].mean()
        curr_rate = current_df['passed'].mean() if 'passed' in current_df.columns else 0.5
        
        diff = curr_rate - hist_rate
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        else:
            return "stable"
    
    def _find_pattern_matches(
        self,
        current_df: pd.DataFrame,
        historical_df: pd.DataFrame,
        diagnostic_cols: list
    ) -> list:
        """Find historical patterns that match current creatives."""
        patterns = []
        
        # Calculate historical statistics for pass/fail groups
        if 'passed' not in historical_df.columns:
            return patterns
        
        passed_hist = historical_df[historical_df['passed'] == True]
        failed_hist = historical_df[historical_df['passed'] == False]
        
        if len(passed_hist) < 3 or len(failed_hist) < 3:
            return patterns
        
        # Define diagnostic thresholds based on historical data
        for col in diagnostic_cols:
            if col not in historical_df.columns:
                continue
                
            passed_mean = passed_hist[col].mean()
            failed_mean = failed_hist[col].mean()
            threshold = (passed_mean + failed_mean) / 2
            
            # Check which current creatives match fail pattern
            for idx, row in current_df.iterrows():
                creative_name = row.get('creative_name', f'Creative {idx}')
                value = row.get(col, 0)
                
                if value < threshold and failed_mean < passed_mean:
                    # Matches historical fail pattern
                    metric_name = self.DIAGNOSTIC_NAMES.get(col, col)
                    
                    # Calculate historical pass rate for this pattern
                    similar = historical_df[historical_df[col] < threshold]
                    if len(similar) > 0 and 'passed' in similar.columns:
                        pattern_pass_rate = similar['passed'].mean() * 100
                        
                        patterns.append({
                            'creative_name': creative_name,
                            'pattern': f"Low {metric_name} (<{threshold:.0f})",
                            'value': round(value, 1),
                            'threshold': round(threshold, 1),
                            'historical_pass_rate': round(pattern_pass_rate, 0),
                            'historical_sample_size': len(similar)
                        })
        
        return patterns
    
    # =========================================================================
    # Stage 3: Pattern Mining
    # =========================================================================
    
    def _stage3_pattern_mining(
        self,
        df: pd.DataFrame,
        lift_col: str
    ) -> PatternMiningResults:
        """
        Mine patterns using ML techniques.
        
        - Decision tree rules
        - Feature importance
        - Clustering
        """
        results = PatternMiningResults()
        
        if not SKLEARN_AVAILABLE:
            results.decision_rules = ["Pattern mining unavailable (scikit-learn not installed)"]
            return results
        
        diagnostic_cols = self._get_diagnostic_columns(df)
        
        if not diagnostic_cols or 'passed' not in df.columns:
            return results
        
        # Prepare data
        analysis_df = df[diagnostic_cols + ['passed']].dropna()
        
        if len(analysis_df) < 4:
            results.decision_rules = ["Insufficient data for pattern mining (need at least 4 samples)"]
            return results
        
        X = analysis_df[diagnostic_cols].values
        y = analysis_df['passed'].astype(int).values
        
        # 1. Decision Tree for interpretable rules
        try:
            tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1)
            tree.fit(X, y)
            
            # Extract rules
            tree_rules = export_text(tree, feature_names=diagnostic_cols)
            results.decision_rules = self._parse_tree_rules(tree_rules, diagnostic_cols)
            
            # Calculate thresholds
            results.diagnostic_thresholds = self._extract_thresholds(tree, diagnostic_cols)
        except Exception as e:
            results.decision_rules = [f"Could not generate decision rules: {str(e)}"]
        
        # 2. Feature Importance (Random Forest)
        try:
            rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
            rf.fit(X, y)
            
            importances = rf.feature_importances_
            results.feature_importance = {
                self.DIAGNOSTIC_NAMES.get(col, col): round(imp, 3)
                for col, imp in sorted(
                    zip(diagnostic_cols, importances),
                    key=lambda x: x[1],
                    reverse=True
                )
            }
        except Exception:
            pass
        
        # 3. Clustering (if enough data)
        if len(analysis_df) >= 4:
            try:
                n_clusters = min(3, len(analysis_df))
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                
                # Analyze each cluster
                analysis_df_with_clusters = analysis_df.copy()
                analysis_df_with_clusters['cluster'] = clusters
                
                for i in range(n_clusters):
                    cluster_df = analysis_df_with_clusters[analysis_df_with_clusters['cluster'] == i]
                    
                    if len(cluster_df) == 0:
                        continue
                    
                    # Find cluster characteristics
                    characteristics = []
                    for col in diagnostic_cols:
                        cluster_mean = cluster_df[col].mean()
                        overall_mean = analysis_df[col].mean()
                        if cluster_mean > overall_mean * 1.1:
                            characteristics.append(f"High {self.DIAGNOSTIC_NAMES.get(col, col)}")
                        elif cluster_mean < overall_mean * 0.9:
                            characteristics.append(f"Low {self.DIAGNOSTIC_NAMES.get(col, col)}")
                    
                    pass_rate = cluster_df['passed'].mean() * 100
                    
                    # Get creative names in this cluster
                    cluster_indices = cluster_df.index.tolist()
                    creative_names = df.loc[cluster_indices, 'creative_name'].tolist() if 'creative_name' in df.columns else []
                    
                    results.clusters.append({
                        'cluster_id': i,
                        'name': self._name_cluster(characteristics, pass_rate),
                        'size': len(cluster_df),
                        'pass_rate': round(pass_rate, 0),
                        'characteristics': characteristics[:3],  # Top 3
                        'creatives': creative_names
                    })
            except Exception:
                pass
        
        # 4. Winning combinations (simple rule extraction)
        results.winning_combinations = self._find_winning_combinations(analysis_df, diagnostic_cols)
        
        return results
    
    def _parse_tree_rules(self, tree_text: str, feature_names: list) -> list:
        """Parse decision tree text into human-readable rules."""
        rules = []
        lines = tree_text.strip().split('\n')
        
        # Simple parsing - extract leaf nodes with their conditions
        current_conditions = []
        
        for line in lines:
            depth = len(line) - len(line.lstrip('|'))
            content = line.strip('| ').strip()
            
            if '<=' in content or '>' in content:
                # This is a condition
                current_conditions = current_conditions[:depth//4]
                current_conditions.append(content)
            elif 'class:' in content:
                # This is a leaf node
                class_val = int(content.split(':')[1].strip())
                if current_conditions:
                    conditions_str = ' AND '.join(current_conditions)
                    # Replace feature indices with names
                    for i, name in enumerate(feature_names):
                        conditions_str = conditions_str.replace(
                            f'feature_{i}', 
                            self.DIAGNOSTIC_NAMES.get(name, name)
                        )
                    result = "PASS" if class_val == 1 else "FAIL"
                    rules.append(f"IF {conditions_str} THEN {result}")
        
        return rules[:5]  # Top 5 rules
    
    def _extract_thresholds(self, tree, feature_names: list) -> dict:
        """Extract optimal thresholds from decision tree."""
        thresholds = {}
        
        if hasattr(tree, 'tree_'):
            tree_model = tree.tree_
            for i, (feature_idx, threshold) in enumerate(zip(tree_model.feature, tree_model.threshold)):
                if feature_idx >= 0 and feature_idx < len(feature_names):
                    feature_name = feature_names[feature_idx]
                    if feature_name not in thresholds:
                        thresholds[feature_name] = round(threshold, 1)
        
        return thresholds
    
    def _name_cluster(self, characteristics: list, pass_rate: float) -> str:
        """Generate a descriptive name for a cluster."""
        if pass_rate >= 80:
            if 'High Attention' in characteristics:
                return "Hook Masters"
            elif 'High Emotional' in ' '.join(characteristics):
                return "Emotional Connectors"
            else:
                return "Top Performers"
        elif pass_rate >= 50:
            return "Middle Performers"
        else:
            if 'Low Attention' in characteristics:
                return "Attention Challenged"
            else:
                return "Struggling Creatives"
    
    def _find_winning_combinations(self, df: pd.DataFrame, diagnostic_cols: list) -> list:
        """Find diagnostic combinations that predict success."""
        combinations = []
        
        if 'passed' not in df.columns or len(df) < 4:
            return combinations
        
        # Check single thresholds
        for col in diagnostic_cols:
            if col not in df.columns:
                continue
            
            # Find threshold that maximizes separation
            median_val = df[col].median()
            high_group = df[df[col] >= median_val]
            
            if len(high_group) > 0 and 'passed' in high_group.columns:
                pass_rate = high_group['passed'].mean() * 100
                
                if pass_rate >= 70:
                    combinations.append({
                        'condition': f"{self.DIAGNOSTIC_NAMES.get(col, col)} ≥ {median_val:.0f}",
                        'pass_rate': round(pass_rate, 0),
                        'sample_size': len(high_group)
                    })
        
        # Sort by pass rate
        combinations.sort(key=lambda x: x['pass_rate'], reverse=True)
        
        return combinations[:5]
    
    # =========================================================================
    # Generate Recommendations
    # =========================================================================
    
    def _generate_recommendations(
        self,
        df: pd.DataFrame,
        statistical: StatisticalFindings,
        historical: HistoricalContext,
        patterns: PatternMiningResults,
        lift_col: str
    ) -> list:
        """Generate specific recommendations for each creative."""
        recommendations = []
        diagnostic_cols = self._get_diagnostic_columns(df)
        
        # Calculate diagnostic averages for comparison
        diagnostic_means = {}
        for col in diagnostic_cols:
            if col in df.columns:
                diagnostic_means[col] = df[col].mean()
        
        for idx, row in df.iterrows():
            creative_name = row.get('creative_name', f'Creative {idx}')
            creative_id = row.get('creative_id', str(idx))
            lift = row.get(lift_col, 0)
            passed = row.get('passed', False)
            
            # Determine category using improved logic
            stat_sig = row.get('awareness_significant', row.get('awareness_stat_sig', False))
            
            if passed or (stat_sig and lift > 0):
                category = "run"
            elif stat_sig and lift <= 0:
                category = "do_not_run"
            elif lift < -2:  # Clearly negative even if not significant
                category = "do_not_run"
            else:
                category = "optimize_retest"
            
            # Find strengths and weaknesses
            strengths = []
            weaknesses = []
            
            for col in diagnostic_cols:
                if col not in df.columns:
                    continue
                    
                value = row.get(col, 0)
                mean_val = diagnostic_means.get(col, 50)
                metric_name = self.DIAGNOSTIC_NAMES.get(col, col)
                
                # Compare to test average and absolute thresholds
                threshold = patterns.diagnostic_thresholds.get(col, 60)
                
                if value >= mean_val * 1.15 and value >= 65:
                    strengths.append(f"{metric_name} ({value:.0f})")
                elif value < mean_val * 0.85 or value < threshold:
                    weaknesses.append(f"{metric_name} ({value:.0f})")
            
            # Generate specific fixes based on weaknesses
            specific_fixes = self._generate_fixes(weaknesses, statistical.key_predictors)
            
            # Find pattern match
            pattern_match = None
            for pm in historical.pattern_matches:
                if pm['creative_name'] == creative_name:
                    pattern_match = f"{pm['pattern']} - historically {pm['historical_pass_rate']:.0f}% pass rate"
                    break
            
            recommendations.append(CreativeRecommendation(
                creative_name=creative_name,
                creative_id=creative_id,
                category=category,
                lift=round(lift, 1),
                stat_sig=bool(stat_sig),
                strengths=strengths[:3],
                weaknesses=weaknesses[:3],
                specific_fixes=specific_fixes[:3],
                percentile_rank=historical.percentile_ranks.get(creative_name),
                pattern_match=pattern_match,
                confidence_interval=statistical.confidence_intervals.get(creative_name)
            ))
        
        return recommendations
    
    def _generate_fixes(self, weaknesses: list, key_predictors: list) -> list:
        """Generate specific fixes based on weaknesses."""
        fixes = []
        
        fix_suggestions = {
            'attention': [
                "Strengthen opening hook in first 3 seconds",
                "Add motion or action in the opening",
                "Consider human face in opening frame",
                "Increase visual contrast and movement",
            ],
            'message_clarity': [
                "Simplify to single key message",
                "Reduce competing visual elements",
                "Add clear end card with CTA",
                "Use on-screen text to reinforce message",
            ],
            'emotional': [
                "Add human stories or testimonials",
                "Include relatable moments",
                "Use music to enhance emotional tone",
                "Show product solving real problems",
            ],
            'brand_recall': [
                "Increase logo presence throughout",
                "Add brand cues in first 3 seconds",
                "Use consistent brand colors",
                "Include audio brand mention",
            ],
            'uniqueness': [
                "Differentiate from competitor creative",
                "Add unexpected visual elements",
                "Try unconventional storytelling",
                "Use distinctive music or sound design",
            ],
        }
        
        for weakness in weaknesses:
            weakness_lower = weakness.lower()
            
            for key, suggestions in fix_suggestions.items():
                if key in weakness_lower:
                    fixes.append(suggestions[0])
                    break
        
        # If no specific fixes, add generic ones based on top predictors
        if not fixes and key_predictors:
            top_predictor = key_predictors[0].get('metric', '') if isinstance(key_predictors[0], dict) else ''
            if top_predictor:
                fixes.append(f"Focus on improving {self.DIAGNOSTIC_NAMES.get(top_predictor, top_predictor)}")
        
        return fixes
    
    def _generate_optimization_playbook(
        self,
        statistical: StatisticalFindings,
        patterns: PatternMiningResults
    ) -> list:
        """Generate general optimization playbook based on findings."""
        playbook = []
        
        # Based on key predictors
        for predictor in statistical.key_predictors[:3]:
            if isinstance(predictor, dict):
                metric = predictor.get('metric', '')
                metric_name = predictor.get('metric_name', metric)
                correlation = predictor.get('correlation', 0)
                
                if correlation > 0.3:
                    playbook.append({
                        'to_improve': metric_name,
                        'importance': 'High' if correlation > 0.6 else 'Medium',
                        'correlation': correlation,
                    })
        
        # Add threshold recommendations
        for metric, threshold in patterns.diagnostic_thresholds.items():
            metric_name = self.DIAGNOSTIC_NAMES.get(metric, metric)
            playbook.append({
                'to_improve': metric_name,
                'target_threshold': threshold,
                'recommendation': f"Ensure {metric_name} score is above {threshold:.0f}",
            })
        
        return playbook


# Singleton instance
_analysis_service = None


def get_analysis_service() -> AdvancedAnalysisService:
    """Get singleton analysis service instance."""
    global _analysis_service
    if _analysis_service is None:
        _analysis_service = AdvancedAnalysisService()
    return _analysis_service
