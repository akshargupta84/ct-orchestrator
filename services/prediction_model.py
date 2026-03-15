"""
Prediction Model for Creative Scoring.

Uses machine learning to predict creative test outcomes before actual testing.
Combines structured features (from video analysis) with historical patterns.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

# ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from .local_vision import VideoAnalysisResult, FrameAnalysis


@dataclass
class PredictionFeatures:
    """Features extracted for prediction."""
    # Video structure
    duration_seconds: float = 0
    frame_count: int = 0
    
    # Human presence (most important!)
    has_human_in_opening: bool = False
    first_human_appearance_sec: float = -1
    human_frame_ratio: float = 0  # % of frames with humans
    human_looking_at_camera_ratio: float = 0
    
    # Emotions detected
    has_positive_emotion: bool = False
    has_emotional_content: bool = False
    dominant_emotion: str = ""
    
    # Brand elements
    logo_first_appearance_sec: float = -1
    logo_in_first_3_sec: bool = False
    logo_frame_ratio: float = 0
    product_visible_ratio: float = 0
    
    # CTA
    has_cta: bool = False
    cta_in_last_5_sec: bool = False
    
    # Scene composition
    scene_type_diversity: int = 0  # Number of different scene types
    dominant_scene_type: str = ""
    dominant_mood: str = ""
    
    # Pacing
    avg_scene_duration: float = 0  # Estimated from scene changes
    
    # Visual style
    visual_complexity_score: float = 0  # 0-1
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'duration_seconds': self.duration_seconds,
            'frame_count': self.frame_count,
            'has_human_in_opening': int(self.has_human_in_opening),
            'first_human_appearance_sec': self.first_human_appearance_sec,
            'human_frame_ratio': self.human_frame_ratio,
            'human_looking_at_camera_ratio': self.human_looking_at_camera_ratio,
            'has_positive_emotion': int(self.has_positive_emotion),
            'has_emotional_content': int(self.has_emotional_content),
            'logo_first_appearance_sec': self.logo_first_appearance_sec,
            'logo_in_first_3_sec': int(self.logo_in_first_3_sec),
            'logo_frame_ratio': self.logo_frame_ratio,
            'product_visible_ratio': self.product_visible_ratio,
            'has_cta': int(self.has_cta),
            'cta_in_last_5_sec': int(self.cta_in_last_5_sec),
            'scene_type_diversity': self.scene_type_diversity,
            'visual_complexity_score': self.visual_complexity_score,
        }
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        d = self.to_dict()
        return np.array(list(d.values())).reshape(1, -1)


@dataclass
class DiagnosticPrediction:
    """Predicted diagnostic score."""
    name: str
    predicted_value: float
    confidence_interval: Tuple[float, float]
    benchmark: float
    status: str  # "good", "warning", "poor"
    
    
@dataclass
class SimilarCreative:
    """A similar historical creative."""
    name: str
    similarity_score: float
    lift: float
    passed: bool
    key_similarities: List[str]


@dataclass
class RiskFactor:
    """A risk factor identified in the creative."""
    factor: str
    impact: str
    confidence: str  # "high", "medium", "low"
    evidence: str


@dataclass
class Recommendation:
    """An actionable recommendation."""
    priority: str  # "high", "medium", "low"
    action: str
    expected_impact: str
    rationale: str


@dataclass
class CreativeScore:
    """Complete scoring result for a creative."""
    # Core predictions
    pass_probability: float
    confidence_interval: Tuple[float, float]
    
    predicted_lift: float
    lift_range: Tuple[float, float]
    
    # Diagnostic predictions
    predicted_diagnostics: Dict[str, DiagnosticPrediction] = field(default_factory=dict)
    
    # Risk analysis
    risk_factors: List[RiskFactor] = field(default_factory=list)
    risk_level: str = "medium"  # "low", "medium", "high"
    
    # Historical comparison
    similar_creatives: List[SimilarCreative] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[Recommendation] = field(default_factory=list)
    
    # Features used
    features: PredictionFeatures = None
    
    # AI summary
    ai_summary: str = ""
    
    # Metadata
    model_version: str = "1.0"
    scoring_timestamp: str = ""
    video_analysis: VideoAnalysisResult = None


class CreativePredictionModel:
    """
    Machine learning model for predicting creative test outcomes.
    """
    
    # Feature names in order
    FEATURE_NAMES = [
        'duration_seconds',
        'frame_count',
        'has_human_in_opening',
        'first_human_appearance_sec',
        'human_frame_ratio',
        'human_looking_at_camera_ratio',
        'has_positive_emotion',
        'has_emotional_content',
        'logo_first_appearance_sec',
        'logo_in_first_3_sec',
        'logo_frame_ratio',
        'product_visible_ratio',
        'has_cta',
        'cta_in_last_5_sec',
        'scene_type_diversity',
        'visual_complexity_score',
    ]
    
    # Historical benchmarks for diagnostics (defaults - will be overridden by data if available)
    DIAGNOSTIC_BENCHMARKS = {
        'attention_score': 60,
        'message_clarity_score': 60,
        'emotional_resonance_score': 55,
        'brand_recall_score': 55,
        'uniqueness_score': 50,
    }
    
    # Track if benchmarks are from data or defaults
    benchmarks_from_data: bool = False
    
    @classmethod
    def load_benchmarks_from_historical_data(cls, data_dir: str = "data") -> bool:
        """
        Calculate benchmarks from historical test results.
        Uses median of passed creatives as the benchmark.
        
        Returns True if benchmarks were updated from data.
        """
        try:
            from services.persistence import get_persistence_service
            persistence = get_persistence_service()
            
            # Collect all results
            all_scores = {
                'attention_score': [],
                'message_clarity_score': [],
                'emotional_resonance_score': [],
                'brand_recall_score': [],
                'uniqueness_score': [],
            }
            
            results_list = persistence.list_results()
            for result_info in results_list:
                result_data = persistence.load_results(result_info['results_id'])
                if result_data and 'results' in result_data:
                    for r in result_data['results']:
                        # Only use passed creatives for benchmarks
                        if r.get('passed', False):
                            for score_name in all_scores.keys():
                                if score_name in r and r[score_name]:
                                    all_scores[score_name].append(float(r[score_name]))
            
            # Calculate median for each score (need at least 5 data points)
            updated = False
            for score_name, values in all_scores.items():
                if len(values) >= 5:
                    median_val = np.median(values)
                    cls.DIAGNOSTIC_BENCHMARKS[score_name] = round(median_val, 0)
                    updated = True
            
            cls.benchmarks_from_data = updated
            return updated
            
        except Exception as e:
            print(f"Could not load historical benchmarks: {e}")
            return False
    
    # Feature importance for diagnostics (learned or heuristic)
    DIAGNOSTIC_FEATURE_WEIGHTS = {
        'attention_score': {
            'has_human_in_opening': 15,
            'human_frame_ratio': 10,
            'visual_complexity_score': -5,  # Too complex = lower attention
            'human_looking_at_camera_ratio': 8,
        },
        'message_clarity_score': {
            'has_cta': 10,
            'scene_type_diversity': -5,  # Too many scene types = less clear
            'cta_in_last_5_sec': 8,
        },
        'emotional_resonance_score': {
            'has_emotional_content': 15,
            'has_positive_emotion': 10,
            'human_frame_ratio': 8,
        },
        'brand_recall_score': {
            'logo_in_first_3_sec': 12,
            'logo_frame_ratio': 10,
            'product_visible_ratio': 8,
        },
        'uniqueness_score': {
            'scene_type_diversity': 5,
            'visual_complexity_score': 5,
        },
    }
    
    def __init__(self, model_path: str = None):
        """
        Initialize the prediction model.
        
        Args:
            model_path: Path to saved model (or None to use heuristic model)
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.training_stats = {}  # Stats about the trained model
        
        # Historical data for comparison
        self.historical_features = []
        self.historical_results = []
        
        # Feature importance learned from data
        self.learned_feature_importance = {}
        
        # Diagnostic prediction models (video features → diagnostic scores)
        self.diagnostic_models = {}  # {diagnostic_name: model}
        self.diagnostic_models_trained = False
        self.diagnostic_training_stats = {}
        
        # Video features for diagnostic prediction
        self.video_features_for_diagnostics = []
        self.actual_diagnostics = []
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        
        # Load historical data and train models
        self._load_historical_data()
        self._train_diagnostic_models()  # NEW: Train diagnostic predictors
        self._train_from_historical_data()
    
    def _load_historical_data(self):
        """Load historical test results AND video features for ML training."""
        try:
            from services.persistence import get_persistence_service
            persistence = get_persistence_service()
            
            # Load all video features (from Ollama analysis)
            video_features_map = {}
            all_video_features = persistence.load_all_video_features()
            for vf in all_video_features:
                creative_id = vf.get('creative_id')
                if creative_id:
                    video_features_map[creative_id] = vf
            
            print(f"Loaded video features for {len(video_features_map)} creatives")
            
            # Load test results and join with video features
            results_list = persistence.list_results()
            for result_info in results_list:
                result_data = persistence.load_results(result_info['results_id'])
                if result_data and 'results' in result_data:
                    campaign_name = result_data.get('campaign_name', 'Unknown')
                    
                    for r in result_data['results']:
                        creative_id = r.get('creative_id', '')
                        
                        # Diagnostic scores from test results (actual values)
                        actual_diags = {
                            'attention_score': r.get('attention_score', 50),
                            'message_clarity_score': r.get('message_clarity_score', 50),
                            'emotional_resonance_score': r.get('emotional_resonance_score', 50),
                            'brand_recall_score': r.get('brand_recall_score', 50),
                            'uniqueness_score': r.get('uniqueness_score', 50),
                        }
                        
                        # Combined features for pass/fail model
                        features = actual_diags.copy()
                        
                        # Add video features if available (from Ollama)
                        if creative_id in video_features_map:
                            vf = video_features_map[creative_id]
                            
                            # Video features dict
                            video_feats = {
                                'has_human_in_opening': 1 if vf.get('has_human_in_opening') else 0,
                                'first_human_appearance_sec': vf.get('first_human_appearance_sec', -1),
                                'human_frame_ratio': vf.get('human_frame_ratio', 0),
                                'human_looking_at_camera_ratio': vf.get('human_looking_at_camera_ratio', 0),
                                'has_positive_emotion': 1 if vf.get('has_positive_emotion') else 0,
                                'has_emotional_content': 1 if vf.get('has_emotional_content') else 0,
                                'logo_in_first_3_sec': 1 if vf.get('logo_in_first_3_sec') else 0,
                                'logo_frame_ratio': vf.get('logo_frame_ratio', 0),
                                'product_visible_ratio': vf.get('product_visible_ratio', 0),
                                'has_cta': 1 if vf.get('has_cta') else 0,
                                'cta_in_last_5_sec': 1 if vf.get('cta_in_last_5_sec') else 0,
                                'scene_type_diversity': vf.get('scene_type_diversity', 0),
                                'visual_complexity_score': vf.get('visual_complexity_score', 50),
                            }
                            
                            features.update(video_feats)
                            
                            # Store for diagnostic model training (video features → diagnostics)
                            self.video_features_for_diagnostics.append(video_feats)
                            self.actual_diagnostics.append(actual_diags)
                        
                        self.historical_features.append(features)
                        
                        # Store result info
                        self.historical_results.append({
                            'creative_id': creative_id,
                            'name': f"{r.get('creative_name', 'Unknown')} ({campaign_name})",
                            'creative_name': r.get('creative_name', 'Unknown'),
                            'campaign': campaign_name,
                            'passed': r.get('passed', False),
                            'lift': r.get('awareness_lift_pct', r.get('primary_kpi_lift', 0)),
                            'has_video_features': creative_id in video_features_map,
                        })
            
            n_with_video = sum(1 for r in self.historical_results if r.get('has_video_features'))
            print(f"Loaded {len(self.historical_results)} historical creatives ({n_with_video} with video features)")
            
        except Exception as e:
            print(f"Could not load historical data: {e}")
            import traceback
            traceback.print_exc()
    
    def _train_diagnostic_models(self):
        """
        Train models to predict diagnostic scores from video features.
        
        This allows us to predict diagnostics (attention_score, brand_recall_score, etc.)
        for NEW creatives based only on their video features extracted by Ollama.
        
        Models: One Ridge Regression per diagnostic score
        """
        if not SKLEARN_AVAILABLE:
            print("sklearn not available - cannot train diagnostic models")
            return
        
        if len(self.video_features_for_diagnostics) < 10:
            print(f"Not enough data for diagnostic models ({len(self.video_features_for_diagnostics)} samples, need 10+)")
            return
        
        try:
            from sklearn.linear_model import Ridge
            from sklearn.model_selection import cross_val_score
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            
            # Build feature matrix from video features
            X = pd.DataFrame(self.video_features_for_diagnostics)
            
            # Replace -1 sentinel with NaN
            X = X.replace(-1, np.nan)
            
            # Store feature names for later
            self.diagnostic_feature_names = X.columns.tolist()
            
            # Train a model for each diagnostic
            diagnostic_names = ['attention_score', 'message_clarity_score', 
                              'emotional_resonance_score', 'brand_recall_score', 'uniqueness_score']
            
            self.diagnostic_training_stats = {}
            
            for diag_name in diagnostic_names:
                # Target: actual diagnostic score
                y = np.array([d[diag_name] for d in self.actual_diagnostics])
                
                # Create pipeline with imputation + Ridge regression
                model = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('regressor', Ridge(alpha=1.0))  # Regularized to prevent overfitting
                ])
                
                # Cross-validation to estimate performance
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                cv_r2 = cv_scores.mean()
                
                # Also calculate RMSE
                from sklearn.model_selection import cross_val_predict
                cv_predictions = cross_val_predict(model, X, y, cv=5)
                rmse = np.sqrt(((cv_predictions - y) ** 2).mean())
                
                # Fit on all data
                model.fit(X, y)
                self.diagnostic_models[diag_name] = model
                
                # Get feature importance from Ridge coefficients
                coefs = model.named_steps['regressor'].coef_
                importance = {name: round(abs(float(c)), 4) 
                            for name, c in zip(self.diagnostic_feature_names, coefs)}
                
                self.diagnostic_training_stats[diag_name] = {
                    'cv_r2': round(cv_r2, 3),
                    'rmse': round(rmse, 1),
                    'top_predictors': sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
                }
            
            self.diagnostic_models_trained = True
            
            print(f"✓ Trained diagnostic prediction models on {len(X)} creatives")
            for diag_name, stats in self.diagnostic_training_stats.items():
                top_preds = [p[0].replace('_', ' ') for p in stats['top_predictors']]
                print(f"  {diag_name}: R²={stats['cv_r2']:.2f}, RMSE={stats['rmse']:.1f}, predictors: {top_preds}")
            
        except Exception as e:
            print(f"Could not train diagnostic models: {e}")
            import traceback
            traceback.print_exc()
    
    def predict_diagnostics_from_video(self, features: 'PredictionFeatures') -> dict:
        """
        Predict diagnostic scores from video features using trained models.
        
        Args:
            features: PredictionFeatures extracted from video
            
        Returns:
            Dict of predicted diagnostic scores
        """
        if not self.diagnostic_models_trained:
            # Fallback to heuristic
            return self._predict_diagnostics_heuristic(features)
        
        try:
            # Build feature vector from video features
            feature_dict = {
                'has_human_in_opening': 1 if features.has_human_in_opening else 0,
                'first_human_appearance_sec': features.first_human_appearance_sec,
                'human_frame_ratio': features.human_frame_ratio,
                'human_looking_at_camera_ratio': features.human_looking_at_camera_ratio,
                'has_positive_emotion': 1 if features.has_positive_emotion else 0,
                'has_emotional_content': 1 if features.has_emotional_content else 0,
                'logo_in_first_3_sec': 1 if features.logo_in_first_3_sec else 0,
                'logo_frame_ratio': features.logo_frame_ratio,
                'product_visible_ratio': features.product_visible_ratio,
                'has_cta': 1 if features.has_cta else 0,
                'cta_in_last_5_sec': 1 if features.cta_in_last_5_sec else 0,
                'scene_type_diversity': features.scene_type_diversity,
                'visual_complexity_score': features.visual_complexity_score,
            }
            
            X = pd.DataFrame([feature_dict])[self.diagnostic_feature_names]
            X = X.replace(-1, np.nan)
            
            # Predict each diagnostic
            predicted = {}
            for diag_name, model in self.diagnostic_models.items():
                pred = model.predict(X)[0]
                # Clamp to valid range
                predicted[diag_name] = max(0, min(100, pred))
            
            return predicted
            
        except Exception as e:
            print(f"Error predicting diagnostics: {e}")
            return self._predict_diagnostics_heuristic(features)
    
    def _predict_diagnostics_heuristic(self, features: 'PredictionFeatures') -> dict:
        """Fallback heuristic diagnostic prediction."""
        predicted = {}
        for diag_name, benchmark in self.DIAGNOSTIC_BENCHMARKS.items():
            value = benchmark
            weights = self.DIAGNOSTIC_FEATURE_WEIGHTS.get(diag_name, {})
            for feature_name, weight in weights.items():
                if hasattr(features, feature_name):
                    feat_val = getattr(features, feature_name)
                    if isinstance(feat_val, bool):
                        value += weight if feat_val else 0
                    elif isinstance(feat_val, (int, float)):
                        value += weight * feat_val
            predicted[diag_name] = max(0, min(100, value))
        return predicted

    def _train_from_historical_data(self):
        """
        Train ensemble prediction model from historical test results.
        
        Uses:
        - Logistic Regression (interpretable, regularized)
        - Random Forest (captures interactions)
        - Ensemble: Average predictions from both
        
        Validation: Leave-One-Out Cross-Validation (LOOCV)
        """
        if not SKLEARN_AVAILABLE:
            print("sklearn not available - using heuristic model")
            return
        
        if len(self.historical_features) < 10:
            print(f"Not enough historical data to train ({len(self.historical_features)} samples, need 10+)")
            return
        
        try:
            from sklearn.model_selection import LeaveOneOut, cross_val_predict
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            from sklearn.ensemble import VotingClassifier
            
            # Build training data
            X = pd.DataFrame(self.historical_features)
            y = np.array([r['passed'] for r in self.historical_results]).astype(int)
            
            # Track which features we're using
            self.feature_names = X.columns.tolist()
            print(f"Training with {len(self.feature_names)} features: {self.feature_names}")
            
            # Handle missing values (some creatives may not have video features)
            # Replace -1 (sentinel for "not detected") with NaN for imputation
            X = X.replace(-1, np.nan)
            
            # Create preprocessing + model pipelines
            # Logistic Regression with L2 regularization
            lr_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    C=0.5,  # Regularization strength
                    class_weight='balanced',  # Handle class imbalance
                    max_iter=1000,
                    random_state=42
                ))
            ])
            
            # Random Forest with limited depth
            rf_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=4,  # Limit depth to prevent overfitting
                    min_samples_leaf=3,  # Require at least 3 samples per leaf
                    class_weight='balanced',
                    random_state=42
                ))
            ])
            
            # Ensemble: Soft voting (average probabilities)
            self.model = VotingClassifier(
                estimators=[
                    ('lr', lr_pipeline),
                    ('rf', rf_pipeline)
                ],
                voting='soft',
                weights=[0.4, 0.6]  # Slightly favor Random Forest for non-linear patterns
            )
            
            # Leave-One-Out Cross-Validation for honest accuracy estimate
            loo = LeaveOneOut()
            
            # Get cross-validated predictions
            cv_probs = cross_val_predict(
                self.model, X, y, 
                cv=loo, 
                method='predict_proba'
            )
            cv_predictions = (cv_probs[:, 1] >= 0.5).astype(int)
            
            # Calculate metrics
            accuracy = (cv_predictions == y).mean()
            
            # Precision/Recall for pass class
            true_positives = ((cv_predictions == 1) & (y == 1)).sum()
            false_positives = ((cv_predictions == 1) & (y == 0)).sum()
            false_negatives = ((cv_predictions == 0) & (y == 1)).sum()
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            # Now fit on ALL data for final model
            self.model.fit(X, y)
            self.is_trained = True
            
            # Store scaler and imputer for prediction (extract from pipeline)
            self.imputer = SimpleImputer(strategy='median')
            self.imputer.fit(X)
            self.scaler = StandardScaler()
            self.scaler.fit(self.imputer.transform(X))
            
            # Extract feature importance from Random Forest
            rf_model = self.model.named_estimators_['rf'].named_steps['classifier']
            importances = rf_model.feature_importances_
            self.learned_feature_importance = {
                name: round(float(imp), 4) 
                for name, imp in zip(self.feature_names, importances)
            }
            # Sort by importance
            self.learned_feature_importance = dict(
                sorted(self.learned_feature_importance.items(), 
                       key=lambda x: x[1], reverse=True)
            )
            
            # Also get LR coefficients for interpretability
            lr_model = self.model.named_estimators_['lr'].named_steps['classifier']
            self.lr_coefficients = {
                name: round(float(coef), 4)
                for name, coef in zip(self.feature_names, lr_model.coef_[0])
            }
            
            # Training stats
            pass_rate = y.mean()
            self.training_stats = {
                'n_samples': len(X),
                'n_passed': int(y.sum()),
                'n_failed': int(len(y) - y.sum()),
                'pass_rate': round(pass_rate * 100, 1),
                'n_features': len(self.feature_names),
                'n_with_video_features': sum(1 for r in self.historical_results if r.get('has_video_features')),
                'loocv_accuracy': round(accuracy * 100, 1),
                'loocv_precision': round(precision * 100, 1),
                'loocv_recall': round(recall * 100, 1),
                'feature_importance': self.learned_feature_importance,
                'lr_coefficients': self.lr_coefficients,
            }
            
            print(f"✓ Trained ensemble model on {len(X)} historical creatives")
            print(f"  Features: {len(self.feature_names)} ({self.training_stats['n_with_video_features']} with video analysis)")
            print(f"  Pass rate: {self.training_stats['pass_rate']}%")
            print(f"  LOOCV Accuracy: {self.training_stats['loocv_accuracy']}%")
            print(f"  LOOCV Precision: {self.training_stats['loocv_precision']}%")
            print(f"  LOOCV Recall: {self.training_stats['loocv_recall']}%")
            print(f"  Top 3 predictors: {list(self.learned_feature_importance.keys())[:3]}")
            
        except Exception as e:
            print(f"Could not train model: {e}")
            import traceback
            traceback.print_exc()
    
    def extract_features(self, video_analysis: VideoAnalysisResult) -> PredictionFeatures:
        """
        Extract prediction features from video analysis.
        
        Args:
            video_analysis: Complete video analysis result
            
        Returns:
            PredictionFeatures for prediction
        """
        features = PredictionFeatures(
            duration_seconds=video_analysis.duration_seconds,
            frame_count=video_analysis.frame_count,
        )
        
        frame_analyses = video_analysis.frame_analyses
        if not frame_analyses:
            return features
        
        # Human presence analysis
        human_frames = [fa for fa in frame_analyses if fa.humans_present]
        features.human_frame_ratio = len(human_frames) / len(frame_analyses)
        
        looking_at_camera = [fa for fa in human_frames if fa.human_looking_at_camera]
        features.human_looking_at_camera_ratio = (
            len(looking_at_camera) / len(human_frames) if human_frames else 0
        )
        
        features.has_human_in_opening = video_analysis.has_human_in_opening
        features.first_human_appearance_sec = video_analysis.first_human_appearance
        
        # Emotion analysis
        positive_emotions = ['happy', 'joyful', 'excited', 'smile', 'positive', 'enthusiastic']
        emotional_keywords = ['emotional', 'touching', 'heartfelt', 'moving']
        
        for fa in frame_analyses:
            for emotion in fa.human_emotions:
                if any(pe in emotion.lower() for pe in positive_emotions):
                    features.has_positive_emotion = True
                if any(ek in emotion.lower() for ek in emotional_keywords):
                    features.has_emotional_content = True
        
        # Brand elements
        logo_frames = [fa for fa in frame_analyses if fa.logo_visible]
        features.logo_frame_ratio = len(logo_frames) / len(frame_analyses)
        features.logo_first_appearance_sec = video_analysis.logo_first_appearance
        features.logo_in_first_3_sec = (
            video_analysis.logo_first_appearance >= 0 and 
            video_analysis.logo_first_appearance <= 3
        )
        
        product_frames = [fa for fa in frame_analyses if fa.product_visible]
        features.product_visible_ratio = len(product_frames) / len(frame_analyses)
        
        # CTA analysis
        features.has_cta = video_analysis.cta_present
        
        cta_frames = [fa for fa in frame_analyses if fa.cta_present]
        if cta_frames:
            last_cta_time = max(fa.timestamp for fa in cta_frames)
            features.cta_in_last_5_sec = last_cta_time >= video_analysis.duration_seconds - 5
        
        # Scene analysis
        features.scene_type_diversity = len(video_analysis.scene_types)
        features.dominant_scene_type = max(
            video_analysis.scene_types.keys(), 
            key=lambda k: video_analysis.scene_types[k]
        ) if video_analysis.scene_types else ""
        features.dominant_mood = video_analysis.dominant_mood
        
        # Visual complexity
        complexity_map = {'low': 0.25, 'medium': 0.5, 'high': 0.75}
        complexity_scores = [
            complexity_map.get(fa.visual_complexity, 0.5) 
            for fa in frame_analyses
        ]
        features.visual_complexity_score = np.mean(complexity_scores) if complexity_scores else 0.5
        
        return features
    
    def predict(
        self, 
        features: PredictionFeatures,
        video_analysis: VideoAnalysisResult = None,
    ) -> CreativeScore:
        """
        Predict creative test outcome.
        
        Args:
            features: Extracted features
            video_analysis: Original video analysis (for context)
            
        Returns:
            CreativeScore with predictions
        """
        score = CreativeScore(
            pass_probability=0.5,
            confidence_interval=(0.3, 0.7),
            predicted_lift=0.0,
            lift_range=(-2.0, 5.0),
            features=features,
            scoring_timestamp=datetime.now().isoformat(),
            video_analysis=video_analysis,
        )
        
        # Use trained model if available
        if self.is_trained and self.model is not None:
            score = self._predict_with_model(features, score)
        else:
            # Use heuristic model
            score = self._predict_heuristic(features, score)
        
        # Predict diagnostics
        score.predicted_diagnostics = self._predict_diagnostics(features)
        
        # Identify risk factors
        score.risk_factors = self._identify_risk_factors(features, video_analysis)
        score.risk_level = self._calculate_risk_level(score.risk_factors)
        
        # Find similar historical creatives
        score.similar_creatives = self._find_similar_creatives(features)
        
        # Generate recommendations
        score.recommendations = self._generate_recommendations(
            features, score.predicted_diagnostics, score.risk_factors
        )
        
        return score
    
    def _predict_heuristic(
        self, 
        features: PredictionFeatures, 
        score: CreativeScore
    ) -> CreativeScore:
        """
        Heuristic-based prediction when no trained model available.
        Based on industry research on creative effectiveness.
        """
        base_prob = 0.45  # Industry average pass rate
        
        # Factors that increase pass probability
        adjustments = []
        
        # Human in opening is HUGE
        if features.has_human_in_opening:
            adjustments.append(('human_opening', 0.15))
        elif features.first_human_appearance_sec > 0 and features.first_human_appearance_sec <= 5:
            adjustments.append(('human_early', 0.08))
        elif features.human_frame_ratio > 0.3:
            adjustments.append(('human_present', 0.05))
        
        # Logo timing
        if features.logo_in_first_3_sec:
            adjustments.append(('logo_early', 0.08))
        
        # Emotional content
        if features.has_positive_emotion:
            adjustments.append(('positive_emotion', 0.07))
        if features.has_emotional_content:
            adjustments.append(('emotional', 0.05))
        
        # CTA
        if features.has_cta and features.cta_in_last_5_sec:
            adjustments.append(('cta_present', 0.05))
        
        # Product visibility
        if features.product_visible_ratio > 0.3:
            adjustments.append(('product_visible', 0.04))
        
        # Negative factors
        if not features.has_human_in_opening and features.first_human_appearance_sec < 0:
            adjustments.append(('no_human', -0.10))
        
        if features.logo_first_appearance_sec > 10:
            adjustments.append(('logo_late', -0.05))
        
        if features.scene_type_diversity > 5:
            adjustments.append(('too_complex', -0.05))
        
        # Calculate final probability
        total_adjustment = sum(adj[1] for adj in adjustments)
        pass_prob = base_prob + total_adjustment
        pass_prob = max(0.1, min(0.9, pass_prob))  # Clamp to [0.1, 0.9]
        
        # Confidence interval based on certainty
        uncertainty = 0.15 - (abs(total_adjustment) * 0.3)  # More adjustment = more confident
        uncertainty = max(0.08, min(0.2, uncertainty))
        
        score.pass_probability = pass_prob
        score.confidence_interval = (
            max(0, pass_prob - uncertainty),
            min(1, pass_prob + uncertainty)
        )
        
        # Predict lift based on pass probability
        if pass_prob >= 0.7:
            score.predicted_lift = 5.0 + (pass_prob - 0.7) * 15
            score.lift_range = (3.0, 10.0)
        elif pass_prob >= 0.5:
            score.predicted_lift = 2.0 + (pass_prob - 0.5) * 15
            score.lift_range = (0.5, 5.0)
        else:
            score.predicted_lift = -1.0 + (pass_prob - 0.3) * 15
            score.lift_range = (-3.0, 2.0)
        
        return score
    
    def _predict_with_model(
        self, 
        features: PredictionFeatures, 
        score: CreativeScore
    ) -> CreativeScore:
        """Predict using trained ensemble model based on historical data."""
        
        # Predict diagnostics using learned models (or fallback to heuristic)
        predicted_diags = self.predict_diagnostics_from_video(features)
        
        # Build feature vector matching training data format
        # Include BOTH diagnostic scores AND video features
        feature_dict = {
            # Diagnostic scores (predicted from video using learned models)
            'attention_score': predicted_diags.get('attention_score', 50),
            'message_clarity_score': predicted_diags.get('message_clarity_score', 50),
            'emotional_resonance_score': predicted_diags.get('emotional_resonance_score', 50),
            'brand_recall_score': predicted_diags.get('brand_recall_score', 50),
            'uniqueness_score': predicted_diags.get('uniqueness_score', 50),
            # Video features (from Ollama analysis)
            'has_human_in_opening': 1 if features.has_human_in_opening else 0,
            'first_human_appearance_sec': features.first_human_appearance_sec,
            'human_frame_ratio': features.human_frame_ratio,
            'human_looking_at_camera_ratio': features.human_looking_at_camera_ratio,
            'has_positive_emotion': 1 if features.has_positive_emotion else 0,
            'has_emotional_content': 1 if features.has_emotional_content else 0,
            'logo_in_first_3_sec': 1 if features.logo_in_first_3_sec else 0,
            'logo_frame_ratio': features.logo_frame_ratio,
            'product_visible_ratio': features.product_visible_ratio,
            'has_cta': 1 if features.has_cta else 0,
            'cta_in_last_5_sec': 1 if features.cta_in_last_5_sec else 0,
            'scene_type_diversity': features.scene_type_diversity,
            'visual_complexity_score': features.visual_complexity_score,
        }
        
        # Only include features that were used in training
        X_dict = {k: v for k, v in feature_dict.items() if k in self.feature_names}
        
        # If we're missing features, fill with defaults
        for feat in self.feature_names:
            if feat not in X_dict:
                X_dict[feat] = 0 if 'has_' in feat else 50
        
        # Create DataFrame in correct column order
        X = pd.DataFrame([X_dict])[self.feature_names]
        
        # Replace -1 sentinel with NaN
        X = X.replace(-1, np.nan)
        
        # Get probability from trained ensemble model
        try:
            proba = self.model.predict_proba(X)[0]
            pass_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
        except Exception as e:
            print(f"Prediction error: {e}")
            # Fallback to heuristic
            return self._predict_heuristic(features, score)
        
        score.pass_probability = pass_prob
        
        # Confidence interval based on training set size and model uncertainty
        n_samples = self.training_stats.get('n_samples', 50)
        loocv_accuracy = self.training_stats.get('loocv_accuracy', 70) / 100
        
        # Wider CI if model accuracy is lower
        base_std = np.sqrt(pass_prob * (1 - pass_prob) / n_samples)
        uncertainty_factor = 1 + (1 - loocv_accuracy)  # Higher uncertainty if lower accuracy
        std_error = base_std * uncertainty_factor * 1.5
        
        score.confidence_interval = (
            max(0, pass_prob - 1.96 * std_error),
            min(1, pass_prob + 1.96 * std_error)
        )
        
        # Predict lift based on historical lift distribution
        if self.historical_results:
            passed_lifts = [r['lift'] for r in self.historical_results if r['passed']]
            failed_lifts = [r['lift'] for r in self.historical_results if not r['passed']]
            
            avg_pass_lift = np.mean(passed_lifts) if passed_lifts else 4.0
            avg_fail_lift = np.mean(failed_lifts) if failed_lifts else -1.0
            
            # Weighted average based on pass probability
            score.predicted_lift = pass_prob * avg_pass_lift + (1 - pass_prob) * avg_fail_lift
            
            # Lift range from historical data
            all_lifts = [r['lift'] for r in self.historical_results]
            score.lift_range = (
                round(np.percentile(all_lifts, 10), 1),
                round(np.percentile(all_lifts, 90), 1)
            )
        else:
            # Fallback
            score.predicted_lift = 3.0 * (pass_prob - 0.5)
            score.lift_range = (-3.0, 6.0)
        
        return score
        
        # Predict lift (would need a separate regression model in production)
        score.predicted_lift = (pass_prob - 0.45) * 20  # Simple linear mapping
        score.lift_range = (score.predicted_lift - 2, score.predicted_lift + 2)
        
        return score
    
    def _predict_diagnostics(self, features: PredictionFeatures) -> Dict[str, DiagnosticPrediction]:
        """Predict diagnostic scores based on features using learned models."""
        diagnostics = {}
        
        # Use learned diagnostic models if available
        if self.diagnostic_models_trained:
            predicted_values = self.predict_diagnostics_from_video(features)
            source = "learned"
        else:
            predicted_values = self._predict_diagnostics_heuristic(features)
            source = "heuristic"
        
        for diag_name, benchmark in self.DIAGNOSTIC_BENCHMARKS.items():
            predicted = predicted_values.get(diag_name, benchmark)
            
            # Clamp to reasonable range
            predicted = max(20, min(95, predicted))
            
            # Determine status
            if predicted >= benchmark + 5:
                status = "good"
            elif predicted >= benchmark - 5:
                status = "warning"
            else:
                status = "poor"
            
            # Confidence interval based on model quality
            if self.diagnostic_models_trained and diag_name in self.diagnostic_training_stats:
                rmse = self.diagnostic_training_stats[diag_name].get('rmse', 8)
                ci_width = rmse * 1.5  # ~90% CI
            else:
                ci_width = 10  # Default uncertainty
            
            diagnostics[diag_name] = DiagnosticPrediction(
                name=diag_name,
                predicted_value=round(predicted, 1),
                confidence_interval=(predicted - ci_width, predicted + ci_width),
                benchmark=benchmark,
                status=status,
            )
        
        return diagnostics
    
    def _identify_risk_factors(
        self, 
        features: PredictionFeatures,
        video_analysis: VideoAnalysisResult = None
    ) -> List[RiskFactor]:
        """Identify risk factors in the creative."""
        risks = []
        
        # No human in opening
        if not features.has_human_in_opening:
            if features.first_human_appearance_sec < 0:
                risks.append(RiskFactor(
                    factor="No human presence in video",
                    impact="-10 to -15 attention points",
                    confidence="high",
                    evidence="Creatives with human faces get 23% higher attention on average"
                ))
            else:
                risks.append(RiskFactor(
                    factor=f"Human first appears at {features.first_human_appearance_sec:.1f}s (not in opening)",
                    impact="-8 to -12 attention points",
                    confidence="high",
                    evidence="Human face in first 3 seconds increases attention by 12 points on average"
                ))
        
        # Logo timing
        if features.logo_first_appearance_sec < 0:
            risks.append(RiskFactor(
                factor="No logo detected in video",
                impact="-10 brand recall points",
                confidence="medium",
                evidence="Logo presence correlates with +15% brand recall"
            ))
        elif features.logo_first_appearance_sec > 10:
            risks.append(RiskFactor(
                factor=f"Logo appears late ({features.logo_first_appearance_sec:.1f}s)",
                impact="-5 brand recall points",
                confidence="medium",
                evidence="Logo in first 3 seconds increases brand recall by 18%"
            ))
        
        # No CTA
        if not features.has_cta:
            risks.append(RiskFactor(
                factor="No call-to-action detected",
                impact="-5 message clarity points",
                confidence="medium",
                evidence="Clear CTA improves message clarity by 12%"
            ))
        elif not features.cta_in_last_5_sec:
            risks.append(RiskFactor(
                factor="CTA not in final frames",
                impact="-3 message clarity points",
                confidence="low",
                evidence="End-frame CTA is 20% more effective than mid-roll"
            ))
        
        # Emotional content
        if not features.has_positive_emotion and not features.has_emotional_content:
            if features.human_frame_ratio > 0:
                risks.append(RiskFactor(
                    factor="Humans present but no clear emotional expression",
                    impact="-5 emotional resonance points",
                    confidence="medium",
                    evidence="Positive emotions increase emotional resonance by 25%"
                ))
        
        # Complexity
        if features.scene_type_diversity > 5:
            risks.append(RiskFactor(
                factor=f"High scene diversity ({features.scene_type_diversity} different types)",
                impact="-3 message clarity points",
                confidence="low",
                evidence="Simpler narratives have 15% higher clarity scores"
            ))
        
        return risks
    
    def _calculate_risk_level(self, risks: List[RiskFactor]) -> str:
        """Calculate overall risk level."""
        high_confidence_risks = [r for r in risks if r.confidence == "high"]
        medium_confidence_risks = [r for r in risks if r.confidence == "medium"]
        
        if len(high_confidence_risks) >= 2:
            return "high"
        elif len(high_confidence_risks) >= 1 or len(medium_confidence_risks) >= 3:
            return "medium"
        else:
            return "low"
    
    def _find_similar_creatives(self, features: PredictionFeatures) -> List[SimilarCreative]:
        """Find similar historical creatives based on predicted diagnostics."""
        if not self.historical_features or not self.historical_results:
            return []
        
        # Get predicted diagnostic values for this creative
        # We need to calculate them first
        predicted_diagnostics = {}
        for diag_name, benchmark in self.DIAGNOSTIC_BENCHMARKS.items():
            predicted = benchmark
            weights = self.DIAGNOSTIC_FEATURE_WEIGHTS.get(diag_name, {})
            for feature_name, weight in weights.items():
                if hasattr(features, feature_name):
                    value = getattr(features, feature_name)
                    if isinstance(value, bool):
                        predicted += weight if value else 0
                    elif isinstance(value, (int, float)):
                        predicted += weight * value
            predicted_diagnostics[diag_name] = max(0, min(100, predicted))
        
        # Calculate similarity to each historical creative
        similarities = []
        
        for i, hist_features in enumerate(self.historical_features):
            # Calculate distance based on diagnostic scores
            distance = 0
            count = 0
            for diag_name in self.DIAGNOSTIC_BENCHMARKS.keys():
                if diag_name in hist_features and hist_features[diag_name]:
                    pred_val = predicted_diagnostics.get(diag_name, 50)
                    hist_val = hist_features[diag_name]
                    distance += (pred_val - hist_val) ** 2
                    count += 1
            
            if count > 0:
                # Convert distance to similarity (0-100)
                rmse = (distance / count) ** 0.5
                similarity = max(0, 100 - rmse)
                similarities.append((i, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 5 similar creatives
        result = []
        for idx, sim in similarities[:5]:
            if idx < len(self.historical_results):
                hist_result = self.historical_results[idx]
                
                # Find key similarities
                key_sims = []
                hist_features = self.historical_features[idx]
                for diag_name in ['attention_score', 'message_clarity_score', 'emotional_resonance_score']:
                    if diag_name in hist_features and hist_features[diag_name]:
                        pred_val = predicted_diagnostics.get(diag_name, 50)
                        hist_val = hist_features[diag_name]
                        if abs(pred_val - hist_val) < 10:
                            key_sims.append(diag_name.replace('_score', '').replace('_', ' '))
                
                result.append(SimilarCreative(
                    name=hist_result.get('name', f'Historical #{idx}'),
                    similarity_score=round(sim, 1),
                    lift=hist_result.get('lift', 0),
                    passed=hist_result.get('passed', False),
                    key_similarities=key_sims[:3] if key_sims else ['overall profile']
                ))
        
        return result
    
    def _generate_recommendations(
        self,
        features: PredictionFeatures,
        diagnostics: Dict[str, DiagnosticPrediction],
        risks: List[RiskFactor],
    ) -> List[Recommendation]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Based on risks
        for risk in risks:
            if risk.confidence == "high":
                if "human" in risk.factor.lower() and "opening" not in risk.factor.lower():
                    recommendations.append(Recommendation(
                        priority="high",
                        action="Add human face in opening 2-3 seconds",
                        expected_impact="+10-15 attention score",
                        rationale=risk.evidence
                    ))
                elif "opening" in risk.factor.lower():
                    recommendations.append(Recommendation(
                        priority="high",
                        action="Re-edit to show human face within first 2 seconds",
                        expected_impact="+8-12 attention score",
                        rationale=risk.evidence
                    ))
                elif "logo" in risk.factor.lower():
                    recommendations.append(Recommendation(
                        priority="high",
                        action="Add logo/brand element in first 3 seconds",
                        expected_impact="+5-8 brand recall score",
                        rationale=risk.evidence
                    ))
        
        # Based on weak diagnostics
        for name, diag in diagnostics.items():
            if diag.status == "poor":
                if "attention" in name:
                    recommendations.append(Recommendation(
                        priority="high",
                        action="Strengthen opening hook - add motion, human face, or pattern interrupt",
                        expected_impact=f"+{diag.benchmark - diag.predicted_value:.0f} attention points",
                        rationale="Attention score predicted below benchmark"
                    ))
                elif "clarity" in name:
                    recommendations.append(Recommendation(
                        priority="medium",
                        action="Simplify message - focus on single key takeaway with clear CTA",
                        expected_impact=f"+{diag.benchmark - diag.predicted_value:.0f} clarity points",
                        rationale="Message clarity predicted below benchmark"
                    ))
                elif "emotional" in name:
                    recommendations.append(Recommendation(
                        priority="medium",
                        action="Add emotional element - human story, music, or relatable moment",
                        expected_impact=f"+{diag.benchmark - diag.predicted_value:.0f} emotional resonance",
                        rationale="Emotional resonance predicted below benchmark"
                    ))
        
        # Deduplicate by action
        seen_actions = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec.action not in seen_actions:
                seen_actions.add(rec.action)
                unique_recommendations.append(rec)
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        unique_recommendations.sort(key=lambda r: priority_order.get(r.priority, 2))
        
        return unique_recommendations[:5]  # Top 5 recommendations
    
    def train(self, training_data: List[dict]):
        """
        Train the model on historical data.
        
        Args:
            training_data: List of dicts with 'features' and 'passed' keys
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for training")
        
        if len(training_data) < 10:
            raise ValueError("Need at least 10 samples for training")
        
        # Prepare data
        X = []
        y = []
        
        for item in training_data:
            if isinstance(item['features'], PredictionFeatures):
                X.append(list(item['features'].to_dict().values()))
            else:
                X.append(list(item['features'].values()))
            y.append(1 if item['passed'] else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Store for similarity search
        self.historical_features = [item['features'] if isinstance(item['features'], dict) else item['features'].to_dict() for item in training_data]
        self.historical_results = [{'name': item.get('name', ''), 'lift': item.get('lift', 0), 'passed': item['passed']} for item in training_data]
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        if XGBOOST_AVAILABLE:
            base_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
        else:
            base_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                random_state=42
            )
        
        # Calibrate for better probability estimates
        self.model = CalibratedClassifierCV(base_model, cv=3)
        self.model.fit(X_scaled, y)
        
        self.is_trained = True
        
        # Evaluate
        scores = cross_val_score(base_model, X_scaled, y, cv=3, scoring='roc_auc')
        return {'auc_mean': scores.mean(), 'auc_std': scores.std()}
    
    def save_model(self, path: str):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'historical_features': self.historical_features,
            'historical_results': self.historical_results,
            'version': '1.0',
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def _load_model(self, path: str):
        """Load trained model from disk."""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.historical_features = model_data.get('historical_features', [])
            self.historical_results = model_data.get('historical_results', [])
            self.is_trained = True
        except Exception as e:
            print(f"Failed to load model: {e}")


def get_prediction_model(model_path: str = None) -> CreativePredictionModel:
    """Get prediction model instance."""
    return CreativePredictionModel(model_path)
