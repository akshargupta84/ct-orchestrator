"""
CSV Parser Service.

Handles parsing and validation of test results CSV files from the 3rd party vendor.
Supports multiple column naming conventions.
"""

import io
from typing import Optional
import pandas as pd
from pydantic import BaseModel, Field

from models import (
    TestResults,
    CreativeTestResult,
    DiagnosticMetric,
    AssetType,
    KPIType,
    LiftResult,
)


# Expected columns in the results CSV
REQUIRED_COLUMNS = [
    "creative_id",
    "creative_name", 
    "asset_type",
]

# Sample size columns - multiple possible names
SAMPLE_SIZE_COLUMNS = {
    "control": ["control_sample_size", "control_n", "control_size"],
    "exposed": ["exposed_sample_size", "test_sample_size", "exposed_n", "test_n"],
}

# KPI columns (control, exposed, lift, stat_sig for each)
KPI_COLUMNS = ["awareness", "consideration", "preference", "purchase_intent", "ad_recall"]

# Column name variations for KPIs
KPI_COLUMN_VARIANTS = {
    "control": [
        "control_{kpi}",
        "control_{kpi}_pct",
        "{kpi}_control",
        "{kpi}_control_pct",
    ],
    "exposed": [
        "exposed_{kpi}",
        "test_{kpi}",
        "exposed_{kpi}_pct",
        "test_{kpi}_pct",
        "{kpi}_exposed",
        "{kpi}_test",
    ],
    "lift": [
        "{kpi}_lift",
        "{kpi}_lift_pct",
        "lift_{kpi}",
        "{kpi}_absolute_lift",
    ],
    "stat_sig": [
        "{kpi}_stat_sig",
        "{kpi}_significant",
        "{kpi}_sig",
        "stat_sig_{kpi}",
        "significant_{kpi}",
    ],
}

# Diagnostic metric columns - multiple possible names
DIAGNOSTIC_COLUMNS = {
    "brand_strength": ["brand_strength", "brand_recall_score", "brand_recall"],
    "relevance": ["relevance", "relevance_score"],
    "emotional_engagement": ["emotional_engagement", "emotional_resonance_score", "emotional_resonance"],
    "uniqueness": ["uniqueness", "uniqueness_score"],
    "credibility": ["credibility", "credibility_score"],
    "call_to_action_clarity": ["call_to_action_clarity", "cta_clarity"],
    "brand_fit": ["brand_fit", "brand_fit_score"],
    "message_clarity": ["message_clarity", "message_clarity_score"],
    "likability": ["likability", "likability_score"],
    "memorability": ["memorability", "attention_score", "attention"],
}


class CSVValidationError(BaseModel):
    """An error found during CSV validation."""
    row: Optional[int] = None
    column: Optional[str] = None
    error: str


class CSVParseResult(BaseModel):
    """Result of parsing a CSV file."""
    success: bool
    results: Optional[TestResults] = None
    errors: list[CSVValidationError] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    row_count: int = 0


class CSVParser:
    """
    Parser for creative testing results CSV files.
    
    Expected CSV format:
    - creative_id, creative_name, asset_type
    - control_awareness, exposed_awareness, awareness_lift, awareness_stat_sig
    - (same pattern for consideration, preference, purchase_intent)
    - diagnostic metrics (brand_strength, relevance, etc.)
    - control_sample_size, exposed_sample_size
    """
    
    def __init__(self, primary_kpi: KPIType = KPIType.AWARENESS):
        """
        Initialize the parser.
        
        Args:
            primary_kpi: The primary KPI for this campaign (determines pass/fail)
        """
        self.primary_kpi = primary_kpi
    
    def parse(
        self, 
        csv_content: str | bytes | io.IOBase,
        campaign_id: str,
        test_plan_id: str,
    ) -> CSVParseResult:
        """
        Parse a CSV file containing test results.
        
        Args:
            csv_content: CSV content as string, bytes, or file-like object
            campaign_id: ID of the campaign these results belong to
            test_plan_id: ID of the test plan these results are for
            
        Returns:
            CSVParseResult with parsed data or errors
        """
        errors = []
        warnings = []
        
        # Load CSV
        try:
            if isinstance(csv_content, str):
                df = pd.read_csv(io.StringIO(csv_content))
            elif isinstance(csv_content, bytes):
                df = pd.read_csv(io.BytesIO(csv_content))
            else:
                df = pd.read_csv(csv_content)
        except Exception as e:
            return CSVParseResult(
                success=False,
                errors=[CSVValidationError(error=f"Failed to parse CSV: {str(e)}")],
            )
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
        
        # Store column mapping for later use
        self._column_map = {}
        
        # Validate required columns
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            errors.append(CSVValidationError(
                error=f"Missing required columns: {', '.join(missing_cols)}"
            ))
        
        # Find sample size columns
        for sample_type, variants in SAMPLE_SIZE_COLUMNS.items():
            found = None
            for variant in variants:
                if variant in df.columns:
                    found = variant
                    break
            if found:
                self._column_map[f"{sample_type}_sample_size"] = found
            else:
                # Not critical - will use default
                warnings.append(f"No {sample_type} sample size column found, using default")
        
        # Check for KPI columns and build mapping
        kpi_found = False
        for kpi in KPI_COLUMNS:
            kpi_cols_found = {}
            
            for col_type, variants in KPI_COLUMN_VARIANTS.items():
                for variant_template in variants:
                    variant = variant_template.format(kpi=kpi)
                    if variant in df.columns:
                        kpi_cols_found[col_type] = variant
                        break
            
            if kpi_cols_found:
                kpi_found = True
                self._column_map[kpi] = kpi_cols_found
                
                # Check completeness
                missing_types = [t for t in ["lift", "stat_sig"] if t not in kpi_cols_found]
                if missing_types:
                    warnings.append(f"Incomplete columns for {kpi}: missing {', '.join(missing_types)}")
            else:
                # Only warn, not error
                pass  # Not all KPIs need to be present
        
        if not kpi_found:
            errors.append(CSVValidationError(
                error="No KPI columns found. Need at least lift and stat_sig for one KPI."
            ))
        
        if errors:
            return CSVParseResult(
                success=False,
                errors=errors,
                warnings=warnings,
                row_count=len(df),
            )
        
        # Parse each row
        creative_results = []
        for idx, row in df.iterrows():
            try:
                result = self._parse_row(row, idx)
                creative_results.append(result)
            except Exception as e:
                errors.append(CSVValidationError(
                    row=idx + 2,  # +2 for 1-indexed + header row
                    error=str(e)
                ))
        
        if errors:
            return CSVParseResult(
                success=False,
                errors=errors,
                warnings=warnings,
                row_count=len(df),
            )
        
        # Build TestResults
        test_results = TestResults(
            id=f"results_{campaign_id}_{test_plan_id}",
            campaign_id=campaign_id,
            test_plan_id=test_plan_id,
            results=creative_results,
            total_creatives_tested=len(creative_results),
            creatives_passed=sum(1 for r in creative_results if r.passed),
            creatives_failed=sum(1 for r in creative_results if not r.passed),
        )
        
        return CSVParseResult(
            success=True,
            results=test_results,
            warnings=warnings,
            row_count=len(df),
        )
    
    def _parse_row(self, row: pd.Series, idx: int) -> CreativeTestResult:
        """Parse a single row into a CreativeTestResult."""
        
        # Determine asset type
        asset_type_str = str(row.get("asset_type", "")).lower()
        if "video" in asset_type_str:
            asset_type = AssetType.VIDEO
        elif "display" in asset_type_str or "banner" in asset_type_str:
            asset_type = AssetType.DISPLAY
        else:
            asset_type = AssetType.VIDEO  # Default to video
        
        # Helper to safely get float from row using possible column names
        def get_float(col: str, default: float = 0.0) -> float:
            val = row.get(col)
            if pd.isna(val):
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        
        # Helper to safely get bool from row
        def get_bool(col: str, default: bool = False) -> bool:
            val = row.get(col)
            if pd.isna(val):
                return default
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                return val > 0
            val_str = str(val).lower()
            return val_str in ("true", "yes", "1", "significant", "sig")
        
        # Helper to get KPI value using the column mapping
        def get_kpi_value(kpi: str, value_type: str, default=None):
            """Get a KPI value using the column mapping."""
            if kpi in self._column_map and value_type in self._column_map[kpi]:
                col = self._column_map[kpi][value_type]
                if value_type == "stat_sig":
                    return get_bool(col, default if default is not None else False)
                else:
                    return get_float(col, default if default is not None else 0.0)
            return default
        
        # Extract KPI data using mapping
        control_awareness = get_kpi_value("awareness", "control", 0.0)
        exposed_awareness = get_kpi_value("awareness", "exposed", 0.0)
        awareness_lift = get_kpi_value("awareness", "lift", exposed_awareness - control_awareness if exposed_awareness else 0.0)
        awareness_stat_sig = get_kpi_value("awareness", "stat_sig", False)
        
        control_consideration = get_kpi_value("consideration", "control", 0.0)
        exposed_consideration = get_kpi_value("consideration", "exposed", 0.0)
        consideration_lift = get_kpi_value("consideration", "lift", exposed_consideration - control_consideration if exposed_consideration else 0.0)
        consideration_stat_sig = get_kpi_value("consideration", "stat_sig", False)
        
        control_preference = get_kpi_value("preference", "control")
        exposed_preference = get_kpi_value("preference", "exposed")
        preference_lift = get_kpi_value("preference", "lift")
        preference_stat_sig = get_kpi_value("preference", "stat_sig")
        
        control_purchase_intent = get_kpi_value("purchase_intent", "control")
        exposed_purchase_intent = get_kpi_value("purchase_intent", "exposed")
        purchase_intent_lift = get_kpi_value("purchase_intent", "lift")
        purchase_intent_stat_sig = get_kpi_value("purchase_intent", "stat_sig")
        
        # Also check for ad_recall
        ad_recall_lift = get_kpi_value("ad_recall", "lift")
        ad_recall_stat_sig = get_kpi_value("ad_recall", "stat_sig")
        
        # Check for "passed" column directly
        passed_direct = None
        if "passed" in row.index:
            passed_direct = get_bool("passed")
        
        # Determine primary KPI values
        primary_kpi = self.primary_kpi
        kpi_mapping = {
            KPIType.AWARENESS: (awareness_lift, awareness_stat_sig),
            KPIType.CONSIDERATION: (consideration_lift, consideration_stat_sig),
            KPIType.PREFERENCE: (preference_lift or 0, preference_stat_sig or False),
            KPIType.PURCHASE_INTENT: (purchase_intent_lift or 0, purchase_intent_stat_sig or False),
            KPIType.AD_RECALL: (ad_recall_lift or 0, ad_recall_stat_sig or False),
        }
        primary_kpi_lift, primary_kpi_stat_sig = kpi_mapping.get(
            primary_kpi, (awareness_lift, awareness_stat_sig)
        )
        
        # Parse diagnostic metrics using flexible column names
        diagnostics = []
        for diag_name, variants in DIAGNOSTIC_COLUMNS.items():
            for variant in variants:
                if variant in row.index and not pd.isna(row[variant]):
                    diagnostics.append(DiagnosticMetric(
                        name=diag_name,
                        value=get_float(variant),
                        benchmark=None,
                        percentile=None,
                    ))
                    break  # Found this diagnostic, move to next
        
        # Determine pass/fail
        if passed_direct is not None:
            passed = passed_direct
        else:
            passed = primary_kpi_stat_sig and primary_kpi_lift > 0
        
        return CreativeTestResult(
            creative_id=str(row["creative_id"]),
            creative_name=str(row["creative_name"]),
            asset_type=asset_type,
            
            control_awareness=control_awareness,
            control_consideration=control_consideration,
            control_preference=control_preference,
            control_purchase_intent=control_purchase_intent,
            
            exposed_awareness=exposed_awareness,
            exposed_consideration=exposed_consideration,
            exposed_preference=exposed_preference,
            exposed_purchase_intent=exposed_purchase_intent,
            
            awareness_lift=awareness_lift,
            consideration_lift=consideration_lift,
            preference_lift=preference_lift,
            purchase_intent_lift=purchase_intent_lift,
            
            awareness_stat_sig=awareness_stat_sig,
            consideration_stat_sig=consideration_stat_sig,
            preference_stat_sig=preference_stat_sig,
            purchase_intent_stat_sig=purchase_intent_stat_sig,
            
            primary_kpi=primary_kpi,
            primary_kpi_lift=primary_kpi_lift or 0,
            primary_kpi_stat_sig=primary_kpi_stat_sig or False,
            
            passed=passed,
            
            diagnostics=diagnostics,
            
            control_sample_size=int(get_float(self._column_map.get("control_sample_size", "control_sample_size"), 500)),
            exposed_sample_size=int(get_float(self._column_map.get("exposed_sample_size", "test_sample_size"), 500)),
        )


def generate_sample_csv() -> str:
    """Generate a sample CSV for testing/documentation."""
    
    header = [
        "creative_id", "creative_name", "asset_type",
        "control_awareness", "exposed_awareness", "awareness_lift", "awareness_stat_sig",
        "control_consideration", "exposed_consideration", "consideration_lift", "consideration_stat_sig",
        "control_preference", "exposed_preference", "preference_lift", "preference_stat_sig",
        "control_purchase_intent", "exposed_purchase_intent", "purchase_intent_lift", "purchase_intent_stat_sig",
        "brand_strength", "relevance", "emotional_engagement", "uniqueness", 
        "credibility", "call_to_action_clarity", "brand_fit", "message_clarity",
        "control_sample_size", "exposed_sample_size"
    ]
    
    rows = [
        ["VID001", "Summer Campaign 30s", "video",
         45.2, 52.1, 6.9, "true",
         32.5, 38.2, 5.7, "true", 
         28.1, 31.5, 3.4, "false",
         15.2, 18.9, 3.7, "true",
         72, 68, 75, 65, 70, 82, 78, 74,
         1500, 1500],
        ["VID002", "Product Demo 15s", "video",
         45.2, 48.5, 3.3, "false",
         32.5, 35.1, 2.6, "false",
         28.1, 29.8, 1.7, "false",
         15.2, 16.1, 0.9, "false",
         58, 52, 48, 61, 55, 45, 60, 52,
         1500, 1500],
        ["DSP001", "Banner 300x250", "display",
         42.1, 47.8, 5.7, "true",
         30.2, 34.9, 4.7, "true",
         25.5, 28.2, 2.7, "false",
         12.8, 15.1, 2.3, "false",
         70, 72, 65, 58, 68, 75, 71, 69,
         2000, 2000],
    ]
    
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(x) for x in row))
    
    return "\n".join(lines)
