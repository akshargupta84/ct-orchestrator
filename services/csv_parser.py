"""
CSV Parser Service.

Handles parsing and validation of test results CSV files from the 3rd party vendor.
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
    "control_sample_size",
    "exposed_sample_size",
]

# KPI columns (control, exposed, lift, stat_sig for each)
KPI_COLUMNS = ["awareness", "consideration", "preference", "purchase_intent"]

# Diagnostic metric columns
DIAGNOSTIC_COLUMNS = [
    "brand_strength",
    "relevance", 
    "emotional_engagement",
    "uniqueness",
    "credibility",
    "call_to_action_clarity",
    "brand_fit",
    "message_clarity",
    "likability",
    "memorability",
]


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
        
        # Validate required columns
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            errors.append(CSVValidationError(
                error=f"Missing required columns: {', '.join(missing_cols)}"
            ))
        
        # Check for KPI columns
        for kpi in KPI_COLUMNS:
            expected = [f"control_{kpi}", f"exposed_{kpi}", f"{kpi}_lift", f"{kpi}_stat_sig"]
            present = [col for col in expected if col in df.columns]
            if len(present) == 0:
                warnings.append(f"No columns found for KPI: {kpi}")
            elif len(present) < 4:
                missing = [col for col in expected if col not in df.columns]
                warnings.append(f"Incomplete columns for {kpi}: missing {', '.join(missing)}")
        
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
            asset_type = AssetType.DISPLAY  # Default
        
        # Helper to safely get float
        def get_float(col: str, default: float = 0.0) -> float:
            val = row.get(col)
            if pd.isna(val):
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        
        # Helper to safely get bool
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
        
        # Extract KPI data
        control_awareness = get_float("control_awareness")
        exposed_awareness = get_float("exposed_awareness")
        awareness_lift = get_float("awareness_lift", exposed_awareness - control_awareness)
        awareness_stat_sig = get_bool("awareness_stat_sig")
        
        control_consideration = get_float("control_consideration")
        exposed_consideration = get_float("exposed_consideration")
        consideration_lift = get_float("consideration_lift", exposed_consideration - control_consideration)
        consideration_stat_sig = get_bool("consideration_stat_sig")
        
        control_preference = get_float("control_preference") if "control_preference" in row else None
        exposed_preference = get_float("exposed_preference") if "exposed_preference" in row else None
        preference_lift = get_float("preference_lift") if "preference_lift" in row else None
        preference_stat_sig = get_bool("preference_stat_sig") if "preference_stat_sig" in row else None
        
        control_purchase_intent = get_float("control_purchase_intent") if "control_purchase_intent" in row else None
        exposed_purchase_intent = get_float("exposed_purchase_intent") if "exposed_purchase_intent" in row else None
        purchase_intent_lift = get_float("purchase_intent_lift") if "purchase_intent_lift" in row else None
        purchase_intent_stat_sig = get_bool("purchase_intent_stat_sig") if "purchase_intent_stat_sig" in row else None
        
        # Determine primary KPI values
        primary_kpi = self.primary_kpi
        kpi_mapping = {
            KPIType.AWARENESS: (awareness_lift, awareness_stat_sig),
            KPIType.CONSIDERATION: (consideration_lift, consideration_stat_sig),
            KPIType.PREFERENCE: (preference_lift or 0, preference_stat_sig or False),
            KPIType.PURCHASE_INTENT: (purchase_intent_lift or 0, purchase_intent_stat_sig or False),
        }
        primary_kpi_lift, primary_kpi_stat_sig = kpi_mapping.get(
            primary_kpi, (awareness_lift, awareness_stat_sig)
        )
        
        # Parse diagnostic metrics
        diagnostics = []
        for diag_col in DIAGNOSTIC_COLUMNS:
            if diag_col in row and not pd.isna(row[diag_col]):
                diagnostics.append(DiagnosticMetric(
                    name=diag_col,
                    value=get_float(diag_col),
                    benchmark=get_float(f"{diag_col}_benchmark") if f"{diag_col}_benchmark" in row else None,
                    percentile=int(get_float(f"{diag_col}_percentile")) if f"{diag_col}_percentile" in row else None,
                ))
        
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
            primary_kpi_lift=primary_kpi_lift,
            primary_kpi_stat_sig=primary_kpi_stat_sig,
            passed=primary_kpi_stat_sig and primary_kpi_lift > 0,
            
            diagnostics=diagnostics,
            
            control_sample_size=int(get_float("control_sample_size", 0)),
            exposed_sample_size=int(get_float("exposed_sample_size", 0)),
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
