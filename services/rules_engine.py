"""
Rules Engine Service.

Handles:
1. Parsing CT Rules from PDF documents
2. Extracting structured rules for validation
3. Embedding rules text for RAG
4. Validating test plans against rules
"""

import os
import json
import re
from typing import Optional
from pathlib import Path

import pdfplumber

from models.rules import CTRules, BudgetTier, AssetCost, Turnaround, MinimumRequirements, DEFAULT_CT_RULES
from utils.llm import get_structured_output, get_completion


class RulesEngine:
    """
    Service for managing and enforcing CT Rules.
    
    The rules are stored in a PDF document that is:
    1. Parsed to extract structured rules (tiers, costs, limits)
    2. Embedded in vector store for Q&A
    """
    
    def __init__(self, rules_pdf_path: Optional[str] = None):
        """
        Initialize the rules engine.
        
        Args:
            rules_pdf_path: Path to the CT Rules PDF. If None, uses default rules.
        """
        self.rules_pdf_path = rules_pdf_path
        self._rules: Optional[CTRules] = None
        self._raw_text: Optional[str] = None
    
    @property
    def rules(self) -> CTRules:
        """Get the current rules, loading from PDF if not already loaded."""
        if self._rules is None:
            if self.rules_pdf_path and os.path.exists(self.rules_pdf_path):
                self._rules = self._load_rules_from_pdf()
            else:
                self._rules = DEFAULT_CT_RULES
        return self._rules
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all text from a PDF file."""
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
                
                # Also extract tables
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        # Convert table to text format
                        for row in table:
                            text_parts.append(" | ".join(str(cell) if cell else "" for cell in row))
        
        return "\n\n".join(text_parts)
    
    def _load_rules_from_pdf(self) -> CTRules:
        """
        Load and parse rules from the PDF document.
        
        Uses Claude to extract structured rules from the PDF text.
        """
        if not self.rules_pdf_path:
            return DEFAULT_CT_RULES
        
        # Extract text from PDF
        self._raw_text = self._extract_text_from_pdf(self.rules_pdf_path)
        
        # Use Claude to extract structured rules
        extraction_prompt = f"""Extract the creative testing rules from this document and return them as structured JSON.

Document text:
{self._raw_text}

Extract:
1. Budget tiers (min_budget, max_budget, video_limit, display_limit for each tier)
2. Costs (video_cost, display_cost, audio_cost per asset)
3. Turnaround times (standard and expedited days for each asset type)
4. Any minimum requirements

Return a JSON object with this structure:
{{
    "version": "extracted version or 1.0",
    "effective_date": "extracted date or today",
    "budget_allocation_pct": 0.015,
    "budget_tiers": [
        {{"min_budget": 0, "max_budget": 5000000, "video_limit": 2, "display_limit": 5, "audio_limit": 0}}
    ],
    "costs": {{
        "video_cost": 5000,
        "display_cost": 3000,
        "audio_cost": 2500,
        "expedited_fee_pct": 0.5
    }},
    "turnaround": {{
        "video_standard_days": 14,
        "video_expedited_days": 7,
        "display_standard_days": 10,
        "display_expedited_days": 5,
        "audio_standard_days": 10,
        "audio_expedited_days": 5
    }},
    "minimum_requirements": {{
        "min_budget_for_required_video_test": 2000000,
        "tv_spend_requires_tv_test": true
    }}
}}

Only output the JSON, no other text."""

        response = get_completion(
            prompt=extraction_prompt,
            system="You extract structured data from documents. Output only valid JSON.",
            temperature=0.1,
        )
        
        # Clean and parse response
        clean_response = response.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]
        if clean_response.startswith("```"):
            clean_response = clean_response[3:]
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]
        
        try:
            data = json.loads(clean_response.strip())
            
            # Build CTRules from extracted data
            rules = CTRules(
                version=data.get("version", "1.0"),
                effective_date=data.get("effective_date", "2025-01-01"),
                budget_allocation_pct=data.get("budget_allocation_pct", 0.015),
                budget_tiers=[
                    BudgetTier(**tier) for tier in data.get("budget_tiers", [])
                ],
                costs=AssetCost(**data.get("costs", {})),
                turnaround=Turnaround(**data.get("turnaround", {})),
                minimum_requirements=MinimumRequirements(**data.get("minimum_requirements", {})),
            )
            rules.raw_text = self._raw_text
            return rules
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error parsing rules from PDF: {e}")
            # Fall back to defaults but keep raw text
            rules = DEFAULT_CT_RULES.model_copy()
            rules.raw_text = self._raw_text
            return rules
    
    def reload_rules(self, pdf_path: Optional[str] = None) -> CTRules:
        """Reload rules from PDF (called when superuser updates the document)."""
        if pdf_path:
            self.rules_pdf_path = pdf_path
        self._rules = None  # Clear cache
        return self.rules
    
    def validate_plan(self, budget: float, video_count: int, display_count: int) -> dict:
        """
        Validate a test plan against the current rules.
        
        Args:
            budget: Campaign budget
            video_count: Number of video assets to test
            display_count: Number of display assets to test
            
        Returns:
            dict with validation results
        """
        return self.rules.validate_plan(budget, video_count, display_count)
    
    def get_limits_for_budget(self, budget: float) -> dict:
        """Get the testing limits for a given budget."""
        return self.rules.get_limits(budget)
    
    def calculate_cost(self, video_count: int, display_count: int, audio_count: int = 0, expedited: bool = False) -> float:
        """Calculate the total cost for testing."""
        return self.rules.calculate_cost(video_count, display_count, audio_count, expedited)
    
    def get_raw_text(self) -> Optional[str]:
        """Get the raw text from the rules PDF for RAG."""
        if self._raw_text is None and self.rules_pdf_path:
            self._raw_text = self._extract_text_from_pdf(self.rules_pdf_path)
        return self._raw_text
    
    def answer_question(self, question: str) -> str:
        """
        Answer a question about the CT rules.
        
        Uses RAG over the rules document to provide accurate answers.
        """
        raw_text = self.get_raw_text()
        if not raw_text:
            # Use structured rules if no PDF
            raw_text = f"""Creative Testing Rules Summary:

Budget Tiers:
{chr(10).join(f"- ${tier.min_budget:,.0f} to ${tier.max_budget:,.0f}: {tier.video_limit} videos, {tier.display_limit} display" for tier in self.rules.budget_tiers)}

Costs:
- Video: ${self.rules.costs.video_cost:,.0f} per asset
- Display: ${self.rules.costs.display_cost:,.0f} per asset
- Audio: ${self.rules.costs.audio_cost:,.0f} per asset
- Expedited fee: +{self.rules.costs.expedited_fee_pct * 100:.0f}%

Turnaround:
- Video: {self.rules.turnaround.video_standard_days} days standard, {self.rules.turnaround.video_expedited_days} days expedited
- Display: {self.rules.turnaround.display_standard_days} days standard, {self.rules.turnaround.display_expedited_days} days expedited

Requirements:
- {self.rules.budget_allocation_pct * 100:.1f}% of campaign budget allocated to testing
- Campaigns over ${self.rules.minimum_requirements.min_budget_for_required_video_test:,.0f} must test at least 1 video
"""
        
        prompt = f"""Based on these Creative Testing Rules:

{raw_text}

Answer this question: {question}

Provide a clear, specific answer based only on the rules document. If the answer isn't in the document, say so."""

        return get_completion(
            prompt=prompt,
            system="You answer questions about creative testing rules based on the provided documentation. Be specific and cite the rules when relevant.",
        )


# Global instance for convenience
_rules_engine: Optional[RulesEngine] = None


def get_rules_engine(pdf_path: Optional[str] = None) -> RulesEngine:
    """Get or create the global rules engine instance."""
    global _rules_engine
    if _rules_engine is None:
        _rules_engine = RulesEngine(pdf_path)
    return _rules_engine
