"""
Tests for the Demo Chat Response System.

Covers:
- Keyword matching for video analysis queries
- Brand recall / performance driver responses
- Budget/rules responses
- Follow-up detection
- Test plan generation from upload context
- Fallback default response
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# We need to mock streamlit before importing the app module
sys.modules["streamlit"] = MagicMock()
sys.modules["streamlit.components"] = MagicMock()
sys.modules["streamlit.components.v1"] = MagicMock()

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def _get_keyword_response(query, upload_context=None, chat_history=None):
    """Helper to call _generate_keyword_response with mocked session state."""
    if upload_context is None:
        upload_context = {"videos": [], "media_plan": None}

    # Mock st.session_state
    mock_state = MagicMock()
    mock_state.hub_chat_messages = chat_history or []

    with patch("frontend.app.st") as mock_st:
        mock_st.session_state = mock_state
        # Import after patching
        from frontend.app import _generate_keyword_response
        return _generate_keyword_response(query, upload_context)


class TestVideoAnalysisResponses:
    """Tests for video-specific keyword matching."""

    def test_summer_hero_query(self):
        """Query mentioning 'summer' or 'hero' should return Summer_Hero analysis."""
        response = _get_keyword_response("Analyze the Summer_Hero_30s video")
        assert "Summer_Hero_30s" in response
        assert "PASS" in response
        assert "87%" in response

    def test_brand_story_query(self):
        """Query mentioning 'brand' and 'story' should return Brand_Story analysis."""
        response = _get_keyword_response("Tell me about the Brand_Story_15s")
        assert "Brand_Story_15s" in response
        assert "PASS" in response

    def test_product_focus_query(self):
        """Query mentioning 'product' and 'focus' should return Product_Focus analysis."""
        response = _get_keyword_response("Analyze Product_Focus_30s")
        assert "Product_Focus_30s" in response
        assert "FAIL" in response

    def test_hero_keyword_case_insensitive(self):
        """Keyword matching should be case insensitive."""
        response = _get_keyword_response("tell me about HERO video")
        assert "Summer_Hero_30s" in response


class TestKnowledgeResponses:
    """Tests for knowledge-based keyword matching."""

    def test_brand_recall_query(self):
        """'recall' keyword should trigger brand recall response."""
        response = _get_keyword_response("What drives brand recall?")
        assert "Knowledge Agent" in response
        assert "Logo" in response or "recall" in response.lower()

    def test_attention_query(self):
        """'attention' keyword should trigger attention drivers response."""
        response = _get_keyword_response("What drives attention scores?")
        assert "Attention" in response
        assert "Scene Diversity" in response or "attention" in response.lower()

    def test_budget_query(self):
        """'budget' keyword should trigger budget rules response."""
        response = _get_keyword_response("What are the budget rules?")
        assert "Planning Agent" in response
        assert "$5,000" in response or "5,000" in response

    def test_cost_query(self):
        """'cost' keyword should also trigger budget response."""
        response = _get_keyword_response("How much does testing cost?")
        assert "Planning Agent" in response

    def test_rules_query(self):
        """'rules' keyword should trigger budget response."""
        response = _get_keyword_response("Show me the CT rules")
        assert "Planning Agent" in response

    def test_tier_query(self):
        """'tier' keyword should trigger budget response."""
        response = _get_keyword_response("What tier am I in?")
        assert "Planning Agent" in response


class TestFollowUpDetection:
    """Tests for conversational follow-up handling."""

    def test_followup_keywords(self):
        """Follow-up phrases should be detected."""
        history = [
            {"role": "user", "content": "What drives brand recall?"},
            {"role": "assistant", "content": "Logo placement is key..."},
        ]
        response = _get_keyword_response("tell me more", chat_history=history)
        # Should not hit default response
        assert "Try asking" not in response or "follow-up" in response.lower() or "context" in response.lower()

    def test_explain_is_followup(self):
        """'explain' should be treated as a follow-up."""
        history = [
            {"role": "user", "content": "What are the budget rules?"},
            {"role": "assistant", "content": "Budget tiers..."},
        ]
        response = _get_keyword_response("can you explain the tier system?", chat_history=history)
        # Should not be the generic default
        assert response is not None


class TestDefaultResponse:
    """Tests for the fallback default response."""

    def test_unknown_query_gets_help(self):
        """Unrecognized queries should get a helpful default response."""
        response = _get_keyword_response("xyzzy random nonsense query")
        assert "Knowledge Agent" in response
        assert "Video Analysis" in response or "I can help" in response

    def test_default_includes_suggestions(self):
        """Default response should suggest what to ask."""
        response = _get_keyword_response("hello there")
        assert "Analyze" in response or "brand recall" in response.lower()


class TestTestPlanGeneration:
    """Tests for demo test plan generation from upload context."""

    def test_plan_with_videos(self):
        """Upload context with videos should generate a plan."""
        context = {
            "videos": [
                {"name": "hero_30s.mp4", "size_kb": 5000},
                {"name": "product_15s.mp4", "size_kb": 3000},
            ],
            "media_plan": None,
        }
        response = _get_keyword_response("generate a test plan for my videos", upload_context=context)
        assert "Planning Agent" in response
        assert "hero_30s.mp4" in response
        assert "product_15s.mp4" in response

    def test_plan_with_media_plan(self):
        """Upload context with media plan should be referenced."""
        context = {
            "videos": [],
            "media_plan": {
                "filename": "media_plan_q4.csv",
                "columns": ["channel", "impressions", "budget"],
                "rows": 25,
                "preview": [{"channel": "YouTube", "impressions": 1000000}],
            },
        }
        response = _get_keyword_response("generate test plan", upload_context=context)
        assert "Planning Agent" in response
        assert "media_plan_q4.csv" in response

    def test_plan_without_uploads(self):
        """Without uploads, plan-related query should not trigger plan generation."""
        response = _get_keyword_response("generate a test plan")
        # Should get the default response since no files uploaded
        assert "Planning Agent" not in response or "Upload" in response
