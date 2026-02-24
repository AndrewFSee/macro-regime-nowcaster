"""Tests for the narrative agent prompt construction and placeholder behaviour."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from src.agent.prompts import (
    MACRO_ANALYST_SYSTEM_PROMPT,
    NARRATIVE_TEMPLATE,
    format_narrative_prompt,
)
from src.agent.narrative_agent import NarrativeAgent, NarrativeReport
from src.agent.fed_scraper import FedDocument, _extract_date_from_url, _clean_text


# ---------------------------------------------------------------------------
# Prompt tests
# ---------------------------------------------------------------------------


def test_system_prompt_non_empty():
    assert len(MACRO_ANALYST_SYSTEM_PROMPT) > 100


def test_narrative_template_has_placeholders():
    assert "{current_regime}" in NARRATIVE_TEMPLATE
    assert "{gdp_nowcast" in NARRATIVE_TEMPLATE


def test_format_narrative_prompt_fills_values(sample_nowcast_result):
    docs = [
        {
            "date": "2024-12-15",
            "document_type": "minutes",
            "title": "FOMC Minutes",
            "text": "Economic activity expanded at a moderate pace.",
        }
    ]
    prompt = format_narrative_prompt(sample_nowcast_result.to_dict(), docs)
    assert "expansion" in prompt.lower()
    assert "2.40" in prompt or "2.4" in prompt
    assert "FOMC Minutes" in prompt


def test_format_narrative_prompt_empty_docs(sample_nowcast_result):
    prompt = format_narrative_prompt(sample_nowcast_result.to_dict(), [])
    assert "No recent Fed documents" in prompt


# ---------------------------------------------------------------------------
# NarrativeAgent tests
# ---------------------------------------------------------------------------


def test_narrative_agent_placeholder_without_api_key(sample_nowcast_result):
    """Agent should return a placeholder report when no API key is set."""
    agent = NarrativeAgent(api_key="")
    report = agent.generate(sample_nowcast_result.to_dict())
    assert isinstance(report, NarrativeReport)
    assert len(report.summary) > 0
    assert "expansion" in report.summary.lower()


def test_narrative_report_dataclass():
    report = NarrativeReport(
        summary="Test summary",
        key_drivers=["driver 1"],
        risk_flags=["risk 1"],
        corroboration_score="neutral",
        raw_response="raw",
    )
    assert report.corroboration_score == "neutral"
    assert len(report.key_drivers) == 1


def test_narrative_agent_with_mock_llm(sample_nowcast_result):
    """Test that the agent parses LLM output correctly."""
    fake_response = (
        "**Summary** Expansion regime with strong growth signals.\n\n"
        "**Key Drivers**\n- Strong labor market\n- Solid consumer spending\n\n"
        "**Fed Alignment** The Fed language corroborates the expansion narrative.\n\n"
        "**Risk Flags**\n- Inflation persistence\n- Credit tightening"
    )
    agent = NarrativeAgent(api_key="fake-key")
    # Replace the LLM call directly
    agent._call_llm = MagicMock(return_value=fake_response)
    agent._llm = MagicMock()  # mark as initialised

    report = agent.generate(sample_nowcast_result.to_dict())
    assert "Expansion" in report.summary or "expansion" in report.summary.lower()
    assert report.corroboration_score == "corroborates"


# ---------------------------------------------------------------------------
# FedScraper helper tests
# ---------------------------------------------------------------------------


def test_extract_date_from_url():
    url = "/monetarypolicy/files/FOMC20231213minutes.htm"
    date_str = _extract_date_from_url(url)
    assert date_str == "2023-12-13"


def test_extract_date_from_url_no_match():
    url = "/some/path/without/date"
    assert _extract_date_from_url(url) == ""


def test_clean_text_removes_extra_whitespace():
    text = "Hello   world\n\n\n\nNext paragraph"
    cleaned = _clean_text(text)
    assert "\n\n\n" not in cleaned
    assert "  " not in cleaned
