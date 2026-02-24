"""LLM-powered macro narrative agent.

Generates a human-readable narrative overlay for the nowcast result by
combining quantitative model output with recent Federal Reserve documents
through a large language model (OpenAI GPT or local Ollama).

Gracefully degrades when ``OPENAI_API_KEY`` is not set — returns a
placeholder narrative with a log warning rather than raising an exception.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from src.agent.prompts import MACRO_ANALYST_SYSTEM_PROMPT, format_narrative_prompt
from src.agent.fed_scraper import FedDocument


@dataclass
class NarrativeReport:
    """Structured narrative report from the LLM agent.

    Attributes:
        summary: 2-3 sentence regime summary.
        key_drivers: List of bullet-point driver descriptions.
        risk_flags: List of identified tail risks.
        corroboration_score: Qualitative label describing Fed alignment
            (``"corroborates"``, ``"neutral"``, ``"contradicts"``).
        raw_response: The full LLM response text.
    """

    summary: str
    key_drivers: list[str]
    risk_flags: list[str]
    corroboration_score: str
    raw_response: str


class NarrativeAgent:
    """Generates macro narrative reports using an LLM.

    Args:
        model: OpenAI model name (e.g. ``"gpt-4o-mini"``).
        temperature: LLM sampling temperature.
        max_tokens: Maximum response tokens.
        api_key: OpenAI API key.  Defaults to ``OPENAI_API_KEY`` env var.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 1500,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._llm = None

        if self._api_key:
            self._init_llm()
        else:
            logger.warning(
                "OPENAI_API_KEY not set — NarrativeAgent will return placeholder reports"
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        nowcast_result: dict,
        fed_documents: Optional[list[FedDocument]] = None,
    ) -> NarrativeReport:
        """Generate a narrative report for the given nowcast.

        Args:
            nowcast_result: Dict from ``NowcastResult.to_dict()``.
            fed_documents: Optional list of :class:`FedDocument` objects.

        Returns:
            :class:`NarrativeReport` with summary, drivers, risks, etc.
        """
        fed_docs_dicts = [
            {
                "date": d.date,
                "document_type": d.document_type,
                "title": d.title,
                "text": d.text,
            }
            for d in (fed_documents or [])
        ]

        if self._llm is None:
            return self._placeholder_report(nowcast_result)

        prompt = format_narrative_prompt(nowcast_result, fed_docs_dicts)

        try:
            response_text = self._call_llm(prompt)
            return self._parse_response(response_text)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"LLM call failed: {exc}")
            return self._placeholder_report(nowcast_result)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_llm(self) -> None:
        """Initialise the langchain LLM client."""
        try:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self._api_key,
            )
            logger.info(f"NarrativeAgent initialised with model {self.model}")
        except ImportError:
            logger.warning("langchain-openai not installed — narrative agent disabled")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to init LLM: {exc}")

    def _call_llm(self, user_prompt: str) -> str:
        """Call the LLM and return the response text."""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=MACRO_ANALYST_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        response = self._llm.invoke(messages)
        return response.content

    def _parse_response(self, text: str) -> NarrativeReport:
        """Extract structured fields from the LLM response text."""
        summary = _extract_section(text, "Summary", "Key Drivers") or text[:300]
        key_drivers = _extract_bullets(text, "Key Drivers", "Fed Alignment")
        risk_flags = _extract_bullets(text, "Risk Flags", None)
        corroboration = _extract_section(text, "Fed Alignment", "Risk Flags") or ""
        if "corroborat" in corroboration.lower():
            score = "corroborates"
        elif "contradict" in corroboration.lower():
            score = "contradicts"
        else:
            score = "neutral"

        return NarrativeReport(
            summary=summary.strip(),
            key_drivers=key_drivers,
            risk_flags=risk_flags,
            corroboration_score=score,
            raw_response=text,
        )

    def _placeholder_report(self, nowcast_result: dict) -> NarrativeReport:
        """Return a minimal placeholder when LLM is unavailable."""
        regime = nowcast_result.get("current_regime", "unknown")
        gdp = nowcast_result.get("gdp_nowcast", 0.0)
        return NarrativeReport(
            summary=f"Current regime: {regime}. GDP nowcast: {gdp:.2f}% annualised. (LLM narrative unavailable — set OPENAI_API_KEY.)",
            key_drivers=["LLM narrative agent not configured."],
            risk_flags=["Set OPENAI_API_KEY to enable narrative generation."],
            corroboration_score="neutral",
            raw_response="",
        )


# ---------------------------------------------------------------------------
# Text parsing helpers
# ---------------------------------------------------------------------------


def _extract_section(text: str, start_marker: str, end_marker: Optional[str]) -> str:
    """Extract text between two markdown section headers."""
    import re

    pattern = (
        rf"\*\*{re.escape(start_marker)}\*\*[:\s]*(.*?)"
        + (rf"(?=\*\*{re.escape(end_marker)}\*\*)" if end_marker else r"$")
    )
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _extract_bullets(text: str, start_marker: str, end_marker: Optional[str]) -> list[str]:
    """Extract bullet-point items from a section."""
    section = _extract_section(text, start_marker, end_marker)
    bullets = []
    for line in section.split("\n"):
        line = line.strip().lstrip("-•*").strip()
        if line:
            bullets.append(line)
    return bullets
